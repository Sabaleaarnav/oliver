import os, time, subprocess, tempfile, wave, requests
import numpy as np, sounddevice as sd
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from openwakeword import Model as WakeModel

# --- config
TARGET_SR   = 16000
RING_SEC    = 1.0                      # context window passed to OWW
IN_DEV      = os.getenv("SD_INPUT_INDEX")          # e.g. "6"
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD","0.45"))
OPENTTS     = os.getenv("OPENTTS_URL","http://127.0.0.1:5500")
VOICE       = os.getenv("OPENTTS_VOICE","larynx:mary_ann-glow_tts")
AGENT       = os.getenv("AGENT_URL","http://127.0.0.1:8080/command")
WAKE_KEYS   = ("hey_mycroft","hey_jarvis","alexa") # limit to these

def _detect_in_sr():
    try:
        d = sd.query_devices(int(IN_DEV), 'input') if IN_DEV is not None else sd.query_devices(None,'input')
        return int(d.get('default_samplerate') or 48000)
    except Exception:
        return 48000

IN_SR  = int(os.getenv("SD_INPUT_SR", _detect_in_sr()))
BLOCK  = int(round(IN_SR * 0.032))     # ~32 ms blocks
RING_N = int(TARGET_SR * RING_SEC)

wake = WakeModel()                      # bundled wakewords
stt  = WhisperModel("base", device="cpu", compute_type="int8")

def _resample_to_16k(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32) if IN_SR == TARGET_SR else resample_poly(x, TARGET_SR, IN_SR).astype(np.float32)

def _to_wav_16k(arr16: np.ndarray) -> str:
    p = tempfile.mktemp(suffix=".wav")
    with wave.open(p,'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(TARGET_SR)
        f.writeframes((np.clip(arr16, -1, 1)*32767).astype(np.int16).tobytes())
    return p

def _speak(text: str):
    if not text: return
    try:
        r = requests.get(f"{OPENTTS}/api/tts", params={"voice":VOICE,"text":text}, timeout=60)
        r.raise_for_status()
        wav = tempfile.mktemp(suffix=".wav")
        open(wav,"wb").write(r.content)
        subprocess.run(["ffplay","-nodisp","-autoexit","-loglevel","quiet", wav])
    except Exception as e:
        print("TTS error:", e)

def _transcribe(wav_path: str) -> str:
    segs,_ = stt.transcribe(wav_path, language="en")
    return " ".join(s.text for s in segs).strip()

def _top(scores: dict, n=3):
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return ", ".join(f"{k}:{v:.3f}" for k,v in items)

print(f"Mic dev={IN_DEV or 'default'} IN_SR={IN_SR}, BLOCK={BLOCK} (~32 ms) → 16 kHz, ring={RING_SEC:.1f}s")
print("Listening for: 'hey mycroft', 'hey jarvis', or 'alexa' (Ctrl+C to quit)")

# ring buffer (1s @ 16 kHz)
ring = np.zeros(RING_N, dtype=np.float32)

def push_ring(x16: np.ndarray):
    global ring
    n = len(x16)
    if n >= RING_N:
        ring = x16[-RING_N:]
    else:
        ring = np.roll(ring, -n)
        ring[-n:] = x16

with sd.InputStream(channels=1, samplerate=IN_SR, blocksize=BLOCK, dtype="float32",
                    device=None if IN_DEV is None else int(IN_DEV)) as stream:
    last_dbg = 0.0
    cooldown = 0.0
    while True:
        audio, _ = stream.read(BLOCK)            # float32 @ IN_SR, ~32 ms
        mono16 = _resample_to_16k(audio[:,0])    # -> ~512 @ 16 kHz
        push_ring(mono16)                        # keep ~1s context

        # call OWW on the full 1s window
        scores = wake.predict(mono16)

        # periodic diagnostics
        now = time.time()
        if now - last_dbg > 0.8:
            rms = float(np.sqrt(np.mean(ring**2)))
            interesting = {k:scores.get(k,0.0) for k in WAKE_KEYS}
            print(f"rms={rms:.3f}  scores: {_top(interesting)}")
            last_dbg = now

        if cooldown > now:
            continue

        if scores:
            hot = max(WAKE_KEYS, key=lambda k: scores.get(k, 0.0))
            if scores.get(hot, 0.0) >= WAKE_THRESHOLD:
                _speak("Yes?")
                rec = sd.rec(int(6*IN_SR), samplerate=IN_SR, channels=1, dtype="float32",
                             device=None if IN_DEV is None else int(IN_DEV))
                sd.wait()
                rec16 = _resample_to_16k(rec[:,0])
                wav_path = _to_wav_16k(rec16)
                text = _transcribe(wav_path)
                print("You:", text or "[no speech]")

                if text:
                    try:
                        r = requests.post(AGENT, json={"text":text, "mode":"mini"}, timeout=90)
                        reply = r.json().get("reply","(no reply)") if r.headers.get("content-type","").startswith("application/json") else r.text
                        print("Oliver:", reply)
                        _speak(reply)
                    except Exception as e:
                        print("Agent error:", e)
                cooldown = now + 1.0               # debounce 1s
