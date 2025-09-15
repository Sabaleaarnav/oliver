import os, sounddevice as sd, numpy as np, requests, tempfile, wave, subprocess
from faster_whisper import WhisperModel

SR=16000
MODEL="base"  # try "small" later for better accuracy
OPENTTS=os.getenv("OPENTTS_URL","http://localhost:5500")
VOICE=os.getenv("OPENTTS_VOICE","en_US-amy-low")

model=WhisperModel(MODEL, device="cpu", compute_type="int8")

def record(seconds=6):
    print("Listening… (Ctrl+C to stop)")
    audio = sd.rec(int(seconds*SR), samplerate=SR, channels=1, dtype='float32')
    sd.wait()
    return audio[:,0]

def to_wav(arr):
    path=tempfile.mktemp(suffix=".wav")
    with wave.open(path,'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((arr*32767).astype(np.int16).tobytes())
    return path

def speak(text):
    if not text: return
    r = requests.get(f"{OPENTTS}/api/tts", params={"voice":VOICE, "text":text}, timeout=60)
    r.raise_for_status()
    wav = tempfile.mktemp(suffix=".wav")
    open(wav,"wb").write(r.content)
    # play on default system output (PulseAudio/pipewire)
    subprocess.run(["ffplay","-nodisp","-autoexit","-loglevel","quiet", wav])

while True:
    input("Press Enter, speak (auto-stops ~6s)…")
    a=record(6); p=to_wav(a)
    segments, _ = model.transcribe(p, language="en")
    text=" ".join(s.text for s in segments).strip()
    print("You:", text if text else "[no speech]")
    if text:
        r = requests.post("http://localhost:8080/command",
                          json={"text":text,"mode":"mini"}, timeout=60)
        reply = (r.json().get("reply") if r.headers.get("content-type","").startswith("application/json")
                 else r.text)
        print("Oliver:", reply)
        try: speak(reply)
        except Exception as e: print("TTS error:", e)
