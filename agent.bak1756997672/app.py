from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import os, json, asyncio, subprocess, tempfile
import httpx
from openai import AsyncOpenAI
from dateutil import parser
from datetime import datetime, timedelta
from ics import Calendar

# --- Config ---
OLLAMA = os.getenv("OLLAMA_HOST","http://localhost:11434")
CHROMA = os.getenv("CHROMA_HOST","http://localhost:8000")
USE_OPENAI = bool(os.getenv("OPENAI"))
OPENAI_MODEL_DEFAULT = os.getenv("OPENAI_MODEL","gpt-4o-mini")
OPENAI = AsyncOpenAI() if USE_OPENAI else None
OPENTTS_URL = os.getenv("OPENTTS_URL")
OPENTTS_VOICE = os.getenv("OPENTTS_VOICE","en_US-amy-low")
LAT = float(os.getenv("OPEN_METEO_LAT","40.4237"))
LON = float(os.getenv("OPEN_METEO_LON","-86.9212"))
NOTES_DIR = os.getenv("NOTES_DIR","/data/notes")
CAL_DIR = os.getenv("CAL_DIR","/data/calendars")

ASSISTANT_SYSTEM = """You are Oliver, a crisp, professional assistant.
- Be factual and concise. If missing info, state what you need.
- No emojis or exclamations. Prefer bullet points when listing.
- If notes/calendars not ingested, say how to add them (paths). Max ~120 words."""

INTENT_SYSTEM = """You are Oliver's router. Return ONE intent + slots as strict JSON.
Intents:
- weather()
- timer_start(label,duration_min)
- timer_extend(label,duration_min)
- timer_cancel(label)
- rag_ingest(path)
- rag_query(question)
- schedule_propose(participants_csv,duration_min)
- smalltalk()
Return only JSON. Example: {"intent":"weather","slots":{}}"""

app = FastAPI(title="Oliver Agent")

CHAT_HTML = """
<!doctype html><meta charset="utf-8"><title>Oliver — Chat</title>
<style>
  body{font:16px system-ui;margin:0;background:#0b0f17;color:#e5ecff}
  #app{max-width:800px;margin:40px auto;padding:16px}
  .card{background:#121826;border:1px solid #1e2a44;border-radius:16px;padding:16px;margin:12px 0}
  .me{background:#1b2436}
  .row{display:flex;gap:8px}
  input,button,select{font:16px system-ui}
  input{flex:1;border-radius:12px;border:1px solid #294065;background:#0f1524;color:#e5ecff;padding:12px}
  button{border-radius:12px;border:0;padding:12px 16px;background:#3b82f6;color:white;cursor:pointer}
  select{border-radius:12px;border:1px solid #294065;background:#0f1524;color:#e5ecff;padding:12px}
  button:disabled{opacity:.6;cursor:not-allowed}
</style>
<div id="app">
  <h1>Oliver</h1>
  <div id="log" class="card" style="min-height:200px"></div>
  <div class="row card">
    <select id="mode">
      <option value="mini" selected>Mini (cheap)</option>
      <option value="reasoning">Reasoning (o3-mini)</option>
      <option value="local">Local (Ollama)</option>
    </select>
    <input id="txt" placeholder="Type to talk to Oliver…"/>
    <button id="go">Send</button>
  </div>
</div>
<script>
const log=document.getElementById('log'), txt=document.getElementById('txt'), go=document.getElementById('go'), modeSel=document.getElementById('mode');
function addBubble(who,msg){const d=document.createElement('div'); d.className='card '+(who==='me'?'me':''); d.textContent=(who==='me'?'You: ':'Oliver: ')+msg; log.appendChild(d); log.scrollTop=log.scrollHeight;}
async function speak(text){
  try{
    const r=await fetch('/tts?text='+encodeURIComponent(text));
    if(r.ok){const b=await r.blob(); const url=URL.createObjectURL(b); new Audio(url).play(); return;}
  }catch(e){}
  try{
    const u=new SpeechSynthesisUtterance(text);
    const vs=speechSynthesis.getVoices();
    const best=vs.find(v=>/google/i.test(v.name)&&/en[-_]?US/i.test(v.lang))||vs.find(v=>/en[-_]?US/i.test(v.lang))||vs[0];
    if(best) u.voice=best; u.rate=0.9; u.pitch=1.0; speechSynthesis.speak(u);
  }catch(e){}
}
async function send(){
  const q=txt.value.trim(); if(!q) return; addBubble('me',q); txt.value=''; go.disabled=true;
  try{
    const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:q,mode:modeSel.value})});
    if(!r.ok){ addBubble('bot','Error: '+(await r.text())); }
    else{ const j=await r.json(); const reply=j.reply||'(no reply)'; addBubble('bot',reply); speak(reply); }
  }catch(e){ addBubble('bot','Network error: '+e); }
  go.disabled=false; txt.focus();
}
go.onclick=send; txt.onkeydown=(e)=>{if(e.key==='Enter') send();};
</script>
"""

@app.get("/", response_class=HTMLResponse)
async def index(): return CHAT_HTML

# --- Model caller (OpenAI or Ollama) ---
async def model_generate(prompt:str, mode:str="mini", options=None):
    if USE_OPENAI and mode in ("mini","reasoning"):
        model = "gpt-4o-mini" if mode=="mini" else "o3-mini"
        resp = await OPENAI.responses.create(
            model=model,
            input=[
              {"role":"system","content":"You are Oliver. Crisp, professional, factual. If missing info, say what you need. No emojis/exclamations."},
              {"role":"user","content":prompt}
            ],
            temperature=0.15,
            max_output_tokens=256
        )
        return resp.output_text.strip()
    payload={"model":"llama3:8b","prompt":prompt,"stream":False,
             "options": options or {"temperature":0.15,"top_p":0.9,"repeat_penalty":1.1,"num_predict":256}}
    async with httpx.AsyncClient(timeout=600) as cx:
        r=await cx.post(f"{OLLAMA}/api/generate",json=payload); r.raise_for_status()
        return r.json().get("response","").strip()

# --- Utilities (weather / RAG / scheduler) – unchanged from your last file ---
# (functions: quick_weather, extract_text, chroma_query, chroma_upsert,
#  load_busy, invert_busy, intersect, propose_slots) — omitted here for brevity,
#  but your file already has them. Keep them as-is.

# ↓↓↓ Paste the SAME utilities from your previous file here ↓↓↓
# (I kept them identical so this remains a drop-in replacement.)
import string
def extract_text(path):
    path=os.path.abspath(path)
    if not os.path.exists(path): return ""
    if path.lower().endswith(".pdf"):
        try:
            tmp=tempfile.mktemp()
            subprocess.run(["pdftotext",path,tmp],check=True)
            return open(tmp,"r",errors="ignore").read()
        except Exception:
            return ""
    return open(path,"r",errors="ignore").read()

async def chroma_query(q:str, mode:str, k:int=4):
    async with httpx.AsyncClient(timeout=30) as cx:
        await cx.post(f"{CHROMA}/api/v1/collections", json={"name":"oliver_notes"})
        coll=(await cx.get(f"{CHROMA}/api/v1/collections/by_name",params={"name":"oliver_notes"})).json()
        cid=coll["id"]
        rq=await cx.post(f"{CHROMA}/api/v1/collections/{cid}/query", json={"query_texts":[q],"n_results":k})
        docs=rq.json().get("documents",[[]])[0]
    ctx="\n\n".join(docs)
    if not ctx: return ""
    prompt=f"Use the context to answer.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer concisely:"
        low = text.strip().lower()
    # memory: add
    if low.startswith(("remember ","remember:","note ","note:","save ")):
        fact = text.split(" ",1)[1] if " " in text else text
        n = memory.add_fact(fact, "user")
        return f"Got it — I’ll remember that. (#{n})"
    # memory: list/recall
    if low in {"recall","memories","list memories","what do you remember"}:
        facts = memory.list_facts()
        if not facts:
            return "I don't have any stored memories yet."
        return "Here's what I remember:\\n" + "\\n".join(f"- {f['text']}" for f in facts[-10:])
return await model_generate(prompt, mode)

async def chroma_upsert(path:str)->int:
    text=extract_text(path)
    if not text: return 0
    chunks=[]; i=0
    while i<len(text):
        chunks.append(text[i:i+1000]); i+=850
    async with httpx.AsyncClient(timeout=30) as cx:
        coll=(await cx.get(f"{CHROMA}/api/v1/collections/by_name",params={"name":"oliver_notes"})).json()
        if "id" not in coll:
            coll=(await cx.post(f"{CHROMA}/api/v1/collections",json={"name":"oliver_notes"})).json()
        cid=coll["id"]
        ids=[f"{os.path.basename(path)}-{i}" for i,_ in enumerate(chunks)]
        await cx.post(f"{CHROMA}/api/v1/collections/{cid}/upsert", json={"ids":ids,"documents":chunks,"metadatas":[{"path":path}]*len(ids)})
    return len(chunks)

from datetime import datetime, timedelta
from ics import Calendar
from dateutil import parser
def load_busy(name,start,end):
    path=os.path.join(CAL_DIR,f"{name}.ics"); busy=[]
    if not os.path.exists(path): return busy
    with open(path) as f: cal=Calendar(f.read())
    for e in cal.events:
        s,e2=e.begin.datetime,e.end.datetime
        if e2<start or s>end: continue
        busy.append((max(s,start),min(e2,end)))
    busy.sort(); return busy
def invert_busy(busy,start,end):
    free=[]; cur=start
    for s,e in busy:
        if s>cur: free.append((cur,s))
        cur=max(cur,e)
    if cur<end: free.append((cur,end))
    return free
def intersect(frees):
    if not frees: return []
    out=frees[0]
    for L in frees[1:]:
        tmp=[]
        for a0,a1 in out:
            for b0,b1 in L:
                s,e=max(a0,b0),min(a1,b1)
                if s<e: tmp.append((s,e))
        out=tmp
    return out
def propose_slots(parts,duration_min=30,days=7,earliest=8,latest=22):
    now=datetime.now(); end=now+timedelta(days=7)
    per=[]
    for p in parts:
        free=invert_busy(load_busy(p,now,end),now,end)
        clipped=[]; cur=now
        while cur<end:
            ds=cur.replace(hour=earliest,minute=0,second=0,microsecond=0)
            de=cur.replace(hour=latest,minute=0,second=0,microsecond=0)
            for f0,f1 in free:
                s,e=max(f0,ds),min(f1,de)
                if s<e: clipped.append((s,e))
            cur+=timedelta(days=1)
        per.append(clipped)
    common=intersect(per)
    dur=timedelta(minutes=duration_min)
    out=[]
    for s,e in common:
        t=s
        while t+dur<=e:
            out.append({"start":t.isoformat(timespec="minutes"),"end":(t+dur).isoformat(timespec="minutes")})
            t+=timedelta(minutes=15)
    return sorted(out,key=lambda x:x["start"])

# --- Router + endpoints ---
class Command(BaseModel):
    text: str
    mode: str | None = "mini"

async def route(text:str, mode:str="mini"):
    try:
        payload={"model":"llama3:8b","prompt":f"{INTENT_SYSTEM}\n\nUser: {text}\nJSON:", "stream":False,
                 "options":{"temperature":0.0,"num_predict":128}}
        async with httpx.AsyncClient(timeout=60) as cx:
            r=await cx.post(f"{OLLAMA}/api/generate",json=payload)
            data=json.loads(r.json().get("response","{}"))
    except Exception:
        data={"intent":"smalltalk","slots":{}}
    intent=data.get("intent","smalltalk"); slots=data.get("slots",{})

    if intent=="weather":
        async with httpx.AsyncClient(timeout=10) as cx:
            url=f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&current=temperature_2m,precipitation&hourly=precipitation_probability"
            w=(await cx.get(url)).json()
        temp=w.get("current",{}).get("temperature_2m")
        p=(w.get("hourly",{}).get("precipitation_probability") or [0])[0]
        return f"{temp}°C now; precip ~{p}%. Meal windows: 8–9a, 11a–1p, 5–6:30p."
    if intent=="timer_start":
        label=slots.get("label","timer"); mins=int(slots.get("duration_min",25))
        return f"Started {label} for {mins} minutes."
    if intent=="timer_extend":
        label=slots.get("label","timer"); mins=int(slots.get("duration_min",5))
        return f"Extended {label} by {mins} minutes."
    if intent=="timer_cancel":
        label=slots.get("label","timer"); return f"Canceled {label}."
    if intent=="rag_ingest":
        path=slots.get("path",""); n=await chroma_upsert(path)
        return f"Ingested {n} chunks from {path}." if n else f"Couldn't ingest {path}."
    if intent=="rag_query":
        q=slots.get("question",""); ans=await chroma_query(q, mode); return ans or "I couldn't find that in your notes yet."
    if intent=="schedule_propose":
        names=[s.strip() for s in slots.get("participants_csv","").split(",") if s.strip()]
        dur=int(slots.get("duration_min",30)); slots_=propose_slots(names,duration_min=dur)
        if not slots_: return "No common slots in the next week."
        top=slots_[:5]; return "Top options:\n"+"\n".join(f"- {s['start']} → {s['end']}" for s in top)
    return await model_generate(f"{ASSISTANT_SYSTEM}\n\nUser: {text}\nAssistant:", mode)

@app.get("/", response_class=HTMLResponse)
async def home(): return CHAT_HTML

@app.get("/tts")
async def tts(text: str):
    if not OPENTTS_URL:
        return JSONResponse({"error":"opentts disabled"}, status_code=404)
    async with httpx.AsyncClient(timeout=60) as cx:
        r=await cx.get(f"{OPENTTS_URL}/api/tts", params={"voice":OPENTTS_VOICE,"text":text})
        r.raise_for_status()
        return Response(content=r.content, media_type=r.headers.get("content-type","audio/wav"))

@app.post("/command")
async def command(c: Command):
    mode=(c.mode or "mini").lower()
    if mode=="local":
        try:
            async with httpx.AsyncClient(timeout=6) as cx:
                tags=(await cx.get(f"{OLLAMA}/api/tags")).json().get("models",[])
            if not any("llama3" in m.get("name","") for m in tags):
                return JSONResponse({"reply":"Downloading local model. Try again in a minute."})
        except Exception as e:
            return JSONResponse({"reply":f"Ollama not reachable: {e}"})
    m=_memory_reply(c.text)
    if m is not None:
        return {"reply": m}
    reply=await route(c.text, mode)
    return JSONResponse({"reply":reply})
