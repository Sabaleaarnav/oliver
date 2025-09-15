import os, json, time, re, httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel

# --- config ---
OPENAI_ON   = os.getenv("OPENAI","") == "1"
OPENAI_KEY  = os.getenv("OPENAI_API_KEY","")
OPENAI_MODEL= os.getenv("OPENAI_MODEL","gpt-4o-mini")
OLLAMA_URL  = os.getenv("OLLAMA_HOST","http://ollama:11434")
OLLAMA_MODEL= os.getenv("OLLAMA_MODEL","llama3.2:3b-instruct")
CHROMA_HOST = os.getenv("CHROMA_HOST","http://chroma:8000")

MEMORY_FILE = os.getenv("MEMORY_FILE","/data/memory.json")
ASSISTANT_SYSTEM = "You are Oliver. Be concise, friendly, and helpful."

# --- tiny memory store ---
def _mload():
    try:
        with open(MEMORY_FILE,"r") as f: return json.load(f)
    except Exception:
        return {"facts":[]}

def _msave(db):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE,"w") as f: json.dump(db,f,indent=2)

def add_fact(text, source="user"):
    db=_mload(); db["facts"].append({"text":text,"source":source,"ts":int(time.time())}); _msave(db); return len(db["facts"])

def list_facts(limit=10):
    return _mload()["facts"][-limit:]

def clear_all():
    _msave({"facts":[]})

# --- app ---
app = FastAPI()

class Command(BaseModel):
    text: str
    mode: str | None = None  # "mini" or "local"

@app.get("/")
def root():
    return {"ok": True, "mini": OPENAI_ON, "model_mini": OPENAI_MODEL, "model_local": OLLAMA_MODEL}

# memory interceptor
def memory_reply(text: str) -> str | None:
    low = text.strip().lower()

    if low.startswith(("remember ", "remember:", "note ", "note:", "save ")):
        fact = text.split(" ",1)[1] if " " in text else text
        n = add_fact(fact, "user")
        return f"Got it — I’ll remember that. (#{n})"

    m = re.match(r"recall\s*(\d+)?$", low)
    if low in {"recall","memories","list memories","what do you remember"} or m:
        n = int(m.group(1)) if m and m.group(1) else 10
        facts = list_facts(n)
        if not facts: return "I don't have any stored memories yet."
        return "Here's what I remember:\n" + "\n".join(f"- {f['text']}" for f in facts)

    if low in {"forget all","clear memories","wipe memory"}:
        clear_all(); return "All memories cleared."

    return None

async def openai_chat(prompt: str) -> str:
    # use Chat Completions (robust & simple)
    headers = {"Authorization": f"Bearer {OPENAI_KEY}"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": ASSISTANT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        j=r.json()
        return j["choices"][0]["message"]["content"].strip()

async def ollama_generate(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate",
                              json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        r.raise_for_status()
        j=r.json()
        return j.get("response","").strip()

@app.post("/command")
async def command(body: Command):
    # 1) memory first
    m = memory_reply(body.text)
    if m is not None:
        return {"reply": m}

    # 2) route to model
    mode = body.mode or ("mini" if OPENAI_ON else "local")
    prompt = f"{ASSISTANT_SYSTEM}\n\nUser: {body.text}\nAssistant:"
    if mode == "mini":
        if not OPENAI_ON or not OPENAI_KEY:
            return {"reply":"OpenAI mode not configured."}
        try:
            out = await openai_chat(body.text)
        except Exception as e:
            return {"reply": f"OpenAI error: {e}"}
        return {"reply": out}
    else:
        try:
            out = await ollama_generate(prompt)
        except Exception as e:
            return {"reply": f"Ollama error: {e}"}
        return {"reply": out}
