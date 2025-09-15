import json, os, time
MEMORY_FILE = os.getenv('MEMORY_FILE','/data/memory.json')
def _load():
    try:
        with open(MEMORY_FILE,'r') as f: return json.load(f)
    except Exception: return {"facts":[]}
def _save(db):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE,'w') as f: json.dump(db,f,indent=2)
def add_fact(text, source="user"):
    db=_load(); db["facts"].append({"text":text,"source":source,"ts":int(time.time())}); _save(db); return len(db["facts"])
def list_facts(limit=50): return _load()["facts"][-limit:]
def clear_all(): _save({"facts":[]})
