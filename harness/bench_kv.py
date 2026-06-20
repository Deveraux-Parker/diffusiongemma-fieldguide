#!/usr/bin/env python3
"""
DiffusionGemma KV-utilization / needle-index benchmark.

Separates two axes the prior single-point test conflated:
  Exp A: retrieval vs KV fill, needle pinned at front (depth ~0.05)
  Exp B: retrieval vs needle index (depth sweep), at two KV fills
  Exp C: generation-length control -- long forced output at low KV fill

Filler = time-series sensor log lines (matches the front-loaded sequential
use case). Needle = an anomalous reading carrying a random incident code.
Retrieval is objective: case-insensitive substring match of the code.
"""
import json, time, random, os, urllib.request

BASE = "http://localhost:8001"
MODEL = "dg-awq"
MAXLEN = 8000
OUTDIR = os.path.dirname(os.path.abspath(__file__))
N_TRIALS = 5

def _post(path, payload):
    req = urllib.request.Request(BASE + path,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())

def tok_count(text):
    return _post("/tokenize", {"model": MODEL, "prompt": text})["count"]

# ---- content generation ----------------------------------------------------
def rand_code(rng):
    color = rng.choice(["MAGENTA","CYAN","AMBER","INDIGO","CRIMSON","JADE","SLATE","OCHRE"])
    return f"{color}-{rng.randint(1000,9999)}"

def log_line(i, rng):
    return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C "
            f"pressure={rng.uniform(99,103):.1f}kPa flow={rng.uniform(8,16):.1f}L/min "
            f"vibration={rng.uniform(0.1,0.9):.2f}mm status=nominal")

def needle_line(i, code):
    return (f"[t={i:05d}] ANOMALY sensor_temp=87.31C pressure=142.7kPa "
            f"flow=0.0L/min vibration=4.88mm status=CRITICAL "
            f"incident_code={code} (log this code)")

def build_prompt(target_tok, depth, rng):
    """Return (user_text, code, actual_tok). Needle placed at fractional depth."""
    code = rand_code(rng)
    # build a pool of filler lines well over target, then trim by token budget
    lines = [log_line(i, rng) for i in range(1, 600)]
    # binary-ish grow: add lines until near target (calibrate w/ one tok call/iter is slow;
    # estimate ~ tokens-per-line then verify+trim once)
    sample = "\n".join(lines[:20])
    tpl = tok_count(sample) / 20.0
    n = max(1, int(target_tok / tpl))
    n = min(n, len(lines))
    filler = lines[:n]
    pos = min(len(filler), max(0, int(depth * len(filler))))
    nl = needle_line(900000 + rng.randint(0,99999), code)
    filler.insert(pos, nl)
    body = "\n".join(filler)
    user = ("You are monitoring a sensor telemetry stream. Below are log lines.\n\n"
            + body +
            "\n\nQuestion: One line is an ANOMALY with status=CRITICAL. "
            "Reply with ONLY the incident_code from that anomalous line, nothing else.")
    return user, code, tok_count(user)

def ask_retrieval(user):
    t0 = time.time()
    r = _post("/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": 40, "temperature": 0.0})
    dt = time.time() - t0
    out = r["choices"][0]["message"]["content"]
    return out, r["usage"], dt

def hit(out, code):
    return code.lower() in (out or "").lower()

raw = open(os.path.join(OUTDIR, "raw.jsonl"), "w")
def lograw(rec):
    raw.write(json.dumps(rec) + "\n"); raw.flush()

# ---- Experiment A: retrieval vs KV fill (front needle) ----------------------
def exp_A():
    fills = [500, 1000, 2000, 4000, 6000, 7500]
    rows = []
    for f in fills:
        hits = 0; toks = []; lats = []
        for tr in range(N_TRIALS):
            rng = random.Random(1000 + f*10 + tr)
            user, code, at = build_prompt(f, 0.05, rng)
            out, usage, dt = ask_retrieval(user)
            h = hit(out, code); hits += h; toks.append(at); lats.append(dt)
            lograw({"exp":"A","fill":f,"depth":0.05,"trial":tr,"prompt_tok":at,
                    "code":code,"out":out,"hit":h,"lat":dt})
        at_avg = sum(toks)/len(toks)
        rows.append({"fill":f,"prompt_tok":round(at_avg),"kv_util":round(at_avg/MAXLEN,3),
                     "hit_rate":hits/N_TRIALS,"lat_ms":round(1000*sum(lats)/len(lats))})
        print(f"  A fill={f:5d} promptTok~{round(at_avg):5d} kvUtil={at_avg/MAXLEN:.2f} "
              f"hit={hits}/{N_TRIALS} lat~{round(1000*sum(lats)/len(lats))}ms")
    return rows

# ---- Experiment B: retrieval vs needle index (depth) at 2 fills -------------
def exp_B():
    depths = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    rows = []
    for f in [2000, 6000]:
        for d in depths:
            hits = 0; toks = []
            for tr in range(N_TRIALS):
                rng = random.Random(5000 + f*10 + int(d*100) + tr)
                user, code, at = build_prompt(f, d, rng)
                out, usage, dt = ask_retrieval(user)
                h = hit(out, code); hits += h; toks.append(at)
                lograw({"exp":"B","fill":f,"depth":d,"trial":tr,"prompt_tok":at,
                        "code":code,"out":out,"hit":h,"lat":dt})
            at_avg = sum(toks)/len(toks)
            rows.append({"fill":f,"kv_util":round(at_avg/MAXLEN,3),"depth":d,
                         "hit_rate":hits/N_TRIALS})
            print(f"  B fill={f:5d} depth={d:.2f} hit={hits}/{N_TRIALS}")
    return rows

# ---- Experiment C: generation-length control (low fill, long output) -------
def exp_C():
    rng = random.Random(99)
    user = ("Write a detailed, internally consistent technical description of how a "
            "block-diffusion language model denoises a 256-token canvas. Keep it "
            "coherent and non-repetitive throughout.")
    rows = []
    for mt in [128, 512, 1024, 1500]:
        t0 = time.time()
        r = _post("/v1/chat/completions", {
            "model": MODEL, "messages":[{"role":"user","content":user}],
            "max_tokens": mt, "min_tokens": mt, "temperature": 0.0, "ignore_eos": True})
        dt = time.time()-t0
        out = r["choices"][0]["message"]["content"]
        # crude repetition proxy: fraction of unique 8-grams
        words = out.split()
        grams = [" ".join(words[i:i+8]) for i in range(max(0,len(words)-7))]
        uniq = len(set(grams))/len(grams) if grams else 0
        rows.append({"forced_tok":mt,"comp_tok":r["usage"]["completion_tokens"],
                     "lat_ms":round(1000*dt),"uniq_8gram":round(uniq,3)})
        lograw({"exp":"C","forced_tok":mt,"out":out,"uniq_8gram":uniq,"lat":dt})
        print(f"  C forced={mt:5d} compTok={r['usage']['completion_tokens']:5d} "
              f"uniq8gram={uniq:.3f} lat~{round(1000*dt)}ms")
    return rows

if __name__ == "__main__":
    print("== Exp A: retrieval vs KV fill (front needle, depth 0.05) ==")
    A = exp_A()
    print("== Exp B: retrieval vs needle index ==")
    B = exp_B()
    print("== Exp C: generation-length control ==")
    C = exp_C()
    json.dump({"A":A,"B":B,"C":C}, open(os.path.join(OUTDIR,"results.json"),"w"), indent=2)
    print("\nsaved -> DIFFBENCH/results.json, raw.jsonl")
