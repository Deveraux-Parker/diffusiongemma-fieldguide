#!/usr/bin/env python3
"""
DiffusionGemma — Wave 3: MITIGATING the dead zone.

A needle (incident_code) is buried ~1000 tokens before the question (the dead zone,
where baseline retrieval ~ 0/5). We test six placement strategies — including the
"duplicate it right after itself" idea — to see which actually rescue retrieval.

All copies carry the SAME code. Sentence-form question (avoids the terse-dropout quirk).
Distances are ABSOLUTE token-distance from the question.
"""
import json, random, os, urllib.request

BASE="http://localhost:8001"; MODEL="dg-awq"; MAXLEN=8000
OUTDIR=os.path.dirname(os.path.abspath(__file__)); N=8

def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat(user,max_tokens=60,**kw):
    pl={"model":MODEL,"messages":[{"role":"user","content":user}],
        "max_tokens":max_tokens,"temperature":0.0}; pl.update(kw)
    return _post("/v1/chat/completions",pl)["choices"][0]["message"]["content"]

def lev(a,b):
    if a==b: return 0
    m,n=len(a),len(b); d=list(range(n+1))
    for i in range(1,m+1):
        prev=d[0]; d[0]=i
        for j in range(1,n+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=t
    return d[n]
def found(out,code):
    o=(out or "").upper(); c=code.upper()
    if c in o: return True
    for tk in o.replace("="," ").replace(","," ").replace("."," ").split():
        if lev(tk,c)<=1: return True
    return c[1:] in o

def nominal(i,rng):
    return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C "
            f"pressure={rng.uniform(99,103):.1f}kPa flow={rng.uniform(8,16):.1f}L/min status=nominal")
def rand_code(rng):
    return f"{rng.choice(['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE'])}-{rng.randint(1000,9999)}"
def needle(i,code):
    return f"[t={i:05d}] ANOMALY sensor_temp=87.3C status=CRITICAL incident_code={code}"

raw=open(os.path.join(OUTDIR,"raw_wave3.jsonl"),"w")
def lr(r): raw.write(json.dumps(r)+"\n"); raw.flush()

Q=("\n\nOne or more lines have status=CRITICAL with an incident_code. "
   "In one short sentence, state that incident_code.")

# token-per-line calibration
TPL=tok_count("\n".join(nominal(i,random.Random(i)) for i in range(20)))/20.0
NLINES=int(5500/TPL)
def idx_for(distance, nlines):  # absolute token-distance from question -> line index
    return max(1, min(nlines-1, nlines - round(distance/TPL)))

# placements: list of absolute token-distances at which to drop a copy of the needle
STRATS = {
 "baseline_deadzone":      [1000],                 # single copy, in the dead zone
 "dup_adjacent_x2":        [1000, 1000],           # user's idea: two back-to-back in the dead zone
 "dup_adjacent_x3":        [1000, 1000, 1000],     # three back-to-back in the dead zone
 "dup_recency":            [1000, 150],            # dead-zone copy + copy right before the question
 "dup_primacy":            [1000, 5300],           # dead-zone copy + copy at the very top
 "dup_primacy_recency":    [1000, 5300, 150],      # belt & suspenders: top + dead + recency
}

def run():
    results={}
    for name,dists in STRATS.items():
        hits=0
        for tr in range(N):
            rng=random.Random(80000+hash(name)%1000+tr); code=rand_code(rng)
            fl=[nominal(i,rng) for i in range(1,NLINES+1)]
            # for adjacent duplicates, nudge copies to consecutive line slots
            placements=[]
            base_idx={}
            for k,dd in enumerate(dists):
                ix=idx_for(dd,len(fl))
                # if same distance repeated, stack consecutively
                while ix in base_idx: ix+=1
                base_idx[ix]=True; placements.append(ix)
            for ix in sorted(placements,reverse=True):
                fl.insert(ix, needle(900000+ix%9999, code))
            user="Telemetry log:\n"+"\n".join(fl)+Q
            o=chat(user,max_tokens=60); ok=found(o,code); hits+=ok
            lr({"strat":name,"trial":tr,"dists":dists,"code":code,"out":o,"hit":ok})
        results[name]=hits
        print(f"  {name:22s} {hits}/{N}  {'#'*hits}{'.'*(N-hits)}")
    return results

if __name__=="__main__":
    print(f"== Wave 3: dead-zone mitigation (fill~5.5k, needle @~1000 tok, {N} trials) ==")
    print(f"   (tokens/line~{TPL:.1f}, filler lines~{NLINES})")
    R=run()
    json.dump(R,open(os.path.join(OUTDIR,"results_wave3.json"),"w"),indent=2)
    print("\nsaved -> DIFFBENCH/results_wave3.json, raw_wave3.jsonl")
