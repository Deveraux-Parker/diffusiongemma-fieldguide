#!/usr/bin/env python3
"""
DiffusionGemma — Wave 2 battery (8k server, sequential).
All retrieval prompts use SENTENCE form (not terse) to avoid the empty-dropout quirk.

W1 dead-zone CLIFF : fix fill=6k, slide needle by ABSOLUTE token-distance from question
W2 canvas boundary : force output lengths across the 256-tok block boundary; seam/latency?
W3 temperature     : does temp do anything in the diffusion sampler? (0..2.0, diversity)
W4 JSON adherence  : strict-JSON requests -> valid-rate & schema conformance
W5 not-found       : answer absent -> admit NOT FOUND, or hallucinate? (+present control)
"""
import json, time, random, os, urllib.request

BASE="http://localhost:8001"; MODEL="dg-awq"; MAXLEN=8000
OUTDIR=os.path.dirname(os.path.abspath(__file__))

def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat_full(user,max_tokens=60,**kw):
    pl={"model":MODEL,"messages":[{"role":"user","content":user}],
        "max_tokens":max_tokens,"temperature":0.0}; pl.update(kw)
    t0=time.time(); r=_post("/v1/chat/completions",pl); dt=time.time()-t0
    return r["choices"][0]["message"]["content"], r["usage"], dt
def chat(user,max_tokens=60,**kw): return chat_full(user,max_tokens,**kw)[0]

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
def uniq8(t):
    w=(t or "").split(); g=[" ".join(w[i:i+8]) for i in range(max(0,len(w)-7))]
    return round(len(set(g))/len(g),3) if g else 0.0

def nominal(i,rng):
    return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C "
            f"pressure={rng.uniform(99,103):.1f}kPa flow={rng.uniform(8,16):.1f}L/min status=nominal")
def rand_code(rng):
    return f"{rng.choice(['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE'])}-{rng.randint(1000,9999)}"
def needle(i,code):
    return f"[t={i:05d}] ANOMALY sensor_temp=87.3C status=CRITICAL incident_code={code}"

raw=open(os.path.join(OUTDIR,"raw_wave2.jsonl"),"w")
def lr(r): raw.write(json.dumps(r)+"\n"); raw.flush()

SENT_Q=("\n\nOne line has status=CRITICAL with an incident_code. "
        "In one short sentence, state that incident_code.")

# ---- W1: dead-zone cliff by absolute token-distance-from-question -----------
def W1():
    print("== W1: dead-zone cliff (fill=6k, needle distance from question) ==")
    rng0=random.Random(1); tpl=tok_count("\n".join(nominal(i,rng0) for i in range(1,21)))/20.0
    NLINES=int(5800/tpl)  # ~5.8k-token fill, leaves room for needle+question+output under 8k
    out=[]
    for D in [200,400,600,800,1000,1200,1500,2000,3000,4000]:
        h=0; dists=[]
        for tr in range(5):
            rng=random.Random(71000+D+tr); code=rand_code(rng)
            fl=[nominal(i,rng) for i in range(1,NLINES+1)]
            lines_after=max(0,min(len(fl)-1,round(D/tpl)))
            idx=len(fl)-lines_after; fl.insert(idx, needle(900000+tr,code))
            body="\n".join(fl)
            user="Telemetry log:\n"+body+SENT_Q
            actual_dist=tok_count("\n".join(fl[idx+1:])+SENT_Q)
            o=chat(user,max_tokens=60); ok=found(o,code); h+=ok; dists.append(actual_dist)
            lr({"w":"W1","targetD":D,"actualD":actual_dist,"trial":tr,"code":code,"out":o,"hit":ok})
        ad=round(sum(dists)/len(dists))
        print(f"  dist~{ad:5d} tok (target {D:5d})  hit={h}/5"); out.append({"dist":ad,"hit":h})
    return out

# ---- W2: canvas-boundary / multi-block coherence ---------------------------
def W2():
    print("== W2: 256-tok canvas boundary (forced lengths, coherence+latency) ==")
    user=("Write a continuous, internally consistent technical explanation of how a "
          "diffusion language model denoises text. Do not repeat yourself.")
    out=[]
    for mt in [128,224,256,288,384,512,640]:
        o,usage,dt=chat_full(user,max_tokens=mt,min_tokens=mt,ignore_eos=True)
        u=uniq8(o); ct=usage["completion_tokens"]
        print(f"  forced={mt:4d} compTok={ct:4d} uniq8={u:.3f} lat={round(1000*dt)}ms")
        lr({"w":"W2","forced":mt,"compTok":ct,"uniq8":u,"lat_ms":round(1000*dt),"out":o})
        out.append({"forced":mt,"uniq8":u,"lat_ms":round(1000*dt)})
    return out

# ---- W3: does temperature do anything? -------------------------------------
def W3():
    print("== W3: temperature effect (diversity across 4 samples per temp) ==")
    user="Write the opening line of a short story about a lighthouse keeper."
    out=[]
    for temp in [0.0,0.5,1.0,1.5,2.0]:
        samples=[]
        for s in range(4):
            o=chat(user,max_tokens=50,temperature=temp,seed=1000+s)  # seed varied if honored
            samples.append((o or "").strip())
        uniq=len(set(samples))
        # mean pairwise normalized edit distance = diversity
        import itertools
        pds=[]
        for a,b in itertools.combinations(samples,2):
            m=max(len(a),len(b),1); pds.append(lev(a,b)/m)
        div=round(sum(pds)/len(pds),3) if pds else 0.0
        print(f"  temp={temp:.1f} unique={uniq}/4 diversity={div:.3f}  e.g. {samples[0][:50]!r}")
        lr({"w":"W3","temp":temp,"unique":uniq,"diversity":div,"samples":samples})
        out.append({"temp":temp,"unique":uniq,"diversity":div})
    return out

# ---- W4: strict JSON adherence ---------------------------------------------
def W4():
    print("== W4: strict-JSON adherence ==")
    rng=random.Random(4)
    series="\n".join(needle(5,"AMBER-2207") if i==5 else nominal(i,rng) for i in range(1,20))
    user=("Given the telemetry below, output ONLY a JSON object with keys "
          '"device_status" (string), "critical_count" (integer), "incident_code" '
          "(string or null). No prose, no code fences.\n\n"+series)
    valid=0; conform=0; n=8; egs=[]
    for tr in range(n):
        o=(chat(user,max_tokens=80) or "").strip()
        s=o
        if s.startswith("```"): s=s.strip("`"); s=s[s.find("{"):s.rfind("}")+1]
        ok=False; cf=False
        try:
            j=json.loads(s[s.find("{"):s.rfind("}")+1]); ok=True
            cf=all(k in j for k in ("device_status","critical_count","incident_code"))
        except Exception: pass
        valid+=ok; conform+=cf
        if tr<4: egs.append(o[:70])
        lr({"w":"W4","trial":tr,"valid":ok,"conform":cf,"out":o})
    print(f"  valid-JSON={valid}/{n}  schema-conform={conform}/{n}")
    for e in egs: print(f"    {e!r}")
    return {"valid":valid,"conform":conform,"n":n}

# ---- W5: hallucinate vs admit not-found ------------------------------------
def W5():
    print("== W5: not-found honesty (absent vs present) ==")
    out={}
    for cond in ["absent","present"]:
        admit=0; halluc=0; hit=0; n=8
        for tr in range(n):
            rng=random.Random(5000+tr); code=rand_code(rng)
            lines=[nominal(i,rng) for i in range(1,40)]
            if cond=="present":
                lines.insert(20,needle(900,code))
            body="\n".join(lines)
            user=("Telemetry log below. If a line has status=CRITICAL, state its incident_code "
                  "in one sentence. If there is NO CRITICAL line, reply exactly: NOT FOUND.\n\n"+body)
            o=(chat(user,max_tokens=50) or "").strip()
            nf="NOT FOUND" in o.upper()
            if cond=="absent":
                if nf: admit+=1
                else: halluc+=1
            else:
                if found(o,code): hit+=1
            lr({"w":"W5","cond":cond,"trial":tr,"code":code if cond=='present' else None,"out":o})
        if cond=="absent":
            print(f"  absent : admit-NOT-FOUND={admit}/{n}  hallucinated={halluc}/{n}")
            out["absent"]={"admit":admit,"halluc":halluc,"n":n}
        else:
            print(f"  present: correct-retrieve={hit}/{n}")
            out["present"]={"hit":hit,"n":n}
    return out

if __name__=="__main__":
    R={}
    R["W1_cliff"]=W1(); R["W2_canvas"]=W2(); R["W3_temp"]=W3()
    R["W4_json"]=W4(); R["W5_notfound"]=W5()
    json.dump(R,open(os.path.join(OUTDIR,"results_wave2.json"),"w"),indent=2,default=str)
    print("\nsaved -> DIFFBENCH/results_wave2.json, raw_wave2.jsonl")
