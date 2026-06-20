#!/usr/bin/env python3
"""
Disentangle: is the "dead zone" a POSITIONAL attention gap, or NEEDLE-NEIGHBOUR
INTERFERENCE (the target being confusable with similar surrounding lines)?

2x2 at the rock-solid dead position (fill 5500, abs pos ~4466 = dist ~1000, which
was 0/10 with a telemetry-blended needle):
  needle  {blends  = telemetry-formatted CRITICAL line,
           distinct = ">>> NOTICE: authorization code = X <<<"}
  filler  {telemetry nominal lines, lorem ipsum}
Prediction (interference): telemetry+blends -> ~0; telemetry+distinct -> high.
Prediction (pure position): all four ~0 at this position.
"""
import json, random, os, urllib.request, datetime
BASE="http://127.0.0.1:8001"; MODEL="dg-awq"; FILL=5500; REPS=8; POS=4466
HERE=os.path.dirname(os.path.abspath(__file__))
def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat(u,mt=70): return _post("/v1/chat/completions",{"model":MODEL,
    "messages":[{"role":"user","content":u}],"max_tokens":mt,"temperature":0.0})["choices"][0]["message"]["content"]
def lev(a,b):
    if a==b: return 0
    m,n=len(a),len(b); d=list(range(n+1))
    for i in range(1,m+1):
        prev=d[0]; d[0]=i
        for j in range(1,n+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=t
    return d[n]
def has_code(o,c):
    o=(o or "").upper(); c=c.upper()
    if c in o: return True
    for tk in o.replace("="," ").replace("."," ").replace(","," ").split():
        if lev(tk,c)<=1: return True
    return False
LOREM=("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut "
 "labore et dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi "
 "aliquip ex ea commodo consequat duis aute irure in reprehenderit voluptate velit esse cillum").split()
COLORS=['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE']
def telem(i,rng): return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C pressure={rng.uniform(99,103):.1f}kPa "
                          f"flow={rng.uniform(8,16):.1f}L/min status=nominal")
def lorem(rng,n=12): return " ".join(rng.choice(LOREM) for _ in range(n)).capitalize()+"."
def needle_blend(color,num): return f"[t=90012] ANOMALY sensor_temp=87.3C status=CRITICAL incident_code={color}-{num}"
def needle_distinct(color,num): return f">>> NOTICE: authorization code = {color}-{num} <<<"
Q=("\n\nOne line is special (it is marked CRITICAL or NOTICE) and carries a code. "
   "State that code in one short sentence.")
raw=None
def lr(r): raw.write(json.dumps(r)+"\n"); raw.flush()
def main():
    global raw
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"interference-2x2-{stamp}"); os.makedirs(outdir,exist_ok=True)
    raw=open(os.path.join(outdir,"raw.jsonl"),"a")
    tpl_t=tok_count("\n".join(telem(i,random.Random(i)) for i in range(20)))/20.0
    tpl_l=tok_count("\n".join(lorem(random.Random(i)) for i in range(20)))/20.0
    print(f"== Interference 2x2 (fill~{FILL}, pos~{POS}, {REPS} reps) ==")
    grid={}
    for fname,(lf,tpl) in {"telemetry":(telem,tpl_t),"lorem":(lorem,tpl_l)}.items():
        nlines=int(FILL/tpl); idx=max(1,min(nlines-1,round(POS/tpl)))
        for nname,nf in {"blends":needle_blend,"distinct":needle_distinct}.items():
            hits=0
            for rep in range(REPS):
                rng=random.Random(60000+rep)
                color=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; code=f"{color}-{num}"
                fl=[ (lf(i,rng) if fname=="telemetry" else lf(rng)) for i in range(1,nlines+1)]
                fl.insert(idx, nf(color,num))
                out=chat("Reference document:\n"+"\n".join(fl)+Q); ok=has_code(out,code); hits+=ok
                lr({"filler":fname,"needle":nname,"rep":rep,"code":code,"out":out,"hit":bool(ok)})
            grid[(fname,nname)]=hits
            print(f"  filler={fname:9s} needle={nname:8s}  {hits}/{REPS}  {'#'*hits}{'.'*(REPS-hits)}")
    L=["# Interference 2x2 (dead position)","",f"- fill~{FILL} pos~{POS} reps {REPS}","",
       "| | blends-in needle | distinct needle |","| --- | --- | --- |",
       f"| telemetry filler | {grid[('telemetry','blends')]}/{REPS} | {grid[('telemetry','distinct')]}/{REPS} |",
       f"| lorem filler | {grid[('lorem','blends')]}/{REPS} | {grid[('lorem','distinct')]}/{REPS} |"]
    open(os.path.join(outdir,"summary.md"),"w").write("\n".join(L))
    json.dump({f"{a}|{b}":v for (a,b),v in grid.items()},open(os.path.join(outdir,"results.json"),"w"),indent=2)
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/")
if __name__=="__main__": main()
