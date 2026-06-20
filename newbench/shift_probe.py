#!/usr/bin/env python3
"""
THE 1024-SHIFT PROBE — decisive test of ABSOLUTE block-grid parity vs a
position-relative window.

Take a dead config (needle ~1000 tok before the question, total ~5500 = diff1,
even needle block). Then pad the START with K tokens. This shifts BOTH needle and
question absolute positions by ~K, so the distance D and the block-diff are UNCHANGED,
but the absolute block parity flips.
  Relative-window theory: distance unchanged -> outcome unchanged for all K.
  Absolute-parity theory: outcome flips with period 2048 (alive at K~1024, dead at K~0,2048).
"""
import json, random, os, urllib.request, datetime
BASE="http://127.0.0.1:8001"; MODEL="dg-awq"; HERE=os.path.dirname(os.path.abspath(__file__))
def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat(u,mt=64): return _post("/v1/chat/completions",{"model":MODEL,
    "messages":[{"role":"user","content":u}],"max_tokens":mt,"temperature":0.0})["choices"][0]["message"]["content"]
def lev(a,b):
    if a==b:return 0
    m,n=len(a),len(b); d=list(range(n+1))
    for i in range(1,m+1):
        pv=d[0]; d[0]=i
        for j in range(1,n+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,pv+(a[i-1]!=b[j-1])); pv=t
    return d[n]
def hc(o,c):
    o=(o or"").upper();c=c.upper()
    if c in o:return True
    for t in o.replace("="," ").replace("."," ").split():
        if lev(t,c)<=1:return True
    return False
LOREM=("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua").split()
COLORS=["MAGENTA","CYAN","AMBER","INDIGO","CRIMSON","JADE"]
def lorem(rng,n=12): return " ".join(rng.choice(LOREM) for _ in range(n)).capitalize()+"."
def rec(c,n): return f"RECORD: secret token = {c}-{n}."
Q="\n\nState the secret token from the RECORD line, in one short sentence."
TPL=tok("\n".join(lorem(random.Random(i)) for i in range(20)))/20.0
PRE_N=4500          # base tokens before the needle -> needle ~ block 4 (EVEN = DEAD base)
POST_N=1000         # tokens between needle and the question (distance ~1000, held constant)

def build(pad, rng):
    c=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; code=f"{c}-{num}"
    pre=round(PRE_N/TPL); post=round(POST_N/TPL)   # pre is FIXED; START padding ADDS on top
    pad_lines=round(pad/TPL)
    body=([lorem(rng) for _ in range(pad_lines)]            # START padding shifts needle abs pos by ~pad
          +[lorem(rng) for _ in range(pre)]
          +[rec(c,num)]
          +[lorem(rng) for _ in range(post)])
    prompt="LOG\n\n"+"\n".join(body)+Q
    nidx=pad_lines+max(0,pre)
    n_abs=tok("LOG\n\n"+"\n".join(body[:nidx]))
    q_abs=tok(prompt); D=q_abs-n_abs
    return prompt, code, n_abs, q_abs, D

def main():
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"shift-probe-{stamp}"); os.makedirs(outdir,exist_ok=True)
    raw=open(os.path.join(outdir,"raw.jsonl"),"a")
    print(f"(tpl~{TPL:.1f}) base needle pos ~{PRE_N} (even block=DEAD), distance ~{POST_N} held constant; vary START padding K")
    print("predict: DEAD at K~0 & K~2048 (even block), ALIVE at K~1024 (parity flipped to odd) — distance never changes\n")
    rows=[]
    for K in [0,256,512,768,1024,1280,1536,2048]:
        h=0; ns=[]; Ds=[]
        for rep in range(12):
            p,code,n,q,D=build(K, random.Random(123000+K+rep))
            o=chat(p); ok=hc(o,code); h+=ok; ns.append(n); Ds.append(D)
            raw.write(json.dumps({"K":K,"rep":rep,"n":n,"q":q,"D":D,"code":code,"hit":bool(ok),"out":o})+"\n")
        navg=round(sum(ns)/len(ns)); Davg=round(sum(Ds)/len(Ds)); Bn=navg//1024
        rows.append({"pad":K,"n":navg,"D":Davg,"Bn":Bn,"hits":h,"reps":12})
        print(f"  pad K={K:4d}  needle_pos~{navg:5d} (Bn={Bn},par{Bn%2}) D~{Davg:4d}  {h}/12  {'#'*h}{'.'*(12-h)}")
    raw.close()
    json.dump(rows,open(os.path.join(outdir,"results.json"),"w"),indent=2)
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/")
if __name__=="__main__": main()
