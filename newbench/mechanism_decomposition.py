#!/usr/bin/env python3
"""
MECHANISM DECOMPOSITION — what actually governs the retrieval dead zone?

Candidates: distance-from-question D, absolute position n, question position q,
or block alignment (n mod B). All scans use IDENTICAL filler/record/question so
the only things that change are positions.

  MAP2D : q x D on a 500-token lattice (so n=q-D collides across many cells).
          Project along q, D, n to see which axis the dead cells line up with.
  PERIOD: D fixed at 1000, q swept fine (step 120) -> exact oscillation period.
  FIXN  : absolute position n fixed (~2600), D varied by padding AFTER the needle
          -> if flat, the hazard is NOT about distance; if a valley, it IS.

Every trial: actual measured n, D, q (tokenized), plus the model output.
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
def has_code(o,c):
    o=(o or "").upper(); c=c.upper()
    if c in o:return True
    for tk_ in o.replace("="," ").replace("."," ").replace(","," ").replace("\n"," ").split():
        if lev(tk_,c)<=1:return True
    return False
LOREM=("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et "
 "dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo").split()
COLORS=["MAGENTA","CYAN","AMBER","INDIGO","CRIMSON","JADE","SLATE","OCHRE"]
HEADER="OPERATIONS LOG\n\n"
def lorem(rng,n=12): return " ".join(rng.choice(LOREM) for _ in range(n)).capitalize()+"."
def rec(c,n): return f"RECORD: secret token = {c}-{n}."
Q="\n\nState the secret token from the RECORD line, in one short sentence."
TPL=tok("\n".join(lorem(random.Random(i)) for i in range(20)))/20.0
QTOK=tok(Q); HTOK=tok(HEADER)
raw=None
def lr(r): raw.write(json.dumps(r)+"\n"); raw.flush()

def trial(seed, n_target, q_target):
    """Place a record so it sits ~n_target tokens from the document start; total ~q_target."""
    rng=random.Random(seed); c=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; code=f"{c}-{num}"
    nlines=max(2,round((q_target-QTOK-HTOK)/TPL))
    nidx=max(1,min(nlines-1,round((n_target-HTOK)/TPL)))
    fl=[lorem(rng) for _ in range(nlines)]
    fl.insert(nidx, rec(c,num))
    prompt=HEADER+"\n".join(fl)+Q
    # measured positions
    n_act=tok(HEADER+"\n".join(fl[:nidx]))            # tokens before the record
    q_act=tok(prompt)                                  # total prompt tokens
    D_act=q_act-n_act                                  # record -> end (incl. question)
    out=chat(prompt); hit=has_code(out,code)
    lr({"seed":seed,"n":n_act,"q":q_act,"D":D_act,"code":code,"hit":bool(hit),"out":out})
    return n_act,q_act,D_act,hit

def map2d(reps=8):
    print("== MAP2D (q x D, 500-lattice; n=q-D collides) ==")
    QS=[4000,4500,5000,5500,6000,6500,7000]; DS=[500,1000,1500,2000,2500]
    rows=[]
    for q in QS:
        line=[]
        for D in DS:
            if q-D<400: line.append(None); continue
            h=0; nn=[]; dd=[]; qq=[]
            for rep in range(reps):
                na,qa,Da,hit=trial(700000+q*10+D+rep, q-D, q)
                h+=hit; nn.append(na); dd.append(Da); qq.append(qa)
            cell={"q":round(sum(qq)/len(qq)),"D":round(sum(dd)/len(dd)),"n":round(sum(nn)/len(nn)),"hits":h,"reps":reps}
            line.append(cell)
            print(f"  q~{cell['q']:5d} D~{cell['D']:5d} n~{cell['n']:5d}  {h}/{reps}  {'#'*h}{'.'*(reps-h)}")
        rows.append(line)
    return {"QS":QS,"DS":DS,"rows":rows}

def period(reps=8):
    print("\n== PERIOD (D=1000, fine q step 120) ==")
    out=[]
    for q in range(4000,7561,120):
        h=0; nn=[]
        for rep in range(reps):
            na,qa,Da,hit=trial(800000+q+rep, q-1000, q); h+=hit; nn.append(na)
        out.append({"q":q,"n":round(sum(nn)/len(nn)),"hits":h,"reps":reps})
        print(f"  q~{q:5d} n~{round(sum(nn)/len(nn)):5d}  {h}/{reps}  {'#'*h}{'.'*(reps-h)}")
    return out

def fixn(reps=8):
    print("\n== FIXN (absolute position fixed ~2600, vary distance D) ==")
    out=[]
    for D in [400,600,800,1000,1200,1400,1700,2100,2600,3200,4000]:
        h=0; nn=[]; dd=[]
        for rep in range(reps):
            na,qa,Da,hit=trial(900000+D+rep, 2600, 2600+D); h+=hit; nn.append(na); dd.append(Da)
        out.append({"n":round(sum(nn)/len(nn)),"D":round(sum(dd)/len(dd)),"hits":h,"reps":reps})
        print(f"  n~{round(sum(nn)/len(nn)):5d} D~{round(sum(dd)/len(dd)):5d}  {h}/{reps}  {'#'*h}{'.'*(reps-h)}")
    return out

def main():
    global raw
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"mechanism-decomp-{stamp}"); os.makedirs(outdir,exist_ok=True)
    raw=open(os.path.join(outdir,"raw.jsonl"),"a")
    print(f"(tokens/line~{TPL:.1f}, Qtok={QTOK}, Htok={HTOK})")
    res={"tpl":TPL}
    res["map2d"]=map2d(); res["period"]=period(); res["fixn"]=fixn()
    json.dump(res,open(os.path.join(outdir,"results.json"),"w"),indent=2,default=str)
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/  (raw.jsonl, results.json)")
    print("OUTDIR="+outdir)
if __name__=="__main__": main()
