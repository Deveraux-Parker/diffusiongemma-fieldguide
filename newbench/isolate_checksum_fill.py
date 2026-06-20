#!/usr/bin/env python3
"""
Controlled isolation: is the dead zone closed by the needle's trailing checksum
landmark, or is it fill/position dependent?

Fixed: distance ~1000 tok from question (the contested point), sentence-form question.
Vary: needle_type {plain, checksum} x fill {4500,5500,6500,7300}. 10 reps.
plain    = ...status=CRITICAL incident_code=CODE
checksum = ...status=CRITICAL incident_code=CODE checksum=CK-####
"""
import json, random, os, urllib.request, datetime

BASE="http://127.0.0.1:8001"; MODEL="dg-awq"; REPS=10; DIST=1000
HERE=os.path.dirname(os.path.abspath(__file__))

def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat(u): return _post("/v1/chat/completions",{"model":MODEL,
    "messages":[{"role":"user","content":u}],"max_tokens":60,"temperature":0.0})["choices"][0]["message"]["content"]
def lev(a,b):
    if a==b: return 0
    m,n=len(a),len(b); d=list(range(n+1))
    for i in range(1,m+1):
        prev=d[0]; d[0]=i
        for j in range(1,n+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=t
    return d[n]
def has_code(out,code):
    o=(out or "").upper(); c=code.upper()
    if c in o: return True
    for tk in o.replace("="," ").replace("."," ").split():
        if lev(tk,c)<=1: return True
    return False

COLORS=['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE']
def nominal(i,rng):
    return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C "
            f"pressure={rng.uniform(99,103):.1f}kPa flow={rng.uniform(8,16):.1f}L/min status=nominal")
def needle(i,color,num,ck,kind):
    base=f"[t={i:05d}] ANOMALY sensor_temp=87.3C status=CRITICAL incident_code={color}-{num}"
    return base+(f" checksum=CK-{ck}" if kind=="checksum" else "")
Q="\n\nThe one line with status=CRITICAL has an incident_code. State that incident_code in one short sentence."
TPL=tok_count("\n".join(nominal(i,random.Random(i)) for i in range(20)))/20.0

def main():
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"isolate-checksum-fill-{stamp}"); os.makedirs(outdir,exist_ok=True)
    rawf=open(os.path.join(outdir,"raw.jsonl"),"a")
    def lr(r): rawf.write(json.dumps(r)+"\n"); rawf.flush()
    print(f"== Isolation: needle-type x fill at dist~{DIST}, {REPS} reps ==")
    grid={}
    for kind in ["plain","checksum"]:
        for fill in [4500,5500,6500,7300]:
            nlines=int(fill/TPL); idx=max(1,min(nlines-1,nlines-round(DIST/TPL)))
            hits=0; ad=[]; ap=[]
            for rep in range(REPS):
                rng=random.Random(33000+fill+rep)  # SAME seed across kinds -> identical filler
                color=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; ck=f"{rng.randint(1000,9999)}"
                code=f"{color}-{num}"
                fl=[nominal(i,rng) for i in range(1,nlines+1)]
                fl.insert(idx,needle(900000+rep,color,num,ck,kind))
                user="Telemetry log:\n"+"\n".join(fl)+Q
                dist=tok_count("\n".join(fl[idx+1:])+Q); pos=tok_count("\n".join(fl[:idx]))
                out=chat(user); ok=has_code(out,code); hits+=ok; ad.append(dist); ap.append(pos)
                lr({"kind":kind,"fill":fill,"rep":rep,"dist":dist,"pos":pos,"code":code,"out":out,"hit":bool(ok)})
            grid[(kind,fill)]=hits
            print(f"  {kind:8s} fill={fill:5d} dist~{round(sum(ad)/len(ad)):4d} pos~{round(sum(ap)/len(ap)):5d}  {hits}/{REPS}  {'#'*hits}{'.'*(REPS-hits)}")
    rawf.close()
    # summary table
    L=["# Isolation: checksum landmark vs fill (dist~1000)","",f"- {REPS} reps · dist~{DIST} · same filler seed across needle types","",
       "| needle | 4500 | 5500 | 6500 | 7300 |","| --- | --- | --- | --- | --- |"]
    for kind in ["plain","checksum"]:
        L.append(f"| {kind} | "+" | ".join(f"{grid[(kind,f)]}/{REPS}" for f in [4500,5500,6500,7300])+" |")
    open(os.path.join(outdir,"summary.md"),"w").write("\n".join(L))
    json.dump({"grid":{f"{k}@{fl}":v for (k,fl),v in grid.items()}},
              open(os.path.join(outdir,"results.json"),"w"),indent=2)
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/")

if __name__=="__main__":
    main()
