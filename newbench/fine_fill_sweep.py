#!/usr/bin/env python3
"""
Fine fill-sweep at fixed distance(s) — characterize the pass/fail OSCILLATION and
discriminate whether it tracks the QUERY position (= fill) or the NEEDLE position
(= fill - distance).

Fixed distance D (needle stays D tokens from the question); sweep total fill.
Two distances (1000, 1500): if dead bands line up by FILL, the effect is anchored to
query/end position; if they're offset by ~500, it's anchored to the needle's absolute
position. Period of the oscillation hints at the attention block size.
"""
import json, random, os, urllib.request, datetime

BASE="http://127.0.0.1:8001"; MODEL="dg-awq"; REPS=6
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
def needle(i,color,num):
    return f"[t={i:05d}] ANOMALY sensor_temp=87.3C status=CRITICAL incident_code={color}-{num}"
Q="\n\nThe one line with status=CRITICAL has an incident_code. State that incident_code in one short sentence."
TPL=tok_count("\n".join(nominal(i,random.Random(i)) for i in range(20)))/20.0
FILLS=list(range(4000,7601,300))
DISTS=[1000,1500]

def main():
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"fine-fill-sweep-{stamp}"); os.makedirs(outdir,exist_ok=True)
    rawf=open(os.path.join(outdir,"raw.jsonl"),"a")
    def lr(r): rawf.write(json.dumps(r)+"\n"); rawf.flush()
    print(f"== Fine fill-sweep ({REPS} reps, tokens/line~{TPL:.1f}) ==")
    series={D:[] for D in DISTS}
    for D in DISTS:
        print(f"-- distance ~{D} --")
        for fill in FILLS:
            nlines=int(fill/TPL); idx=max(1,min(nlines-1,nlines-round(D/TPL)))
            hits=0; ad=[]; ap=[]
            for rep in range(REPS):
                rng=random.Random(51000+fill+rep)
                color=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; code=f"{color}-{num}"
                fl=[nominal(i,rng) for i in range(1,nlines+1)]
                fl.insert(idx,needle(900000+rep,color,num))
                user="Telemetry log:\n"+"\n".join(fl)+Q
                dist=tok_count("\n".join(fl[idx+1:])+Q); pos=tok_count("\n".join(fl[:idx]))
                out=chat(user); ok=has_code(out,code); hits+=ok; ad.append(dist); ap.append(pos)
                lr({"D":D,"fill":fill,"rep":rep,"dist":dist,"pos":pos,"code":code,"out":out,"hit":bool(ok)})
            adist=round(sum(ad)/len(ad)); apos=round(sum(ap)/len(ap))
            series[D].append({"fill":fill,"dist":adist,"pos":apos,"hits":hits})
            print(f"  fill={fill:5d} needlePos~{apos:5d}  {hits}/{REPS}  {'#'*hits}{'.'*(REPS-hits)}")
    rawf.close()
    json.dump({"reps":REPS,"series":series},open(os.path.join(outdir,"results.json"),"w"),indent=2)
    # summary with both axes
    L=["# Fine fill-sweep — oscillation characterization","",f"- {REPS} reps · distances {DISTS} · tokens/line ~{TPL:.1f}",""]
    for D in DISTS:
        L.append(f"## distance ~{D}")
        L.append("| fill | needle pos | hits | |")
        L.append("| --- | --- | --- | --- |")
        for r in series[D]:
            L.append(f"| {r['fill']} | {r['pos']} | {r['hits']}/{REPS} | {'#'*r['hits']}{'.'*(REPS-r['hits'])} |")
        L.append("")
    open(os.path.join(outdir,"summary.md"),"w").write("\n".join(L))
    # quick phase report: list FAIL fills (hits<=REPS//3) per distance
    print("\n--- FAIL fills (hits <= %d) ---"%(REPS//3))
    for D in DISTS:
        fails=[(r['fill'],r['pos']) for r in series[D] if r['hits']<=REPS//3]
        print(f"  dist~{D}: fails at fills {[f for f,_ in fails]}  (needle pos {[p for _,p in fails]})")
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/")

if __name__=="__main__":
    main()
