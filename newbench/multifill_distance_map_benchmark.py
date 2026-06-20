#!/usr/bin/env python3
"""
Multi-fill distance map — adjudicates the two-mechanism model of the dead zone.

Two-mechanism hypothesis:
  recency coverage  = sliding-window layers -> anchored to DISTANCE from the question
                      (constant near-edge ~600-1024 tok, independent of fill)
  primacy coverage  = full-attention layers -> anchored to absolute POSITION near the
                      start (constant far-edge ~first ~1800 tok, independent of fill)
  dead zone         = the gap between them -> WIDENS with fill, VANISHES below ~1.5-2k.

Test: sweep needle absolute token-distance-from-question at fills 2k/4.5k/6.5k/7.3k.
For every trial record both distance AND absolute position. Then check whether the
near edge is constant in distance and the far edge is constant in position.

Exact-code retrieval, sentence-form question (avoids the terse-dropout quirk).
Robust: raw log opened append-mode inside main (no import side effects).
"""
import json, random, os, math, urllib.request, datetime

BASE="http://127.0.0.1:8001"; MODEL="dg-awq"; REPS=8
HERE=os.path.dirname(os.path.abspath(__file__))

def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat(user,max_tokens=60):
    return _post("/v1/chat/completions",{"model":MODEL,
        "messages":[{"role":"user","content":user}],"max_tokens":max_tokens,
        "temperature":0.0})["choices"][0]["message"]["content"]

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
    for tk in o.replace("="," ").replace(","," ").replace("."," ").split():
        if lev(tk,c)<=1: return True
    return False
def wilson(k,n,z=1.96):
    if n==0: return (0.0,0.0)
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(max(0,c-h),2), round(min(1,c+h),2))

COLORS=['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE']
def nominal(i,rng):
    return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C "
            f"pressure={rng.uniform(99,103):.1f}kPa flow={rng.uniform(8,16):.1f}L/min status=nominal")
def crit_line(i,color,num,ck):
    return (f"[t={i:05d}] ANOMALY sensor_temp=87.3C status=CRITICAL "
            f"incident_code={color}-{num} checksum=CK-{ck}")
Q="\n\nThe one line with status=CRITICAL has an incident_code. State that incident_code in one short sentence."

TPL=tok_count("\n".join(nominal(i,random.Random(i)) for i in range(20)))/20.0

# distance ladders per fill (token-distance from the question); capped below fill
FILLS={
 2000:[150,400,700,1000,1300,1600,1800],
 4500:[150,400,700,1000,1400,1900,2600,3500,4250],
 6500:[150,400,700,1000,1300,1700,2200,3000,4200,5500,6250],
 7300:[150,400,700,1000,1300,1700,2200,3000,4200,5500,7050],
}
PASS=0.75  # hit-rate threshold to call a cell "covered"

def run(lr):
    rows=[]
    for fill,dists in FILLS.items():
        nlines=int(fill/TPL)
        for D in dists:
            idx=max(1,min(nlines-1,nlines-round(D/TPL)))
            hits=0; ad=[]; ap=[]
            for rep in range(REPS):
                rng=random.Random(7000+fill+D+rep)
                color=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; ck=f"{rng.randint(1000,9999)}"
                code=f"{color}-{num}"
                fl=[nominal(i,rng) for i in range(1,nlines+1)]
                fl.insert(idx, crit_line(900000+rep,color,num,ck))
                user="Telemetry log:\n"+"\n".join(fl)+Q
                dist=tok_count("\n".join(fl[idx+1:])+Q)          # actual distance from question
                pos=tok_count("Telemetry log:\n"+"\n".join(fl[:idx])) # actual absolute position from start
                out=chat(user); ok=has_code(out,code); hits+=ok; ad.append(dist); ap.append(pos)
                lr({"fill":fill,"target_dist":D,"rep":rep,"dist":dist,"pos":pos,
                    "code":code,"out":out,"hit":bool(ok)})
            adist=round(sum(ad)/len(ad)); apos=round(sum(ap)/len(ap))
            lo,hi=wilson(hits,REPS)
            rows.append({"fill":fill,"dist":adist,"pos":apos,"hits":hits,"reps":REPS,
                         "rate":round(hits/REPS,2),"ci":[lo,hi]})
            bar="#"*hits+"."*(REPS-hits)
            print(f"  fill={fill:5d} dist~{adist:5d} pos~{apos:5d}  {hits}/{REPS} [{bar}]")
    return rows

def analyze(rows):
    """Per fill: find the failing band (cells below PASS), report its distance & position edges."""
    out=[]
    byfill={}
    for r in rows: byfill.setdefault(r["fill"],[]).append(r)
    for fill,rs in byfill.items():
        rs=sorted(rs,key=lambda r:r["dist"])
        fails=[r for r in rs if r["rate"]<PASS]
        if not fails:
            out.append({"fill":fill,"deadzone":None}); continue
        near=min(f["dist"] for f in fails); far=max(f["dist"] for f in fails)
        near_pos=max(f["pos"] for f in fails); far_pos=min(f["pos"] for f in fails)
        out.append({"fill":fill,
            "deadzone_dist":[near,far], "deadzone_pos":[far_pos,near_pos],
            "n_fail_cells":len(fails)})
    return out

def main():
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"multifill-distance-map-{stamp}"); os.makedirs(outdir,exist_ok=True)
    rawf=open(os.path.join(outdir,"raw.jsonl"),"a")
    def lr(r): rawf.write(json.dumps(r)+"\n"); rawf.flush()
    print(f"== Multi-fill distance map ({REPS} reps, tokens/line~{TPL:.1f}) ==")
    rows=run(lr); rawf.close()
    edges=analyze(rows)
    json.dump({"started":stamp,"reps":REPS,"pass_threshold":PASS,"rows":rows,"edges":edges},
              open(os.path.join(outdir,"results.json"),"w"),indent=2)
    # summary
    L=["# Multi-fill distance map — two-mechanism test","",
       f"- {REPS} reps/cell · model `{MODEL}` · pass threshold {PASS:.0%} · tokens/line ~{TPL:.1f}","",
       "## Dead-zone edges per fill","",
       "| fill | dead zone (distance from Q) | dead zone (absolute position) | # fail cells |",
       "| --- | --- | --- | --- |"]
    for e in edges:
        if e.get("deadzone_dist") is None and e.get("deadzone") is None:
            L.append(f"| {e['fill']} | — none — | — none — | 0 |")
        else:
            dd=e["deadzone_dist"]; pp=e["deadzone_pos"]
            L.append(f"| {e['fill']} | {dd[0]}–{dd[1]} tok | pos {pp[0]}–{pp[1]} | {e['n_fail_cells']} |")
    L+=["","## Prediction check","",
        "- two-mechanism model expects: **near edge ~constant in DISTANCE** across fills,",
        "  **far edge ~constant in POSITION** across fills, dead zone **widens with fill**, **absent at 2k**.",
        "","## Full grid (hit-rate)",""]
    byfill={}
    for r in rows: byfill.setdefault(r["fill"],[]).append(r)
    for fill in sorted(byfill):
        L.append(f"**fill {fill}**  "+"  ".join(
            f"`d{r['dist']}:{r['hits']}/{r['reps']}`" for r in sorted(byfill[fill],key=lambda x:x['dist'])))
        L.append("")
    open(os.path.join(outdir,"summary.md"),"w").write("\n".join(L))
    print("\n--- dead-zone edges ---")
    for e in edges: print("  ",e)
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/ (results.json, summary.md, raw.jsonl)")

if __name__=="__main__":
    main()
