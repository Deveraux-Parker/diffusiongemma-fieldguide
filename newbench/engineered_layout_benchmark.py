#!/usr/bin/env python3
"""
Engineered layout: does reserving the dead band for lorem-ipsum and placing real
records in the working zones actually let us use a long prompt reliably?

Dead-phase fill 5500 (dist~1000 is confirmed dead here). Dead band ~ dist 600-1500
before the question. Distinctive RECORD lines, lorem-ipsum junk.

  Stage A (map): single record swept across distance-from-question -> confirm the band.
  Stage B (the scheme): 4 records, all retrieved via "list all codes":
     IN_DEADBAND : 4 records placed at dist 700/950/1200/1450 (inside the band)
     ENGINEERED  : same 4 records relocated to working zones (dist 250/420 recency +
                   dist 1800/2400), lorem left occupying the dead band.
"""
import json, random, os, urllib.request, datetime
BASE="http://127.0.0.1:8001"; MODEL="dg-awq"; FILL=5500; HERE=os.path.dirname(os.path.abspath(__file__))
def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat(u,mt=160): return _post("/v1/chat/completions",{"model":MODEL,
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
    for tk in o.replace("="," ").replace("."," ").replace(","," ").replace("\n"," ").split():
        if lev(tk,c)<=1: return True
    return False
LOREM=("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore "
 "et dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip ex ea "
 "commodo consequat duis aute irure in reprehenderit voluptate velit esse cillum dolore eu fugiat nulla").split()
COLORS=['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE','SLATE','OCHRE']
def lorem(rng,n=12): return " ".join(rng.choice(LOREM) for _ in range(n)).capitalize()+"."
def record(color,num): return f"RECORD: authorization code = {color}-{num}."
Q="\n\nList every authorization code from the RECORD lines, one per line."
TPL=tok_count("\n".join(lorem(random.Random(i)) for i in range(20)))/20.0
NLINES=int(FILL/TPL)
def idx_for(dist): return max(1,min(NLINES-1,NLINES-round(dist/TPL)))
raw=None
def lr(r): raw.write(json.dumps(r)+"\n"); raw.flush()

def build(dists, rng):
    specs=[]
    for dd in dists:
        color=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; specs.append((dd,color,num,f"{color}-{num}"))
    fl=[lorem(rng) for _ in range(NLINES)]
    for dd,color,num,code in sorted(specs,key=lambda x:-idx_for(x[0])):  # insert deepest-index first
        fl.insert(idx_for(dd), record(color,num))
    return "Reference document:\n"+"\n".join(fl)+Q, specs

def stageA(reps=6):
    print("== Stage A: single-record map vs distance (fill %d, lorem) =="%FILL)
    rows=[]
    for dd in [250,500,750,1000,1250,1500,1800,2200,3000,4200]:
        hits=0
        for rep in range(reps):
            rng=random.Random(40000+dd+rep)
            user,specs=build([dd],rng); code=specs[0][3]
            out=chat(user,mt=60); ok=has_code(out,code); hits+=ok
            lr({"stage":"A","dist":dd,"rep":rep,"code":code,"out":out,"hit":bool(ok)})
        rows.append({"dist":dd,"hits":hits,"reps":reps})
        print(f"  dist~{dd:4d}  {hits}/{reps}  {'#'*hits}{'.'*(reps-hits)}")
    return rows

def stageB(reps=10):
    print("\n== Stage B: 4 records — in dead band vs engineered (lorem in dead band) ==")
    layouts={"IN_DEADBAND":[700,950,1200,1450], "ENGINEERED":[250,420,1800,2400]}
    out={}
    for name,dists in layouts.items():
        per=[0,0,0,0]; allf=0
        for rep in range(reps):
            rng=random.Random(48000+rep)
            user,specs=build(dists,rng)
            o=chat(user,mt=160)
            found=[has_code(o,code) for *_,code in specs]
            for i,f in enumerate(found): per[i]+=f
            if all(found): allf+=1
            lr({"stage":"B","layout":name,"dists":dists,"rep":rep,
                "codes":[c for *_,c in specs],"found":[bool(x) for x in found],"out":o})
        tot=sum(per)
        print(f"  {name:11s} per-record {per} /{reps} · total {tot}/{4*reps} · all-4 {allf}/{reps}")
        out[name]={"per":per,"total":tot,"all4":allf,"reps":reps,"dists":dists}
    return out

def main():
    global raw
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"engineered-layout-{stamp}"); os.makedirs(outdir,exist_ok=True)
    raw=open(os.path.join(outdir,"raw.jsonl"),"a")
    print(f"(tokens/line~{TPL:.1f}, lines~{NLINES})")
    a=stageA(); b=stageB()
    json.dump({"fill":FILL,"stageA":a,"stageB":b},open(os.path.join(outdir,"results.json"),"w"),indent=2)
    L=[f"# Engineered layout (fill~{FILL}, lorem junk)","","## Map (single record vs distance)","","| dist | hits |","|---|---|"]
    for r in a: L.append(f"| {r['dist']} | {r['hits']}/{r['reps']} {'#'*r['hits']}{'.'*(r['reps']-r['hits'])} |")
    L+=["","## The scheme (4 records)",""]
    for n,v in b.items(): L.append(f"- **{n}** dists {v['dists']}: total {v['total']}/{4*v['reps']}, all-4 {v['all4']}/{v['reps']}, per-record {v['per']}")
    open(os.path.join(outdir,"summary.md"),"w").write("\n".join(L))
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/")
if __name__=="__main__": main()
