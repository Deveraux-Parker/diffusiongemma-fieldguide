#!/usr/bin/env python3
"""
Dead-zone DODGING: can we park lorem-ipsum junk in the dead bands and keep real
content in the safe slots, so a long prompt is fully usable?

Stage 1 (comb): fix fill ~7400, sweep a single needle across absolute positions with
  LOREM-IPSUM filler -> map the safe/dead "teeth" at this length.
Stage 2 (dodge): place 3 needles at once. SAFE-aligned (in the comb's pass teeth) vs
  DEAD-aligned (in the dead teeth), junk everywhere else. Does dodging recover them?
"""
import json, random, os, urllib.request, datetime

BASE="http://127.0.0.1:8001"; MODEL="dg-awq"; FILL=7400
HERE=os.path.dirname(os.path.abspath(__file__))
def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat(u,mt=80): return _post("/v1/chat/completions",{"model":MODEL,
    "messages":[{"role":"user","content":u}],"max_tokens":mt,"temperature":0.0})["choices"][0]["message"]["content"]
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
    for tk in o.replace("="," ").replace("."," ").replace(","," ").split():
        if lev(tk,c)<=1: return True
    return False

LOREM=("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
 "incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud "
 "exercitation ullamco laboris nisi aliquip ex ea commodo consequat duis aute irure "
 "in reprehenderit voluptate velit esse cillum dolore eu fugiat nulla pariatur "
 "excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt "
 "mollit anim id est laborum").split()
COLORS=['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE','GULL','SLATE']
def lorem_line(rng,nwords=14):
    return " ".join(rng.choice(LOREM) for _ in range(nwords)).capitalize()+"."
def needle(color,num): return f"NOTICE: the authorization code is {color}-{num}."
Q=("\n\nThe NOTICE line(s) each contain an authorization code. "
   "List every authorization code you find, in one short sentence.")
TPL=tok_count("\n".join(lorem_line(random.Random(i)) for i in range(20)))/20.0
NLINES=int(FILL/TPL)
def idx_for(pos): return max(1,min(NLINES-1,round(pos/TPL)))

raw=None
def lr_open(outdir):
    global raw; raw=open(os.path.join(outdir,"raw.jsonl"),"a")
def lr(r): raw.write(json.dumps(r)+"\n"); raw.flush()

def comb(reps=5):
    print("== Stage 1: comb (fill~%d, lorem filler, single needle vs position) =="%FILL)
    rows=[]
    for pos in range(200,FILL-150,300):
        idx=idx_for(pos); hits=0; aps=[]
        for rep in range(reps):
            rng=random.Random(90000+pos+rep)
            color=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; code=f"{color}-{num}"
            fl=[lorem_line(rng) for _ in range(NLINES)]
            fl.insert(idx, needle(color,num))
            user="Reference document:\n"+"\n".join(fl)+Q
            ap=tok_count("\n".join(fl[:idx])); out=chat(user); ok=has_code(out,code); hits+=ok; aps.append(ap)
            lr({"stage":"comb","pos_target":pos,"pos":ap,"rep":rep,"code":code,"out":out,"hit":bool(ok)})
        apos=round(sum(aps)/len(aps))
        rows.append({"pos":apos,"hits":hits,"reps":reps})
        print(f"  pos~{apos:5d}  {hits}/{reps}  {'#'*hits}{'.'*(reps-hits)}")
    return rows

def dodge(comb_rows,reps=8):
    print("\n== Stage 2: dodge (3 needles at once; SAFE vs DEAD aligned) ==")
    safe=[r["pos"] for r in sorted(comb_rows,key=lambda r:-r["hits"]) if r["hits"]>=r["reps"]]
    dead=[r["pos"] for r in sorted(comb_rows,key=lambda r:r["hits"]) if r["hits"]==0]
    # spread picks across the prompt
    def spread(cands,k=3):
        cands=sorted(cands);
        if len(cands)<k: return cands
        step=len(cands)/k; return [cands[int(i*step)] for i in range(k)]
    safe3=spread(safe); dead3=spread(dead)
    print(f"  safe slots: {safe3}   dead slots: {dead3}")
    out={}
    for name,slots in [("SAFE_aligned",safe3),("DEAD_aligned",dead3)]:
        if len(slots)<3: print(f"  ({name}: not enough slots found: {slots})"); continue
        per=[0,0,0]; allfound=0
        for rep in range(reps):
            rng=random.Random(95000+rep)
            specs=[]
            for s in slots:
                color=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; specs.append((s,color,f"{num}",f"{color}-{num}"))
            fl=[lorem_line(rng) for _ in range(NLINES)]
            for s,color,num,code in sorted(specs,key=lambda x:-x[0]):
                fl.insert(idx_for(s),needle(color,num))
            user="Reference document:\n"+"\n".join(fl)+Q
            o=chat(user,mt=120)
            found=[has_code(o,code) for _,_,_,code in specs]
            for i,f in enumerate(found): per[i]+=f
            if all(found): allfound+=1
            lr({"stage":"dodge","cond":name,"slots":slots,"rep":rep,
                "codes":[c for *_,c in specs],"out":o,"found":[bool(x) for x in found]})
        print(f"  {name:13s} per-slot {per} /{reps}   all-3 {allfound}/{reps}")
        out[name]={"per_slot":per,"all3":allfound,"reps":reps,"slots":slots}
    return out

def main():
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"deadzone-dodge-{stamp}"); os.makedirs(outdir,exist_ok=True)
    lr_open(outdir)
    print(f"(tokens/line~{TPL:.1f}, lines~{NLINES})")
    c=comb(); d=dodge(c)
    json.dump({"fill":FILL,"comb":c,"dodge":d},open(os.path.join(outdir,"results.json"),"w"),indent=2)
    # summary
    L=[f"# Dead-zone dodging (fill~{FILL}, lorem-ipsum filler)","","## Comb (single needle vs absolute position)","","| pos | hits |","| --- | --- |"]
    for r in c: L.append(f"| {r['pos']} | {r['hits']}/{r['reps']} {'#'*r['hits']}{'.'*(r['reps']-r['hits'])} |")
    L+=["","## Dodge (3 needles at once)",""]
    for name,v in d.items():
        L.append(f"- **{name}** slots {v['slots']}: per-slot {v['per_slot']}/{v['reps']}, all-3 {v['all3']}/{v['reps']}")
    open(os.path.join(outdir,"summary.md"),"w").write("\n".join(L))
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/")

if __name__=="__main__": main()
