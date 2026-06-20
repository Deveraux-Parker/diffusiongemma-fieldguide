#!/usr/bin/env python3
"""
Binding-failure vs coverage-gap adjudication.

The dead zone (~1000 tok before the question) gives 0/8 exact-code retrieval, but
misses are dominated by needle_neighbor_value / code_fragment — suggesting the model
REACHES the region but cannot bind the exact high-entropy string. This benchmark
varies the QUERY along an information-content gradient at a FIXED dead-zone distance:

  coverage-gap  predicts: every probe fails (model can't see the region)
  binding-fail  predicts: coarse/low-entropy probes pass, only exact high-entropy fails

Probes (same buried CRITICAL line, fill ~5.5k, needle ~1000 tok from question):
  presence       "is there a CRITICAL line?"            (1 bit)
  neighbor_temp  "what temp did it hit?"  -> 87.3        (co-located, mid-entropy)
  color_prefix   "what color starts the code?"          (~2.6 bits, 1 of 6)
  exact_code     "state the full incident_code"         (~high entropy)  EXPECT FAIL
  exact_twice    "state it twice"                       (redundancy mitigation)
Control:
  exact_recency  exact_code but needle ~220 tok (safe)  -> question is answerable

Robust: raw log opened append-mode inside main (no import side effects).
"""
import json, random, os, math, urllib.request, datetime

BASE="http://127.0.0.1:8001"; MODEL="dg-awq"; REPS=10; FILL=5500
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
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(max(0,c-h),2), round(min(1,c+h),2))

COLORS=['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE']
def nominal(i,rng):
    return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C "
            f"pressure={rng.uniform(99,103):.1f}kPa flow={rng.uniform(8,16):.1f}L/min status=nominal")
def crit_line(i,color,num,ck):
    return (f"[t={i:05d}] ANOMALY sensor_temp=87.3C pressure=142.7kPa flow=0.0L/min "
            f"status=CRITICAL incident_code={color}-{num} checksum=CK-{ck}")

TPL=tok_count("\n".join(nominal(i,random.Random(i)) for i in range(20)))/20.0
NLINES=int(FILL/TPL)
def idx_for(dist): return max(1,min(NLINES-1,NLINES-round(dist/TPL)))

PROBES={
 "presence":     ("Does the log contain any line with status=CRITICAL? Answer yes or no in one short sentence.", "presence"),
 "neighbor_temp":("What sensor_temp value does the status=CRITICAL line report? Answer in one short sentence.", "temp"),
 "color_prefix": ("The incident_code on the status=CRITICAL line begins with a COLOR word. Which color is it? Answer in one short sentence.", "color"),
 "exact_code":   ("State the full incident_code on the status=CRITICAL line. Answer in one short sentence.", "code"),
 "exact_twice":  ("State the full incident_code on the status=CRITICAL line, and repeat it a second time. Answer in one short sentence.", "code"),
}

def classify_miss(out,color,num):
    o=(out or "").strip()
    if o=="" : return "empty"
    if "87.3" in o or "ANOMALY" in o.upper() or "CRITICAL" in o.upper(): return "needle_neighbor_value"
    if color in o.upper() or num[:2] in o or "-" in o: return "code_fragment"
    if any(ch.isdigit() for ch in o): return "wrong_number"
    return "other"

def grade(out,kind,color,num,code):
    o=(out or "").lower()
    if kind=="presence": return ("yes" in o) and ("no critical" not in o) and ("not found" not in o)
    if kind=="temp":     return "87.3" in (out or "")
    if kind=="color":    return color.lower() in o
    if kind=="code":     return has_code(out,code)
    return False

def run(lr):
    rows=[]
    # main probes at dead-zone distance
    conds=[(name,q,kind,1000) for name,(q,kind) in PROBES.items()]
    conds.append(("exact_recency",PROBES["exact_code"][0],"code",220))  # control: safe zone
    for name,q,kind,dist in conds:
        hits=0; miss_types={}
        for rep in range(REPS):
            rng=random.Random(424242+rep)  # same buried record across conds for comparability
            color=rng.choice(COLORS); num=f"{rng.randint(1000,9999)}"; ck=f"{rng.randint(1000,9999)}"
            code=f"{color}-{num}"
            fl=[nominal(i,rng) for i in range(1,NLINES+1)]
            fl.insert(idx_for(dist), crit_line(900000+rep, color, num, ck))
            user="Telemetry log:\n"+"\n".join(fl)+"\n\n"+q
            out=chat(user); ok=grade(out,kind,color,num,code); hits+=ok
            if not ok and kind=="code":
                mt=classify_miss(out,color,num); miss_types[mt]=miss_types.get(mt,0)+1
            lr({"cond":name,"kind":kind,"dist":dist,"rep":rep,"code":code,
                "color":color,"out":out,"hit":bool(ok)})
        lo,hi=wilson(hits,REPS)
        rows.append({"cond":name,"kind":kind,"dist":dist,"hits":hits,"reps":REPS,
                     "rate":round(hits/REPS,2),"ci":[lo,hi],"miss_types":miss_types})
        bar="#"*hits+"."*(REPS-hits)
        print(f"  {name:14s} kind={kind:8s} d~{dist:4d}  {hits}/{REPS} [{bar}] "
              f"CI[{lo:.2f},{hi:.2f}] {miss_types if miss_types else ''}")
    return rows

def main():
    stamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir=os.path.join(HERE,f"binding-vs-coverage-{stamp}"); os.makedirs(outdir,exist_ok=True)
    rawf=open(os.path.join(outdir,"raw.jsonl"),"a")
    def lr(r): rawf.write(json.dumps(r)+"\n"); rawf.flush()
    print(f"== Binding-vs-coverage (fill~{FILL}, dead-zone d~1000, {REPS} reps) ==")
    print(f"   tokens/line~{TPL:.1f}, filler lines~{NLINES}")
    rows=run(lr); rawf.close()
    json.dump({"started":stamp,"fill":FILL,"reps":REPS,"rows":rows},
              open(os.path.join(outdir,"results.json"),"w"),indent=2)
    # summary.md
    L=["# Binding-failure vs coverage-gap","",f"- fill `{FILL}` tok · dead-zone d~1000 · {REPS} reps · model `{MODEL}`","",
       "| probe | info content | dist | hit rate | 95% CI | miss types |","| --- | --- | --- | --- | --- | --- |"]
    label={"presence":"coarse (1 bit)","neighbor_temp":"co-located value","color_prefix":"low (1 of 6)",
           "exact_code":"high (exact)","exact_twice":"high (x2)","exact_recency":"high (SAFE control)"}
    for r in rows:
        mt=", ".join(f"{k}:{v}" for k,v in r["miss_types"].items()) or "—"
        L.append(f"| {r['cond']} | {label.get(r['cond'],'')} | {r['dist']} | {r['hits']}/{r['reps']} "
                 f"({r['rate']:.2f}) | [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | {mt} |")
    L+=["","## Read","",
        "If coarse/low-entropy probes pass while exact_code fails (and the safe-zone control passes),",
        "the dead zone is a BINDING failure under attention geometry, not blindness to the region."]
    open(os.path.join(outdir,"summary.md"),"w").write("\n".join(L))
    print(f"\nsaved -> {os.path.relpath(outdir,HERE)}/ (results.json, summary.md, raw.jsonl)")

if __name__=="__main__":
    main()
