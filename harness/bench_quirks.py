#!/usr/bin/env python3
"""
DiffusionGemma — Wave 1 quirks & weaknesses battery (8k server, sequential).

P1 dead-zone threshold : retrieval @ worst depth (0.8), sweep fill -> where does it break?
P2 computed retrieval  : max / first-threshold-crossing / count, front-loaded vs buried
P3 top-summary rescue  : does a top-of-context POINTER rescue a dead-zone needle?
P4 determinism         : same prompt, temp 0, x5 -> identical?
P5 empty-dropout fix   : reproduce empties @ depth 0.95; does min_tokens cure them?
P6 INT4 string fidelity: echo random exact codes -> char-drop rate (isolates quant artifact)
"""
import json, time, random, os, urllib.request

BASE="http://localhost:8001"; MODEL="dg-awq"; MAXLEN=8000
OUTDIR=os.path.dirname(os.path.abspath(__file__))

def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]
def chat(user,max_tokens=40,**kw):
    pl={"model":MODEL,"messages":[{"role":"user","content":user}],
        "max_tokens":max_tokens,"temperature":0.0}; pl.update(kw)
    r=_post("/v1/chat/completions",pl)
    return r["choices"][0]["message"]["content"]

def lev(a,b):
    if a==b: return 0
    m,n=len(a),len(b); d=list(range(n+1))
    for i in range(1,m+1):
        prev=d[0]; d[0]=i
        for j in range(1,n+1):
            t=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=t
    return d[n]
def found(out,code):  # lenient: ignores INT4 single-char drops
    o=(out or "").upper(); c=code.upper()
    if c in o: return True
    for tk in o.replace("="," ").replace(","," ").split():
        if lev(tk,c)<=1: return True
    return c[1:] in o

raw=open(os.path.join(OUTDIR,"raw_quirks.jsonl"),"w")
def lr(r): raw.write(json.dumps(r)+"\n"); raw.flush()

def nominal(i,rng,cap=79.0):
    return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C "
            f"pressure={rng.uniform(99,103):.1f}kPa flow={rng.uniform(8,16):.1f}L/min status=nominal")
def filler(target,rng):
    L=[nominal(i,rng) for i in range(1,700)]
    tpl=tok_count("\n".join(L[:20]))/20.0
    return L[:min(len(L),max(1,int(target/tpl)))]
def rand_code(rng):
    return f"{rng.choice(['MAGENTA','CYAN','AMBER','INDIGO','CRIMSON','JADE'])}-{rng.randint(1000,9999)}"
def needle(i,code):
    return f"[t={i:05d}] ANOMALY sensor_temp=87.3C status=CRITICAL incident_code={code}"

# ---------- P1: dead-zone threshold (depth 0.8) -----------------------------
def P1():
    print("== P1: dead-zone threshold (retrieval @ depth 0.8) ==")
    out=[]
    for f in [2000,3000,4000,5000,6000,7000]:
        h=0
        for tr in range(5):
            rng=random.Random(11000+f+tr); code=rand_code(rng)
            fl=filler(f,rng); pos=int(0.8*len(fl)); fl.insert(pos,needle(900000+tr,code))
            user=("Telemetry log:\n"+"\n".join(fl)+
                  "\n\nReply with ONLY the incident_code from the one status=CRITICAL line.")
            o=chat(user); ok=found(o,code); h+=ok
            lr({"p":"P1","fill":f,"trial":tr,"code":code,"out":o,"hit":ok})
        print(f"  fill={f:5d} kv={f/MAXLEN:.2f} hit={h}/5"); out.append({"fill":f,"hit":h})
    return out

# ---------- P2: computed retrieval (front-loaded vs buried) ------------------
def build_series(rng):
    """Return (lines, gt) with controlled ground truth."""
    M=60; lines=[]; crit_ts=[]; crit_temps=[]
    crit_idx=sorted(rng.sample(range(8,M),3))
    peaks=[round(rng.uniform(85,95),1) for _ in range(3)]
    for k,i in enumerate(range(1,M+1)):
        ts=f"t={i:05d}"
        if i-1 in crit_idx:
            j=crit_idx.index(i-1); tp=peaks[j]
            lines.append(f"[{ts}] sensor_temp={tp:.1f}C status=CRITICAL")
            crit_ts.append(ts); crit_temps.append(tp)
        else:
            lines.append(f"[{ts}] sensor_temp={rng.uniform(18,26):.1f}C status=nominal")
    gt={"max":max(crit_temps),"first_cross":crit_ts[0],"count":3}
    return lines,gt

def P2():
    print("== P2: computed retrieval (front-loaded vs buried @0.8/6k) ==")
    res={}
    for cond in ["front","buried"]:
        acc={"max":0,"first_cross":0,"count":0}
        for tr in range(5):
            rng=random.Random(22000+tr); series,gt=build_series(rng)
            block="\n".join(series)
            if cond=="front":
                ctx="SERIES (analyze this):\n"+block
            else:
                fl=filler(6000,random.Random(22500+tr)); pos=int(0.8*len(fl))
                fl[pos:pos]=series; ctx="Telemetry stream:\n"+"\n".join(fl)
            qs={"max":("What was the highest sensor_temp reading? Reply with ONLY the number.",f"{gt['max']:.1f}"),
                "first_cross":("At what timestamp did sensor_temp FIRST exceed 80.0C? Reply ONLY the timestamp like t=00042.",gt["first_cross"]),
                "count":("How many readings have status=CRITICAL? Reply ONLY the number.","3")}
            for key,(q,ans) in qs.items():
                o=chat(ctx+"\n\n"+q,max_tokens=24)
                ok=ans.lower() in (o or "").lower(); acc[key]+=ok
                lr({"p":"P2","cond":cond,"trial":tr,"q":key,"gt":ans,"out":o,"hit":ok})
        print(f"  {cond:6s}: max={acc['max']}/5 first_cross={acc['first_cross']}/5 count={acc['count']}/5")
        res[cond]=acc
    return res

# ---------- P3: top-summary rescue of a dead-zone needle ---------------------
def P3():
    print("== P3: does a top-of-context pointer rescue a buried (0.8/6k) needle? ==")
    res={}
    for cond in ["no_pointer","pointer"]:
        h=0
        for tr in range(5):
            rng=random.Random(33000+tr); code=rand_code(rng)
            fl=filler(6000,rng); pos=int(0.8*len(fl)); fl.insert(pos,needle(900000+tr,code))
            body="\n".join(fl)
            head=("" if cond=="no_pointer" else
                  "NOTE: exactly one line below has status=CRITICAL with an incident_code; "
                  "you will be asked to report that code.\n\n")
            user=head+"Telemetry log:\n"+body+"\n\nReply with ONLY that incident_code."
            o=chat(user); ok=found(o,code); h+=ok
            lr({"p":"P3","cond":cond,"trial":tr,"code":code,"out":o,"hit":ok})
        print(f"  {cond:10s}: hit={h}/5"); res[cond]=h
    return res

# ---------- P4: determinism at temp 0 ---------------------------------------
def P4():
    print("== P4: determinism (same prompt, temp 0, x5) ==")
    rng=random.Random(44); fl=filler(1500,rng)
    user="Telemetry:\n"+"\n".join(fl)+"\n\nIn one sentence, summarize the overall status."
    outs=[chat(user,max_tokens=80) for _ in range(5)]
    uniq=len(set(outs))
    for i,o in enumerate(outs): lr({"p":"P4","run":i,"out":o})
    print(f"  unique outputs: {uniq}/5  ({'DETERMINISTIC' if uniq==1 else 'NON-deterministic'})")
    return {"unique":uniq,"sample":outs[0][:120]}

# ---------- P5: empty-dropout fix -------------------------------------------
def P5():
    print("== P5: empty-output dropout @0.95 — does min_tokens cure it? ==")
    res={}
    for cond,extra in [("plain",{}),("min_tokens",{"min_tokens":30})]:
        empt=0
        for tr in range(8):
            rng=random.Random(55000+tr); fl=filler(6000,rng); pos=int(0.95*len(fl))
            fl.insert(pos,"EPISODE: device DV-7741 peaked 91.3C at t=00412 (coolant_pump_failure).")
            user=("Logs+one EPISODE line:\n"+"\n".join(fl)+
                  "\n\nWrite a one-line incident summary naming the device and peak temp.")
            o=chat(user,max_tokens=120,**extra)
            e=not (o and o.strip()); empt+=e
            lr({"p":"P5","cond":cond,"trial":tr,"empty":e,"out":o})
        print(f"  {cond:10s}: empties={empt}/8"); res[cond]=empt
    return res

# ---------- P6: INT4 exact-string fidelity ----------------------------------
def P6():
    print("== P6: INT4 exact-string fidelity (echo random codes) ==")
    exact=0; n=20; errs=[]
    for tr in range(n):
        rng=random.Random(66000+tr)
        code="".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(10))
        o=(chat(f"Repeat this exact string and nothing else: {code}",max_tokens=20) or "").strip()
        ok=code in o; exact+=ok
        if not ok: errs.append((code,o))
        lr({"p":"P6","trial":tr,"code":code,"out":o,"exact":ok})
    print(f"  exact-match: {exact}/{n}")
    for c,o in errs[:6]: print(f"    {c} -> {o!r}")
    return {"exact":exact,"n":n,"errs":errs[:6]}

if __name__=="__main__":
    R={}
    R["P1"]=P1(); R["P2"]=P2(); R["P3"]=P3(); R["P4"]=P4(); R["P5"]=P5(); R["P6"]=P6()
    json.dump(R,open(os.path.join(OUTDIR,"results_quirks.json"),"w"),indent=2,default=str)
    print("\nsaved -> DIFFBENCH/results_quirks.json, raw_quirks.jsonl")
