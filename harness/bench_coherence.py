#!/usr/bin/env python3
"""
DiffusionGemma — COHERENCE vs KV utilization, and RESPONSE SHAPING vs KV index.

Distinct from bench_kv.py (which measured binary needle retrieval). Here the
model must GENERATE a coherent single-canvas (<=256 tok) structured unit from a
front-loaded spec, while the context is filled to varying KV utilization.

  Exp D: coherence vs KV fill   (spec pinned at FRONT, fill grows 0.5k->7.5k)
  Exp E: response shaping vs KV index (spec block MOVED to depth 0..0.95 at fixed fill)

Coherence is scored objectively per generated unit:
  completeness  = # of the 5 required sections present      (0-5)
  faithfulness  = # of the 4 front-loaded facts used        (0-4)
  uniq8gram     = non-repetition proxy                       (0-1)
  len_tok       = output length (shape)
Outputs are saved verbatim for qualitative coherence reading.
"""
import json, time, random, os, urllib.request

BASE="http://localhost:8001"; MODEL="dg-awq"; MAXLEN=8000
OUTDIR=os.path.dirname(os.path.abspath(__file__)); N_TRIALS=5

def _post(p,pl):
    req=urllib.request.Request(BASE+p,data=json.dumps(pl).encode(),
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=300) as r: return json.loads(r.read())
def tok_count(t): return _post("/tokenize",{"model":MODEL,"prompt":t})["count"]

# ---- the front-loaded spec + the required single-canvas output -------------
SPEC = (
 "EPISODE SPEC (use these exact values):\n"
 "  device_id: DV-7741\n"
 "  baseline_temp: 22.4C\n"
 "  alert_threshold: 80.0C\n"
 "  peak_temp: 91.3C\n"
 "  peak_time: t=00412\n"
 "  root_cause: coolant_pump_failure\n"
 "TASK: Produce an INCIDENT SUMMARY of at most ~200 words with EXACTLY these five "
 "sections, each on its own line prefixed by the header in caps: "
 "HEADLINE:, TIMELINE:, ROOT CAUSE:, SEVERITY:, RECOMMENDATION:. "
 "You MUST explicitly state the device_id (DV-7741), the peak_temp (91.3C), and "
 "the peak_time (t=00412) somewhere in the summary.\n")

REQ_SECTIONS=["HEADLINE","TIMELINE","ROOT CAUSE","SEVERITY","RECOMMENDATION"]
REQ_FACTS=["DV-7741","91.3","00412","coolant"]

def log_line(i,rng):
    return (f"[t={i:05d}] sensor_temp={rng.uniform(18,26):.2f}C "
            f"pressure={rng.uniform(99,103):.1f}kPa flow={rng.uniform(8,16):.1f}L/min "
            f"vibration={rng.uniform(0.1,0.9):.2f}mm status=nominal")

def make_filler(target_tok,rng):
    lines=[log_line(i,rng) for i in range(1,700)]
    tpl=tok_count("\n".join(lines[:20]))/20.0
    n=min(len(lines),max(1,int(target_tok/tpl)))
    return lines[:n]

def build(target_tok,spec_depth,rng):
    """spec_depth=0.0 -> spec at very front; 0.95 -> near tail. Question always last."""
    filler=make_filler(target_tok,rng)
    pos=min(len(filler),max(0,int(spec_depth*len(filler))))
    block="\n".join(filler[:pos]) + ("\n\n" if pos else "") + SPEC + "\n" + "\n".join(filler[pos:])
    user=("You monitor a telemetry stream. Context (logs + one EPISODE SPEC) follows.\n\n"
          + block +
          "\n\nNow produce the INCIDENT SUMMARY exactly as the EPISODE SPEC instructs.")
    return user, tok_count(user)

def gen(user):
    t0=time.time()
    r=_post("/v1/chat/completions",{"model":MODEL,
        "messages":[{"role":"user","content":user}],
        "max_tokens":256,"temperature":0.0})
    return r["choices"][0]["message"]["content"], r["usage"], time.time()-t0

def score(out):
    up=(out or "").upper()
    comp=sum(1 for s in REQ_SECTIONS if s+":" in up or s in up)
    faith=sum(1 for f in REQ_FACTS if f.upper() in up)
    w=(out or "").split()
    g=[" ".join(w[i:i+8]) for i in range(max(0,len(w)-7))]
    uniq=len(set(g))/len(g) if g else 0.0
    return comp,faith,round(uniq,3),len(w)

raw=open(os.path.join(OUTDIR,"raw_coherence.jsonl"),"w")
def lr(rec): raw.write(json.dumps(rec)+"\n"); raw.flush()

def expD():
    rows=[]
    for f in [500,1000,2000,4000,6000,7200]:  # cap so prompt + 256-tok canvas + template <= 8000
        agg=[0.0,0.0,0.0,0.0]; toks=[]; lats=[]
        for tr in range(N_TRIALS):
            rng=random.Random(2000+f*7+tr)
            user,at=build(f,0.0,rng)
            out,usage,dt=gen(user)
            c,fa,u,ln=score(out);
            agg[0]+=c; agg[1]+=fa; agg[2]+=u; agg[3]+=ln; toks.append(at); lats.append(dt)
            lr({"exp":"D","fill":f,"prompt_tok":at,"trial":tr,"comp":c,"faith":fa,
                "uniq":u,"len":ln,"out":out})
        at_avg=sum(toks)/len(toks)
        rows.append({"fill":f,"kv_util":round(at_avg/MAXLEN,3),
            "completeness":round(agg[0]/N_TRIALS,2),"faithfulness":round(agg[1]/N_TRIALS,2),
            "uniq8":round(agg[2]/N_TRIALS,3),"len_w":round(agg[3]/N_TRIALS),
            "lat_ms":round(1000*sum(lats)/len(lats))})
        print(f"  D kv={at_avg/MAXLEN:.2f} comp={agg[0]/N_TRIALS:.1f}/5 "
              f"faith={agg[1]/N_TRIALS:.1f}/4 uniq8={agg[2]/N_TRIALS:.2f} "
              f"len~{round(agg[3]/N_TRIALS)}w lat~{round(1000*sum(lats)/len(lats))}ms")
    return rows

def expE():
    rows=[]
    for d in [0.0,0.25,0.5,0.75,0.95]:
        agg=[0.0,0.0,0.0,0.0]
        for tr in range(N_TRIALS):
            rng=random.Random(7000+int(d*100)+tr)
            user,at=build(6000,d,rng)
            out,usage,dt=gen(user)
            c,fa,u,ln=score(out)
            agg[0]+=c; agg[1]+=fa; agg[2]+=u; agg[3]+=ln
            lr({"exp":"E","spec_depth":d,"fill":6000,"trial":tr,"comp":c,"faith":fa,
                "uniq":u,"len":ln,"out":out})
        rows.append({"spec_depth":d,"completeness":round(agg[0]/N_TRIALS,2),
            "faithfulness":round(agg[1]/N_TRIALS,2),"uniq8":round(agg[2]/N_TRIALS,3),
            "len_w":round(agg[3]/N_TRIALS)})
        print(f"  E specDepth={d:.2f} comp={agg[0]/N_TRIALS:.1f}/5 "
              f"faith={agg[1]/N_TRIALS:.1f}/4 uniq8={agg[2]/N_TRIALS:.2f} len~{round(agg[3]/N_TRIALS)}w")
    return rows

if __name__=="__main__":
    print("== Exp D: coherence vs KV utilization (spec front-loaded) ==")
    D=expD()
    print("== Exp E: response shaping vs KV index (spec block moved) ==")
    E=expE()
    json.dump({"D":D,"E":E},open(os.path.join(OUTDIR,"results_coherence.json"),"w"),indent=2)
    print("\nsaved -> DIFFBENCH/results_coherence.json, raw_coherence.jsonl")
