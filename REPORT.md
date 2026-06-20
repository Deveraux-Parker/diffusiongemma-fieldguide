# DiffusionGemma — KV-utilization vs needle-index benchmark

**Model:** `dg-awq` (DiffusionGemma-26B-A4B-it, block-diffusion, AWQ/compressed-tensors INT4)
**Server:** vLLM, 4090, `max_model_len=8000`, `--max-num-seqs 1`
**Date:** 2026-06-20 · 5 trials/cell · filler = sensor telemetry log lines · needle = anomalous reading carrying a random incident code · retrieval = case-insensitive match

---

## Headline

**The weakness is driven by needle POSITION at high KV fill — not by KV fill itself, and not by generation length.**

- **Generation length: harmless.** Forced 128→1500-token continuous outputs stayed fully coherent (unique-8-gram = 1.000, no repetition collapse). Long outputs are *not* a coherence risk. *(Exp C)*
- **KV fill alone: harmless to front-loaded info.** A front needle (depth 0.05) was retrieved 5/5 at every fill from 1k up to **7.6k tokens = 95% of max context**. The one "miss" at the smallest fill was an empty reply, not a wrong answer. *(Exp A)*
- **Position at high fill: this is the real limit.** At 76% KV fill the model has a **dead zone at depth ≈0.6–0.95**, where it fails to locate the needle and instead emits a plausible number copied from a nearby nominal line. Beginning (primacy) and exact tail (recency) survive. *(Exp B)*

**For your front-loaded time-series use case: this is the favorable corner.** Important-info-up-front lands squarely in the strong primacy zone, and it holds at 95% KV utilization. Your ~1000-token NITH retrieval is comfortably safe — at that fill, position barely matters at all.

---

## Exp A — retrieval vs KV utilization (front needle, depth 0.05)

found-rate (edit-tolerant; ignores INT4 char-drops — see caveat)

```
kvUtil   promptTok   found      latency
0.07       577      ####.  4/5   168ms   (the .  = one empty reply)
0.14      1087      #####  5/5   231ms
0.26      2094      #####  5/5   339ms
0.51      4101      #####  5/5   510ms
0.76      6110      #####  5/5   706ms
0.95      7599      #####  5/5  1130ms
```
Front retrieval is flat-perfect across the whole KV range. Latency is pure prefill cost and scales with fill (~170ms→1.1s), as expected.

## Exp B — retrieval vs needle index (the position curve)

hit-rate out of 5, strict match:

```
depth:     0.00  0.10  0.25  0.50  0.75  0.90  1.00
fill 2000   5     5     5     5     4     5     5     (26% KV — essentially flat)
fill 6000   5     5     5     5     0     0     5     (76% KV — DEAD ZONE at 0.75–0.90)

fill=6000 profile:
 1.0 |#####           #####     #####     #####                    #####
     |  ^begin/primacy region............^           tail/recency^
 0.0 |                                    #####0   #####0
         0.0   0.1   0.25  0.5   0.75  0.90  1.0
```

At moderate fill (26% KV) position is nearly irrelevant. At high fill (76% KV) a clear **primacy + recency, lost-in-the-late-middle** pattern emerges: depths 0.0–0.5 perfect, 0.75–0.90 collapse to 0/5, and 1.0 recovers (needle sits inside the sliding window next to the question).

**Failure mode is substitution, not omission.** In the dead zone the model returned values like `00127`, `9667`, `82759` — timestamps/readings lifted from other nominal lines. It doesn't say "not found"; it confidently grabs a nearby number. Treat dead-zone retrieval as silently wrong, not blank.

## Exp C — generation-length control (low fill, forced long output)

```
forced_tok  compTok  uniq_8gram  latency
   128        128      1.000       704ms
   512        512      1.000      1127ms
  1024       1024      1.000      2098ms
  1500       1500      1.000      2264ms
```
No degradation with output length. Confirms the limit is input-side, not generation-side.

---

## Why (architecture)

DiffusionGemma is sliding-window-heavy (window 1024; full attention only every 6th layer). At high fill, mid/late-context tokens fall outside the per-layer sliding window and rely on the sparse full-attention layers, which appear to preferentially preserve the **start** (primacy anchor) and the **in-window tail**. The ~0.6–0.95 band gets neither. At low fill the whole prompt fits comfortably within reach, so position washes out.

## Practical guidance

- **Put what must be retrieved in the first ~50% of the prompt** (or the final ~1k tokens). Front-loaded is safest and survives to 95% KV util.
- **Avoid depth ~0.6–0.95 for must-retrieve facts at high fill.** If unavoidable, keep total fill ≤ ~2–3k tokens, where position is flat.
- **Dead-zone errors are silent substitutions** — if you need a "not found" signal, ask for a verbatim quote of the surrounding line and validate it, don't trust a bare value.
- **INT4 transcription caveat (separate axis):** even when the needle is *found*, INT4 occasionally drops a leading char/digit of an exact token (`JADE-3839`→`ADE-3839`). For exact-ID retrieval, prefer codes with redundancy/checksums or ask the model to repeat the value twice.

---

# Part 2 — Coherence vs KV utilization, and response shaping vs KV index

Part 1 measured binary *retrieval*. This part measures **generation quality**: the
model must emit a coherent single-canvas (≤256-tok) structured INCIDENT SUMMARY from
a front-loaded spec, graded objectively — completeness (5 required sections),
faithfulness (4 front-loaded facts used), non-repetition (unique-8-gram). Harness:
`bench_coherence.py` · raw: `raw_coherence.jsonl` · aggregates: `results_coherence.json`.

## Exp D — coherence vs KV utilization (spec front-loaded)

```
kvUtil   completeness  faithfulness  uniq8   out_len   latency
0.09        5.0/5         4.0/4       1.00     52w       374ms
0.15        5.0/5         4.0/4       1.00     53w       460ms
0.27        5.0/5         4.0/4       1.00     55w       550ms
0.52        5.0/5         4.0/4       1.00     45w       712ms
0.78        5.0/5         4.0/4       1.00     46w       955ms
0.92        5.0/5         4.0/4       1.00     46w      1173ms

coherence |#####################################  FLAT AT CEILING
   (5/5)  |  0.09   0.27   0.52   0.78   0.92   (KV util)
```

**Flat-perfect across the entire KV range.** Single-canvas generation quality has
**no measurable relationship with KV utilization** when the source info is
front-loaded — exactly your hypothesis, confirmed to 92% fill. Output length and
structure stay stable; only latency rises (pure prefill cost).

## Exp E — response shaping vs KV index (spec block moved, fill ~6k = 76% KV)

per-trial breakdown (5 trials), failure mode characterized:

```
spec_depth   clean(5/5,4/4)   failure mode at this index
  0.00          5/5           none — flawless
  0.25          5/5           none — flawless
  0.50          3/5           2 trials drop 1 section + 1 fact (still coherent, non-empty)
  0.75          3/5           2 trials drop 1 section (all facts kept, coherent)
  0.95          4/5           1 trial TOTAL DROPOUT (empty output); other 4 flawless
```

**Shape of the degradation:** front-loaded → perfect. As the controlling spec moves
into the mid-context (0.5–0.75) the model occasionally **omits one of the five
sections** (a completeness nick, not a collapse — prose stays clean, facts mostly
intact). At the tail (0.95) the dominant risk flips to **occasional total dropout**
(empty reply), while successful generations remain flawless. No repetition collapse
at any index (every non-empty output had uniq-8gram = 1.0).

Note the contrast with retrieval (Part 1): retrieval *recovered* at the exact tail
(needle adjacent to the question, inside the sliding window); generation does **not**
get that recency rescue here because filler still follows the spec at depth 0.95, and
it adds a dropout failure mode instead.

## Combined verdict for the front-loaded / time-series use case

Your instinct is correct and now quantified: **a coherent semantic unit generated
from front-loaded context is robust to KV fill** — perfect single-canvas output to
92% utilization (Exp D), and front-loaded retrieval is perfect to 95% (Exp A). The
only fragile regimes are *mid-to-late source position at high fill*: silent number
substitution for retrieval (Part 1) and section-omission / occasional dropout for
generation (Exp E). For ~1k-token front-loaded time-series episodes you sit far inside
the safe corner on both axes.

---

# Part 3 — Quirks & weaknesses (Wave 1)

Harness: `bench_quirks.py` · raw: `raw_quirks.jsonl`. Every headline below was
confirmed by reading raw outputs, not just aggregate scores — several aggregates
were misleading until inspected.

## ★ QUIRK 1 — terse "reply with ONLY X" prompts cause empty-output dropouts

The biggest practical finding. Same easy counting task (3 CRITICAL lines in 24), three answer styles, 8 trials each:

```
answer style                         empties   correct
"Reply with ONLY the number"          7/8        1/8
"Answer in one short sentence"        0/8        8/8
"List each CRITICAL line, then count" 1/8        7/8
```

The model counts **perfectly** when allowed a sentence. Demanding a single bare
token makes the diffusion denoiser early-stop to **nothing** ~7/8 of the time. This
one quirk masqueraded as "bad counting," caused most of the P2 computed-retrieval
"failures," and explains the occasional empty in Exp E (Part 2).

**Workaround:** never request a bare terse answer. Ask for a short sentence (or
"list, then state"), and parse the value out yourself. Costs a few tokens, removes
the dropout entirely.

## Computed / multi-fact retrieval — works (once terseness is controlled)

Front-loaded time-series with known ground truth, asked for max reading, first
threshold-crossing timestamp, and CRITICAL count. When non-empty, answers are
correct: max (`94.5`, `94.2`, `94.6` ✓), first-crossing (`t=00015`, `t=00020` ✓),
count (8/8 in sentence form). The model genuinely *reasons over* the series — the
only failures were terse-dropouts and one char-drop (`93.7`→`3.7`). **It is useful
for real time-series queries, not just string lookup.**

## Dead zone is a sharp sliding-window boundary effect (not a smooth fill curve)

Retrieval at depth 0.8, sweeping fill, came out **non-monotonic**: hit-rate
4/5, 5/5, **2/5**, 5/5, **0/5**, 5/5 across fills 2k→7k. Inspecting misses: the 0/5
and 2/5 cells fail by **wrong-value substitution** (returning a timestamp from a
neighboring line: `90000`, `00001`…), i.e. genuine dead-zone — while 5k and 7k are
perfect. The failure tracks the needle's **absolute token-distance from the question**
crossing the ~1024-token sliding window, *not* depth-percent. Parametrizing by depth%
makes it look erratic; it's actually a sharp boundary. → remap by token-distance (Wave 2).

## Top-of-context pointer does NOT rescue a dead-zone needle

Buried needle at 0.8/6k: no-pointer 0/5, with a top "NOTE: one CRITICAL line below,
you'll be asked for its code" → 1/5. Priming attention from the top does not overcome
the sliding-window dead zone. (Other mitigations — duplicating the needle, a real TOC
with the value — remain to test.)

## Near-deterministic at temperature 0 (not byte-exact)

Same prompt × 5 at temp 0 → 5 *semantically identical* summaries with minor lexical
jitter ("maintains a nominal status" vs "…throughout, with…"). The diffusion sampler
is not bit-reproducible even at temp 0. Meaning is stable; exact wording is not — don't
build exact-string caching/asserts on output.

## CORRECTION — INT4 char-drop is retrieval-under-load, not pure quantization

Earlier (Part 1) I attributed `JADE-3839`→`ADE-3839` to an INT4 transcription artifact.
**Clean echo of 20 random 10-char codes: 20/20 exact.** So quantization transcribes
perfectly in isolation; the dropped leading char/digit only appears when *reading a
value out of a filled/working context* (`93.7`→`3.7` in P2, `MAGENTA-73…` in P1). It's
a retrieval-under-load artifact, not a quant echo defect. Mitigation unchanged (ask for
the value twice / use checksummed IDs), but the cause is reattributed.

## Wave-1 scorecard

| probe | result |
|---|---|
| Terse-answer dropout | **Major quirk** — 7/8 empty; sentence form fixes it |
| Computed retrieval | Works (max/count/crossing correct in sentence form) |
| Dead-zone shape | Real, sharp ~1024-tok boundary effect, not a fill curve |
| Pointer rescue | Doesn't work (0→1 / 5) |
| Temp-0 determinism | Semantically stable, lexically jittery |
| INT4 clean echo | 20/20 — char-drop is retrieval-load, not quant |

---

# Part 4 — Mitigating the dead zone (Wave 3)

Harness `bench_wave3.py`. A needle is buried ~1000 tokens before the question (dead
zone, baseline ~0). Six placement strategies, 8 trials each, all copies carry the
same code, sentence-form question.

```
strategy                       found    what it tells us
baseline (1 copy, dead zone)    0/8     reproduces the dead zone
dup adjacent ×2 (back-to-back)  0/8     local repetition does NOTHING
dup adjacent ×3 (back-to-back)  0/8     even 3 stacked copies fail completely
dup → recency (copy pre-question) 8/8   ✓ rescued
dup → primacy (copy at top)     8/8     ✓ rescued
primacy + recency (×3 spread)   8/8     ✓ rescued
```

**The finding:** duplicating content *within* the dead zone is useless — three
back-to-back copies still scored 0/8. The dead zone is an **attention-coverage gap**
(sliding-window), not a signal-strength problem, so repetition there can't help. The
fix is to relocate **one** copy into a zone attention actually reaches — the recency
window (last ~500 tokens) or the primacy slot (very top). Either gives 8/8.

Verified: the 3 copies were genuinely placed back-to-back; misses are real
substitutions (the model returns `87.3` — the needle line's *temperature* — or a
garbled `AMBER-5817`→`BER-57`), confirming it partially attends to the region but
cannot reliably read it.

**Actionable rule:** if a critical fact must live in the middle of a long prompt,
echo a copy of it into a header at the top **or** into the lines just before your
question. Don't bother repeating it in place.

---

# Part 5 — CORRECTION: the dead zone is position-anchored and oscillates with length (NEWBENCH)

A multi-fill distance map (`NEWBENCH/multifill_distance_map_benchmark.py`) **disconfirmed**
the earlier "fixed dead zone 600–1500 tokens before the question." Holding distance
constant and sweeping fill:

```
needle at a FIXED ~1000 tok from the question, exact-code retrieval:
  fill   4500  5500  6500  7300       (10 reps, isolate_checksum_fill.py)
  hits   10/10 0/10  10/10 0/10        <- identical for plain AND checksum needles
```

The result **oscillates with fill** at the same distance — so the dead zone is *not* a
fixed distance from the question, and *not* a function of needle content (the trailing
checksum landmark changed nothing).

A fine fill-sweep at two distances (`fine_fill_sweep.py`, dist 1000 & 1500) shows the
dead bands align by **needle ABSOLUTE POSITION**, not by fill or distance:

```
dist~1000  dead at needle-pos ~ 2959, 4200–5058, 6279–6556
dist~1500  dead at needle-pos ~ 2448, 4265            (overlaps the ~4200–4800 band)
```

So there are **quasi-periodic dead bands at certain absolute positions** (~2500, ~4400,
~6400; spacing ~1900 tok), with pass bands between and at the extremes. The original
"distance valley" (Parts 1/Wave-2) was the ~4200–5100 position band viewed at a single
~5.8k-token fill, misread as a distance effect.

**Corrected model.** Two reliable zones — the **very start** (primacy) and the **last
few hundred tokens** before the question (recency / sliding window). Everything in
between is a position-dependent minefield: the same fact at the same distance can be
read or confabulated depending only on total prompt length. You cannot pick a safe
*distance* in the middle.

**Practical guidance is unchanged and now better-justified:** put must-retrieve values
at the very top or right before the question; never rely on a middle position; echo a
compact index (key+value+checksum) into a safe zone; validate exact IDs. Likely mechanism:
block/chunk-aligned sparse attention whose tiling shifts against absolute position as the
prompt grows. Open: pin the exact period and tie it to the attention block size.

Harnesses: `NEWBENCH/multifill_distance_map_benchmark.py`, `isolate_checksum_fill.py`,
`fine_fill_sweep.py` (timestamped run dirs with raw.jsonl + summary.md).

## Files
- `bench_kv.py` / `results.json` / `raw.jsonl` — Part 1 (retrieval)
- `bench_coherence.py` / `results_coherence.json` / `raw_coherence.jsonl` — Part 2 (coherence + shaping)
- `bench_quirks.py` / `results_quirks.json` / `raw_quirks.jsonl` — Part 3 (quirks, Wave 1)
- `bench_wave2.py` / `results_wave2.json` / `raw_wave2.jsonl` — Wave 2 (cliff, canvas, temp, JSON, honesty)
- `bench_wave3.py` / `results_wave3.json` / `raw_wave3.jsonl` — Part 4 (dead-zone mitigation)
