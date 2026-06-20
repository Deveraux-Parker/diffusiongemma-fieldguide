# DiffusionGemma on a 4090 — tuning, context scaling & benchmarks

A working record of how to run **DiffusionGemma-26B-A4B-it** (AWQ/compressed-tensors INT4)
fast and correctly on a single RTX 4090 via vLLM, and the investigation behind it.

Model: `dg-awq` · image `vllm/vllm-openai:gemma` (v0.22.1rc1) · 24 GB RTX 4090 (Ada) ·
weights ~16.4 GB · single-user (`--max-num-seqs 1`).

---

## TL;DR

1. **Set `--max-num-batched-tokens ≥ --max-model-len`.** This is the single most important
   flag for long-context use. It forces single-chunk prefill and makes retrieval clean at
   every position. (Default chunked prefill silently breaks long-range retrieval — see below.)
2. **Use `--kv-cache-dtype fp8` for more context *and* more speed.** It halves KV memory and
   decode is **~8–27 % faster** (decode is bandwidth-bound; smaller KV = less to read).
   No measurable retrieval-quality loss.
3. **Context ceilings on the 4090** (with clean single-chunk prefill):
   - fp16 KV → **~12k** (tight) / 10k comfortable
   - fp8 KV → **~18k** (20k just OOMs on a fixed 256 MB buffer)
4. **Prompt in sentences**, not bare tokens ("state X in one short sentence", then parse).

Ready-to-run scripts: `launch_dg_10k_fp16.sh` (comfortable) and `launch_dg_18k_fp8.sh` (max ctx).

---

## The finding: long-context retrieval & the chunked-prefill dead zone

With the default vLLM config, the model appeared to have a "dead zone": a fact placed
~1,000 tokens before the question was confabulated instead of read, while facts near the
start or right before the question were fine. Extensive testing (600+ trials, a 1024-shift
probe, an adversarial multi-agent verification) showed this was **deterministic** —
governed by 1024-token block alignment — and we eventually traced the root cause:

> **It is a vLLM *chunked-prefill* × sliding-window-attention artifact, not a model
> limitation.** vLLM prefills long prompts in `max_num_batched_tokens`-sized chunks
> (default ~2048); the model's 1024 sliding window drops some long-range attention *inside*
> a chunk. The dead zone tracks the chunk boundary exactly:
>
> | chunk size | dead zone |
> | --- | --- |
> | 2048 (default) | present, repeats every ~2048 tokens |
> | 4096 | moves to the 4096 boundary |
> | ≥ prompt (single chunk) | **gone — 100 % everywhere** |

**Fix:** `--max-num-batched-tokens ≥ max-model-len` → the whole prompt prefills in one chunk
→ retrieval is **100 % at every depth** (verified 250–16,000 tokens deep). No prompt
engineering or routing required.

---

## Context scaling — how far the 4090 goes

The KV cache is *not* the bottleneck (sliding-window attention keeps long-context KV cheap;
fp8 halves it again). The real wall is the **single-chunk prefill batch** (needed to keep
retrieval clean) plus a **fixed ~256 MB fp32 logits buffer** the diffusion sampler allocates
per request. Both compete with KV for the ~8 GB left after weights.

| KV dtype | max clean ctx | util | free VRAM | notes |
| --- | --- | --- | --- | --- |
| fp16 | 8,000 | 0.85 | ~1 GB | original comfortable config |
| fp16 | 12,000 | 0.90 | 78 MB | works, zero margin |
| **fp8** | **18,000** | **0.90** | 239 MB | **max; benchmarked stable** |
| fp8 (attempt) | 20,000 | 0.92 | — | KV fits (22k pool) but the 256 MB buffer OOMs |

`ValueError` at 12k/fp16 estimated max model length **10,976**; at 20k/fp8 estimated **17,696**
(batch 20480) — i.e. the big prefill batch, not the KV, sets the ceiling. Pushing util to 0.92
to fit 20k crashes on the diffusion buffer (`CUDA out of memory. Tried to allocate 256.00 MiB`).

---

## Benchmarks (measured)

**Retrieval at depth — fixed (single-chunk) server.** Needle ("secret token") placed at
varying distance from the question; 100 % is perfect.

```
@ 7k context  (fixed 8k server):   dist 250→6500 tok  =  110/110  (100% everywhere)
@ 11k context (12k/fp16 server):   dist 300/1000/3000/6000/9000   =  5/5 each
@ 17k context (18k/fp8  server):   dist 300/1000/5000/10000/16000 =  5/5 each
```

**Decode throughput — fp8 vs fp16 KV.** *Pure* decode, measured from the first generated
token (prefill excluded via the marginal method: `(t1024 − t256) / 768`, same prompt so
prefill cancels exactly). Low-entropy continuation (best case):

```
context   fp16 KV    fp8 KV     
  2k      595 t/s    818 t/s    fp8 faster
  8k      493 t/s    514 t/s    fp8 faster
```

→ **fp8 KV is faster or equal, never a tax** (decode is bandwidth-bound; half-size KV = less
to read). The exact margin is noisy at small per-canvas times (~+4 % to +38 %).

**Decode is content-dependent.** The diffusion sampler runs more denoising steps for complex/
less-predictable text. So decode is a *range*: a predictable continuation hits **500–800 tok/s**,
while a dense 800-token technical explainer ran **~305 tok/s** (high entropy → more steps). Both
measured post-prefill. (Earlier "219/285 tok/s" figures were total-time-including-prefill or
prefix-warmed — they undercount; use the pure-decode numbers above.)

**Prefill / latency.** Prefilling a 17k prompt (single chunk) ≈ **2.0 s** (this is the TTFT cost,
separate from decode). Short answers on small prompts return in ~106 ms (the denoiser
early-stops). Latency rises one 256-token canvas at a time; throughput *rises* with output
length (opposite of autoregressive). An 800-token explainer: coherent, uniq-8gram 1.00.

Other (from the broader benchmark): generation coherence flat to 92 % context fill, strict
JSON 8/8, computed retrieval (max/threshold/count) correct, tool use clean, multilingual fine.

---

## Run configs

**Comfortable — 10k / fp16** (`launch_dg_10k_fp16.sh`):
```
--max-model-len 10000 --max-num-batched-tokens 10240 \
--gpu-memory-utilization 0.88 --kv-cache-dtype float16 --max-num-seqs 1
```

**Max context — 18k / fp8** (`launch_dg_18k_fp8.sh`):
```
--max-model-len 18000 --max-num-batched-tokens 18432 \
--gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --max-num-seqs 1
```

Both keep `--max-num-batched-tokens ≥ --max-model-len` (clean retrieval), `--enable-auto-tool-choice
--tool-call-parser gemma4 --reasoning-parser gemma4`, and `-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

To rescale: pick `max-model-len`, set `max-num-batched-tokens` to it (rounded up), choose KV
dtype (fp8 for >12k or for speed), and nudge `gpu-memory-utilization` until it fits — keep
≥ ~250 MB free for the diffusion buffer (don't hit 0.92).

---

## Prompting notes

- **Ask for a sentence / JSON, not a bare token.** "Reply with only the number" makes the
  diffusion sampler return an empty string (~7/8); "state it in one short sentence" answers
  correctly — parse the value out.
- **Temp 0 is not byte-deterministic** (meaning stable, wording varies); don't build exact-string caches.
- For exact IDs deep in context, asking the model to repeat the value once is a cheap self-check.

---

## Reproducibility

Harnesses and raw run records under `DIFFBENCH/` and `NEWBENCH/` (timestamped run dirs with
`raw.jsonl` + `summary.md`): mechanism decomposition, the 1024-shift probe, chunk-size sweeps,
retrieval-at-depth, and the fp8-vs-fp16 decode benchmark. Public showcase:
https://deveraux-parker.github.io/diffusiongemma-fieldguide/
