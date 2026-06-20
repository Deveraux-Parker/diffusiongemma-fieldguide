# DiffusionGemma — a fast, faithful long-context model

A short showcase of **DiffusionGemma-26B-A4B** (a block-diffusion LM) running on a single
RTX 4090 via vLLM.

**→ Live page: https://deveraux-parker.github.io/diffusiongemma-fieldguide/**

## What it does

- **Long-context retrieval** — pulls a fact from anywhere in the prompt; 110/110 across the
  full depth of a 7k-token context.
- **Coherent under load** — structured generation stays complete and non-repetitive to 92%
  context fill.
- **Computed retrieval** — max value, first threshold-crossing, anomaly count, all correct
  over a long series.
- **Reliable strict JSON** — 8/8 valid against a fixed schema.
- **Fast** — denoises a 256-token canvas at once: ~106 ms short answers, ~806 tok/s peak,
  throughput rises with output length.
- **Tool use & multilingual** — valid single/parallel tool calls; fluent JA / ZH / FR.

## Getting the best from it

Ask for answers as a short sentence or JSON (not a bare token) and parse the value out — it
keeps the diffusion sampler's responses crisp.

## Serve it

```
docker run -d --gpus all --ipc=host \
  -v $MODEL_DIR:/model:ro -p 8001:8000 vllm/vllm-openai:gemma \
  --model /model --served-model-name dg-awq \
  --max-model-len 8000 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 --max-num-batched-tokens 8192 \
  --kv-cache-dtype float16 --host 0.0.0.0 --port 8000
```

Use `/v1/chat/completions` with the chat template; the served model id is `dg-awq`.

## In this repo

- `index.html` — the live showcase (self-contained, no build step).
- `newbench/`, `REPORT.md` — measurement harnesses and data.
