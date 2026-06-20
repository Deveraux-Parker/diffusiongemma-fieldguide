# DiffusionGemma — a field guide to its behavior

A visual showcase of what we measured about **DiffusionGemma-26B-A4B** (a block-diffusion
language model) running on a single RTX 4090 via vLLM.

**→ Live page: https://deveraux-parker.github.io/diffusiongemma-fieldguide/**

It's a single self-contained `index.html` (no build step, no external JS libraries —
charts are hand-built inline SVG/CSS).

## What's inside

- **`index.html`** — the showcase page.
- **`REPORT.md`** — the full written findings (Parts 1–3 + Wave 2).
- **`harness/`** — the reproducible benchmark scripts and their raw aggregate results:
  - `bench_kv.py` — retrieval vs context fill & needle position
  - `bench_coherence.py` — generation coherence vs fill, response shaping vs position
  - `bench_quirks.py` — terse-prompt dropout, computed retrieval, determinism, INT4 fidelity
  - `bench_wave2.py` — the dead-zone cliff, canvas-boundary latency, temperature, JSON, honesty

## Headline findings

- **Front-load it.** Retrieval stays ~100% to 95% context fill and generation coherence
  holds to 92% — *when the important information is near the top*.
- **The valley.** A fact stranded ~600–1500 tokens before the question gets lost (sliding-window
  attention), while facts near the start (primacy) or right before the question (recency) are safe.
- **Never ask for a one-word answer.** "Reply with only X" makes the denoiser return empty ~7/8 of
  the time; ask for a short sentence instead and it answers correctly.

All numbers were measured locally; every headline was confirmed by reading raw model outputs.
