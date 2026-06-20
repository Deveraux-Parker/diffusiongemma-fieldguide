# Serving Config Finding: Dead Zone Was Chunked-Prefill Artifact

Date: 2026-06-20

## Correction

The previously observed 1024-block dead zone is not an intrinsic
DiffusionGemma model behavior under a correctly sized single-request prefill.

It appears when vLLM serves the 8k model with the default/old chunked prefill
shape:

```text
max_num_batched_tokens: default / 2048 effective compile range
compile_ranges_endpoints: [2048]
```

It disappears when the same model is served with:

```text
--max-num-batched-tokens 8192
compile_ranges_endpoints: [8192]
```

Active verified container:

```text
name: dg-nochunk
args include: --max-num-batched-tokens 8192
log: Chunked prefill is enabled with max_num_batched_tokens=8192
log: compile_ranges_endpoints: [8192]
```

This makes the prompt prefill effectively single-chunk for the configured
`max_model_len=8000` / `max_num_seqs=1` setup.

## Decisive Reruns

### Shift Probe

Old 2048-compiled/chunked behavior:

```text
K=0    Bn=4 even  0/12
K=1024 Bn=5 odd   12/12
K=2048 Bn=6 even  0/12
```

New 8192-compiled behavior:

```text
K=0    Bn=4 even  12/12
K=256  Bn=4 even  12/12
K=512  Bn=4 even  12/12
K=768  Bn=5 odd   12/12
K=1024 Bn=5 odd   12/12
K=1280 Bn=5 odd   12/12
K=1536 Bn=5 odd   12/12
K=2048 Bn=6 even  12/12
```

Result directory:

```text
NEWBENCH/shift-probe-20260620-083422
```

### Dead-Zone Router Benchmark

Old 2048-compiled/chunked behavior:

```text
single raw_bad: 0/12 exact
single ballast_router: 12/12 exact
multi raw_mixed: 0.438 mean exact
multi all_registry: 1.000 mean exact
```

New 8192-compiled behavior:

```text
single raw_bad: 6/6 exact
multi raw_mixed: 4/4 exact
all strategies: 1.000 mean exact
```

Result directory:

```text
NEWBENCH/deadzone-router-20260620-083856
```

### Engineered Layout Benchmark

Old 2048-compiled/chunked behavior:

```text
dist 1000: 0/6
dist 1250: 0/6
IN_DEADBAND 4-record layout: 10/40 total, all-4 0/10
```

New 8192-compiled behavior:

```text
dist 1000: 6/6
dist 1250: 6/6
IN_DEADBAND 4-record layout: 40/40 total, all-4 10/10
```

Result directory:

```text
NEWBENCH/engineered-layout-20260620-083948
```

## Current Interpretation

The earlier block-parity law was a real, reproducible behavior of the old
serving setup, but it was not a property of DiffusionGemma itself. The most
likely cause is an interaction between vLLM chunked prefill, the 2048 compile
range, and Gemma's 1024-token sliding-window attention.

Corrected hierarchy:

1. First fix serving config: set `--max-num-batched-tokens` at least as large as
   the max prompt/request length for this single-user 8k setup.
2. Then benchmark again.
3. Keep compact tail registries for production validation and exact-value
   grounding, but do not use dead-zone ballast as a primary mitigation under the
   8192 prefill config.

## Operational Recommendation

For this local single-user setup:

```text
--max-model-len 8000
--max-num-seqs 1
--max-num-batched-tokens 8192
```

This trades away the old 2048-token prefill chunking behavior and removes the
dead-zone artifact in the tests above.

