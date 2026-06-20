# Binding-failure vs coverage-gap

- fill `5500` tok · dead-zone d~1000 · 10 reps · model `dg-awq`

| probe | info content | dist | hit rate | 95% CI | miss types |
| --- | --- | --- | --- | --- | --- |
| presence | coarse (1 bit) | 1000 | 7/10 (0.70) | [0.40, 0.89] | — |
| neighbor_temp | co-located value | 1000 | 0/10 (0.00) | [0.00, 0.28] | — |
| color_prefix | low (1 of 6) | 1000 | 0/10 (0.00) | [0.00, 0.28] | — |
| exact_code | high (exact) | 1000 | 0/10 (0.00) | [0.00, 0.28] | wrong_number:2, needle_neighbor_value:4, code_fragment:4 |
| exact_twice | high (x2) | 1000 | 0/10 (0.00) | [0.00, 0.28] | code_fragment:5, needle_neighbor_value:4, wrong_number:1 |
| exact_recency | high (SAFE control) | 220 | 10/10 (1.00) | [0.72, 1.00] | — |

## Read

If coarse/low-entropy probes pass while exact_code fails (and the safe-zone control passes),
the dead zone is a BINDING failure under attention geometry, not blindness to the region.
## Observed failure mode (from raw outputs)

It is not a clean entropy gradient — it is **confabulation from priors**, total for
specific fields:

- `presence` 7/10: weak gist that *a* CRITICAL line exists (3/10 it denied one existed).
- `neighbor_temp` 0/10: answered with **nominal-range** temps (21.90C, 24.73C…); real value 87.3C.
- `color_prefix` 0/10: answered **"green"** every time — not even in the code's color set.
- `exact_recency` 10/10: same questions are answerable when the record sits in a safe zone.

So the dead zone yields no reliable read of any field; the model fills the requested slot
with a plausible prior value. Low entropy does NOT make it bindable. Implication: you cannot
rescue a dead-zone value by asking a "smaller" question — relocate a copy (or a compact index)
into a safe zone, and validate exact IDs with checksum / source-quote.
