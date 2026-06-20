# Routing Index Ablation

- Started: 2026-06-20T05:45:07-0600
- Fill target: `5500` prompt tokens
- Target distance requested: `1000` tokens
- Reps: `8`

## Results

| strategy | top | tail | values in safe zone | pointer only | hits | exact rate | field score | miss types |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_deadzone | None | None | False | False | 0/8 | 0.000 | 0.292 | `{"empty": 1, "window_only": 7}` |
| tail_pointer | None | pointer | False | True | 0/8 | 0.000 | 0.250 | `{"empty": 2, "window_only": 6}` |
| tail_exact_index | None | exact_index | True | False | 8/8 | 1.000 | 1.000 | `{"hit": 0}` |
| tail_summary_duplicate | None | summary | True | False | 8/8 | 1.000 | 1.000 | `{"hit": 0}` |
| tail_full_duplicate | None | full | True | False | 8/8 | 1.000 | 1.000 | `{"hit": 0}` |
| top_pointer | pointer | None | False | True | 0/8 | 0.000 | 0.333 | `{"window_only": 8}` |
| top_exact_index | exact_index | None | True | False | 8/8 | 1.000 | 1.000 | `{"hit": 0}` |
| top_summary_duplicate | summary | None | True | False | 8/8 | 1.000 | 1.000 | `{"hit": 0}` |
| top_full_duplicate | full | None | True | False | 8/8 | 1.000 | 1.000 | `{"hit": 0}` |

## Interpretation

Pointer-only rows test whether a safe-zone pointer can route the model back into the dead-zone source. Exact indexes and duplicated summaries test whether putting the actual answer values in a safe zone is enough.
