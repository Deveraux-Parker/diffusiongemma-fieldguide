# Mitigation Strategy Benchmark

- Started: 2026-06-20T05:13:12-0600
- Model: `dg-awq`
- Fill target: `5500` prompt tokens
- Reps: `8`

## Results

| strategy | copies | hits | hit rate | actual copy distances | miss types |
| --- | --- | --- | --- | --- | --- |
| baseline_deadzone | 1.0 | 0/8 | 0.000 | 1045 | `{"code_fragment": 1, "other": 2, "wrong_number": 5}` |
| dup_adjacent_x2 | 2.0 | 0/8 | 0.000 | 1109 / 1062 | `{"code_fragment": 4, "needle_neighbor_value": 2, "wrong_number": 2}` |
| dup_adjacent_x3 | 3.0 | 0/8 | 0.000 | 1169 / 1122 / 1061 | `{"code_fragment": 3, "needle_neighbor_value": 5}` |
| dup_spread_deadzone | 3.0 | 0/8 | 0.000 | 1402 / 1218 / 972 | `{"needle_neighbor_value": 7, "wrong_number": 1}` |
| dup_recency | 2.0 | 8/8 | 1.000 | 1114 / 227 | `{"hit": 0}` |
| dup_primacy | 2.0 | 8/8 | 1.000 | 5396 / 1116 | `{"hit": 0}` |
| dup_primacy_recency | 3.0 | 8/8 | 1.000 | 5440 / 1167 / 273 | `{"hit": 0}` |

## Interpretation

Back-to-back duplication inside the dead zone tests salience. A recency or primacy copy tests relocation into a safer attention region. If adjacent copies fail while relocated copies pass, duplication works only when at least one copy leaves the valley.
