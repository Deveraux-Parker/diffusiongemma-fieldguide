# Dead-Zone Router Benchmark

- Started: 2026-06-20T08:38:56-0600
- Model: `dg-awq`
- Block size: `1024`
- Calibrated filler tokens/line: `15.45`
- Single reps: `6`
- Multi reps: `4`

## Results

| suite | strategy | n | mean_exact | min_exact | mean_lenient | pass_rate_exact_1 | mean_dead_count | mean_pad_tokens | mean_prompt_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi | all_registry | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 | 5734.500 |
| multi | ballast_router | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 1216 | 6883.250 |
| multi | combo_original_dead_registry | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 1216 | 6923.250 |
| multi | dead_only_registry | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 | 5689.250 |
| multi | raw_mixed | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 | 5649.250 |
| single | ballast_router | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 1301.333 | 6945.500 |
| single | combo_router | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 1301.333 | 6984.500 |
| single | raw_bad | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 | 5629.500 |
| single | tail_registry | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 | 5668.500 |

## Interpretation

- `raw_bad` deliberately places the target in `diff==1` with an even needle block.
- `ballast_router` adds computed start padding until the classifier predicts no dead-zone source record.
- `tail_registry` duplicates exact answer-bearing fields immediately before the question.
- `dead_only_registry` copies only records classified as dead in the base prompt.
- `combo_original_dead_registry` adds ballast and still copies records that were dead before padding.

Use `raw.jsonl` for full prompts' position reports and model outputs.
