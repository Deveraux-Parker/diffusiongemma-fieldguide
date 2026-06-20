# Dead-Zone Router Benchmark

- Started: 2026-06-20T08:07:49-0600
- Model: `dg-awq`
- Block size: `1024`
- Calibrated filler tokens/line: `15.45`
- Single reps: `12`
- Multi reps: `8`

## Results

| suite | strategy | n | mean_exact | min_exact | mean_lenient | pass_rate_exact_1 | mean_dead_count | mean_pad_tokens | mean_prompt_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multi | all_registry | 8 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 | 5741 |
| multi | ballast_router | 8 | 0.906 | 0.750 | 0.938 | 0.625 | 0 | 1216 | 6887.750 |
| multi | combo_original_dead_registry | 8 | 0.906 | 0.750 | 0.969 | 0.625 | 0 | 1216 | 6927.875 |
| multi | dead_only_registry | 8 | 0.938 | 0.750 | 0.938 | 0.750 | 1 | 0 | 5695.875 |
| multi | raw_mixed | 8 | 0.438 | 0.250 | 0.500 | 0.000 | 1 | 0 | 5655.875 |
| single | ballast_router | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 1162.667 | 6804.833 |
| single | combo_router | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 1162.667 | 6843.833 |
| single | raw_bad | 12 | 0.000 | 0.000 | 0.083 | 0.000 | 1 | 0 | 5628 |
| single | tail_registry | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 0 | 5667 |

## Interpretation

- `raw_bad` deliberately places the target in `diff==1` with an even needle block.
- `ballast_router` adds computed start padding until the classifier predicts no dead-zone source record.
- `tail_registry` duplicates exact answer-bearing fields immediately before the question.
- `dead_only_registry` copies only records classified as dead in the base prompt.
- `combo_original_dead_registry` adds ballast and still copies records that were dead before padding.

Use `raw.jsonl` for full prompts' position reports and model outputs.
