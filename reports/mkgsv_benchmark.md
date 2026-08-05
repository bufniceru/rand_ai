# Markov Gap-Space Vector promotion report

Generated from `data/lotto_results_2019.yaml` with
`scripts/benchmark_mkgsv.py`. The 770 evaluated predictions are split into 320
warm-up draws, 200 development-validation draws, and an untouched latest-250
holdout.

## Selected development configuration

| Prior level | Strength |
| --- | ---: |
| Single state | 64 |
| Pair state | 4 |
| Ordered triple state | 2 |

The completed history contains 37,381 triple exposures across 10,473 unique
ordered states. Median exact-triple support is two exposures.

| Exact-triple support | State count |
| --- | ---: |
| 1 exposure | 4,647 |
| 2 exposures | 1,804 |
| 3–5 exposures | 2,163 |
| 6–10 exposures | 1,117 |
| More than 10 | 742 |

## Results

| Scope and strategy | Hits | Mean | Zero | One | Two-plus |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation — MKGSV | 143 | 0.715 | 89 | 83 | 28 |
| Validation — Markov 100 | 148 | 0.740 | 88 | 83 | 29 |
| Validation — Freshness | 149 | 0.745 | 88 | 81 | 31 |
| Validation — Proximity | 161 | 0.805 | 78 | 86 | 36 |
| Validation — Combined | 140 | 0.700 | 89 | 83 | 28 |
| Holdout — MKGSV | 178 | 0.712 | 119 | 87 | 44 |
| Holdout — Markov 100 | 207 | 0.828 | 93 | 117 | 40 |
| Holdout — Freshness | 191 | 0.764 | 102 | 109 | 39 |
| Holdout — Proximity | 173 | 0.692 | 114 | 103 | 33 |
| Holdout — Combined | 177 | 0.708 | 101 | 126 | 23 |

The random expectations are 146.94 validation hits and 183.67 holdout hits.
MKGSV's all-number Brier scores are `0.117864` on validation and `0.116041` on
holdout.

| Paired against strongest scope baseline | Wins | Ties | Losses | Mean hit difference | 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation vs Proximity | 56 | 75 | 69 | -0.090 | [-0.2358, 0.0558] |
| Holdout vs Markov 100 | 70 | 89 | 91 | -0.116 | [-0.2551, 0.0231] |

## Promotion decision

**Failed.** MKGSV did not beat the strongest related baseline on validation and
trailed Markov 100 on holdout. It remains a selectable experimental strategy
and is disabled by default. The complete machine-readable result is preserved
in `reports/mkgsv_benchmark.json`.
