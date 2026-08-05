# CIS v3 promotion report

Generated from `data/lotto_results_2019.yaml` with
`scripts/benchmark_cis_v3.py`. The dataset contains 770 evaluated predictions:
320 warm-up draws, 200 development-validation draws, and an untouched latest-250
holdout.

## Selected development configuration

| Parameter | Value |
| --- | ---: |
| Recent window | 80 |
| Recent weight | 0.60 |
| Correlation threshold | 0.80 |
| Minimum independent support | 0.20 |
| Minimum peer gain | 0.10 |
| Maximum replacements | 1 |

## Results

| Scope | Current CIS | CIS v3 | Adaptive champion | Strongest fixed expert |
| --- | ---: | ---: | ---: | ---: |
| Validation (200) | 143 | 172 | 172 | 178 (SVC) |
| Holdout (250) | 174 | 188 | 188 | 207 (Markov 100) |

Against the strongest fixed expert, CIS v3 had 35 wins, 127 ties, and 38 losses
on validation (mean difference -0.030 hits/draw; 95% interval -0.145 to 0.085).
On holdout it had 73 wins, 93 ties, and 84 losses (mean difference -0.076;
95% interval -0.192 to 0.040).

The correction generator produced 726 evaluable shadow proposals with a net
loss of 13 hits. The evidence gate therefore applied zero production
corrections, leaving CIS v3 equal to its adaptive champion.

## Promotion decision

**Failed.** CIS v3 improved on current CIS but did not beat the adaptive
champion or strongest fixed expert on validation and trailed Markov 100 on the
holdout. The public `cis` implementation remains unchanged.

