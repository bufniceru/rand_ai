# MKGSV v3 promotion report

Leakage-safe 320/200/250 warm-up, validation, and holdout benchmark.

## Selected configuration

Correction off; no configuration had positive raw validation gain.

## Results

| Scope | Gated MKGSV | Raw motif | Markov 100 | Gated gain | Raw gain | Proposals | Active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 148 | 148 | 148 | +0 | +0 | 0 | 0 |
| Holdout | 207 | 207 | 207 | +0 | +0 | 0 | 0 |

Validation gated/raw/Markov Brier: `0.107562` / `0.107562` / `0.107562`.
Holdout gated/raw/Markov Brier: `0.107471` / `0.107471` / `0.107471`.

Complete distributions, paired differences, replacements, component ablations, null support, and related baselines are in the JSON report.

## Promotion decision

**Failed.** MKGSV remains experimental and disabled by default; production output is exactly Markov 100.
