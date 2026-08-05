# MKGSV v2 promotion report

Leakage-safe 320/200/250 warm-up, validation, and holdout benchmark.

## Selected configuration

Singles `32`, pairs `128`, triples `512`, `historical` evidence, replacement margin `0.0025`.

## Results

| Scope | Gated MKGSV | Raw shadow | Markov 100 | Gated gain | Proposals | Active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 148 | 146 | 148 | +0 | 97 | 27 |
| Holdout | 207 | 194 | 207 | +0 | 107 | 0 |

Validation gated/Markov Brier: `0.107557` / `0.107562`.
Holdout gated/Markov Brier: `0.107471` / `0.107471`.

Complete distributions, paired differences, replacement identities, related baselines, and state support are in the JSON report.

## Promotion decision

**Failed.** MKGSV remains selectable, experimental, and disabled by default.
