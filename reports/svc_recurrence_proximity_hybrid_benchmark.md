# SVC–Recurrence–Proximity Hybrid Benchmark

## Protocol

- Dataset: `data/lotto_results_2019.yaml`, 771 chronological draws.
- Evaluation: leakage-free walk-forward Top-6 predictions.
- Validation: target draws 121–520 (400 forecasts).
- Holdout: target draws 521–770 (250 forecasts).
- SRPH formula: 75% of the existing adaptive SVC–Recurrence score plus 25%
  Proximity rank strength.
- The theoretical random expectation is `36/49 = 0.734694` hits per draw.

Proximity and its 25% weight were chosen after comparing multiple strategies and
weights on this dataset. All results below are selection-biased regression
evidence, including the nominal holdout.

## Results

| Scope | Strategy | Hits | Mean hits/draw |
|---|---|---:|---:|
| Full 770 | SVC–Recurrence–Proximity Hybrid | 643 | 0.835065 |
| Full 770 | SVC–Recurrence Hybrid | 634 | 0.823377 |
| Full 770 | Recurrence Dynamics V2 | 625 | 0.811688 |
| Full 770 | SVC | 625 | 0.811688 |
| Validation 121–520 | SVC–Recurrence–Proximity Hybrid | 328 | 0.820000 |
| Validation 121–520 | SVC–Recurrence Hybrid | 325 | 0.812500 |
| Holdout 521–770 | SVC–Recurrence–Proximity Hybrid | 205 | 0.820000 |
| Holdout 521–770 | SVC–Recurrence Hybrid | 204 | 0.816000 |

## Planning-baseline reconciliation

The planning experiment reported 644 full-history hits and 329 validation hits.
The production benchmark reproducibly reports 643 and 328 because it applies the
specified gap tie-break to a mathematically exact score tie at target draw 161.
Numbers 23 and 30 both score `965377 / 1327776`; number 30 wins the tie because
its current gap is 2 rather than 1. The exploratory calculation's floating-point
evaluation order placed number 23 sixth instead, and number 23 happened to be in
the target draw. The holdout total is unaffected at 205.

This correction is intentionally retained: changing the tie outcome merely to
recover the planning total would violate the fixed formula and existing gap/number
tie-break.

## Decision

SRPH remains a default-disabled shadow strategy. Its historical improvement is
small, especially on the holdout, and cannot be treated as independent evidence
after candidate and weight selection. The 25% Proximity weight is frozen for
evaluation on future untouched draws.
