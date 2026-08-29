# Recurrence Dynamics V1/V2 Benchmark

## Protocol

- Dataset: `data/lotto_results_2019.pkl`, 771 chronological draws.
- Evaluation: leakage-free walk-forward Top-6 predictions.
- V2 fixed scopes: target draws 121–520 for 400 validation forecasts and target
  draws 521–770 for 250 holdout forecasts.
- Additional scopes: all 770 evaluable targets and the latest 250 targets
  (522–771), retained for direct comparison with the historical V1 report.
- V2 parameters: 18-value three-draw embedding, 8 analogues, inverse-distance
  weights, and prior strength 8; no adjustment was made after this run.
- Comparators: theoretical random expectation, deterministic Random baseline,
  Earth Mover Distance, and Markov Spaces.

No separate full-history pickle is currently present in the repository, so the
previously benchmarked 2,562-draw dataset could not be rerun for this change.

V2 was designed after exploratory analysis of this dataset. These results are
therefore regression and locked-slice evidence, not independent confirmation.

## Historical V1 baseline

The theoretical random expectation is **0.734694 hits/draw**.

| Scope | Strategy | Hits | Mean hits/draw | Lift vs theoretical random |
|---|---|---:|---:|---:|
| Full 770 | Recurrence Dynamics | 558 | 0.724675 | -0.010019 |
| Recent 250 | Recurrence Dynamics | 170 | 0.680000 | -0.054694 |

## Fixed V2 results

| Scope | Strategy | Hits | Mean hits/draw | Lift vs theoretical random |
|---|---|---:|---:|---:|
| Full 770 | Recurrence Dynamics V2 | 625 | 0.811688 | +0.076994 |
| Full 770 | Deterministic Random | 573 | 0.744156 | +0.009462 |
| Full 770 | Earth Mover Distance | 592 | 0.768831 | +0.034137 |
| Full 770 | Markov Spaces | 596 | 0.774026 | +0.039332 |
| Validation 121–520 | Recurrence Dynamics V2 | 342 | 0.855000 | +0.120306 |
| Validation 121–520 | Deterministic Random | 304 | 0.760000 | +0.025306 |
| Validation 121–520 | Earth Mover Distance | 316 | 0.790000 | +0.055306 |
| Validation 121–520 | Markov Spaces | 307 | 0.767500 | +0.032806 |
| Holdout 521–770 | Recurrence Dynamics V2 | 199 | 0.796000 | +0.061306 |
| Holdout 521–770 | Deterministic Random | 184 | 0.736000 | +0.001306 |
| Holdout 521–770 | Earth Mover Distance | 187 | 0.748000 | +0.013306 |
| Holdout 521–770 | Markov Spaces | 195 | 0.780000 | +0.045306 |
| Recent 250 | Recurrence Dynamics V2 | 197 | 0.788000 | +0.053306 |
| Recent 250 | Deterministic Random | 184 | 0.736000 | +0.001306 |
| Recent 250 | Earth Mover Distance | 186 | 0.744000 | +0.009306 |
| Recent 250 | Markov Spaces | 194 | 0.776000 | +0.041306 |

The production ranking uses the application's normal gap-and-number tie-break,
which explains its small difference from the exploratory core-ranking estimate.
Over the report's latest 750-draw window, V2 averages `0.825100` hits/draw with a
95% lower bound of `0.767367`, but the combined Nonlinear Dynamics verdict remains
**weak** because the unchanged surrogate determinism test has `p = 0.96`.

V2 remains opt-in and experimental. Promotion requires replication on an untouched
larger dataset or future draws, including a lower confidence bound above random and
stable comparison with Markov Spaces.
