# SVC–Recurrence Hybrid Benchmark

## Protocol

- Dataset: `data/lotto_results_2019.yaml`, 771 chronological draws.
- Evaluation: leakage-free walk-forward Top-6 predictions.
- Validation: target draws 121–520 (400 forecasts).
- Holdout: target draws 521–770 (250 forecasts).
- Hybrid: full-rank blend of SVC and Recurrence Dynamics using cumulative
  effectiveness available before each target draw and a neutral 24-draw prior.
- The theoretical random expectation is `36/49 = 0.734694` hits per draw.

The hybrid rule was chosen after inspecting this dataset. These results are
regression and locked-slice evidence, not independent confirmation.

## Results

| Scope | Strategy | Hits | Mean hits/draw |
|---|---|---:|---:|
| Full 770 | SVC–Recurrence Hybrid | 634 | 0.823377 |
| Full 770 | Recurrence Dynamics V2 | 625 | 0.811688 |
| Full 770 | SVC | 625 | 0.811688 |
| Validation 121–520 | SVC–Recurrence Hybrid | 325 | 0.812500 |
| Validation 121–520 | Recurrence Dynamics V2 | 342 | 0.855000 |
| Validation 121–520 | SVC | 335 | 0.837500 |
| Holdout 521–770 | SVC–Recurrence Hybrid | 204 | 0.816000 |
| Holdout 521–770 | Recurrence Dynamics V2 | 199 | 0.796000 |
| Holdout 521–770 | SVC | 187 | 0.748000 |

## Decision

The hybrid improves on both sources over the full history and holdout, but it
trails both sources on the earlier validation slice. It therefore remains an
opt-in experimental strategy. Promotion requires stable results on future,
untouched draws; historical improvement is not evidence that lottery outcomes
have become predictable.
