# SRPH Residual Diversity Hybrid Benchmark

## Protocol

- Dataset: `data/lotto_results_2019.yaml`, 771 chronological draws.
- Evaluation: leakage-free walk-forward Top-6 predictions.
- Validation: target draws 121–520 (400 forecasts).
- Holdout: target draws 521–770 (250 forecasts).
- Residual candidates: Freshness, EMD, Bayesian, and Doublet/Triplet Markov.
- Candidate blend: 90% SRPH score plus 10% candidate rank strength.
- Selector: cumulative counterfactual Top-6 quality with a neutral 24-draw prior;
  fall back to SRPH unless the best candidate is strictly better.

The pool and 10% residual weight were fixed after exploratory comparisons on this
dataset. The result is therefore selection-biased and the nominal holdout is not
independent confirmation.

## Results

| Scope | Strategy | Hits | Mean hits/draw |
|---|---|---:|---:|
| Full 770 | SRPH Residual Diversity Hybrid | 642 | 0.833766 |
| Full 770 | SVC–Recurrence–Proximity Hybrid | 643 | 0.835065 |
| Validation 121–520 | SRPH Residual Diversity Hybrid | 333 | 0.832500 |
| Validation 121–520 | SVC–Recurrence–Proximity Hybrid | 328 | 0.820000 |
| Holdout 521–770 | SRPH Residual Diversity Hybrid | 200 | 0.800000 |
| Holdout 521–770 | SVC–Recurrence–Proximity Hybrid | 205 | 0.820000 |

Across all 770 forecasts, SRD fell back to SRPH 248 times. It selected EMD 29
times, Bayesian 15 times, Doublet/Triplet Markov 478 times, and Freshness zero
times.

## Decision

SRD remains a default-disabled shadow strategy. Its validation improvement does
not offset the five-hit holdout deficit, and historical selection cannot establish
predictability. The candidate pool, prior, fallback rule, tie order, and 10%
weight are frozen for evaluation on future untouched draws.
