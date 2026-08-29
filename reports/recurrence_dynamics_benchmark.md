# Recurrence Dynamics v1 Benchmark

## Protocol

- Dataset: `data/lotto_results_2019.pkl`, 771 chronological draws.
- Evaluation: leakage-free walk-forward Top-6 predictions.
- Scopes: all 770 evaluable targets and the latest 250 targets.
- Parameters: the fixed v1 parameters documented before this evaluation.
- Comparators: theoretical random expectation, deterministic Random baseline,
  Earth Mover Distance, and Markov Spaces.

No separate full-history pickle is currently present in the repository, so the
previously benchmarked 2,562-draw dataset could not be rerun for this change.

## Results

The theoretical random expectation is **0.734694 hits/draw**.

| Scope | Strategy | Hits | Mean hits/draw | Lift vs theoretical random |
|---|---|---:|---:|---:|
| Full 770 | Recurrence Dynamics | 558 | 0.724675 | -0.010019 |
| Full 770 | Deterministic Random | 573 | 0.744156 | +0.009462 |
| Full 770 | Earth Mover Distance | 592 | 0.768831 | +0.034137 |
| Full 770 | Markov Spaces | 596 | 0.774026 | +0.039332 |
| Recent 250 | Recurrence Dynamics | 170 | 0.680000 | -0.054694 |
| Recent 250 | Deterministic Random | 184 | 0.736000 | +0.001306 |
| Recent 250 | Earth Mover Distance | 186 | 0.744000 | +0.009306 |
| Recent 250 | Markov Spaces | 194 | 0.776000 | +0.041306 |

The default Nonlinear Dynamics diagnostic over its latest-750-draw window reports
**weak evidence**, with surrogate determinism `p = 0.96`. Recurrence Dynamics
does not beat theoretical random in either benchmark scope, so it remains opt-in
and experimental. These results were recorded without parameter tuning.
