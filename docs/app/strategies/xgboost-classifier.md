# XGBoost Classifier

`xgboost_classifier` is an opt-in strategy in **Relationships & Machine Learning**.
Enable it in Settings → Strategies. It returns all 49 candidates and uses the first
six for prediction grids, hit histories, comparisons, exports, and portfolios.

## Historical inputs and labels

The classifier shares the exact 14 [Lagged Logistic](lagged-logistic.md) features:
membership in the latest three draws, lag-hit share, clipped gap, three gap
indicators, overdue ratio, lifetime frequency residual, recent 5/20/100 frequency
residuals, and the recent 5-vs-20 trend. It consumes no other strategy outputs.
Feature snapshots precede their target draw. Each subsequent draw labels the
snapshot's 49 rows: six positives and 43 negatives. No future draw is an input.

## Training and scores

The first fit follows 120 completed transitions; later fits occur every ten
transitions, from scratch, on at most the latest 500 transitions. Between fits
the estimator stays fixed. Before activation, scores are
`(appearances + 24 * 6/49) / (completed_draws + 24)`.

The fixed CPU XGBClassifier uses binary logistic loss, histogram trees, 100
estimators, depth 2, learning rate 0.05, minimum child weight 10, L2 penalty 10,
L1 penalty 0, full row/column sampling, class scale 1, base score 6/49, one thread,
and seed 20260626. There is no tuning or early stopping. Unit example weights
preserve the six-to-43 label ratio. Dependencies and fitting errors use the
application's analysis error path.

Positive-class scores sort descending, with ties resolved by ascending number.
Details show warm-up status, completed transitions, buffer size, and last fit.
Scores are uncalibrated marginal estimates, not joint ticket probabilities.
Exactly six picks are selected by ranking, without independent thresholds.

## Evaluation

Run `uv run python scripts/study_xgboost.py` for the frozen chronological study.
It compares trained XGBoost predictions with Lagged Logistic, smoothed frequency,
and the seeded random strategy on identical target draws. Reports include five
chronological blocks, hit distributions, paired moving-block bootstrap intervals,
and runtime. The theoretical random mean is 36/49 hits per six-number ticket.
The bundled history has influenced prior development, so results are exploratory.
Fair independent draws provide no guaranteed predictive advantage.

Dataset edits and reloads rebuild causal state. Strategy caches use schema 26.
Models are not persisted separately or shared between datasets.
