# ARIMA (Experimental)

This opt-in strategy is available in **Settings → Strategies → Markov & Sequence**.
It adapts the independent positional ARIMA(5,1,0) baseline in
[Lotto-prediction](https://github.com/michaelkupfer97/Lotto-prediction/blob/master/lottoProd.py).
This is an independent implementation; no upstream source code is copied.

## Basic implementation

Six sorted number positions are modeled separately. Each series is differenced
once; its preceding five differences predict the next difference with no drift
or moving-average terms. NumPy conditional least squares estimates the five AR
coefficients. This differs from the reference's statsmodels likelihood fit and
will not reproduce its numerical results exactly. The latest 250 completed draws
bound fitting cost. Before 24 observations, positional historical means are used;
empty history uses the uniform order-statistic means. Solver failures also use
positional means.

Each forecast is clipped to its position's feasible 6/49 range. Gaussian kernels
with residual-RMS widths (minimum one ball) turn forecasts into 49 relative
scores. Each positional kernel is normalized before averaging. The existing
ranking system selects six distinct numbers. Scores are not calibrated inclusion
probabilities. Forecasts are computed from completed draws only, so the existing
walk-forward hit statistics can evaluate the strategy without future outcomes.
It is excluded from default desktop selection and residual coverage dependencies.

## Further study

Use chronological expanding or rolling training folds, then an untouched final
time block. Compare mean hits, the full hit distribution, and uncertainty against
uniform random selection, positional means, and last-position forecasts. A random
six-number ticket has expected hits 36/49 per draw; predictive benefit has not
been established for this implementation.

1. Compare conditional least squares with statsmodels maximum likelihood on
   identical prefixes, including residual autocorrelation and unstable AR roots.
2. Tune small p/d/q grids and training windows using past-only validation;
   freeze choices before the final test to control selection bias.
3. Compare sorted-position models with per-number occurrence or gap series.
   Sorted positions are order statistics, not persistent ball identities.
4. Study stationarity constraints, uncertainty calibration, and a joint
   ordered-position forecast instead of independent kernels.
5. Measure replay runtime and cache compatibility before adding richer models.

These are research directions, not demonstrated improvements. No study results
are claimed until a reproducible evaluation has been run.
