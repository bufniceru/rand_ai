# First Number Direction

Enable **First Number Direction** in report settings and open its statistics
view or View menu entry. “First” means the smallest of the six sorted numbers,
not the first ball extracted. The forecast follows the latest loaded draw;
it does not imply the stored dataset is current.

The view shows lower, equal and higher probabilities, the most probable
direction, the active model, chronological evaluation and expandable forecast
history. This report is independent of ticket rankings. No draws produces an
empty state; fewer than 120 draws produces an exact baseline forecast without
scored backtests. Dated draws must be chronological; undated history retains
its stored order.

## Exact reference model

For previous minimum m, and D = C(49,6), probabilities are:

- Lower: 1 − C(50−m,6)/D.
- Equal: C(49−m,5)/D.
- Higher: C(49−m,6)/D.

These follow from the minimum distribution P(M=m)=C(49−m,5)/D under independent,
fair 6-from-49 draws. Thus direction probabilities are not generally one-third.
Impossible directions are removed and probabilities normalized. Ties choose
lower, then equal, then higher.

## Learned candidates

The historical minimum distribution uses the exact minimum distribution as
a 50-draw prior. The transition model conditions on the previous minimum and
uses the smoothed historical distribution as a 50-transition prior.

Logistic regression uses the last five minima, last three directions encoded
as lower=0, equal=1, higher=2, and population mean and standard deviation over
10 and 30 draws. It uses training-only standardization, L2 C=0.1, and refits
every 20 draws from the 120-draw warm-up boundary. Training examples begin
with 30 observed draws. With one training class it retains the baseline;
with two classes it maps predictions to those classes. Three-class training
uses multinomial regression.

## Evidence gate and evaluation

Backtesting starts at draw 121, forecasting before observing its result.
At 20-forecast intervals, after at least 200 completed forecasts, the lowest
mean-log-loss candidate can activate. It must have a positive 95% bootstrap
lower confidence bound on paired baseline-minus-candidate loss differences,
and Holm-adjusted one-sided p ≤ 0.05 across all three candidates. The gate uses
2,000 moving-block bootstrap samples, contiguous blocks of length 20 sampled
with replacement and truncated to the evaluation length. P-values use the
centered bootstrap null distribution with an add-one correction. Seed: 20261101.
Otherwise the exact baseline stays active. The selected model remains fixed
between reviews; historical decisions are never rewritten using later results.

Reported metrics include log loss (primary), summed multiclass Brier score,
accuracy, confusion matrices, class recall, and five consecutive evaluation
blocks. Missing-class recall is shown as unavailable. Log-loss scoring clips
the realized class probability at 1e-15. The gated policy is scored separately.
The gate does not establish a permanent advantage or correct for repeated
reviews over time. Historical findings remain exploratory because this history
has previously informed research. Twenty independent synthetic histories
provide a false-activation diagnostic.

Run the complete study from the repository root:

```powershell
uv run python scripts/study_first_number_direction.py --stage all
```

Use `--stage historical` or `--stage synthetic` for separate runs. `--dataset`
selects a YAML source and `--output` selects the report directory, defaulting
to `reports/first_number_direction`. Outputs include protocol, dataset
fingerprints, per-dataset summaries and forecast JSONL, aggregate results and
a Markdown report. Synthetic histories use seed 20261101 plus history index
and match the source draw count. App caches include all draw dates, numbers,
and protocol configuration; edits invalidate them automatically.
