# Number Directions

Enable **Reports → Number Directions**, then open **View → Number Directions**
or its tab in the Statistics workspace. The internal report ID remains
`first-number-direction`, so existing saved enable/disable preferences apply.
Positions refer to ascending sorted values, not extraction order.

Six cards show the First through Sixth number forecasts together. Each compares
that position in the next draw with the same position in the latest loaded draw.
Each card shows the previous value, predicted direction, active model, and lower,
equal and higher probabilities alongside the exact baseline. The forecast follows
the latest loaded draw; it does not imply the stored dataset is current.

Use the position selector below the cards to inspect evaluation, activation
evidence, chronological blocks, confusion matrices and historical forecasts.
It starts on First. This report is independent of ticket rankings. No draws
produces an empty state; fewer than 120 draws produces exact baseline forecasts
without scored backtests. Dated draws must be chronological; undated history
retains its stored order.

## Exact reference model

For sorted position k (1 through 6), the exact distribution is
P(X_k=x) = C(x−1,k−1) C(49−x,6−k) / C(49,6), for k ≤ x ≤ 43+k.
Lower and higher probabilities sum this distribution below and above the
previous value; equality uses its probability at that value. These probabilities
assume independent, fair 6-from-49 draws and are not generally one-third.
Impossible directions are removed and probabilities normalized. Ties choose
lower, then equal, then higher. Position one retains the original minimum model.

## Learned candidates

Each position has independent models and evidence. Its historical value
distribution uses its exact sorted-position distribution as a 50-draw prior.
Its transition model conditions on the previous value and
uses the smoothed historical distribution as a 50-transition prior.

Logistic regression uses the last five values of that position, last three directions encoded
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
and Holm-adjusted one-sided p ≤ 0.05 across its three candidates. Correction is
performed separately for each position; it provides no simultaneous guarantee
across all six positions. The gate uses
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
provide a false-activation diagnostic. The same synthetic draw history is used
for all six positions within each replicate.

Run the complete study from the repository root:

```powershell
uv run python scripts/study_first_number_direction.py --stage all
```

Use `--stage historical` or `--stage synthetic` for separate runs. `--dataset`
selects a YAML source and `--output` selects the report directory, defaulting
to `reports/first_number_direction`. Outputs include protocol, dataset
fingerprints, per-dataset summaries, `position-1` through `position-6` folders
with summaries and forecast JSONL, aggregate results and
a Markdown report. Synthetic histories use seed 20261101 plus history index
and match the source draw count. App caches include all draw dates, numbers,
and protocol configuration; edits invalidate them automatically. Payload schema
and protocol version 2 invalidate the original single-position cache. The Python
`baseline_probabilities`, `normalize`, `analyze_direction` and `cached_direction`
entry points remain compatible with position one, including legacy minimum field
names. The expanded GUI payload uses `referenceValue` and `actualValue` and
contains one report per position.
