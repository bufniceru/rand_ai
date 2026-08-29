# Categorical Chi-square

## Introduction

**Categorical Chi-square** is the production strategy with identifier
`categorical_chi_square` and short engine name **Cat χ²**. It is a
default-enabled member of the **Frequency & Recency** family. For every number
from 1 through 49, the engine learns whether that number appeared after exact
combinations of three historical state variables:

- its current recurrence gap;
- the empty circular space immediately to its left when it last appeared;
- the empty circular space immediately to its right when it last appeared.

Separate contingency tables measure each single state, each pair, and the full
three-way state. Their effects are shrunk through a fixed hierarchy and combined
on the log-odds scale to produce one score for every candidate. The first six
numbers in the final 1–49 ranking form the strategy's Top-6 prediction.

```{admonition} Distinct from Chi-square Frequency
:class: important

`categorical_chi_square` is not the `chi_square` engine documented in
**Chi-square Frequency**. Chi-square Frequency ranks signed lifetime count
residuals. Categorical Chi-square learns number-specific conditional hit tables
from exact gap and spacing states. Neither strategy is an alias for the other.
```

## Scope

This page documents the production state variables, seven table views,
incremental contingency statistics, baseline prior, hierarchical smoothing,
association-weighted logit adjustments, causal update order, ranking,
interpretation, retrospective replay, and limitations.

The model's output is displayed as an estimated probability, but it is not
validated as a calibrated probability of a future lottery result. Its
chi-square statistics and corrected Cramér's V values control descriptive model
adjustments; the strategy does not perform or report a conventional global
hypothesis test or p-value.

## Prediction problem

For each candidate {math}`n\in\{1,\ldots,49\}`, every completed draw supplies
one binary outcome

```{math}
y_{t,n}=\mathbf{1}[n\in D_t].
```

There are therefore 49 exposures per draw: six hits and 43 misses. Before
outcome {math}`y_{t,n}` is learned, the model captures the candidate's state
from draws through {math}`t-1`. The strategy estimates how the binary hit rate
for that particular number has varied across exact historical categories of
that state.

Every candidate has its own baseline counts and its own seven contingency
tables. Evidence is not pooled between different numbers.

## Exact categorical state

### Zero-based recurrence gap

After {math}`h` draws have been remembered, let {math}`\ell_n` be the zero-based
index of the most recent draw containing number {math}`n`. The state gap is

```{math}
g_n=
\begin{cases}
h, & n\text{ has never appeared},\\
h-\ell_n-1, & \text{otherwise}.
\end{cases}
```

A number present in the latest remembered draw has {math}`g_n=0`; one absent
for one complete intervening draw has {math}`g_n=1`. An unseen number's gap
grows with the completed history length.

### Circular left and right spaces

For a sorted draw
{math}`n_1<n_2<\cdots<n_6`, define its six circular empty spaces as

```{math}
e_1=(n_1-1)+(49-n_6),
```

```{math}
e_i=n_i-n_{i-1}-1,\qquad i=2,\ldots,6.
```

The spaces are nonnegative and satisfy

```{math}
\sum_{i=1}^{6}e_i=43.
```

When number {math}`n_i` appears, its stored left space is {math}`e_i`. Its
stored right space is {math}`e_{i+1}`, with circular wrap so that the right
space of {math}`n_6` is {math}`e_1`. These values remain unchanged until the
number appears again. A number that has never appeared stores `unseen`,
represented internally by `None`, for both spaces.

Thus the current state for candidate {math}`n` is

```{math}
x_n=(g_n,L_n,R_n).
```

The spaces describe the candidate's neighborhood at its most recent
appearance, not the spacing of the latest draw when the candidate was absent.

## Seven exact table views

The model expands {math}`x_n` into seven categorical views:

| Level | View | Exact category |
|---|---|---|
| Single | Gap | {math}`g_n` |
| Single | Left | {math}`L_n` |
| Single | Right | {math}`R_n` |
| Pair | Gap–left | {math}`(g_n,L_n)` |
| Pair | Gap–right | {math}`(g_n,R_n)` |
| Pair | Left–right | {math}`(L_n,R_n)` |
| Triple | Full state | {math}`(g_n,L_n,R_n)` |

For each candidate and view, an exact category cell stores two counts:

```{math}
h_c=\text{hits observed in category }c,
\qquad
n_c=\text{total exposures observed in category }c.
```

No gap bins or space bins are used. Exact categories preserve detail but create
many sparse cells, especially for pairs and triples. The fixed smoothing and
support factors described below are the production response to that sparsity.

## Contingency-table mathematics

### Table-wide chi-square statistic

Within one number and one view, let {math}`N` be total exposures,
{math}`H` total hits, and {math}`M=N-H` total misses across all observed
categories. Production maintains

```{math}
S=\sum_c\frac{h_c^2}{n_c}
```

incrementally and calculates the categorical-by-binary Pearson statistic as

```{math}
\chi^2=
\frac{N^2}{HM}
\left(S-\frac{H^2}{N}\right).
```

This is algebraically equivalent to summing Pearson contributions across the
hit and miss columns of the category-by-outcome contingency table. Production
returns 0 when the table is empty, contains no hits, or contains no misses, and
clips a small negative floating-point result to 0.

### Current-cell residual

For the candidate's current category {math}`c`, the expected hit count under
the table-wide hit rate is

```{math}
E_c=n_c\frac{H}{N}.
```

The displayed hit-column residual is

```{math}
r_c=\frac{h_c-E_c}{\sqrt{E_c}}.
```

It is 0 when the current cell has no support, the table is empty, or its
expected hit count is nonpositive. A positive residual means the current
category's observed hit rate is above the table's overall rate; a negative
residual means it is below.

### Bias-corrected Cramér's V

The model converts the table-wide statistic to a bias-corrected association
magnitude. With {math}`R` observed category rows and {math}`C=2` outcome
columns, define

```{math}
\phi^2=\frac{\chi^2}{N},
\qquad
\phi^2_{\mathrm{corr}}=
\max\left(0,\phi^2-\frac{(R-1)(C-1)}{N-1}\right),
```

```{math}
R_{\mathrm{corr}}=R-\frac{(R-1)^2}{N-1},
\qquad
C_{\mathrm{corr}}=C-\frac{(C-1)^2}{N-1}.
```

The production value is

```{math}
V_{\mathrm{corr}}=
\min\left(
1,
\sqrt{
\frac{\phi^2_{\mathrm{corr}}}
{\min(R_{\mathrm{corr}}-1,C_{\mathrm{corr}}-1)}
}
\right).
```

It falls back to 0 when there are too few exposures or categories, or when the
corrected denominator is nonpositive. Cramér's V supplies association
magnitude; the current residual supplies direction.

## Baseline and hierarchical smoothing

### Number-specific baseline

Let {math}`H_n` and {math}`N_n` be the lifetime hits and exposures for candidate
{math}`n`. The baseline is

```{math}
b_n=\frac{H_n+6}{N_n+49}.
```

The added 6 hits and 43 misses form 49 pseudo-exposures centered exactly on
{math}`6/49`. Algebraically, this is the posterior mean associated with a
Beta-style {math}`(6,43)` prior for the candidate's binary hit rate, although
the complete strategy is not a formal Bayesian generative model.

### Smoothed cell probability

Given a prior probability {math}`p` and prior strength {math}`\lambda`, a
current category cell is smoothed as

```{math}
\widetilde p_c=
\frac{h_c+\lambda p}{n_c+\lambda}.
```

The cell's reliability factor is

```{math}
\rho_c=\frac{n_c}{n_c+\lambda}.
```

An unseen cell has {math}`\rho_c=0` and therefore contributes no adjustment.
Support must grow before a category can substantially move the baseline.

### Association-weighted logit adjustment

Let

```{math}
\operatorname{logit}(p)=\log\frac{p}{1-p}.
```

Production bounds logit inputs to
{math}`[10^{-9},1-10^{-9}]`. It first computes the magnitude of the smoothed
cell's log-odds difference from its prior, then forces its sign to agree with
the current Pearson residual:

```{math}
\delta_c=
\begin{cases}
0, & r_c=0,\\
\operatorname{sign}(r_c)
\left|\operatorname{logit}(\widetilde p_c)-
\operatorname{logit}(p)\right|, & r_c\neq0.
\end{cases}
```

The final adjustment from that table is

```{math}
A_c=\delta_c\,V_{\mathrm{corr}}\,\rho_c.
```

A category can therefore affect the score only when it has current-cell
direction, table-wide association, and nonzero support reliability. The
chi-square statistic is not added directly to the candidate score.

## Exact hierarchy

The three single views use baseline {math}`b_n` as their prior and fixed
strength 12:

```{math}
\widetilde p_g,\widetilde p_L,\widetilde p_R
\quad\text{with}\quad \lambda=12.
```

The three pair priors are averages of the corresponding single probabilities:

```{math}
p_{gL}=\frac{\widetilde p_g+\widetilde p_L}{2},
\qquad
p_{gR}=\frac{\widetilde p_g+\widetilde p_R}{2},
```

```{math}
p_{LR}=\frac{\widetilde p_L+\widetilde p_R}{2},
```

and each pair uses fixed strength 24. The triple prior is the mean of the three
smoothed pair probabilities,

```{math}
p_{gLR}=\frac{
\widetilde p_{gL}+\widetilde p_{gR}+\widetilde p_{LR}
}{3},
```

with fixed strength 48.

Let {math}`A_g,A_L,A_R` be the three single adjustments,
{math}`A_{gL},A_{gR},A_{LR}` the pair adjustments, and {math}`A_{gLR}` the
triple adjustment. Production combines them as

```{math}
\overline A_{\mathrm{single}}=\frac{A_g+A_L+A_R}{3},
```

```{math}
\overline A_{\mathrm{pair}}=
\frac{A_{gL}+A_{gR}+A_{LR}}{3},
```

```{math}
s_n=\sigma\left(
\operatorname{logit}(b_n)
+\overline A_{\mathrm{single}}
+\overline A_{\mathrm{pair}}
+A_{gLR}
\right),
```

where {math}`\sigma(z)=1/(1+e^{-z})` and production bounds {math}`z` to
{math}`[-35,35]` before applying the sigmoid.

The displayed **effective backoff** label is the highest hierarchy level with a
nonzero adjustment: triple, otherwise pair, otherwise single, otherwise
baseline. It is an audit summary, not an exclusive switch. The final formula
always includes every nonzero single, pair, and triple adjustment.

## Causal walk-forward lifecycle

When draw {math}`D_t` becomes observed, the integration performs these steps in
order:

1. For every candidate, capture {math}`(g_n,L_n,R_n)` from history through
   {math}`D_{t-1}`.
2. Record whether the candidate appears in {math}`D_t` in its baseline counts
   and all seven exact-state tables.
3. Only after learning those outcomes, update last-seen and circular-space state
   for the six numbers in {math}`D_t`.
4. Increment the completed-draw count.
5. Score the resulting states for target draw {math}`D_{t+1}`.

The target outcome never defines its own predictor state. A number's spaces
from {math}`D_t` can influence the forecast of {math}`D_{t+1}`, but the tables
learn whether those spaces were predictive only from previously completed
state/outcome pairs. Appending future draws cannot alter a prediction already
produced for an earlier prefix.

## Cold start

Before any completed draw, every candidate has baseline

```{math}
b_n=\frac{6}{49},
```

state {math}`(0,\text{unseen},\text{unseen})`, and empty contingency tables.
All association adjustments are zero, so all 49 scores equal {math}`6/49`.

In the normal lifecycle, the first displayed forecast is built after the first
draw is learned and remembered. A number that appeared has baseline
{math}`7/50=0.14`; a number that did not appear has baseline
{math}`6/50=0.12`. The tables are still degenerate or unsupported for the new
post-draw states, so their corrected associations contribute no adjustment.
The first Top-6 consequently consists of the six numbers from the first draw.

## Ranking and tie-breaking

The final values {math}`s_n` pass directly into the common ranking contract;
they are not min–max scaled. Candidates are ordered by:

1. larger estimated score;
2. larger application-level zero-based current gap when scores tie;
3. smaller number when both score and current gap tie.

The application gap normally matches the recurrence meaning used in the model,
but it is maintained by the surrounding prediction state and is used only for
tie-breaking. The first six candidates form the Top-6.

## Interpreting the application fields

| Field | Meaning |
|---|---|
| Score / Estimated probability | Final hierarchical sigmoid score {math}`s_n`, displayed as a percentage. It is a model estimate, not a verified calibrated probability. |
| Baseline | Number-specific pseudo-count-smoothed lifetime rate {math}`b_n`. |
| Lift | Ratio {math}`s_n/b_n`; values above 1 indicate an upward conditional adjustment and values below 1 a downward adjustment. |
| Exact state | Current gap and the last stored left and right spaces; unseen spaces are displayed explicitly. |
| Support | Number of historical exposures in the current exact category for the displayed table. |
| Residual | Current hit-cell Pearson residual {math}`r_c`, supplying adjustment direction. |
| Corrected V | Bias-corrected Cramér's V for the full categorical-by-outcome table, supplying association magnitude. |
| Chi-square | Table-wide Pearson statistic for that view. It is descriptive and has no displayed p-value. |
| Triple support | Exposures observed in the exact {math}`(g,L,R)` category. |
| Effective backoff | Highest hierarchy level with a nonzero adjustment. All lower nonzero adjustments remain included. |
| Rank and Top-6 | Final position and membership after score, gap, and number ordering. |

Per-number details display complete evidence lines for the three single views
(Gap, Left, and Right), plus triple support and the effective-backoff label.
Pair tables and the triple's full statistic still participate in scoring even
though their complete evidence lines are not printed in the tooltip.

## Statistical interpretation

### What chi-square contributes

For each number and view, {math}`\chi^2` summarizes how much hit rates differ
among all observed categories. Corrected Cramér's V converts that statistic into
a bounded association magnitude after a finite-sample correction. The current
cell residual determines whether that category points upward or downward.

This separation prevents a large table-wide association from assigning the
same direction to every category. It also means a category with no support, a
zero residual, or corrected {math}`V=0` cannot move the baseline.

### Why this is not a significance test

The model creates seven tables for each of 49 numbers and repeatedly evaluates
time-dependent exact categories. Standard fixed-table chi-square p-values would
require assumptions and multiplicity treatment not implemented here.
Sequential gaps and stored spaces are derived from the same outcomes being
modeled, exact cells can be sparse, and candidate outcomes within a draw are
dependent because exactly six distinct numbers are selected.

Production therefore uses {math}`\chi^2` and corrected V as effect-size
ingredients only. It does not compare them with a critical value, assign
degrees of freedom, or label individual tables statistically significant.

### Top-6 null baseline

For any six-number prediction evaluated against an independent uniform
six-number draw,

```{math}
H\sim\operatorname{Hypergeometric}(N=49,K=6,n=6),
```

with

```{math}
\mathbb{E}[H]=\frac{36}{49}\approx0.734694,
```

```{math}
\operatorname{Var}(H)=
6\frac{6}{49}\frac{43}{49}\frac{43}{48}
\approx0.577572.
```

This is the theoretical overlap reference for efficacy totals, not a
calibration guarantee for {math}`s_n`.

## Leakage-free replay evidence

A source-concordance replay through the production strategy path uses the
repository's 771 chronological YAML draws and yields 770 target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 600 | 0.779221 | 565.714 |
| Validation, target draws 121–520 | 400 | 321 | 0.802500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 188 | 0.752000 | 183.673 |

The holdout excess over the theoretical random mean is small, and the stronger
validation behavior does not persist at the same level. These results verify a
reproducible causal implementation on the current dataset; they do not
demonstrate stable future predictability or calibrated probabilities.

No statistical significance is claimed. The model, hierarchy, and strategy
comparisons were developed in the context of the available history, and many
candidate engines are evaluated on overlapping slices.

## Limitations and responsible interpretation

- **Exact-state sparsity:** gaps, space pairs, and triples create many cells
  with little or no support.
- **Fixed shrinkage:** prior strengths 12, 24, and 48 are design constants, not
  estimated uncertainty for each table.
- **Baseline dependence:** the number-specific baseline carries lifetime
  frequency differences into every conditional score.
- **Stale spacing context:** left and right spaces remain those of the number's
  last appearance, even across a long absence.
- **Association is not causation:** a nonzero corrected V does not show that a
  gap or spacing state causes the next occurrence.
- **Many comparisons:** seven tables across 49 numbers provide numerous chances
  for historical associations to appear.
- **Sequential dependence:** overlapping states and repeated candidate
  exposures violate a simple collection of independent fixed contingency
  tables.
- **Heuristic hierarchy:** averaging probabilities for priors and adding
  association-weighted logit effects is a fixed engineering rule, not a fitted
  coherent joint probability model.
- **Forced adjustment direction:** the magnitude of a logit difference is
  assigned the sign of the current residual, even when hierarchical priors make
  the raw direction more complicated.
- **Uncalibrated score:** values lie between 0 and 1 but have not been shown to
  match future empirical occurrence frequencies.
- **Tie-break influence:** current gap can decide equal-score ranks outside the
  categorical probability calculation.
- **Dataset and selection dependence:** retrospective performance depends on
  the exact history, chosen states, hierarchy, and evaluation slices.
- **No guaranteed predictability:** apparent conditional patterns may be random
  fluctuations and can disappear or reverse on untouched draws.

Use the model as an auditable conditional-frequency ranking, not as evidence
that a fair lottery's outcomes are causally controlled by past gaps or spaces.

## Implementation map

| Responsibility | Production location |
|---|---|
| Constants, evidence record, incremental cells, chi-square, residual, corrected V, and cell adjustment | `src/rand_ai/categorical_chi_square.py`, `ContingencyEvidence` and `ContingencyTable` |
| Per-number state, circular spaces, seven views, baseline, hierarchy, scoring, and details | `src/rand_ai/categorical_chi_square.py`, `CategoricalChiSquareModel` |
| Logit and sigmoid numerical guards | `src/rand_ai/categorical_chi_square.py`, `_logit` and `_sigmoid` |
| Causal learn-before-remember integration | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train` and `_StrategyState.remember` |
| Final ranking, Top-6, efficacy, and walk-forward orchestration | `src/rand_ai/strategy_prediction.py`, `_ranking_from_scores`, `_strategy`, `_StrategyState.build_strategies`, and `build_prediction_suites` |
| Desktop bridge serialization | `src/rand_ai/gui_bridge.py`, `_strategy_payload` and `_suite_payload` |
| Display name, default registration, family, details, and score rendering | `web/electron/main.cjs`, `web/src/lib/strategyFamilies.ts`, `web/src/components/SettingsDialog.vue`, and `web/src/views/CombinedPredictionGridView.vue` |
| Table statistics, edge cases, state capture, hierarchy, and negative-adjustment tests | `tests/test_categorical_chi_square.py` |
| Strategy ranking shape and serialized detail coverage | `tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py` |
| Concordance replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 scores, ranks, application gaps, detail
strings, Top-6 numbers, and standard completed efficacy. The strategy does not
attach separate experimental evidence metadata.
