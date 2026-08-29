# Recurrence Dynamics

## Introduction

**Recurrence Dynamics** is the production strategy with identifier
`recurrence_dynamics`. It is a default-disabled, opt-in member of the
**Shape & Similarity** family. The strategy represents the latest three draws
as one geometric state, finds similar states earlier in the observed sequence,
and scores each number from the draws that followed those historical states.

The strategy returns a complete ranking of numbers 1 through 49. Its first six
numbers form the Top-6 prediction used by prediction grids, audits,
effectiveness histories, comparisons, portfolios, exports, and selected
ensemble consumers.

The strategy selector identifies the engine as **Recurrence Dynamics
(Experimental)**. Active-result views use the shorter **Recurrence Dynamics**
label. The experimental status remains important: recurrence in recorded data
does not show that a lottery is chaotic, deterministic, or predictably
different from its stated random mechanism.

## Scope

This page documents the production V2 forecast, including its six-value draw
representation, 18-value delay embedding, causal analogue search,
inverse-distance weights, prior-smoothed number scores, evidence metadata, and
application ranking rules. It also explains how this forecast differs from the
associated **Nonlinear Dynamics** diagnostic report.

It does not describe the formulas of ensemble strategies that consume the
Recurrence Dynamics ranking. Those consumers do not alter the source ranking
described here.

## Prediction problem

Let the chronological observed history be
{math}`D_1,D_2,\ldots,D_h`, where every {math}`D_t` is a set of six unique
numbers from 1 through 49. The task is to assign a score to every candidate
{math}`n` for the unobserved draw {math}`D_{h+1}`.

Recurrence Dynamics uses a local analogue assumption:

> If an earlier three-draw state is close to the latest three-draw state, the
> draw that followed the earlier state may provide a useful local reference for
> the next draw.

This is a similarity rule, not a fitted global classifier. It does not assign
binary training labels to all 49 candidates. Instead, a candidate receives
weighted evidence whenever it appeared in the observed successor of a selected
historical analogue.

The uniform per-number reference rate is

```{math}
p_0=\frac{6}{49}\approx0.122449.
```

## Production forecast representation

### Six-value draw vector

Write the sorted values in draw {math}`D_t` as

```{math}
n_{t,1}<n_{t,2}<\cdots<n_{t,6}.
```

The V2 forecast represents that draw by

```{math}
q_t=\frac{1}{\sqrt{6}}
\left(
\frac{n_{t,1}-1}{48},
\frac{n_{t,2}-1}{48},
\ldots,
\frac{n_{t,6}-1}{48}
\right)\in\mathbb{R}^{6}.
```

Subtracting 1 and dividing by 48 maps every sorted position to the interval
{math}`[0,1]`. Dividing the entire block by {math}`\sqrt{6}` prevents its
Euclidean magnitude from growing merely because the block has six fields. The
representation is independent of the input order because each draw is sorted
and validated first.

The production forecast deliberately does not append explicit parity,
frequency, gap, spacing, or overlap fields. Those features are either absent or
belong to the separate diagnostic representation described later.

### Three-draw delay embedding

Once at least three draws are available, the state ending at draw {math}`t` is
the 18-value vector

```{math}
z_t=
\left[
0.50q_{t-2},\;
0.75q_{t-1},\;
1.00q_t
\right]\in\mathbb{R}^{18}.
```

The fixed lag weights emphasize the newest draw while retaining the preceding
two draws. The embedding dimension is 3; it is not estimated from the dataset.
For {math}`h` observed draws, valid embedding end indexes run from the third
draw through draw {math}`h`.

## Causal analogue selection

Let {math}`c` be the end index of the latest embedding. A historical embedding
ending at index {math}`j` is eligible only when

```{math}
j\leq c-3.
```

The exact inequality excludes embeddings ending at {math}`c`, {math}`c-1`, and
{math}`c-2`; an embedding ending at {math}`c-3` is eligible. Its successor draw
{math}`D_{j+1}` is already observed. This temporal exclusion prevents the
current state from selecting itself and ensures that every analogue has a
known successor.

For every eligible state, the model computes Euclidean distance

```{math}
d_j=\lVert z_j-z_c\rVert_2.
```

It uses a stable ascending distance sort and retains at most the nearest eight
states. Stable sorting makes the earlier candidate order decisive when two
floating-point distances are equal. If {math}`k\leq8` analogues are retained,
their raw and normalized weights are

```{math}
r_i=\frac{1}{d_i+10^{-9}}, \qquad
w_i=r_i\frac{k}{\sum_{\ell=1}^{k}r_\ell}.
```

The {math}`10^{-9}` term keeps an exact zero-distance match finite. The
normalization makes

```{math}
\sum_{i=1}^{k}w_i=k,
```

so the selected analogues contribute the same total evidence as {math}`k`
equally weighted draws while closer states receive a larger share.

### Effective neighbor count

Weight concentration is summarized by the Kish-style effective count

```{math}
N_{\mathrm{eff}}=
\frac{\left(\sum_iw_i\right)^2}{\sum_iw_i^2}.
```

It approaches {math}`k` when the weights are similar and approaches 1 when one
analogue dominates. A list may therefore contain eight analogues but carry much
less than eight effectively balanced neighbors.

## Prior-smoothed successor score

For number {math}`n`, its weighted successor count is

```{math}
a_n=\sum_{i=1}^{k}w_i\mathbf{1}[n\in D_{j_i+1}],
```

where {math}`j_i` is the end index of selected analogue {math}`i`. The
production score uses a fixed prior strength of eight draws at the uniform
{math}`6/49` rate:

```{math}
s(n)=\frac{a_n+8p_0}{\sum_iw_i+8}
=\frac{a_n+8(6/49)}{k+8}.
```

The prior reduces extreme estimates when few analogues exist. Because every
successor contains six numbers and the normalized analogue weights sum to
{math}`k`, the 49 scores satisfy

```{math}
\sum_{n=1}^{49}s(n)=6.
```

This makes each score interpretable as a prior-smoothed local occurrence rate,
but not as a calibrated probability. The analogue rule, feature selection, and
reuse of historical draws do not constitute a validated probabilistic
generative model.

## Cold start and fallback

The model returns a neutral prediction when it cannot build a current
three-draw embedding or when no historical embedding passes the temporal
exclusion. Every number then receives exactly

```{math}
s(n)=p_0=\frac{6}{49}.
```

The per-number details state **Neutral 6/49 prior** and **Insufficient embedded
history**. At least six observed draws are needed for the first non-neutral
forecast: three draws form the current embedding, and the exact exclusion must
leave at least one earlier embedding with an observed successor.

Equal neutral scores still pass through the application's normal tie-break.
They are therefore not interpreted as a random selection.

## Causal walk-forward lifecycle

The production integration follows an evaluate-then-observe-then-predict
sequence. When draw {math}`D_t` becomes known:

1. Evaluate the pending Top-6 forecast made after {math}`D_{t-1}` against
   {math}`D_t`, then clear that pending forecast.
2. Append the sorted observed draw {math}`D_t` to the recurrence history.
3. Construct the latest embedding using only draws through {math}`D_t`.
4. Find eligible historical analogues and score the candidates for
   {math}`D_{t+1}`.
5. Apply the application's final tie-break and store that exact Top-6 for
   evaluation only when {math}`D_{t+1}` is later observed.

The target draw is never part of its own feature history, analogue search, or
score calculation. Extending the dataset cannot alter a forecast already made
for an earlier prefix. This prefix invariance is the strategy's principal
target-leakage protection.

The stored efficacy Top-6 uses the final application ranking rather than an
internal score-only ordering. This keeps the evidence history aligned with the
numbers actually displayed to the user.

## Ranking and tie-breaking

Unlike SVC, Recurrence Dynamics does not min–max scale its 49 scores. The
prior-smoothed values {math}`s(n)` pass directly into the common strategy
ranking contract. Candidates are ordered by:

1. larger recurrence score;
2. larger zero-based current gap when scores tie;
3. smaller number when both score and current gap tie.

The current gap used here is an application-level ranking field: a number in
the latest reference draw has gap 0. The first six ranked candidates form the
Top-6.

## Forecast evidence metadata

### Completed forecast history

The model records the number of Top-6 hits only after the corresponding target
draw occurs. The completed history includes forecasts made during the neutral
cold-start period. If the hit counts are {math}`H_1,\ldots,H_N`, it reports

```{math}
\bar H=\frac{1}{N}\sum_{i=1}^{N}H_i
```

and, when {math}`N\geq2`, the lower-bound statistic

```{math}
L=\bar H-1.96\frac{s_H}{\sqrt{N}},
```

where {math}`s_H` is the sample standard deviation of the completed hit counts.
For fewer than two observations the implementation uses {math}`L=0`.

The status gates are fixed:

| Status | Production condition |
|---|---|
| Insufficient | Fewer than 100 completed forecasts, or fewer than eight analogues for the current forecast. |
| Weak | The gates are met, but {math}`\bar H\leq36/49`. |
| Suggestive | {math}`\bar H>36/49`, but {math}`L\leq36/49`. |
| Supported | {math}`L>36/49`. |

The word “supported” means only that this fixed descriptive forecast gate was
cleared. The bound does not correct for strategy selection, parameter
exploration, repeated use of one time series, or multiple comparisons.

### Distance support index

For every non-neutral forecast, the nearest distance is compared with all
previous non-neutral nearest distances plus the current one. Its causal
percentile is

```{math}
P_d=\frac{
\#\{d\text{ in the accumulated nearest-distance history}:d\leq d_{\min}\}
}{
\text{number of accumulated distances}
}.
```

Small values mean the current nearest state is unusually close relative to
earlier forecasts. The displayed evidence index is

```{math}
I=\min\left(1,\frac{N_{\mathrm{eff}}}{8}\right)(1-P_d).
```

The index combines neighbor balance with relative closeness. It is bounded by
0 and 1, but it is not a probability of a correct prediction or a statistical
significance level.

## Interpreting the application fields

| Field | Meaning |
|---|---|
| Score / Posterior occurrence | The prior-smoothed analogue successor score {math}`s(n)`, displayed as a percentage. It is a local relative estimate, not a calibrated lottery probability. |
| Causal V2 analogues | The current number of eligible neighbors retained, from 1 through 8 for a non-neutral forecast. |
| Effective neighbors | {math}`N_{\mathrm{eff}}`, showing whether the inverse-distance weights are balanced or concentrated. |
| Nearest-distance percentile | {math}`P_d`, calculated causally from nearest distances observed up to the current forecast. Lower means an unusually close analogue. |
| Evidence | The forecast-only status: insufficient, weak, suggestive, or supported. |
| Rank | Position in the complete 1–49 ordering after gap and number tie-breaking. |
| Top-6 membership | Whether the candidate occupies one of the first six positions. |

The strategy-level evidence panel also exposes the evidence index, evaluated
forecast count, average hits per draw, analogue count, effective neighbors, and
distance percentile. Candidate scores and the evidence index answer different
questions: the score ranks a number, while the evidence index describes the
current analogue neighborhood as a whole.

## Relation to the Nonlinear Dynamics report

The strategy and diagnostic report share recurrence concepts but use different
representations and evidence rules.

### Diagnostic 20-value draw vector

The report represents each draw with 20 values:

- six normalized sorted values, divided by {math}`\sqrt6` as in the forecast;
- six circular empty-space counts divided by {math}`43\sqrt6`;
- normalized sum, span, odd share, low-number share, prime share, and
  consecutive-pair share, with the six-value block divided by {math}`\sqrt6`;
- overlap share with each of the preceding two draws, with the two-value block
  divided by {math}`\sqrt2`.

For sorted values {math}`n_1<\cdots<n_6`, the circular empty spaces are

```{math}
\left(
(n_1-1)+(49-n_6),
n_2-n_1-1,
\ldots,
n_6-n_5-1
\right),
```

which sum to 43. The six shape fields before block scaling are

```{math}
\left(
\frac{\sum_i n_i-21}{258},
\frac{n_6-n_1}{48},
\frac{\#\mathrm{odd}}6,
\frac{\#\{n_i\leq24\}}6,
\frac{\#\mathrm{prime}}6,
\frac{\#\mathrm{consecutive\ pairs}}5
\right).
```

Three such vectors form a 60-value diagnostic embedding with the same
0.50/0.75/1.00 lag weights. These 60-value states are not used to rank numbers
in the production V2 strategy.

### Recurrence quantification and surrogates

The report analyzes at most the latest 750 draws. It computes pairwise Euclidean
distances between diagnostic embeddings, excludes pairs whose index separation
is at most 3, and sets the recurrence threshold to the 10th percentile of the
remaining distances. From the resulting Boolean recurrence matrix it reports
recurrence rate, determinism, mean and maximum diagonal length, laminarity, and
trapping time. Its displayed plot contains at most the latest 250 embedded
states.

Observed determinism is compared with 99 draw-order-shuffled surrogates from the
fixed random seed `20260829`. The empirical upper-tail value is

```{math}
p_{\mathrm{surr}}=
\frac{1+\#\{\mathrm{surrogate\ determinism}\geq
\mathrm{observed\ determinism}\}}{1+99}.
```

The report verdict requires both forecast evidence and
{math}`p_{\mathrm{surr}}\leq0.05`. The prediction strategy's evidence metadata
deliberately omits this surrogate requirement and describes only causal
forecast performance. A strong strategy label and a strong dynamical claim are
therefore not the same thing.

## Statistical baseline

For any six-number Top-6 evaluated against an independent uniform six-number
draw, the hit count follows

```{math}
H\sim\operatorname{Hypergeometric}(N=49,K=6,n=6).
```

Its mean and variance are

```{math}
\mathbb{E}[H]=\frac{36}{49}\approx0.734694,
```

```{math}
\operatorname{Var}(H)=
6\frac{6}{49}\frac{43}{49}\frac{43}{48}
\approx0.577572.
```

This is the theoretical overlap baseline used by the evidence gates and
benchmark reports. It does not account for model selection or establish that a
historical deviation is statistically significant.

## Leakage-free replay evidence

The fixed V2 benchmark uses 771 chronological draws and produces 770 evaluable
target forecasts. The production gap-and-number tie-break is included.

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 625 | 0.811688 | 565.714 |
| Validation, target draws 121–520 | 400 | 342 | 0.855000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 199 | 0.796000 | 183.673 |

The benchmark also records 197 hits, or 0.788000 per target, over its latest
250-target comparison slice. The older V1 recurrence rule is not the production
strategy; its historical full-replay result was 558 hits, or 0.724675 per
target.

Over the report's latest 750-draw diagnostic window, the recorded V2 forecast
mean is 0.825100 and its implemented lower-bound statistic is 0.767367. The
combined Nonlinear Dynamics verdict nevertheless remains **weak** because the
order-shuffled surrogate determinism test records {math}`p=0.96`. This is a
useful reminder that forecast replay performance and evidence for nonlinear
recurrence are different claims.

V2 was designed after exploratory analysis of the included dataset. The fixed
slices and results are regression evidence, not independent confirmation. No
statistical significance, future advantage, or predictable lottery mechanism
is claimed.

## Limitations and responsible interpretation

- **Exploratory selection:** the representation, three-draw embedding, eight
  neighbors, lag weights, prior strength, and evidence gates were chosen after
  exploratory work on the available history.
- **Dataset dependence:** nearest states, distance percentiles, and measured
  efficacy depend on the exact chronological dataset and preprocessing.
- **Distance concentration:** an 18-dimensional Euclidean neighborhood may be
  unstable or weakly discriminative in finite history.
- **Correlated analogues:** overlapping delay embeddings and successor draws are
  not independent experimental samples.
- **Near-zero dominance:** inverse-distance weighting can concentrate most
  evidence on one almost-identical state despite the finite epsilon.
- **Representation bias:** sorted values retain positional geometry but omit
  explicit frequency, gap, parity, and other potentially relevant or
  irrelevant summaries by design.
- **Fixed temporal rule:** the exact three-index exclusion is a design choice,
  not an inferred property of the drawing process.
- **Prior interpretation:** the eight-draw uniform prior stabilizes scores but
  is not proof that the resulting percentages are calibrated posteriors.
- **Approximate lower bound:** the evidence gate uses a normal-style 1.96
  standard-error calculation despite sequential dependence and adaptive model
  development.
- **Multiple comparisons:** strategy comparisons and diagnostic metrics create
  selection opportunities not represented by the displayed evidence status.
- **Recurrence is not chaos:** repeated or nearby historical states do not
  establish a low-dimensional deterministic lottery system.
- **No guaranteed predictability:** apparent historical lift may shrink,
  disappear, or reverse on untouched future draws.

Use the strategy as an auditable experimental ranking and its evidence as a
description of completed replay history, not as assurance about the next draw.

## Implementation map

| Responsibility | Production location |
|---|---|
| Forecast constants, draw validation, six-value features, and 18-value embeddings | `src/rand_ai/nonlinear_dynamics.py`, constants, `_validated_draw`, `forecast_features`, and `forecast_delay_embeddings` |
| Causal history, analogue selection, weighting, score construction, pending forecast, and evidence | `src/rand_ai/nonlinear_dynamics.py`, `RecurrenceDynamicsModel` |
| Effective-neighbor weights and evidence gates | `src/rand_ai/nonlinear_dynamics.py`, `_inverse_distance_weights`, `_lower_confidence_bound`, and `classify_forecast_evidence` |
| 20-value diagnostic features, 60-value embeddings, recurrence matrix, RQA, and surrogate analysis | `src/rand_ai/nonlinear_dynamics.py`, `draw_features`, `delay_embeddings`, `_recurrence_matrix`, `_rqa_metrics`, and `nonlinear_dynamics_analysis` |
| Strategy lifecycle, final gap/number tie-break, and source evidence conversion | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train`, `_StrategyState.remember`, `_StrategyState.build_strategies`, `_ranking_from_scores`, `_recurrence_evidence`, and `build_prediction_suites` |
| Desktop bridge serialization | `src/rand_ai/gui_bridge.py`, `_strategy_payload` and `_suite_payload` |
| Selector registration, active display name, family, detail fields, and evidence panel | `web/electron/main.cjs`, `web/src/lib/strategyFamilies.ts`, `web/src/components/SettingsDialog.vue`, and `web/src/views/CombinedPredictionGridView.vue` |
| Fixed V1/V2 replay record | `reports/recurrence_dynamics_benchmark.md` |
| Feature, embedding, weighting, evidence, cold-start, causal, diagnostic, and prefix-invariance tests | `tests/test_nonlinear_dynamics.py` |
| Strategy selection and serialized evidence tests | `tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py` |

The serializer carries all 49 scores, ranks, gaps, per-number detail strings,
Top-6 numbers, standard efficacy, and recurrence-specific evidence to the
desktop interface. Rendering those fields does not change the model state or
ranking.
