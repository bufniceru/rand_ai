# Support Vector Classifier

## Introduction

**Support Vector Classifier** is the production strategy with identifier
`svc` and short engine name **SVC**. It is a default-enabled member of the
**Relationships & Machine Learning** family. The strategy learns a single
linear scoring function from the draw sequence, evaluates every number from 1
through 49, and returns a complete ordered ranking. The first six entries form
its Top-6 prediction.

SVC supplies rankings to prediction grids, audits, effectiveness histories,
comparisons, portfolios, exports, and other selected application consumers. A
few ensemble engines can also consume its complete ranking. Those consumers do
not change the SVC calculation described here.

```{admonition} Distinct from Scikit Online SVM
:class: important

`svc` is Rand AI's custom online linear hinge-loss classifier. It is not the
separate `sklearn_svm` strategy, displayed as **Scikit Online SVM**. The two
strategies have different implementations and must not be treated as aliases.
```

## Scope

This page describes the production `svc` implementation: its candidate labels,
11 input features, online weight update, causal lifecycle, ranking rules,
displayed fields, and retrospective replay evidence. It does not describe the
internals of strategies that merely consume the SVC ranking.

The model is an exploratory ranking mechanism. It does not establish a causal
process in lottery drawings, produce calibrated probabilities, or guarantee
more future matches than random selection.

## Prediction problem

Let draw {math}`t` contain a six-number set {math}`D_t`. Training expands that
single draw into 49 binary candidate examples. For candidate number
{math}`n\in\{1,\ldots,49\}`, the label is

```{math}
y_{t,n}=
\begin{cases}
+1, & n\in D_t,\\
-1, & n\notin D_t.
\end{cases}
```

Each draw therefore contributes six positive examples and 43 negative
examples. All candidates share the historical state available before that
draw, but their number-specific properties, gaps, and occurrence counts can
differ.

The natural per-number base rate is

```{math}
p_0=\frac{6}{49}\approx0.122449.
```

The classifier produces a real-valued margin for all 49 candidates. It does
not independently accept or reject candidates; the application turns those
margins into one complete ranking and takes its first six numbers.

## Exact 11-feature vector

For candidate {math}`n`, the model constructs
{math}`x(n)\in\mathbb{R}^{11}`. Let {math}`h` be the number of remembered
draws, {math}`A_n` the lifetime appearances of {math}`n`, and
{math}`C_{n,k}` its count in the most recent {math}`\min(h,k)` draws. Define

```{math}
E_h=\max(hp_0,1), \qquad
E_{h,k}=\max(\min(h,k)p_0,1),
```

and the clipping function

```{math}
\operatorname{clip}_{[-1,1]}(z)=\max(-1,\min(1,z)).
```

The features, in their exact production order, are:

| Index | Feature | Formula and range | Interpretation |
|---:|---|---|---|
| 1 | Bias | {math}`1` | Allows a global intercept. |
| 2 | Normalized number | {math}`n/49\in[1/49,1]` | Encodes numeric position. |
| 3 | Low-number indicator | {math}`\mathbf{1}[n\leq24]\in\{0,1\}` | Separates 1–24 from 25–49. |
| 4 | Odd-number indicator | {math}`\mathbf{1}[n\bmod2=1]\in\{0,1\}` | Separates odd and even candidates. |
| 5 | Clipped gap | {math}`\min(g_n/40,1)\in[0,1]` | Smoothly increases with time since appearance, capped at 40. |
| 6 | Gap-one band | {math}`\mathbf{1}[g_n=1]` | Candidate appeared in the latest remembered draw. |
| 7 | Gap-two-to-four band | {math}`\mathbf{1}[2\leq g_n\leq4]` | Candidate appeared in the short intermediate gap band. |
| 8 | Long-gap band | {math}`\mathbf{1}[g_n\geq12]` | Candidate has a comparatively long absence. |
| 9 | Lifetime frequency residual | {math}`\operatorname{clip}_{[-1,1]}((E_h-A_n)/E_h)` | Positive when lifetime appearances are below their base-rate expectation. |
| 10 | Recent-8 residual | {math}`\operatorname{clip}_{[-1,1]}((E_{h,8}-C_{n,8})/E_{h,8})` | Compares the recent eight-draw count with its base-rate expectation. |
| 11 | Recent-24 residual | {math}`\operatorname{clip}_{[-1,1]}((E_{h,24}-C_{n,24})/E_{h,24})` | Compares the recent 24-draw count with its base-rate expectation. |

Here, {math}`\mathbf{1}[P]` equals 1 when condition {math}`P` is true and 0
otherwise.

### Gap and short-history behavior

The feature gap {math}`g_n` is one-based. A number present in the latest
remembered draw has gap 1. After {math}`h\geq1` remembered draws, a number that
has never appeared has gap {math}`h+1`. During the very first training pass,
before any draw has been remembered, the implementation's guarded history
length gives unseen candidates a gap of 2.

The expected-count denominators never fall below 1. This protects the lifetime
and recent residuals against division by zero and limits unstable amplification
in short histories. An unseen number has {math}`A_n=0` and recent counts of 0,
so its three residuals are positive and bounded by the clipping rule. Recent
windows use all available history when fewer than 8 or 24 draws exist.

The feature gap is not the same convention used by the final ranking
tie-break. The tie-break's current gap is zero-based: a number in the latest
reference draw has current gap 0. Keeping these conventions distinct is
necessary to reproduce production rankings.

## Online mathematics

### Linear margin and hinge condition

The model state is one weight vector {math}`w\in\mathbb{R}^{11}`. A candidate's
raw margin is

```{math}
m(n)=w^\top x(n).
```

For a labeled training example, the hinge loss is positive when
{math}`y\,m<1`. Production uses this strict condition; an example exactly on
the unit margin does not receive the classification correction.

Because every draw contains six positive and 43 negative examples, positive
examples receive the balancing weight

```{math}
c_{+}=\frac{43}{6},
```

while negative examples use {math}`c_{-}=1`. This balances the total nominal
positive and negative correction weight within a draw. It does not make the
classes equally likely and does not calibrate the resulting score.

### Learning rate and sequential update

Before training on a draw, let {math}`h` be the number of already remembered
draws. The learning rate is

```{math}
\eta_h=\frac{0.08}{\sqrt{h+1}}.
```

For each candidate example, production applies shrinkage coefficient 0.0008
and then, when the hinge condition is active, the class-weighted correction:

```{math}
w\leftarrow(1-0.0008\eta_h)w+
\begin{cases}
\eta_h c_y yx, & yw^\top x<1,\\
0, & \text{otherwise}.
\end{cases}
```

Candidates are processed sequentially in numeric order from 1 to 49. The
margin for a candidate is computed from the weight state reached after all
earlier candidates in that draw have been processed. Shrinkage also occurs for
examples outside the hinge boundary. Consequently, this is an online
hinge-loss approximation with order-dependent updates, not a batch
maximum-margin fit.

The decreasing learning rate makes later observations move the weights less
than early observations. Shrinkage continuously pulls every coordinate toward
zero, while the class weights prevent the 43 negative examples from dominating
the six positives solely through their count.

## Causal walk-forward lifecycle

The implementation enforces a train-then-remember-then-score sequence. If the
state before observing draw {math}`t` contains only draws through
{math}`t-1`, one walk-forward step is:

1. Build candidate features from history through {math}`t-1` and train on the
   now-observed labels from {math}`D_t`.
2. Remember {math}`D_t` by updating lifetime counts, last-seen positions,
   recent windows, and the completed-draw count.
3. Build features from the resulting state and score all 49 candidates for
   draw {math}`t+1`.
4. Compare that saved prediction with {math}`D_{t+1}` only after the next draw
   becomes observed.

Thus a target draw cannot train its own forecast. Appending later draws cannot
change rankings that were already produced for an earlier prefix, assuming the
same configuration and input prefix. This is the strategy's prefix-invariance
property and the central protection against target leakage.

All 11 weights start at zero. A hypothetical score before any training would
give every candidate the same margin. In the normal replay lifecycle, however,
the first observed draw trains the zero-weight model, that draw is remembered,
and the first forecast is then produced for the second draw.

## Scoring, normalization, and ranking

After training and remembering the reference history, SVC computes all 49 raw
margins. Let

```{math}
m_{\min}=\min_n m(n), \qquad m_{\max}=\max_n m(n).
```

When {math}`m_{\max}>m_{\min}`, the displayed score is the min–max transform

```{math}
s(n)=\frac{m(n)-m_{\min}}{m_{\max}-m_{\min}}.
```

This maps the lowest margin in that draw to 0 and the highest to 1. If all 49
margins are equal, production assigns every score the fallback value 0.

The complete ranking sorts candidates by:

1. larger normalized score;
2. larger zero-based current gap when scores tie;
3. smaller number when both score and gap tie.

Because min–max scaling is monotone whenever the margins are not all equal,
sorting normalized scores is equivalent to sorting raw margins. The first six
ranked numbers are the SVC Top-6.

## Interpreting the application fields

The prediction details expose the information needed to interpret a candidate:

| Field | Meaning |
|---|---|
| Score | The normalized {math}`s(n)`, displayed as a percentage. It is a relative within-draw score, not the probability that the number will be drawn. |
| Margin | The raw linear value {math}`w^\top x(n)`, displayed to three decimal places. |
| Recent 8 | The candidate's observed count in the available portion of the most recent eight draws. |
| Recent 24 | The corresponding count over at most 24 recent draws. |
| Rank | The candidate's position in the complete 1–49 ordering after tie-breaking. |
| Top-6 membership | Whether the rank is one of the first six positions. |

A higher score means only that the current fitted model ranks the candidate
more strongly relative to the other 48 candidates for the same target. The
percentage is not calibrated against observed event frequencies. Raw margins
can reveal separation that min–max scaling hides, but their magnitude is also
state-dependent. Neither normalized scores nor raw margins should be compared
as if they had a fixed probabilistic meaning across different draws.

## Statistical baseline

For a six-number prediction evaluated against a uniformly random six-number
draw from 49, the number of hits {math}`H` follows

```{math}
H\sim\operatorname{Hypergeometric}(N=49,K=6,n=6).
```

Its mean is

```{math}
\mathbb{E}[H]=n\frac{K}{N}=\frac{36}{49}
\approx0.734694,
```

and its variance is

```{math}
\operatorname{Var}(H)
=n\frac{K}{N}\left(1-\frac{K}{N}\right)\frac{N-n}{N-1}
=6\frac{6}{49}\frac{43}{49}\frac{43}{48}
\approx0.577572.
```

This null model describes the overlap of two six-element sets when the actual
draw is uniform and independent of the prediction. It is a reference point,
not by itself a hypothesis test for an adaptively developed strategy.

## Leakage-free replay evidence

The fixed 771-draw benchmark report replays the strategy causally, producing
770 target predictions. Its recorded SVC results are:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 625 | 0.811688 | 565.714 |
| Validation, target draws 121–520 | 400 | 335 | 0.837500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 187 | 0.748000 | 183.673 |

These figures are retrospective regression evidence from the established
benchmark dataset and slices. They confirm reproducibility of the recorded
implementation under its causal replay protocol. They are not proof of a
future advantage: the application contains multiple strategies, strategy and
ensemble choices were informed by historical comparisons, and the draws in one
sequence are not independent development experiments. No statistical
significance or guaranteed predictive benefit is claimed here.

## Limitations and responsible interpretation

- **Retrospective selection:** design decisions and later comparisons were made
  with knowledge of historical behavior, which can favor the observed dataset.
- **Multiple-strategy comparison:** inspecting many engines or variants raises
  the chance that an apparently strong historical result occurs by selection.
- **Dataset dependence:** learned weights and measured efficacy depend on the
  exact draw sequence, preprocessing, slice boundaries, and replay setup.
- **Class imbalance:** each draw supplies only six positives against 43
  negatives. The {math}`43/6` weight balances update mass but cannot remove all
  consequences of imbalance.
- **Feature bias:** low/high, parity, gaps, and frequency residuals encode
  chosen structural assumptions. They do not demonstrate that the drawing
  mechanism uses those structures.
- **Sequential order:** candidates update the model from 1 through 49, so the
  online approximation is order-dependent.
- **Uncalibrated output:** the linear margin and displayed percentage are
  ranking signals, not event probabilities.
- **Min–max information loss:** normalization preserves order but suppresses
  absolute scale and forces each nonconstant draw to span the full 0–1 range.
- **No guaranteed predictability:** a historical excess over the random mean
  may shrink, disappear, or reverse on untouched future draws.

Use efficacy history as an audit of what the application did, not as assurance
of what the next draw will do.

## Implementation map

The following production locations are the source of truth for this page. The
map identifies responsibilities without reproducing the source code wholesale.

| Responsibility | Production location |
|---|---|
| Strategy state, including the 11 weights, counts, gaps, and recent history | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Feature construction and short-history guards | `src/rand_ai/strategy_prediction.py`, `_svc_features` |
| Sequential hinge update and learning-rate schedule | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train` path for `svc` |
| Raw margins, min–max scaling, full ranking, and Top-6 creation | `src/rand_ai/strategy_prediction.py`, `_StrategyState.build_strategies`, `_scale_scores`, `_ranking_from_scores`, and `_strategy` |
| Causal walk-forward orchestration | `src/rand_ai/strategy_prediction.py`, `build_prediction_suites` |
| Desktop bridge serialization of ranks, scores, gaps, details, and Top-6 | `src/rand_ai/gui_bridge.py`, `_strategy_payload` and `_suite_payload` |
| Display name, default registration, family, ordering, and detail rendering | `web/electron/main.cjs`, `web/src/lib/strategyFamilies.ts`, `web/src/components/SettingsDialog.vue`, and `web/src/views/CombinedPredictionGridView.vue` |
| Fixed replay figures and reproduction command | `reports/svc_recurrence_hybrid_benchmark.md` and `scripts/benchmark_svc_recurrence_hybrid.py` |
| Ranking shape, Top-6, efficacy, and causal dependency regression coverage | `tests/test_strategy_prediction.py`, including `test_builds_thirty_six_named_rankings_and_reports_progress` and the SVC-hybrid prefix-invariance tests |
| Serialized prediction payload coverage | `tests/test_gui_bridge.py` |

The serializer exposes the complete ranking, scores, Top-6 numbers, and detail
text to the desktop interface. UI formatting does not retrain the model or
alter its ordering.
