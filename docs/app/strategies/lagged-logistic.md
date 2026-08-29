# Lagged Logistic

## Introduction

**Lagged Logistic** is the production strategy with identifier `lag_logistic`.
It is a default-disabled, opt-in member of the **Relationships & Machine
Learning** family. The engine trains one compact online logistic classifier from
each candidate's exact membership in the latest three draws, together with gap,
overdue, lifetime-frequency, and recent-frequency context.

The strategy produces a complete ranking of numbers 1 through 49. Its first six
entries form the Top-6 prediction used by prediction grids, audits,
effectiveness histories, comparisons, portfolios, exports, and selected
research consumers.

```{admonition} Compact candidate model, not a ticket probability model
:class: important

Lagged Logistic estimates 49 separate candidate scores with one shared linear
model. It does not model the joint constraint that exactly six distinct numbers
must be drawn, and its probabilities are not passed through a calibration
procedure. The Top-6 is a ranking of marginal model outputs, not a generated
joint six-number distribution.
```

## Scope

This page documents the exact production `lag_logistic` engine: its delayed
next-draw labels, 14-feature vector, three-draw lag convention, scikit-learn
configuration, one-epoch `partial_fit` lifecycle, 48-transition activation
gate, prior-smoothed fallback, probability scoring, causal safeguards, endpoint
state, and retrospective walk-forward behavior.

Lagged Logistic has no source-strategy dependencies. A separate Sparse Neural
Ticket research strategy can consume its first four lag features, but that
consumer does not alter the standalone calculation documented here.

## Prediction problem

Let {math}`D_t` be the six-number set observed at draw {math}`t`. A target draw
becomes 49 binary candidate examples. For
{math}`n\in\{1,\ldots,49\}`,

```{math}
y_{t,n}=
\begin{cases}
1, & n\in D_t,\\
0, & n\notin D_t.
\end{cases}
```

Every target supplies six positive and 43 negative labels. All candidates share
one coefficient vector, but each receives a feature vector from its own recent
membership, gap, and frequency history.

The logistic linear predictor is

```{math}
z(n)=w^\top x(n)+b,
```

with {math}`x(n)\in\mathbb R^{14}`. The fitted positive-class output is

```{math}
\widehat p_n=\sigma(z(n))
=\frac{1}{1+e^{-z(n)}}.
```

The application ranks all 49 values and returns the first six; it does not apply
49 independent classification thresholds.

## Historical candidate state

Let {math}`h` be the number of completed draws available when a feature matrix
is constructed. Production maintains:

- the latest at most 100 draws;
- lifetime appearance count {math}`A_n`;
- zero-based current gap {math}`g_n`;
- ordered occurrence indexes for each candidate.

The uniform per-number reference rate is

```{math}
p_0=\frac{6}{49}\approx0.122449.
```

Define

```{math}
\operatorname{clip}(v)=\max(-1,\min(1,v)).
```

The lifetime rate and residual are

```{math}
f_n=\frac{A_n}{\max(h,1)},
\qquad
F_n=\operatorname{clip}\left(\frac{f_n-p_0}{p_0}\right).
```

For recent window {math}`k\in\{5,20,100\}`, let {math}`C_{n,k}` count
appearances in the latest {math}`\min(h,k)` stored draws. When recent history
exists,

```{math}
F_{n,k}=\operatorname{clip}\left(
\frac{C_{n,k}/\min(h,k)-p_0}{p_0}
\right).
```

With no recent draw, production returns zero for the recent residuals. Positive
residuals mean above-reference historical frequency and negative residuals mean
below-reference frequency.

## Exact three-draw lag pattern

For candidate {math}`n`, production reads the latest three completed draws in
reverse chronological order:

```{math}
L_{n,1}=\mathbf 1[n\in D_h],
\qquad
L_{n,2}=\mathbf 1[n\in D_{h-1}],
\qquad
L_{n,3}=\mathbf 1[n\in D_{h-2}].
```

The application detail **Lag 1/2/3 pattern** displays these bits in exactly that
order. Lag 1 is always the newest completed draw, not the oldest.

If fewer than three draws exist, unavailable older positions are filled with
zero. The lag-hit share is always

```{math}
\bar L_n=\frac{L_{n,1}+L_{n,2}+L_{n,3}}{3}.
```

The denominator remains 3 during short history. Missing history is therefore
encoded identically to non-membership in that lag position. For example, after
one completed draw, a number present in it has pattern `1/0/0` and hit share
{math}`1/3`.

## Gap and overdue state

A number in the latest completed draw has current gap 0. If it has never
appeared, its gap is {math}`h`; otherwise the gap counts completed draws since
its most recent appearance, excluding that appearance draw.

If its occurrence indexes are {math}`o_1<\cdots<o_s`, the mean
inter-appearance distance is

```{math}
\bar g_n=
\begin{cases}
\dfrac{1}{s-1}\displaystyle\sum_{i=1}^{s-1}(o_{i+1}-o_i), & s\geq2,\\[8pt]
0, & s<2.
\end{cases}
```

The overdue residual is

```{math}
O_n=
\begin{cases}
0, & \bar g_n\leq0,\\
\operatorname{clip}((g_n-\bar g_n)/\bar g_n), & \bar g_n>0.
\end{cases}
```

Positive values indicate a current absence longer than the candidate's own
historical mean interval; negative values indicate a shorter interval.

## Exact 14-feature vector

The fixed production feature order is:

| Index | Feature | Exact value and range |
|---:|---|---|
| 1 | Lag 1 | {math}`L_{n,1}\in\{0,1\}` |
| 2 | Lag 2 | {math}`L_{n,2}\in\{0,1\}` |
| 3 | Lag 3 | {math}`L_{n,3}\in\{0,1\}` |
| 4 | Lag-hit share | {math}`\bar L_n\in\{0,1/3,2/3,1\}` |
| 5 | Clipped gap | {math}`\min(g_n/40,1)` |
| 6 | Gap-one indicator | {math}`\mathbf 1[g_n=1]` |
| 7 | Gap-two-to-four indicator | {math}`\mathbf 1[2\leq g_n\leq4]` |
| 8 | Long-gap indicator | {math}`\mathbf 1[g_n\geq12]` |
| 9 | Overdue ratio | {math}`O_n\in[-1,1]` |
| 10 | Lifetime frequency residual | {math}`F_n\in[-1,1]` |
| 11 | Recent-5 residual | {math}`F_{n,5}\in[-1,1]` |
| 12 | Recent-20 residual | {math}`F_{n,20}\in[-1,1]` |
| 13 | Recent-100 residual | {math}`F_{n,100}\in[-1,1]` |
| 14 | Recent trend | {math}`\operatorname{clip}(F_{n,5}-F_{n,20})` |

All features are bounded within {math}`[-1,1]`. There is no explicit bias
column because the estimator fits a separate intercept.

The vector contains no normalized number value, low/high indicator, parity,
prime status, calendar field, pair relationship, source-strategy rank, or raw
draw-shape feature. Two candidates with identical lag, gap, occurrence, and
recent-frequency histories receive identical feature rows and probabilities.

## Online logistic mathematics

For label {math}`y\in\{0,1\}` and logit {math}`z=w^\top x+b`, binary logistic
loss can be written as

```{math}
\ell(y,z)=\log(1+e^z)-yz.
```

Its unregularized coefficient gradient is

```{math}
\nabla_w\ell=(\sigma(z)-y)x.
```

Production configures scikit-learn's `SGDClassifier` as follows:

| Parameter | Production value |
|---|---|
| Library version in the lockfile | scikit-learn 1.9.0 |
| Loss | `log_loss` |
| Penalty | `l2` |
| Regularization strength | `alpha=0.001` |
| Learning-rate schedule | `optimal` |
| Intercept | enabled |
| Averaged SGD | enabled from the beginning |
| Shuffle | enabled |
| Random seed | `20260626` |

Conceptually, the estimator optimizes a regularized objective of the form

```{math}
\frac{\alpha}{2}\lVert w\rVert_2^2
+\text{binary log loss}.
```

For the optimal schedule, scikit-learn defines

```{math}
\eta_t=\frac{1}{\alpha(t_0+t)},
```

where {math}`t_0` is chosen by the library's Bottou heuristic. Production does
not specify a custom step size. The estimator defaults `eta0=0.01` and
`power_t=0.5`, but neither controls the optimal schedule.

No `sample_weight` or class-weight argument is supplied. The six positive and
43 negative examples therefore each have unit weight. Unlike the custom SVC,
TBL, and Scikit Online SVM implementations, Lagged Logistic does not balance
the two classes to equal total correction mass.

With `average=True`, the public coefficient vector and intercept are averaged
across online SGD updates rather than exposing only the final instantaneous
state. Shuffle order, L2 updates, optimal-schedule state, intercept handling,
and averaging follow the locked scikit-learn implementation.

## Delayed 49-row training batches

After every completed reference draw, production saves a feature matrix

```{math}
X_t\in\mathbb R^{49\times14}
```

for the next target. When that target draw occurs, its six positive and 43
negative labels are joined to the previously saved rows. The estimator receives
one `partial_fit` call, which scikit-learn defines as one SGD epoch over the
49-row batch.

The first call supplies the class list `[0, 1]`; later calls retain it. With
`shuffle=True`, the row sequence within each epoch is deterministically
shuffled from the fixed random-state sequence. There is no multi-target replay
buffer, convergence loop, early stopping, manual refit, or periodic reset.

The first observed draw has no earlier pending matrix, so it cannot train the
model. Draw 2 labels the matrix saved after draw 1 and becomes the first trained
transition. After 771 draws, 770 target batches have been processed.

## Activation gate and cold-start fallback

The estimator trains from the first available transition, but production does
not expose its `predict_proba` output until at least 48 target batches have been
processed. Before that threshold, candidate scores use a 24-draw neutral prior:

```{math}
C_n(h)=
\frac{A_n+24p_0}{h+24}.
```

This is a Beta-style posterior mean with total prior strength 24 and mean
{math}`6/49`, applied independently as an engineering fallback. Since all
candidates share the denominator and prior term, the fallback ranking is
exactly a lifetime-frequency ranking, with current gap and number resolving
count ties.

The 49 fallback scores sum to six:

```{math}
\sum_{n=1}^{49}C_n(h)=6.
```

The first 48 evaluable forecasts—targets 2 through 49—use this fallback. After
draw 49 is observed, **Trained transitions 48** is reached and the forecast for
draw 50 is the first one built from logistic probabilities.

The activation gate changes only which score is displayed and ranked. The
online classifier has already been learning every available transition during
the fallback period.

### First forecast

After the first draw, its six observed numbers each receive

```{math}
\frac{1+24(6/49)}{25},
```

while the other 43 receive

```{math}
\frac{24(6/49)}{25}.
```

The first Top-6 therefore repeats the first observed draw. This follows from
prior-smoothed lifetime frequency and is not a learned lag transition.

## Causal walk-forward lifecycle

For target draw {math}`D_t`, production advances as follows:

1. A 49×14 feature matrix was saved after {math}`D_{t-1}` using only history
   available at that time.
2. When {math}`D_t` occurs, build its label vector and train one `partial_fit`
   epoch on the saved matrix.
3. Remember {math}`D_t` by updating appearances, last-seen indexes, occurrence
   intervals, and the latest-100 draw history.
4. Build the new lag patterns, gaps, overdue values, and frequency residuals
   through {math}`D_t`.
5. Save those rows and rank either the prior fallback or fitted probabilities
   for target {math}`D_{t+1}`, according to the 48-transition gate.

The unknown target cannot enter its own lag pattern, frequency windows, gap,
training matrix, or probability. Tests replay identical prefixes with changed
future draws and verify that the earlier ranking remains unchanged. The fixed
scikit-learn seed makes repeated identical replays deterministic.

## Fitted probability and ranking

Once the activation gate passes, production takes the positive-class column
directly from

```{math}
\widehat p_n=operatorname{predict\_proba}(X)_{n,1}.
```

It does not min–max scale, temperature-adjust, renormalize, or force the 49
values to sum to six. The fallback score is also serialized directly. Thus a
score remains in {math}`[0,1]`, but its meaning changes at the activation
boundary from prior-smoothed lifetime frequency to the model's logistic output.

Candidates are ranked by:

1. larger current fallback or fitted score;
2. larger zero-based current gap when scores tie;
3. smaller number when score and gap both tie.

The gap tie-break is outside scikit-learn, although several gap features already
enter the model. The first six candidates form the Top-6.

## Interpreting the application fields

The prediction payload exposes the following fields. The first four rows are
the strategy-specific detail lines:

| Field | Meaning |
|---|---|
| Cold-start probability | Prior-smoothed lifetime score {math}`C_n(h)` while fewer than 48 transitions are trained. |
| Estimated hit probability | Positive-class `predict_proba` output after activation. It is not externally calibrated. |
| Lag 1/2/3 pattern | Membership in the newest, second-newest, and third-newest completed draws, with unavailable lags shown as zero. |
| Trained transitions | Number of completed next-draw batches processed by `partial_fit`, including batches learned while fallback scores were still displayed. |
| Compact shared model; no future-draw inputs | A lifecycle reminder, not a statistical diagnostic. |
| Gap | Zero-based current gap used in features and as the final exact-score tie-break. |
| Rank | Position in the complete 1–49 ordering. |
| Top-6 membership | Whether the candidate occupies one of the first six ranks. |

The tooltip does not expose the other ten feature values, coefficient vector,
intercept, or logit. The visible three-bit pattern alone is insufficient to
reproduce a score.

## Endpoint diagnostic

After all 771 repository draws, the model has processed 770 transition batches
and 37,730 labeled candidate rows. Scikit-learn reports
{math}`t_=37{,}731`, including its initial counter offset.

| Diagnostic | Endpoint value |
|---|---:|
| Trained transitions | 770 |
| Candidate examples processed | 37,730 |
| Averaged intercept | -0.069660 |
| Minimum next-forecast probability | 0.079508 |
| Maximum next-forecast probability | 0.200593 |
| Sum of 49 next-forecast probabilities | 5.949776 |
| Next Top-6 ranking | 6, 42, 45, 8, 24, 1 |

The endpoint averaged coefficients, ordered by absolute magnitude, are:

| Feature | Endpoint coefficient |
|---|---:|
| Gap two to four | -2.206654 |
| Lag 1 | -1.892700 |
| Clipped gap | -1.353138 |
| Gap one | -1.229848 |
| Recent-5 residual | 1.189826 |
| Overdue ratio | 0.895521 |
| Lag-hit share | -0.833578 |
| Long-gap indicator | -0.656741 |
| Lag 2 | -0.640028 |
| Recent trend | 0.379196 |
| Lifetime frequency residual | -0.197169 |
| Recent-20 residual | 0.129003 |
| Lag 3 | 0.031993 |
| Recent-100 residual | -0.015764 |

At this endpoint all six Top-6 candidates have lag pattern `0/0/0`. This does
not mean lag history is ignored: their other temporal features differ, and the
negative endpoint Lag-1 coefficient makes immediate repetition less favored in
this fitted state.

Coefficients are path-dependent diagnostics, not causal effects. Features are
correlated and differently distributed, averaging incorporates the entire SGD
trajectory, and a new draw changes both the model and candidate state.

## Top-6 efficacy reference

For a fixed six-number prediction evaluated against an independent uniformly
random six-number draw,

```{math}
H\sim\operatorname{Hypergeometric}(N=49,K=6,n=6),
```

with

```{math}
\mathbb E[H]=\frac{36}{49}=0.734694,
\qquad
\operatorname{Var}(H)=0.577572.
```

A leakage-free production replay over the repository's 771 chronological draws
produces 770 evaluable target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 588 | 0.763636 | 565.714 |
| Validation, target draws 121–520 | 400 | 307 | 0.767500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 188 | 0.752000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, records 189 hits
or 0.756000 per target. The two scoring regimes contribute:

| Regime | Targets | Total hits | Mean hits per target |
|---|---:|---:|---:|
| Prior fallback, targets 2–49 | 48 | 41 | 0.854167 |
| Logistic output, targets 50–771 | 722 | 547 | 0.757618 |

The short fallback interval is not an independent validation set, and its
higher average must not be interpreted as evidence that the fallback is
superior. The feature design, activation threshold, and strategy collection
were developed with historical context. These retrospective results do not
establish statistical significance, calibration, independent replication, or
future advantage.

## Core mathematical and statistical concepts

- **Lagged binary state:** three exact membership indicators encode immediate
  candidate recurrence.
- **Logistic regression:** a shared linear logit maps bounded candidate features
  to positive-class probabilities.
- **L2 regularization:** coefficient shrinkage limits unrestricted online
  separation.
- **Stochastic gradient descent:** each completed target contributes one
  shuffled 49-row epoch.
- **Averaged SGD:** published coefficients average the online trajectory.
- **Delayed supervision:** each target labels only the matrix saved before it
  occurred.
- **Prior smoothing:** a 24-draw neutral fallback stabilizes the first 48
  forecasts.
- **Multi-horizon frequency:** lifetime, recent-100, recent-20, and recent-5
  residuals provide overlapping time scales.
- **Hypergeometric overlap:** standard Top-6 efficacy uses the null overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Default-disabled status:** the compact learner remains an opt-in research
  strategy.
- **No class balancing:** 43 negative examples and six positive examples carry
  unit weight in every target batch.
- **Independent-candidate approximation:** logistic outputs do not enforce
  exactly six unique selected values or model candidate dependence.
- **No external calibration:** `predict_proba` is the logistic transformation of
  an online regularized score, not a validated next-draw probability.
- **Single epoch per target:** each `partial_fit` call advances the optimizer but
  does not converge the cumulative objective.
- **Activation discontinuity:** score semantics switch abruptly after 48
  transitions from lifetime fallback to logistic output.
- **Fallback is a hot-frequency ranker:** its common prior cannot change the
  ordering of lifetime appearance counts.
- **Missing-lag ambiguity:** an unavailable early lag and a known non-hit are
  both encoded as zero.
- **Overlapping temporal features:** lag bits, gap bands, overdue ratio, and
  recent residuals reuse the same observations.
- **No candidate identity fields:** candidates with identical histories are
  indistinguishable until the application tie-break.
- **Path and version dependence:** shuffling, optimal learning-rate state,
  averaging, and numerical behavior depend on the locked scikit-learn version
  and full update sequence.
- **Unnormalized probability sum:** the 49 outputs need not sum to six, and Top-6
  selection discards their joint scale.
- **Gap reuse:** current gap appears in several model features and again as the
  exact-score tie-break.
- **Dataset and multiple-comparison dependence:** historical lift can reflect
  chance, feature selection, activation choices, or comparison across many
  strategies.
- **No guaranteed predictability:** lagged historical behavior does not
  establish a stable mechanism for future lottery outcomes.

Use Lagged Logistic as a compact, reproducible, leakage-protected ranking
experiment—not as a calibrated ticket distribution or guarantee of future
efficacy.

## Implementation map

| Responsibility | Production location |
|---|---|
| Lag count, ordered feature names, 48-transition gate, and 24-draw prior | `src/rand_ai/strategy_prediction.py`, `_LAG_LOGISTIC_*` constants |
| `SGDClassifier` construction and pending model state | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Recent residual helper shared with Scikit Online SVM | `src/rand_ai/strategy_prediction.py`, `_sklearn_svm_recent_residual` |
| Exact lag ordering and zero padding | `src/rand_ai/strategy_prediction.py`, `_lag_logistic_pattern` |
| Exact 14-feature row and 49-row matrix | `src/rand_ai/strategy_prediction.py`, `_lag_logistic_features` and `_lag_logistic_feature_matrix` |
| Delayed one-epoch `partial_fit` | `src/rand_ai/strategy_prediction.py`, `_train_lag_logistic` |
| Pending row capture, activation gate, prior fallback, fitted probabilities, and details | `src/rand_ai/strategy_prediction.py`, `_lag_logistic_scores` |
| Appearance, gap, occurrence, and recent-history state updates | `src/rand_ai/strategy_prediction.py`, `_StrategyState.remember` |
| Final gap/number tie-break, Top-6 construction, efficacy, and causal orchestration | `src/rand_ai/strategy_prediction.py`, `_ranking_from_scores`, `_strategy`, and `build_prediction_suites` |
| Desktop serialization and default-disabled registration | `src/rand_ai/gui_bridge.py` and `web/electron/main.cjs` |
| Settings description, family, color, names, and detail rendering | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Feature shape and lag order, activation threshold, determinism, and prefix-invariance tests | `tests/test_strategy_prediction.py` |
| Default-disabled and serialization coverage | `tests/test_gui_bridge.py` |
| Shared research feature consumer and comparative benchmark harness | `scripts/benchmark_sparse_neural_ticket.py` and `tests/test_sparse_neural_ticket_benchmark.py` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 scores, ranks, current gaps, four detail
strings, Top-6 numbers, and standard completed efficacy. The complete feature
matrix, coefficient vector, intercept, and optimizer state remain internal.
