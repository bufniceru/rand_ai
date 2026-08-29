# Scikit Online SVM

## Introduction

**Scikit Online SVM** is the production strategy with identifier
`sklearn_svm`. It is a default-disabled, opt-in member of the
**Relationships & Machine Learning** family. The engine uses scikit-learn's
incremental `SGDClassifier` to learn a linear support-vector ranking from 32
candidate features combining temporal behavior, frequency, latest-draw
relationships, six source rankings, and leakage-safe source efficacy.

The strategy produces a complete ranking of numbers 1 through 49. Its first six
entries form the Top-6 prediction used by prediction grids, audits,
effectiveness histories, comparisons, portfolios, exports, and selected
research consumers.

```{admonition} Distinct from Support Vector Classifier
:class: important

`sklearn_svm` is the scikit-learn model documented here. It is not the custom
`svc` engine displayed as **Support Vector Classifier**. Scikit Online SVM uses
a 32-feature expert-stacking schema, delayed 49-row `partial_fit` batches,
scikit-learn's optimal learning-rate schedule, an explicit intercept, and
averaged SGD coefficients. The custom SVC uses a different 11-feature vector
and handwritten sequential update.
```

## Scope

This page documents the exact production `sklearn_svm` strategy: its hidden
dependencies, binary candidate labels, 32-feature vector, source-quality
tracker, delayed training batch, scikit-learn configuration, cold-start
consensus, fitted margins, score transformation, causal safeguards,
interpretation, and retrospective walk-forward behavior.

It does not reproduce the internal algorithms of the six source strategies.
Their complete rankings and completed Top-6 hit histories are inputs to this
model, but each source remains an independent strategy.

## Prediction problem

Let {math}`D_t` be the six-number set at draw {math}`t`. One target draw becomes
49 binary examples. For candidate {math}`n\in\{1,\ldots,49\}`,

```{math}
y_{t,n}=
\begin{cases}
+1, & n\in D_t,\\
-1, & n\notin D_t.
\end{cases}
```

The application supplies scikit-learn with class labels 1 and 0; the binary
linear SVM internally uses the corresponding positive and negative sides of
the decision boundary. Every batch therefore contains six positive examples
and 43 negative examples.

The fitted decision function is

```{math}
f(n)=w^\top x(n)+b,
```

where {math}`x(n)\in\mathbb R^{32}`. Scikit Online SVM does not produce
independent yes/no decisions for the interface. The 49 margins become one
complete ordering and the first six ranks form the prediction.

## Hidden source dependencies

Scikit Online SVM directly depends on six ranking experts, in this fixed order:

1. **Markov Spaces** (`mksp`);
2. **Doublet & Triplet Markov** (`doublet_triplet_markov`);
3. **Bayesian** (`bayesian`);
4. **Temporal Behavior Learning** (`tbl`);
5. **Markov Normalized Positions** (`mknp`);
6. **Earth Mover Distance** (`emd`).

TBL recursively activates Freshness, Proximity, and the deterministic Random
baseline. Requesting only `sklearn_svm` calculates this dependency graph but
serializes only Scikit Online SVM as the requested output.

For source {math}`j`, let {math}`r_j(n)` be candidate {math}`n`'s complete rank.
Production converts it to rank strength

```{math}
q_j(n)=\frac{49-r_j(n)}{48}.
```

Rank 1 maps to 1 and rank 49 maps to 0. Source raw scores are not passed to the
SVM; only complete ranks, Top-6 membership, top-quarter membership, and
completed source efficacy enter its features.

## Leakage-safe source quality

For expert {math}`j`, production records only predictions whose target draw has
already occurred:

- cumulative Top-6 hits {math}`T_j` over {math}`N_j` evaluated draws;
- a deque of the latest at most 40 completed hit counts
  {math}`H_{j,1},\ldots,H_{j,m_j}`.

The theoretical random Top-6 mean is

```{math}
\mu_0=\frac{36}{49}\approx0.734694.
```

Both long-term and recent qualities use a neutral 24-draw prior:

```{math}
L_j=\frac{T_j+24\mu_0}{N_j+24},
\qquad
R_j=\frac{\sum_{k=1}^{m_j}H_{j,k}+24\mu_0}{m_j+24}.
```

Confidence in the completed long-term history is

```{math}
c_j=\frac{N_j}{N_j+24}.
```

Production blends the two qualities as

```{math}
B_j=L_j(1-0.70c_j)+R_j(0.70c_j).
```

The resulting efficacy factor is

```{math}
\omega_j=
\operatorname{clip}_{[0.5,1.5]}
\left(\frac{B_j}{\mu_0}\right).
```

Before any completed evaluation, {math}`L_j=R_j=\mu_0`,
{math}`c_j=0`, and every expert factor is exactly 1. As evidence grows, the
formula approaches a 30% long-term and 70% recent blend. The recent component
still carries the fixed neutral prior, and the final factor cannot fall below
0.5 or rise above 1.5.

The factor is a feature input, not a direct guarantee that the SVM follows the
best recent expert. Once fitted, learned coefficients can reinforce, ignore, or
oppose both source strength and its efficacy interaction.

## Historical candidate state

Let {math}`h` be the number of completed draws in the feature state. For number
{math}`n`, production maintains lifetime appearances {math}`A_n`, zero-based
current gap {math}`g_n`, ordered occurrence indexes, at most 100 recent draws,
pair co-occurrence counts, and the latest completed draw.

The uniform per-number reference rate is

```{math}
p_0=\frac{6}{49}.
```

For any value {math}`z`, define

```{math}
\operatorname{clip}(z)=\max(-1,\min(1,z)).
```

The lifetime frequency residual is

```{math}
F_n=
\operatorname{clip}\left(
\frac{A_n/\max(h,1)-p_0}{p_0}
\right).
```

For a recent window {math}`k\in\{5,20,100\}`, let {math}`C_{n,k}` be the
number of appearances in the available latest {math}`\min(h,k)` draws. When at
least one recent draw exists,

```{math}
F_{n,k}=
\operatorname{clip}\left(
\frac{C_{n,k}/\min(h,k)-p_0}{p_0}
\right).
```

With no recent history, production returns 0 for these recent residuals. A
positive residual means above-reference frequency; a negative residual means
below-reference frequency.

### Gap and overdue state

A number in the latest completed draw has current gap 0. A never-seen candidate
has gap {math}`h`. If occurrence indexes are
{math}`o_1<\cdots<o_s`, its mean inter-appearance distance is

```{math}
\bar g_n=
\begin{cases}
\dfrac{1}{s-1}\displaystyle\sum_{i=1}^{s-1}(o_{i+1}-o_i), & s\geq2,\\[8pt]
0, & s<2.
\end{cases}
```

The overdue feature is

```{math}
O_n=
\begin{cases}
0, & \bar g_n\leq0,\\
\operatorname{clip}((g_n-\bar g_n)/\bar g_n), & \bar g_n>0.
\end{cases}
```

### Latest-draw relationship residual

Let {math}`L` be the latest completed draw and {math}`P(a,n)` the lifetime count
of draws containing both distinct numbers {math}`a` and {math}`n`. For each
{math}`a\in L` with {math}`a\neq n`, production forms

```{math}
Q(a,n)=\frac{P(a,n)}{\max(A_a,1)}.
```

It averages the available conditional rates and compares them with the
{math}`6/49` reference:

```{math}
J_n=
\operatorname{clip}\left(
\frac{\operatorname{mean}_{a\in L,\ a\neq n}Q(a,n)-p_0}{p_0}
\right).
```

When there is no latest draw or no distinct member to average, {math}`J_n=0`.
The value is a smoothed engineering residual only in the sense of clipping; it
has no pseudocount and is not proof of conditional dependence.

## Exact 32-feature vector

The production feature schema is fixed and ordered. All values lie within
{math}`[-1,1]`; most lie within {math}`[0,1]`.

### Candidate and temporal features

| Index | Feature | Exact production value |
|---:|---|---|
| 1 | Normalized number | {math}`n/49` |
| 2 | Low-number indicator | {math}`\mathbf 1[n\leq24]` |
| 3 | Odd indicator | {math}`\mathbf 1[n\bmod2=1]` |
| 4 | Prime indicator | {math}`\mathbf 1[n\in\{2,3,5,7,11,13,17,19,23,29,31,37,41,43,47\}]` |
| 5 | Clipped gap | {math}`\min(g_n/60,1)` |
| 6 | Gap-one indicator | {math}`\mathbf 1[g_n=1]` |
| 7 | Gap-two-to-four indicator | {math}`\mathbf 1[2\leq g_n\leq4]` |
| 8 | Long-gap indicator | {math}`\mathbf 1[g_n\geq12]` |
| 9 | Overdue ratio | {math}`O_n` |
| 10 | Lifetime frequency residual | {math}`F_n` |
| 11 | Recent-5 residual | {math}`F_{n,5}` |
| 12 | Recent-20 residual | {math}`F_{n,20}` |
| 13 | Recent-100 residual | {math}`F_{n,100}` |
| 14 | Recent trend | {math}`\operatorname{clip}(F_{n,5}-F_{n,20})` |
| 15 | Latest-draw compatibility | {math}`J_n` |

There is no explicit bias feature because `SGDClassifier` fits a separate
intercept {math}`b`.

### Expert strength and efficacy interactions

Each of the six experts contributes two consecutive features in the order
listed earlier:

```{math}
\left(q_j(n),\;q_j(n)\frac{\omega_j}{1.5}\right).
```

| Indexes | Source |
|---:|---|
| 16–17 | Markov Spaces strength and efficacy interaction |
| 18–19 | Doublet & Triplet Markov strength and efficacy interaction |
| 20–21 | Bayesian strength and efficacy interaction |
| 22–23 | Temporal Behavior Learning strength and efficacy interaction |
| 24–25 | Markov Normalized Positions strength and efficacy interaction |
| 26–27 | Earth Mover Distance strength and efficacy interaction |

Dividing by 1.5 keeps the interaction in {math}`[0,1]` despite the efficacy
factor's upper bound. The base strength and interaction are deliberately both
present, allowing the fitted linear model to distinguish rank position from
quality-adjusted rank position.

### Cross-expert summary features

Let {math}`q_1,\ldots,q_6` be the six strengths. The final five features are:

| Index | Feature | Exact production value |
|---:|---|---|
| 28 | Expert mean | {math}`\frac16\sum_jq_j` |
| 29 | Expert median | Mean of the third and fourth ordered strengths |
| 30 | Top-6 support | {math}`\frac16\sum_j\mathbf 1[r_j(n)\leq6]` |
| 31 | Top-quarter support | {math}`\frac16\sum_j\mathbf 1[r_j(n)\leq13]` |
| 32 | Expert rank variance | {math}`\frac16\sum_j(q_j-\bar q)^2` |

The top-quarter boundary is exactly 13 because production uses
{math}`\lceil49\times0.25\rceil`. Variance uses the population divisor 6, not
the sample divisor 5.

## Linear SVM mathematics

For signed label {math}`y\in\{-1,+1\}`, margin
{math}`f=w^\top x+b`, and sample weight {math}`c_y`, the weighted hinge loss is

```{math}
\ell(y,f)=c_y\max(0,1-yf).
```

The class-imbalance weights are

```{math}
c_{+}=\frac{43}{6},\qquad c_{-}=1.
```

Thus the six positive examples carry total nominal weight 43 and the 43
negative examples also carry total nominal weight 43. This balances correction
mass; it does not imply equal event probabilities or calibrate the margin.

The configured scikit-learn estimator is:

| Parameter | Production value |
|---|---|
| Library version in the lockfile | scikit-learn 1.9.0 |
| Loss | `hinge` |
| Penalty | `l2` |
| Regularization strength | `alpha=0.0001` |
| Learning-rate schedule | `optimal` |
| Intercept | enabled |
| Averaged SGD | enabled from the beginning |
| Shuffle | enabled |
| Random seed | `20260626` |

Conceptually, the estimator minimizes a regularized weighted hinge objective of
the form

```{math}
\frac{\alpha}{2}\lVert w\rVert_2^2
+\text{weighted hinge loss}.
```

For the optimal schedule, scikit-learn defines

```{math}
\eta_t=\frac{1}{\alpha(t_0+t)},
```

where {math}`t_0` is selected by the library's Bottou heuristic. Production
does not set a custom step size. The estimator's default `eta0=0.01` and
`power_t=0.5` are not used by the optimal schedule.

A single unaveraged SGD step has the familiar structure of L2 shrinkage plus a
sample-weighted hinge correction when {math}`yf<1`. The exact update sequence,
intercept handling, shuffling, optimal-schedule state, and coefficient averaging
are owned by the locked scikit-learn implementation. With `average=True`, the
published `coef_` and `intercept_` are averages accumulated across SGD updates,
not merely the final instantaneous parameters.

## Delayed batch training

After each reference draw, production saves:

- one 32-feature row for every candidate 1 through 49;
- each expert's complete ranking used to build those rows.

When the next draw occurs, these saved rows become one labeled matrix

```{math}
X_t\in\mathbb R^{49\times32},
```

with a 49-value label vector and the {math}`43/6` versus 1 sample-weight vector.
The estimator receives one `partial_fit` call. Scikit-learn defines that call as
one SGD epoch over the supplied batch; it is not a convergence fit. With
`shuffle=True`, row presentation inside the epoch is deterministically shuffled
from the fixed random-state sequence.

The first call supplies the full class list `[0, 1]`; later calls retain the
known classes. Every completed target contributes exactly one 49-row epoch.
There is no replay buffer, minibatch accumulation across targets, early-stopping
gate, manual refit, or reset.

After fitting the target batch, production compares that same completed target
with the six saved expert Top-6 rankings and updates their cumulative and
recent-40 quality trackers. These updated qualities affect features only for the
next target.

## Causal walk-forward lifecycle

For target {math}`D_t`, the production sequence is:

1. A complete expert-ranking set and 49×32 feature matrix were saved after
   observing only draws through {math}`D_{t-1}`.
2. When {math}`D_t` occurs, create its 49 labels and call `partial_fit` on the
   saved matrix. The target changes model parameters only after its own forecast
   has already been fixed.
3. Compare each saved expert Top-6 with {math}`D_t` and update source-quality
   history.
4. Remember {math}`D_t` in the temporal, frequency, pair, and expert source
   states.
5. Rebuild all six source rankings through {math}`D_t`, calculate new efficacy
   factors using only completed targets, and construct the feature matrix for
   {math}`D_{t+1}`.
6. Score and save the SVM ranking for {math}`D_{t+1}`.

The unknown target cannot enter its own features, expert weights, or fitted
margin. Tests replay identical prefixes with different future draws and verify
that the earlier prediction remains unchanged. The fixed random seed also makes
repeated identical replays deterministic.

## Cold start

Before the first forecast there is no pending feature batch, so the first
observed draw cannot train the SVM. After that draw is remembered, production
builds the six expert rankings and all 32 features but the estimator remains
unfitted with **Trained draws 0**.

For this first forecast, all expert efficacy factors are 1 and the cold-start
margin is the equal-weight source consensus

```{math}
m_n^{(0)}=\frac16\sum_{j=1}^{6}q_j(n).
```

The common score scaling and tie-break then produce the forecast for draw 2.
When draw 2 occurs, that saved batch receives its labels and becomes the first
`partial_fit` epoch. Fitted `decision_function` margins are therefore first used
for the forecast of draw 3.

Cold start is not a neutral or random fallback. It is a rank-strength average
of six fully calculated source strategies.

## Fitted scoring and ranking

Once fitted, production evaluates the saved current rows with scikit-learn's
decision function:

```{math}
m_n=w^\top x(n)+b.
```

Let {math}`m_{\min}` and {math}`m_{\max}` be the smallest and largest of the 49
margins. When their spread is positive, the serialized score is

```{math}
s_n=\frac{m_n-m_{\min}}{m_{\max}-m_{\min}}.
```

If all margins are equal, every score falls back to 0. Min–max scaling is
monotone and therefore preserves the margin order whenever the spread is
nonzero.

Candidates are ranked by:

1. larger normalized score;
2. larger zero-based current gap when scores tie;
3. smaller number when score and gap both tie.

The gap tie-break is outside scikit-learn, although gap-derived fields also
appear in the feature vector. The first six candidates form the Top-6.

## Interpreting the application fields

| Field | Meaning |
|---|---|
| Score | Min–max normalized current consensus or fitted margin, shown as a percentage. It is not a probability. |
| Margin | Fitted `decision_function` value after at least one training batch. Positive values lie on the learned positive side; magnitude is uncalibrated. |
| Cold-start consensus | Six-expert mean rank strength used only before the first fitted batch. |
| Trained draws | Number of completed 49-row target batches processed by `partial_fit`. |
| Strongest expert inputs | Three source IDs with largest current {math}`q_j(n)\omega_j` products, with source ID resolving exact ties. This list explains current inputs but is not a coefficient attribution. |
| Gap | Zero-based current gap, used in features and as the final exact-score tie-break. |
| Rank | Position in the complete 1–49 ordering. |
| Top-6 membership | Whether the candidate occupies one of the first six ranks. |

The tooltip does not expose the 32 individual feature values, learned
coefficients, intercept, or all source weights. A named “strongest” expert can
still have a negative fitted coefficient, and correlated aggregate features can
change the effect of any one source.

## Endpoint diagnostic

After all 771 repository draws, the estimator has processed 770 labeled target
batches and 37,730 candidate rows. Scikit-learn reports
{math}`t_=37{,}731`, including its initial counter offset.

| Diagnostic | Endpoint value |
|---|---:|
| Trained target batches | 770 |
| Candidate examples processed | 37,730 |
| Averaged intercept | 0.270407 |
| Minimum next-forecast margin | -8.930267 |
| Maximum next-forecast margin | 16.172326 |
| Next Top-6 ranking | 36, 33, 39, 35, 21, 27 |

The 12 largest averaged coefficients by absolute magnitude are:

| Feature | Endpoint coefficient |
|---|---:|
| Clipped gap | 12.413166 |
| TBL strength | -11.135399 |
| TBL efficacy interaction | -9.861881 |
| Markov Spaces strength | 6.935912 |
| Overdue ratio | 5.499877 |
| Doublet & Triplet Markov strength | 4.825806 |
| Markov Normalized Positions strength | 4.428848 |
| Gap-one indicator | 3.999041 |
| Markov Spaces efficacy interaction | 3.944158 |
| Earth Mover Distance strength | 3.835113 |
| Bayesian strength | 3.388703 |
| Prime indicator | -2.898229 |

The completed source histories and current efficacy factors are:

| Expert | Cumulative hits / 770 | Recent-40 hits | Current factor |
|---|---:|---:|---:|
| Markov Spaces | 596 | 27 | 0.982201 |
| Doublet & Triplet Markov | 601 | 33 | 1.071577 |
| Bayesian | 596 | 41 | 1.184322 |
| Temporal Behavior Learning | 594 | 41 | 1.183220 |
| Markov Normalized Positions | 583 | 30 | 1.018356 |
| Earth Mover Distance | 592 | 24 | 0.936688 |

These are state diagnostics, not causal feature importances. Inputs are
correlated, strength and interaction fields duplicate related information,
coefficients use different feature scales, and averaged SGD parameters depend
on the full update path. New draws will change the histories, factors,
coefficients, and ranking.

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
| Full replay | 770 | 594 | 0.771429 | 565.714 |
| Validation, target draws 121–520 | 400 | 305 | 0.762500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 194 | 0.776000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, records 195 hits
or 0.780000 per target. The cold-start consensus, all recursive dependencies,
source-quality updates, fitted margins, and final gap-and-number tie-break are
included.

Every slice is above the theoretical random mean, but the feature schema,
expert pool, efficacy formula, and wider strategy collection were developed
with historical data available. These retrospective results do not establish
statistical significance, calibrated margins, independent replication, or
future advantage.

## Core mathematical and statistical concepts

- **Maximum-margin classification:** hinge loss rewards signed margins of at
  least one.
- **L2 regularization:** coefficient shrinkage discourages an unrestricted
  linear separator.
- **Stochastic gradient descent:** each completed target contributes one
  shuffled 49-row epoch through `partial_fit`.
- **Polyak-style averaging:** scikit-learn exposes coefficients averaged across
  online updates.
- **Class weighting:** six positives receive weight {math}`43/6` to balance 43
  negative examples.
- **Causal feature storage:** target labels are joined only with rows saved
  before that target occurred.
- **Rank stacking:** six heterogeneous strategies contribute bounded complete
  rank strengths.
- **Prior-smoothed source efficacy:** cumulative and recent Top-6 hits modulate
  rank interactions only after evaluation.
- **Consensus statistics:** mean, median, support fractions, and rank variance
  summarize agreement and disagreement.
- **Relative score scaling:** min–max transformation preserves order but removes
  absolute margin scale.
- **Hypergeometric overlap:** standard Top-6 efficacy uses the null overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Default-disabled status:** the engine is opt-in because it is a larger
  research model with recursive strategy dependencies.
- **Retrospective expert selection:** the six-source pool and 32-feature schema
  were chosen with historical development context.
- **Single epoch per target:** `partial_fit` does not converge each batch or the
  cumulative objective before the next forecast.
- **Library-version dependence:** shuffle order, optimizer details, and numerical
  behavior are tied to the locked scikit-learn implementation.
- **Averaging path dependence:** coefficients depend on every prior shuffled SGD
  update, not only aggregate counts.
- **Correlated candidate labels:** each 49-row batch has exactly six positives;
  rows are not independent Bernoulli observations.
- **Correlated features:** source strengths, efficacy interactions, consensus
  summaries, gaps, frequencies, and relationships reuse overlapping history.
- **Adaptive recent quality:** a 40-target recent window may react to chance
  streaks; the 24-draw prior reduces but does not remove this risk.
- **Double use of source evidence:** expert ranks enter directly and again
  through aggregates and efficacy interactions.
- **Conditional-rate sparsity:** early pair rates can be extreme despite
  clipping and lack an explicit pseudocount.
- **No feature standardization:** coefficient magnitudes are not directly
  comparable across differently distributed inputs.
- **Uncalibrated margins:** decision values and normalized percentages do not
  estimate the chance of next-draw inclusion.
- **Min–max information loss:** normalized scores hide absolute margin spread
  and cannot be compared directly across targets.
- **Tie-break influence:** gap and number can determine exact-margin ties outside
  the SVM.
- **Multiple comparisons and dataset dependence:** apparent historical lift can
  arise from chance, tuning, or evaluating many strategies.
- **No guaranteed predictability:** a fitted historical separator does not show
  that fair lottery outcomes have a stable learnable boundary.

Use Scikit Online SVM as a reproducible, leakage-protected ranking experiment,
not as a calibrated probability model or evidence of guaranteed future lift.

## Implementation map

| Responsibility | Production location |
|---|---|
| Expert IDs, recent window, neutral prior, feature names, and feature count | `src/rand_ai/strategy_prediction.py`, `_SKLEARN_SVM_*` constants |
| Hidden dependency graph and strategy ordering | `src/rand_ai/strategy_prediction.py`, `_STRATEGY_DEPENDENCIES` and `STRATEGY_IDS` |
| `SGDClassifier` construction and estimator state | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Recent frequency and latest-draw relationship residuals | `src/rand_ai/strategy_prediction.py`, `_sklearn_svm_recent_residual` and `_sklearn_svm_relationship_residual` |
| Prior-smoothed long/recent expert factor | `src/rand_ai/strategy_prediction.py`, `_sklearn_svm_expert_weight` |
| Exact 32-feature vector | `src/rand_ai/strategy_prediction.py`, `_sklearn_svm_features` |
| Delayed 49-row weighted `partial_fit` and completed expert-quality updates | `src/rand_ai/strategy_prediction.py`, `_train_sklearn_svm` |
| Pending feature/ranking capture, cold-start consensus, fitted margins, scaling, and details | `src/rand_ai/strategy_prediction.py`, `_sklearn_svm_scores` |
| Temporal, occurrence, pair, latest-draw, and source-strategy state updates | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train` and `_StrategyState.remember` |
| Final gap/number tie-break, Top-6 construction, efficacy, and causal orchestration | `src/rand_ai/strategy_prediction.py`, `_ranking_from_scores`, `_strategy`, and `build_prediction_suites` |
| Desktop serialization and default-disabled strategy registry | `src/rand_ai/gui_bridge.py` and `web/electron/main.cjs` |
| Settings description, family, color, names, and detail rendering | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Feature bounds, recursive dependencies, cold start, weighted batch, determinism, and prefix invariance tests | `tests/test_strategy_prediction.py` |
| Default-disabled and serialization coverage | `tests/test_gui_bridge.py` |
| Shared research feature consumer and comparative benchmark harness | `scripts/benchmark_sparse_neural_ticket.py` and `tests/test_sparse_neural_ticket_benchmark.py` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 normalized scores, ranks, current gaps,
detail strings, Top-6 numbers, and standard completed efficacy. The full feature
matrix, source-quality tables, learned coefficients, intercept, and optimizer
state remain internal.
