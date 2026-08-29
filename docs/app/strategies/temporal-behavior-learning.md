# Temporal Behavior Learning

## Introduction

**Temporal Behavior Learning** is the production strategy with identifier
`tbl` and short engine name **TBL**. It is a default-enabled member of the
**Relationships & Machine Learning** family. The engine learns an online
linear logistic score from candidate history, then supplements it with fixed
nonlinear adjustments for overdue behavior, short-versus-medium recency,
co-occurrence compatibility, and three source-strategy ranks.

TBL produces a complete ranking of numbers 1 through 49. Its first six entries
form the Top-6 prediction used by prediction grids, audits, effectiveness
histories, comparisons, portfolios, exports, and selected ensemble or
meta-model consumers.

```{admonition} A ranking model, not a calibrated probability model
:class: important

The interface labels TBL's pre-normalization sigmoid output **Probability**.
Positive examples are weighted by {math}`43/6`, fixed nonlinear terms are added
only during prediction, and the 49 outputs are subsequently min–max scaled.
The displayed value is therefore an internal ranking score, not an estimated
chance that the number will occur in the next draw.
```

## Scope

This page documents the exact production `tbl` engine: its binary candidate
problem, 14-feature vector, hidden strategy dependencies, sequential weighted
logistic update, fixed nonlinear scoring layer, causal lifecycle, cold start,
score transformation, interpretation, and retrospective walk-forward behavior.

It does not document the internal formulas of Freshness, Proximity, or the
deterministic Random baseline. TBL consumes their complete rankings as features;
those source strategies remain separate engines with their own outputs.

## Prediction problem

Let {math}`D_t` be the six-number set observed at draw {math}`t`. Each completed
draw becomes 49 binary candidate examples. For
{math}`n\in\{1,\ldots,49\}`, define

```{math}
y_{t,n}=
\begin{cases}
1, & n\in D_t,\\
0, & n\notin D_t.
\end{cases}
```

Every draw therefore supplies six positive examples and 43 negative examples.
The model maintains one shared coefficient vector and evaluates all candidates
against the history available before the observed label. It does not train 49
independent classifiers and does not directly generate a six-number ticket.

After all candidates are scored, the application creates one complete ranking
and takes its first six numbers.

## Hidden source dependencies

Selecting TBL activates three source engines recursively:

- **Freshness**;
- **Proximity**;
- the deterministic **Random baseline**.

When only TBL is requested, these dependencies can be calculated invisibly and
only TBL is serialized as a selected strategy. Their current complete rankings
provide two TBL features: Freshness rank strength and the mean of Proximity and
Random rank strength.

For a source ranking {math}`R` and candidate rank
{math}`r_R(n)\in\{1,\ldots,49\}`, production defines

```{math}
\rho_R(n)=\frac{49-r_R(n)}{48}.
```

Rank 1 maps to 1, rank 49 maps to 0, and intermediate positions are evenly
spaced. Only rank is transferred; source raw scores are not mixed into TBL.
The Random ranking is deterministic for a given draw index and fixed seed, so
identical replays receive identical values.

## Historical quantities

Let {math}`h` be the number of draws remembered before a feature vector is
constructed. For candidate {math}`n`, production maintains:

- lifetime appearances {math}`A_n`;
- zero-based current gap {math}`g_n`;
- ordered occurrence indexes;
- a recent-draw history capped at 100 draws;
- pair co-occurrence counts;
- the latest completed draw;
- the current or previously saved dependency rankings, according to the
  lifecycle stage.

The current gap is

```{math}
g_n=
\begin{cases}
h, & n\text{ has never appeared},\\
h-1-\ell_n, & \ell_n\text{ is its zero-based last-seen index}.
\end{cases}
```

A candidate in the latest remembered draw has gap 0. Before the first draw is
remembered, all gaps are 0.

When candidate {math}`n` has occurrence indexes
{math}`o_1<\cdots<o_k`, its mean inter-appearance distance is

```{math}
\bar g_n=
\begin{cases}
\dfrac{1}{k-1}\displaystyle\sum_{j=1}^{k-1}(o_{j+1}-o_j), & k\geq2,\\[8pt]
0, & k<2.
\end{cases}
```

## Exact 14-feature vector

For candidate {math}`n`, TBL constructs
{math}`x(n)\in\mathbb R^{14}`. Define the clipping function

```{math}
\operatorname{clip}(z)=\max(-1,\min(1,z)),
```

the count {math}`C_{n,k}` of appearances in the latest
{math}`\min(h,k)` draws, and recent rates

```{math}
R_{n,k}=\frac{C_{n,k}}{\max(\min(h,k),1)}.
```

The features appear in this exact production order:

| Index | Feature | Exact formula and range | Interpretation |
|---:|---|---|---|
| 1 | Bias | {math}`1` | Global intercept. |
| 2 | Normalized number | {math}`n/49\in[1/49,1]` | Numeric position. |
| 3 | Low-number indicator | {math}`\mathbf 1[n\leq24]` | Separates 1–24 from 25–49. |
| 4 | Odd indicator | {math}`\mathbf 1[n\bmod2=1]` | Odd-versus-even partition. |
| 5 | Prime indicator | {math}`\mathbf 1[n\in\mathcal P]` | Membership in the primes from 2 through 47. |
| 6 | Clipped current gap | {math}`\min(g_n/60,1)` | Linear gap scale capped at 60 draws. |
| 7 | Overdue ratio | {math}`0` if {math}`\bar g_n\leq0`; otherwise {math}`\operatorname{clip}((g_n-\bar g_n)/\bar g_n)` | Relative deviation from the candidate's own mean interval. |
| 8 | Lifetime frequency | {math}`A_n/\max(h,1)` | Historical per-draw appearance rate. |
| 9 | Recent-5 frequency | {math}`R_{n,5}` | Appearance rate over at most five recent draws. |
| 10 | Recent-20 frequency | {math}`R_{n,20}` | Appearance rate over at most 20 recent draws. |
| 11 | Recent trend | {math}`R_{n,5}-R_{n,20}\in[-1,1]` | Short-window rate minus medium-window rate. |
| 12 | Previous compatibility | {math}`K_n` defined below | Co-occurrence compatibility with the latest completed draw. |
| 13 | Freshness strength | {math}`\rho_{\mathrm{Freshness}}(n)` | Current complete-ranking strength from Freshness. |
| 14 | Proximity/Random strength | {math}`(\rho_{\mathrm{Proximity}}(n)+\rho_{\mathrm{Random}}(n))/2` | Equal average of two complete-ranking strengths. |

Here, {math}`\mathcal P` is the fixed set

```{math}
\{2,3,5,7,11,13,17,19,23,29,31,37,41,43,47\}.
```

The low-number partition is asymmetric by one value: 1–24 are low and 25–49
are high. No feature standardization, centering, learned embedding, or feature
selection is applied.

### Previous-draw compatibility

Let {math}`L` be the latest completed draw and let {math}`P(a,n)` be the
lifetime count of draws containing both distinct numbers {math}`a` and
{math}`n`. Production computes

```{math}
K_n=
\frac{
\displaystyle\sum_{a\in L,\ a\neq n}P(a,n)
}{\max(|L|h,1)}.
```

With a normal latest draw, {math}`|L|=6`. If {math}`n\in L`, its self-pair is
excluded from the numerator but the denominator remains {math}`6h`. When no
previous draw exists, compatibility is 0. This is a globally normalized
co-occurrence score, not a conditional probability and not a causal
relationship measure.

### Short-history safeguards

The guarded denominators make lifetime and recent rates zero before history
exists. The overdue ratio is zero until at least two appearances establish a
positive mean interval. Compatibility is zero before a latest draw and pair
history exist.

The initial dependency rankings are the numeric sequence 1 through 49, so the
first training pass does not begin with neutral rank features: lower-numbered
candidates initially have larger seeded rank strengths. After the first
forecast, actual Freshness, Proximity, and deterministic Random rankings replace
those seeds.

## Online logistic learning

### Linear score and sigmoid

TBL maintains one coefficient vector {math}`w\in\mathbb R^{14}`, initialized to
zero. During training, its linear score and guarded sigmoid are

```{math}
z_{t,n}=w^\top x_{t,n},
```

```{math}
\sigma_c(z)=
\frac{1}{1+\exp(-\operatorname{clip}_{[-35,35]}(z))}.
```

Clipping the logit avoids exponential overflow. It does not calibrate the
output.

### Class weighting

Because each draw contains six positive and 43 negative examples, production
uses

```{math}
c_1=\frac{43}{6},\qquad c_0=1.
```

The weighted residual for the current candidate is

```{math}
e_{t,n}=c_{y_{t,n}}\left(y_{t,n}-\sigma_c(z_{t,n})\right).
```

The nominal total positive and negative correction weights are thereby
balanced within a draw. This deliberately changes the scale and intercept that
would arise from the natural {math}`6/49` class prevalence, so the sigmoid
cannot be read as that prevalence.

### Learning rate, shrinkage, and sequential update

Before draw {math}`t` is remembered, let {math}`h=t-1`. The learning rate is

```{math}
\eta_h=\frac{0.09}{\sqrt{h+1}}.
```

For each candidate, production applies shrinkage coefficient 0.0006 and the
weighted logistic correction:

```{math}
w\leftarrow
(1-0.0006\eta_h)w
+\eta_h e_{t,n}x_{t,n}.
```

Candidates are processed sequentially in numeric order from 1 through 49. The
sigmoid residual for candidate {math}`n` is calculated from the weight state
left by all earlier candidates in the same draw. Shrinkage is also applied for
every candidate example. The final result is therefore order-dependent online
learning, not a batch logistic regression fit.

Only the linear score {math}`w^\top x` participates in training. The nonlinear
terms described next are not included in the residual and have no trainable
coefficients.

## Fixed nonlinear prediction layer

For prediction, TBL adds four hand-specified adjustments to the trained linear
score. Using the 1-based feature indexes above,

```{math}
q(n)=
0.20\tanh(1.4x_7)
+0.16\tanh(4x_{11})
+0.12\tanh(30x_{12})
+0.16\tanh\!\left(1.5(x_{13}+x_{14}-1)\right).
```

The terms represent:

1. overdue behavior relative to the candidate's mean interval;
2. the recent-5 versus recent-20 trend;
3. previous-draw compatibility;
4. joint source-ranking strength from Freshness and the
   Proximity/Random average.

The pre-normalization prediction value is

```{math}
p_n=\sigma_c\!\left(w^\top x(n)+q(n)\right).
```

The hyperbolic tangent bounds each contribution. Before coefficients and signs
inside the terms are considered, their maximum absolute outer amplitudes are
0.20, 0.16, 0.12, and 0.16. These constants are fixed; production does not
estimate, adapt, or validate them during the walk-forward update.

Because training optimizes the linear sigmoid while prediction ranks a different
linear-plus-nonlinear sigmoid, TBL is best understood as an online logistic
ranking model with a deterministic heuristic overlay rather than one coherent
maximum-likelihood logistic model.

## Causal walk-forward lifecycle

For an observed draw {math}`D_t`, production follows this sequence:

1. The pending forecast for {math}`D_t`, made from draws through
   {math}`D_{t-1}`, is ready for evaluation.
2. Build all 49 training feature vectors from state ending at
   {math}`D_{t-1}`. The source-rank features are the exact rankings saved with
   the prior forecast.
3. Process labels from the now-observed {math}`D_t` in numeric order and update
   the linear weights.
4. Remember {math}`D_t`: update appearances, last-seen indexes, occurrences,
   the recent history, pair counts, latest draw, Proximity state, and completed
   draw count.
5. Rebuild the current Freshness, Proximity, and deterministic Random rankings
   from state through {math}`D_t` and save them as the next rank-feature state.
6. Construct current TBL features, add the nonlinear prediction layer, and rank
   candidates for {math}`D_{t+1}`.

The target draw is never used to train or describe its own forecast. Its labels
are incorporated only after that target occurs. Appending future draws cannot
change a prediction already produced from an earlier prefix, provided the same
configuration and prefix are replayed.

The source rankings used in prediction and those retained for the next training
step are identical. This alignment lets the next observed labels train against
the rank features that were actually available when their forecast was made.

## Cold start

All 14 linear coefficients begin at zero. Before any draw, the history-derived
features are neutral except for candidate identity indicators and the seeded
numeric source rankings. In the normal application lifecycle, no untrained TBL
forecast is emitted: the first observed draw trains the model, is remembered,
and then generates the forecast for draw 2.

That first update is substantial because {math}`\eta_0=0.09`; later learning
rates decrease with {math}`1/\sqrt{h+1}`. Numeric candidate order, the six labels
in the first draw, and the non-neutral initial rank seeds can therefore have a
persistent effect. There is no minimum-history gate, uniform-prior fallback, or
periodic reset.

## Score normalization and ranking

After calculating all 49 values {math}`p_n`, production finds

```{math}
p_{\min}=\min_n p_n,\qquad p_{\max}=\max_n p_n.
```

If {math}`p_{\max}>p_{\min}`, the serialized strategy score is

```{math}
s_n=\frac{p_n-p_{\min}}{p_{\max}-p_{\min}}.
```

The smallest current value becomes 0 and the largest becomes 1. If all values
are equal, every score falls back to 0. The transformation is monotone, so it
does not change the ordering when the spread is positive.

Candidates are ranked by:

1. larger normalized TBL score;
2. larger zero-based current gap when scores tie;
3. smaller number when both score and gap tie.

Gap is already a learned feature, but the common ranking contract also uses it
as an exact-score tie-break. The first six candidates form the Top-6.

## Interpreting the application fields

| Field | Meaning |
|---|---|
| Score | Min–max normalized {math}`s_n`, shown as a percentage. It expresses relative position within the current 49-number range. |
| Probability | Raw linear-plus-nonlinear sigmoid {math}`p_n`, displayed to two decimals as a percentage. It is not calibrated. |
| Lifetime frequency | {math}`A_n/\max(h,1)`, displayed as a percentage. |
| Recent 20 | Candidate appearance count in the available portion of the latest 20 draws. |
| Gap | Zero-based current gap, used both in features and as the common tie-break. |
| Rank | Position in the complete 1–49 ordering after score and tie-breaking rules. |
| Top-6 membership | Whether the candidate occupies one of the first six ranks. |

The tooltip does not expose every feature, linear coefficient, linear logit, or
individual nonlinear contribution. Two candidates with similar displayed
fields can still differ through number identity, parity, prime status, recent-5
rate, overdue ratio, compatibility, or source rankings.

## Endpoint diagnostic

After all 771 repository draws, the learned linear coefficients are:

| Feature | Endpoint coefficient |
|---|---:|
| Bias | 0.026844 |
| Normalized number | -0.260159 |
| Low-number indicator | 0.062198 |
| Odd indicator | -0.024687 |
| Prime indicator | 0.049482 |
| Clipped current gap | 0.311371 |
| Overdue ratio | 0.018558 |
| Lifetime frequency | 0.215060 |
| Recent-5 frequency | 0.047161 |
| Recent-20 frequency | 0.058427 |
| Recent trend | -0.011266 |
| Previous compatibility | 0.194096 |
| Freshness strength | 0.104722 |
| Proximity/Random strength | -0.048776 |

For the next forecast at that endpoint:

| Diagnostic | Endpoint value |
|---|---:|
| Minimum linear score | -0.132761 |
| Maximum linear score | 0.223478 |
| Minimum nonlinear adjustment | -0.235869 |
| Maximum nonlinear adjustment | 0.275784 |
| Minimum raw sigmoid value | 0.432736 |
| Maximum raw sigmoid value | 0.621195 |
| Top-6 ranking | 6, 17, 38, 40, 8, 34 |

These coefficients are one endpoint of an order-dependent online fit. Feature
scales and correlations differ, and the nonlinear layer can reinforce or oppose
the linear contribution. Coefficient magnitude alone is therefore not a causal
importance measure. Every value will change when a new draw is learned or the
historical sequence changes.

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
| Validation, target draws 121–520 | 400 | 293 | 0.732500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 206 | 0.824000 | 183.673 |

The latest 250-target slice, target draws 522–771, also records 206 hits or
0.824000 per target. The production dependency rankings, nonlinear layer,
min–max transform, and gap-and-number tie-break are included.

The validation slice is slightly below the theoretical random mean while the
later slice is substantially higher. That reversal may reflect chance,
non-stationarity, dependence, or broader historical selection; it is not proof
of a regime that TBL can identify in advance. These retrospective measurements
do not establish statistical significance, calibration, independent
replication, or future advantage.

## Core mathematical and statistical concepts

- **Binary candidate expansion:** each six-number draw creates 49 labels with
  exactly six positives.
- **Online logistic learning:** one shared coefficient vector is updated after
  each candidate rather than refitted in a batch.
- **Class weighting:** positive corrections receive weight {math}`43/6` to
  balance their smaller count.
- **Multiplicative shrinkage:** every example contracts the coefficient vector
  before adding the weighted residual correction.
- **Temporal features:** lifetime rate, recent rates, trend, gap, and overdue
  ratio summarize different historical horizons.
- **Relationship feature:** normalized co-occurrence with the latest draw adds a
  pair-history signal.
- **Rank stacking:** complete source rankings enter as bounded rank-strength
  features without raw-score mixing.
- **Nonlinear basis functions:** fixed `tanh` terms saturate four selected
  behaviors during prediction.
- **Relative normalization:** min–max scaling preserves current ordering but
  removes absolute output scale.
- **Hypergeometric overlap:** standard efficacy uses the null Top-6 overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Uncalibrated “Probability”:** class weighting, heuristic prediction terms,
  and min–max scaling prevent a literal probability interpretation.
- **Training/scoring mismatch:** the online residual trains only the linear
  sigmoid, while ranking uses a linear-plus-nonlinear sigmoid.
- **Hand-selected nonlinear constants:** amplitudes and multipliers were fixed
  engineering choices, not learned in the production update.
- **Order dependence:** processing candidates 1 through 49 sequentially makes
  the fitted state sensitive to numeric order.
- **Early-history influence:** the decreasing learning rate gives the first
  observations and seeded source ranks comparatively persistent influence.
- **No minimum-history gate:** the model emits a learned ranking immediately
  after one training draw.
- **Correlated examples:** the 49 labels within a draw are constrained to
  exactly six positives and are not independent Bernoulli trials.
- **Overlapping time windows:** lifetime, recent-20, recent-5, trend, and gap
  features reuse the same observations.
- **Co-occurrence ambiguity:** pair counts can reflect marginal frequency and do
  not demonstrate dependence or causality.
- **Source-feature dependence:** Freshness, Proximity, and Random rankings can
  inject their tie-breaks, assumptions, and deterministic artifacts into TBL.
- **Random-rank feature:** a deterministic pseudo-random ordering is a model
  input, not evidence about the drawing mechanism.
- **No feature standardization:** coefficient magnitudes are not directly
  comparable across differently scaled inputs.
- **Min–max information loss:** normalized scores conceal absolute sigmoid
  spread and cannot be compared directly between forecasts.
- **Retrospective instability:** validation and later-slice behavior differ
  materially on the included history.
- **Dataset and multiple-comparison dependence:** apparent replay lift can arise
  from chance, development choices, or comparison across many strategies.
- **No guaranteed predictability:** historical temporal patterns do not
  establish a stable mechanism for future lottery outcomes.

Use TBL as an auditable, causal ranking experiment—not as a calibrated forecast
or evidence that a fair lottery has learnable temporal behavior.

## Implementation map

| Responsibility | Production location |
|---|---|
| Strategy identifier, ordering, short name, and hidden dependencies | `src/rand_ai/strategy_prediction.py`, `STRATEGY_IDS` and `_STRATEGY_DEPENDENCIES` |
| Weight initialization, source-ranking seeds, occurrences, recent draws, pair counts, and latest draw | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Rank-strength conversion, mean interval, and previous compatibility | `src/rand_ai/strategy_prediction.py`, `_rank_score`, `_mean_gap`, and `_previous_compatibility` |
| Exact 14-feature vector | `src/rand_ai/strategy_prediction.py`, `_StrategyState._tbl_features` |
| Guarded sigmoid and clipping | `src/rand_ai/strategy_prediction.py`, `_sigmoid` and `_clamp` |
| Sequential weighted logistic update, learning rate, and shrinkage | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train` |
| Appearance, gap, recent, pair, latest-draw, and Proximity state updates | `src/rand_ai/strategy_prediction.py`, `_StrategyState.remember` |
| Current dependency ranks, nonlinear layer, raw values, details, and min–max scaling | `src/rand_ai/strategy_prediction.py`, `_StrategyState.build_strategies` |
| Final gap/number tie-break, Top-6 construction, efficacy, and causal orchestration | `src/rand_ai/strategy_prediction.py`, `_ranking_from_scores`, `_strategy`, and `build_prediction_suites` |
| Desktop serialization | `src/rand_ai/gui_bridge.py`, `_strategy_payload` and `_suite_payload` |
| Default-enabled plugin registration and display label | `web/electron/main.cjs` |
| Settings description, family, color, names, and detail rendering | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Strategy-suite construction, enabled-strategy behavior, ranking contract, and bridge serialization tests | `tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 normalized scores, ranks, current gaps,
detail strings, Top-6 numbers, and standard completed efficacy. The learned
coefficient vector, full feature vectors, linear logits, and separate nonlinear
contributions are not serialized to the interface.
