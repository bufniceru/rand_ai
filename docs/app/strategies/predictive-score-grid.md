# Predictive Score Grid

## Introduction

**Predictive Score Grid** is the production strategy with identifier
`predictive_grid` and short engine name **Grid**. It is a default-enabled member
of the **Shape & Similarity** family. The engine constructs six independently
normalized historical score columns, combines them into a fixed 70% history
block, and reserves 30% of the final score for Earth Mover Distance similarity.

The strategy produces a complete ranking of numbers 1 through 49. Its first six
entries form the Top-6 prediction used by prediction grids, audits,
effectiveness histories, comparisons, portfolios, exports, and selected
ensemble or meta-model consumers.

```{admonition} A seven-component grid, not a fitted learner
:class: important

Predictive Score Grid has no learned regression coefficients or adaptive blend
weights. Each raw history column is min–max normalized across the current 49
candidates, then combined with fixed constants. The displayed component
percentages are normalized column positions—not probabilities and not their
already-weighted contributions to the final score.
```

## Scope

This page documents the exact production `predictive_grid` engine: its hidden
dependencies, causal state updates, seven component definitions, independent
normalization, fixed blend, cold start, ranking rules, interpretation, endpoint
diagnostics, and retrospective walk-forward behavior.

It does not reproduce the complete standalone Earth Mover Distance or Markov
100 implementations. It explains precisely which state and score each source
contributes to the grid.

## Prediction problem

After chronological draws {math}`D_1,\ldots,D_h` have been observed, the engine
assigns a score to every candidate
{math}`n\in\{1,\ldots,49\}` for the unknown draw {math}`D_{h+1}`.

The seven score columns are:

1. gap-state Markov probability;
2. last-draw transition rate;
3. lifetime frequency;
4. recent-20 activity;
5. current gap;
6. pair affinity with the latest draw;
7. Earth Mover Distance successor similarity.

The engine assumes that these heterogeneous historical views may contain
complementary ranking information. It does not claim that the columns are
independent, probabilistically calibrated, or causally related to the next
draw.

## Hidden dependencies

Selecting Predictive Score Grid recursively activates:

- **Markov 100** (`markov100`), which maintains the decayed gap-state tables;
- **Earth Mover Distance** (`emd`), which supplies its complete 1–49 score map.

When only Predictive Score Grid is selected, those dependencies can run
invisibly and only `predictive_grid` is serialized as a requested strategy.

The grid does not blend the final Markov 100 ranking. It reads the same Markov
gap tables and calculates a gap-bucket probability before normalizing that
column. In contrast, it consumes the already normalized standalone EMD score
directly as its seventh component.

## Common column normalization

For any raw component values {math}`a_1,\ldots,a_{49}`, define

```{math}
\mathcal N(a_n)=
\begin{cases}
\dfrac{a_n-a_{\min}}{a_{\max}-a_{\min}}, & a_{\max}>a_{\min},\\[7pt]
0, & a_{\max}=a_{\min}.
\end{cases}
```

Every one of the six history columns is transformed separately. The smallest
raw value becomes 0 and the largest becomes 1. If all 49 raw values are equal,
the entire component contributes zero rather than a neutral constant.

Independent scaling makes unlike units combinable, but it also removes their
absolute magnitude. A very small spread in transition rates can occupy the same
0–1 range as a large spread in gaps or frequencies.

## Component 1: gap-state Markov

### Causal gap observations

Let {math}`p_0=6/49`, and cap each zero-based current gap at bucket 35. Before a
newly observed draw is remembered, every candidate contributes one opportunity
to its pre-target gap bucket, and an additional hit if it occurs in that draw.

Starting with the second observed draw, production first decays all gap tables
by

```{math}
\lambda=2^{-1/500},
```

then adds the current 49 opportunities and six hits. For bucket
{math}`b\in\{0,\ldots,35\}`, write the decayed totals as {math}`O_b` and
{math}`H_b`. The prior-smoothed probability is

```{math}
P_b=\frac{H_b+8p_0}{O_b+8}.
```

For the next forecast, candidate {math}`n` receives raw value

```{math}
a_n^{(M)}=P_{\min(g_n,35)},
```

where {math}`g_n` is its current zero-based gap after the latest draw is
remembered. The displayed component is

```{math}
M_n=\mathcal N(a_n^{(M)}).
```

All candidates in the same current gap bucket have the same raw Markov value.
The eight-draw prior stabilizes bucket estimates but does not make the
min–max-normalized percentage a calibrated probability.

## Component 2: last-draw transitions

For predecessor number {math}`p` and candidate {math}`n`, production stores
{math}`C_{p,n}`, the number of times {math}`n` appeared in the draw immediately
following a draw that contained {math}`p`. It also stores

```{math}
T_p=\sum_{n=1}^{49}C_{p,n}.
```

Every predecessor occurrence adds six successor balls, so {math}`T_p`
increases by 6 whenever {math}`p` belongs to the preceding draw. Repetition of
the same number across consecutive draws is allowed in the matrix; there is no
self-transition exclusion.

Let {math}`L=D_h` be the latest six-number draw. The raw transition column is

```{math}
a_n^{(T)}=
\frac{1}{6}\sum_{p\in L}
\begin{cases}
C_{p,n}/T_p, & T_p>0,\\
0, & T_p=0.
\end{cases}
```

and the displayed component is

```{math}
T_n=\mathcal N(a_n^{(T)}).
```

This averages six empirical successor-ball shares conditioned on members of
the latest draw. It is not a fitted joint transition probability for the next
six-number set.

## Component 3: lifetime frequency

If {math}`A_n` is candidate {math}`n`'s appearance count over {math}`h`
completed draws, the raw value is

```{math}
a_n^{(F)}=\frac{A_n}{\max(h,1)}.
```

The grid uses

```{math}
F_n=\mathcal N(a_n^{(F)}).
```

Because every candidate has the same denominator, this component ranks exactly
by lifetime count whenever those counts are not all equal. It favors larger
historical frequency; it does not compare the count with an uncertainty band or
apply a frequency prior.

## Component 4: recent-20 activity

Let {math}`R_n` be the count of draws containing {math}`n` within the latest at
most 20 completed draws. The raw and normalized values are

```{math}
a_n^{(R)}=R_n,
\qquad
R_n^*=\mathcal N(a_n^{(R)}).
```

Production stores at most 100 recent draws, but this component reads only the
latest 20. It uses a count rather than a rate; because the available window
length is common to every candidate, the ordering would be unchanged by
dividing all counts by that length.

## Component 5: current gap

The current gap is zero for a number in the latest draw. A never-seen candidate
has gap {math}`h`; otherwise it is the number of completed draws since its most
recent appearance, excluding the appearance draw itself.

The component is

```{math}
G_n=\mathcal N(g_n).
```

Larger gaps receive larger values. Unlike the Markov column, this column uses
the uncapped gap before normalization and assumes a monotone preference for
longer absence. It also later acts as the common exact-score tie-break.

## Component 6: pair affinity

Let {math}`P(i,j)` be the lifetime number of draws containing the unordered
distinct pair {math}`\{i,j\}`. For latest draw {math}`L`, production calculates

```{math}
a_n^{(P)}=
\operatorname{mean}_{p\in L,\ p\neq n}P(n,p).
```

The candidate itself is omitted when {math}`n\in L`, leaving five terms;
otherwise all six latest-draw members contribute. If no term exists, the raw
value is 0. The displayed component is

```{math}
P_n^*=\mathcal N(a_n^{(P)}).
```

These are raw co-occurrence counts. They are not divided by candidate or latest
number frequency, so commonly occurring numbers can receive systematically
larger affinity values.

## Component 7: Earth Mover Distance

Earth Mover Distance compares the latest sorted six-number draw with each
historical predecessor draw. A predecessor at distance {math}`d_i` receives
weight {math}`1/(1+d_i)`, and its already observed successor lends that weight
to each of its six numbers. The source strategy divides candidate weighted
evidence by the largest candidate evidence, producing

```{math}
E_n\in[0,1].
```

Predictive Score Grid consumes {math}`E_n` directly. It does not independently
min–max scale the EMD column, select a nearest-neighbor subset, or alter the EMD
history. See the dedicated Earth Mover Distance guide for the complete source
calculation and diagnostic bands.

## Exact seven-component blend

The six normalized history columns first form

```{math}
H_n=
0.35M_n
+0.20T_n
+0.15F_n
+0.15R_n^*
+0.10G_n
+0.05P_n^*.
```

Those internal weights sum to 1. The final grid score is

```{math}
S_n=0.70H_n+0.30E_n.
```

Equivalently, the effective final shares are:

| Component | Inside-history weight | Effective final weight |
|---|---:|---:|
| Gap-state Markov | 35% | 24.5% |
| Last-draw transition | 20% | 14.0% |
| Lifetime frequency | 15% | 10.5% |
| Recent-20 activity | 15% | 10.5% |
| Current gap | 10% | 7.0% |
| Pair affinity | 5% | 3.5% |
| Earth Mover Distance | — | 30.0% |

There is no final min–max transformation after this blend. Although every input
lies in {math}`[0,1]`, the highest final score need not be 1 because different
candidates can maximize different columns. No agreement bonus, quota, adaptive
weight, recent efficacy adjustment, or second-stage learner is applied.

## Causal walk-forward lifecycle

For newly observed draw {math}`D_t`, production advances in this order:

1. The forecast for {math}`D_t` already exists from state ending at
   {math}`D_{t-1}`.
2. Decay the Markov gap tables and record each candidate's pre-target gap
   opportunity plus the six observed hits from {math}`D_t`.
3. Remember {math}`D_t`: increment lifetime counts and unordered pair counts,
   add the immediate transition {math}`D_{t-1}\rightarrow D_t`, update the
   latest-draw pointer, append the recent history, and append the sorted EMD
   vector.
4. Calculate current gaps and all seven columns using only draws through
   {math}`D_t`.
5. Blend, tie-break, and store the Top-6 forecast for {math}`D_{t+1}`.

The target cannot enter its own score grid. The Markov outcome, transition,
pair, frequency, recent activity, gap, and EMD successor evidence from
{math}`D_t` become available only for the following target. Prefix-invariance
tests verify that appending later draws does not alter an already generated
grid ranking.

## Cold start

Before any draw is observed, every historical raw column is equal and a zero
EMD input yields 49 final scores of zero. In the normal application lifecycle,
the first forecast is emitted after draw {math}`D_1` is remembered.

At that point:

- the Markov and transition columns are all equal and normalize to zero;
- the six numbers in {math}`D_1` have normalized lifetime, recent, and pair
  values of 1;
- unseen numbers have normalized current gap 1, while the six observed numbers
  have gap component 0;
- EMD has no completed predecessor-successor transition and contributes zero.

Therefore each just-observed number receives

```{math}
0.70(0.15+0.15+0.05)=0.245,
```

while each unseen number receives

```{math}
0.70(0.10)=0.070.
```

The first Top-6 is exactly the first observed draw. This is a deterministic
consequence of the frequency, recent, and pair columns—not a special repeat
rule or evidence of a learned transition.

## Ranking and tie-breaking

Candidates are ordered by:

1. larger final grid score {math}`S_n`;
2. larger zero-based current gap when scores tie;
3. smaller number when score and gap both tie.

The final score is already bounded between 0 and 1 and is serialized directly.
The gap tie-break is separate from the 7% effective gap contribution and can
still decide candidates with exactly equal blended scores. The first six ranks
form the Top-6.

## Interpreting the application fields

Each number exposes seven detail lines:

| Field | Meaning |
|---|---|
| Score | Final fixed blend {math}`S_n`. It is a relative composite score, not a probability. |
| Gap-state Markov | Independently min–max-normalized smoothed gap-bucket probability {math}`M_n`; its effective final weight is 24.5%. |
| Last-draw transition | Normalized mean successor share {math}`T_n`; effective weight 14%. |
| Lifetime frequency | Normalized lifetime appearance rate {math}`F_n`; effective weight 10.5%. |
| Recent-20 activity | Normalized recent count {math}`R_n^*`; effective weight 10.5%. |
| Current gap | Normalized uncapped gap {math}`G_n`; effective weight 7%. |
| Pair affinity | Normalized mean raw co-occurrence count {math}`P_n^*`; effective weight 3.5%. |
| Earth-mover similarity | Standalone EMD score {math}`E_n`; effective weight 30%. |
| Rank | Position in the complete ordering after score and tie-breaking. |
| Top-6 membership | Whether the candidate occupies one of the first six ranks. |

The component percentages show normalized inputs before their weights. For
example, **Gap-state Markov 100%** means the candidate has the largest current
raw Markov value; it contributes 24.5 percentage points to the final score, not
100 percentage points.

## Endpoint diagnostic

After all 771 repository draws, the next-forecast state contains:

| Diagnostic | Endpoint value |
|---|---:|
| Completed draws | 771 |
| Decayed Markov opportunity mass | 23,207.122 |
| Decayed Markov hit mass | 2,841.688 |
| Immediate-transition ball count | 27,720 |
| Stored unordered pair appearances | 11,565 |
| Recent-history capacity in use | 100 draws |
| Stored EMD draw vectors | 771 |
| Next Top-6 ranking | 40, 6, 39, 4, 8, 20 |

The transition total equals {math}`770\times6\times6`: each of 770 completed
draw-to-draw transitions supplies six predecessor numbers and six successor
balls. The pair total equals {math}`771\times\binom62=11{,}565`.

For endpoint rank 1, number 40 has final score 0.856268 and normalized inputs:

| Component | Number 40 endpoint value |
|---|---:|
| Gap-state Markov | 100.0% |
| Last-draw transition | 100.0% |
| Lifetime frequency | 83.7% |
| Recent-20 activity | 33.3% |
| Current gap | 41.0% |
| Pair affinity | 84.5% |
| Earth Mover Distance | 96.7% |

These values describe one fitted historical endpoint. New draws change the
normalization minima and maxima as well as every causal state table, so both the
component percentages and final ranking can move.

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
| Full replay | 770 | 585 | 0.759740 | 565.714 |
| Validation, target draws 121–520 | 400 | 307 | 0.767500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 181 | 0.724000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, records 179 hits
or 0.716000 per target. The full dependency state, first-draw behavior, fixed
blend, and final gap-and-number tie-break are included.

The holdout and latest slices are below the theoretical random mean even though
the full and validation summaries are above it. That instability prevents a
claim of robust improvement. All figures are retrospective and do not establish
statistical significance, calibration, independent replication, or future
advantage.

## Core mathematical and statistical concepts

- **Exponentially decayed counts:** Markov gap evidence has an approximate
  500-draw half-life.
- **Prior smoothing:** each gap bucket begins with eight draws of neutral
  {math}`6/49` evidence.
- **First-order conditional counts:** the transition matrix summarizes
  predecessor-number to successor-number shares.
- **Frequency and recency:** lifetime and recent-20 columns represent long and
  short historical horizons.
- **Co-occurrence affinity:** unordered pair counts relate candidates to the
  latest draw.
- **Optimal transport similarity:** EMD transfers evidence from successors of
  geometrically similar historical draws.
- **Independent min–max normalization:** six unlike raw columns are mapped to a
  shared 0–1 scale.
- **Fixed linear aggregation:** seven predefined weights form the final score
  without fitting.
- **Hypergeometric overlap:** standard Top-6 efficacy uses the null overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Fixed retrospective weights:** 70/30 and all six internal history weights
  are engineering choices, not learned or independently validated.
- **Independent scaling distortion:** a tiny raw spread can receive the same
  0–1 influence range as a large raw spread.
- **Cross-draw incomparability:** a component percentage depends on the current
  49-number minimum and maximum.
- **Correlated inputs:** gap Markov, raw gap, frequency, recency, transitions,
  pairs, and EMD all reuse overlapping draw history.
- **Frequency duplication:** lifetime frequency influences its own column and
  can indirectly influence pair counts and EMD successor evidence.
- **Sparse transition rates:** predecessor rows with limited history can be
  unstable and receive no explicit prior.
- **Raw pair confounding:** pair affinity is not adjusted for either member's
  marginal frequency.
- **Monotone gap assumption:** the raw gap column always favors longer absence,
  independently of the learned Markov bucket probability.
- **Broad EMD history:** every eligible EMD transition receives positive weight,
  including distant states.
- **No uncertainty estimate:** the final score contains no standard error,
  credible interval, or calibration check.
- **Deterministic cold start:** the first ranking repeats the first observed
  draw because three columns favor it.
- **Tie-break reuse:** current gap influences both the blended score and exact
  score ties.
- **Negative later replay:** the recorded holdout and latest slices trail the
  theoretical random expectation.
- **Dataset and multiple-comparison dependence:** apparent full-history lift can
  reflect chance, tuning, or comparison across many strategies.
- **No guaranteed predictability:** historical grids and similarities do not
  establish a stable mechanism for future lottery outcomes.

Use Predictive Score Grid as an auditable fixed composite of historical
rankings, not as a calibrated probability model or evidence of guaranteed
future improvement.

## Implementation map

| Responsibility | Production location |
|---|---|
| Number count, base rate, gap cap, Markov prior and decay, and 30% EMD weight | `src/rand_ai/strategy_prediction.py`, module constants |
| Hidden Markov 100 and EMD dependencies | `src/rand_ai/strategy_prediction.py`, `_STRATEGY_DEPENDENCIES` |
| Appearances, gaps, recent history, pairs, transitions, and Markov tables | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Pre-target decayed Markov opportunity and hit updates | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train` |
| Lifetime, pair, immediate-transition, recent, latest-draw, and EMD history updates | `src/rand_ai/strategy_prediction.py`, `_StrategyState.remember` |
| Standalone EMD score source | `src/rand_ai/strategy_prediction.py`, `_earth_mover_scores` |
| Six raw columns, independent scaling, fixed 70/30 blend, and detail strings | `src/rand_ai/strategy_prediction.py`, `_predictive_grid_scores` |
| Final gap/number tie-break, Top-6 construction, efficacy, and causal orchestration | `src/rand_ai/strategy_prediction.py`, `_ranking_from_scores`, `_strategy`, and `build_prediction_suites` |
| Desktop serialization and default-enabled registration | `src/rand_ai/gui_bridge.py` and `web/electron/main.cjs` |
| Settings description, family, color, names, and detail rendering | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| EMD blend mathematics, composite prefix invariance, strategy ordering, and serialization coverage | `tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 final scores, ranks, current gaps, seven
detail strings, Top-6 numbers, and standard completed efficacy. Raw unnormalized
columns and internal state tables are not serialized.
