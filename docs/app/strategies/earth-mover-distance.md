# Earth Mover Distance

## Introduction

**Earth Mover Distance** is the production strategy with identifier `emd` and
short engine name **EMD**. It is a default-enabled member of the **Shape &
Similarity** family. The engine treats each completed draw as six equally
weighted points on the number line, measures how much the latest draw differs
from every eligible historical draw, and transfers weighted evidence from the
draws that followed those historical states.

The strategy produces a complete ranking of numbers 1 through 49. Its first six
entries form the Top-6 prediction used by prediction grids, audits,
effectiveness histories, comparisons, portfolios, exports, and selected
ensemble or meta-model consumers.

```{admonition} What “Earth Mover” means here
:class: important

Production does not call a general optimal-transport library or solve a
multidimensional transport program. Because both draws contain six equally
weighted points on a one-dimensional line, the exact first Wasserstein distance
is obtained by sorting both draws and averaging the six position-wise absolute
differences. This closed form is the Earth Mover Distance used by the engine.
```

## Scope

This page documents the exact production draw representation, one-dimensional
transport distance, historical transition set, inverse-distance kernel,
per-number weighted evidence, diagnostic distance bands, normalization, causal
lifecycle, ranking rules, and retrospective walk-forward behavior of `emd`.

It does not describe the formulas of strategies that consume the EMD ranking.
Those consumers can combine or transform its output, but they do not alter the
standalone source ranking documented here.

## Prediction problem

Let the observed chronological history be
{math}`D_1,D_2,\ldots,D_h`, where every draw contains six unique values from 1
through 49. After observing {math}`D_h`, the strategy must score every candidate
{math}`n\in\{1,\ldots,49\}` for the unobserved target {math}`D_{h+1}`.

The production rule uses an analogue-transition assumption:

> Historical draws that are geometrically similar to the latest draw lend more
> weight to the numbers in their observed successor draws.

Unlike a nearest-neighbor classifier, the engine does not select a fixed number
of closest analogues. Every historical predecessor with a known successor is
included, and distance changes only the size of its contribution.

## Draw representation and transport distance

Write two draws in ascending order as

```{math}
A=(a_1<a_2<\cdots<a_6),\qquad
B=(b_1<b_2<\cdots<b_6).
```

Associate each draw with an equal-mass empirical distribution,

```{math}
\mu_A=\frac{1}{6}\sum_{j=1}^{6}\delta_{a_j},
\qquad
\mu_B=\frac{1}{6}\sum_{j=1}^{6}\delta_{b_j}.
```

On the one-dimensional number line, the monotone optimal matching pairs equal
order statistics. The production distance is therefore

```{math}
d(A,B)=W_1(\mu_A,\mu_B)
=\frac{1}{6}\sum_{j=1}^{6}|a_j-b_j|.
```

The value is an average displacement in lottery-number units. It is zero for
identical draws and grows as their sorted positions move farther apart. For
valid six-number draws its range is 0 through 43: the maximum is attained, for
example, by comparing {math}`(1,2,3,4,5,6)` with
{math}`(44,45,46,47,48,49)`.

Sorting makes the representation independent of the order in which balls were
stored. The model deliberately uses no explicit frequency, current-gap, parity,
calendar, sum, spacing, or overlap feature inside this distance.

### Defensive sequence behavior

The helper computes over the shorter of two supplied sequences and returns zero
when both are empty. Normal application data always pass validated six-number
draws, so the production prediction path uses all six ordered positions.

## Historical analogue transitions

The latest observed draw {math}`D_h` is the query. For each historical index

```{math}
i\in\{1,2,\ldots,h-1\},
```

the engine compares {math}`D_h` with predecessor {math}`D_i` and uses the
already observed following draw {math}`D_{i+1}` as its outcome. The distance and
kernel weight are

```{math}
d_i=d(D_h,D_i),
\qquad
w_i=\frac{1}{1+d_i}.
```

The added 1 keeps an exact match finite: {math}`d_i=0` receives weight 1. Every
non-exact match receives a positive weight below 1, so even distant transitions
remain in the aggregate. There is no cutoff, nearest-neighbor quota, recency
decay, trainable coefficient, or prior term.

After {math}`h` observed draws, exactly {math}`h-1` transitions are eligible.
This includes the newest completed transition
{math}`D_{h-1}\rightarrow D_h`: its predecessor is compared with the current
query, and its known successor is the latest observed draw. Using that transition
is causal because the forecast target is {math}`D_{h+1}`, not {math}`D_h`.

## Per-number evidence and score

For candidate {math}`n`, production accumulates the weighted successor evidence

```{math}
u_n=
\sum_{i=1}^{h-1}
w_i\,\mathbf{1}[n\in D_{i+1}].
```

Each historical transition contributes its weight to all six numbers in its
successor draw. Consequently,

```{math}
\sum_{n=1}^{49}u_n=6\sum_{i=1}^{h-1}w_i.
```

Dividing {math}`u_n` by {math}`\sum_iw_i` would produce a kernel-weighted
historical occurrence rate. Production instead performs maximum-only scaling:

```{math}
s_n=
\begin{cases}
\dfrac{u_n}{\max_m u_m}, & \max_m u_m>0,\\[6pt]
0, & \text{otherwise}.
\end{cases}
```

The most strongly supported candidate therefore receives score 1 and all other
scores lie between 0 and 1. This transformation is monotone and does not change
the ordering of the raw evidence. It is not min–max scaling because no minimum
is subtracted.

The displayed percentage is a relative score against the current maximum. It
is not a calibrated inclusion probability, posterior probability, p-value, or
confidence level. Scores from different target draws are not directly
comparable because their common denominator changes with the history and query.

## Diagnostic support and distance bands

For each candidate, **Support draws** is the unweighted number of eligible
successor draws containing it:

```{math}
c_n=\sum_{i=1}^{h-1}\mathbf{1}[n\in D_{i+1}].
```

When {math}`u_n>0`, the displayed average distance is the candidate-specific
weighted mean

```{math}
\bar d_n=
\frac{
\sum_{i=1}^{h-1}d_iw_i\mathbf{1}[n\in D_{i+1}]
}{u_n}.
```

Production maps that value to one of six labels:

| Label | Exact production condition |
|---|---|
| Overlap | {math}`\bar d_n\leq1` |
| Near | {math}`1<\bar d_n\leq3` |
| Close | {math}`3<\bar d_n\leq5` |
| Middle | {math}`5<\bar d_n\leq8` |
| Far | {math}`8<\bar d_n\leq12` |
| Distant | {math}`\bar d_n>12` |

An unsupported candidate is also labeled **Distant** and displays average
distance 0.00. The label is explanatory only: it does not add a bonus, remove a
candidate, or change the score. It summarizes the weighted distances of the
transitions that specifically supported that number; it is not the nearest
distance to the latest draw and not a classification of the latest draw as a
whole.

## Cold start

Before any draw is stored, the internal scorer returns 49 zero scores and no
detail rows. The normal desktop lifecycle emits its first forecast only after
the first draw has been stored. With one observed draw, there is no predecessor
with a known successor, so all scores remain zero; every candidate then shows
**Distant**, **Average distance 0.00**, and **Support draws 0**.

Equal scores pass through the common tie-break. After the first draw, unseen
numbers have a larger current gap than its six observed numbers, so the first
EMD Top-6 is a deterministic gap-and-number fallback rather than an
Earth-Mover-informed prediction.

After two draws, one transition is eligible. Its successor is the second draw,
so those six numbers all receive score 1 and the other 43 receive score 0. The
model becomes a progressively broader kernel aggregate as more completed
transitions accumulate.

## Causal walk-forward lifecycle

For target draw {math}`D_{t+1}`, the production sequence is:

1. The application evaluates the previously stored forecast for {math}`D_t`
   only after {math}`D_t` is observed.
2. The observed values of {math}`D_t` are sorted and appended to the EMD draw
   history.
3. {math}`D_t` becomes the query; all pairs
   {math}`D_i\rightarrow D_{i+1}` with {math}`i<t` are already complete.
4. Distances from {math}`D_t` to each predecessor {math}`D_i`, kernel weights,
   successor evidence, detail fields, and scores are calculated.
5. The common tie-break forms and stores the Top-6 forecast for
   {math}`D_{t+1}`.

The unknown target cannot enter its own query, transition outcomes, weights, or
scores. Appending later draws cannot alter an EMD forecast already generated
from an earlier history prefix. No separate fitting step or mutable coefficient
update occurs in `train`; the strategy state is its chronological list of
completed sorted draws.

## Ranking and tie-breaking

Candidates are ordered by:

1. larger maximum-scaled EMD score;
2. larger zero-based current gap when scores tie;
3. smaller number when both score and current gap tie.

A number in the latest reference draw has current gap 0. Current gap is not part
of the transport distance, kernel weight, or raw evidence; it only resolves
exact score ties. The first six candidates after all three rules form the
Top-6.

## Interpreting the application fields

| Field | Meaning |
|---|---|
| Score | {math}`u_n/\max_m u_m`, shown as a relative percentage. It is not a calibrated probability. |
| Distance label | Overlap, Near, Close, Middle, Far, or Distant from the exact band containing {math}`\bar d_n`; unsupported candidates use Distant. |
| Average distance | Candidate-specific kernel-weighted mean predecessor distance, displayed to two decimals. |
| Support draws | Unweighted count of historical successor draws that contain the candidate. |
| Gap | Zero-based number of completed draws since the candidate last appeared; used only for score ties. |
| Rank | Position in the complete 1–49 ordering after score, gap, and number rules. |
| Top-6 membership | Whether the candidate occupies one of the first six ranks. |

Two candidates can have identical support counts but different scores because
their supporting transitions have different distances. Conversely, a candidate
with more support draws can rank below one with fewer but more heavily weighted
supports.

## Core mathematical and statistical concepts

- **Empirical distributions:** each draw is represented by six equal point
  masses on the discrete number line.
- **One-dimensional optimal transport:** monotone matching of sorted order
  statistics gives the exact first Wasserstein distance.
- **Analogue transitions:** distances are calculated on predecessors while
  evidence is taken from their observed successor draws.
- **Kernel weighting:** {math}`1/(1+d)` continuously reduces, but never removes,
  the influence of a historical transition.
- **Kernel-weighted frequency:** raw evidence is a weighted historical count of
  successor membership.
- **Maximum normalization:** dividing by the largest raw evidence creates a
  relative 0–1 score without probabilistic calibration.
- **Hypergeometric overlap:** standard Top-6 efficacy uses the null overlap
  distribution for two six-element subsets of 49.

If all historical distances happened to be equal, every transition would have
the same weight and EMD would reduce to ranking lifetime appearances over draws
{math}`D_2,\ldots,D_h`. Its distinct behavior comes specifically from the
association between predecessor similarity and successor membership.

## Endpoint diagnostic

After all 771 repository draws, the latest-query state has:

| Diagnostic | Endpoint value |
|---|---:|
| Stored draw vectors | 771 |
| Eligible historical transitions | 770 |
| Minimum predecessor distance | 1.500 |
| Mean predecessor distance | 11.055 |
| Maximum predecessor distance | 24.167 |
| Sum of kernel weights | 73.747 |
| Kish-style effective weight count | 635.101 |
| Candidates with zero support | 0 |

The effective weight count is a diagnostic derived as

```{math}
N_{\mathrm{eff}}=
\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
```

Production does not use this diagnostic to select transitions or change scores.
Its large endpoint value illustrates that the EMD engine is a broad historical
kernel average, not a sparse nearest-neighbor lookup. The endpoint values depend
on the exact dataset and latest draw and will change when a new draw is added.

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
| Full replay | 770 | 592 | 0.768831 | 565.714 |
| Validation, target draws 121–520 | 400 | 316 | 0.790000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 187 | 0.748000 | 183.673 |

The benchmark's latest 250-target comparison slice, target draws 522–771,
records 186 hits or 0.744000 per target. The production gap-and-number tie-break
is included in every figure.

These are retrospective measurements from a dataset used during broader model
development and strategy comparison. In particular, the holdout result is only
slightly above the theoretical random mean. The table does not establish
statistical significance, calibration, an independent replication, or a future
advantage.

## Limitations and responsible interpretation

- **Analogue assumption:** similar sorted draws are not known to imply similar
  successors in a fair lottery.
- **One-dimensional representation:** sorted positions retain location and
  spread implicitly but omit many other summaries; adding such summaries would
  define a different model.
- **All-history dilution:** every eligible transition receives positive weight,
  so numerous distant states can collectively dominate a few close states.
- **No recency model:** an old and a new transition at the same distance receive
  identical weight.
- **Kernel choice:** {math}`1/(1+d)` and the added 1 are fixed engineering
  choices, not estimated or probabilistically justified by production.
- **Overlapping evidence:** adjacent predecessor-successor pairs share draws and
  are not independent samples.
- **Query reuse:** all weights are calculated relative to the same latest draw,
  which further couples the candidate evidence.
- **Frequency confounding:** commonly occurring successor numbers accumulate
  evidence under many distances; EMD does not subtract a per-number base rate.
- **No prior or uncertainty interval:** sparse early histories can produce
  extreme normalized scores without shrinkage or an uncertainty estimate.
- **Maximum-scaling information loss:** the score hides absolute evidence mass
  and cannot be compared directly across target draws.
- **Diagnostic-band ambiguity:** a per-number weighted average can hide a mix of
  very close and very distant supporting transitions.
- **Deterministic cold start and ties:** gap and number rules can determine the
  Top-6 when transport scores are equal.
- **Dataset and selection dependence:** historical results can reflect chance,
  tuning elsewhere in the strategy collection, or repeated comparisons.
- **No guaranteed predictability:** historical geometric similarity does not
  demonstrate a causal or stable lottery mechanism.

Use EMD as an auditable similarity-weighted ranking, not as evidence that earth
mover geometry predicts future random draws.

## Implementation map

| Responsibility | Production location |
|---|---|
| Strategy identifier, ordering, dependencies, and short-name registry | `src/rand_ai/strategy_prediction.py`, `STRATEGY_IDS`, dependency tables, and strategy-name constants |
| Distance-band labels | `src/rand_ai/strategy_prediction.py`, `_EARTH_MOVER_BUCKETS` |
| Stored sorted draw vectors | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` and `_StrategyState.remember` |
| Closed-form one-dimensional transport distance | `src/rand_ai/strategy_prediction.py`, `_StrategyState._earth_mover_distance` |
| Diagnostic band mapping | `src/rand_ai/strategy_prediction.py`, `_StrategyState._earth_mover_bucket` |
| Transition weighting, raw evidence, maximum scaling, support, and details | `src/rand_ai/strategy_prediction.py`, `_StrategyState._earth_mover_scores` |
| Final gap/number tie-break, Top-6 construction, efficacy, and causal orchestration | `src/rand_ai/strategy_prediction.py`, `_ranking_from_scores`, `_strategy`, and `build_prediction_suites` |
| Desktop serialization | `src/rand_ai/gui_bridge.py`, `_strategy_payload` and `_suite_payload` |
| Default-enabled plugin registration and display label | `web/electron/main.cjs` |
| Settings description, family, color, names, and detail rendering | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Empty-history, distance-band, disabled-dependency, ranking, and serialization coverage | `tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py` |
| Fixed comparative replay record | `reports/recurrence_dynamics_benchmark.md` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 scores, ranks, current gaps, detail strings,
Top-6 numbers, and standard completed efficacy. EMD has no separate experimental
evidence metadata object; its historical performance is carried through the
application's common efficacy fields.
