(svc-recurrence-hybrid)=
# SVC–Recurrence Hybrid

## Introduction

**SVC–Recurrence Hybrid** is the production strategy with identifier
`svc_recurrence_hybrid` and short engine name **SRH**. It is a
default-disabled experimental member of the **Ensembles & Coverage** family.

SRH combines the complete 1–49 rankings from two structurally different
sources:

- [Support Vector Classifier](support-vector-classifier.md), an online linear
  candidate classifier using gap and frequency-residual features; and
- [Recurrence Dynamics](recurrence-dynamics.md), a nearest-analogue model over
  normalized three-draw recurrence states.

The strategy converts both rankings to a common linear strength scale, then
weights them by cumulative, completed walk-forward Top-6 efficacy. Neither
source implementation is changed.

```{admonition} Rank pooling, not probability pooling
:class: important

SRH averages rank strengths. It does not average SVC margins, Recurrence
scores, calibrated probabilities, or evidence statuses. Its displayed score
is a relative ensemble strength, not a chance that a number will be drawn.
```

## Scope and role

SVC–Recurrence Hybrid asks:

> Can two causally produced rankings with different representations cancel
> some source-specific ranking error, while completed source Top-6 history
> determines their relative influence?

It produces one complete ranking. The first six numbers form the Top-6 used by
prediction grids, audits, effectiveness histories, comparisons, portfolios,
exports, and Possible Draw.

The strategy deliberately contains no:

- raw-score mixing;
- agreement bonus;
- source quota;
- recent efficacy window;
- streak or regime detector;
- adaptive learning rate;
- residual third source; or
- ticket-shape correction.

Adaptation is limited to the two cumulative source weights. The score formula
and 24-draw prior remain fixed.

## Hidden source dependencies

Selecting only SRH recursively activates `svc` and `recurrence_dynamics`.
Those sources calculate and update their full production states, but only SRH
is serialized unless the user explicitly selects the sources too.

The source engines remain independent:

- SVC retains its online 11-feature hinge-loss learner, class weighting,
  margin scaling, and ranking tie-break.
- Recurrence Dynamics retains its three-draw embedding, analogue selection,
  prior, score construction, and causal evidence tracker.

SRH reads only their complete permutations. It does not feed its result back
into either source.

## Why ranks are used

The source score scales have different meanings:

- SVC begins with a linear margin and min–max scales the 49 current margins.
- Recurrence Dynamics produces analogue-weighted candidate evidence under its
  own prior and distance model.

A numerical value such as 0.8 is therefore not commensurate across the two
engines. Mixing raw scores would let source calibration and spread influence
the ensemble in addition to ordering.

SRH discards source magnitude and retains only ordinal position. Let

```{math}
r_{R,t}(n),r_{S,t}(n)\in\{1,\ldots,49\}
```

be the Recurrence and SVC ranks for number (n\) in the forecast of target
draw (t\).

## Rank-strength transformation

Every source rank becomes

```{math}
q(r)=\frac{49-r}{48}.
```

Thus:

| Source rank | Rank strength |
|---:|---:|
| 1 | 1.000000 |
| 6 | 0.895833 |
| 25 | 0.500000 |
| 49 | 0.000000 |

Adjacent ranks differ by exactly

```{math}
\frac1{48}=0.0208333.
```

Define

```{math}
q_{R,t}(n)=\frac{49-r_{R,t}(n)}{48},
\qquad
q_{S,t}(n)=\frac{49-r_{S,t}(n)}{48}.
```

Because each source is a complete permutation, each strength vector has mean
0.5 and sum 24.5.

## Completed Top-6 source hits

For each already observed target draw (i\), let (D_i\) be its six actual
numbers and define source hits

```{math}
h_{R,i}=
\left|\operatorname{Top6}(R_i)\cap D_i\right|,
```

```{math}
h_{S,i}=
\left|\operatorname{Top6}(S_i)\cap D_i\right|.
```

Each value lies in ({0,1,\ldots,6}\). After (E_t\) completed forecast
targets, production stores

```{math}
H_{R,t}=\sum_{i=1}^{E_t}h_{R,i},
\qquad
H_{S,t}=\sum_{i=1}^{E_t}h_{S,i}.
```

Only source Top-6 membership matters to weight adaptation. A source's ranks
7–49 affect the hybrid number score, but they do not affect its quality until
a number enters that source's Top-6.

## Neutral 24-draw prior

For a fixed six-number forecast under the uniform null, expected hits per draw
are

```{math}
\mu_0=6\times\frac6{49}=\frac{36}{49}=0.734694.
```

Production uses a neutral prior equivalent to 24 draws at that mean. Its
pseudo-hit total is

```{math}
A=24\mu_0
=24\times\frac{36}{49}
=17.632653.
```

The smoothed source qualities are

```{math}
Q_{R,t}=\frac{H_{R,t}+A}{E_t+24},
\qquad
Q_{S,t}=\frac{H_{S,t}+A}{E_t+24}.
```

Quality is measured in mean Top-6 hits per completed target. It is not a
probability bounded by 1; one target can contribute up to six hits.

The prior prevents an early one-draw difference from giving either source an
extreme share. It also ensures both qualities are positive before any source
has been evaluated.

## Adaptive source weights

The qualities are normalized:

```{math}
w_{R,t}=\frac{Q_{R,t}}{Q_{R,t}+Q_{S,t}},
\qquad
w_{S,t}=\frac{Q_{S,t}}{Q_{R,t}+Q_{S,t}}.
```

Therefore

```{math}
w_{R,t}+w_{S,t}=1.
```

Because both sources always have the same evaluated-history length, their
quality denominators cancel in the ratio:

```{math}
w_{R,t}=
\frac{H_{R,t}+A}{H_{R,t}+H_{S,t}+2A},
```

```{math}
w_{S,t}=
\frac{H_{S,t}+A}{H_{R,t}+H_{S,t}+2A}.
```

This makes the adaptation cumulative. There is no decay or rolling window, so
a recent streak contributes only its hits to the entire completed history.

### Cold-start weights

Before any completed target,

```{math}
H_{R,0}=H_{S,0}=E_0=0,
```

so

```{math}
Q_{R,0}=Q_{S,0}=\frac{36}{49},
\qquad
w_{R,0}=w_{S,0}=50\%.
```

Whenever the cumulative source hit totals are equal, weights return to exactly
50/50, regardless of the path by which those totals were reached.

## Exact hybrid score

For every number, production calculates

```{math}
S_t(n)=
w_{R,t}q_{R,t}(n)
+w_{S,t}q_{S,t}(n).
```

Expanded in source ranks:

```{math}
S_t(n)=
w_{R,t}\frac{49-r_{R,t}(n)}{48}
+w_{S,t}\frac{49-r_{S,t}(n)}{48}.
```

This is a convex combination, so

```{math}
0\le S_t(n)\le1.
```

No final min–max scaling is applied. Since both source strength vectors have
mean 0.5,

```{math}
\frac1{49}\sum_{n=1}^{49}S_t(n)=0.5
```

for every target and every valid weight pair. The score distribution's mean is
fixed; only its ordering and concentration vary.

## Ranking and tie-break

Numbers are ordered by:

1. larger hybrid score;
2. larger current gap; then
3. smaller number.

Equal scores can occur even though each source rank is unique. At 50/50, any
two numbers with the same sum of source ranks have the same hybrid score.

In the extreme case where one ranking is the exact reverse of the other and
weights are 50/50, every number receives 0.5. The result then falls entirely
to current gap and number. Rank averaging reduces source-scale mismatch, but
it does not guarantee a uniquely informative consensus.

## How error cancellation can occur

Rank pooling changes which mistakes dominate:

- **Shared support:** a number ranked highly by both sources usually rises
  because both terms are large.
- **One-source extreme:** a very high rank in one source can be moderated by a
  weak rank in the other.
- **Near-top consensus:** a number just outside both source Top-6 lists can
  enter the hybrid Top-6 when its combined strength exceeds one-sided picks.
- **Quality adjustment:** a source with more completed cumulative hits receives
  a larger share on subsequent targets.

This is the mechanism commonly described as error cancellation. It is a
property of combining imperfect rankings, not proof that the errors are
independent, low-correlated, or predictable. SVC and Recurrence can still
share dataset effects, gap-related ordering, tie-breaks, and historical
selection bias.

## Causal lifecycle and leakage protection

For target draw (t\), production follows this sequence:

1. retain the Recurrence and SVC rankings issued after draw (t-1\);
2. calculate SRH weights using source hits completed only through draw
   (t-1\);
3. build and retain the SRH ranking for target (t\);
4. when draw (t\) occurs, run the fixed source-training and hybrid-tracking
   update pass;
5. compare the saved source Top-6 lists—not rebuilt rankings—with the actual
   numbers, add those completed hits, and increment evaluated history;
6. finish remembering the completed draw in the underlying source states; and
7. use newly built source rankings and updated cumulative qualities for target
   (t+1\).

The concrete method order trains some source state in the same overall update
pass, but SRH efficacy always reads copied pending rankings produced before the
actual target existed. The target cannot alter the weight or source ranking
used for its own forecast.

Appending future draws therefore leaves earlier SRH forecasts unchanged.
Tests compare identical history prefixes in shorter and longer datasets to
enforce that invariant.

## Source and hybrid efficacy are separate

SRH maintains source hit totals solely to calculate (w_R\) and (w_S\). The
application's standard efficacy tracker separately records the hybrid's own
Top-6 results for charts, audits, comparisons, and exports.

The hybrid's own completed hits do not affect either source weight. Otherwise,
the strategy would be adapting to a third quantity not present in its stated
formula.

Recurrence Dynamics also exposes experimental evidence metadata based on its
own analogue and causal-forecast diagnostics. SRH does not copy that metadata:

- `recurrence_dynamics` may carry a `StrategyEvidence` object;
- `svc_recurrence_hybrid` carries standard `StrategyEfficacy`; and
- SRH's `evidence` field is deliberately `None`.

This prevents categorical Recurrence evidence from being mistaken for evidence
about the hybrid blend.

## Interpreting application fields

Every number carries three SRH-specific detail lines:

- **Recurrence weight; rank; Top 6 yes/no** reports the current normalized
  weight and complete source position.
- **SVC weight; rank; Top 6 yes/no** reports the equivalent SVC information.
- **Effectiveness history … completed draws** reports (E_t\), the number of
  source forecasts already resolved before this target.

The standard payload adds hybrid score, current gap, final rank, Top-6
membership, and completed hybrid efficacy.

Weights are displayed to one decimal percentage point. Source ranks and Top-6
membership are exact. The displayed score is the rank-strength combination,
not a hit probability or confidence level.

## Endpoint diagnostic

After all 771 repository draws, the target-772 forecast has 770 completed
source evaluations:

| Source | Cumulative hits | Smoothed quality | Weight |
|---|---:|---:|---:|
| Recurrence Dynamics | 625 | 0.809361 | 50.0% |
| Support Vector Classifier | 625 | 0.809361 | 50.0% |

The resulting Top-6 is:

| Hybrid rank | Number | Score | Gap | Recurrence rank | SVC rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.968750 | 8 | 4 | 1 |
| 2 | 4 | 0.958333 | 4 | 3 | 3 |
| 3 | 2 | 0.906250 | 0 | 1 | 10 |
| 4 | 9 | 0.864583 | 5 | 7 | 8 |
| 5 | 10 | 0.802083 | 1 | 16 | 5 |
| 6 | 30 | 0.729167 | 2 | 2 | 26 |

Numbers 8 and 4 receive strong agreement. Number 9 is outside both source
Top-6 lists, but ranks 7 and 8 combine into hybrid rank 4. Numbers 2, 10, and
30 demonstrate one-sided support moderated by the other source.

The next two candidates, 40 and 16, both score 0.697917 because their source
rank sums are equal. Their current gaps are 16 and 0, so the standard gap
tie-break ranks 40 first.

These values are deterministic endpoint diagnostics, not evidence that the
same source balance or number ordering will persist.

## Top-6 statistical reference

For any fixed six-number forecast compared with an independent uniformly
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

This expectation is used by the weight prior. It is also the theoretical
reference for the hybrid's standard Top-6 efficacy, but the two roles remain
separate in state.

## Leakage-free replay reference

The fixed benchmark replays the repository's 771 chronological YAML draws,
creating 770 target forecasts:

| Slice | Strategy | Hits | Mean hits per target | Random expected total |
|---|---|---:|---:|---:|
| Full 770 | **SVC–Recurrence Hybrid** | **634** | **0.823377** | 565.714 |
| Full 770 | Recurrence Dynamics | 625 | 0.811688 | 565.714 |
| Full 770 | SVC | 625 | 0.811688 | 565.714 |
| Validation 121–520 | **SVC–Recurrence Hybrid** | **325** | **0.812500** | 293.878 |
| Validation 121–520 | Recurrence Dynamics | 342 | 0.855000 | 293.878 |
| Validation 121–520 | SVC | 335 | 0.837500 | 293.878 |
| Holdout 521–770 | **SVC–Recurrence Hybrid** | **204** | **0.816000** | 183.673 |
| Holdout 521–770 | Recurrence Dynamics | 199 | 0.796000 | 183.673 |
| Holdout 521–770 | SVC | 187 | 0.748000 | 183.673 |

The latest 250-target slice, target draws 522–771, records 205 hybrid hits,
197 Recurrence hits, and 188 SVC hits.

The hybrid exceeds both sources over the full replay and holdout, but trails
both on the earlier validation slice. That reversal is why the strategy
remains default-disabled and experimental. The formula was selected after
inspection of this dataset, so these results are regression evidence rather
than independent confirmation. They do not establish statistical significance,
stable future lift, or predictability.

## Core mathematical and statistical concepts

- **Ordinal normalization:** source-specific score scales are replaced by a
  common ((49-r)/48\) rank scale.
- **Convex combination:** non-negative source weights sum to one and keep every
  hybrid score between 0 and 1.
- **Neutral-prior smoothing:** 24 null-mean pseudo-draws stabilize early
  cumulative source qualities.
- **Walk-forward adaptation:** only completed pending Top-6 outcomes update
  future weights.
- **Cumulative estimation:** all resolved source targets retain equal influence;
  there is no decay or recent window.
- **Rank aggregation:** shared support, near-top consensus, and moderated
  one-source extremes can change the Top-6.
- **Error cancellation:** complementary ordinal errors can partially offset,
  although independence is neither assumed nor demonstrated.
- **Prefix invariance:** later observations cannot alter an already issued
  forecast.
- **Hypergeometric overlap:** the null expectation for source quality and
  standard efficacy is (36/49\) hits per target.

## Limitations and responsible interpretation

- **Selection-biased design:** the source pair, rank blend, prior, and reported
  slices were fixed after historical exploration.
- **Validation underperformance:** SRH trails both sources on target draws
  121–520 despite stronger full and holdout totals.
- **Only two sources:** one shared failure mode can affect the entire blend.
- **Correlated errors:** both sources use the same draw history and common
  gap/number tie-break; their errors are not independent trials.
- **Top-6-only weighting:** ranks 7–49 influence scores but cannot improve a
  source's quality until they enter its Top-6.
- **Cumulative inertia:** old source hits never expire, so the weights react
  increasingly slowly to a genuine change.
- **No uncertainty interval:** normalized point qualities do not express
  confidence in the source-weight difference.
- **Rank information loss:** raw margin, analogue distance, support, and score
  separation are discarded.
- **Linear rank spacing:** the difference between ranks 1 and 2 is treated the
  same as the difference between ranks 48 and 49.
- **Tie-break influence:** current gap can decide equal hybrid scores without
  belonging to the weight-quality formula.
- **Uncalibrated output:** a score such as 0.86 is not an 86% occurrence
  probability.
- **Dependency sensitivity:** any SVC or Recurrence implementation change can
  change source ranks, completed hit paths, future weights, and replay totals.
- **Experimental evidence separation:** Recurrence evidence neither validates
  nor weights the hybrid.
- **No guaranteed predictability:** historical error cancellation can be a
  chance property that disappears on untouched draws.

Use SRH as an auditable experiment in causal rank aggregation, not as proof
that combining two historical models creates reliable knowledge of future
lottery outcomes.

## Implementation map

The production blend is implemented in
`src/rand_ai/strategy_prediction.py`:

- `_SVC_RECURRENCE_EXPERT_IDS` fixes source order to Recurrence Dynamics and
  SVC;
- `_SVC_RECURRENCE_PRIOR_DRAWS` fixes the neutral prior at 24 draws;
- `_STRATEGY_DEPENDENCIES` activates both hidden source engines;
- `_StrategyState` stores cumulative source hits, common evaluated length, and
  copied pending rankings;
- `_train_svc_recurrence_effectiveness` evaluates only saved source Top-6 lists
  after the target occurs;
- `_svc_recurrence_weights` calculates prior-smoothed qualities and normalized
  weights;
- `_svc_recurrence_scores` converts complete ranks to strength, blends scores,
  records the next pending rankings, and builds detail strings;
- `_ranking_from_scores` applies the shared score/gap/number ordering; and
- `build_strategies` serializes engine `SRH` without Recurrence evidence.

Registration and presentation are distributed across:

- `src/rand_ai/gui_bridge.py` for default-disabled state, cache, standard
  efficacy, and payload serialization;
- `web/electron/main.cjs` for plugin registration;
- `web/src/lib/strategyFamilies.ts` and `strategyColors.ts` for the
  **Ensembles & Coverage** family and color;
- Settings, prediction, comparison, effectiveness, portfolio, export, and
  Possible Draw views for selection and display.

`scripts/benchmark_svc_recurrence_hybrid.py` reproduces the fixed full,
validation, and holdout comparison recorded in
`reports/svc_recurrence_hybrid_benchmark.md`.

Tests in `tests/test_strategy_prediction.py` verify exact 50/50 cold start,
prior arithmetic, directional weight updates, hidden dependencies, selected-only
output, causal prefix invariance, complete ranking details, and preservation of
both sources. `tests/test_gui_bridge.py` verifies default-disabled state,
standard efficacy serialization, and the absence of copied Recurrence evidence.
