(srph-residual-diversity-hybrid)=
# SRPH Residual Diversity Hybrid

## Introduction

**SRPH Residual Diversity Hybrid** is the production strategy with identifier
`srph_residual_diversity_hybrid` and short engine name **SRD**. It is a
default-disabled experimental member of the **Ensembles & Coverage** family.

SRD treats [SVC–Recurrence–Proximity Hybrid](svc-recurrence-proximity-hybrid.md)
as its base. It continuously replays four fixed 90/10 counterfactual blends,
each injecting one candidate ranking into the exact SRPH score. A guarded
selector uses only completed counterfactual Top-6 history:

- if the best candidate blend has strictly higher cumulative prior-smoothed
  quality than SRPH, SRD emits that blend;
- otherwise SRD reproduces SRPH.

```{admonition} A selector over counterfactual blends
:class: important

SRD does not compare raw candidate efficacy with SRPH. It compares SRPH with
the actual Top-6 ranking that each 90/10 blend would have produced. Candidate
quality therefore means **counterfactual blend quality**.
```

## Scope and role

SRPH Residual Diversity Hybrid asks:

> Has any frozen 10% residual injection produced more completed cumulative
> Top-6 hits than unmodified SRPH, using only forecasts that existed before
> their targets occurred?

It produces a complete 1–49 ranking. Its first six numbers form the Top-6 used
by prediction grids, audits, effectiveness histories, comparisons, portfolios,
exports, and Possible Draw.

The strategy is a guarded shadow experiment, not a replacement for SRPH. It
adds no:

- recent efficacy window;
- adaptive residual percentage;
- agreement bonus;
- regime detector;
- source quota;
- joint-ticket optimization; or
- use of residual diagnostic lift in selection.

The candidate pool, 10% share, 24-draw prior, strict fallback rule, and stable
tie order are frozen.

## Recursive hidden dependencies

Selecting only SRD activates the following dependency tree:

```text
srph_residual_diversity_hybrid
├── svc_recurrence_proximity_hybrid
│   ├── svc_recurrence_hybrid
│   │   ├── recurrence_dynamics
│   │   └── svc
│   └── proximity
├── freshness
├── emd
├── bayesian
└── doublet_triplet_markov
```

All dependencies can calculate invisibly. Only SRD is serialized unless the
user explicitly requests source strategies too.

SRD does not change SRPH or any candidate engine. Downstream selection has no
feedback path into their training, ranks, or efficacy.

## Frozen candidate pool and tie order

The residual candidates are fixed in this exact order:

| Stable order | Strategy ID | Displayed label |
|---:|---|---|
| 1 | `freshness` | Freshness |
| 2 | `emd` | EMD |
| 3 | `bayesian` | Bayesian |
| 4 | `doublet_triplet_markov` | Doublet/Triplet Markov |

When two candidate counterfactuals have exactly equal quality, the earlier
candidate in this table wins the candidate tie. Stable ordering does not
override the fallback rule: if that winning quality merely equals SRPH, SRD
still falls back.

The pool was chosen after exploratory comparisons on the repository history.
It is not an exhaustive or independently selected set of residual models.

## Base score and candidate rank strengths

Let

```{math}
B_t(n)=S_{\mathrm{SRPH},t}(n)
```

be the exact SRPH score for number (n\) at target (t\). SRD receives this
score before converting SRPH's final ranking into anything else.

For candidate (c\), let

```{math}
r_{c,t}(n)\in\{1,\ldots,49\}
```

be its complete source rank. Production converts it to strength:

```{math}
q_{c,t}(n)=\frac{49-r_{c,t}(n)}{48}.
```

Raw Freshness probabilities, EMD distances, Bayesian posteriors, and
Doublet/Triplet scores are discarded. Only their complete rank orders enter
the counterfactuals.

## Four exact counterfactual scores

For every candidate (c\), production builds a complete score vector:

```{math}
C_{c,t}(n)=0.90B_t(n)+0.10q_{c,t}(n).
```

Each counterfactual is then ranked by:

1. larger counterfactual score;
2. larger current gap; then
3. smaller number.

Let

```{math}
T_{c,t}=\operatorname{Top6}(C_{c,t})
```

be the six-number result of the complete 90/10 ranking. Also let

```{math}
T_{0,t}=\operatorname{Top6}(B_t)
```

be unmodified SRPH's Top-6.

All four counterfactuals are built and retained at every target, including
when the selector falls back. This is necessary to evaluate what each fixed
blend actually would have done.

## Score invariants

SRPH score (B_t\) has mean 0.5 and sum 24.5 across numbers. Every candidate
rank-strength vector has the same invariants. Therefore every counterfactual
satisfies

```{math}
0\le C_{c,t}(n)\le1,
```

```{math}
\frac1{49}\sum_{n=1}^{49}C_{c,t}(n)=0.5,
```

```{math}
\sum_{n=1}^{49}C_{c,t}(n)=24.5.
```

No final min–max scaling is applied. On fallback, SRD copies the exact SRPH
score dictionary; with the same gap/number tie-break, its complete ranking is
bit-for-bit identical to SRPH.

## Completed counterfactual tracking

For each completed target (i\) with actual set (D_i\), production records

```{math}
h_{0,i}=|T_{0,i}\cap D_i|
```

for SRPH and

```{math}
h_{c,i}=|T_{c,i}\cap D_i|
```

for every candidate blend.

After (E_t\) completed predictions, cumulative hits are

```{math}
H_{0,t}=\sum_{i=1}^{E_t}h_{0,i},
\qquad
H_{c,t}=\sum_{i=1}^{E_t}h_{c,i}.
```

The candidate tracker never substitutes the raw candidate Top-6 for
(T_{c,i}\). It always evaluates the exact 90/10 result, including score ties
and the production gap/number tie-break.

## Neutral 24-draw quality prior

The uniform Top-6 null expectation is

```{math}
\mu_0=\frac{36}{49}=0.734694
```

hits per target. With prior mass

```{math}
A=24\mu_0=17.632653,
```

the displayed qualities are

```{math}
Q_{0,t}=\frac{H_{0,t}+A}{E_t+24},
```

```{math}
Q_{c,t}=\frac{H_{c,t}+A}{E_t+24}.
```

Quality is mean Top-6 hits per completed target, not an individual-number
probability. It can exceed 1 because a prediction can hit several numbers in
one draw.

### What the common prior does—and does not do

All five trackers share the same (A\) and (E_t\). Consequently,

```{math}
Q_{c,t}>Q_{0,t}
\iff
H_{c,t}>H_{0,t}.
```

Candidate quality ordering is likewise identical to cumulative hit ordering.
The prior makes early reported means less extreme, but it does not create an
additional activation margin. One cumulative extra hit is sufficient for a
candidate counterfactual to be strictly better.

At cold start, every quality equals (36/49\), so the selector necessarily
falls back to SRPH.

## Exact selector and fallback rule

First choose the candidate

```{math}
c_t^*=\arg\max_c Q_{c,t},
```

using the frozen candidate order to resolve ties. Then production emits

```{math}
S_{\mathrm{SRD},t}(n)=
\begin{cases}
C_{c_t^*,t}(n),&Q_{c_t^*,t}>Q_{0,t},\\[4pt]
B_t(n),&Q_{c_t^*,t}\le Q_{0,t}.
\end{cases}
```

The comparison is strictly greater:

- positive cumulative counterfactual lift activates 90/10;
- zero or negative lift reproduces SRPH exactly.

There is no confidence interval, minimum hit margin, persistence requirement,
or penalty for choosing the best of four histories.

## Effective weights

On fallback, user-facing effective weights are:

```{math}
W_{\mathrm{SRPH}}=100\%,
\qquad W_c=0\%.
```

When a candidate is selected:

```{math}
W_{\mathrm{SRPH}}=90\%,
\qquad W_c=10\%.
```

If desired, the selected state can be expanded through SRPH. For example, when
SRPH internally uses 37.5% Recurrence, 37.5% SVC, and 25% Proximity, the final
selected blend has:

```{math}
33.75\%\text{ Recurrence},
\quad33.75\%\text{ SVC},
\quad22.5\%\text{ Proximity},
\quad10\%\text{ residual candidate}.
```

The application details show only the direct 90/10 decomposition, not this
expanded source tree.

## Unique additions, displacements, and net lift

For each completed counterfactual, production also records:

```{math}
a_{c,i}=
|D_i\cap(T_{c,i}\setminus T_{0,i})|
```

as **unique added hits**, and

```{math}
d_{c,i}=
|D_i\cap(T_{0,i}\setminus T_{c,i})|
```

as **displaced SRPH hits**.

Their cumulative difference is

```{math}
L_{c,t}=
\sum_{i=1}^{E_t}a_{c,i}
-\sum_{i=1}^{E_t}d_{c,i}.
```

Because both Top-6 sets have size six,

```{math}
H_{c,t}-H_{0,t}=L_{c,t}.
```

These counts explain where the counterfactual hit difference came from. The
selector nevertheless uses only (Q_c\), which is equivalent to cumulative
net hits; it does not separately reward additions or penalize turnover.

## Causal lifecycle and leakage protection

For target draw (t\), production follows this sequence:

1. build SRPH and all four raw candidate rankings from history through
   (t-1\);
2. construct the four exact counterfactual score vectors and rankings;
3. select or fall back using qualities completed only through target
   (t-1\);
4. retain SRPH and all counterfactual Top-6 lists as pending state;
5. when draw (t\) occurs, compare those saved lists with the actual numbers,
   update hits, additions, displacements, and common history length;
6. complete the fixed source-training and history-update pass; and
7. use the updated trackers and source states only for target (t+1\).

The target cannot choose its own residual candidate, update the quality used
for its own selector decision, or change its own counterfactual rankings.

Appending later draws leaves every earlier SRD output unchanged. Tests also
verify that enabling SRD does not mutate the SRPH or candidate rankings it
consumes.

## Efficacy and evidence separation

SRD has three layers of retrospective accounting:

- internal SRPH/counterfactual qualities drive the selector;
- addition/displacement totals provide diagnostics; and
- standard SRD efficacy reports the actually emitted SRD Top-6 in charts and
  exports.

Only the first layer selects. Standard SRD efficacy does not feed back into
the selector, and diagnostics do not add a second objective.

Recurrence-specific experimental evidence remains attached only to standalone
Recurrence Dynamics. SRD serializes standard efficacy and `evidence=None`.

## Interpreting application fields

Every number carries seven SRD-specific detail lines:

- **Selector selected …** or **Selector fallback to SRPH** reports the current
  decision.
- **SRPH effective weight; rank; Top 6 yes/no** reports the direct base share
  and raw SRPH rank.
- **Candidate effective weight; rank; Top 6 yes/no** reports the selected
  candidate's direct share and raw candidate rank.
- **Quality SRPH …; candidate …** compares the base with the selected
  candidate's **counterfactual blend** quality.
- **Candidate qualities …** lists all four counterfactual qualities.
- **Cumulative residual lift … hits (… added; … displaced)** reports
  (L_c\) and its components.
- **Selector history … completed draws** reports (E_t\).

The candidate rank shown in details is the raw candidate source rank. The
candidate quality is the 90/10 counterfactual quality. They describe different
objects and should not be read as one model's raw efficacy.

The standard payload separately contains emitted SRD score, current gap, final
rank, Top-6 membership, and completed standard efficacy. Scores and qualities
are not calibrated individual-number probabilities.

## Endpoint diagnostic

After all 771 repository draws, the selector has 770 completed counterfactual
targets:

| Tracked ranking | Hits | Smoothed quality | Added hits | Displaced hits | Net lift vs SRPH |
|---|---:|---:|---:|---:|---:|
| SRPH base | 643 | 0.832031 | — | — | 0 |
| 90/10 Freshness | 614 | 0.795507 | 73 | 102 | -29 |
| 90/10 EMD | 629 | 0.814399 | 46 | 60 | -14 |
| 90/10 Bayesian | 620 | 0.803064 | 56 | 79 | -23 |
| 90/10 Doublet/Triplet Markov | 645 | 0.834550 | 67 | 65 | +2 |

The Doublet/Triplet counterfactual is strictly best and exceeds SRPH, so
target draw 772 uses 90% SRPH and 10% Doublet/Triplet Markov.

| Rank | Number | SRD score | Gap | SRPH rank | Candidate rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.901302 | 8 | 1 | 18 |
| 2 | 4 | 0.822396 | 4 | 2 | 39 |
| 3 | 10 | 0.814323 | 1 | 3 | 26 |
| 4 | 40 | 0.754948 | 16 | 6 | 5 |
| 5 | 26 | 0.752604 | 7 | 4 | 14 |
| 6 | 9 | 0.744010 | 5 | 5 | 17 |

The emitted Top-6 set is identical to endpoint SRPH's set
`(8, 4, 10, 26, 9, 40)`; only its order changes. Candidate selection therefore
does not imply that the residual source must replace a Top-6 member on every
target. Here, candidate rank 5 lifts number 40 from SRPH rank 6 to SRD rank 4,
while all six members remain present.

These endpoint values are fitted historical diagnostics, not a prediction that
the selected candidate or +2 cumulative lift will remain favorable.

## Top-6 statistical reference

For a fixed six-number forecast evaluated against an independent uniformly
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

This mean supplies the common quality prior. It does not compensate for
choosing the best of four counterfactual histories.

## Leakage-free replay reference

The fixed production benchmark replays 771 chronological YAML draws, creating
770 target forecasts:

| Slice | Strategy | Hits | Mean hits per target | Random expected total |
|---|---|---:|---:|---:|
| Full 770 | **SRPH Residual Diversity Hybrid** | **642** | **0.833766** | 565.714 |
| Full 770 | SRPH | 643 | 0.835065 | 565.714 |
| Validation 121–520 | **SRPH Residual Diversity Hybrid** | **333** | **0.832500** | 293.878 |
| Validation 121–520 | SRPH | 328 | 0.820000 | 293.878 |
| Holdout 521–770 | **SRPH Residual Diversity Hybrid** | **200** | **0.800000** | 183.673 |
| Holdout 521–770 | SRPH | 205 | 0.820000 | 183.673 |

The latest 250-target slice, target draws 522–771, records 201 SRD hits and 206
SRPH hits.

Across the full 770 emitted forecasts, selector states were:

| State | Forecast count |
|---|---:|
| Fallback to SRPH | 248 |
| Selected Freshness | 0 |
| Selected EMD | 29 |
| Selected Bayesian | 15 |
| Selected Doublet/Triplet Markov | 478 |

SRD gains five hits over SRPH on the validation slice but loses five on the
holdout and one over the full history. The negative holdout prevents promotion.
The results are retrospective and do not establish statistical significance,
stable residual value, or predictability.

## Selection history and experimental status

The candidate pool and 10% weight were selected after exploratory comparisons
on the same dataset. In addition, every target chooses the best cumulative
history among four counterfactuals. This creates a multiple-selection context
that the simple 24-draw prior does not correct.

The complete rule is now frozen for future untouched draws. Validation
improvement is an acceptance regression, not independent evidence that the
selector found a persistent residual source.

## Core mathematical and statistical concepts

- **Counterfactual ranking:** each candidate is evaluated through the exact
  90/10 ranking it would have emitted, not its standalone Top-6.
- **Rank-strength normalization:** candidate positions use the common
  ((49-r)/48\) scale.
- **Convex residual injection:** 90% base plus 10% candidate preserves score
  range and mean.
- **Prior-smoothed cumulative quality:** completed hit totals are expressed as
  means with 24 null pseudo-draws.
- **Strict guarded fallback:** equal or lower best-candidate quality reproduces
  SRPH exactly.
- **Stable argmax:** frozen candidate order makes equal-quality selection
  deterministic.
- **Set difference diagnostics:** added and displaced actual hits decompose
  counterfactual net lift.
- **Walk-forward causality:** saved rankings are scored only after the target
  occurs.
- **Hypergeometric overlap:** all trackers count intersections of predicted
  and actual six-number sets.

## Limitations and responsible interpretation

- **Negative holdout:** SRD trails SRPH by five hits on the nominal holdout.
- **No full-history improvement:** emitted SRD also trails SRPH by one hit over
  all 770 targets.
- **Candidate and weight selection bias:** the pool and 10% share were chosen
  after historical exploration.
- **Best-of-four bias:** selecting the strongest cumulative counterfactual can
  capitalize on noise among several comparisons.
- **Common prior gives no activation margin:** one cumulative extra hit is
  enough to select a candidate, regardless of uncertainty.
- **Cumulative inertia:** old outcomes never expire, so an early advantage can
  dominate many later choices.
- **No uncertainty estimate:** qualities have no confidence or credible
  interval and are treated as exact ordering statistics.
- **Top-6-only selector:** score improvements outside the first six do not
  affect quality until they cross the boundary.
- **Raw/counterfactual distinction:** displayed raw candidate rank is not the
  object whose quality is selected.
- **Diagnostic non-use:** additions and displacements explain lift but do not
  guard activation or penalize turnover.
- **Selection without membership change:** an active 90/10 blend can reorder
  the same Top-6 set and add no current residual coverage.
- **Fixed residual share:** every selected candidate receives exactly 10%,
  whether its quality advantage is one hit or much larger.
- **Correlated sources:** candidates and SRPH reuse the same chronological
  history and may encode overlapping frequency, gap, shape, or sequence
  information.
- **No direct diversity statistic:** the name refers to the candidate role;
  production does not optimize correlation, disagreement, entropy, or distance
  between rankings.
- **Uncalibrated scores:** blend strengths and qualities are not
  individual-number probabilities.
- **Dependency sensitivity:** any source or SRPH change propagates through all
  counterfactual histories and selector decisions.
- **No guaranteed predictability:** positive historical residual lift can be a
  transient random fluctuation.

Use SRD as a frozen audit of whether predefined minority rank injections have
earned cumulative walk-forward selection, not as evidence that residual
diversity reliably predicts future lottery outcomes.

## Implementation map

The production selector is implemented in
`src/rand_ai/strategy_prediction.py`:

- `_SRPH_RESIDUAL_BASE_ID` fixes SRPH as the base;
- `_SRPH_RESIDUAL_CANDIDATE_IDS` fixes candidate order;
- `_SRPH_RESIDUAL_LABELS`, `_SRPH_RESIDUAL_WEIGHT`, and
  `_SRPH_RESIDUAL_PRIOR_DRAWS` define labels, 10%, and 24 draws;
- `_STRATEGY_DEPENDENCIES` recursively activates the full hidden source tree;
- `_StrategyState` stores base/counterfactual hits, unique additions,
  displacements, common history length, and pending rankings;
- `_train_srph_residual_effectiveness` evaluates saved rankings after the
  target occurs and updates all diagnostics;
- `_srph_residual_quality` applies the neutral prior;
- `_srph_residual_scores` builds all counterfactuals, applies gap/number
  ranking, resolves candidate ties, enforces strict fallback, and constructs
  detail strings;
- `_ranking_from_scores` creates the emitted complete ranking; and
- `build_strategies` serializes engine `SRD` without source evidence.

Registration and presentation are distributed across the Python bridge,
Electron plugin registration, strategy family/color registries, Settings,
prediction grids, audits, effectiveness, comparisons, portfolios, exports,
and Possible Draw views. SRD remains excluded from default-enabled IDs and
from Residual Coverage's implicit dependency pool.

`scripts/benchmark_srph_residual_diversity_hybrid.py` reproduces the fixed
slice comparison and selector counts recorded in
`reports/srph_residual_diversity_hybrid_benchmark.md`.

Tests in `tests/test_strategy_prediction.py` verify cold-start fallback,
counterfactual tracking, additions and displacements, strict positive-quality
selection, stable candidate order, hidden dependencies, selected-only output,
prefix invariance, and preservation of source rankings. Bridge tests verify
default-disabled state, engine name, standard efficacy, evidence separation,
and selected-strategy-only serialization.
