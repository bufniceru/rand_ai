(svc-recurrence-proximity-hybrid)=
# SVC–Recurrence–Proximity Hybrid

## Introduction

**SVC–Recurrence–Proximity Hybrid** is the production strategy with identifier
`svc_recurrence_proximity_hybrid` and short engine name **SRPH**. It is a
default-disabled experimental member of the **Ensembles & Coverage** family.

SRPH extends the existing [SVC–Recurrence Hybrid](svc-recurrence-hybrid.md) by
reserving exactly 25% of every number's score for the complete Proximity
ranking. The remaining 75% preserves SRH's leakage-free adaptive split between
Recurrence Dynamics and the Support Vector Classifier.

```{admonition} A frozen three-source rank blend
:class: important

Proximity always receives 25%. Its historical efficacy does not raise or lower
that share. SVC and Recurrence divide only the remaining 75% using SRH's
cumulative completed-hit weights.
```

## Scope and role

SVC–Recurrence–Proximity Hybrid asks:

> Does a fixed minority allocation to historical nearest-neighbor spacing add
> useful ranking diversity to the adaptive SVC–Recurrence pair?

It produces a complete 1–49 ranking. Its first six numbers form the Top-6 used
by prediction grids, audits, effectiveness histories, comparisons, portfolios,
exports, and Possible Draw.

SRPH is a shadow strategy rather than a replacement for SRH. It does not alter
the source strategies, SRH's score, or SRH's benchmark. It adds no:

- Proximity efficacy tracker;
- adaptive third-source weight;
- recent performance window;
- agreement bonus;
- source quota;
- raw-score mixing;
- residual candidate selector; or
- ticket-shape correction.

The 25% share is a fixed experimental design choice.

## Recursive hidden dependencies

Selecting only SRPH activates:

```text
svc_recurrence_proximity_hybrid
├── svc_recurrence_hybrid
│   ├── recurrence_dynamics
│   └── svc
└── proximity
```

All four hidden calculations can run while only SRPH is serialized. Selecting
SRPH does not automatically display source cards or the intermediate SRH card.

The dependency chain has two state roles:

- SRH maintains cumulative completed Top-6 hits for Recurrence and SVC and
  calculates their current pair weights.
- Proximity maintains its own historical nearest-distance profile, but SRPH
  consumes only its current complete ranking.

SRPH itself has no separate learned state beyond the source states it reuses.

## Source 1 and 2: the adaptive SRH pair

Let (r_{R,t}(n)\) and (r_{S,t}(n)\) be the complete Recurrence and SVC ranks
for number (n\) at target (t\). Convert each to rank strength:

```{math}
q_{R,t}(n)=\frac{49-r_{R,t}(n)}{48},
\qquad
q_{S,t}(n)=\frac{49-r_{S,t}(n)}{48}.
```

After (E_t\) completed forecasts, let (H_{R,t}\) and (H_{S,t}\) be cumulative
source Top-6 hits. With

```{math}
A=24\times\frac{36}{49}=17.632653,
```

the inherited SRH weights are

```{math}
w_{R,t}=\frac{H_{R,t}+A}{H_{R,t}+H_{S,t}+2A},
```

```{math}
w_{S,t}=\frac{H_{S,t}+A}{H_{R,t}+H_{S,t}+2A}.
```

They satisfy (w_{R,t}+w_{S,t}=1\). The intermediate pair score is

```{math}
S_{\mathrm{SRH},t}(n)=
w_{R,t}q_{R,t}(n)+w_{S,t}q_{S,t}(n).
```

SRPH reuses this exact score dictionary before SRH's final gap/number sort. It
does not convert the intermediate SRH ranking into another rank strength.
Consequently, a tie resolved in the displayed SRH ranking does not change the
underlying pair score supplied to SRPH.

## Source 3: Proximity

The production Proximity strategy summarizes where each number historically
appeared relative to its nearest neighbor in the same completed draw.

For a drawn number, production finds its smallest available distance to the
adjacent number on the left or right in sorted order. End numbers have one
available neighbor. The distance enters one of six fixed buckets:

| Bucket | Nearest-neighbor distance |
|---|---:|
| Paired | 1 |
| Tight | 2–3 |
| Near | 4–6 |
| Balanced | 7–10 |
| Wide | 11–15 |
| Isolated | 16 or more |

Let (C_{n,k}(t)\) be the cumulative appearances of number (n\) in bucket
(k\), and let

```{math}
T_k(t)=\sum_{n=1}^{49}C_{n,k}(t).
```

With (d_t\) completed draws, the global bucket share is

```{math}
p_k(t)=\frac{T_k(t)}{6d_t},
```

subject to production's zero-history safeguards. Proximity's raw number score
is

```{math}
U_{P,t}(n)=
\frac1{d_t}\sum_k C_{n,k}(t)p_k(t).
```

The 49 raw values are min–max scaled and ranked by score, larger current gap,
then smaller number. Let that complete position be (r_{P,t}(n)\), and define

```{math}
q_{P,t}(n)=\frac{49-r_{P,t}(n)}{48}.
```

SRPH uses only (q_P\). It discards Proximity's raw score magnitude, bucket
counts, and support details.

## Exact SRPH score

The fixed Proximity allocation is

```{math}
\lambda_P=0.25,
```

leaving

```{math}
1-\lambda_P=0.75
```

for the adaptive pair. Production calculates

```{math}
S_{\mathrm{SRPH},t}(n)=
0.75S_{\mathrm{SRH},t}(n)
+0.25q_{P,t}(n).
```

Expanded into all three sources:

```{math}
S_{\mathrm{SRPH},t}(n)=
0.75w_{R,t}q_{R,t}(n)
+0.75w_{S,t}q_{S,t}(n)
+0.25q_{P,t}(n).
```

The effective source weights are therefore

```{math}
W_{R,t}=0.75w_{R,t},
\qquad
W_{S,t}=0.75w_{S,t},
\qquad
W_P=0.25.
```

They remain non-negative and sum to one.

### Cold-start allocation

Before any completed source forecast, SRH assigns 50/50 to Recurrence and SVC.
Thus SRPH starts at

```{math}
W_{R,0}=37.5\%,
\qquad
W_{S,0}=37.5\%,
\qquad
W_P=25\%.
```

Only the Recurrence/SVC division can change. Proximity remains exactly 25% at
every target, even if its completed efficacy is stronger or weaker.

## Score invariants

All three source rankings are complete permutations, so each rank-strength
vector has mean 0.5 and sum 24.5. The convex blend therefore satisfies

```{math}
0\le S_{\mathrm{SRPH},t}(n)\le1,
```

```{math}
\frac1{49}\sum_{n=1}^{49}S_{\mathrm{SRPH},t}(n)=0.5,
```

```{math}
\sum_{n=1}^{49}S_{\mathrm{SRPH},t}(n)=24.5.
```

No final min–max scaling is applied. These invariants hold regardless of the
adaptive pair split.

## Ranking and tie-break

Numbers are ordered by:

1. larger SRPH score;
2. larger current gap; then
3. smaller number.

Exact score ties are possible because every source contributes equally spaced
rational rank strengths and the source weights can also be rational. The gap
tie-break is therefore part of the strategy definition, not merely a UI
presentation choice.

The Top-6 is the first six individual ranks. There are no parity, sum, spacing,
coverage, or joint-ticket constraints after sorting.

## How the Proximity injection changes SRH

A fixed 25% source can alter the boundary in several ways:

- a candidate already strong in SRH and Proximity can be reinforced;
- a Proximity Top-6 candidate with moderate SVC and Recurrence ranks can enter
  SRPH;
- an SRH candidate weak in Proximity can be displaced; and
- a number outside every individual Top-6 can still enter through broadly
  favorable complete ranks.

This is rank diversification, not a requirement that one or two SRPH numbers
come from Proximity. The formula has no quota, so Proximity may change zero,
one, or several final Top-6 positions.

A component does not need stronger standalone Top-6 efficacy to improve a
particular historical blend. It needs sufficiently different complete-rank
errors near the ensemble boundary. That mechanism also makes retrospective
candidate and weight selection especially vulnerable to overfitting.

## Causal lifecycle and leakage protection

For target draw (t\), production follows this causal sequence:

1. use Recurrence, SVC, and Proximity state available after draw (t-1\);
2. calculate SRH weights from Recurrence/SVC source outcomes completed only
   through (t-1\);
3. build the three complete source rankings and exact SRH pair score;
4. calculate and retain the SRPH ranking for target (t\);
5. when draw (t\) occurs, run the fixed source-training and hybrid-tracking
   update pass, comparing copied pending Recurrence/SVC Top-6 lists with the
   actual numbers rather than rebuilt rankings;
6. finish updating and remembering the completed draw in all three source
   states; and
7. build target (t+1\) with the newly available state.

The target never affects its own source ranks, pair weights, Proximity rank, or
SRPH score. Appending later draws leaves every earlier prediction unchanged.

Proximity outcomes are not evaluated for adaptive weighting. The fixed 25%
share therefore cannot leak through a current-target performance calculation.

## Efficacy and evidence separation

Three distinct records should not be conflated:

- cumulative Recurrence and SVC source hits determine the inherited 75% pair
  split;
- standard SRPH efficacy records SRPH's own completed Top-6 hits for charts and
  exports, but does not alter the formula; and
- Recurrence Dynamics evidence metadata remains attached only to the
  standalone Recurrence strategy.

SRPH's serialized `evidence` is `None`. Selecting only SRPH serializes standard
SRPH efficacy and no hidden source efficacy cards.

SRPH Residual Diversity Hybrid can consume SRPH as a later base strategy, but
that downstream selector does not feed any information back into SRPH.

## Interpreting application fields

Every number carries four SRPH-specific detail lines:

- **Recurrence effective weight; rank; Top 6 yes/no** reports
  (0.75w_R\) and the exact source position.
- **SVC effective weight; rank; Top 6 yes/no** reports (0.75w_S\).
- **Proximity fixed weight; rank; Top 6 yes/no** reports the frozen 25% share.
- **Effectiveness history … completed draws** reports the common Recurrence/SVC
  evaluation length used by SRH's weight calculation.

The standard payload adds SRPH score, current gap, final rank, Top-6 membership,
and completed SRPH efficacy.

Weights are displayed to one decimal percentage point. The score is a rank
strength average, not an occurrence probability, confidence estimate, or
calibrated posterior.

## Endpoint diagnostic

After all 771 repository draws, Recurrence and SVC have equal cumulative hit
totals. Target draw 772 therefore uses effective weights:

```{math}
W_R=37.5\%,
\qquad W_S=37.5\%,
\qquad W_P=25.0\%.
```

The SRPH Top-6 is:

| Rank | Number | Score | Gap | Recurrence rank | SVC rank | Proximity rank |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.929688 | 8 | 4 | 1 | 10 |
| 2 | 4 | 0.890625 | 4 | 3 | 3 | 16 |
| 3 | 10 | 0.851562 | 1 | 16 | 5 | 1 |
| 4 | 26 | 0.755208 | 7 | 14 | 18 | 3 |
| 5 | 9 | 0.752604 | 5 | 7 | 8 | 29 |
| 6 | 40 | 0.736979 | 16 | 6 | 25 | 8 |

Numbers 8 and 4 retain strong pair support despite middling Proximity ranks.
Number 10 combines SVC rank 5 with Proximity rank 1. Number 26 demonstrates
the fixed injection most clearly: Proximity rank 3 lifts Recurrence/SVC ranks
14 and 18 into the final Top-6. Number 9 remains competitive through near-top
pair consensus despite Proximity rank 29.

For comparison, the endpoint SRH Top-6 is `(8, 4, 2, 9, 10, 30)`. The 25%
injection replaces 2 and 30 with 26 and 40 at this target. That replacement is
a deterministic rank consequence, not a guarantee of improved outcomes.

Candidates 6 and 30 both score 0.713542. Their gaps are 11 and 2, so the
standard gap tie-break ranks number 6 first.

## Top-6 statistical reference

For any fixed six-number prediction evaluated against an independent uniformly
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

This null expectation is used in the inherited 24-draw Recurrence/SVC prior.
Proximity's weight is fixed and does not use the null prior.

## Leakage-free replay reference

The production benchmark replays 771 chronological YAML draws, creating 770
target forecasts:

| Slice | Strategy | Hits | Mean hits per target | Random expected total |
|---|---|---:|---:|---:|
| Full 770 | **SVC–Recurrence–Proximity Hybrid** | **643** | **0.835065** | 565.714 |
| Full 770 | SVC–Recurrence Hybrid | 634 | 0.823377 | 565.714 |
| Full 770 | Recurrence Dynamics | 625 | 0.811688 | 565.714 |
| Full 770 | SVC | 625 | 0.811688 | 565.714 |
| Full 770 | Proximity | 601 | 0.780519 | 565.714 |
| Validation 121–520 | **SVC–Recurrence–Proximity Hybrid** | **328** | **0.820000** | 293.878 |
| Validation 121–520 | SVC–Recurrence Hybrid | 325 | 0.812500 | 293.878 |
| Validation 121–520 | Proximity | 325 | 0.812500 | 293.878 |
| Holdout 521–770 | **SVC–Recurrence–Proximity Hybrid** | **205** | **0.820000** | 183.673 |
| Holdout 521–770 | SVC–Recurrence Hybrid | 204 | 0.816000 | 183.673 |
| Holdout 521–770 | Proximity | 174 | 0.696000 | 183.673 |

The latest 250-target slice, target draws 522–771, records 206 SRPH hits, 205
SRH hits, and 173 Proximity hits.

The injection adds nine full-replay hits, three validation hits, and one
holdout hit over SRH. The small holdout difference is especially fragile, and
Proximity itself is below random expectation on that holdout. The result is
compatible with useful boundary diversification, retrospective selection, or
ordinary finite-sample variation; it does not identify which explanation will
persist.

## Planning-baseline reconciliation

An earlier exploratory calculation recorded 644 full-history hits and 329
validation hits. Production reproducibly records 643 and 328.

At target draw 161, numbers 23 and 30 have the mathematically identical score

```{math}
\frac{965377}{1327776}.
```

Number 30 has current gap 2 and number 23 has gap 1, so the fixed tie-break
places 30 sixth. The exploratory floating-point evaluation order placed 23
sixth, and 23 happened to occur in that target. The holdout total is unaffected
at 205.

Changing the tie merely to restore the planning total would violate the
specified formula and deterministic ranking rule. The production figure is
therefore the documented acceptance baseline.

## Selection history and experimental status

Proximity and the 25% share were chosen after comparing multiple candidate
strategies and weights on the same historical dataset. The nominal validation
and holdout labels do not undo that reuse: the design decision had access to
their outcomes.

The formula is now frozen for evaluation on future untouched draws. Historical
improvement is an implementation regression target, not independent evidence
for promotion or future predictability.

## Core mathematical and statistical concepts

- **Recursive ensemble composition:** a two-source adaptive score becomes 75%
  of a separate three-source strategy.
- **Ordinal normalization:** all three complete rankings use the common
  ((49-r)/48\) strength scale.
- **Convex combination:** effective weights are non-negative, sum to one, and
  preserve the 0–1 range and mean score 0.5.
- **Neutral-prior smoothing:** 24 random-expectation pseudo-draws stabilize the
  Recurrence/SVC division.
- **Fixed diversification allocation:** Proximity contributes 25% without
  performance adaptation.
- **Cumulative walk-forward learning:** only completed Recurrence/SVC Top-6
  outcomes affect future pair weights.
- **Boundary error cancellation:** a minority source can reorder candidates
  near the final Top-6 cutoff.
- **Deterministic tie resolution:** current gap and number make equal rational
  scores reproducible.
- **Hypergeometric overlap:** standard efficacy compares six selected numbers
  with six actual numbers from 49.

## Limitations and responsible interpretation

- **Explicit selection bias:** Proximity and 25% were selected after testing
  alternatives on the available history.
- **Tiny holdout increment:** the nominal holdout improves by only one hit over
  SRH.
- **Weak Proximity holdout:** Proximity alone records 174 hits versus 183.673
  expected on the holdout.
- **Frozen third-source weight:** Proximity remains at 25% even when completed
  evidence is unfavorable.
- **Asymmetric adaptation:** Recurrence and SVC adapt cumulatively, while
  Proximity never participates in the weight tracker.
- **Correlated sources:** all three use the same chronological history, and
  SVC/Proximity both encode aspects of gap or spacing structure.
- **Rank information loss:** raw margins, recurrence distances, proximity
  counts, supports, and score spreads are discarded.
- **Linear rank spacing:** every adjacent rank difference contributes exactly
  (1/48\), regardless of source confidence.
- **Cumulative inertia:** old SVC/Recurrence hits never expire.
- **No joint-ticket model:** the final six marginal ranks are not checked for
  parity, sum, group shape, or mutual spacing.
- **Tie-break influence:** current gap can alter Top-6 efficacy at exact score
  ties without being part of the three-source formula.
- **Uncalibrated output:** displayed scores and effective weights are not
  individual-number probabilities.
- **Dependency sensitivity:** any source or SRH change propagates into SRPH and
  its benchmark.
- **Downstream comparison risk:** SRPH is also reused by a later residual
  strategy, increasing the broader multiple-comparison context.
- **No guaranteed predictability:** a historically favorable blend can regress
  to random expectation or worse on untouched future draws.

Use SRPH as a frozen experiment in whether a minority spacing rank diversifies
an adaptive two-model ensemble, not as proof that adding a third model creates
reliable future information.

## Implementation map

The production strategy is implemented in
`src/rand_ai/strategy_prediction.py`:

- `_SVC_RECURRENCE_PROXIMITY_WEIGHT` fixes Proximity at 0.25;
- `_STRATEGY_DEPENDENCIES` activates SRH and Proximity, with SRH recursively
  activating Recurrence and SVC;
- `_svc_recurrence_weights` supplies the inherited prior-smoothed pair split;
- `_svc_recurrence_scores` creates the exact intermediate pair score and saves
  pending source rankings;
- `_svc_recurrence_proximity_scores` expands effective weights, converts the
  Proximity rank, applies the 75/25 score, and creates all detail strings;
- `_proximity_bucket`, `_StrategyState.remember`, and `_proximity_scores`
  implement the spacing-profile source;
- `_ranking_from_scores` applies score, gap, and number ordering; and
- `build_strategies` serializes engine `SRPH` without source evidence.

Registration and presentation are distributed across the Python bridge,
Electron plugin list, strategy family/color registries, Settings, prediction
grids, audits, effectiveness, comparisons, portfolios, exports, and Possible
Draw views. The strategy remains excluded from default-enabled identifiers.

`scripts/benchmark_svc_recurrence_proximity_hybrid.py` reproduces the fixed
full, validation, and holdout comparison recorded in
`reports/svc_recurrence_proximity_hybrid_benchmark.md`.

Tests in `tests/test_strategy_prediction.py` verify the exact 75/25 formula,
37.5/37.5/25 cold start, fixed Proximity share after source adaptation, hidden
dependencies, selected-only output, prefix invariance, and preservation of
SRH. `tests/test_gui_bridge.py` verifies default-disabled state, engine name,
standard efficacy serialization, and selected-strategy-only history.
