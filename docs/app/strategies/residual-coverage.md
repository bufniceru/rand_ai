(residual-coverage)=
# Residual Coverage

## Introduction

**Residual Coverage** is the production strategy with identifier
`residual_coverage` and short engine name **RCOV**. It is a default-enabled
member of the **Ensembles & Coverage** family.

Rather than searching for agreement among strategies, Residual Coverage looks
at the complement of their Top-6 selections. It gives an absolute first tier
to numbers outside every available base Top-6, then favors larger current gaps
within that tier. A tiny average-rank term distinguishes equally overdue
uncovered candidates.

```{admonition} A diversification rule, not independent evidence
:class: important

Being absent from every source Top-6 does not make a number more likely to be
drawn. RCOV deliberately covers what the base portfolio omitted; it does not
prove that omission creates predictive residual value.
```

## Scope and role

Residual Coverage asks:

> Which numbers are not selected by any available source Top-6, and how can
> those uncovered candidates be ordered reproducibly for a complementary
> six-number ranking?

It produces a complete 1–49 ranking. Its first six numbers form the Top-6 used
by prediction grids, audits, effectiveness histories, comparisons, portfolios,
exports, and Possible Draw.

RCOV is not a score average, majority vote, learned stacker, or recent-efficacy
selector. It uses only:

- whether each source places a number in its Top-6;
- the number's complete source ranks, for a very small uncovered tie term; and
- the current causal gap of the number.

There is no independent training state, adaptive weight, warm-up, agreement
bonus, ticket quota, or number-level probability model.

## Hidden baseline source pool

Selecting only Residual Coverage recursively activates a fixed baseline pool
of 24 strategies. Those dependencies can calculate rankings invisibly; only
RCOV is serialized unless the user also selected the sources themselves.

| Family or role | Implicit source strategies |
|---|---|
| Frequency and recency | Freshness, Chi-square Frequency, Entropy, Bayesian |
| Shape and similarity | Proximity, Earth Mover Distance, Recurrence Dynamics |
| Markov and sequence | Markov 100, Markov Freshness, Markov Spaces, Doublet & Triplet Markov |
| Relationships and ML | Next Draw Co-occurrence, Support Vector Classifier, Temporal Behavior Learning |
| Ensembles | Mixed Prediction, Collective Intelligence Strategy |
| Grid | Predictive Score Grid |
| Border space groups | Statistical, Markov, Bayesian, ML, and Hybrid |
| Random controls | Random baseline and Fresh Random |

The exact implicit identifiers are:

```text
proximity, freshness, emd, recurrence_dynamics, randomness, fresh_random,
chi_square, entropy, markov100, mkfr, mksp, bayesian, predictive_grid,
co_occurrence, doublet_triplet_markov, mixed, svc, tbl, cis,
border_group_statistical, border_group_markov, border_group_bayesian,
border_group_ml, border_group_hybrid
```

Random baseline and Fresh Random count like every other source. The pool does
not weight a learned engine more heavily than a deterministic comparison
control.

### Sources not activated implicitly

The dependency declaration does not automatically activate:

- Categorical Chi-square;
- Markov Gap-Space Vector;
- Markov Normalized Positions;
- Markov Relative Dispersion;
- the SVC–Recurrence hybrids and SRPH Residual Diversity Hybrid;
- Scikit Online SVM and Lagged Logistic;
- Sparse Neural Ticket; or
- Decision Tree Selector.

Chained Strategy is built after Residual Coverage and cannot be one of its
sources.

## Selection-dependent source extension

The fixed 24-strategy pool describes a run in which only RCOV is requested and
the application's standard default configuration. Production ultimately
consumes every ranking already available at the RCOV build stage except:

- `mknp` — Markov Normalized Positions; and
- `mkrd` — Markov Relative Dispersion.

Consequently, another explicitly selected strategy can augment the source pool
even when RCOV would not activate it by itself. For example, explicitly
enabling Categorical Chi-square, Scikit Online SVM, or an SVC hybrid makes its
ranking available to RCOV in that run. This can change source count, coverage
support, uncovered membership, scores, and Top-6 output.

```{admonition} Compare like with like
:class: warning

RCOV forecasts are comparable across datasets only when the enabled strategy
set and every source implementation are also held fixed. The strategy is
prefix invariant for one fixed configuration, but it is intentionally not
invariant to adding another source ranking.
```

## Complete source ranks

Let (\mathcal J_t\) be the available source set for target draw (t\), with

```{math}
C_t=|\mathcal J_t|.
```

Every source (j\in\mathcal J_t\) supplies a permutation of 1–49. Let

```{math}
r_{t,j}(n)\in\{1,\ldots,49\}
```

be number (n\)'s rank in source (j\).

Production calculates the source Top-6 support count

```{math}
u_t(n)=
\sum_{j\in\mathcal J_t}
\mathbf1[r_{t,j}(n)\le6].
```

A number is **uncovered** exactly when

```{math}
u_t(n)=0.
```

It is **covered** when at least one source ranks it in the first six. Repeated
support from correlated sources counts repeatedly; RCOV does not collapse
similar strategies into one vote.

## Average rank and near-miss strength

The diagnostic average rank is

```{math}
\bar r_t(n)=
\frac1{C_t}\sum_{j\in\mathcal J_t}r_{t,j}(n).
```

Each complete source rank is converted to linear strength:

```{math}
q_{t,j}(n)=\frac{49-r_{t,j}(n)}{48}.
```

The mean near-miss strength is

```{math}
a_t(n)=
\frac1{C_t}\sum_{j\in\mathcal J_t}q_{t,j}(n)
=\frac{49-\bar r_t(n)}{48}.
```

Thus (a_t(n)\in[0,1]\), and a smaller average source rank produces larger
strength. Despite the implementation name *near miss*, the calculation uses
all 49 rank positions, not only rank 7 or the top quarter.

This strength affects only uncovered scores, and its coefficient is
`0.0001`. It is a fine-ordering term, not a substantive blend with source
rankings.

## Current-gap normalization

Let (g_t(n)\ge0\) be the number's current recurrence gap after the latest
completed draw. Define

```{math}
G_t=\max\!\left(1,\max_{1\le m\le49}g_t(m)\right).
```

The normalized value (g_t(n)/G_t\) lies between 0 and 1. It measures relative
waiting time within the current 49-number state; it is not an estimated
probability or evidence that a number is due.

At the earliest histories, unseen-number and zero-gap behavior follows the
shared causal gap state used by every strategy. No future appearance is used.

## Exact two-branch score

For an uncovered number, production assigns

```{math}
S_t(n)=
0.55
+0.4498\frac{g_t(n)}{G_t}
+0.0001a_t(n),
\qquad u_t(n)=0.
```

For a covered number, it assigns

```{math}
S_t(n)=
0.43\left(1-\frac{u_t(n)}{C_t}\right)
+0.01\frac{g_t(n)}{G_t},
\qquad u_t(n)>0.
```

There is no final min–max transformation. These branch scores are the values
stored in the standard strategy payload.

### Why every uncovered number ranks first

The minimum uncovered score is 0.55. For any covered number,

```{math}
S_t(n)<0.43+0.01=0.44.
```

The score bands are separated by more than 0.11. Therefore every uncovered
number ranks ahead of every covered number, regardless of gap or average rank.

This produces two cases:

- if at least six numbers are uncovered, RCOV's Top-6 contains only uncovered
  numbers;
- if fewer than six are uncovered, every uncovered number appears first and
  the remaining Top-6 positions come from the covered branch.

The method does not guarantee that six uncovered candidates always exist.

### Ordering inside the uncovered branch

The gap term spans almost the entire uncovered band, from 0 to 0.4498. The
near-miss strength contributes at most 0.0001. For equal gaps, better average
base rank produces the higher score.

The constants strongly prioritize gap on the current repository scale, but
the exact formula—not an undocumented lexicographic sort—is authoritative.

### Ordering inside the covered branch

Covered candidates receive less score as Top-6 support increases. The support
term can contribute up to just under 0.43, while the gap term contributes at
most 0.01. Average base rank is displayed but does not enter the covered
branch score.

For the standard 24-source pool, one additional source of support changes the
score by

```{math}
\frac{0.43}{24}=0.0179167,
```

which exceeds the complete 0.01 gap range. With that pool, fewer supporting
Top-6 lists always outrank more supporting lists inside the covered branch.

## Final ranking and tie-break

The complete 1–49 ranking is sorted by:

1. larger RCOV score;
2. larger current gap; then
3. smaller number.

The second criterion is usually already represented in the score, but remains
the standard deterministic tie-break. The first six ranks form the prediction.

There are no ticket-shape checks, low/high quotas, odd/even quotas, number
replacement steps, or agreement bonuses after this sort. The six marginal
selections need not resemble a historically typical draw.

## Causal lifecycle and leakage protection

For completed draw (t\), production performs the following sequence:

1. update each stateful source using the newly completed draw (D_t\);
2. advance source gap, frequency, sequence, relationship, and other causal
   state through (t\);
3. build every enabled source's complete ranking for target (t+1\);
4. collect the available source Top-6 support and ranks;
5. calculate current gaps, RCOV branch scores, and the complete ranking; and
6. retain the Top-6 until draw (t+1\) occurs and can be evaluated.

The actual target draw (D_{t+1}\) cannot enter its own source rankings,
coverage set, gaps, or RCOV score. It affects only later forecasts.

Appending future draws leaves earlier RCOV forecasts unchanged when strategy
selection and settings are unchanged. Any source that has its own delayed
training or warm-up remains responsible for its causal output; RCOV reads the
ranking that source legitimately produced at the target time.

## Cold start and adaptation

RCOV has no separate cold-start mode. From the first forecastable target, each
enabled source returns a complete ranking under its own cold-start behavior,
and RCOV immediately calculates coverage.

The output can still change substantially as history grows because:

- source models leave their own warm-up or fallback modes;
- current number gaps change;
- base Top-6 overlaps and unions change; and
- optional source selection or application settings change.

RCOV does not track whether its residual choices have succeeded. Standard
walk-forward efficacy is attached for reporting, but those completed hit
counts are never fed back into the score.

## Interpreting application fields

Each number carries four RCOV-specific detail lines:

- **Base Top-6 support _u/C_** reports the number of current source Top-6 lists
  containing the number and the current source count.
- **Outside every base Top-6** or **Already covered by the base portfolio**
  identifies the score branch.
- **Current gap** reports (g_t(n)\).
- **Average base rank** reports (\bar r_t(n)\), rounded to one decimal place.

The standard payload separately contains raw RCOV score, final rank, current
gap, Top-6 membership, and completed efficacy.

```{admonition} The score is not a percentage chance
:class: warning

A score near 0.60 identifies an uncovered candidate at a particular relative
gap. It does not mean a 60% chance of appearing. The intentional discontinuity
between covered and uncovered score bands makes probability interpretation
especially inappropriate.
```

## Endpoint diagnostic

After all 771 repository draws, requesting only RCOV creates the fixed
24-source pool for target draw 772. Its 144 source Top-6 slots contain 34
distinct numbers, leaving 15 numbers outside every source Top-6.

The support-count distribution is:

| Source Top-6 support | Number count |
|---:|---:|
| 0 | 15 |
| 1 | 5 |
| 2 | 11 |
| 3 | 4 |
| 4 | 2 |
| 5 | 3 |
| 6 | 3 |
| 7 | 1 |
| 8 | 1 |
| 10 | 2 |
| 13 | 1 |
| 16 | 1 |

The weighted count check is

```{math}
\sum_{n=1}^{49}u_{772}(n)=24\times6=144.
```

The maximum current gap is 39, but that number is already covered. Because
coverage tier dominates gap, the RCOV Top-6 is:

| Rank | Number | Score | Gap | Support | Average base rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 21 | 0.619247 | 6 | 0/24 | 26.3 |
| 2 | 27 | 0.619245 | 6 | 0/24 | 27.2 |
| 3 | 23 | 0.607719 | 5 | 0/24 | 23.8 |
| 4 | 36 | 0.607715 | 5 | 0/24 | 26.0 |
| 5 | 9 | 0.607708 | 5 | 0/24 | 29.2 |
| 6 | 24 | 0.596193 | 4 | 0/24 | 20.5 |

Numbers 21 and 27 share gap 6, so their tiny near-miss terms order them by
average source rank. The same mechanism orders numbers 23, 36, and 9 within
gap 5. Number 24 takes the final position from the remaining uncovered pool.

These values describe one fixed history, source set, and code version. They
are a reproducibility diagnostic, not evidence that neglected numbers become
more likely.

## Top-6 efficacy reference

Conditional on completed history, RCOV's Top-6 is a fixed six-number set. If
the next actual draw is independent and uniformly random, the overlap (H\)
follows

```{math}
H\sim\operatorname{Hypergeometric}(N=49,K=6,n=6),
```

with

```{math}
\mathbb E[H]=\frac{36}{49}=0.734694,
\qquad
\operatorname{Var}(H)=0.577572.
```

A leakage-free production replay with the fixed implicit 24-source pool over
the repository's 771 chronological YAML draws creates 770 target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 536 | 0.696104 | 565.714 |
| Validation, target draws 121–520 | 400 | 263 | 0.657500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 184 | 0.736000 | 183.673 |

The latest 250-target slice, target draws 522–771, records 186 hits or
0.744000 per target, against an expectation of 183.673.

The full and validation totals are below theoretical random expectation. The
holdout and latest slices are close to expectation and do not recover the
earlier deficit. This replay does not demonstrate positive residual efficacy,
statistical significance, or a stable future advantage.

## Core mathematical and statistical concepts

- **Set union and complement:** the source Top-6 union defines covered numbers;
  its complement defines the residual pool.
- **Support count:** (u_t(n)\) measures how many source Top-6 lists contain a
  number without treating those sources as independent.
- **Separated score bands:** constants below 0.44 for covered and at least
  0.55 for uncovered implement a hard coverage tier through scalar scores.
- **Relative waiting time:** current gap divided by maximum gap orders residual
  candidates without changing source membership.
- **Rank-strength transform:** ((49-r)/48) supplies a bounded fine-ordering
  term for uncovered candidates.
- **Anti-consensus diversification:** the strategy intentionally seeks numbers
  rejected by the current source consensus.
- **Hidden dependency expansion:** complete source models can run without
  being serialized as selected output.
- **Conditional causal evaluation:** all rankings are fixed before the target
  outcome is observed.
- **Hypergeometric overlap:** standard efficacy counts the intersection of two
  six-element subsets of 49.

## Limitations and responsible interpretation

- **No positive replay result:** full and validation efficacy are below random
  expectation; holdout is approximately random.
- **Contrarian fallacy risk:** source omission does not imply that a number is
  due or underpriced.
- **Equal source treatment:** one Top-6 vote counts equally from a random
  control, a simple model, and a complex learned ensemble.
- **Correlated sources:** many engines reuse gaps, frequencies, ranks, or one
  another, so support is not independent evidence.
- **Selection dependence:** explicitly enabling another ranking can alter the
  source pool and every downstream RCOV value.
- **Dependency sensitivity:** any source implementation, warm-up, setting, or
  tie-break change can change the residual complement.
- **Coverage saturation:** 24 sources contribute 144 Top-6 slots; sometimes
  fewer than six or no numbers may remain uncovered.
- **Hard discontinuity:** moving from support 0 to support 1 causes a large
  score drop even when the source rank changes only from 7 to 6.
- **Gap dominance:** waiting time receives almost all within-residual weight,
  although a fair lottery has no obligation to compensate overdue numbers.
- **Tiny rank term:** average source rank can break close uncovered cases but
  contributes no more than 0.0001.
- **No quality adaptation:** completed RCOV or source hits do not change its
  formula or source weights.
- **No joint-ticket model:** six individually selected residual numbers are
  not validated for spacing, parity, sums, group signatures, or relationships.
- **Score non-calibration:** the two branch formulas are engineering priorities,
  not probabilities or confidence estimates.
- **Computational scope:** selecting only RCOV still evaluates 24 hidden source
  strategies, including stateful and experimental calculations.
- **Retrospective design:** source membership, constants, settings, and slices
  were defined or inspected with historical data available.
- **No guaranteed predictability:** diversification can change exposure but
  cannot manufacture information about a future random outcome.

Use Residual Coverage to inspect and export the complement of the current
strategy portfolio, not as evidence that disagreement or neglect predicts the
next draw.

## Implementation map

The production strategy is implemented in
`src/rand_ai/strategy_prediction.py`:

- `_BASE_STRATEGY_IDS` defines the complete base registry;
- `_STRATEGY_DEPENDENCIES["residual_coverage"]` declares the 24 rankings that
  activate when RCOV is requested by itself;
- `_rank_strength` implements ((49-r)/48\);
- `_StrategyState._residual_coverage_scores` calculates support, average rank,
  normalized gaps, the two score branches, and detail text;
- `_StrategyState.build_strategies` collects rankings available before RCOV,
  excludes `mknp` and `mkrd`, applies the standard tie-break, and serializes
  engine `RCOV`; and
- `_EfficacyTracker` attaches completed Top-6 history without feeding it back
  into the ranking.

Registration and presentation are distributed across the application:

- `src/rand_ai/gui_bridge.py` includes RCOV in default-enabled strategy and
  serialized efficacy payloads;
- `web/electron/main.cjs` registers **Residual Coverage**;
- `web/src/lib/strategyFamilies.ts` places it in **Ensembles & Coverage**;
- `web/src/lib/strategyColors.ts` assigns its family tone position; and
- prediction, comparison, effectiveness, portfolio, export, and Possible Draw
  views consume the standard strategy payload.

Tests in `tests/test_strategy_prediction.py` verify hidden dependency
selection, complete 49-number output, exclusion of standalone `mknp` and
`mkrd`, uncovered-first behavior, gap priority, detail strings, and
serialization. Bridge and frontend tests cover default state, display name,
family, color, and efficacy output.
