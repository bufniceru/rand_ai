(fresh-random)=
# Fresh Random

## Introduction

**Fresh Random** is the production strategy with identifier `fresh_random` and
short engine name **FRnd**. It is a default-enabled member of the **Random
Baselines** family.

The strategy combines a reproducible pseudo-random ranking with the causal
Freshness ranking:

- 65% of each score comes from a target-specific seeded shuffle; and
- 35% comes from the historical exact-gap Freshness rank.

Fresh Random is designed as a randomized comparison baseline with a modest
data-guided tilt. It is not a stochastic simulation during display: the same
target draw number, history prefix, and code version reproduce the same
ranking exactly.

```{admonition} A rank blend, not a probability model
:class: important

Fresh Random averages two rank strengths. Its displayed score is not a hit
probability, posterior probability, calibrated confidence, or estimate of
lottery odds.
```

## Scope and role

Fresh Random asks:

> What ranking results when a deterministic random permutation provides most
> of the ordering, while historically estimated gap freshness supplies a
> smaller directional influence?

The first six displayed ranks form its Top-6 prediction. It is included in
prediction grids, audits, effectiveness history, comparisons, exports, and
Possible Draw when selected.

The application deliberately excludes **Random baseline** and **Fresh Random**
from predictive portfolio construction. They remain available as controls for
judging whether more structured strategies add stable value beyond a seeded
ranking.

Selecting only Fresh Random recursively activates its hidden `freshness` and
`randomness` dependencies. Only `fresh_random` is serialized unless those
sources were also requested explicitly.

## Distinct from Random baseline

Fresh Random and **Random baseline** are separate deterministic streams:

- Random baseline uses seed `20260626` and no data-guided component.
- Fresh Random uses seed `20260626 + 7919 = 20268545` for its random leg and
  blends that leg with Freshness.

The Fresh Random implementation generates its own offset-seed permutation. It
does not directly reuse the ordinary Random baseline's permutation, even
though `randomness` is activated as a dependency.

Consequently, two strategies can have different random ranks for the same
target while remaining fully reproducible.

## Target-specific deterministic shuffle

Let \(q\) be the one-based target draw number and let

```{math}
s_F=20{,}268{,}545
```

be the fixed Fresh Random seed. Production initializes a 32-bit state:

```{math}
z_0=
s_F\;\operatorname{XOR}\;
\left(q\times2{,}654{,}435{,}761\pmod{2^{32}}\right).
```

The state is explicitly restricted to 32 bits. Starting with the ordered array

```{math}
(1,2,\ldots,49),
```

the strategy performs a descending Fisher–Yates-style shuffle. For array index
\(i=48,47,\ldots,1\):

```{math}
z\leftarrow
(1{,}664{,}525z+1{,}013{,}904{,}223)\pmod{2^{32}},
```

```{math}
j=z\bmod(i+1),
```

then entries \(i\) and \(j\) are exchanged.

The resulting permutation defines random rank

```{math}
r_R(n)\in\{1,\ldots,49\}.
```

### Reproducibility properties

- The shuffle reads only the fixed seed and target draw number.
- It does not inspect actual target numbers, system time, operating-system
  randomness, or user interaction.
- Rebuilding the same historical prefix produces the same random rank.
- Changing the target index changes the initial state and permutation.

The linear congruential generator is an engineering mechanism for a stable
baseline, not a cryptographically secure random generator. The modulo operation
can also introduce tiny permutation bias because powers of two are not evenly
divisible by every \(i+1\).

## Freshness source model

The data-guided leg comes from the production **Freshness** strategy. For every
historical target draw and every candidate number, the prediction engine records
the candidate's exact pre-draw gap and whether the number appeared.

For exact gap \(g\), let

```{math}
N_g=\text{number of completed candidate opportunities at gap }g,
```

```{math}
H_g=\text{number of those opportunities that were hits}.
```

The reference hit rate is

```{math}
p_0=\frac6{49},
```

and production uses total prior strength 2:

```{math}
\widehat p_g=
\frac{H_g+2p_0}{N_g+2}.
```

An unseen gap therefore returns \(p_0\). As support grows, its estimate moves
toward the empirical exact-gap hit rate.

For the latest completed draw, each candidate is assigned the estimate for its
current exact gap. These 49 values are min–max scaled and ranked by:

1. larger Freshness score;
2. larger current gap; then
3. smaller number.

This produces Freshness rank

```{math}
r_F(n)\in\{1,\ldots,49\}.
```

Fresh Random consumes the complete rank only. It does not mix raw Freshness hit
rates with random values.

## Rank-strength transformation

Both source ranks are converted to the same linear strength scale:

```{math}
\rho(r)=\frac{49-r}{48}.
```

Therefore:

| Rank | Strength |
|---:|---:|
| 1 | 1.000000 |
| 25 | 0.500000 |
| 49 | 0.000000 |

Each source is a permutation, so every strength from 0 through 1 in steps of
\(1/48\) occurs exactly once. Each source's mean strength is 0.5.

Rank strength discards the distance between raw Freshness probabilities. A
small and a large probability separation both become one rank step.

## Exact blended score

For number \(n\), Fresh Random calculates

```{math}
S_{FR}(n)=
0.65\rho(r_R(n))+0.35\rho(r_F(n)).
```

Equivalently,

```{math}
S_{FR}(n)=
0.65\frac{49-r_R(n)}{48}
+0.35\frac{49-r_F(n)}{48}.
```

The score lies in \([0,1]\), but the largest candidate need not reach 1 because
the two sources need not assign rank 1 to the same number. Likewise, the
smallest need not be zero.

Because both source permutations have mean strength 0.5,

```{math}
\frac1{49}\sum_{n=1}^{49}S_{FR}(n)=0.5
```

for every target. The blend receives no additional min–max scaling.

The 65/35 weights and seed offset are fixed. They do not adapt from historical
Fresh Random efficacy, source agreement, or recent winning streaks.

## Displayed ranking and tie-breaking

The serialized strategy is ranked by:

1. larger blended score;
2. larger current gap; then
3. smaller number.

Exact score ties are possible because both ranks are integers. Gap is not part
of the blend formula, but can determine the displayed order of tied scores.

The implementation also retains an internal Fresh Random ordering whose exact
score ties use better Freshness rank and then smaller number. That internal
ordering is available to downstream strategy calculations; the user-visible
`StrategyPrediction` uses the standard score/gap/number ordering above.

## Causal Freshness lifecycle

The random leg requires no learning. Freshness is updated walk-forward:

1. after reference draw \(t-1\), calculate every number's exact gap and retain
   the forecast state for target \(t\);
2. when draw \(t\) becomes known, increment \((H_g,N_g)\) for all 49 candidates
   using the gaps that existed before that draw;
3. observe draw \(t\) to update last-seen positions;
4. calculate current gap estimates and Freshness ranks for target \(t+1\);
5. generate the deterministic target-\(t+1\) random permutation; and
6. blend the two rank strengths.

The target cannot affect its own Freshness rank. Its random rank depends only
on its target number, not its outcome. Prefix-invariance tests confirm that
appending later draws leaves an earlier Fresh Random Top-6 unchanged.

## Cold start and short history

The first prediction is built after draw 1 for target draw 2. No historical
target opportunity has yet been completed, so every exact gap estimate equals
the prior rate \(6/49\). Freshness min–max scores are therefore all zero.

Freshness ranking still resolves those equal scores using current gap and
number:

- the 43 numbers unseen in draw 1 have gap 1 and rank before the six observed
  numbers with gap 0;
- smaller number resolves ties within each gap group.

The seeded random permutation remains the 65% primary source. As more targets
are completed, exact-gap hit-rate ranks can move the remaining 35%.

Rare exact gaps continue to receive strong prior influence because their
support \(N_g\) is small. Fresh Random does not use Freshness support directly;
only the final Freshness ordering matters.

## Interpreting prediction details

Each number exposes three Fresh Random details:

- **Random rank** is \(r_R(n)\) from the offset-seed target permutation.
- **Freshness rank** is \(r_F(n)\) from the causal exact-gap source.
- **Blend 65% random / 35% freshness** identifies the fixed score formula.

The standard prediction payload separately includes blended score, current
gap, displayed rank, and Top-6 membership.

The score is a relative rank-strength average. For example, 80% means a blend
value of 0.8 on the two linear rank scales; it does not mean an 80% chance of
being drawn.

## Endpoint diagnostic

After all 771 repository draws, the forecast for target draw 772 uses seed
`20268545`. Its random permutation begins:

```text
47, 5, 15, 23, 9, 49, 3, 33, 41, 38
```

The displayed Fresh Random Top-6 is:

| Rank | Number | Random rank | Freshness rank | Current gap | Blended score |
|---:|---:|---:|---:|---:|---:|
| 1 | 47 | 1 | 6 | 9 | 0.963542 |
| 2 | 38 | 10 | 1 | 24 | 0.878125 |
| 3 | 42 | 12 | 8 | 11 | 0.800000 |
| 4 | 20 | 11 | 11 | 1 | 0.791667 |
| 5 | 3 | 7 | 22 | 2 | 0.765625 |
| 6 | 5 | 2 | 32 | 7 | 0.760417 |

Number 47 leads because its random rank is first while Freshness still places
it sixth. Number 38 demonstrates the opposite influence: Freshness rank 1
lifts random rank 10 to displayed rank 2. These values are deterministic for
this target and history prefix.

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

A leakage-free production replay over the repository's 771 chronological YAML
draws produces 770 target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 575 | 0.746753 | 565.714 |
| Validation, target draws 121–520 | 400 | 289 | 0.722500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 207 | 0.828000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, also records 207
hits or 0.828000 per target.

The full replay is close to theoretical random expectation, while validation
is below and the later slice is above it. This sharp reversal is not evidence
that the fixed seed predicts a regime. The seed, weights, strategy collection,
and reported slices are known in retrospect, and repeated comparisons can
produce unusually strong or weak subsequences by chance. No stable future lift
or predictability is established.

## Core mathematical and statistical concepts

- **Pseudo-random permutation:** a fixed 32-bit generator and Fisher–Yates-style
  shuffle create a reproducible target-specific ordering.
- **Exact-gap empirical rate:** Freshness groups historical 49-candidate
  opportunities by their precise pre-draw waiting gap.
- **Prior smoothing:** total strength 2 pulls sparse gap rates toward \(6/49\).
- **Rank transformation:** \((49-r)/48\) maps both source permutations to a
  common linear scale.
- **Convex combination:** 65/35 weights preserve the score range and a fixed
  mean of 0.5.
- **Hidden dependency expansion:** Freshness and Random baseline state can run
  without being serialized.
- **Determinism:** seed and target index replace runtime randomness, supporting
  exact audits and prefix comparisons.
- **Hypergeometric overlap:** Top-6 evaluation uses the null overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Baseline purpose:** 65% of the score is intentionally pseudo-random, so
  Fresh Random should not be interpreted as a conventional predictive model.
- **Slice instability:** historical validation and holdout move in opposite
  directions despite one fixed formula.
- **Seed dependence:** another fixed seed would produce different rankings and
  retrospective totals.
- **Index dependence:** inserting, deleting, or reordering historical draws can
  realign target numbers and therefore every subsequent random permutation.
- **Non-cryptographic generator:** the LCG and modulo shuffle prioritize
  reproducibility, not cryptographic or perfect permutation randomness.
- **Rank information loss:** Fresh Random discards the magnitude and support of
  Freshness probability differences.
- **Sparse exact gaps:** rare gap estimates remain uncertain, but rank blending
  provides no confidence adjustment.
- **Fixed source allocation:** random and Freshness shares never adapt to their
  realized efficacy.
- **Gap tie influence:** current gap can affect Freshness rank and later resolve
  exact blended-score ties.
- **No calibrated output:** blended strengths are not occurrence
  probabilities.
- **Portfolio exclusion:** the application does not treat this baseline as a
  predictive portfolio component.
- **Retrospective multiple comparison:** one deterministic seed can look strong
  on a selected interval when many strategies and slices are inspected.
- **No guaranteed predictability:** deterministic reproducibility does not
  convert a pseudo-random ordering into knowledge of future outcomes.

## Implementation map

Freshness state is implemented in `src/rand_ai/prediction.py`:

- `_PredictionEngine._gap_before_draw` derives causal exact gaps;
- `_PredictionEngine.learn_draw` records all 49 candidate opportunities and
  outcomes before observing the current draw;
- `_PredictionEngine._smoothed_rate` applies the strength-2 prior toward
  \(6/49\);
- `_PredictionEngine.observe_draw` advances last-seen state; and
- `_PredictionEngine.build_prediction` creates current Freshness scores and
  ranks.

The Fresh Random blend is implemented in
`src/rand_ai/strategy_prediction.py`:

- `_RANDOM_SEED`, `_FRESH_RANDOM_SEED_OFFSET`, and
  `_FRESH_RANDOM_INFLUENCE` define the fixed constants;
- `_STRATEGY_DEPENDENCIES` activates `freshness` and `randomness` when only
  `fresh_random` is requested;
- `_random_ranking` implements the 32-bit target-specific shuffle;
- `build_strategies` converts random and Freshness ranks to strength, applies
  the 65/35 blend, builds details, and serializes engine `FRnd`; and
- `_strategy` applies the user-visible score/gap/number ranking.

Registration, dependency behavior, complete 49-number output, details,
serialization, and prefix invariance are covered in
`tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py`. Frontend
tests cover its **Random Baselines** family, color, selection, effectiveness
display, and deliberate portfolio exclusion.
