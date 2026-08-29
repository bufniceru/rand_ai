(random-baseline)=
# Random baseline

## Introduction

**Random baseline** is the production strategy with identifier `randomness`
and short engine name **Rand**. It is a default-enabled member of the
**Random Baselines** family.

The strategy creates a complete 1–49 ranking from a fixed seed and the
one-based position of the target draw. It does not learn from historical
outcomes. Its purpose is to provide a stable comparison control: structured
strategies can be evaluated against the same reproducible pseudo-random
predictions instead of against a different random sample on every run.

```{admonition} Deterministic control, not a prediction model
:class: important

Random baseline contains no evidence about which numbers will occur. The word
*random* describes how its permutation is constructed; after the seed and
target index are fixed, the output is completely deterministic.
```

## Scope and role

Random baseline asks:

> What Top-6 overlap would a fixed, target-specific pseudo-random ranking
> produce on the same chronological forecast targets?

The first six numbers in its permutation form its Top-6. When selected, it is
available in prediction grids, audits, effectiveness history, comparisons,
exports, and Possible Draw.

The application deliberately excludes Random baseline and **Fresh Random**
from predictive portfolio construction. These controls remain visible for
comparison, but the portfolio builder does not present them as predictive
components.

Random baseline is standalone:

- it has no hidden strategy dependencies;
- it has no learned parameters or warm-up threshold;
- it does not consume frequency, gap, spacing, relationship, or draw-shape
  features; and
- selecting it does not activate another strategy.

## Distinct from Fresh Random

Random baseline and **Fresh Random** are separate engines:

- Random baseline uses seed `20260626` and ranks solely by its seeded
  permutation.
- Fresh Random uses the offset seed `20268545` and blends its own random rank
  strength 65/35 with the learned Freshness rank strength.

Consequently, their random permutations differ for the same target. Random
baseline is the pure comparison control; Fresh Random is a data-guided rank
blend.

## Target numbering

Let (q\geq2) denote the one-based chronological position of the draw being
forecast. A prediction built after repository draw (q-1) targets draw (q).

The target number is a sequence position used by the prediction suite, not an
official lottery draw identifier. If historical rows are inserted, removed,
or reordered, later targets receive different positions and therefore
different permutations.

This mapping is deliberate. Production builds the suite after a zero-based
reference index (d), requests the random ranking for index (d+1), and the
shuffle initializes its state from that argument plus one. Thus:

```{math}
q=d+2.
```

The efficacy tracker uses the one-based reference draw number (q-1), which
the same shuffle advances to (q). The displayed Random baseline and its
recorded comparison baseline therefore use exactly the same Top-6.

## Fixed 32-bit initial state

Let the fixed production seed be

```{math}
s=20{,}260{,}626.
```

For target (q), the initial unsigned 32-bit state is

```{math}
z_0=
s\;\operatorname{XOR}\;
\left(q\times2{,}654{,}435{,}761\pmod{2^{32}}\right).
```

`XOR` is bitwise exclusive OR. Production masks the multiplication and the
resulting state to 32 bits, so arithmetic is reproducible rather than
dependent on the host language's integer width.

The multiplier (2{,}654{,}435{,}761) spreads consecutive target indices
through the 32-bit state space before the shuffle begins. It does not encode
historical draw information.

## Linear congruential state transitions

Start with the ordered array

```{math}
A=(1,2,\ldots,49).
```

For descending array index (i=48,47,\ldots,1), update the state with the
32-bit linear congruential recurrence

```{math}
z\leftarrow
(1{,}664{,}525z+1{,}013{,}904{,}223)\pmod{2^{32}},
```

then calculate

```{math}
j=z\bmod(i+1)
```

and exchange (A_i) with (A_j). The final array is the full random ranking.

This is a descending Fisher–Yates-style shuffle driven by an explicitly
implemented linear congruential generator (LCG). It has several useful
engineering properties:

- all 49 numbers occur exactly once;
- the same seed and target index reproduce the same permutation;
- no system clock, operating-system entropy, or global random state is read;
- advancing to a new target changes the initialized stream; and
- production, tests, exports, and efficacy comparison can reproduce one
  another exactly.

### What the generator does not guarantee

The LCG is not cryptographically secure. In addition, (z\bmod(i+1)) has a
tiny modulo bias whenever (i+1) does not divide (2^{32}). The
implementation prioritizes a stable application baseline, not cryptographic
unpredictability or a proof that every one of the (49!) permutations is
sampled with exactly equal probability.

## From permutation to score

Let (r_q(n)\in\{1,\ldots,49\}) be the position of number (n) in the
permutation for target (q). Production converts rank to linear strength:

```{math}
S_q(n)=\frac{49-r_q(n)}{48}.
```

Therefore:

- rank 1 receives score (1);
- rank 25 receives score (0.5);
- rank 49 receives score (0); and
- adjacent ranks differ by exactly (1/48\approx0.0208333).

The 49 scores are unique. Their sum and mean are fixed for every target:

```{math}
\sum_{n=1}^{49}S_q(n)=24.5,
\qquad
\frac1{49}\sum_{n=1}^{49}S_q(n)=0.5.
```

The common strategy builder orders candidates by descending score, then by
larger current gap and smaller number. Because the Random baseline scores are
all distinct, neither tie-break can alter its permutation. Current gaps can be
present in the standard prediction payload, but they have no influence on
this strategy's ranking.

The Top-6 is simply

```{math}
T_q=\{A_0,A_1,\ldots,A_5\}.
```

No calibration, threshold fitting, agreement bonus, quota, or ticket-shape
constraint is applied.

## Causal lifecycle

Random baseline has no training lifecycle. For a target draw (q):

1. the application identifies the target's chronological position;
2. the fixed seed and (q) initialize the 32-bit state;
3. the 1–49 array is shuffled and converted to rank strengths; and
4. only after the target occurs can its Top-6 overlap be evaluated.

The actual numbers in target (q) are never an input to its forecast. Earlier
draw values are not inputs either; only their count determines the target
position.

### Prefix invariance

If two datasets have identical first (q-1) chronological rows, they produce
the same forecast for target (q). Appending later draws cannot change that
forecast. This is prefix invariant and leakage-free by construction.

The stronger statement that history values are irrelevant also follows: two
different histories of equal length receive the same Random baseline ranking
for their next target. That property makes the strategy useful as a control,
not as a model of historical behavior.

## Cold start and reproducibility

There is no cold-start degradation or warm-up. The first forecastable target,
draw 2, receives a complete 49-number permutation and the same score scale as
every later target.

Reproducibility depends on three frozen elements:

- seed `20260626`;
- the target-position mapping; and
- the exact 32-bit transition and shuffle procedure.

Changing any one of them creates a different baseline series and different
retrospective totals. Re-running the unchanged implementation does not.

## Interpreting the application output

For every number, the prediction payload includes the standard fields:

- **Score** is (S_q(n)), usually displayed as a percentage of the rank
  scale.
- **Current gap** is supplied by the shared prediction view but is not used by
  the random calculation.
- **Rank** is the number's position in the seeded permutation.
- **Top-6 membership** states whether the number occupies ranks 1–6.
- **Detail** reads `Deterministic PyLotto baseline` for every number.

```{admonition} The percentage is not a chance of occurrence
:class: warning

A displayed score of 100% means *first in this permutation*. It does not mean
a 100% probability of being drawn. Likewise, 50% means the middle rank, not a
50% lottery probability.
```

Scores are directly comparable as ordinal rank positions across targets, but
they contain no changing evidence magnitude: every target always has one of
each strength value from 0 to 1.

## Endpoint diagnostic

After all 771 repository draws, the forecast for target draw 772 uses seed
`20260626`. Its Top-6 permutation prefix is:

```text
37, 4, 14, 42, 49, 13
```

| Rank | Number | Rank-strength score | Current gap |
|---:|---:|---:|---:|
| 1 | 37 | 1.000000 | 4 |
| 2 | 4 | 0.979167 | 4 |
| 3 | 14 | 0.958333 | 4 |
| 4 | 42 | 0.937500 | 11 |
| 5 | 49 | 0.916667 | 14 |
| 6 | 13 | 0.895833 | 0 |

The unequal gaps illustrate that gap does not drive the order: number 42 has
a larger gap than the first three numbers but remains fourth because its
seeded rank is fourth. These values are a reproducibility diagnostic, not
evidence for target 772.

## Top-6 null distribution

If the actual draw is an independent, uniformly random six-number subset and
the predicted Top-6 is fixed, their overlap (H) follows

```{math}
H\sim\operatorname{Hypergeometric}(N=49,K=6,n=6).
```

Its probability mass function is

```{math}
\Pr(H=k)=
\frac{\binom6k\binom{43}{6-k}}{\binom{49}6},
\qquad k=0,1,\ldots,6.
```

The expectation and variance are

```{math}
\mathbb E[H]=\frac{36}{49}=0.734694,
```

```{math}
\operatorname{Var}(H)
=6\left(\frac6{49}\right)
\left(1-\frac6{49}\right)
\left(\frac{43}{48}\right)
=0.577572.
```

This distribution applies to any fixed six-number prediction under the stated
uniform independent-draw null. It does not require the prediction itself to
be randomly regenerated.

## Leakage-free replay reference

A production replay over the repository's 771 chronological YAML draws
creates 770 forecast targets:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 573 | 0.744156 | 565.714 |
| Validation, target draws 121–520 | 400 | 304 | 0.760000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 184 | 0.736000 | 183.673 |

The per-target hit counts were:

| Slice | 0 hits | 1 hit | 2 hits | 3 or more hits |
|---|---:|---:|---:|---:|
| Full replay | 324 | 335 | 95 | 16 |
| Validation | 164 | 177 | 50 | 9 |
| Holdout | 104 | 112 | 30 | 4 |

No full-replay target recorded more than three hits. The latest 250-target
slice, target draws 522–771, also records 184 hits or 0.736000 per target.

The full replay is only 7.286 hits above its theoretical expectation, and the
holdout is almost exactly at expectation. Those descriptive differences do
not establish significance, predictability, or future advantage. The fixed
seed and reported slices are already known retrospectively.

### Efficacy identity

The application evaluates every strategy against this same deterministic
Random baseline. When `randomness` itself is evaluated, its strategy hits and
random comparison hits are identical by definition:

```{math}
H_{\text{strategy}}=H_{\text{random}},
\qquad
H_{\text{strategy}}-H_{\text{random}}=0.
```

This identity is an implementation consistency check. It is not a claim that
the observed total must equal the theoretical expectation on every finite
slice.

## Core mathematical and statistical concepts

- **Pseudo-random determinism:** a fixed seed creates repeatable values that
  resemble a randomized ordering without reading live entropy.
- **Unsigned modular arithmetic:** masks and modulo (2^{32}) reproduce a
  fixed finite-state transition system.
- **Linear congruential generator:** an affine recurrence advances the state
  once per shuffle position.
- **Fisher–Yates-style permutation:** descending swaps construct one complete
  ranking without duplicates.
- **Rank-strength transform:** ((49-r)/48) maps ordinal positions linearly to
  the closed interval from 0 to 1.
- **Prefix invariance:** future rows cannot alter a target's already determined
  permutation.
- **Hypergeometric overlap:** Top-6 efficacy counts successes when two
  six-element subsets of a 49-element population intersect.
- **Finite-sample variation:** observed totals naturally move above and below
  their expectation even under a valid null model.

## Limitations and responsible interpretation

- **No historical signal:** the strategy ignores all historical values and
  cannot learn frequency, recency, shape, or relationships.
- **Seed dependence:** a different fixed seed yields a different forecast
  series and replay total.
- **Index dependence:** inserting, deleting, or reordering draws realigns all
  subsequent target permutations.
- **Non-cryptographic generator:** the LCG should not be used for security,
  gambling integrity, or adversarial randomness.
- **Small modulo bias:** the shuffle does not prove perfectly uniform access
  to all (49!) permutations.
- **Ordinal-only score:** the rank transform has fixed spacing and contains no
  evidence magnitude or uncertainty.
- **Uncalibrated display:** the score is not a probability, confidence, or
  estimate of expected lottery frequency.
- **Retrospective slices:** the seed, dataset, strategies, and evaluation
  intervals are known after inspection.
- **Multiple comparison risk:** among many seeds, strategies, and slices, some
  controls will appear unusually strong or weak by chance.
- **Portfolio exclusion:** the application intentionally keeps both random
  controls outside predictive portfolio construction.
- **Null assumptions:** the hypergeometric reference assumes a uniformly
  random independent actual draw; it does not prove that a particular data
  source satisfies that process.
- **No guaranteed predictability:** deterministic reproducibility does not
  provide information about future winning numbers.

## Implementation map

The strategy is implemented in `src/rand_ai/strategy_prediction.py`:

- `_RANDOM_SEED` defines the fixed seed;
- `_random_ranking` implements target initialization, the 32-bit LCG, and the
  descending shuffle;
- `build_strategies` converts the permutation to rank-strength scores, creates
  the uniform detail text, and serializes engine `Rand` when `randomness` is
  requested;
- `_strategy` constructs the complete 49-number prediction and applies the
  standard score/gap/number ordering; and
- `_EfficacyTracker.compare` regenerates the same target-specific Top-6 for
  the random comparison record.

Registration and presentation are distributed across the Python bridge and
Electron/Vue frontend:

- `src/rand_ai/gui_bridge.py` includes the strategy in the default-enabled
  identifiers and strategy-cache payloads;
- `web/electron/main.cjs` registers the **Random baseline** plugin;
- `web/src/lib/strategyFamilies.ts` assigns it to **Random Baselines**;
- prediction, comparison, and effectiveness views provide its display name;
  and
- `web/src/lib/drawPortfolio.ts` excludes it from portfolio candidates.

Production tests in `tests/test_strategy_prediction.py` cover complete
49-number rankings, Top-6 consistency, progress, efficacy serialization, and
the exact zero difference between this strategy's hits and its random
comparison hits. Frontend tests cover family grouping and deterministic
portfolio exclusion.
