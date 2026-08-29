(entropy)=
# Entropy

## Introduction

**Entropy** is the production strategy with identifier `entropy` and short
engine name **Entr**. It is a default-enabled member of the **Frequency &
Recency** family.

The strategy measures how evenly each completed draw divides the circular
1–49 number range into six arcs. For every number, it remembers the entropy of
the draws in which that number appeared, the share of those appearances in
high-entropy draws, and the number's current overdue gap. These three terms are
combined into a complete ranking of numbers 1–49.

```{admonition} Structural entropy, not forecast uncertainty
:class: important

The model calculates entropy from the six circular distances inside a draw. It
does **not** calculate the entropy of a number's future probability, the
uncertainty of the Top-6 forecast, or the entropy of lifetime number
frequencies.
```

## Scope and role

Entropy asks:

> Which numbers have historically appeared in more evenly spaced circular
> draws, especially in draws above the fixed high-entropy threshold, while
> also receiving a bounded overdue adjustment?

Its first six ranks form the Top-6 prediction used by prediction grids, audits,
effectiveness histories, comparisons, portfolios, exports, Possible Draw, and
selected ensemble consumers.

The engine is standalone and has no hidden strategy dependencies. It has no
trainable coefficients, transition table, classifier, recent entropy window,
decay factor, or probability calibration stage. Its three weights and
thresholds are fixed production constants.

## Distinct entropy concepts in the application

Several application features use the word *entropy*:

- the **Entropy strategy** documented here uses six circular draw distances and
  associates each draw's result with its six numbers;
- the **Possible Draw** entropy status computes the same circular ticket-shape
  measure for the six currently selected numbers, but does not alter this
  strategy's historical state;
- the randomness diagnostic's **normalized frequency entropy** measures how
  evenly all historical appearances are distributed over 49 number categories;
  and
- Markov Relative Dispersion uses an internal five-gap shape entropy inside a
  different model.

These quantities have different sample spaces, denominators, and
interpretations.

## Circular gap construction

Let a completed draw be sorted as

```{math}
1\le n_1<n_2<\cdots<n_6\le49.
```

Production calculates five forward distances and one wraparound distance:

```{math}
d_i=n_{i+1}-n_i,
\qquad i=1,\ldots,5,
```

```{math}
d_6=49+n_1-n_6.
```

Every \(d_i\) is a positive integer and

```{math}
\sum_{i=1}^{6}d_i=49.
```

These are distances between drawn numbers, not counts of empty numbers. The
corresponding empty spaces would be \(d_i-1\) and would sum to 43.

The wraparound term makes the representation circular. Rotating a ticket
around the number ring permutes its distances without changing its entropy.

## Normalized Shannon entropy

Each circular distance becomes a share of the ring:

```{math}
p_i=\frac{d_i}{49},
\qquad
\sum_{i=1}^{6}p_i=1.
```

The draw's base-2 Shannon entropy is

```{math}
H(D)=-\sum_{i=1}^{6}p_i\log_2p_i.
```

Production normalizes by the six-category maximum and expresses the result as
a percentage:

```{math}
E(D)=100\frac{H(D)}{\log_2 6}.
```

The logarithm base does not affect the normalized value as long as the same
base is used in numerator and denominator.

### Geometric interpretation

- **High entropy** means the six circular distances are relatively similar, so
  the selected numbers are distributed more evenly around the ring.
- **Low entropy** means one or a few distances dominate, so the draw is more
  clustered or uneven.
- Permuting the six distances changes neither entropy nor the score assigned to
  that completed draw.

The mathematical normalization permits 100% when all six shares equal
\(1/6\). Exact equality is impossible with 49 integer positions. The most even
integer composition is a permutation of

```{math}
(9,8,8,8,8,8),
```

whose entropy is approximately 99.9434%. The most uneven valid composition is
a permutation of

```{math}
(44,1,1,1,1,1),
```

whose normalized entropy is approximately 27.5580%.

## Per-number historical state

After \(t\) completed draws, let

```{math}
A_n(t)=\sum_{i=1}^{t}\mathbf1[n\in D_i]
```

be the appearance count for number \(n\). The strategy accumulates the entropy
of every draw containing that number:

```{math}
T_n(t)=\sum_{i=1}^{t}E(D_i)\mathbf1[n\in D_i].
```

It also counts appearances in draws meeting the fixed high-entropy threshold:

```{math}
K_n(t)=
\sum_{i=1}^{t}
\mathbf1[n\in D_i]\mathbf1[E(D_i)\ge92].
```

One completed draw contributes the same \(E(D_i)\) to all six numbers it
contains. Numbers absent from that draw receive no entropy update.

The state is cumulative for the full available prefix. There is no rolling
window or recency weighting of old entropy observations.

## Average entropy term

For a number that has appeared at least once, its conditional historical mean
is

```{math}
\bar E_n(t)=\frac{T_n(t)}{A_n(t)}.
```

An unseen number receives the fixed fallback

```{math}
\bar E_n(t)=50.
```

The fallback is an engineering default, not a posterior mean or a pseudocount.
After the number's first appearance, it is replaced completely by that draw's
entropy.

This term measures the typical structural entropy of draws *conditioned on the
number having appeared*. It does not reward a number merely for having more
appearances because the total is divided by its appearance count.

## High-entropy share

For a previously observed number, production calculates

```{math}
Q_n(t)=\frac{K_n(t)}{A_n(t)}.
```

An unseen number receives \(Q_n(t)=0\). The threshold event is exactly

```{math}
E(D_i)\ge92\%.
```

The threshold is inclusive. The ratio is an unsmoothed empirical proportion:
there is no Beta prior, confidence interval, minimum appearance requirement, or
shrinkage toward the overall high-entropy rate.

Average entropy and high-entropy share are related rather than independent.
The threshold term deliberately gives extra weight to the upper tail of the
same draw-entropy values already represented by \(\bar E_n\).

## Overdue adjustment

Let \(G_n(t)\) be the current gap after draw \(t\): the number of consecutive
completed draws since number \(n\) last appeared. A number in the latest draw
has gap 0. An unseen number has gap \(t\).

Production converts gap to a capped fraction:

```{math}
O_n(t)=
\operatorname{clamp}\!\left(\frac{G_n(t)}{28},0,1\right).
```

The overdue contribution grows linearly through gap 28 and remains at its
maximum for every larger gap. The divisor 28 is fixed and is not inferred from
the historical waiting-time distribution.

This adjustment is a ranking heuristic. Under independent uniform draws, a
long absence does not make a number causally due.

## Exact raw score

The three terms are combined on a 100-point scale:

```{math}
R_n(t)=
0.55\bar E_n(t)
+0.30\bigl(100Q_n(t)\bigr)
+0.15\bigl(100O_n(t)\bigr).
```

Equivalently,

```{math}
R_n(t)=0.55\bar E_n(t)+30Q_n(t)+15O_n(t).
```

The nominal contributions are:

| Component | Fixed weight | Maximum point contribution |
|---|---:|---:|
| Conditional average entropy | 55% | 55 |
| High-entropy appearance share | 30% | 30 |
| Capped overdue fraction | 15% | 15 |

The weights are not fitted by regression and do not adapt from effectiveness.
The first two components reuse the same entropy history in different forms.

## Min–max score and ranking

Raw scores are transformed across the 49 candidates:

```{math}
S_n(t)=
\begin{cases}
\dfrac{R_n(t)-R_{\min}(t)}
      {R_{\max}(t)-R_{\min}(t)},
&R_{\max}(t)>R_{\min}(t),\\[10pt]
0,&R_{\max}(t)=R_{\min}(t).
\end{cases}
```

Numbers are ranked by:

1. larger scaled score;
2. larger current gap; then
3. smaller number.

The Top-6 consists of the first six ranks. Because gap is already a 15% raw
component and is also the first tie-break, overdue state can affect ordering in
two distinct ways when raw scores tie exactly.

Min–max scaling preserves non-tied raw-score order but removes absolute score
level and spread. A displayed score of 100% means the largest raw value for
that target; it is not a calibrated occurrence probability.

## Causal lifecycle and leakage protection

For each completed draw \(t\), production proceeds as follows:

1. calculate \(E(D_t)\) from the six known numbers in draw \(t\);
2. for each number in \(D_t\), increment its appearance count, add
   \(E(D_t)\), and increment its high-entropy count when \(E(D_t)\ge92\);
3. update last-seen positions and current gaps;
4. calculate all 49 raw and min–max scores; and
5. retain the resulting ranking as the forecast for draw \(t+1\).

The entropy of target draw \(t+1\) is unavailable when its forecast is built
and cannot update that forecast. It enters the state only after the draw is
completed, affecting later targets. Appending later draws therefore cannot
alter an earlier prefix ranking.

## Cold start and early behavior

Before any completed draw:

- every number is unseen;
- average entropy is 50%;
- high-entropy share is 0%;
- gap and overdue fraction are 0; and
- every raw score is 27.5.

Min–max scaling therefore returns zero for every number, and ranking falls
through to smaller number after equal gaps.

After the first draw, its six numbers use that draw's actual entropy and a
high-entropy share of either 0% or 100%. Their gaps are zero. All 43 unseen
numbers retain the 50% fallback and receive gap 1. This can create a sharp
early separation because the historical proportions are not smoothed.

## Interpreting application fields

Each number exposes two Entropy-specific details:

- **Average entropy** is \(\bar E_n\), the mean normalized circular-gap entropy
  of completed draws containing the number.
- **High-entropy share** is \(Q_n\), the fraction of that number's appearances
  in draws with entropy at least 92%.

The standard prediction payload separately supplies current gap, normalized
score, rank, and Top-6 membership. The overdue fraction and unscaled raw score
are not displayed directly, but can be reconstructed from the documented
formula and current gap.

The percentage score shown in the grid is the min–max value \(S_n\), not
\(\bar E_n\), \(Q_n\), or a chance that the number will appear.

## Endpoint diagnostic

Across the repository's 771 completed draws, the observed draw-entropy
distribution is:

| Diagnostic | Value |
|---|---:|
| Mean entropy | 83.4772% |
| Population standard deviation | 8.7462 percentage points |
| Minimum observed entropy | 49.0985% |
| Maximum observed entropy | 99.2902% |
| Draws at or above 92% | 117 of 771 (15.1751%) |

These are descriptive values for this dataset, not a calibrated random-null
distribution.

After all 771 draws, the next-forecast Top-6 is:

| Rank | Number | Appearances | Gap | Average entropy | High-entropy share | Raw score | Scaled score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 33 | 85 | 39 | 83.7935% | 16.4706% | 66.0276 | 1.000000 |
| 2 | 38 | 87 | 24 | 84.9252% | 20.6897% | 65.7729 | 0.985444 |
| 3 | 29 | 79 | 23 | 84.6854% | 16.4557% | 63.8351 | 0.874705 |
| 4 | 40 | 112 | 16 | 84.0137% | 16.9643% | 59.8682 | 0.648006 |
| 5 | 45 | 86 | 20 | 83.2037% | 10.4651% | 59.6159 | 0.633583 |
| 6 | 48 | 88 | 14 | 83.9416% | 19.3182% | 59.4633 | 0.624865 |

The endpoint raw-score range is 48.5291–66.0276. Number 33's gap exceeds 28,
so its overdue contribution is already capped. These values describe one
fitted historical state and will change with new draws.

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
| Full replay | 770 | 582 | 0.755844 | 565.714 |
| Validation, target draws 121–520 | 400 | 312 | 0.780000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 178 | 0.712000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, records 177 hits
or 0.708000 per target.

The full and validation slices exceed theoretical random expectation, but the
holdout and latest slice are below it. The apparent historical excess does not
persist. These retrospective results do not establish statistical
significance, calibration, stable future lift, or predictability.

## Core mathematical and statistical concepts

- **Circular composition:** six positive distances sum to the fixed ring size
  49.
- **Shannon entropy:** \(-\sum p_i\log p_i\) measures evenness of the six arc
  shares.
- **Normalized entropy:** division by \(\log 6\) maps the theoretical
  six-category maximum to one.
- **Rotation invariance:** entropy depends on the multiset of circular
  distances, not the starting number or traversal position.
- **Conditional sample mean:** average entropy is estimated only from draws
  containing the candidate number.
- **Empirical tail proportion:** high-entropy share counts the conditional
  frequency above a fixed 92% threshold.
- **Censoring or saturation:** overdue contribution stops increasing after gap
  28.
- **Fixed linear scoring:** three engineered quantities are combined using
  55/30/15 weights.
- **Min–max normalization:** raw values become relative within-target ranking
  scores.
- **Hypergeometric overlap:** Top-6 efficacy uses the without-replacement null
  for overlap between two six-element subsets of 49.

## Limitations and responsible interpretation

- **Negative holdout behavior:** the current holdout and latest-slice Top-6
  results are below theoretical random expectation.
- **No causal entropy mechanism:** a number's historical association with
  evenly spaced draws does not imply that it will occur in another such draw.
- **Conditional-selection noise:** numbers with fewer appearances have noisier
  conditional means and proportions, but receive no uncertainty penalty.
- **Unsmoothed high share:** one early high-entropy appearance produces 100%,
  while one non-high appearance produces 0%.
- **Arbitrary unseen fallback:** 50% is a fixed default rather than a
  data-derived prior.
- **Double use of entropy:** average entropy and the ≥92% indicator are
  correlated summaries of the same completed-draw values.
- **Fixed threshold:** 92% is neither an estimated quantile nor a significance
  boundary.
- **Cumulative history:** old entropy associations never decay and can dominate
  recent behavior.
- **Overdue fallacy risk:** gap is a heuristic; under independent draws, past
  absence does not increase the next-draw probability.
- **Gap saturation:** gaps 28 and 100 receive the same 15-point overdue
  contribution.
- **Scale compression:** min–max scoring hides absolute raw-score differences
  and prevents direct comparison across target draws.
- **Tie-break reuse:** current gap can influence both raw score and exact-score
  ties.
- **Shape compression:** one entropy value discards which arcs are large, their
  order, number identities, and all other draw relationships.
- **No calibrated output:** neither raw score nor scaled score is an occurrence
  probability.
- **Retrospective selection:** the constants and wider strategy collection were
  developed with historical data available.
- **No guaranteed predictability:** observed structural associations can be
  random fluctuations and may disappear on untouched draws.

## Implementation map

The production implementation is concentrated in
`src/rand_ai/strategy_prediction.py`:

- `_gap_entropy_percent` sorts the draw, constructs six circular distances,
  and calculates normalized base-2 entropy;
- `_StrategyState.entropy_totals` stores cumulative entropy points for each
  number;
- `_StrategyState.high_entropy_hits` stores conditional threshold counts;
- `_StrategyState.remember` updates entropy state only after a draw is
  completed;
- `_StrategyState.current_gaps` derives the overdue state;
- `_StrategyState._entropy_scores` applies the unseen fallbacks, 92% threshold
  history, 55/30/15 formula, and min–max scaling;
- `_ranking_from_scores` applies score, gap, and number ordering; and
- `build_strategies` serializes `entropy` with engine name `Entr` when
  requested.

Registration, ordering, and defaults are exercised in
`tests/test_strategy_prediction.py`, `tests/test_gui_bridge.py`, and the
frontend strategy-family and selection tests. The shared walk-forward suite
tests cover causal serialization, rankings, details, and efficacy history.
