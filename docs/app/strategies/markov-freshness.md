(markov-freshness)=
# Markov Freshness

## Introduction

**Markov Freshness** is the production strategy with identifier `mkfr` and
short engine name **MKFR**. It is a default-enabled member of the **Markov &
Sequence** family.

For each number 1–49, MKFR stores its latest 20 drawn/not-drawn outcomes and
learns separate binary successor counts for every context order from 1 through
20. It shrinks supported context estimates through shorter contexts toward the
number's smoothed lifetime appearance rate, then ranks numbers by the context's
probability lift over that personal baseline.

```{admonition} Lift, not absolute probability
:class: important

MKFR does not rank candidates directly by their context probability. It ranks
the difference between context probability and each number's own smoothed
lifetime baseline. A number with a lower predicted probability can rank higher
when its current binary pattern creates a larger positive lift.
```

## Scope and role

MKFR asks:

> For each number independently, how did that number behave after previous
> occurrences of its current drawn/not-drawn suffix, and how much does that
> context change its estimate relative to its lifetime rate?

It produces a complete 1–49 ranking. The first six form the Top-6 used by
prediction grids, audits, effectiveness histories, comparisons, portfolios,
exports, Possible Draw, and selected ensemble consumers.

The strategy is standalone and has no hidden strategy dependencies. It does
not use exact numerical gaps as states, combine evidence across different
numbers, model six-number tickets jointly, or enforce ticket-shape constraints.

Some older internal or ensemble labels call `mkfr` **Markov Frequency**. The
Settings and main prediction interfaces display **Markov Freshness**; both
labels refer to the same production identifier and state.

## Binary candidate process

For draw \(t\) and candidate number \(n\), define

```{math}
Y_{t,n}=\mathbf1[n\in D_t]
=
\begin{cases}
1,&n\text{ is drawn},\\
0,&n\text{ is not drawn}.
\end{cases}
```

The UI description abbreviates these outcomes as **D** and **!D**. Every
completed draw supplies:

- six outcomes equal to 1; and
- 43 outcomes equal to 0.

MKFR maintains 49 independent binary sequences:

```{math}
(Y_{1,n},Y_{2,n},\ldots,Y_{t,n}),
\qquad n=1,\ldots,49.
```

Only the latest 20 outcomes for each number remain in its active context deque.
Transition counts learned from older contexts remain cumulative.

## Relationship to freshness and gap

The trailing zeros in a binary history encode the current waiting gap:

- history ending in `1` means the number appeared in the latest draw;
- history ending in `10` means one subsequent non-hit;
- history ending in `1000` means three subsequent non-hits.

MKFR retains more than gap alone. Two numbers with the same trailing-zero count
can have different earlier hit patterns inside the 20-draw context. This is why
the model is called Markov Freshness rather than an exact-gap lookup table.

It is distinct from the separate **Freshness** source used by Fresh Random,
which groups historical opportunities by one exact gap and estimates a
smoothed hit rate without a full binary suffix.

## Context orders and encoding

After \(t\) completed draws, an order-\(r\) context for number \(n\) is the
chronological suffix

```{math}
C_{t,n}^{(r)}=
(Y_{t-r+1,n},\ldots,Y_{t,n}),
\qquad 1\le r\le\min(t,20).
```

Production stores each binary context as an integer. The most recent outcome
is bit 0, the next most recent is bit 1, and so on:

```{math}
c_{t,n}^{(r)}=
\sum_{k=0}^{r-1}Y_{t-k,n}2^k.
```

For example, chronological display context `10` has most recent value 0 and is
stored as binary integer `10₂`, or 2. Transition tables are separated by order,
so the same integer value at two lengths remains two distinct states.

The detail display writes the selected context chronologically, oldest to
newest, using `1` and `0` rather than the internal reversed bit significance.

## Transition counts

For each candidate \(n\), order \(r\), and context \(c\), MKFR stores

```{math}
F_{n,r,c}=\text{number of observed next outcomes equal to 0},
```

```{math}
H_{n,r,c}=\text{number of observed next outcomes equal to 1}.
```

Their support is

```{math}
N_{n,r,c}=F_{n,r,c}+H_{n,r,c}.
```

When draw \(t\) becomes known, its target \(Y_{t,n}\) is added to every
available context ending at draw \(t-1\), from order 1 through 20. Only after
those transition rows are updated is \(Y_{t,n}\) appended to the active
history.

The model therefore accumulates up to 20 supervised transition observations
per number per completed draw once histories are long enough.

## Smoothed lifetime baseline

Let

```{math}
A_n(t)=\sum_{i=1}^{t}Y_{i,n}
```

be the lifetime appearance count after \(t\) draws. The uniform 6-from-49
reference rate is

```{math}
p_0=\frac6{49}=0.122449\ldots.
```

MKFR uses total prior strength 8:

```{math}
b_n(t)=
\frac{A_n(t)+8p_0}{t+8}.
```

This is the number-specific lifetime hit rate shrunk toward \(6/49\). At cold
start it equals \(p_0\); as history grows, the eight prior observations have
less influence.

The baseline allows candidates with different lifetime frequencies to be
evaluated relative to their own usual rates.

## Variable-order hierarchical backoff

For a forecast after draw \(t\), MKFR begins each number at

```{math}
p_n^{(0)}=b_n(t).
```

Orders are visited from 1 through the active maximum 20. For the current
order-\(r\) context, a row is ignored unless

```{math}
N_{n,r,c}\ge8.
```

For every supported row, production updates

```{math}
p_n^{(r)}=
\frac{H_{n,r,c}+8p_n^{(r-1)}}{N_{n,r,c}+8}.
```

If an order is unsupported, the current probability is retained unchanged and
the loop continues. A later, longer context can still be used when it reaches
support 8, shrinking toward the most recent probability produced by the
lifetime baseline and any shorter supported contexts.

This is hierarchical empirical-Bayes-style smoothing:

- sparse contexts remain close to their shorter-context prior;
- larger rows have more influence;
- valid but unseen next hits do not force zero probability; and
- the selected order is the longest supported current context, not one global
  order shared by all candidates.

The smoothing strength 8 and minimum support 8 are separate constants that
happen to have the same numeric value.

## Exact transition lift

Let \(p_n^*(t+1)\) be the probability after all supported orders have been
processed. MKFR's raw ranking value is

```{math}
L_n(t+1)=p_n^*(t+1)-b_n(t).
```

The lift is expressed in probability points:

- \(L_n>0\): the current binary context raises the estimate above the number's
  lifetime baseline;
- \(L_n=0\): no supported context changes the baseline, or supported updates
  cancel exactly; and
- \(L_n<0\): the context lowers the estimate below baseline.

This relative construction is crucial. Suppose candidate A has context
probability 21% and baseline 20%, while candidate B has context probability 15%
and baseline 10%. Their lifts are +1 and +5 percentage points, so candidate B
ranks higher before scaling despite its lower absolute probability.

## Min–max score and ranking

The 49 lifts are min–max scaled:

```{math}
S_n(t+1)=
\begin{cases}
\dfrac{L_n-L_{\min}}{L_{\max}-L_{\min}},
&L_{\max}>L_{\min},\\[8pt]
0,&L_{\max}=L_{\min}.
\end{cases}
```

Numbers are ranked by:

1. larger scaled lift;
2. larger current gap; then
3. smaller number.

The first six form the Top-6. Min–max scaling preserves non-tied lift order but
hides the absolute size and sign of lift. A score of 100% means the largest
current lift, not certainty of occurrence; a score of 0% can still represent a
negative, zero, or positive raw lift when it is merely the smallest of the 49.

## Causal lifecycle and leakage protection

For completed draw \(t\), production uses this order:

1. for every number, derive all contexts from histories ending at draw
   \(t-1\);
2. use actual \(Y_{t,n}\) to update failure/hit counts for those prior contexts;
3. update lifetime appearance and last-seen state with draw \(t\);
4. append \(Y_{t,n}\) to each number's 20-value history;
5. calculate baselines, supported context estimates, and lifts from history
   through \(t\); and
6. rank the forecast for target draw \(t+1\).

The target draw cannot train its own forecast. Its outcome is unavailable until
the pending prediction has already been issued. Prefix-invariance tests compare
the same reference prefix with and without a later appended draw and require
identical numbers, scores, details, and Top-6.

## Cold start and early history

Before any draw:

- every baseline equals \(6/49\);
- every history is empty;
- selected order and support are zero;
- context probability equals baseline; and
- every raw lift and scaled score is zero.

Ranking therefore falls through equal gaps to smaller number.

After draw 1, each history contains one bit, but no transition row has yet been
observed because no prior context existed for that target. Lift remains zero.
Numbers absent from draw 1 have current gap 1 and rank before numbers present in
draw 1 with gap 0 when scores tie.

Transitions begin accumulating with draw 2, but rows remain ignored until their
support reaches 8. Common all-zero short contexts generally qualify before rare
hit-rich or long contexts.

## Interpreting prediction details

Each number exposes five MKFR-specific fields:

- **Context probability** is the final hierarchically smoothed
  \(p_n^*(t+1)\).
- **Baseline probability** is the smoothed lifetime \(b_n(t)\).
- **Transition lift** is \(100(p_n^*-b_n)\) percentage points, including its
  sign.
- **Order r/20: context** reports the longest supported current order and its
  chronological binary suffix. An em dash indicates order 0.
- **Context support** is the failure-plus-hit count of the selected context.

The standard strategy payload separately supplies current gap, min–max score,
rank, and Top-6 membership.

Context probability is a smoothed model estimate, not a validated calibrated
probability. The grid's displayed percentage is the min–max lift score, not the
context probability shown in details.

## Endpoint diagnostic

After all 771 repository draws, every number has a 20-bit active history. The
longest supported current orders are distributed as follows:

| Selected order | Candidate count |
|---:|---:|
| 3 | 2 |
| 4 | 2 |
| 5 | 1 |
| 6 | 1 |
| 7 | 3 |
| 8 | 2 |
| 9 | 5 |
| 10 | 4 |
| 11 | 3 |
| 12 | 4 |
| 13 | 2 |
| 14 | 1 |
| 15 | 2 |
| 17 | 4 |
| 18 | 1 |
| 19 | 2 |
| 20 | 10 |

No candidate currently falls back to order 0–2 or order 16. Selected-context
support ranges from 8 to 95. This distribution is target-specific; it does not
mean an order-20 model is globally best.

The next-forecast Top-6 is:

| Rank | Number | Gap | Baseline | Context probability | Lift | Selected order | Support |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 8 | 12.0641% | 26.2498% | +14.1857 pp | 17 | 15 |
| 2 | 37 | 4 | 10.9088% | 21.9911% | +11.0823 pp | 11 | 22 |
| 3 | 40 | 16 | 14.5032% | 24.6286% | +10.1255 pp | 17 | 9 |
| 4 | 34 | 16 | 9.8818% | 19.4587% | +9.5769 pp | 20 | 12 |
| 5 | 31 | 1 | 11.9358% | 21.2224% | +9.2866 pp | 9 | 33 |
| 6 | 26 | 7 | 14.2464% | 23.3669% | +9.1205 pp | 8 | 35 |

Across all 49 candidates, endpoint lift ranges from −13.1910 percentage points
to +14.1857 percentage points. These figures describe one fitted prefix and
will change after new draws.

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
| Full replay | 770 | 572 | 0.742857 | 565.714 |
| Validation, target draws 121–520 | 400 | 297 | 0.742500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 191 | 0.764000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, records 189 hits
or 0.756000 per target.

All totals remain close to theoretical random expectation, with modestly
positive differences on these chosen slices. They are retrospective and arise
within a broad collection of compared strategies. They do not establish
statistical significance, calibration, stable future lift, or predictability.

## Core mathematical and statistical concepts

- **Binary time series:** each candidate has its own drawn/not-drawn sequence.
- **Variable-order Markov context:** suffix lengths 1–20 provide candidate- and
  target-specific states.
- **Bit encoding:** reversed bit significance stores binary suffixes compactly
  while order-specific tables preserve length.
- **Bernoulli transition counts:** each context row records failures and hits.
- **Prior smoothing:** lifetime rates shrink toward \(6/49\) with strength 8.
- **Hierarchical backoff:** supported longer contexts shrink toward the estimate
  from baseline and shorter contexts.
- **Minimum support:** rows with fewer than eight successors are skipped.
- **Transition lift:** context probability is centered on a number-specific
  baseline before candidates are compared.
- **Min–max normalization:** signed lifts become relative 0–1 display scores.
- **Hypergeometric overlap:** Top-6 efficacy uses the correct
  without-replacement null.

## Limitations and responsible interpretation

- **Near-random replay:** historical Top-6 totals are close to theoretical
  random expectation and do not demonstrate a stable edge.
- **Independent candidate models:** 49 binary processes ignore the constraint
  that exactly six numbers occur together and omit pair or ticket structure.
- **Severe class imbalance:** every draw supplies 43 failures and six hits;
  context rows can be dominated by non-hits.
- **Combinatorial sparsity:** order \(r\) permits \(2^r\) binary states per
  number, so most long contexts have little or no support.
- **Multiple context inspection:** production evaluates up to 20 orders for 49
  candidates, increasing opportunities for noisy historical patterns.
- **Longest-supported selection:** support 8 is enough to activate a long row
  even when its estimate is unstable; there is no held-out order selector.
- **Cumulative transitions:** old rows never decay and cannot rapidly adapt to
  a genuine distribution change.
- **Fixed smoothing:** prior strength and support threshold are engineering
  constants rather than learned uncertainty estimates.
- **Dynamic empirical prior:** longer contexts shrink toward shorter estimates
  derived from the same sequence, not independent evidence.
- **Lift can favor low baselines:** a large relative improvement can outrank a
  higher absolute context probability.
- **Min–max information loss:** scaled scores hide lift sign and magnitude and
  are not comparable across target draws.
- **Gap tie influence:** current gap is not in the raw lift formula but resolves
  exact score ties.
- **No calibrated probability guarantee:** context estimates are descriptive
  smoothed rates, not validated occurrence probabilities.
- **Label ambiguity:** legacy “Markov Frequency” labels can obscure that the
  state is a binary freshness pattern.
- **Retrospective comparison:** orders, thresholds, and the broader strategy
  set were developed with historical outcomes available.
- **No guaranteed predictability:** repeated binary patterns can arise by
  chance under independent draws and need not recur with the same successor.

## Implementation map

The production implementation is concentrated in
`src/rand_ai/strategy_prediction.py`:

- `_MKFR_MAX_ORDER`, `_MKFR_PRIOR_STRENGTH`, and
  `_MKFR_MIN_CONTEXT_SUPPORT` define order 20, strength 8, and support 8;
- `_StrategyState.mkfr_histories` owns 49 bounded binary deques;
- `_StrategyState.mkfr_transitions` owns per-number, per-order context rows;
- `train` encodes histories ending before the current draw and records its
  binary successor;
- `remember` appends all 49 current outcomes after transition training;
- `_mkfr_baseline_probability` calculates the lifetime Bernoulli baseline;
- `_mkfr_probability` applies variable-order support gating and hierarchical
  backoff;
- `_mkfr_scores` calculates transition lift, min–max scores, and details;
- `_ranking_from_scores` applies score, gap, and number tie-breaking; and
- `build_strategies` serializes `mkfr` with engine name `MKFR` when requested.

The exact behavior is covered in `tests/test_strategy_prediction.py`, including
joint binary context encoding, failure/hit counts, unsupported-order backoff,
supported order 20, lift rather than absolute probability ranking, cold-start
prior, 20-draw truncation, details, and future-draw prefix invariance.
