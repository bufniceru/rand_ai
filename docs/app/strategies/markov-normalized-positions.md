(markov-normalized-positions)=
# Markov Normalized Positions

## Introduction

**Markov Normalized Positions** is the production strategy with identifier
`mknp` and short engine name `MKNP`. It is a default-enabled member of the
**Markov & Sequence** family.

The strategy models the *shape* of a sorted six-number draw independently of
where that shape starts. It translates every draw so that its first number is
1, learns how the remaining five normalized positions evolve, predicts a
distribution for each position and for the starting location, and decodes
those distributions into valid six-number tickets. The final number scores are
the marginal weights of the decoded tickets that contain each number.

This is a categorical, variable-order sequence model with analogue matching
and constrained decoding. It is not a claim that lottery draws contain a
predictable Markov process.

## Scope and role

MKNP answers this question:

> Given the normalized shapes seen through the latest completed draw, which
> numbers receive the most mass after plausible next shapes are translated
> across all valid starting locations?

It produces a complete ranking of numbers 1–49. The first six form its Top-6;
the full ranking can also be consumed by comparison views, effectiveness
history, portfolios, exports, Possible Draw, and ensemble strategies.

MKNP is distinct from two nearby engines:

- **Markov Spaces** models six circular empty-space counts whose sum is 43.
- **Markov Normalized Positions** models five cumulative positions relative to
  the first ball and learns a separate first-number anchor.
- **Markov Relative Dispersion** builds on normalized positions but adds
  additional relative-shape diagnostics and scoring terms.

MKNP is standalone when selected; it does not require another strategy to run
as a hidden dependency.

## Prediction representation

Let a completed draw, sorted in increasing order, be

```{math}
1\le x_1<x_2<\cdots<x_6\le49.
```

The internal anchor offset is

```{math}
a=x_1-1\in\{0,\ldots,43\},
```

and the normalized position vector is

```{math}
\mathbf u=(u_1,\ldots,u_6),
\qquad
u_i=x_i-x_1+1.
```

Therefore

```{math}
u_1=1,
\qquad
1=u_1<u_2<\cdots<u_6\le49,
\qquad
x_i=a+u_i.
```

For example,

```{math}
(4,11,18,29,33,47)
\longmapsto
a=3,
\quad
\mathbf u=(1,8,15,26,30,44).
```

Adding the same offset to every original number leaves \(\mathbf u\)
unchanged. The representation is consequently translation-invariant until the
anchor distribution restores absolute location.

The last normalized position is the inclusive spread,

```{math}
u_6=x_6-x_1+1.
```

The associated wraparound empty space is

```{math}
49-u_6=(x_1-1)+(49-x_6).
```

Thus MKNP retains the internal cumulative geometry of a draw while separating
its position on the 1–49 line.

## Valid position domains

Only positions 2–6 are learned because the first normalized position is always
1. For ordinal position \(j\in\{2,\ldots,6\}\), the production domain is

```{math}
V_j=\{j,j+1,\ldots,43+j\}.
```

The concrete ranges are:

| Normalized position | Valid values |
|---|---:|
| \(u_2\) | 2–45 |
| \(u_3\) | 3–46 |
| \(u_4\) | 4–47 |
| \(u_5\) | 5–48 |
| \(u_6\) | 6–49 |

These bounds ensure enough room for the remaining ordered positions. The
decoder additionally enforces strict increase, so independently plausible
position values cannot form an invalid or duplicated draw.

## Learned state

For each of the five variable positions, MKNP maintains:

- the latest 20 observed normalized values;
- categorical transition tables for context orders 1 through 20; and
- lifetime counts for every valid position value.

It also stores:

- 44 first-number anchor counts, indexed internally from 0 through 43; and
- the chronological observation sequence \((\mathbf u_t,a_t)\), used for
  analogue matching.

The history deques are bounded at 20, while transition counts, lifetime counts,
anchor counts, and the observation sequence accumulate across completed draws.

## Lifetime categorical baseline

Let \(C_j(v)\) be the lifetime count for value \(v\) at normalized position
\(j\), and let \(h\) be the number of remembered draws. The baseline is

```{math}
b_j(v)=
\frac{C_j(v)+8/|V_j|}{h+8},
\qquad v\in V_j.
```

Values outside \(V_j\) receive zero. The total prior strength is 8 and is
distributed uniformly over the position's valid categories. This prevents
zero probabilities, makes unseen values possible, and yields a uniform
position baseline before any draw is remembered.

## Exact variable-order Markov branch

For a proposed next value \(v\), MKNP begins with \(p_0(v)=b_j(v)\). It then
examines the current suffix at every available order from 1 through 20. A
context is ignored unless it has at least 8 recorded successor observations.

For every supported order \(r\), the probability is updated sequentially:

```{math}
p_r(v)=
\frac{N_r(v)+8p_{r-1}(v)}{N_r+8},
```

where \(N_r(v)\) is the number of times the order-\(r\) context was followed by
\(v\), and \(N_r\) is that context's total successor support.

This is hierarchical backoff. A longer supported context does not replace the
shorter estimate outright; it shrinks toward the probability already produced
by the lifetime and shorter-context levels. The displayed exact order is the
longest supported order reached for the most probable value at that position.
An order of 0 and support of 0 mean no exact context met the threshold.

## Analogue branch

Exact order-20 contexts are sparse. MKNP therefore also compares the current
normalized-position history with historical contexts and borrows their known
successors.

At most the latest 512 eligible historical target observations are considered.
For an analogue target \(t\), the comparison length is

```{math}
o_t=\min(20,\text{available predecessor history}).
```

At lag \(\ell\), the five-position distance is

```{math}
\delta_{t,\ell}
=\frac{1}{5\times48}
\sum_{j=2}^{6}
\left|u_{h-\ell+1,j}-u_{t-\ell,j}\right|.
```

Recent lags receive more weight through \(\gamma=0.86\):

```{math}
d_t=
\frac{\sum_{\ell=1}^{o_t}\gamma^{\ell-1}\delta_{t,\ell}}
     {\sum_{\ell=1}^{o_t}\gamma^{\ell-1}}.
```

The analogue weight is

```{math}
w_t=
\exp(-10d_t)
\times 2^{-A_t/800}
\times\left(0.35+0.65\frac{o_t}{20}\right),
```

where \(A_t\) is the source age used by the implementation. The three factors
reward shape-context similarity, decay older examples with an 800-draw
half-life, and reduce confidence in short contexts. Each weight is attached to
the already observed successor shape and anchor of that historical context.

Effective analogue support is reported as

```{math}
n_{\mathrm{eff}}=
\frac{\left(\sum_t w_t\right)^2}{\sum_t w_t^2}.
```

It equals the raw count only when weights are equal and becomes smaller when a
few analogues dominate.

## Position and anchor distributions

For each normalized position, weighted analogue counts are smoothed toward the
lifetime baseline with prior strength 4 and normalized:

```{math}
q_j(v)=
\operatorname{Norm}\!\left(
\sum_t w_t\mathbf1[u_{t,j}=v]+4b_j(v)
\right).
```

The exact-context probabilities are separately normalized to \(e_j(v)\). The
final position distribution is the fixed blend

```{math}
P_j(v)=0.70q_j(v)+0.30e_j(v),
```

followed by normalization. There is no adaptive blend weight.

The anchor uses a separate lifetime baseline

```{math}
b_a(k)=\frac{C_a(k)+4/44}{h+4},
\qquad k\in\{0,\ldots,43\},
```

then combines weighted analogue anchor counts with another total prior of 4:

```{math}
P_a(k)=
\operatorname{Norm}\!\left(
\sum_t w_t\mathbf1[a_t=k]+4b_a(k)
\right).
```

The anchor does not have its own exact variable-order Markov branch.

## Constrained shape decoding

Independent position maxima can violate ordering, so MKNP decodes complete
shapes with a beam search.

The beam starts at \((1,)\). At each of the five learned positions it appends
every valid value larger than the previous one and adds its log probability:

```{math}
L(1,u_2,\ldots,u_6)=
\sum_{j=2}^{6}\log P_j(u_j).
```

For each ending value, only the best 8 partial paths are retained. This is a
per-ending-value beam width, not a single global set of eight tickets. Equal
path weights are resolved deterministically by the position tuple.

For every retained shape with spread \(s=u_6\), MKNP enumerates all anchors

```{math}
a\in\{0,\ldots,49-s\},
```

and reconstructs a valid ticket

```{math}
T=(a+u_1,\ldots,a+u_6).
```

Its log weight is

```{math}
L(T)=L(\mathbf u)+\log P_a(a).
```

The implementation subtracts the largest generated log weight before
exponentiation. This standard log-sum-exp stabilization changes no relative
ticket weight and avoids numerical underflow.

## Number marginals, display score, and ranking

Let \(\mathcal T\) be the generated tickets and let \(W(T)\) be their stabilized
positive weights. The true decoded marginal for number \(n\) is

```{math}
M(n)=
\frac{\sum_{T\in\mathcal T}W(T)\mathbf1[n\in T]}
     {\sum_{T\in\mathcal T}W(T)}.
```

Because every generated ticket contains six distinct numbers,

```{math}
\sum_{n=1}^{49}M(n)=6.
```

The application shows this marginal in the number details. The strategy score
used for the grid and ranking is instead min–max scaled across all 49
marginals:

```{math}
S(n)=\frac{M(n)-M_{\min}}{M_{\max}-M_{\min}}.
```

If all marginals are equal, all scaled scores are zero. The displayed strategy
percentage is therefore a relative within-draw score, not another calibrated
probability. Use the **Marginal probability** detail when interpreting ticket
mass.

Numbers are ranked by:

1. larger scaled score;
2. larger current gap; then
3. smaller number.

Min–max scaling is monotone, so it does not change score order unless all
marginals are equal. The Top-6 is formed from the first six individual-number
ranks; it is not necessarily one of the generated beam tickets.

## Causal lifecycle and leakage protection

For each completed draw \(t\), production processing occurs in this order:

1. train transition tables using histories ending at \(t-1\) and the observed
   normalized values from draw \(t\);
2. remember draw \(t\) in histories, lifetime counts, anchor counts, and the
   analogue sequence;
3. build the forecast for draw \(t+1\); and
4. evaluate that forecast only after draw \(t+1\) becomes known.

The draw being forecast cannot train its own prediction. Appending later draws
does not alter rankings already produced for an earlier prefix; the automated
tests explicitly check this prefix invariance.

Historical analogue targets are also causal: their successor shapes and
anchors are already completed observations. No future successor is read for
the current query.

## Cold start and short history

Before observations, every position and anchor baseline is symmetric and no
analogue exists. Nevertheless, beam constraints and deterministic tie handling
can make the final number marginals nonuniform: the decoder retains a bounded
subset of otherwise equally weighted shapes and translates each shape through
only its valid anchors.

After one remembered draw, lifetime position and anchor counts are available,
but there are still no eligible analogues. Exact contexts remain unavailable
until a context has accumulated the minimum support of 8. As history grows,
the analogue branch can operate with shorter contexts; its length-confidence
factor explicitly reduces their influence.

## Interpreting prediction details

Each number exposes the following fields:

- **Marginal probability** is \(M(n)\), the normalized mass of generated
  tickets containing the number.
- **Random baseline** is \(6/49=12.2449\%\), shown as a reference for one
  uniformly random six-number ticket.
- **Best generated draw** is the highest-weight generated ticket containing
  that number.
- **Best normalized positions** is the translated shape of that explanatory
  ticket.
- **Spread** is its inclusive first-to-last span; **wraparound space** is
  \(49-\text{spread}\).
- **First-number anchor** is the first displayed number of that ticket. The
  internal anchor offset is one less.
- **Analogue support** reports effective support and raw candidate count.
- **Exact orders /20** reports the selected order for the five learned
  positions and the minimum-to-maximum support of those selected contexts.
- **Valid-shape beam width 8** identifies the pruning limit.

The best generated draw explains one number's largest ticket contribution. It
is not the same object as the strategy's independently ranked Top-6.

## Endpoint diagnostic

After all 771 repository draws, the next-forecast state has:

| Diagnostic | Endpoint value |
|---|---:|
| Stored observations | 771 |
| Analogue candidates | 512 |
| Effective analogue support | 475.617 |
| Selected exact orders for positions 2–6 | 1, 1, 1, 1, 1 |
| Corresponding exact-context supports | 24, 40, 38, 15, 8 |
| Retained complete normalized shapes | 342 |
| Generated valid translated tickets | 7,483 |

The endpoint uses only order-1 exact contexts despite supporting orders up to
20, illustrating the sparsity of exact categorical suffixes. The figures
describe one fitted historical state and will change when new draws are added.

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
draws produces 770 evaluable target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 583 | 0.757143 | 565.714 |
| Validation, target draws 121–520 | 400 | 302 | 0.755000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 185 | 0.740000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, also records 185
hits or 0.740000 per target.

All totals are retrospective. The feature representation, constants, decoder,
and broader strategy collection were developed with historical information
available. These figures do not establish statistical significance,
calibration, stable future lift, or predictability.

## Core mathematical and statistical concepts

- **Translation invariance:** normalized positions remove absolute location;
  the anchor distribution restores it during decoding.
- **Categorical Dirichlet-style smoothing:** total prior strengths 8 and 4
  prevent zero-probability categories.
- **Variable-order Markov modeling:** exact suffixes from order 1 through 20
  condition position distributions when support reaches 8.
- **Hierarchical backoff:** longer-context estimates shrink toward supported
  shorter-context and lifetime probabilities.
- **Analogue or nearest-context estimation:** exponentially weighted shape
  similarity transfers observed historical successor mass.
- **Recency decay:** an 800-draw half-life gradually discounts older analogue
  targets without a hard window beyond the 512-candidate cap.
- **Effective sample size:** \((\sum w)^2/\sum w^2\) measures concentration of
  analogue weights.
- **Constrained beam search:** joint monotone shapes are approximated while
  preserving legal six-number tickets.
- **Marginalization:** ticket weights are summed into number-level inclusion
  mass.
- **Hypergeometric evaluation:** Top-6 overlap has the correct without-
  replacement null model.

## Limitations

- Lottery draws may be independent; a Markov interpretation can fit historical
  variation without representing a persistent data-generating mechanism.
- Translation normalization deliberately discards absolute location from the
  shape model and restores it only through a separate anchor distribution.
- The five positions are estimated separately before constrained decoding, so
  dependence is represented only approximately by context matching and the
  beam constraints.
- The 512-candidate cap, order 20, support threshold 8, decay constants,
  smoothing strengths, 70/30 blend, and beam width 8 are fixed modeling
  choices rather than universally optimal values.
- Per-ending-value beam pruning omits valid shapes and can affect marginals,
  especially at cold start or under near ties.
- Min–max scaling removes the absolute level and spread of marginals from the
  displayed score; scores from different target draws are not directly
  comparable.
- The Top-6 is selected from number marginals and need not itself be a valid
  high-weight joint ticket from the decoder.
- Historical replay is dataset-dependent and vulnerable to retrospective
  selection and multiple-strategy comparison.
- No displayed score, marginal, benchmark total, or fitted context guarantees a
  future advantage.

## Implementation map

The production implementation is concentrated in
`src/rand_ai/strategy_prediction.py`:

- `_normalized_positions_for_numbers` builds the translated representation;
- `_StrategyState` owns histories, transition tables, lifetime counts, anchor
  counts, and observations;
- `train` updates exact transition tables before the current draw is remembered;
- `remember` updates histories and analogue observations;
- `_mknp_valid_values` defines legal categorical domains;
- `_mknp_baseline_probability` and `_mknp_probability` implement smoothing and
  exact-context backoff;
- `_mknp_analogue_weights` computes analogue distances and weights;
- `_mknp_distributions` forms the five position and anchor distributions;
- `_mknp_shape_beam` performs constrained shape decoding;
- `_mknp_scores` translates shapes, marginalizes ticket weights, and creates
  user-facing details; and
- `build_strategies` applies the standard gap-and-number ranking and serializes
  `mknp` when requested.

The causal and mathematical behavior is covered in
`tests/test_strategy_prediction.py`, including normalization and spread,
state updates, categorical smoothing and backoff, valid decoded shapes,
score/details behavior, maximum history order, and prefix invariance.
