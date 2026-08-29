(markov-relative-dispersion)=
# Markov Relative Dispersion

## Introduction

**Markov Relative Dispersion** is the production strategy with identifier
`mkrd` and short engine name **MKRD**. It is a default-disabled, opt-in member
of the **Markov & Sequence** family.

The strategy translates each six-number draw to normalized positions, derives
a relative-dispersion profile, compares the latest profile sequence with
historical contexts, and borrows their known successor shapes and anchors. A
separate exact variable-order Markov branch models the five normalized
positions. The two branches are combined and decoded into valid translated
tickets whose number marginals form the final ranking.

```{admonition} Relative shape with retained coverage
:class: important

Most profile fields are translation- and scale-normalized, but MKRD is not
completely scale-free. Its analogue distance assigns 20% weight to absolute
span coverage, preserving whether a shape occupies a narrow or wide part of
the 1–49 range.
```

## Scope and role

MKRD asks:

> When recent completed draws resemble the relative shape, dispersion, and
> coverage history preceding an earlier draw, which normalized successor
> shapes and starting anchors followed those historical contexts?

It produces a complete ranking of numbers 1–49. Its first six form the Top-6
used by prediction grids, audits, effectiveness histories, comparisons,
portfolios, exports, and Possible Draw.

The engine is standalone when selected and maintains state independent of
Markov Normalized Positions. It shares helper mathematics and the constrained
decoder, but does not require `mknp` to run as a hidden strategy.

MKRD differs from related engines:

- **Markov Spaces** models six circular empty-space counts summing to 43.
- **Markov Normalized Positions** compares histories using absolute translated
  position differences.
- **Markov Relative Dispersion** compares histories using a weighted profile of
  relative internal positions, coverage, gap dispersion, internal entropy, and
  center balance; historical successors still contribute integer normalized
  positions and anchors to the decoder.

## Normalized position representation

Let a completed draw be sorted as

```{math}
1\le x_1<x_2<\cdots<x_6\le49.
```

The internal anchor offset is

```{math}
a=x_1-1\in\{0,\ldots,43\},
```

and normalized positions are

```{math}
u_i=x_i-x_1+1.
```

Therefore

```{math}
\mathbf u=(1,u_2,\ldots,u_6),
\qquad
1=u_1<u_2<\cdots<u_6\le49,
```

and the original draw can be reconstructed exactly:

```{math}
x_i=a+u_i.
```

Adding the same offset to every number leaves \(\mathbf u\) unchanged. The
anchor restores absolute location during decoding.

The last normalized position is the inclusive span:

```{math}
u_6=x_6-x_1+1.
```

## Relative-dispersion profile

Let the endpoint extent and inclusive span be

```{math}
e=x_6-x_1,
\qquad
s=e+1.
```

Strictly increasing six-number draws guarantee \(e>0\). Production derives
five related profile blocks.

### Relative positions

Each number is placed on a unit interval spanning the first and last draw
values:

```{math}
r_i=\frac{x_i-x_1}{e}.
```

Thus

```{math}
r_1=0,
\qquad
r_6=1,
\qquad
0<r_2<\cdots<r_5<1.
```

Translation and uniform rescaling of a shape leave these values unchanged,
subject to valid integer coordinates.

### Internal gap shares

The five adjacent distances are normalized by the endpoint extent:

```{math}
g_i=\frac{x_{i+1}-x_i}{e},
\qquad i=1,\ldots,5.
```

They are positive and satisfy

```{math}
\sum_{i=1}^{5}g_i=1.
```

They are also the differences between consecutive relative positions.

### Coverage

Absolute span information is retained as

```{math}
C=\frac{s}{49}.
```

Two shapes can have identical relative positions and gap shares but different
coverage when one is a wider integer-scale version of the other.

### Uniformity deviation

The ideal equal gap share is \(1/5\). Production calculates the population
standard deviation

```{math}
\sigma_g=
\sqrt{\frac15\sum_{i=1}^{5}\left(g_i-\frac15\right)^2}
```

and normalizes it by the simplex maximum 0.4:

```{math}
U=\operatorname{clamp}\!\left(\frac{\sigma_g}{0.4},0,1\right).
```

Despite the internal field name `uniformity`, the displayed quantity is a
**uniformity deviation**:

- \(U=0\) means five equal adjacent shares;
- larger values mean more unequal spacing; and
- the theoretical upper bound is 1.

### Internal-gap entropy

The normalized Shannon entropy of the five gap shares is

```{math}
E=-\frac{\sum_{i=1}^{5}g_i\log g_i}{\log5}.
```

It is 1 for equal gap shares and approaches 0 as one share dominates. This is
an internal five-gap entropy, distinct from the Entropy strategy's six
circular-distance measure.

### Center balance

Let

```{math}
\bar r=\frac16\sum_{i=1}^{6}r_i.
```

Production maps the mean relative position to

```{math}
B=\operatorname{clamp}\!\left(
0.5+1.5(\bar r-0.5),0,1
\right).
```

A symmetric shape has \(\bar r=0.5\) and \(B=0.5\). Values below 0.5 indicate
more internal mass toward the first endpoint; values above 0.5 indicate more
mass toward the last endpoint. Reflection maps this directional summary toward
its opposite side rather than treating mirrored shapes as identical.

## Relative-profile distance

For profiles \(P\) and \(P'\), the internal-position shape difference is

```{math}
D_{\mathrm{shape}}(P,P')=
\frac14\sum_{i=2}^{5}|r_i-r'_i|.
```

The endpoints are excluded because they are always 0 and 1. Production then
uses the fixed weighted distance

```{math}
\begin{aligned}
D(P,P')={}&
0.50D_{\mathrm{shape}}
+0.20|C-C'|\\
&+0.10|U-U'|
+0.10|E-E'|
+0.10|B-B'|.
\end{aligned}
```

All five terms lie in \([0,1]\), and the weights sum to one. The 50% shape term
dominates; coverage preserves scale at 20%; the remaining summaries each
contribute 10%.

The weights are fixed engineering constants. They are not fitted from Top-6
efficacy or adapted over time.

## Valid normalized-position domains

Only positions 2–6 are learned because \(u_1=1\) by construction. For ordinal
position \(j\), production permits

```{math}
V_j=\{j,j+1,\ldots,43+j\}.
```

| Position | Valid values |
|---|---:|
| \(u_2\) | 2–45 |
| \(u_3\) | 3–46 |
| \(u_4\) | 4–47 |
| \(u_5\) | 5–48 |
| \(u_6\) | 6–49 |

These ordinal bounds leave room for later positions. The decoder additionally
enforces strict increase.

## Learned state

For each of the five variable normalized positions, MKRD maintains:

- a deque containing the latest 20 integer position values;
- categorical transition tables for orders 1 through 20; and
- cumulative lifetime counts for every valid value.

It also stores:

- 44 lifetime anchor counts indexed internally from 0 through 43; and
- the chronological observations
  \((\mathbf u_t,a_t,P_t)\), containing normalized positions, anchor, and the
  full relative-dispersion profile.

The position histories are bounded at 20. Transition tables, lifetime counts,
anchor counts, and observations accumulate across the completed prefix.

MKRD state is allocated only when the opt-in strategy is active. It does not
reuse MKNP's histories or observations.

## Lifetime categorical baseline

Let \(C_j(v)\) be the cumulative count of normalized value \(v\) at position
\(j\), and let \(h\) be the number of remembered draws. The baseline is

```{math}
b_j(v)=
\frac{C_j(v)+8/|V_j|}{h+8},
\qquad v\in V_j.
```

Values outside the valid domain receive zero. The total symmetric prior
strength 8 prevents valid unseen position values from receiving zero and yields
a uniform position distribution before history exists.

## Exact variable-order Markov branch

For proposed next value \(v\), MKRD begins with \(p_0(v)=b_j(v)\). It examines
the current integer-position suffix at every available order from 1 through 20.
A context is used only when it has at least 8 recorded successors.

For each supported order \(r\), the probability is updated as

```{math}
p_r(v)=
\frac{N_r(v)+8p_{r-1}(v)}{N_r+8},
```

where \(N_r(v)\) is the number of times the order-\(r\) context was followed by
\(v\), and \(N_r\) is its total successor support.

Orders are evaluated from shortest to longest. Each longer supported context
shrinks toward the probability already produced by the lifetime and shorter
contexts. Unsupported contexts are skipped rather than zeroing the estimate.

The displayed order is the longest supported context reached for the most
probable value of each position. Order 0 and support 0 mean no exact context met
the threshold.

## Relative-dispersion analogue branch

Exact categorical suffixes become sparse quickly. MKRD therefore searches
historical profile contexts and transfers their already observed successor
positions and anchors.

At most the latest 512 eligible historical targets are considered. For
candidate target \(k\), the available comparison order is

```{math}
o_k=\min(20,\text{available predecessor history}).
```

At lag \(\ell\), the current profile is compared with the corresponding
profile preceding historical target \(k\). Recent lags receive more weight
through \(\gamma=0.86\):

```{math}
d_k=
\frac{
\sum_{\ell=1}^{o_k}
\gamma^{\ell-1}D(P_{h-\ell+1},P_{k-\ell})
}{
\sum_{\ell=1}^{o_k}\gamma^{\ell-1}
}.
```

The final analogue weight is

```{math}
w_k=
\exp(-10d_k)
\times2^{-A_k/800}
\times\left(0.35+0.65\frac{o_k}{20}\right),
```

where \(A_k\) is the source age used by production. The factors provide:

- exponential similarity sharpening with coefficient 10;
- recency decay with an 800-draw half-life; and
- lower confidence for short comparison contexts.

The weighted target is historical successor \((\mathbf u_k,a_k)\). Its profile
is retained in the observation record but the output distributions receive its
integer normalized positions and anchor.

Effective analogue support is

```{math}
n_{\mathrm{eff}}=
\frac{\left(\sum_kw_k\right)^2}{\sum_kw_k^2}.
```

It equals the raw candidate count only for equal weights and decreases when a
small subset dominates.

## Position distributions

For each position, analogue successor counts are smoothed toward the lifetime
baseline using total strength 4:

```{math}
q_j(v)=
\operatorname{Norm}\!\left(
\sum_kw_k\mathbf1[u_{k,j}=v]+4b_j(v)
\right).
```

The exact Markov probabilities are separately normalized to \(e_j(v)\). The
final distribution is the fixed blend

```{math}
P_j(v)=0.70q_j(v)+0.30e_j(v),
```

followed by defensive normalization. The 70/30 allocation is not adaptive.

The most probable valid value is used only to select the displayed exact order
and support; the complete distribution continues into decoding.

## Anchor distribution

Let \(C_a(k)\) be the lifetime count for anchor offset
\(k\in\{0,\ldots,43\}\). The anchor baseline is

```{math}
b_a(k)=\frac{C_a(k)+4/44}{h+4}.
```

Weighted analogue anchors are smoothed toward this baseline:

```{math}
P_a(k)=
\operatorname{Norm}\!\left(
\sum_tw_t\mathbf1[a_t=k]+4b_a(k)
\right).
```

The anchor has no independent exact Markov context and no additional 70/30
blend.

## Constrained shape decoding

Independent position distributions can choose crossing or duplicate values.
MKRD uses the same legal-shape beam as Markov Normalized Positions.

The beam starts at \((1,)\). For each of the five learned positions, every
valid value larger than the preceding value is appended and its log probability
is added:

```{math}
L(1,u_2,\ldots,u_6)=
\sum_{j=2}^{6}\log P_j(u_j).
```

For each ending value, production keeps the best 8 partial paths. This is a
per-ending-value width, not a global eight-ticket beam. Exact path ties are
resolved deterministically by the position tuple.

For retained shape \(\mathbf u\) with spread \(s=u_6\), every legal anchor is
enumerated:

```{math}
a\in\{0,\ldots,49-s\}.
```

The decoded ticket and joint log weight are

```{math}
T=(a+u_1,\ldots,a+u_6),
```

```{math}
L(T)=L(\mathbf u)+\log P_a(a).
```

Every generated ticket is strictly increasing and contained in 1–49. Before
exponentiation, the largest log weight is subtracted from every ticket weight
to prevent numerical underflow without changing relative mass.

## Number marginals and ranking

Let \(\mathcal T\) be the generated ticket set and \(W(T)\) its stabilized
positive weights. Number inclusion marginal is

```{math}
M(n)=
\frac{
\sum_{T\in\mathcal T}W(T)\mathbf1[n\in T]
}{
\sum_{T\in\mathcal T}W(T)
}.
```

Every generated ticket has six numbers, so

```{math}
\sum_{n=1}^{49}M(n)=6.
```

The application exposes \(M(n)\) as **Marginal probability**. The strategy
score is a separate min–max transform:

```{math}
S(n)=
\begin{cases}
\dfrac{M(n)-M_{\min}}{M_{\max}-M_{\min}},
&M_{\max}>M_{\min},\\[8pt]
0,&M_{\max}=M_{\min}.
\end{cases}
```

Numbers are ranked by:

1. larger scaled score;
2. larger current gap; then
3. smaller number.

The Top-6 is formed from the six largest individual-number ranks. It need not
equal a generated joint ticket. The min–max score is a relative within-target
display value, not a calibrated probability.

## Causal lifecycle and leakage protection

For completed draw \(t\), production processing is:

1. use normalized-position histories ending at \(t-1\) to record transitions
   whose target is the now observed \(\mathbf u_t\);
2. append \(\mathbf u_t\) to position histories and lifetime counts;
3. add anchor \(a_t\) and observation \((\mathbf u_t,a_t,P_t)\);
4. build exact and analogue distributions from history through \(t\);
5. decode the ranking for target draw \(t+1\); and
6. evaluate it only after draw \(t+1\) becomes known.

The forecast target cannot enter its own transition counts, analogue sequence,
profile context, anchors, or decoder. Automated prefix-invariance tests confirm
that appending a later draw leaves the earlier ranking and details unchanged.

## Cold start and short history

Before any observation:

- position baselines are uniform over their valid ordinal domains;
- the anchor baseline is uniform over 44 offsets;
- no analogue exists; and
- no exact context meets support 8.

Even with symmetric input distributions, the finite beam and legal-anchor
constraints can produce nonuniform number marginals. Beam tie handling selects
a deterministic subset of equally weighted shapes.

After the first draw, lifetime position and anchor counts update, but no
eligible analogue target yet exists. With two observations, one analogue can
be used with a short-context confidence penalty. Exact contexts remain at their
smoothed baselines until support reaches 8.

## Interpreting prediction details

Each number exposes 13 MKRD-specific detail fields:

- **Marginal probability** is \(M(n)\), the normalized mass of generated
  tickets containing the number.
- **Random baseline** is \(6/49=12.2449\%\), supplied as a one-number uniform
  reference.
- **Best generated draw** is the highest-weight generated ticket containing
  that number.
- **Relative positions** lists the six \(r_i\) values for that explanatory
  ticket.
- **Span** and **coverage** report \(s\) and \(s/49\).
- **Gap shares** lists the five normalized adjacent distances.
- **Uniformity deviation** reports \(U\), where lower is more even.
- **Internal-gap entropy** reports \(E\), where higher is more even.
- **Center balance** reports \(B\), with 0.5 indicating a symmetric relative
  center.
- **First-number anchor** is the displayed ticket's first number; the internal
  offset is one less.
- **Analogue support** reports effective support and raw candidate count.
- **Exact orders /20** reports the selected order for all five positions and
  the minimum-to-maximum selected context support.
- **Valid-shape beam width 8** identifies the pruning limit.

The best generated draw is an explanation for one number's largest ticket
contribution. It is not necessarily the globally highest ticket, and the
strategy's Top-6 need not match it.

## Endpoint diagnostic

The repository's latest completed draw is

```{math}
(2,13,16,18,19,22).
```

Its relative-dispersion profile is:

| Field | Value |
|---|---:|
| Span | 21 |
| Coverage | 42.8571% |
| Relative positions | 0.000, 0.550, 0.700, 0.800, 0.850, 1.000 |
| Gap shares | 0.550, 0.150, 0.100, 0.050, 0.150 |
| Uniformity deviation | 0.447214 |
| Internal-gap entropy | 0.794061 |
| Center balance | 0.725000 |

After all 771 repository draws, the next-forecast state has:

| Diagnostic | Endpoint value |
|---|---:|
| Stored observations | 771 |
| Analogue candidates | 512 |
| Effective analogue support | 489.509 |
| Selected exact orders for positions 2–6 | 1, 1, 1, 1, 1 |
| Corresponding exact-context supports | 24, 40, 38, 15, 8 |
| Retained complete normalized shapes | 342 |
| Generated valid translated tickets | 7,483 |

The next-forecast Top-6 is `4, 5, 3, 17, 6, 12`. Its highest marginal is
23.87% for number 4. The best explanatory shape shared by several leading
numbers has span 35, coverage 71.43%, uniformity deviation 0.387, internal-gap
entropy 0.821, and center balance 0.449.

Only order-1 exact contexts meet the threshold at this endpoint, while
effective analogue support is close to the full 512 candidates. This fitted
state is driven primarily by broad profile analogues and short exact contexts,
not by a demonstrated deep order-20 recurrence.

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
| Full replay | 770 | 573 | 0.744156 | 565.714 |
| Validation, target draws 121–520 | 400 | 305 | 0.762500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 178 | 0.712000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, also records 178
hits or 0.712000 per target.

The full replay is close to theoretical random expectation, while the holdout
and latest comparison slice are below it. In the fixed comparative harness,
MKRD also trails Markov Spaces and Markov Normalized Positions over the full
history. It remains default-disabled. These retrospective results do not
establish statistical significance, calibration, stable future lift, or
predictability.

## Core mathematical and statistical concepts

- **Translation normalization:** subtracting the first number separates shape
  from anchor location.
- **Relative coordinates:** endpoint normalization maps internal positions to
  the unit interval.
- **Compositional gap shares:** five positive adjacent shares sum to one.
- **Dispersion and entropy:** normalized standard deviation and Shannon entropy
  summarize spacing inequality in complementary directions.
- **Directional center summary:** the mean relative coordinate distinguishes
  left- and right-concentrated shapes within their span.
- **Weighted metric:** fixed convex weights combine shape, coverage,
  uniformity deviation, entropy, and center balance.
- **Categorical Dirichlet-style smoothing:** lifetime position distributions
  use total prior strength 8; analogue and anchor smoothing use strength 4.
- **Variable-order Markov context:** exact position suffixes from order 1 to 20
  require minimum support 8.
- **Hierarchical backoff:** longer exact contexts shrink toward shorter and
  lifetime estimates.
- **Analogue estimation:** exponentially weighted profile similarity transfers
  historical successor mass.
- **Effective sample size:** \((\sum w)^2/\sum w^2\) measures concentration of
  analogue weights.
- **Constrained beam search:** independent position distributions are converted
  into legal monotone shapes.
- **Marginalization:** joint ticket weights become number-level inclusion mass.
- **Hypergeometric overlap:** Top-6 evaluation uses the correct
  without-replacement null.

## Limitations and responsible interpretation

- **Negative holdout behavior:** current holdout and latest-slice Top-6 results
  are below theoretical random expectation.
- **No deep exact context at the endpoint:** despite maximum order 20, only
  order-1 contexts meet support for the displayed modes.
- **Hand-selected metric:** the 50/20/10/10/10 distance weights are fixed and
  can encode retrospective design bias.
- **Correlated profile terms:** internal relative positions determine gap
  shares, which in turn determine uniformity and entropy; the distance terms
  are not independent information.
- **Partial scale invariance:** coverage deliberately makes uniformly rescaled
  shapes different even when all other relative fields match.
- **Directional reflection sensitivity:** center balance distinguishes mirrored
  shapes; whether this is useful is not established.
- **Independent position estimation:** five categorical distributions are
  modeled separately before beam constraints approximate their joint law.
- **Analogue cap and decay choices:** 512 candidates, order 20, lag decay 0.86,
  similarity coefficient 10, and half-life 800 are engineering constants.
- **Fixed branch blend:** analogue and exact distributions always receive
  70%/30%, regardless of support quality.
- **Sparse categorical states:** higher exact orders rarely reach support 8 in
  the observed endpoint.
- **Beam truncation:** retaining eight paths per ending value omits valid shapes
  and can influence number marginals.
- **Separate anchor model:** location is estimated from lifetime and analogue
  anchors without an exact Markov context.
- **Min–max information loss:** displayed scores preserve order but hide the
  absolute level and spread of marginals.
- **Top-6 coherence:** six marginal leaders need not form a high-weight joint
  ticket from the decoder.
- **Tie-break influence:** current gap can order equal marginals despite not
  belonging to the relative-dispersion model.
- **Retrospective comparison:** the profile, constants, and broader strategy
  collection were developed with historical outcomes available.
- **No guaranteed predictability:** similar historical shapes can have
  unrelated successors under independent random draws.

## Implementation map

The production implementation is concentrated in
`src/rand_ai/strategy_prediction.py`:

- `_normalized_positions_for_numbers` creates translated integer positions;
- `_RelativeDispersionProfile` stores the seven profile outputs;
- `_relative_dispersion_profile` calculates relative positions, gap shares,
  coverage, uniformity deviation, entropy, and center balance;
- `_StrategyState` owns independent MKRD histories, transitions, counts,
  anchors, and profile observations;
- `train` records exact normalized-position transitions before the current draw
  is remembered;
- `remember` appends current positions, anchor, and profile;
- `_mkrd_baseline_probability` and `_mkrd_probability` implement categorical
  smoothing and exact-context backoff;
- `_mkrd_profile_distance` applies the 50/20/10/10/10 profile metric;
- `_mkrd_analogue_weights` calculates causal context distances, recency, and
  length confidence;
- `_mkrd_distributions` creates position and anchor distributions;
- `_mknp_shape_beam` supplies the shared legal-shape decoder;
- `_mkrd_scores` translates shapes, marginalizes ticket weights, and builds
  detailed explanations; and
- `build_strategies` applies standard ranking and serializes `mkrd` when
  requested.

The mathematical and causal behavior is covered by
`tests/test_strategy_prediction.py`, including translation and scaling,
profile components, distance weighting, independent state, categorical
smoothing and backoff, valid shapes, detailed fields, maximum context length,
and prefix invariance. `scripts/benchmark_mkrd.py` provides the reproducible
comparative walk-forward harness.
