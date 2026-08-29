# Markov Spaces

## Introduction

**Markov Spaces** is the production strategy with identifier `mksp` and short
engine name **MKSP**. It is a default-enabled member of the
**Markov & Sequence** family. The strategy represents every draw by the six
empty-number spaces around its sorted values, learns exact variable-order
transitions for each space position, finds similar historical space contexts,
and decodes the combined distributions into complete valid 6-from-49 draws.

Number inclusion marginals from that valid-draw beam provide the complete 1–49
ranking. Its first six ranks form the Top-6 prediction used by grids, audits,
effectiveness histories, comparisons, portfolios, exports, and Possible Draw.
Its complete ranking can also serve as an input to selected ensemble engines.

```{admonition} A ticket-generating space model
:class: important

MKSP does not predict six numbers independently. It forecasts six positional
space distributions, constrains them to sum to 43, combines them with an anchor
distribution, and constructs valid sorted tickets before calculating number
marginals.
```

## Scope

The strategy combines two causal forecasting branches:

- **exact contexts**: variable-order categorical transitions through order 20;
- **similar contexts**: up to 512 analogue targets weighted by space-pattern
  similarity, recency, and available context length.

The final positional distributions reserve 70% for the analogue branch and 30%
for the exact-context branch. A width-eight constrained beam then restores the
joint sum and ticket-validity conditions that separate position models do not
enforce by themselves.

MKSP is distinct from Markov Normalized Positions and Markov Relative
Dispersion, which transform draw shape differently. It is also distinct from
the Border Group strategies: MKSP preserves all six exact space values rather
than thresholding spaces into group signatures.

## Six-space representation

For sorted draw

\[
1\leq n_1<n_2<\cdots<n_6\leq49,
\]

the outer space crossing the 49-to-1 boundary is

\[
s_0=(n_1-1)+(49-n_6),
\]

and the five internal spaces are

\[
s_i=n_{i+1}-n_i-1,
\qquad i=1,\ldots,5.
\]

Each \(s_i\) is an integer from 0 through 43, and every valid draw satisfies

\[
\sum_{i=0}^{5}s_i=49-6=43.
\]

For example,

\[
\{1,10,20,30,40,49\}
\longrightarrow(0,8,9,9,9,8).
\]

The model also stores the location anchor

\[
a=n_1-1,
\qquad a\in\{0,\ldots,43\}.
\]

The space tuple describes the relative empty positions, while the anchor places
the resulting shape on the 1–49 line.

## State maintained for each position

For every position \(p\in\{0,\ldots,5\}\), MKSP maintains:

- the latest 20 observed values \(s_{t,p}\);
- lifetime counts for all 44 values \(0,\ldots,43\);
- transition tables for exact context orders 1 through 20.

It also retains every historical `(spaces, anchor)` observation for analogue
search and lifetime counts for all 44 anchor values.

When current draw \(t\) occurs, each position's target value is recorded against
every available suffix of its preceding history. For order \(o\), the context is

\[
c_{t,p,o}=(s_{t-o,p},\ldots,s_{t-1,p}),
\]

and the transition table counts the target \(s_{t,p}\). Only after these
transitions are updated is the current value appended to the 20-item history.

## Lifetime categorical baseline

Let \(C_p(v)\) be the lifetime number of draws whose position \(p\) has value
\(v\), and let \(N\) be the completed-draw count. The baseline is

\[
B_p(v)=
\frac{C_p(v)+8/44}{N+8},
\qquad v=0,\ldots,43.
\]

This is the posterior mean under a symmetric 44-category Dirichlet prior with
total concentration 8. Each value receives pseudocount \(8/44\), so unseen
values retain nonzero probability.

The baseline treats positions separately. It does not by itself enforce that
six sampled values sum to 43; that constraint is restored by the decoder.

## Exact variable-order transitions

For position \(p\), order \(o\), current context \(c\), and candidate value
\(v\), let \(C_{p,o,c}(v)\) be the transition count and

\[
M_{p,o,c}=\sum_{u=0}^{43}C_{p,o,c}(u)
\]

its support. Contexts with fewer than eight observations are ignored:

\[
M_{p,o,c}<8.
\]

Starting with \(Q_{p,0}(v)=B_p(v)\), supported orders are applied from shortest
to longest:

\[
Q_{p,o}(v)=
\frac{C_{p,o,c}(v)+8Q_{p,o-1}(v)}
     {M_{p,o,c}+8}.
\]

If an order is unsupported, the current probability remains unchanged. Because
an exact longer context cannot have more support than its suffix, this produces
a natural backoff: long contexts influence the forecast only after every needed
history match exists often enough.

The maximum active order is

\[
o_{\max}=\min(20,N).
\]

The production code evaluates the recurrence separately for all 44 values and
normalizes the resulting vector. The reported selected order is the longest
supported exact context for the value with the largest final positional
probability.

## Similar historical contexts

The analogue branch compares the latest space-vector history with the contexts
that preceded historical target draws. With \(N\) stored observations, candidate
target indexes are

\[
j=\max(1,N-512),\ldots,N-1.
\]

Thus no more than 512 targets are considered, and index 0 is excluded because
it has no preceding context. Candidate \(j\) uses context length

\[
o_j=\min(20,j).
\]

### Lag distance

At lag \(\ell\), the normalized six-position Manhattan distance is

\[
d_{j,\ell}
=\frac{\sum_{p=0}^{5}
|s_{N-\ell,p}-s_{j-\ell,p}|}
{6\cdot43}.
\]

Recent context lags receive geometric weight \(0.86^{\ell-1}\). The combined
distance is

\[
d_j=
\frac{\sum_{\ell=1}^{o_j}0.86^{\ell-1}d_{j,\ell}}
     {\sum_{\ell=1}^{o_j}0.86^{\ell-1}}.
\]

### Analogue weight

Candidate target \(j\) receives

\[
w_j=
\exp(-10d_j)
\times2^{-(N-j)/800}
\times\left(0.35+0.65\frac{o_j}{20}\right).
\]

The three factors represent:

- similarity sharpness 10;
- an 800-draw recency half-life;
- confidence increasing from short histories toward a full 20-draw context.

The candidate's own six-space tuple and anchor become weighted target evidence.
The target always follows its historical comparison context, so no future draw
is used.

### Effective analogue support

The UI reports the Kish-style effective sample size

\[
N_{\mathrm{eff}}
=\frac{(\sum_jw_j)^2}{\sum_jw_j^2}.
\]

It equals the candidate count only when all weights are equal and becomes
smaller when a few analogues dominate.

## Analogue positional distributions

For position \(p\), the weighted target count is

\[
A_p(v)=\sum_jw_j\mathbf 1[s_{j,p}=v].
\]

It is smoothed toward the lifetime baseline with strength 4:

\[
P^{\mathrm{analogue}}_p(v)
=\frac{A_p(v)+4B_p(v)}{\sum_jw_j+4}.
\]

When no analogue exists, this branch reduces to the baseline. The exact-context
distribution is the normalized vector

\[
P^{\mathrm{exact}}_p(v)
=\operatorname{normalize}(Q_{p,o_{\max}}(v)).
\]

The final space distribution is

\[
P_p(v)=
\operatorname{normalize}\!\left(
0.70P^{\mathrm{analogue}}_p(v)
+0.30P^{\mathrm{exact}}_p(v)
\right).
\]

The weights are fixed and do not adapt to efficacy or uncertainty.

## Anchor distribution

Let \(C_a(r)\) be the lifetime count of anchor \(r\), and let

\[
A_a(r)=\sum_jw_j\mathbf 1[a_j=r].
\]

The anchor baseline uses symmetric strength 4:

\[
B_a(r)=\frac{C_a(r)+4/44}{N+4}.
\]

The production anchor distribution is

\[
P_a(r)=
\frac{A_a(r)+4B_a(r)}{\sum_jw_j+4}.
\]

There is no separate exact-context Markov branch for anchors. Historical
analogue anchors are smoothed directly toward lifetime anchor frequencies.

## Valid-space beam

Independent position distributions can assign mass to impossible combinations.
MKSP therefore decodes only paths whose spaces can sum to 43.

### Internal spaces

The beam first processes positions \(s_1,\ldots,s_5\). For every partial sum
\(q\in\{0,ldots,43\}\), it retains at most the eight paths with largest

\[
\sum_{i=1}^{m}\log P_i(s_i).
\]

Any extension whose sum exceeds 43 is discarded. Keeping eight paths
**per partial sum** preserves alternatives leading to different valid outer
spaces while bounding computation.

### Completing the outer space

For a retained internal path with total

\[
q=\sum_{i=1}^{5}s_i,
\]

the outer space is forced to

\[
s_0=43-q.
\]

Its positional contribution is \(P_0(s_0)\). This makes every completed state
sum exactly to 43.

### Valid anchors and ticket reconstruction

For a completed state, allowed anchors are

\[
a=0,\ldots,s_0.
\]

The corresponding numbers are

\[
n_1=a+1,
\qquad
n_{i+1}=n_i+s_i+1,
\quad i=1,\ldots,5.
\]

The anchor bound guarantees \(n_6\leq49\). Every generated ticket is sorted,
contains six unique numbers, lies inside 1–49, and reproduces a six-space state
summing to 43.

Its unnormalized log weight is

\[
\log P_0(s_0)
+\sum_{i=1}^{5}\log P_i(s_i)
+\log P_a(a).
\]

The implementation subtracts the maximum log weight before exponentiation for
numerical stability and normalizes over all generated tickets.

## Number marginals and ranking

Let \(\mathcal T\) be the generated ticket beam and \(W(T)\) its normalized
weight. The inclusion marginal for number \(n\) is

\[
M(n)=\sum_{T\in\mathcal T:n\in T}W(T).
\]

Because every generated ticket contains six numbers,

\[
\sum_{n=1}^{49}M(n)=6.
\]

This makes \(M(n)\) an inclusion probability under MKSP's approximate beam
distribution. It has not been demonstrated to be calibrated to future lottery
occurrence frequencies.

The 49 marginals are min–max scaled:

\[
S(n)=
\begin{cases}
\dfrac{M(n)-M_{\min}}{M_{\max}-M_{\min}},
&M_{\max}>M_{\min},\\[8pt]
0,&M_{\max}=M_{\min}.
\end{cases}
\]

Numbers are ranked by descending score, then larger current recurrence gap,
then smaller number. The first six ranks form the Top-6 prediction. The
displayed normalized score is relative within one target and is not the same as
the unscaled marginal shown in details.

## Cold start and short history

With no observations, all 44 values at each position and anchor have symmetric
prior mass. There are no analogues and no supported exact transition contexts,
so both space branches reduce to the uniform baseline.

The constrained width-eight decoder still selects only a deterministic subset
of all possible internal paths. Consequently, structural constraints and beam
tie resolution can make number marginals nonuniform even when every source
distribution is uniform.

The desktop's first emitted forecast occurs after one draw. At that point
lifetime space and anchor counts contain that draw, but the analogue branch has
no candidate until at least two observations exist.

## Causal lifecycle

For target draw \(t\), the production sequence is:

1. A prediction for \(t\) already exists from state ending at draw \(t-1\).
2. When \(D_t\) occurs, convert it to its six-space target.
3. For each position and every available order through 20, update the transition
   from the suffix ending at \(t-1\) to the observed target value at \(t\).
4. Append the six current values to their bounded histories, increment lifetime
   value and anchor counts, and store the complete `(spaces, anchor)` observation.
5. For target \(t+1\), query exact contexts ending at \(t\), compare that current
   context with only historical predecessor contexts, and use their already
   observed following targets as analogues.
6. Blend positional distributions, decode valid tickets, calculate marginals,
   and rank all 49 numbers.

The current target is learned only after its forecast has completed. It cannot
influence its own contexts, analogue targets, beam, or marginals. Prefix tests
verify that appending a later draw leaves the earlier MKSP prediction unchanged.

## Interpreting prediction details

Every number carries seven detail lines:

- **Marginal probability** is \(M(n)\), the number's inclusion mass in the
  normalized valid-ticket beam.
- **Random baseline** displays \(6/49\) for reference; it is not mixed into the
  final marginal at this stage.
- **Best generated draw** is the single highest-weight beam ticket containing
  that number, not necessarily the globally highest ticket for every number.
- **Best six-space state** gives that ticket's six spaces and confirms their sum
  is 43.
- **Analogue support** reports effective support and the raw candidate count.
- **Exact orders /20** reports the selected exact-context order for all six
  positions and the minimum-to-maximum context support.
- **Valid-draw beam width 8** identifies the per-partial-sum pruning limit.

The best generated ticket is an explanation aid. It is not directly returned
as MKSP's Top-6, which is formed from the six largest number marginals.

## Endpoint diagnostic

After all 771 repository draws, the forecast state uses:

| Diagnostic | Endpoint value |
|---|---:|
| Stored observations | 771 |
| Analogue candidates | 512 |
| Effective analogue support | 493.177 |
| Selected exact orders for positions 0–5 | 1, 1, 1, 1, 2, 2 |
| Corresponding exact-context supports | 8, 24, 71, 89, 8, 8 |

The low selected orders illustrate exact-context sparsity even with a maximum
order of 20. At this endpoint, most practical weight comes from the analogue
branch and short exact contexts. This is a description of one fitted historical
state, not evidence that the same pattern will persist.

## Top-6 efficacy reference

For a fixed six-number prediction evaluated against a uniformly random
six-number draw,

\[
H\sim\operatorname{Hypergeometric}(49,6,6),
\]

with

\[
\mathbb E[H]=\frac{36}{49}=0.734694,
\qquad
\operatorname{Var}(H)=0.577572.
\]

A leakage-free replay over the repository's 771 chronological YAML draws
produces 770 target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 596 | 0.774026 | 565.714 |
| Validation, target draws 121–520 | 400 | 307 | 0.767500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 195 | 0.780000 | 183.673 |

All three slices are above the theoretical random mean, but the model,
constants, transformations, and broader strategy collection were developed
with historical data available. These retrospective measurements do not prove
statistical significance, calibration, stable future lift, or predictability.

## Core mathematical and statistical concepts

- **Compositional state:** six non-negative spaces must sum to the fixed total
  43.
- **Categorical Dirichlet smoothing:** lifetime position distributions use a
  symmetric total prior strength of 8.
- **Variable-order Markov context:** exact suffix transitions are evaluated from
  order 1 through 20 with minimum support 8.
- **Hierarchical backoff:** each supported longer context shrinks toward the
  probability produced by shorter contexts.
- **Analogue forecasting:** similar predecessor histories lend their following
  space tuple and anchor as weighted targets.
- **Kernel weighting:** exponential similarity, recency half-life, and context
  length confidence determine analogue influence.
- **Effective sample size:** squared total weight divided by squared weights
  summarizes analogue concentration.
- **Model averaging:** smoothed analogue and exact-context distributions receive
  fixed 70% and 30% shares.
- **Constrained beam search:** partial sums and log probabilities approximate a
  joint distribution while guaranteeing valid space totals.
- **Marginalization:** normalized ticket weights are summed into individual
  number inclusion probabilities.
- **Hypergeometric overlap:** standard efficacy uses the null overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Retrospective specification:** order 20, support 8, 512 analogues, decay
  0.86, half-life 800, sharpness 10, the 70/30 blend, and beam width 8 were fixed
  in the context of historical development.
- **Position independence before decoding:** six marginal space models ignore
  their strong compositional dependence until the beam stage.
- **Sparse exact contexts:** long exact sequences rarely reach support 8, so the
  advertised maximum order need not be the order actually used.
- **Nested evidence reuse:** shorter and longer exact contexts are derived from
  overlapping observations; sequential smoothing is an engineering hierarchy,
  not independent Bayesian confirmation.
- **Overlapping analogues:** historical contexts can share many draws, making
  effective support an importance-weight diagnostic rather than an independent
  sample count.
- **Fixed analogue kernel:** distance normalization and similarity parameters
  may not reflect true predictive relevance.
- **Candidate truncation:** only the latest 512 eligible historical targets are
  considered.
- **Anchor simplification:** anchors use analogue and lifetime frequency but no
  separate variable-order Markov model and are not modeled jointly with a space
  state before decoding.
- **Beam approximation:** retaining eight paths per partial sum discards valid
  lower-probability combinations and can introduce deterministic tie artifacts.
- **Marginal-versus-ticket mismatch:** the six largest marginal numbers need not
  equal any one high-weight generated ticket.
- **No calibration or uncertainty interval:** the beam marginal is a normalized
  model output, not a validated probability or credible interval.
- **Min–max information loss:** displayed scores obscure absolute marginal
  concentration and cannot be compared directly across targets.
- **Tie-break influence:** current gap and number can order equal final scores
  outside the space model.
- **Dataset and selection dependence:** replay improvements may reflect chance,
  repeated strategy comparisons, or historical tuning.
- **No guaranteed predictability:** patterns among past spaces do not establish
  a causal mechanism for future lottery draws.

Use MKSP as an auditable causal model of space sequences and valid-ticket
decoding, not as evidence that a fair lottery follows a stable Markov process.

## Implementation map

| Responsibility | Production location |
|---|---|
| Orders, support, priors, analogue limit and blend, kernel constants, half-life, and beam width | `src/rand_ai/strategy_prediction.py`, `_MKSP_*` constants |
| Six-space encoding | `src/rand_ai/strategy_prediction.py`, `_spaces_for_numbers` |
| Histories, transition tables, lifetime space counts, anchors, and observations | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Exact-context transition learning before current-state storage | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train` |
| Current spaces, anchor, lifetime counts, histories, and analogue observations | `src/rand_ai/strategy_prediction.py`, `_StrategyState.remember` |
| Lifetime baseline and variable-order hierarchical probability | `src/rand_ai/strategy_prediction.py`, `_mksp_baseline_probability` and `_mksp_probability` |
| Context distance, similarity, recency, confidence, and candidate weights | `src/rand_ai/strategy_prediction.py`, `_mksp_analogue_weights` |
| Positional and anchor distributions plus effective support | `src/rand_ai/strategy_prediction.py`, `_mksp_distributions` |
| Constrained internal-space beam | `src/rand_ai/strategy_prediction.py`, `_mksp_internal_beam` |
| Valid ticket reconstruction, number marginals, scaling, and details | `src/rand_ai/strategy_prediction.py`, `_mksp_scores` |
| Final gap/number tie-break, Top-6, efficacy, and causal orchestration | `src/rand_ai/strategy_prediction.py`, `_ranking_from_scores`, `_strategy`, and `build_prediction_suites` |
| Desktop registration, serialization, and default-enabled state | `web/electron/main.cjs` and `src/rand_ai/gui_bridge.py` |
| Settings description, family, color, names, and details rendering | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Encoding, order-20, smoothing, analogue, beam validity, and prefix-invariance tests | `tests/test_strategy_prediction.py` |
| Comparative replay harness using MKSP as a fixed baseline | `scripts/benchmark_mkrd.py` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 normalized scores, ranks, current gaps,
detail strings, Top-6 numbers, and standard completed efficacy. Internal
positional distributions and the generated ticket beam are summarized in
details rather than serialized wholesale.
