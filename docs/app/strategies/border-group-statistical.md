# Border Group Statistical

## Introduction

**Border Group Statistical** is the production strategy with identifier
`border_group_statistical`. It is a default-enabled member of the
**Border Space Groups** family. The strategy converts each historical draw
into a circular grouping signature, estimates the next signature from
smoothed lifetime frequencies, and decodes that categorical forecast into a
complete ranking of numbers 1 through 49.

Its first six ranked numbers form the Top-6 prediction used by prediction
grids, audits, effectiveness histories, comparisons, portfolios, exports, and
Possible Draw. The same border-space setting also affects the application's
Space Groups analysis and the other Border Group strategies.

```{admonition} A two-stage model
:class: important

The statistical model forecasts a **group signature**, not six numbers
directly. A separate leakage-safe decoder translates signature probabilities
through valid ticket shapes and historical anchors to obtain number scores.
```

## Scope

The strategy asks two related questions:

1. Which partition of six numbers into circularly connected groups is likely
   for the next draw?
2. Given that distribution over partitions, which individual numbers receive
   the most mass across valid decoded tickets?

It does not use transitions, contextual Bayesian features, online logistic
regression, or adaptive component weights. Those belong to the separate Border
Group Markov, Bayesian, ML, and Hybrid engines. Border Group Statistical uses
only smoothed cumulative signature counts for its categorical forecast.

The exact-random signature distribution and chi-square diagnostic shown in the
Space Groups report are reference statistics. They determine which signatures
are feasible, but their probabilities and p-value are not inputs to the
statistical forecast.

## Circular spaces

For a sorted draw

\[
1\leq n_1<n_2<\cdots<n_6\leq49,
\]

the engine constructs six counts of empty numbers. The first crosses the
49-to-1 boundary and the remaining five lie between adjacent sorted numbers:

\[
s_0=(n_1-1)+(49-n_6),
\]

\[
s_i=n_{i+1}-n_i-1,
\qquad i=1,\ldots,5.
\]

Every space is a non-negative integer and

\[
\sum_{i=0}^{5}s_i=49-6=43.
\]

This circular representation removes an artificial break between numbers 49
and 1. For example, the draw \(\{1,2,8,17,31,49\}\) has spaces

\[
(0,0,5,8,13,17).
\]

## Border threshold and connected groups

The **Border space** setting is an integer \(b\) from 0 through 43; its default
is 7. The threshold is inclusive:

- a space \(s_i\leq b\) connects the numbers on its two sides;
- a space \(s_i>b\) is a separator between groups.

Groups are maximal connected runs around the six-position circle. If there is
no separator, all six numbers form one group. Otherwise, the number of groups
equals the number of separators.

For the space pattern

\[
(10,6,6,10,6,5)
\]

at \(b=7\), the two spaces equal to 10 are separators. The two circular groups
have three numbers each.

Changing \(b\) changes the definition of the response variable itself. A
larger threshold treats more spaces as connections and generally produces
fewer groups; a smaller threshold creates more separators. At \(b=43\), every
valid draw has one group.

## Canonical signatures

The ordered circular group sizes depend on where traversal starts. The strategy
therefore sorts the group sizes from largest to smallest to obtain a
rotation-independent signature. The 11 integer partitions of six are:

| Index | Internal tuple | Displayed signature | Group count |
|---:|---|---|---:|
| 1 | `(6,)` | `6` | 1 |
| 2 | `(5, 1)` | `5+1` | 2 |
| 3 | `(4, 2)` | `4+2` | 2 |
| 4 | `(4, 1, 1)` | `4+1+1` | 3 |
| 5 | `(3, 3)` | `3+3` | 2 |
| 6 | `(3, 2, 1)` | `3+2+1` | 3 |
| 7 | `(3, 1, 1, 1)` | `3+1+1+1` | 4 |
| 8 | `(2, 2, 2)` | `2+2+2` | 3 |
| 9 | `(2, 2, 1, 1)` | `2+2+1+1` | 4 |
| 10 | `(2, 1, 1, 1, 1)` | `2+1+1+1+1` | 5 |
| 11 | `(1, 1, 1, 1, 1, 1)` | `1+1+1+1+1+1` | 6 |

When equal largest separator spaces could define more than one traversal
anchor, the production profile chooses the separator followed by the smallest
number. This makes ordered groups deterministic. The canonical sorted signature
is unchanged by that choice.

## Exact 6/49 reference distribution

For each border value, the engine enumerates separator masks and counts rooted
non-negative gap compositions summing to 43. There are

\[
\binom{48}{5}=1{,}712{,}304
\]

such compositions. Circular symmetry makes their signature proportions the
exact probabilities for a uniformly random 6-from-49 ticket.

At the default border \(b=7\), the exact reference is:

| Signature | Composition count | Exact probability |
|---|---:|---:|
| `6` | 196,608 | 11.4821% |
| `5+1` | 344,064 | 20.0936% |
| `4+2` | 344,064 | 20.0936% |
| `4+1+1` | 177,408 | 10.3608% |
| `3+3` | 172,032 | 10.0468% |
| `3+2+1` | 354,816 | 20.7216% |
| `3+1+1+1` | 25,536 | 1.4913% |
| `2+2+2` | 59,136 | 3.4536% |
| `2+2+1+1` | 38,304 | 2.2370% |
| `2+1+1+1+1` | 336 | 0.0196% |
| `1+1+1+1+1+1` | 0 | 0% |

Six separate groups are impossible at \(b=7\): six separators would each
require at least eight empty positions, but only 43 empty positions exist.
The implementation marks any zero-count signature as infeasible and assigns it
zero forecast probability.

```{admonition} Reference, not forecast prior
:class: note

The exact probabilities above are used for feasibility, diagnostics, and the
random-null comparison. Border Group Statistical does **not** shrink toward
these unequal probabilities. Its categorical pseudocount is uniform across
feasible signatures.
```

## Smoothed statistical forecast

Let \(\mathcal F_b\) be the signatures feasible at border \(b\), let \(C_s(t)\)
be the number of completed draws through time \(t\) with signature \(s\), and
let

\[
N(t)=\sum_{u\in\mathcal F_b}C_u(t).
\]

The next-signature forecast is

\[
P_{t+1}(s)=
\begin{cases}
\dfrac{C_s(t)+1}{N(t)+|\mathcal F_b|}, & s\in\mathcal F_b,\\[6pt]
0, & s\notin\mathcal F_b.
\end{cases}
\]

This is the posterior mean of a multinomial category probability under a
symmetric Dirichlet prior with concentration 1 for every feasible signature.
The pseudocount prevents a feasible but unseen signature from receiving zero
probability.

In an absolute cold state, all feasible signatures have equal probability.
The desktop's first emitted forecast occurs after the first historical draw has
already been observed, so one signature then has one observed count in addition
to its pseudocount.

### Manual group-count conditioning

The **Predicted groups** setting is **Automatic** by default. A user may instead
fix the next target to an exact feasible group count \(g\in\{1,\ldots,6\}\).
The strategy then conditions the signature distribution:

\[
P_{t+1}(s\mid |s|=g)=
\begin{cases}
\dfrac{P_{t+1}(s)}{\sum_{u:|u|=g}P_{t+1}(u)}, & |s|=g,\\[8pt]
0, & |s|\ne g.
\end{cases}
\]

Impossible border/group-count combinations are rejected. This setting is a
manual constraint, not a conclusion learned by the statistical model, and it
changes every Border Group strategy as well as relevant analysis and portfolio
guidance.

## From signatures to valid tickets

Signature probabilities do not identify particular numbers. The decoder builds
a probability beam of valid sorted tickets for each signature using two kinds
of causal evidence:

- up to the latest 16 observed six-space shapes for that signature;
- deterministic feasible fallback shapes, each with one pseudocount.

Let \(Q_s\) be the multiset of historical and fallback shapes for signature
\(s\). If a particular shape \(q\) has count \(c_{s,q}\), its weight is

\[
w_{s,q}=\frac{c_{s,q}}{\sum_{v\in Q_s}c_{s,v}}.
\]

For a shape \(q=(q_0,\ldots,q_5)\), the first sorted number can begin at anchors

\[
a\in\{0,1,\ldots,q_0\},
\]

where the first number is \(a+1\). If \(A_{s,a}\) is the lifetime count of
anchor \(a\) for signature \(s\), the allowed anchors receive additive-one
smoothing:

\[
w_{s,q}(a)=
\frac{A_{s,a}+1}{\sum_{r=0}^{q_0}(A_{s,r}+1)}.
\]

The remaining sorted numbers are reconstructed recursively:

\[
n_1=a+1,
\qquad
n_{i+1}=n_i+q_i+1,
\quad i=1,\ldots,5.
\]

The anchor range and the requirement that all six spaces sum to 43 guarantee a
sorted, unique ticket inside 1 through 49. If several shape/anchor paths decode
to the same ticket, their weights are added. This produces a normalized
within-signature ticket distribution \(D_s(T)\).

## Number marginals and ranking

The strategy combines its next-signature distribution with the decoded ticket
beams. For number \(n\), the production marginal is

\[
M_t(n)=\sum_s P_{t+1}(s)
       \sum_{T:n\in T}\frac{D_s(T)}{6}.
\]

The division by six makes the 49 marginals sum to one. Consequently,
\(M_t(n)\) is normalized number mass, not a calibrated probability that number
\(n\) will appear. Under the decoder's ticket mixture, the corresponding raw
inclusion mass would be six times this value.

All 49 marginals are min–max scaled for display and ranking:

\[
S_t(n)=
\begin{cases}
\dfrac{M_t(n)-M_{\min}}{M_{\max}-M_{\min}},
& M_{\max}>M_{\min},\\[8pt]
0,&M_{\max}=M_{\min}.
\end{cases}
\]

Numbers are ordered by descending score. Equal scores use the application's
standard tie-break: larger current recurrence gap first, then smaller number.
The first six ranks form the Top-6 prediction.

The displayed score is relative to the smallest and largest marginals in that
one forecast. It is not comparable as an absolute probability across draws,
and min–max scaling discards the original spread.

## Causal lifecycle

For completed draw \(t\), the production sequence is:

1. Evaluate any pending signature forecasts for draw \(t\) before updating
   their diagnostic losses.
2. Convert draw \(t\) to its circular profile and signature.
3. Increment the lifetime signature count and anchor count, and append its
   shape to that signature's 16-item recent buffer.
4. Build the smoothed signature forecast for draw \(t+1\).
5. Decode it using only shapes and anchors observed through draw \(t\), then
   rank all 49 numbers for target \(t+1\).

The actual target can update only the following forecast. It cannot influence
its own pending signature distribution, decoder beam, number marginals, or
ranking. Later appended draws therefore do not change predictions already made
for an earlier prefix.

## Interpreting prediction details

Every number displays four model details:

- **Border space** reports the inclusive connection threshold.
- **Model-selected group count** means the signature distribution is
  unconditioned; **Manual target … groups** identifies an explicit constraint.
- **Decoded marginal** is the number's unscaled normalized mass \(M_t(n)\),
  formatted as a percentage.
- **Leading signatures** lists the three largest forecast probabilities, with
  stable canonical order resolving equal probabilities.

Top-6 membership and rank are supplied by the standard prediction payload. The
leading-signature percentages describe categorical group shapes; they are not
individual-number probabilities.

## Signature-forecast statistics

The Space Groups analysis evaluates pending categorical forecasts only after
100 prior draws. For actual signature index \(Y_t\) and probability vector
\(\mathbf p_t\), it reports:

- log loss \(-\log(\max(p_{t,Y_t},10^{-15}))\);
- multiclass Brier score
  \(\sum_s(p_{t,s}-\mathbf 1[s=Y_t])^2\);
- exact-signature accuracy from the largest forecast category;
- group-count accuracy after mapping the largest category to its number of
  groups;
- mean absolute error in the predicted versus actual group count.

The displayed 95% log-loss interval uses the sample standard deviation of
per-draw losses and a normal \(1.96\) multiplier. These metrics evaluate the
11-category forecast, not the decoded Top-6 ranking.

On the repository's 771 draws at border 7, the 671 post-warm-up signature
evaluations are:

| Forecast | Log loss | Brier score | Signature accuracy | Group-count accuracy | Group-count MAE |
|---|---:|---:|---:|---:|---:|
| Border Group Statistical | 1.928158 | 0.835644 | 19.9702% | 41.2817% | 0.663189 |
| Exact random 6/49 null | 1.913997 | 0.833116 | 21.3115% | 33.9791% | 0.770491 |

The statistical forecast has better group-count accuracy and error on this
history, but the exact random null has better log loss, Brier score, and exact
signature accuracy. This mixed result does not establish predictive value.

### Distribution diagnostic

The application separately compares all 771 observed signature counts with the
exact default-border null. The current dataset gives

\[
\chi^2=5.357270,
\qquad p=0.802119.
\]

This diagnostic does not provide evidence of a departure from the exact null
on this history. It is not used to alter the forecast. Sparse rare categories,
retrospective setting choices, and temporal dependence should also temper a
literal interpretation of the asymptotic chi-square p-value.

## Top-6 efficacy reference

For any fixed six-number prediction compared with a uniformly random six-number
draw,

\[
H\sim\operatorname{Hypergeometric}(49,6,6),
\]

with

\[
\mathbb E[H]=\frac{36}{49}=0.734694,
\qquad
\operatorname{Var}(H)=0.577572.
\]

A leakage-free replay using border 7 and automatic group count over the
repository's 771 chronological YAML draws produces 770 target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 563 | 0.731169 | 565.714 |
| Validation, target draws 121–520 | 400 | 304 | 0.760000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 170 | 0.680000 | 183.673 |

The full replay is slightly below the theoretical random mean, and the
validation excess does not persist in the holdout. These are retrospective
implementation measurements, not a claim of statistical significance or
future advantage.

## Core mathematical and statistical concepts

- **Circular gap composition:** six non-negative spaces summing to 43 encode a
  ticket without privileging the 49-to-1 boundary.
- **Connected components:** an inclusive threshold turns adjacent number
  positions into circular groups separated by large spaces.
- **Integer partitions:** sorting group sizes maps every draw to one of 11
  canonical signatures.
- **Multinomial estimation:** lifetime counts estimate a categorical
  next-signature distribution.
- **Dirichlet smoothing:** one pseudocount per feasible signature prevents zero
  estimates for unseen categories.
- **Conditional probability:** a manual group count renormalizes probability
  over only signatures with the selected number of parts.
- **Mixture distributions:** signature probability, shape weight, and anchor
  weight combine into a distribution over valid tickets.
- **Marginalization:** ticket weights are summed to derive mass for each number.
- **Proper scoring rules:** log loss and Brier score assess categorical
  forecasts without being used to train this cumulative-frequency model.
- **Hypergeometric reference:** standard Top-6 efficacy is compared with the
  overlap distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **No demonstrated Top-6 lift:** the full replay is near and slightly below
  the theoretical random expectation, with weaker holdout behavior.
- **Uniform categorical prior:** early forecasts shrink toward equal feasible
  signature probabilities rather than the unequal exact 6/49 null.
- **Cumulative adaptation:** signature counts never expire, so the categorical
  model cannot respond quickly to a genuine distribution change.
- **Threshold sensitivity:** changing the border can redefine every historical
  signature and materially change all results.
- **Manual-selection risk:** choosing a group count after inspecting the same
  history introduces additional selection bias.
- **Shape truncation:** only the latest 16 observed shapes per signature enter
  the shape beam, while anchor counts use lifetime history.
- **Constructed fallback shapes:** deterministic pseudoshapes ensure coverage
  but are engineering priors, not samples from the exact random ticket space.
- **Anchor bias:** additive-one lifetime anchor frequencies can preserve
  dataset-specific location effects.
- **Two-stage error:** even a useful signature forecast can lose information or
  acquire bias when decoded into individual-number ranks.
- **Uncalibrated number mass:** decoded marginals and min–max scores are not
  occurrence probabilities.
- **Min–max information loss:** score scaling retains order but hides the
  absolute concentration of the marginal distribution.
- **Tie-break influence:** current recurrence gap can order numbers whose
  decoded scores are exactly equal.
- **Overlapping evaluation:** signature and Top-6 results arise from one fixed
  historical sequence, and model choices were developed in that context.
- **No guaranteed predictability:** apparent group frequencies or anchor
  patterns can be random fluctuations and may disappear on future draws.

Use Border Group Statistical as an interpretable structural baseline for
circular ticket shapes, not as evidence that historical grouping frequencies
control future lottery results.

## Implementation map

| Responsibility | Production location |
|---|---|
| Constants, validation, canonical signatures, and model names | `src/rand_ai/space_groups.py`, module constants and validators |
| Circular spaces, separators, ordered groups, and canonical profile | `src/rand_ai/space_groups.py`, `spaces_for_numbers`, `profile_from_spaces`, and `profile_for_numbers` |
| Exact rooted-composition counts, null probabilities, and feasibility | `src/rand_ai/space_groups.py`, `exact_null_signature_counts`, `exact_null_probabilities`, and `exact_null_group_probabilities` |
| Manual group-count conditioning | `src/rand_ai/space_groups.py`, `condition_signature_probabilities` |
| Smoothed cumulative categorical forecast | `src/rand_ai/space_groups.py`, `SpaceGroupForecaster._statistical` and `forecast` |
| Causal observation, counts, recent shapes, and anchors | `src/rand_ai/space_groups.py`, `SpaceGroupForecaster.observe` |
| Fallback shapes, ticket beams, marginals, score scaling, and details | `src/rand_ai/space_groups.py`, `_fallback_shapes`, `_signature_candidates`, `_signature_marginals`, and `number_scores` |
| Walk-forward log loss, Brier score, accuracies, and random-null comparison | `src/rand_ai/space_groups.py`, `_metrics` and `walk_forward_models` |
| Signature chi-square and transition diagnostics | `src/rand_ai/space_groups.py`, `signature_chi_square` and `transition_diagnostics` |
| Learn-before-forecast integration and standard ranking | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train`, `_StrategyState.build_strategies`, and `_ranking_from_scores` |
| Analysis tables, distributions, diagnostics, and forecast payload | `src/rand_ai/statistics.py`, `DrawsStatistics.space_group_analysis` |
| Desktop options, registration, serialization, and default state | `src/rand_ai/gui_bridge.py` and `web/electron/main.cjs` |
| Settings controls, family, color, and prediction details | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Circular grouping, exact-null, conditioning, decoding, metrics, and diagnostics tests | `tests/test_space_groups.py` |
| Selected-only output, manual target, ranking shape, and serialization tests | `tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The standard strategy payload contains all 49 scaled scores, ranks, current
gaps, detail strings, Top-6 numbers, and completed efficacy. The broader Space
Groups report additionally exposes categorical probabilities and diagnostics;
those analysis fields are not embedded in every number's strategy details.
