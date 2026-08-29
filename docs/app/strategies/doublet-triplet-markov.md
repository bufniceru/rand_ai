# Doublet & Triplet Markov

## Introduction

**Doublet & Triplet Markov** is the production strategy with identifier
`doublet_triplet_markov`. It is a default-enabled member of the
**Markov & Sequence** family. The engine learns the recurrence of consecutive
two- and three-number groups, estimates how the latest draw changes their
next-draw rates, and converts supported groups into a complete ranking of
numbers 1 through 49.

The first six ranks form its Top-6 prediction for grids, audits, effectiveness
histories, comparisons, portfolios, exports, and Possible Draw. Its complete
ranking can also be consumed by selected ensemble strategies, without changing
the standalone calculation documented here.

```{admonition} What “Markov” means here
:class: important

The model conditions each candidate consecutive group on the six individual
numbers in the immediately preceding draw. It also maintains a three-state
first-order transition model for whether draws contain no doublet, a doublet
without a triplet, or a triplet. It is not a Markov chain over complete lottery
tickets.
```

## Scope

The strategy models ordinary numerical adjacency:

- a **doublet** is \(\{a,a+1\}\) for \(a=1,\ldots,48\);
- a **triplet** is \(\{a,a+1,a+2\}\) for \(a=1,\ldots,47\).

Adjacency does not wrap around the endpoints. Thus `48-49` is a valid doublet,
but `49-1` is not; similarly, `47-48-49` is valid, but groups crossing from 49
to 1 are not.

A draw can contain several overlapping groups. For example,
\(\{1,2,3,10,20,30\}\) contains doublets starting at 1 and 2 and a triplet
starting at 1. Every observed start is counted.

The strategy does not model arbitrary unordered pairs or triples. That is a
different problem from co-occurrence. It specifically models consecutive runs,
their recent recurrence, and one-step transitions from the latest draw.

## Group indicators

For draw \(D_t\), group size \(k\in\{2,3\}\), and valid start \(a\), define

\[
I_{t,k,a}
=\mathbf 1\!\left[\{a,a+1,\ldots,a+k-1\}\subseteq D_t\right].
\]

The production state records:

- lifetime group count
  \(C_{k,a}(t)=\sum_{i=1}^{t}I_{i,k,a}\);
- the latest at most 120 sets of doublet and triplet starts;
- number-conditioned transition count \(T_{x,k,a}(t)\), incremented when
  number \(x\) was in draw \(D_{i-1}\) and group \((k,a)\) appeared in
  \(D_i\);
- opportunity count \(O_x(t)\), incremented once for every adjacent-draw pair
  whose source draw contains \(x\).

Each fixed group is a binary event within a target draw. Transition
probabilities for different starts are estimated separately and do not need to
sum to one.

## Random-reference probability for a fixed group

If a particular set of \(k\) consecutive numbers is fixed in advance, a
uniformly random 6-from-49 draw contains it with probability

\[
p_{0,k}
=\frac{\binom{49-k}{6-k}}{\binom{49}{6}}.
\]

For the two supported sizes:

| Group size | Valid starts | Fixed-group probability |
|---:|---:|---:|
| Doublet, \(k=2\) | 48 | 0.0127551, or 1.27551% |
| Triplet, \(k=3\) | 47 | 0.00108554, or 0.108554% |

These are probabilities for one specified group, such as `20-21`, not the
probability that a draw contains at least one group of that size. Overlapping
candidate groups prevent obtaining the latter by simply multiplying by the
number of starts.

## Lifetime, recent, and conditional estimates

The fixed smoothing strength is

\[
\alpha=8.
\]

Let \(N=t\) be the number of completed draws and let
\(W=\min(N,120)\) be the number retained in the recent group buffer.

### Lifetime estimate

For group \((k,a)\), the lifetime estimate is

\[
L_{k,a}
=\frac{C_{k,a}+\alpha p_{0,k}}{N+\alpha}.
\]

This is additive shrinkage toward the exact fixed-group probability. At cold
start every group of the same size has the same lifetime estimate.

### Recent estimate

Let \(C^{(W)}_{k,a}\) be the number of appearances in the latest at most 120
draws. The recent estimate shrinks toward the already-smoothed lifetime value:

\[
R_{k,a}
=\frac{C^{(W)}_{k,a}+\alpha L_{k,a}}{W+\alpha}.
\]

The window is a hard trailing window. There is no exponential decay or extra
streak multiplier inside it.

### Previous-number conditional estimate

For each number \(x\) in the latest completed draw, the one-step conditional
estimate is

\[
Q_{x,k,a}
=\frac{T_{x,k,a}+\alpha L_{k,a}}{O_x+\alpha}.
\]

The group-level Markov estimate averages the six latest-number conditionals:

\[
Q_{k,a}
=\frac{1}{|D_t|}\sum_{x\in D_t}Q_{x,k,a}.
\]

If there is no previous draw, the implementation uses \(Q_{k,a}=L_{k,a}\).
The average is an engineering aggregation of six conditional estimates; it is
not an independence-based union probability.

## Relative group score

Lifetime, recent, and conditional estimates can have very different numerical
spreads. For each group size and each component separately, the strategy
min–max scales all valid starts:

\[
\operatorname{scale}(v_a)=
\begin{cases}
\dfrac{v_a-\min_b v_b}{\max_b v_b-\min_b v_b},
&\max_b v_b>\min_b v_b,\\[8pt]
0,&\text{otherwise}.
\end{cases}
\]

It then forms the relative score

\[
G_{k,a}
=0.30\,\operatorname{scale}(L_{k,a})
+0.22\,\operatorname{scale}(R_{k,a})
+0.48\,\operatorname{scale}(Q_{k,a}).
\]

The Markov component has the largest weight. Because doublets and triplets are
scaled in separate pools, \(G_{2,a}\) and \(G_{3,a}\) are relative support
scores rather than directly comparable occurrence probabilities.

At absolute cold start, every start within each size has equal component
values. All scaled components and all group scores are therefore zero.

## Three-state shape model

Every completed draw is assigned exactly one state:

| State | Meaning |
|---:|---|
| 0 | No consecutive doublet |
| 1 | At least one doublet, but no triplet |
| 2 | At least one triplet |

A triplet necessarily contains two doublets, but state 2 takes precedence over
state 1. Let \(Z_t\) be the state of draw \(t\), \(S_z\) its lifetime count,
and \(M_{u,z}\) the observed transitions from previous state \(u\) to next
state \(z\).

When history exists, the lifetime shape prior is empirical:

\[
\pi_z=\frac{S_z}{\sum_r S_r}.
\]

Before any draw exists, the fixed cold-start prior is

\[
(\pi_0,\pi_1,\pi_2)=(0.65,0.30,0.05).
\]

For the latest state \(u=Z_t\), the smoothed next-state distribution is

\[
P(Z_{t+1}=z\mid Z_t=u)
=\frac{M_{u,z}+\alpha\pi_z}
       {\sum_r M_{u,r}+\alpha}.
\]

The derived probabilities shown in details are

\[
P_{\text{doublet}}=P(Z_{t+1}=1)+P(Z_{t+1}=2),
\]

\[
P_{\text{triplet}}=P(Z_{t+1}=2).
\]

They modulate the two group families through

\[
w_2=0.55+0.45P_{\text{doublet}},
\qquad
w_3=0.45+0.55P_{\text{triplet}}.
\]

These are bounded support multipliers, not mixture weights, and they are not
required to sum to one.

## From group scores to number scores

A number can belong to up to two linear doublets and three linear triplets,
with fewer possibilities near 1 and 49. Let \(\mathcal G_k(n)\) be the valid
size-\(k\) groups containing number \(n\).

The doublet support is

\[
D(n)=0.70\max_{a\in\mathcal G_2(n)}G_{2,a}
+0.30\operatorname{mean}_{a\in\mathcal G_2(n)}G_{2,a},
\]

and the triplet support is

\[
T(n)=0.68\max_{a\in\mathcal G_3(n)}G_{3,a}
+0.32\operatorname{mean}_{a\in\mathcal G_3(n)}G_{3,a}.
\]

The preliminary number score is

\[
U(n)=0.46w_2D(n)+0.54w_3T(n).
\]

This construction rewards membership in the strongest supported group while
retaining some information from every containing group.

## Six-member admission bonus

The engine next assembles a priority list containing all triplets with priority
\(w_3G_{3,a}\) and all doublets with priority \(w_2G_{2,a}\). Candidate groups
are ordered by:

1. larger priority;
2. triplet before doublet when priorities tie;
3. smaller start when size and priority tie.

Starting with an empty admission set, the strategy visits positive-priority
groups. New members of a group are ordered by larger preliminary score, then
smaller number, and admitted until the set reaches six distinct numbers. Every
already-admitted or newly admitted member of that visited group receives

\[
0.30\times\text{group priority}
\]

as a score bonus. Processing stops once six numbers have been admitted.

The admission set is not a hard final ticket. It only determines which numbers
receive bonuses; the final Top-6 is still produced by ranking all 49 adjusted
scores. The mechanism favors coherent coverage of highly supported groups and
can change an ordering that the preliminary support alone would produce.

## Final scaling and ranking

After bonuses, all 49 raw scores are min–max scaled to \([0,1]\). If every raw
score is equal, all displayed scores become zero. Numbers are ranked by:

1. larger scaled score;
2. larger current recurrence gap;
3. smaller number.

The first six ranks form the prediction. Scaling preserves unequal-score order
but removes the absolute distance between the weakest and strongest forecasts,
so displayed percentages are relative scores rather than calibrated hit
probabilities.

## Causal lifecycle

For target draw \(t\), the production order is:

1. A prediction for \(t\) already exists from state ending at draw \(t-1\).
2. When \(D_t\) occurs, identify its doublet and triplet starts.
3. Before replacing the latest draw, update transitions from each number in
   \(D_{t-1}\) to every group observed in \(D_t\), and update the shape-state
   transition \(Z_{t-1}\to Z_t\).
4. Add \(D_t\)'s groups to lifetime counts and the 120-draw buffer, increment
   its shape count, and remember \(D_t\) as the latest draw.
5. Use this completed state to score groups and numbers for target \(t+1\).

Draw \(t\) therefore trains only forecasts issued after it occurs. It cannot
alter its own pending prediction. Prefix-invariance tests confirm that adding a
future draw leaves the preceding prediction unchanged.

## Interpreting prediction details

Every number carries five detail lines:

- **Strongest doublet** identifies the containing doublet with the largest
  relative group score and shows that score plus its unscaled Markov estimate.
- **Strongest triplet** reports the equivalent containing triplet.
- **Next shape** reports the derived probabilities of at least one doublet and
  at least one triplet in the next draw.
- **Recent window** reports the current buffer length, up to 120, and the number
  of latest-draw values used for conditioning—normally six after initialization.
- **Lifetime/recent best** shows the unscaled lifetime and recent estimates for
  the strongest containing doublet and triplet.

The “group score” and “Markov” values are different quantities: the former is
the weighted blend of separately scaled components, while the latter is the
unscaled conditional estimate \(Q_{k,a}\).

## Descriptive state statistics

Across the repository's 771 chronological draws, the production shape states
at the current dataset endpoint are:

| State | Draws | Share |
|---|---:|---:|
| No doublet | 397 | 51.4916% |
| Doublet without triplet | 338 | 43.8392% |
| Triplet | 36 | 4.6693% |

The 770 adjacent-draw transitions are:

| Previous state → next state | No doublet | Doublet only | Triplet |
|---|---:|---:|---:|
| No doublet | 209 | 173 | 15 |
| Doublet only | 174 | 144 | 19 |
| Triplet | 14 | 20 | 2 |

These counts are inputs to the fitted endpoint state, not an independent test
of a stable transition law. Overlapping groups, repeated use of the same draws,
and retrospective model design limit inferential interpretation.

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

A leakage-free replay over the repository's 771 draws produces 770 target
forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 601 | 0.780519 | 565.714 |
| Validation, target draws 121–520 | 400 | 303 | 0.757500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 205 | 0.820000 | 183.673 |

The replay is above the theoretical random mean on the full and holdout slices,
but the model and broader strategy collection were developed in the context of
available history. These measurements do not establish statistical
significance, independence from selection effects, or future predictability.

## Core mathematical and statistical concepts

- **Subset indicators:** each fixed consecutive group is treated as a binary
  event within each draw.
- **Combinatorial baseline:** hypergeometric containment supplies the exact
  random probability for one predefined doublet or triplet.
- **Shrinkage:** lifetime rates use an eight-draw-equivalent null prior, recent
  rates shrink toward lifetime, and transition rates shrink toward lifetime.
- **First-order conditioning:** the next group estimate depends on membership
  of each number in the immediately preceding draw.
- **Hard rolling window:** recent recurrence uses at most 120 completed draws.
- **State transition smoothing:** a three-state empirical prior stabilizes the
  latest shape-transition row.
- **Feature scaling:** each component is normalized across starts before fixed
  weighting, emphasizing relative rather than absolute differences.
- **Local-to-global aggregation:** group support is mapped to individual
  numbers through maximum and mean membership effects.
- **Coverage heuristic:** a deterministic six-member admission pass rewards
  coherent high-priority groups before the final ranking.
- **Hypergeometric overlap:** standard Top-6 efficacy uses the null overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Retrospective specification:** group sizes, weights, prior strength, recent
  window, state multipliers, and bonus rules were chosen with historical data
  available.
- **Many candidate starts:** 48 doublets and 47 triplets create many chances for
  noisy historical differences to appear.
- **Overlapping events:** adjacent starts share numbers, so group indicators and
  their estimates are strongly dependent.
- **Conditional averaging:** averaging six previous-number rates is a heuristic,
  not a fitted joint probability model for the latest draw.
- **Cumulative transition history:** number-conditioned transitions do not
  expire, even though recent group recurrence uses a 120-draw window.
- **Empirical shape prior:** after history begins, the cold-start shape prior is
  replaced by unsmoothed lifetime state shares.
- **Separate min–max scaling:** component magnitude and uncertainty are lost,
  and very small raw differences can be expanded to the full 0–1 range.
- **Cross-size comparability:** doublet and triplet group scores are normalized
  separately before fixed weighting.
- **Boundary asymmetry:** numbers near 1 and 49 belong to fewer non-wrapping
  groups than central numbers.
- **Admission heuristic:** the bonus is deterministic coverage engineering, not
  a probability calculation, and its admitted set need not equal the final
  Top-6.
- **No uncertainty interval:** displayed group and number scores do not report
  sampling variance or confidence.
- **Tie-break influence:** current gap can decide ranks when final scores tie.
- **Dataset and selection dependence:** favorable replay slices can arise from
  chance, repeated comparison, or adaptation to the available history.
- **No guaranteed predictability:** consecutive patterns in past lottery draws
  do not imply a causal mechanism governing future draws.

Use the strategy as an auditable consecutive-run and transition model, not as
evidence that a fair lottery has a persistent Markov dependency.

## Implementation map

| Responsibility | Production location |
|---|---|
| Prior strength, recent-window length, identifiers, and registration order | `src/rand_ai/strategy_prediction.py`, `_DOUBLET_TRIPLET_MARKOV_*` and strategy constants |
| Lifetime counts, transition matrices, shape state, and recent buffers | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Consecutive group extraction and three-state classification | `src/rand_ai/strategy_prediction.py`, `_consecutive_group_starts` and `_doublet_triplet_shape_state` |
| Causal group and shape transition updates | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train` |
| Lifetime counts, recent group buffer, and latest-draw state | `src/rand_ai/strategy_prediction.py`, `_StrategyState.remember` |
| Lifetime, recent, conditional, shape, support, admission-bonus, and detail calculations | `src/rand_ai/strategy_prediction.py`, `_doublet_triplet_markov_scores` |
| Min–max scaling, score/gap/number tie-break, Top-6, and efficacy | `src/rand_ai/strategy_prediction.py`, `_scale_scores`, `_ranking_from_scores`, `_strategy`, and `build_prediction_suites` |
| Desktop registration, serialization, and default-enabled state | `web/electron/main.cjs` and `src/rand_ai/gui_bridge.py` |
| Settings description, family, color, names, and displayed details | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Group extraction, transition learning, supported-triplet, and causal-prefix tests | `tests/test_strategy_prediction.py` |
| Strategy registration and serialization coverage | `tests/test_gui_bridge.py` and `web/src/lib/strategyFamilies.test.ts` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The standard desktop payload contains all 49 scaled scores, ranks, current
gaps, detail strings, Top-6 numbers, and completed efficacy. It does not expose
the internal group-start matrices or shape-transition table as separate
strategy payload fields.
