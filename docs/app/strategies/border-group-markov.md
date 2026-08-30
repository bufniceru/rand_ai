# Border Group Markov

## Introduction

**Border Group Markov** is the production strategy with identifier
`border_group_markov`. It is a default-enabled member of the
**Border Space Groups** family. The strategy converts every draw into a
canonical circular group signature, learns first-order transitions between
successive signatures, and shrinks the latest transition row toward the
smoothed lifetime signature distribution.

That categorical next-signature forecast is decoded through valid ticket shapes
and historical anchors into a complete ranking of numbers 1 through 49. Its
first six ranks form the Top-6 prediction used by grids, audits, effectiveness
histories, comparisons, portfolios, exports, and Possible Draw.

```{admonition} First-order signature model
:class: important

The Markov state is the immediately preceding **group signature**, not the
complete draw and not the six individual numbers. Earlier signatures influence
the categorical forecast only through cumulative transition counts and the
lifetime backoff distribution.
```

## Scope

The strategy answers two sequential questions:

1. Given the signature of the latest completed draw, which signature should be
   assigned probability for the next draw?
2. Given that signature distribution, which numbers receive the most mass
   across leakage-safe valid-ticket beams?

Border Group Markov differs from the other family members:

- Statistical ignores the latest state and uses only cumulative signature
  frequencies.
- Bayesian adds several categorical context variables.
- ML uses an online multinomial classifier.
- SVC uses a balanced RBF classifier with bounded scheduled refitting.
- Hybrid combines all five component forecasts using recent log loss.

The Markov strategy uses only one-step signature transitions with a fixed
ten-observation statistical backoff. It has no higher-order transition context,
recent-transition window, or adaptive smoothing strength.

## Circular spaces and groups

For sorted draw

\[
1\leq n_1<n_2<\cdots<n_6\leq49,
\]

the six empty-number spaces are

\[
s_0=(n_1-1)+(49-n_6),
\]

\[
s_i=n_{i+1}-n_i-1,
\qquad i=1,\ldots,5.
\]

They are non-negative integers satisfying

\[
\sum_{i=0}^{5}s_i=43.
\]

The first space crosses the 49-to-1 boundary, so grouping is circular.

The **Border space** setting is an integer \(b\) from 0 through 43, with default
7. Its interpretation is inclusive:

- \(s_i\leq b\) connects adjacent numbers;
- \(s_i>b\) separates two groups.

If no separator exists, all six positions form one group. Otherwise the number
of circular groups equals the number of separators. Changing the border changes
the state assigned to every draw and therefore rebuilds the complete transition
history.

## Canonical signature state

The sizes of the circular connected groups are sorted from largest to smallest.
This removes traversal and rotation ambiguity and maps every draw to one of the
11 integer partitions of six:

| State index | Signature | Group count |
|---:|---|---:|
| 0 | `6` | 1 |
| 1 | `5+1` | 2 |
| 2 | `4+2` | 2 |
| 3 | `4+1+1` | 3 |
| 4 | `3+3` | 2 |
| 5 | `3+2+1` | 3 |
| 6 | `3+1+1+1` | 4 |
| 7 | `2+2+2` | 3 |
| 8 | `2+2+1+1` | 4 |
| 9 | `2+1+1+1+1` | 5 |
| 10 | `1+1+1+1+1+1` | 6 |

Let \(Z_t\) be the state index for completed draw \(t\). At the default border,
state 10 is impossible: six separators would each require at least eight empty
positions, exceeding the available total of 43.

Feasibility is calculated exactly for every border from rooted circular gap
compositions. Infeasible states always receive probability zero.

## Smoothed lifetime backoff

Let \(C_s(t)\) be the number of completed draws through \(t\) with signature
state \(s\), and let \(\mathcal F_b\) be the feasible states for border \(b\).
The statistical backoff is

\[
B_t(s)=
\begin{cases}
\dfrac{C_s(t)+1}
      {\sum_{u\in\mathcal F_b}C_u(t)+|\mathcal F_b|},
&s\in\mathcal F_b,\\[8pt]
0,&s\notin\mathcal F_b.
\end{cases}
\]

This is the posterior mean for a multinomial distribution under one symmetric
pseudocount for each feasible signature. It prevents a feasible but unseen
state from receiving zero backoff mass.

The exact unequal 6/49 signature probabilities are used for feasibility and
diagnostic comparison, not as this backoff distribution. The Markov prior is
therefore data-adaptive and approaches empirical lifetime frequencies as
history grows.

## First-order transition estimate

Let

\[
T_{r,s}(t)
=\sum_{i=2}^{t}\mathbf 1[Z_{i-1}=r,\,Z_i=s]
\]

be the cumulative transition count from state \(r\) to state \(s\), with row
total

\[
M_r(t)=\sum_sT_{r,s}(t).
\]

For the latest state \(r=Z_t\), Border Group Markov forecasts

\[
P_{t+1}(s\mid Z_t=r)
=\frac{T_{r,s}(t)+10B_t(s)}{M_r(t)+10}.
\]

The formula is already normalized because the transition row sums to \(M_r\)
and the backoff sums to one. Production normalizes once more defensively.

This can be interpreted as a Dirichlet posterior mean for the selected row with
prior concentration 10 and prior mean \(B_t\). Because \(B_t\) is itself
estimated from the same history, the construction is an empirical-Bayes-style
backoff rather than a fixed-prior Markov chain.

### Sparse-row behavior

- With no completed profile, the forecast is the statistical backoff.
- After the first profile, its outgoing row has no transition yet, so the
  Markov forecast again equals the backoff.
- As a row accumulates transitions, their relative counts increasingly dominate
  the ten-observation prior.
- A feasible target unseen from the selected row still receives nonzero mass
  through \(10B_t(s)\).

All transition counts are cumulative. There is no recency decay, rolling
window, or minimum-support gate.

## Manual group-count conditioning

The **Predicted groups** setting is **Automatic** by default. If a user selects
an exact feasible group count \(g\), the Markov distribution is conditioned
after transition smoothing:

\[
P_{t+1}(s\mid Z_t=r,|s|=g)=
\begin{cases}
\dfrac{P_{t+1}(s\mid Z_t=r)}
      {\sum_{u:|u|=g}P_{t+1}(u\mid Z_t=r)},
&|s|=g,\\[10pt]
0,&|s|\ne g.
\end{cases}
\]

Impossible border/group-count combinations are rejected. Manual conditioning
is a user constraint, not a state inferred by the Markov model, and it affects
all Border Group forecasts and related portfolio guidance.

## Shared valid-ticket decoder

The categorical forecast does not identify particular numbers. The shared
Border Group decoder builds one ticket beam for each signature using:

- up to the latest 16 observed six-space shapes for that signature;
- deterministic feasible fallback shapes, each contributing one pseudocount;
- lifetime anchor counts for that signature with additive-one smoothing.

### Shape weights

For signature \(s\), let \(c_{s,q}\) be the count of observed and fallback shape
\(q\). Its within-signature weight is

\[
w_{s,q}=\frac{c_{s,q}}{\sum_vc_{s,v}}.
\]

### Anchor weights

For shape \(q=(q_0,\ldots,q_5)\), allowed anchors are

\[
a=0,\ldots,q_0.
\]

If \(A_{s,a}\) is the lifetime anchor count for signature \(s\), the shape's
normalized anchor weight is

\[
w_{s,q}(a)=
\frac{A_{s,a}+1}{\sum_{r=0}^{q_0}(A_{s,r}+1)}.
\]

The decoded sorted ticket is

\[
n_1=a+1,
\qquad
n_{i+1}=n_i+q_i+1,
\quad i=1,\ldots,5.
\]

Every ticket is valid, unique, sorted, and contained in 1 through 49. Duplicate
shape/anchor paths leading to the same ticket have their weights added. The
result is a normalized within-signature distribution \(D_s(T)\).

## Number marginals and ranking

For next-signature probability \(P_{t+1}(s)\), the production number mass is

\[
M_t(n)=\sum_sP_{t+1}(s)
       \sum_{T:n\in T}\frac{D_s(T)}{6}.
\]

The division by six makes the 49 values sum to one. This is normalized number
mass, not a calibrated probability that \(n\) will appear. Under the decoded
ticket mixture, raw inclusion mass is six times \(M_t(n)\).

All marginals are min–max scaled:

\[
S_t(n)=
\begin{cases}
\dfrac{M_t(n)-M_{\min}}{M_{\max}-M_{\min}},
&M_{\max}>M_{\min},\\[8pt]
0,&M_{\max}=M_{\min}.
\end{cases}
\]

Numbers are ranked by descending score, then larger current recurrence gap,
then smaller number. The first six form the Top-6 prediction. The scaled score
is relative within one target and is not directly comparable across draws.

## Causal lifecycle

For completed draw \(t\), the production order is:

1. Evaluate the pending categorical forecast for \(t\) before updating any
   state.
2. Convert \(D_t\) into its circular profile and actual signature \(Z_t\).
3. If a previous profile exists, increment transition
   \(T_{Z_{t-1},Z_t}\).
4. Increment the lifetime signature count, append the profile, and update its
   recent shape and lifetime anchor evidence.
5. Calculate the smoothed lifetime backoff including \(D_t\).
6. Select row \(Z_t\), apply the ten-observation backoff, optionally condition
   on manual group count, and decode the forecast for target \(t+1\).

The observed target is never added before its own forecast is evaluated. It can
affect only later forecasts. The decoder likewise uses shapes and anchors only
through the latest completed draw.

## Interpreting prediction details

Every number displays four shared Border Group details:

- **Border space** reports the inclusive connection threshold.
- **Model-selected group count** indicates automatic signature forecasting;
  **Manual target … groups** identifies an explicit conditioning constraint.
- **Decoded marginal** reports the unscaled normalized number mass \(M_t(n)\).
- **Leading signatures** lists the three largest Markov probabilities, with
  canonical signature order resolving exact ties.

Rank, exact current gap, normalized score, and Top-6 membership are supplied by
the standard prediction payload. The detail strings do not expose the complete
11-by-11 transition matrix or the selected row's raw support.

## Endpoint transition example

After all 771 repository draws at border 7, the latest signature is `5+1`. Its
historical outgoing row has 168 transitions:

| Next signature | Row count | Smoothed forecast |
|---|---:|---:|
| `6` | 18 | 10.7598% |
| `5+1` | 43 | 25.3802% |
| `4+2` | 28 | 16.8381% |
| `4+1+1` | 19 | 11.1921% |
| `3+3` | 8 | 5.0483% |
| `3+2+1` | 38 | 22.5568% |
| `3+1+1+1` | 2 | 1.2027% |
| `2+2+2` | 8 | 4.6814% |
| `2+2+1+1` | 4 | 2.3335% |
| `2+1+1+1+1` | 0 | 0.0072% |
| `1+1+1+1+1+1` | 0 | 0% |

The very rare feasible five-group state retains nonzero probability through the
lifetime backoff despite having no transition in this row. The infeasible
six-group state remains exactly zero.

## Signature-forecast statistics

The Space Groups analysis begins categorical evaluation after 100 prior draws.
It reports log loss, multiclass Brier score, exact-signature accuracy,
group-count accuracy, and group-count mean absolute error. For actual state
\(Y_t\) and probability vector \(\mathbf p_t\):

\[
\operatorname{LogLoss}_t
=-\log(\max(p_{t,Y_t},10^{-15})),
\]

\[
\operatorname{Brier}_t
=\sum_s(p_{t,s}-\mathbf 1[s=Y_t])^2.
\]

On the repository's 671 post-warm-up evaluations at border 7:

| Forecast | Log loss | Brier score | Signature accuracy | Group-count accuracy | Group-count MAE |
|---|---:|---:|---:|---:|---:|
| Border Group Markov | 1.971747 | 0.847768 | 19.5231% | 41.4307% | 0.667660 |
| Border Group Statistical | 1.928158 | 0.835644 | 19.9702% | 41.2817% | 0.663189 |
| Exact random 6/49 null | 1.913997 | 0.833116 | 21.3115% | 33.9791% | 0.770491 |

The Markov forecast is marginally higher on group-count accuracy than the
statistical model, but it is worse on log loss, Brier score, exact-signature
accuracy, and group-count MAE. The exact null has the best log loss, Brier
score, and exact-signature accuracy in this replay. These mixed and mostly
negative comparisons do not establish useful transition predictability.

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

A leakage-free replay using border 7 and automatic group count over the
repository's 771 chronological YAML draws produces 770 target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 571 | 0.741558 | 565.714 |
| Validation, target draws 121–520 | 400 | 307 | 0.767500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 172 | 0.688000 | 183.673 |

The full result is close to the theoretical random mean and the holdout is
below it. The stronger validation slice does not persist. These are
retrospective implementation measurements, not evidence of statistical
significance, stable future lift, or guaranteed predictability.

## Core mathematical and statistical concepts

- **Circular gap composition:** six empty-number spaces summing to 43 define
  connectivity without breaking the 49-to-1 boundary.
- **Integer partitions:** sorted connected-group sizes map every draw to one of
  11 canonical categorical states.
- **First-order Markov transition:** the next signature distribution is selected
  from the row identified by the latest signature.
- **Dirichlet shrinkage:** ten observations of prior strength pull a sparse row
  toward the smoothed lifetime distribution.
- **Empirical Bayes:** the prior mean is itself estimated from cumulative
  signature counts.
- **Conditional probability:** manual group-count selection renormalizes over a
  subset of states.
- **Mixture decoding:** signature, shape, and anchor weights form a distribution
  over valid tickets.
- **Marginalization:** valid-ticket weights are summed into number mass.
- **Proper categorical scoring:** log loss and Brier score evaluate the full
  signature probability vector.
- **Hypergeometric overlap:** Top-6 efficacy uses the null overlap distribution
  for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Negative holdout:** the current Top-6 holdout is below theoretical random
  expectation.
- **Categorical metrics trail baselines:** the Markov row does not improve log
  loss or Brier score over the statistical or exact-null references here.
- **First-order compression:** one canonical signature discards exact spaces,
  group order, number identities, and histories older than one state at row
  selection time.
- **Cumulative transitions:** no recency decay allows old transitions to retain
  full influence indefinitely.
- **Fixed smoothing:** concentration 10 is an engineering constant rather than
  an estimated state-specific uncertainty level.
- **Data-dependent prior:** row evidence and lifetime backoff reuse the same
  sequence, so they are not independent information sources.
- **Rare rows and targets:** some signatures have few outgoing transitions,
  making estimates sensitive to individual observations.
- **Threshold sensitivity:** changing border space reclassifies all draws and
  can substantially change the matrix and decoder.
- **Manual-selection bias:** choosing a group count after inspecting history
  adds an unmodeled selection step.
- **Decoder truncation:** only the latest 16 observed shapes per signature are
  retained, while anchor evidence is cumulative.
- **Constructed fallback shapes:** deterministic pseudoshapes are coverage
  priors, not samples from the exact ticket distribution.
- **Two-stage error:** signature uncertainty and decoder approximation both
  affect the final number ranking.
- **Uncalibrated number mass:** decoded marginals and min–max scores are not
  validated number probabilities.
- **Tie-break influence:** exact current gap can order equal decoded scores
  outside the signature model.
- **Dataset and model-selection dependence:** retrospective differences can
  arise from chance and repeated comparison.
- **No guaranteed predictability:** observed signature transitions do not imply
  a causal Markov mechanism in a fair lottery.

Use Border Group Markov as an auditable first-order categorical transition
baseline, not as evidence that past circular group structure controls future
draws.

## Implementation map

| Responsibility | Production location |
|---|---|
| Borders, canonical signatures, feasibility, identifiers, and names | `src/rand_ai/space_groups.py`, module constants, validators, and exact-null helpers |
| Circular spaces, separators, groups, and canonical profile | `src/rand_ai/space_groups.py`, `spaces_for_numbers`, `profile_from_spaces`, and `profile_for_numbers` |
| Lifetime signature counts and 11-by-11 transition matrix | `src/rand_ai/space_groups.py`, `SpaceGroupForecaster.__init__` and `observe` |
| Smoothed statistical backoff | `src/rand_ai/space_groups.py`, `SpaceGroupForecaster._statistical` |
| Selected-row first-order forecast with concentration 10 | `src/rand_ai/space_groups.py`, `SpaceGroupForecaster._markov` |
| Manual group-count conditioning and family forecast assembly | `src/rand_ai/space_groups.py`, `condition_signature_probabilities` and `SpaceGroupForecaster.forecast` |
| Fallback shapes, recent signature shapes, anchor smoothing, and valid tickets | `src/rand_ai/space_groups.py`, `_fallback_shapes` and `SpaceGroupForecaster._signature_candidates` |
| Number marginals, min–max scaling, and detail strings | `src/rand_ai/space_groups.py`, `_signature_marginals` and `number_scores` |
| Walk-forward metrics and exact-null comparison | `src/rand_ai/space_groups.py`, `_metrics` and `walk_forward_models` |
| Causal integration, standard tie-break, Top-6, and efficacy | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train`, `_StrategyState.build_strategies`, and `_ranking_from_scores` |
| Analysis tables and complete forecast payload | `src/rand_ai/statistics.py`, `DrawsStatistics.space_group_analysis` |
| Desktop registration, serialization, and default-enabled state | `web/electron/main.cjs` and `src/rand_ai/gui_bridge.py` |
| Settings description, family, color, names, and detail rendering | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Profiles, exact null, online transitions, conditioning, decoding, and metrics tests | `tests/test_space_groups.py` |
| Strategy selection, ranking shape, manual target, and serialization coverage | `tests/test_strategy_prediction.py` and `tests/test_gui_bridge.py` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The standard desktop strategy payload contains all 49 scaled scores, ranks,
current gaps, detail strings, Top-6 numbers, and completed efficacy. The broader
Space Groups report exposes the full categorical forecast and model metrics;
the transition matrix itself appears through analysis tables rather than every
number's details.
