(border-group-hybrid)=
# Border Group Hybrid

## Introduction

**Border Group Hybrid** is the production strategy with identifier
`border_group_hybrid`. It is a default-enabled member of the **Border Space
Groups** family.

The engine combines four different forecasts of the next circular group
signature:

1. cumulative statistical frequencies;
2. a first-order Markov transition model;
3. a categorical Bayesian context model; and
4. an online multinomial machine-learning model.

The four categorical probability vectors are blended using their recent causal
log loss. A shared decoder then maps the hybrid signature distribution through
valid space shapes and historical anchors into a complete ranking of numbers
1–49.

```{admonition} A hybrid of signature forecasts
:class: important

The adaptive weights evaluate how well each component predicted complete
**group signatures**. They do not directly optimize Top-6 hits, individual
number probabilities, or ticket profit.
```

## Scope and role

Border Group Hybrid answers three linked questions:

1. How should the next draw be partitioned into circularly connected groups?
2. Which component forecasts have recently assigned more probability to the
   signatures that actually occurred?
3. Given the blended signature distribution, which numbers receive the most
   mass across valid decoded tickets?

Its first six number ranks form the Top-6 prediction used by prediction grids,
audits, effectiveness histories, comparisons, portfolios, exports, and
Possible Draw.

Selecting only the hybrid creates one `SpaceGroupForecaster` that calculates
all four component forecasts internally. The component strategies do not have
to be selected for display, and only the requested hybrid is serialized.

The strategy shares the application-wide **Border space** and optional
**Predicted groups** settings with every Border Group engine. Changing either
setting changes its response space or conditions its forecast.

## Circular draw representation

For a sorted draw

```{math}
1\le n_1<n_2<\cdots<n_6\le49,
```

the engine constructs six empty-number spaces:

```{math}
s_0=(n_1-1)+(49-n_6),
```

```{math}
s_i=n_{i+1}-n_i-1,
\qquad i=1,\ldots,5.
```

The first space crosses the 49-to-1 boundary. All six values are non-negative
integers and

```{math}
\sum_{i=0}^{5}s_i=49-6=43.
```

This representation treats the draw as a circle rather than introducing an
artificial edge between 49 and 1.

## Border threshold and group signatures

The border setting is an integer \(b\) from 0 through 43, with default 7. Its
comparison is inclusive:

- \(s_i\le b\) connects the numbers on the two sides of the space;
- \(s_i>b\) separates two groups.

Groups are maximal connected runs around the six-position circle. If no
separator exists, all six numbers form one group. Otherwise, the number of
groups equals the number of separators.

The circular group sizes are sorted from largest to smallest to form a
rotation-independent signature. Every draw maps to one of the 11 integer
partitions of six:

| Index | Signature | Group count |
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

When equal maximum separators provide several possible traversal anchors, the
separator followed by the smallest number is selected. Sorting the resulting
group sizes makes the canonical signature independent of that traversal.

## Exact feasibility reference

For every border, production enumerates separator masks and counts rooted
non-negative gap compositions summing to 43. The total is

```{math}
\binom{48}{5}=1{,}712{,}304.
```

The counts determine which signatures are feasible and provide the exact
uniform-random reference distribution. At border 7:

| Signature | Exact probability |
|---|---:|
| `6` | 11.4821% |
| `5+1` | 20.0936% |
| `4+2` | 20.0936% |
| `4+1+1` | 10.3608% |
| `3+3` | 10.0468% |
| `3+2+1` | 20.7216% |
| `3+1+1+1` | 1.4913% |
| `2+2+2` | 3.4536% |
| `2+2+1+1` | 2.2370% |
| `2+1+1+1+1` | 0.0196% |
| `1+1+1+1+1+1` | 0% |

Six groups are impossible at border 7 because six separators would each need
at least eight empty positions, exceeding the available total of 43.
Infeasible signatures receive zero from every component and from the hybrid.

The exact probabilities are a diagnostic and feasibility reference. They are
not used as the statistical component's prior and are not one of the four
hybrid components.

## Component 1: statistical frequencies

Let \(C_s(t)\) be the number of completed draws through \(t\) with signature
\(s\), and let \(\mathcal F_b\) be the feasible signatures at border \(b\).
The statistical forecast is

```{math}
P_{\mathrm{stat},t+1}(s)=
\begin{cases}
\dfrac{C_s(t)+1}
      {\sum_{u\in\mathcal F_b}C_u(t)+|\mathcal F_b|},
&s\in\mathcal F_b,\\[8pt]
0,&s\notin\mathcal F_b.
\end{cases}
```

This is cumulative multinomial estimation with one symmetric pseudocount per
feasible signature. It has no rolling window or recency decay.

## Component 2: first-order Markov

Let \(Z_t\) be the latest completed signature, and define cumulative transition
counts

```{math}
T_{r,s}(t)=
\sum_{i=2}^{t}\mathbf1[Z_{i-1}=r,Z_i=s],
\qquad
M_r(t)=\sum_sT_{r,s}(t).
```

For the row \(r=Z_t\), the Markov component shrinks transition evidence toward
the statistical forecast with fixed strength 10:

```{math}
P_{\mathrm{markov},t+1}(s)=
\frac{T_{r,s}(t)+10P_{\mathrm{stat},t+1}(s)}{M_r(t)+10}.
```

It is a first-order cumulative model. There is no higher-order context,
transition decay, recent window, or row-specific learned smoothing strength.

## Component 3: categorical Bayesian context

The Bayesian component derives five categorical features from history ending
at draw \(t\):

| Feature | Categories | Production definition |
|---|---:|---|
| Current signature | 11 | \(Z_t\) |
| Current group count | 6 | number of parts in \(Z_t\), internally 0–5 |
| Maximum-space bucket | 4 | 0 for ≤7, 1 for 8–11, 2 for 12–15, 3 for ≥16 |
| Recent modal signature | 11 | most common signature in the latest 10 draws |
| Group-count trend | 3 | decreasing, stable, or increasing over up to 25 draws |

For the trend, the recent sequence is divided into an earlier and later part.
A mean group-count change greater than 0.2 is increasing, a decrease greater
than 0.2 is decreasing, and all other cases are stable. Histories shorter than
four profiles are neutral.

Let the feature vector be \(\mathbf f=(f_1,\ldots,f_5)\), with cardinalities

```{math}
(K_1,\ldots,K_5)=(11,6,4,11,3).
```

For feasible target signature \(s\), production forms the naïve-Bayes score

```{math}
Q_s=
\frac{C_s+1}{t+11}
\prod_{j=1}^{5}
\frac{N_j(s,f_j)+1}{C_s+K_j},
```

where \(N_j(s,f_j)\) counts completed causal training pairs with target
signature \(s\) and feature value \(f_j\). The values are calculated in log
space, shifted by the maximum finite score, exponentiated, and normalized:

```{math}
P_{\mathrm{bayes},t+1}(s)=
\frac{Q_s}{\sum_uQ_u}.
```

The conditional-independence assumption is an approximation. The features are
strongly related: current signature determines current group count, and the
modal and trend fields are derived from the same recent signature history.

## Component 4: online multinomial model

The ML component uses an averaged `SGDClassifier` with:

- multinomial log-loss training through `predict_proba`;
- L2 penalty;
- regularization coefficient `alpha=0.001`;
- parameter averaging enabled; and
- deterministic random state 0.

Its exact 82-value feature vector contains:

| Block | Width | Values |
|---|---:|---|
| Lags 1–3 | 39 | 11-value signature one-hot, group count / 6, and maximum space / 43 for each lag |
| Latest six spaces | 6 | each circular space / 43 |
| Windows 10, 25, and 100 | 36 | 11 signature frequencies and mean group count / 6 per window |
| Trend | 1 | category divided by 2 |
| **Total** | **82** | |

Unavailable lag blocks are zero. The feature vector built after draw \(t\) is
stored and trained only when signature \(Z_{t+1}\) becomes known. The model is
updated one delayed example at a time with `partial_fit` and all 11 signature
classes declared on the first update.

Until 50 delayed examples have been trained, the ML component returns the
statistical forecast. After warm-up, predicted mass for signatures infeasible
at the selected border is set to zero and the remaining values are normalized.

## Manual group-count conditioning

**Predicted groups** is **Automatic** by default. If the user selects a feasible
exact count \(g\), every component distribution is conditioned before hybrid
weighting:

```{math}
P_m(s\mid |s|=g)=
\begin{cases}
\dfrac{P_m(s)}{\sum_{u:|u|=g}P_m(u)},&|s|=g,\\[8pt]
0,&|s|\ne g.
\end{cases}
```

Impossible border/count combinations are rejected. This is a manual
constraint, not a group-count forecast inferred by the hybrid. Because the
conditioned component forecasts are retained for evaluation, excluding the
eventual actual group count produces a near-zero-probability log-loss penalty.

## Leakage-free component losses

For each component \(m\), the forecaster retains at most the latest 100
completed losses. If the actual next signature is \(Y_t\), its loss is

```{math}
\ell_{m,t}=-\log\!\left(\max(P_{m,t}(Y_t),10^{-15})\right).
```

The loss is computed from the pending forecast before the actual profile is
added or either trainable component is updated. The hybrid's own loss is
recorded for reporting but is not an input to its adaptive weights. The exact
random null is also excluded from weighting.

## Adaptive hybrid weights

Until every component has at least 30 completed losses, all weights are exactly

```{math}
w_m=\frac14=25\%.
```

After warm-up, let \(\bar\ell_m\) be component \(m\)'s mean loss over its
trailing buffer and define

```{math}
r_m=\exp(-\bar\ell_m).
```

Because log loss is the negative log probability assigned to the realized
class, \(r_m\) is the geometric mean probability the component assigned to
recent realized signatures.

Production reserves a 5% floor for each of the four components and distributes
the remaining 80% in proportion to \(r_m\):

```{math}
w_m=0.05+0.80\frac{r_m}{\sum_kr_k}.
```

Consequently,

```{math}
\sum_mw_m=1,
\qquad
w_m\ge0.05.
```

No component can be switched off completely. The weights use only categorical
log loss; they do not use Top-6 hits, Brier score, signature accuracy, or
group-count accuracy.

## Hybrid signature distribution

After optional manual conditioning, the next-signature forecast is the linear
probability pool

```{math}
P_{\mathrm{hybrid},t+1}(s)
=\sum_{m\in\{\mathrm{stat},\mathrm{markov},\mathrm{bayes},\mathrm{ML}\}}
w_mP_{m,t+1}(s).
```

The result is already normalized because every component distribution and the
weights each sum to one. This is probability averaging, not rank averaging,
majority voting, winner selection, or stacking with a fitted meta-model.

Linear pooling tends to moderate component-specific extremes, while the 5%
floor preserves diversity. It cannot create information absent from every
component, and correlated components can receive separate weight for largely
overlapping evidence.

## Shared valid-ticket decoder

A signature does not specify six numbers. For each signature, the shared
decoder builds a leakage-safe beam from:

- up to the latest 16 observed six-space shapes with that signature;
- deterministic valid fallback shapes, each contributing one pseudocount; and
- lifetime first-number anchor counts for that signature with additive-one
  smoothing.

### Shape distribution

For signature \(s\) and shape \(q\), let \(c_{s,q}\) include recent observed
occurrences and fallback pseudocounts. Its within-signature weight is

```{math}
D_s(q)=\frac{c_{s,q}}{\sum_vc_{s,v}}.
```

### Anchor distribution

For space shape \(q=(q_0,\ldots,q_5)\), valid anchor offsets are

```{math}
a\in\{0,\ldots,q_0\}.
```

If \(A_s(a)\) is the lifetime anchor count for the signature, the conditional
anchor weight is

```{math}
D_s(a\mid q)=
\frac{A_s(a)+1}{\sum_{r=0}^{q_0}(A_s(r)+1)}.
```

The sorted ticket is reconstructed by

```{math}
n_1=a+1,
\qquad
n_{i+1}=n_i+q_i+1,
\quad i=1,\ldots,5.
```

Every generated ticket is sorted, contains six distinct numbers from 1–49,
and has spaces summing to 43. Duplicate shape/anchor paths that create the same
ticket have their weights added.

## Number marginals and ranking

Let \(D_s(T)\) be the normalized within-signature ticket distribution. The
decoded number mass is

```{math}
M_t(n)=
\sum_sP_{\mathrm{hybrid},t+1}(s)
\sum_{T:n\in T}\frac{D_s(T)}6.
```

The factor \(1/6\) makes

```{math}
\sum_{n=1}^{49}M_t(n)=1.
```

It is normalized number mass, not a calibrated occurrence probability. Under
the decoder's ticket mixture, the corresponding inclusion marginal is
\(6M_t(n)\).

All 49 masses are min–max scaled:

```{math}
S_t(n)=
\begin{cases}
\dfrac{M_t(n)-M_{\min}}{M_{\max}-M_{\min}},
&M_{\max}>M_{\min},\\[8pt]
0,&M_{\max}=M_{\min}.
\end{cases}
```

If no decoded mass exists, production first substitutes \(1/49\) for every
number; min–max scaling then produces all-zero scores.

Numbers are ranked by:

1. larger scaled score;
2. larger current gap; then
3. smaller number.

The Top-6 consists of the first six individual-number ranks. It need not equal
one decoded ticket, because it is assembled from separate marginals.

## Causal lifecycle and leakage protection

For completed draw \(t\), production performs this sequence:

1. retrieve the pending component forecasts made after draw \(t-1\);
2. score those forecasts against actual signature \(Z_t\) and append their log
   losses;
3. train the ML model on the feature vector saved after \(t-1\) with target
   \(Z_t\);
4. update Bayesian feature counts saved after \(t-1\);
5. update the first-order transition, lifetime signature count, recent shape
   beam, and lifetime anchor counts with draw \(t\);
6. calculate all four component forecasts from the resulting completed state;
7. derive hybrid weights from completed losses and blend the components; and
8. decode and retain the forecast for target draw \(t+1\).

The actual target cannot change its own forecast or its component weights. It
can influence only later targets. Historical ticket shapes and anchors also
enter the decoder only after their draws are completed.

## Cold start and warm-up

With no profile, all five Border Group forecasts return the same uniform
distribution over feasible signatures, optionally conditioned on manual group
count. This differs from the unequal exact 6/49 null distribution.

After the first completed profile:

- statistical frequencies begin adapting;
- the selected Markov row has no outgoing transition and therefore equals the
  statistical backoff;
- Bayesian context features can be prepared for the next delayed target;
- ML still returns the statistical forecast; and
- hybrid component weights remain 25% each.

Adaptive hybrid weights begin only after 30 pending forecasts have been
resolved. The ML component remains a statistical fallback until its 50th
delayed training example. Its early fallback losses are still attributed to
the ML component because those were its actual issued forecasts.

## Interpreting application fields

Each number exposes four shared Border Group details:

- **Border space** reports the inclusive connection threshold.
- **Model-selected group count** identifies automatic signature forecasting;
  **Manual target … groups** identifies explicit conditioning.
- **Decoded marginal** is the unscaled normalized number mass \(M_t(n)\).
- **Leading signatures** lists the three largest hybrid signature
  probabilities; canonical signature index resolves exact ties.

Rank, current gap, normalized score, and Top-6 membership are supplied by the
standard strategy payload.

Per-number details do not show all component vectors or adaptive weights. The
Space Groups analysis exposes the current component weights and the complete
categorical forecast table.

## Endpoint diagnostic

After all 771 repository draws at border 7 with automatic group count, every
component loss buffer contains its maximum 100 completed outcomes:

| Component | Trailing mean log loss | Current hybrid weight |
|---|---:|---:|
| Statistical | 1.949836 | 32.4644% |
| Markov | 1.984477 | 31.5293% |
| Bayesian | 2.152021 | 27.4369% |
| ML | 3.990346 | 8.5693% |

The 5% floor prevents the weaker recent ML component from disappearing. The
remaining weight ordering follows the exponentiated trailing mean losses, not
the full-history metrics below.

The endpoint hybrid signature forecast is:

| Signature | Hybrid probability |
|---|---:|
| `6` | 12.1021% |
| `5+1` | 27.8023% |
| `4+2` | 18.7128% |
| `4+1+1` | 9.0943% |
| `3+3` | 5.7330% |
| `3+2+1` | 20.5131% |
| `3+1+1+1` | 0.8859% |
| `2+2+2` | 3.6690% |
| `2+2+1+1` | 1.4142% |
| `2+1+1+1+1` | 0.0731% |
| `1+1+1+1+1+1` | 0% |

The shared decoder contains 2,874 distinct signature-ticket entries across the
ten feasible signature beams at this endpoint. These are fitted-state
diagnostics, not evidence that the same weights or signature distribution will
persist.

## Signature-forecast statistics

The Space Groups analysis begins reported model evaluation after 100 prior
draws. For actual signature \(Y_t\) and forecast vector \(\mathbf p_t\):

```{math}
\operatorname{LogLoss}_t
=-\log(\max(p_{t,Y_t},10^{-15})),
```

```{math}
\operatorname{Brier}_t
=\sum_s(p_{t,s}-\mathbf1[s=Y_t])^2.
```

Exact-signature accuracy selects the largest-probability state. Group-count
accuracy compares the number of parts in the predicted and actual signatures;
group-count MAE is their absolute difference.

On the repository's 671 post-warm-up evaluations at border 7:

| Forecast | Log loss | Brier | Signature accuracy | Group-count accuracy | Group-count MAE |
|---|---:|---:|---:|---:|---:|
| Statistical | 1.928158 | 0.835644 | 19.9702% | 41.2817% | 0.663189 |
| Markov | 1.971747 | 0.847768 | 19.5231% | 41.4307% | 0.667660 |
| Bayesian | 2.140277 | 0.889186 | 20.2683% | 44.7094% | 0.634873 |
| ML | 6.315110 | 1.321350 | 19.2250% | 42.7720% | 0.676602 |
| **Hybrid** | **1.963763** | **0.848200** | **20.4173%** | **44.1133%** | **0.633383** |
| Exact random 6/49 null | 1.913997 | 0.833116 | 21.3115% | 33.9791% | 0.770492 |

The hybrid has the best group-count MAE among these rows and improves exact
signature accuracy over every learned component. It does not improve log loss
or Brier score over the statistical component or exact null, and the Bayesian
component has slightly higher group-count accuracy. The metrics therefore give
mixed evidence rather than a general categorical advantage.

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

A leakage-free production replay at border 7 with automatic group count over
the repository's 771 chronological YAML draws produces 770 target forecasts:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 585 | 0.759740 | 565.714 |
| Validation, target draws 121–520 | 400 | 310 | 0.775000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 173 | 0.692000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, records 174 hits
or 0.696000 per target.

The full replay and validation slice exceed theoretical random expectation,
but the holdout and latest slice are below it. This lack of persistence is an
important negative result. The measurements are retrospective and do not
establish statistical significance, stable future lift, or predictability.

## Core mathematical and statistical concepts

- **Circular gap composition:** six empty spaces summing to 43 encode the draw
  without privileging the 49-to-1 boundary.
- **Connected components:** an inclusive threshold turns circular neighbors
  into groups separated by large spaces.
- **Integer partitions:** sorted group sizes create 11 canonical categorical
  states.
- **Multinomial smoothing:** feasible lifetime signatures receive symmetric
  additive-one mass.
- **Empirical-Bayes Markov backoff:** the latest transition row shrinks toward
  cumulative signature frequencies.
- **Naïve Bayes:** smoothed categorical likelihoods multiply under a
  conditional-independence approximation.
- **Online multinomial classification:** delayed averaged stochastic-gradient
  updates learn from an 82-value rolling feature vector.
- **Proper scoring rules:** recent log loss controls the adaptive weights;
  Brier score supplies a separate categorical evaluation.
- **Exponentially transformed loss:** \(e^{-\bar\ell}\) converts mean log loss
  into geometric mean realized-class probability.
- **Linear probability pooling:** normalized component vectors are averaged
  with non-negative weights summing to one.
- **Weight regularization:** a 5% floor prevents complete component exclusion.
- **Conditional probability:** manual group-count selection renormalizes each
  component over a subset of signatures.
- **Mixture decoding and marginalization:** signature, shape, and anchor mass
  become number-level rankings.
- **Hypergeometric overlap:** Top-6 evaluation uses the correct
  without-replacement null distribution.

## Limitations and responsible interpretation

- **Negative holdout behavior:** current decoded Top-6 holdout performance is
  below theoretical random expectation despite the stronger full replay.
- **Categorical baselines remain stronger on loss:** the hybrid trails the
  statistical component and exact null on full post-warm-up log loss and Brier
  score.
- **Objective mismatch:** weights optimize recent signature log loss, not
  number hits or the final decoded ranking.
- **Correlated components:** every component uses the same signature history,
  and Markov, Bayesian, and early ML forecasts reuse statistical information.
  The mixture is not four independent sources of evidence.
- **Short adaptive window:** trailing 100-draw losses can make weights sensitive
  to a modest number of rare, high-loss outcomes.
- **Probability floor:** clipping at \(10^{-15}\) bounds numerical loss but a
  manually excluded actual class still produces a very large penalty.
- **Minimum component weight:** the 5% floor preserves diversity even when a
  component's recent categorical performance is poor.
- **Fixed warm-ups and constants:** 30 losses, 50 ML examples, the 100-loss
  buffer, smoothing strengths, feature thresholds, and SGD settings are
  engineering choices rather than universally optimal estimates.
- **Naïve-Bayes dependence:** its context variables overlap substantially, so
  multiplying their likelihoods can produce overconfident probabilities.
- **ML instability and class imbalance:** rare signatures provide few online
  examples, while the current historical ML log loss is weak.
- **First-order and compressed state:** canonical signatures discard exact
  group order, most space geometry, number identities, and longer transition
  history.
- **Threshold sensitivity:** changing border space reclassifies every draw and
  retrains all four components and the decoder.
- **Manual-selection bias:** choosing a target group count after reviewing the
  same history adds an external selection step.
- **Decoder truncation:** only the latest 16 observed shapes per signature are
  retained, while anchor evidence is cumulative.
- **Constructed fallback shapes:** deterministic pseudoshapes ensure coverage
  but are not samples from the exact random ticket distribution.
- **Two-stage error:** a well-scored signature distribution can still become a
  weak number ranking through decoder approximation.
- **Uncalibrated number scores:** decoded mass and min–max scores are not
  validated number occurrence probabilities.
- **Tie-break influence:** current gap can decide exact score ties even though
  it is not part of the group-signature models.
- **Retrospective model selection:** the architecture and constants were
  developed with historical results available and coexist with many compared
  strategies.
- **No guaranteed predictability:** adaptive weights and apparent group
  structure can reflect chance and need not persist on untouched draws.

## Implementation map

The group representation, forecaster, component models, hybrid, decoder, and
categorical evaluation are implemented in `src/rand_ai/space_groups.py`:

- `spaces_for_numbers`, `profile_from_spaces`, and `profile_for_numbers` create
  circular profiles and canonical signatures;
- `exact_null_signature_counts` and `exact_null_probabilities` establish
  feasibility and the exact random reference;
- `SpaceGroupForecaster._statistical`, `_markov`, `_bayesian`, and the online
  classifier produce the four component distributions;
- `_bayes_features` and `_ml_features` construct their causal context inputs;
- `observe` scores pending forecasts, trains delayed components, and remembers
  each completed profile;
- `_hybrid_weights` converts trailing component losses into floored adaptive
  weights;
- `forecast` conditions component distributions and creates the hybrid linear
  pool;
- `_signature_candidates` and `_signature_marginals` implement the shared
  valid-ticket decoder;
- `number_scores` creates number masses and user-facing details; and
- `walk_forward_models` calculates the categorical report metrics after the
  100-profile reporting warm-up.

`src/rand_ai/strategy_prediction.py` owns the application lifecycle, invokes
the shared forecaster, applies the standard score/gap/number ranking, and
serializes `border_group_hybrid` when requested.

Relevant behavior is covered by `tests/test_space_groups.py` and
`tests/test_strategy_prediction.py`, including exact null feasibility,
normalization, delayed training, loss-based weights and their 5% floor, manual
conditioning, valid decoded tickets, number-score fallback, chronological
metrics, strategy registration, and prediction serialization.
