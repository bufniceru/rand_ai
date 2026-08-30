(border-group-bayesian)=
# Border Group Bayesian

## Introduction

**Border Group Bayesian** is the production strategy with identifier
`border_group_bayesian`. It is a default-enabled member of the **Border Space
Groups** family.

The strategy converts each completed draw into a circular group signature,
constructs five categorical context variables, and estimates a smoothed
naïve-Bayes posterior for the next signature. A shared valid-ticket decoder
then turns that categorical distribution into a complete ranking of numbers
1–49.

```{admonition} A contextual signature model
:class: important

The Bayesian response variable is the next draw's **group signature**, not the
presence or absence of an individual number. Its final number ranking is a
second-stage marginalization through historical space shapes and anchors.
```

## Scope and role

Border Group Bayesian answers two linked questions:

1. Given the latest signature, group count, maximum space, recent modal
   signature, and group-count trend, which canonical signature should receive
   probability for the next draw?
2. Given that signature distribution, which numbers receive the most mass
   across valid decoded tickets?

Its first six number ranks form the Top-6 prediction used by prediction grids,
audits, effectiveness histories, comparisons, portfolios, exports, and
Possible Draw.

The engine is distinct from:

- **Bayesian**, the separate number-level strategy in **Frequency & Recency**;
- **Border Group Statistical**, which uses only lifetime signature counts;
- **Border Group Markov**, which conditions only on the latest signature using
  a first-order transition row;
- **Border Group ML**, which learns an online linear classifier from a larger
  rolling feature vector;
- **Border Group SVC**, which fits a balanced RBF classifier to that larger
  causal feature vector with bounded scheduled retraining; and
- **Border Group Hybrid**, which combines all five component forecasts using
  recent log loss.

Selecting Border Group Bayesian creates the shared `SpaceGroupForecaster`, but
only the requested strategy is serialized for display.

## Circular spaces

For a sorted six-number draw

```{math}
1\le n_1<n_2<\cdots<n_6\le49,
```

the engine constructs six counts of empty numbers:

```{math}
s_0=(n_1-1)+(49-n_6),
```

```{math}
s_i=n_{i+1}-n_i-1,
\qquad i=1,\ldots,5.
```

The first space crosses the 49-to-1 boundary. Every space is a non-negative
integer and

```{math}
\sum_{i=0}^{5}s_i=49-6=43.
```

The representation is circular: numbers near 49 and 1 can belong to the same
connected group.

## Border threshold and connected groups

The **Border space** setting is an integer \(b\) from 0 through 43; its default
is 7. The comparison is inclusive:

- \(s_i\le b\) connects the number positions on its two sides;
- \(s_i>b\) is a separator between groups.

Groups are maximal connected runs around the six-position circle. With no
separator, all six numbers form one group. Otherwise, the number of groups
equals the number of separators.

Changing \(b\) reclassifies the complete draw history. A larger border treats
more spaces as connections and generally produces fewer groups; a smaller
border creates more separators.

## Canonical signature response

The sizes of the circular groups are sorted from largest to smallest. This
removes traversal and rotation ambiguity and maps every draw to one of the 11
integer partitions of six:

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

Let \(Z_t\in\{0,\ldots,10\}\) be the state index for completed draw \(t\).
For equal largest separator spaces, production chooses the separator followed
by the smallest number as a deterministic traversal anchor; the sorted
signature remains rotation-independent.

## Exact feasibility

For each border, the engine enumerates separator masks and rooted
non-negative gap compositions summing to 43. There are

```{math}
\binom{48}{5}=1{,}712{,}304
```

such compositions. Their signature counts establish exact feasibility and the
uniform-random diagnostic distribution.

At border 7, the exact reference is:

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

Six separate groups are impossible at border 7 because six separator spaces
would each require at least eight empty numbers, exceeding the total 43.
Infeasible signatures receive exactly zero Bayesian score.

The unequal exact probabilities are used for diagnostics and feasibility, not
as the Bayesian class prior.

## Exact five-feature context

After completed draw \(t\), production builds the categorical vector

```{math}
\mathbf F_t=(F_{t,1},\ldots,F_{t,5}).
```

Its fields are:

| Feature | Internal range | Interpretation |
|---|---:|---|
| Current signature | 0–10 | state index \(Z_t\) |
| Current group count | 0–5 | number of groups minus one |
| Maximum-space bucket | 0–3 | bucket containing \(\max_i s_i\) |
| Recent modal signature | 0–10 | most frequent state in the latest 10 profiles |
| Group-count trend | 0–2 | decreasing, stable, or increasing |

The categorical cardinalities used for smoothing are

```{math}
(K_1,K_2,K_3,K_4,K_5)=(11,6,4,11,3).
```

### Current signature and group count

The first feature retains the complete canonical partition. The second retains
only its number of parts. They are deliberately redundant: the signature
already determines group count, so these are not independent evidence sources.

### Maximum-space bucket

Let \(m_t=\max_i s_{t,i}\). The fixed production buckets are

```{math}
F_{t,3}=
\begin{cases}
0,&m_t\le7,\\
1,&8\le m_t\le11,\\
2,&12\le m_t\le15,\\
3,&m_t\ge16.
\end{cases}
```

These cut points do not change with the selected border-space setting.

### Recent modal signature

The fourth feature is the most common state among at most the latest ten
completed profiles. It summarizes recent categorical concentration but does
not preserve their order.

### Group-count trend

The trend uses at most the latest 25 profiles. With fewer than four profiles it
is stable. Otherwise, the sequence is divided at its midpoint, and the mean
group count of the later part is compared with the earlier part:

```{math}
F_{t,5}=
\begin{cases}
2,&\bar g_{\mathrm{late}}-\bar g_{\mathrm{early}}>0.2,\\
0,&\bar g_{\mathrm{early}}-\bar g_{\mathrm{late}}>0.2,\\
1,&\text{otherwise}.
\end{cases}
```

The categories mean increasing, decreasing, and stable respectively.

## Delayed categorical training counts

For target state \(s\), feature index \(j\), and category \(v\), define

```{math}
N_j(s,v;t)=
\sum_{i=2}^{t}
\mathbf1[Z_i=s,\,F_{i-1,j}=v].
```

The feature vector is built after draw \(i-1\), stored as pending, and counted
only when draw \(i\) becomes known. This creates causal next-signature training
pairs. It also means the first profile contributes to lifetime class counts but
has no preceding feature row.

The five tables are cumulative. There is no recent window, exponential decay,
or feature-specific forgetting.

## Smoothed naïve-Bayes posterior

Let \(C_s(t)\) be the lifetime count of state \(s\) through completed draw
\(t\). For the current feature values \(\mathbf f=\mathbf F_t\), production
starts with the additive-one class term

```{math}
\pi_s(t)=\frac{C_s(t)+1}{t+11}.
```

Each categorical likelihood is

```{math}
L_{j,s}(f_j;t)=
\frac{N_j(s,f_j;t)+1}{C_s(t)+K_j}.
```

For every feasible signature, the unnormalized score is

```{math}
Q_s(t+1)=
\pi_s(t)\prod_{j=1}^{5}L_{j,s}(f_j;t).
```

Infeasible signatures are assigned \(-\infty\) in log space and therefore zero
probability. For numerical stability, the implementation calculates

```{math}
\log Q_s
=\log\pi_s+\sum_{j=1}^{5}\log L_{j,s},
```

subtracts the maximum finite log score, exponentiates, and normalizes:

```{math}
P_{t+1}(s\mid\mathbf F_t)=
\frac{Q_s(t+1)}{\sum_{u\in\mathcal F_b}Q_u(t+1)}.
```

Additive-one smoothing keeps feasible unseen combinations finite. The model
does not back off to the Border Group Statistical probability vector as the
Markov component does; it constructs its own smoothed class and likelihood
terms from the shared counts.

## Bayesian interpretation

The calculation has the algebraic form of a multinomial naïve-Bayes posterior:

```{math}
P(s\mid f_1,\ldots,f_5)
\propto P(s)\prod_{j=1}^{5}P(f_j\mid s).
```

The additive-one terms correspond to symmetric categorical pseudocounts.
However, the word *Bayesian* should be interpreted narrowly:

- it is a plug-in posterior from cumulative counts, not a sampled posterior;
- it does not expose credible intervals or posterior uncertainty over its
  parameters;
- it assumes feature independence conditional on the target signature; and
- its engineered categories and smoothing strengths are fixed.

The first two features are deterministically related, while the recent mode and
trend derive from the same signature history. The independence approximation
can therefore count overlapping evidence more than once and create overly
concentrated probabilities.

## Manual group-count conditioning

**Predicted groups** is **Automatic** by default. If a user selects an exact
feasible group count \(g\), the Bayesian posterior is conditioned after its
normalization:

```{math}
P_{t+1}(s\mid\mathbf F_t,|s|=g)=
\begin{cases}
\dfrac{P_{t+1}(s\mid\mathbf F_t)}
      {\sum_{u:|u|=g}P_{t+1}(u\mid\mathbf F_t)},
&|s|=g,\\[10pt]
0,&|s|\ne g.
\end{cases}
```

Impossible border/count combinations are rejected. Manual conditioning is a
user constraint, not a group-count inference produced by the model, and it
affects all Border Group strategies and associated portfolio guidance.

## Shared valid-ticket decoder

The posterior identifies a group signature but not six numbers. For every
signature, the shared decoder maintains:

- up to the latest 16 observed six-space shapes with that signature;
- deterministic valid fallback shapes, each adding one pseudocount; and
- lifetime first-number anchor counts for the signature.

### Shape weights

For signature \(s\) and space shape \(q\), let \(c_{s,q}\) include recent
observations and fallback pseudocounts. The within-signature shape weight is

```{math}
D_s(q)=\frac{c_{s,q}}{\sum_vc_{s,v}}.
```

### Anchor weights

For \(q=(q_0,\ldots,q_5)\), the legal anchor offsets are

```{math}
a=0,\ldots,q_0.
```

With lifetime anchor count \(A_s(a)\), additive-one smoothing gives

```{math}
D_s(a\mid q)=
\frac{A_s(a)+1}{\sum_{r=0}^{q_0}(A_s(r)+1)}.
```

The decoded sorted ticket is

```{math}
n_1=a+1,
\qquad
n_{i+1}=n_i+q_i+1,
\quad i=1,\ldots,5.
```

Every ticket contains six distinct values from 1–49 and reproduces a valid
space pattern summing to 43. Duplicate paths to one ticket have their weights
added.

## Number marginals and ranking

Let \(D_s(T)\) be the normalized ticket distribution for signature \(s\). The
decoded number mass is

```{math}
M_t(n)=
\sum_sP_{t+1}(s\mid\mathbf F_t)
\sum_{T:n\in T}\frac{D_s(T)}6.
```

The division by six ensures

```{math}
\sum_{n=1}^{49}M_t(n)=1.
```

This is normalized number mass, not a calibrated occurrence probability. The
corresponding inclusion mass under the decoder's ticket mixture is
\(6M_t(n)\).

All number masses are min–max scaled:

```{math}
S_t(n)=
\begin{cases}
\dfrac{M_t(n)-M_{\min}}{M_{\max}-M_{\min}},
&M_{\max}>M_{\min},\\[8pt]
0,&M_{\max}=M_{\min}.
\end{cases}
```

Numbers are ranked by descending score, then larger current gap, then smaller
number. The first six form the Top-6. Because these are individual marginals,
the Top-6 need not be one of the decoder's joint tickets.

## Causal lifecycle and leakage protection

For completed draw \(t\), the Bayesian path is:

1. evaluate the pending forecast for draw \(t\) before updating state;
2. map the actual draw to signature \(Z_t\);
3. pair \(Z_t\) with the pending features \(\mathbf F_{t-1}\) and increment the
   five conditional count tables;
4. increment lifetime class count \(C_{Z_t}\), append the profile, and update
   its recent shape and lifetime anchor evidence;
5. construct \(\mathbf F_t\) from history through draw \(t\);
6. calculate the posterior for draw \(t+1\), optionally condition it on a
   manual group count, and save \(\mathbf F_t\) for delayed training; and
7. decode the posterior into number scores.

The draw being forecast cannot enter its own class counts, feature counts,
shapes, anchors, or number ranking. It affects only subsequent forecasts.

## Cold start and short history

With no completed profile, the forecaster returns a uniform distribution over
feasible signatures. This is the shared cold-start forecast and differs from
the unequal exact 6/49 null distribution.

After the first profile, the Bayesian class counts and current context exist,
but no delayed feature-target pair has yet been completed. Additive-one
likelihoods keep every feasible signature available. After draw 2, the first
pending feature row is associated with an observed next-signature target.

The recent mode uses however many profiles are available up to ten. Trend is
stable until at least four profiles exist, then uses up to 25. No field reads
or pads future observations.

## Interpreting application fields

Every number displays four shared Border Group details:

- **Border space** reports the inclusive connectivity threshold.
- **Model-selected group count** identifies automatic posterior forecasting;
  **Manual target … groups** identifies explicit conditioning.
- **Decoded marginal** is the unscaled normalized number mass \(M_t(n)\).
- **Leading signatures** lists the three largest Bayesian posterior
  probabilities; canonical state index resolves exact ties.

The standard payload adds rank, current gap, normalized score, and Top-6
membership. Per-number details do not expose the five current feature values or
all conditional count tables; the Space Groups analysis exposes the complete
signature forecast.

## Endpoint diagnostic

After all 771 repository draws at border 7, the next-forecast context is:

| Field | Endpoint value |
|---|---|
| Latest signature | `5+1` (state 1) |
| Current group count | 2 (internal category 1) |
| Maximum space | 28 (bucket 3) |
| Recent modal signature | `4+2` (state 2) |
| Group-count trend | stable (category 1) |
| Completed feature-target rows | 770 per feature table |

The Bayesian signature posterior for the next target is:

| Signature | Probability |
|---|---:|
| `6` | 10.3437% |
| `5+1` | 32.7380% |
| `4+2` | 17.1545% |
| `4+1+1` | 8.8906% |
| `3+3` | 3.4272% |
| `3+2+1` | 23.3884% |
| `3+1+1+1` | 0.0618% |
| `2+2+2` | 3.4924% |
| `2+2+1+1` | 0.5014% |
| `2+1+1+1+1` | 0.0020% |
| `1+1+1+1+1+1` | 0% |

These values describe one fitted historical endpoint. New draws or a different
border can change every feature, count, posterior value, decoded marginal, and
rank.

## Signature-forecast statistics

The Space Groups analysis begins reported model evaluation after 100 prior
draws. For actual signature \(Y_t\) and posterior vector \(\mathbf p_t\):

```{math}
\operatorname{LogLoss}_t
=-\log(\max(p_{t,Y_t},10^{-15})),
```

```{math}
\operatorname{Brier}_t
=\sum_s(p_{t,s}-\mathbf1[s=Y_t])^2.
```

On the repository's 671 post-warm-up evaluations at border 7:

| Forecast | Log loss | Brier | Signature accuracy | Group-count accuracy | Group-count MAE |
|---|---:|---:|---:|---:|---:|
| **Border Group Bayesian** | **2.140277** | **0.889186** | **20.2683%** | **44.7094%** | **0.634873** |
| Border Group Statistical | 1.928158 | 0.835644 | 19.9702% | 41.2817% | 0.663189 |
| Exact random 6/49 null | 1.913997 | 0.833116 | 21.3115% | 33.9791% | 0.770492 |

The Bayesian model improves group-count accuracy and MAE over the statistical
and exact-null rows, and its exact-signature accuracy is slightly above the
statistical component. It is worse on log loss and Brier score than both
references and trails the exact null on exact-signature accuracy. This pattern
is consistent with a model that sometimes identifies the broader group-count
class while assigning probabilities too aggressively within the 11-state
distribution.

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
| Full replay | 770 | 592 | 0.768831 | 565.714 |
| Validation, target draws 121–520 | 400 | 317 | 0.792500 | 293.878 |
| Holdout, target draws 521–770 | 250 | 171 | 0.684000 | 183.673 |

The latest 250-target comparison slice, target draws 522–771, records 172 hits
or 0.688000 per target.

The full and validation results exceed theoretical random expectation, but the
holdout and latest slice are below it. The validation pattern does not persist.
These retrospective measurements do not establish statistical significance,
calibration, stable future lift, or predictability.

## Core mathematical and statistical concepts

- **Circular gap composition:** six non-negative spaces summing to 43 encode a
  ticket without privileging the 49-to-1 boundary.
- **Connected components:** an inclusive border threshold creates circular
  groups separated by large spaces.
- **Integer partitions:** sorted group sizes map every draw to one of 11
  canonical states.
- **Categorical feature engineering:** signature, group count, maximum-space
  band, recent mode, and trend summarize the latest context.
- **Delayed supervised counting:** each context is paired only with the next
  completed signature.
- **Multinomial and categorical smoothing:** additive-one terms keep unseen
  feasible classes and feature values finite.
- **Naïve Bayes:** a class prior and five conditional likelihoods are multiplied
  under a conditional-independence approximation.
- **Log-domain normalization:** maximum shifting prevents numerical underflow
  without changing posterior ratios.
- **Conditional probability:** a manual group count restricts and renormalizes
  the state space.
- **Mixture decoding:** posterior, shape, and anchor weights form a distribution
  over valid tickets.
- **Marginalization:** valid-ticket weights are summed into normalized
  number-level mass.
- **Proper scoring rules:** log loss and Brier score evaluate the complete
  posterior, not only its largest state.
- **Hypergeometric overlap:** Top-6 evaluation uses the null overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Negative holdout:** the current decoded Top-6 holdout is below theoretical
  random expectation despite stronger full and validation results.
- **Weak probability calibration:** categorical log loss and Brier score trail
  the statistical and exact-random references in the historical replay.
- **Conditional-independence violation:** current signature and group count are
  deterministic relatives; recent mode and trend reuse overlapping history.
- **Potential overconfidence:** multiplying correlated likelihoods can create
  sharper posterior ratios than the evidence supports.
- **Cumulative counts:** old feature-target pairs never decay, so the model
  cannot adapt quickly to a genuine distribution shift.
- **Fixed category boundaries:** maximum-space buckets and the 0.2 trend
  threshold are engineering choices, not learned cut points.
- **Compressed response:** one sorted partition discards exact group order,
  spaces, anchors, and number identities before prediction.
- **Rare signatures:** very uncommon states have little conditional evidence;
  additive-one smoothing prevents zeros but does not create information.
- **Threshold sensitivity:** changing border space redefines every historical
  response and feature profile.
- **Manual-selection bias:** selecting a target group count after examining the
  same history adds an external, unmeasured selection step.
- **Decoder truncation:** only the latest 16 observed shapes per signature enter
  the shape beam, while anchor counts use lifetime history.
- **Constructed fallback shapes:** deterministic pseudoshapes ensure coverage
  but are not samples from the exact random ticket distribution.
- **Two-stage error:** even a useful signature posterior can lose information
  or acquire bias during number-level decoding.
- **Uncalibrated number mass:** decoded marginals and min–max scores are not
  occurrence probabilities.
- **Min–max information loss:** score scaling preserves order but removes the
  absolute concentration of number mass.
- **Tie-break influence:** current gap can order exact score ties despite not
  belonging to the Bayesian signature model.
- **Retrospective comparison:** feature choices, constants, and the broader
  strategy collection were developed with historical results available.
- **No guaranteed predictability:** apparent contextual associations can be
  random fluctuations and may disappear on untouched draws.

## Implementation map

The production group model is implemented in
`src/rand_ai/space_groups.py`:

- `spaces_for_numbers`, `profile_from_spaces`, and `profile_for_numbers` create
  circular spaces, groups, and canonical signatures;
- `exact_null_signature_counts` and `exact_null_probabilities` establish
  feasibility and the exact random reference;
- `SpaceGroupForecaster._bayes_features` constructs the five current context
  categories;
- `_space_bucket` and `_trend_category` define their fixed transformations;
- `SpaceGroupForecaster.observe` associates pending contexts with newly
  completed targets before remembering the current profile;
- `SpaceGroupForecaster._bayesian` calculates smoothed log scores and the
  normalized posterior;
- `condition_signature_probabilities` applies an optional manual group-count
  constraint;
- `_signature_candidates` and `_signature_marginals` build the valid-ticket
  decoder; and
- `number_scores` creates scaled number scores and user-facing details.

`src/rand_ai/strategy_prediction.py` owns the chronological application state,
invokes the shared forecaster, applies the standard score/gap/number ranking,
and serializes `border_group_bayesian` when requested.

The behavior is covered by `tests/test_space_groups.py` and
`tests/test_strategy_prediction.py`, including exact feasibility,
normalization, sparse and trending context behavior, delayed causal updates,
manual conditioning, valid decoded tickets, number-score fallback,
walk-forward metrics, strategy registration, and serialization.
