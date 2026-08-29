(border-group-ml)=
# Border Group ML

## Introduction

**Border Group ML** is the production strategy with identifier
`border_group_ml`. It is a default-enabled member of the **Border Space
Groups** family.

The strategy converts every completed six-number draw into a circular group
signature, represents the latest history with an exact 82-value feature
vector, and trains an averaged online logistic classifier to forecast the next
signature. A shared valid-ticket decoder then maps that 11-category forecast
into a complete ranking of numbers 1–49.

```{admonition} A signature classifier, not a number classifier
:class: important

The supervised target is the next draw's **circular group signature**. The
model does not classify the 49 numbers independently. Number scores arise only
after the predicted signature distribution is mixed through historical space
shapes and anchors.
```

## Scope and role

Border Group ML answers three linked questions:

1. What recent circular signatures, spaces, and group-count patterns describe
   the completed history?
2. Which of the 11 canonical signatures does an online linear classifier
   favor for the next draw?
3. When that signature distribution is decoded through valid ticket shapes,
   which individual numbers receive the most marginal mass?

Its first six number ranks form the Top-6 used by prediction grids, audits,
effectiveness histories, comparisons, portfolios, exports, and Possible Draw.

The engine is distinct from:

- **Border Group Statistical**, which estimates only cumulative signature
  frequencies;
- **Border Group Markov**, which conditions a transition row on the latest
  signature;
- **Border Group Bayesian**, which uses five discrete context fields and a
  naïve conditional-independence approximation; and
- **Border Group Hybrid**, which includes this ML forecast as one of four
  log-loss-weighted component distributions.

Selecting only Border Group ML creates the shared `SpaceGroupForecaster`, but
only `border_group_ml` is serialized for display. It does not activate another
selectable strategy as a hidden dependency.

## Circular spaces

For a sorted draw

```{math}
1\le n_1<n_2<\cdots<n_6\le49,
```

production constructs six counts of empty numbers:

```{math}
s_0=(n_1-1)+(49-n_6),
```

```{math}
s_i=n_{i+1}-n_i-1,
\qquad i=1,\ldots,5.
```

The first space crosses the 49-to-1 boundary. Every space is non-negative and

```{math}
\sum_{i=0}^{5}s_i=49-6=43.
```

This circular representation avoids treating the edge between 49 and 1 as
structurally different from the other five intervals.

## Border threshold and connected groups

The application-wide **Border space** setting is an integer (b\) from 0
through 43; its default is 7. The threshold is inclusive:

- (s_i\le b\) connects the number positions on its two sides;
- (s_i>b\) separates two groups.

Groups are maximal connected runs around the six-position circle. If no space
is a separator, all six numbers form one group. Otherwise, the number of
groups equals the number of separators.

Changing (b\) reclassifies the entire draw history and therefore changes the
ML labels, lag features, window features, feasible classes, decoder beams,
metrics, and final number ranking. The classifier is rebuilt for the selected
border; it is not transferred between border values.

## Canonical signature classes

The circular group sizes are sorted from largest to smallest. Sorting removes
traversal and rotation ambiguity and maps every draw to one of the 11 integer
partitions of six:

| Class index | Signature | Group count |
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

Let (Z_t\in\{0,\ldots,10\}) be the signature class for completed draw
(t\), let (G_t\) be its group count, and let (M_t=\max_i s_{t,i}\) be its
maximum circular space.

When equal largest separators provide several traversal anchors, production
chooses the separator followed by the smallest number. The sorted signature
does not depend on that choice, but the deterministic convention keeps group
and decoder representations stable.

## Exact feasibility mask

For the selected border, production enumerates all 64 separator masks and
counts rooted non-negative six-space compositions summing to 43. A signature
is feasible only when its exact count is positive.

The classifier is always initialized with all 11 class identifiers, but after
prediction it sets infeasible-class probability to zero and renormalizes the
remaining values. At border 7, six singleton groups are impossible because
six separator spaces would each require at least 8 empty numbers, for a
minimum total of 48 rather than 43. Class 10 therefore receives zero final
probability at the default border.

The feasibility mask is a structural constraint. It is not learned from the
historical sample.

## Exact 82-value feature vector

After completed draw (t\), production builds

```{math}
\mathbf x_t\in\mathbb R^{82}
```

to predict (Z_{t+1}\). Its values are appended in the following exact order.

### Signature and shape lags: 39 values

For each lag (\ell=1,2,3\), append:

1. an 11-value one-hot vector for signature (Z_{t-\ell+1}\);
2. normalized group count (G_{t-\ell+1}/6\); and
3. normalized maximum space (M_{t-\ell+1}/43\).

For class (k\), the one-hot value is

```{math}
x_{\ell,k}=\mathbf1[Z_{t-\ell+1}=k].
```

Each lag contributes (11+1+1=13) values, for (3\times13=39). When a lag
does not yet exist, its complete 13-value block is zero.

The group-count values lie in ([1/6,1]\) when present. Maximum-space values
lie in ([0,1]\).

### Latest circular spaces: 6 values

Append the six spaces of the latest completed draw in their production order:

```{math}
\left(
\frac{s_{t,0}}{43},
\frac{s_{t,1}}{43},
\ldots,
\frac{s_{t,5}}{43}
\right).
```

These six values retain exact normalized geometry that is lost when a draw is
compressed to its sorted signature. Their sum is always 1.

### Rolling signature windows: 36 values

For each window (w\in\{10,25,100\}), let

```{math}
L_w=\min(t,w)
```

and append 11 empirical signature proportions:

```{math}
F_{w,k}(t)=
\frac1{L_w}
\sum_{i=t-L_w+1}^{t}\mathbf1[Z_i=k],
\qquad k=0,\ldots,10.
```

Then append the normalized mean group count:

```{math}
\bar G_w(t)=
\frac1{6L_w}
\sum_{i=t-L_w+1}^{t}G_i.
```

Each window contributes 12 values, so the three windows contribute 36. Short
histories use every available completed profile; they do not read or pad
future draws.

### Group-count trend: 1 value

The trend uses at most the latest 25 profiles. With fewer than four profiles,
its category is stable:

```{math}
C_t=1.
```

Otherwise, the recent sequence is split at
(m=\lfloor L/2\rfloor\). Let (\mu_L\) be the mean group count in the first
(m\) profiles and (\mu_R\) the mean in the remaining profiles. Production
sets

```{math}
C_t=
\begin{cases}
2,&\mu_R-\mu_L>0.2,\\
0,&\mu_L-\mu_R>0.2,\\
1,&\text{otherwise}.
\end{cases}
```

The appended feature is (C_t/2\), giving 0 for decreasing, 0.5 for stable,
and 1 for increasing group count.

### Feature count summary

| Block | Width |
|---|---:|
| Three lag blocks | 39 |
| Latest six normalized spaces | 6 |
| Three rolling windows | 36 |
| Normalized trend category | 1 |
| **Total** | **82** |

No individual number identity, current number gap, Top-6 efficacy, other
strategy rank, draw date, or future value enters this feature vector.

## Online classifier

Production uses scikit-learn's `SGDClassifier` with the explicitly configured
parameters:

| Parameter | Value |
|---|---|
| Loss | `log_loss` |
| Penalty | `l2` |
| Regularization `alpha` | `0.001` |
| Parameter averaging | `True` |
| Random state | `0` |

The locked implementation also uses an intercept, no class weights, and the
library's `optimal` learning-rate schedule. Each delayed example is supplied
through one `partial_fit` call. On the first update, production declares all
classes (0,1,\ldots,10\); later updates reuse that class layout.

For class (k\), a linear decision function has the form

```{math}
a_k(\mathbf x)=\mathbf w_k^\mathsf T\mathbf x+b_k.
```

Logistic loss penalizes a class-specific margin through

```{math}
\ell(y,a)=\log(1+e^{-ya}),
\qquad y\in\{-1,+1\},
```

with L2 shrinkage controlled by `alpha=0.001`. Scikit-learn handles the
11-class reduction and `predict_proba` normalization; production does not
implement a separate custom softmax or closed-form fit.

With parameter averaging enabled, the prediction coefficients are averages
of the sequential SGD states rather than only the last noisy update. This can
stabilize online estimates, but it does not remove feature correlation,
class imbalance, or probability overconfidence.

## Delayed target training

The feature vector built after draw (t) is retained as pending state. It is
not trained immediately because its target (Z_{t+1}\) is still unknown.

When draw (t+1) occurs, production pairs the saved feature vector with the
new signature and performs exactly one online update:

```{math}
(\mathbf x_t,Z_{t+1}).
```

Only then is a new vector (\mathbf x_{t+1}\) built for the following target.
This delayed association makes the classifier genuinely next-step rather than
an in-sample classifier of the latest draw.

## Warm-up and statistical fallback

The ML strategy requires 50 completed delayed feature–target pairs before it
uses `predict_proba`. Before that threshold, it returns the Border Group
Statistical distribution:

```{math}
P_{\mathrm{fallback},t+1}(k)=
\frac{C_k(t)+1}{\sum_{u\in\mathcal F}(C_u(t)+1)},
\qquad k\in\mathcal F,
```

where (C_k(t)\) is the cumulative signature count and (\mathcal F\) is the
set of feasible signatures. Infeasible classes receive zero.

There is no delayed pair after the first completed draw. Observing draw 2
trains the first row; observing draw 51 trains the 50th. Consequently, the
forecast for target draw 52 is the first one that can use the ML probability
vector.

With no completed profile, every Border Group engine returns the same uniform
distribution over feasible signatures. That cold-start distribution differs
from the unequal exact 6/49 signature null.

## Probability masking and optional conditioning

After warm-up, let (\widetilde P_{t+1}(k)\) be the classifier-provided class
probability. Production first masks structurally infeasible signatures and
normalizes:

```{math}
P_{t+1}(k)=
\frac{\mathbf1[k\in\mathcal F]\widetilde P_{t+1}(k)}
{\sum_{u\in\mathcal F}\widetilde P_{t+1}(u)}.
```

**Predicted groups** is **Automatic** by default. If the user selects a
feasible exact group count (g\), the distribution is further conditioned:

```{math}
P_{t+1}(k\mid |S_k|=g)=
\begin{cases}
\dfrac{P_{t+1}(k)}{\sum_{u:|S_u|=g}P_{t+1}(u)},&|S_k|=g,\\[8pt]
0,&|S_k|\ne g.
\end{cases}
```

Here (|S_k|\) is the number of parts in signature (S_k\). Impossible
border/count combinations are rejected. Manual conditioning is a user-imposed
constraint, not a group-count prediction learned by the classifier.

## Shared valid-ticket decoder

A signature distribution does not yet rank numbers. For each signature, the
shared decoder builds a leakage-safe candidate beam from:

- up to the latest 16 observed six-space shapes carrying that signature;
- deterministic valid fallback shapes, each adding one pseudocount; and
- lifetime first-number anchor counts for that signature, with additive-one
  smoothing.

### Shape weights

For signature (k\) and space shape (q\), let (c_{k,q}\) include recent
occurrences and fallback pseudocounts. Its within-signature weight is

```{math}
D_k(q)=\frac{c_{k,q}}{\sum_vc_{k,v}}.
```

### Anchor weights

For (q=(q_0,\ldots,q_5)\), valid anchor offsets are

```{math}
a\in\{0,\ldots,q_0\}.
```

If (A_k(a)\) is the lifetime anchor count for signature (k\), production
uses

```{math}
D_k(a\mid q)=
\frac{A_k(a)+1}{\sum_{r=0}^{q_0}(A_k(r)+1)}.
```

The ticket is reconstructed as

```{math}
n_1=a+1,
\qquad
n_{i+1}=n_i+q_i+1,
\quad i=1,\ldots,5.
```

Every candidate ticket is sorted, contains six distinct numbers from 1–49,
and reproduces a valid space composition summing to 43. Weights from duplicate
shape/anchor paths are added.

## Number marginals and ranking

Let (D_k(T)\) be the within-signature decoded ticket distribution. The
number-level mass is

```{math}
M_t(n)=
\sum_kP_{t+1}(k)
\sum_{T:n\in T}\frac{D_k(T)}6.
```

The factor (1/6) gives

```{math}
\sum_{n=1}^{49}M_t(n)=1.
```

This is normalized number mass, not a calibrated occurrence probability. The
corresponding inclusion marginal under the decoder mixture is (6M_t(n)\).

Production min–max scales all 49 masses:

```{math}
S_t(n)=
\begin{cases}
\dfrac{M_t(n)-M_{\min}}{M_{\max}-M_{\min}},
&M_{\max}>M_{\min},\\[8pt]
0,&M_{\max}=M_{\min}.
\end{cases}
```

If no decoded mass exists, (1/49\) is first assigned to every number; the
equal-value branch then produces all-zero scores.

Numbers are ranked by:

1. larger scaled score;
2. larger current gap; then
3. smaller number.

The Top-6 consists of the first six individual-number marginals. It need not
be one of the valid tickets in the decoder beam.

## Causal lifecycle and leakage protection

For completed draw (t), production follows this order:

1. evaluate the pending forecast for draw (t) before changing state;
2. derive the actual signature (Z_t\);
3. train the classifier on the feature vector saved after (t-1), with
   target (Z_t\);
4. update cumulative signature and transition counts, recent per-signature
   shape buffers, and lifetime anchor counts with draw (t);
5. append the completed profile;
6. build (\mathbf x_t\) from history only through draw (t\);
7. issue the fallback or ML signature distribution for target (t+1\), then
   apply feasibility and optional group-count conditioning; and
8. decode that distribution into the next 49-number ranking.

The target being forecast cannot enter its own training row, rolling windows,
shape beam, anchor counts, or ranking. It can affect only later targets.

Appending later draws leaves every earlier prediction unchanged. Identical
history prefixes, border setting, manual group constraint, locked dependency
version, and random state produce identical forecasts.

## Interpreting application fields

Every number exposes four shared Border Group details:

- **Border space** reports the inclusive connectivity threshold.
- **Model-selected group count** means automatic signature forecasting;
  **Manual target … groups** identifies explicit user conditioning.
- **Decoded marginal** is the unscaled normalized number mass (M_t(n)\).
- **Leading signatures** lists the three largest ML signature probabilities;
  canonical class index resolves exact ties.

The standard strategy payload separately contains min–max score, current gap,
rank, and Top-6 membership.

```{admonition} Three different quantities
:class: warning

Leading-signature percentages are probabilities over group shapes. Decoded
marginals are normalized number mass. The displayed strategy score is a
min–max transformation of those marginals. None should be read as a calibrated
probability that an individual number will be drawn.
```

The per-number details do not expose the 82 feature values or learned
coefficients. The Space Groups analysis supplies the complete current
signature distribution and walk-forward categorical metrics.

## Endpoint diagnostic

After all 771 repository draws at border 7 with automatic group count, the
classifier has completed 770 delayed feature–target updates. The latest profile
has:

| Field | Endpoint value |
|---|---|
| Signature | `5+1` |
| Circular spaces | `(28, 10, 2, 1, 0, 2)` |
| Group count | 2 |
| Maximum space | 28 |
| Feature width | 82 |

Its next-signature distribution is:

| Signature | ML probability |
|---|---:|
| `6` | 24.8626% |
| `5+1` | 43.7757% |
| `4+2` | 26.7906% |
| `4+1+1` | 1.5563% |
| `3+3` | 0.0032% |
| `3+2+1` | 0.0079% |
| `3+1+1+1` | 0.3792% |
| `2+2+2` | 1.7975% |
| `2+2+1+1` | 0.4914% |
| `2+1+1+1+1` | 0.3355% |
| `1+1+1+1+1+1` | 0% |

The decoded forecast for target draw 772 begins:

| Rank | Number | Scaled score | Current gap | Decoded marginal |
|---:|---:|---:|---:|---:|
| 1 | 4 | 1.000000 | 4 | 2.95% |
| 2 | 5 | 0.952129 | 7 | 2.84% |
| 3 | 6 | 0.877734 | 11 | 2.68% |
| 4 | 3 | 0.804820 | 2 | 2.53% |
| 5 | 7 | 0.773933 | 6 | 2.46% |
| 6 | 1 | 0.758169 | 4 | 2.42% |

These fitted-state values are reproducibility diagnostics. They do not imply
that the three leading signatures or six leading numbers have a future
advantage.

## Signature-forecast statistics

The Space Groups analysis starts reported evaluation only when a pending
forecast was made with at least 100 prior profiles. For actual class (Y_t\)
and probability vector (\mathbf p_t\), it reports:

```{math}
\operatorname{LogLoss}_t
=-\log(\max(p_{t,Y_t},10^{-15})),
```

```{math}
\operatorname{Brier}_t
=\sum_k(p_{t,k}-\mathbf1[k=Y_t])^2.
```

Exact-signature accuracy uses the maximum-probability class. Group-count
accuracy compares the number of parts in predicted and actual signatures;
group-count MAE is their absolute difference.

On the repository's 671 post-warm-up evaluations at border 7:

| Forecast | Log loss | Brier | Signature accuracy | Group-count accuracy | Group-count MAE |
|---|---:|---:|---:|---:|---:|
| **Border Group ML** | **6.315110** | **1.321350** | **19.2250%** | **42.7720%** | **0.676602** |
| Border Group Statistical | 1.928158 | 0.835644 | 19.9702% | 41.2817% | 0.663189 |
| Exact random 6/49 null | 1.913997 | 0.833116 | 21.3115% | 33.9791% | 0.770492 |

The ML log-loss 95% normal interval is approximately
([5.891844,6.738375]\). Its group-count accuracy is higher than the statistical
and exact-null rows, but its log loss, Brier score, exact-signature accuracy,
and group-count MAE trail the statistical model. The very high log loss is
consistent with some confidently wrong signature forecasts.

These metrics score the 11-category forecast before reducing it to Top-6
number hits. A component can behave differently under the two evaluations.

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
the repository's 771 chronological YAML draws creates 770 forecast targets:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 551 | 0.715584 | 565.714 |
| Validation, target draws 121–520 | 400 | 286 | 0.715000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 181 | 0.724000 | 183.673 |

The latest 250-target slice, target draws 522–771, also records 181 hits or
0.724000 per target.

All three reported totals are below their theoretical random expectations.
The holdout is closer to expectation than the validation slice, but neither
the categorical metrics nor decoded Top-6 replay demonstrates stable
historical improvement. These are retrospective measurements and not proof of
statistical significance, inferiority on every future sample, or
predictability.

## Core mathematical and statistical concepts

- **Circular gap composition:** six spaces summing to 43 represent a ticket
  without privileging the 49-to-1 boundary.
- **Connected components:** an inclusive threshold turns circular neighbors
  into groups separated by larger spaces.
- **Integer partitions:** sorted group sizes define 11 canonical classes.
- **Feature engineering:** lagged one-hot states, normalized geometry, rolling
  categorical frequencies, group-count means, and trend form 82 inputs.
- **Delayed supervised learning:** each completed feature row is paired only
  with the following observed signature.
- **Online logistic classification:** stochastic-gradient updates optimize
  linear class margins under log loss and L2 regularization.
- **Parameter averaging:** sequential coefficient states are averaged to
  reduce update noise.
- **Structural masking:** exact gap-composition feasibility removes impossible
  output classes.
- **Conditional probability:** a manual group count renormalizes the forecast
  over only signatures with that number of parts.
- **Mixture decoding:** signature, shape, and anchor weights define mass over
  valid tickets.
- **Marginalization:** ticket mass is summed to derive each number's score
  source.
- **Proper scoring rules:** categorical log loss and Brier score evaluate the
  entire probability vector.
- **Hypergeometric overlap:** Top-6 efficacy uses the null overlap
  distribution for two six-element subsets of 49.

## Limitations and responsible interpretation

- **Negative historical efficacy:** full, validation, and holdout Top-6 totals
  are all below theoretical random expectation.
- **Weak categorical calibration:** log loss and Brier score substantially
  trail the statistical and exact-null references on this replay.
- **Linear decision surface:** interactions exist only to the extent they are
  encoded in the fixed features; the classifier does not learn nonlinear
  feature combinations.
- **Correlated features:** signature lags, group counts, spaces, rolling
  frequencies, and trend reuse overlapping information.
- **Class imbalance:** common signatures provide many more positive updates
  than rare signatures, and production applies no class weights.
- **One-example updates:** chronological `partial_fit` order matters, and old
  examples are not revisited as a batch optimization problem.
- **Averaging inertia:** averaged weights can stabilize noise but respond
  slowly to a genuine distribution change.
- **Fixed warm-up:** 50 rows is an engineering threshold, not an uncertainty
  test.
- **Library dependence:** optimizer and probability behavior depend on the
  locked scikit-learn implementation in addition to the explicit parameters.
- **Threshold sensitivity:** changing border space redefines all labels,
  features, feasibility, and results.
- **Manual-selection bias:** choosing a target group count after inspecting
  history adds an external selection step.
- **Compressed response:** the class target discards exact group order, number
  identities, spaces, and anchors.
- **Decoder truncation:** only the latest 16 observed shapes per signature enter
  the shape beam, while anchor counts use lifetime history.
- **Constructed fallback shapes:** deterministic pseudoshapes ensure coverage
  but are not samples from the exact random ticket distribution.
- **Two-stage error:** signature classification and ticket decoding can each
  introduce error or bias.
- **Uncalibrated number scores:** decoded mass and min–max score are not
  validated occurrence probabilities.
- **Tie-break influence:** current gap can decide exact marginal-score ties
  despite not being an ML input.
- **Retrospective comparison:** features, constants, settings, and evaluation
  slices were developed or inspected with historical data available.
- **No guaranteed predictability:** model flexibility cannot create a causal
  future signal when the underlying process supplies none.

Use Border Group ML as an auditable online experiment in forecasting circular
ticket structure, not as evidence that a more complex classifier is more
effective than simpler baselines.

## Implementation map

The production model is implemented in `src/rand_ai/space_groups.py`:

- module constants define the 11 signatures, border range, 50-row ML warm-up,
  and model identifiers;
- `spaces_for_numbers`, `profile_from_spaces`, and `profile_for_numbers` create
  circular spaces, connected groups, and canonical profiles;
- `exact_null_signature_counts` determines structural feasibility;
- `SpaceGroupForecaster.__init__` configures the averaged `SGDClassifier` and
  online state;
- `SpaceGroupForecaster._ml_features` builds the exact 82-value input vector;
- `_trend_category` implements the up-to-25-profile trend category;
- `SpaceGroupForecaster.observe` trains the pending vector before remembering
  the new profile;
- `SpaceGroupForecaster.forecast` applies warm-up fallback, classifier
  prediction, feasibility masking, and optional group-count conditioning;
- `_fallback_shapes`, `_signature_candidates`, and `_signature_marginals`
  construct the valid-ticket decoder; and
- `number_scores` creates decoded marginals, min–max scores, and detail text.

`src/rand_ai/strategy_prediction.py` owns the chronological strategy state,
invokes the shared forecaster, applies the score/gap/number tie-break, and
serializes `border_group_ml` when requested.

`src/rand_ai/statistics.py` builds the broader Space Groups forecast and model
metrics. Python bridge and Electron/Vue files register the default-enabled
strategy, family, color, display names, settings description, export fields,
and prediction details.

Tests in `tests/test_space_groups.py`, `tests/test_strategy_prediction.py`, and
`tests/test_gui_bridge.py` cover feature warm-up, delayed online updates,
normalization, feasibility, manual conditioning, valid-ticket decoding,
number-score fallback, metrics, complete ranking shape, selection, and
serialization. Frontend tests cover the **Border Space Groups** family and
color registration.
