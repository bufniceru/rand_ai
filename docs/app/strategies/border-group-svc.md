(border-group-svc)=
# Border Group SVC

## Introduction

**Border Group SVC** is the production strategy with identifier
`border_group_svc`. It is a separate, default-enabled member of the **Border
Space Groups** family. The existing `border_group_ml` online logistic model is
unchanged and remains independently selectable.

Border Group SVC uses the same leakage-safe 82-value feature vector as Border
Group ML, but fits a balanced radial-basis-function support-vector classifier
on a bounded rolling sample. It forecasts the next draw's complete border-group
signature. A shared valid-ticket decoder then converts that 11-category
distribution into a ranking of numbers 1–49.

```{admonition} Signature classifier, not six independent number classifiers
:class: important

The SVC target is one canonical partition of six, such as `5+1` or `3+2+1`.
Individual number scores are produced only after the signature probabilities
are decoded through valid circular-space shapes and historical anchors.
```

## Circular groups and target classes

For a sorted draw {math}`1\le n_1<\cdots<n_6\le49`, production calculates
six empty-number spaces:

```{math}
s_0=(n_1-1)+(49-n_6),
```

```{math}
s_i=n_{i+1}-n_i-1,\qquad i=1,\ldots,5.
```

The spaces are circular, non-negative, and sum to 43. For the selected Border
space {math}`b`:

- {math}`s_i\le b` connects the number positions on both sides;
- {math}`s_i>b` separates two groups.

Group sizes are sorted from largest to smallest, making the signature
independent of circular rotation. The 11 target classes, in production index
order, are:

| Class | Signature | Groups |
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

Changing Border space reclassifies the history, changes class feasibility, and
rebuilds the SVC training state. Models are not transferred between borders.

## Leakage-safe feature vector

After completed draw {math}`t`, the strategy creates the same exact feature vector
used by Border Group ML:

| Block | Width | Values |
|---|---:|---|
| Previous three profiles | 39 | Per lag: 11-value signature one-hot, group count divided by 6, maximum space divided by 43 |
| Current circular spaces | 6 | Six spaces divided by 43 |
| Windows 10, 25, and 100 | 36 | Per window: 11 signature frequencies and average group count divided by 6 |
| Recent group-count trend | 1 | Decreasing, stable, or increasing category divided by 2 |
| **Total** | **82** | |

Missing lag profiles use zeros. Every rolling window ends at the latest
completed draw. No value from target draw {math}`t+1` enters its own feature
vector.

## Delayed training lifecycle

The feature vector created after draw {math}`t` is retained as a pending
example. Only when draw {math}`t+1` becomes known is that vector paired with
the actual signature {math}`Z_{t+1}`. Production then:

1. scores the pending forecast against {math}`Z_{t+1}`;
2. appends the saved pre-draw feature vector and actual class to SVC training
   history;
3. updates the other Border Group histories with the completed draw; and
4. creates a new feature vector and forecast for draw {math}`t+2`.

This ordering is walk-forward. Modifying a later draw cannot alter the SVC
forecast issued at an earlier common prefix.

## Estimator configuration

The deterministic scikit-learn pipeline is:

1. `StandardScaler`; then
2. `SVC` configured with:

| Parameter | Value |
|---|---|
| Kernel | RBF |
| `C` | `1.0` |
| `gamma` | `"scale"` |
| Class weighting | `"balanced"` |
| Probability estimates | enabled |
| Random state | `0` |

Scaling is fitted only on the current causal training window. Balanced class
weights reduce the dominance of common signatures during fitting, but they do
not create observations for missing classes or guarantee calibrated
probabilities.

## Warm-up, rolling window, and refitting

The strategy uses a bounded schedule:

- fitting starts after 50 labeled pre-draw examples;
- at most the latest 500 labeled examples are retained;
- the first fit occurs at example 50; and
- a fresh scaler and SVC are fitted after each additional 25 examples.

This means scheduled fits occur at labeled counts 50, 75, 100, and so on. The
model is not incrementally updated and is not refitted after every draw.

Until warm-up is complete, the SVC forecast equals the smoothed Border Group
Statistical baseline. The same fallback is used whenever the current rolling
training window contains fewer than two observed signature classes. A
single-class scheduled fit is not attempted.

## Probability mapping and conditioning

`predict_proba` returns only the classes present in the fitted SVC. Production
maps those class indices back into the fixed 11-signature vector. Missing and
border-infeasible signatures receive zero, and the remaining probability mass
is normalized.

If **Predicted groups** specifies an exact feasible group count, the SVC vector
is then restricted to signatures having that number of parts and normalized
again. If the SVC assigned no mass to the requested subset, production uses a
uniform distribution across feasible signatures with the requested group
count. Impossible Border-space/group-count combinations are rejected.

## Number ranking

The shared Border Group decoder combines each signature probability with:

- up to the latest 16 observed circular-space shapes for that signature;
- deterministic valid fallback shapes; and
- smoothed historical first-number anchors.

Signature-ticket mass is marginalized over every number. The resulting number
mass is min–max scaled, with current gap and then number used for exact ranking
ties. Consequently, the Top-6 is a marginal ranking and does not have to equal
one decoded historical or generated ticket.

## Participation in Border Group Hybrid

Border Group Hybrid contains five components:

1. Statistical;
2. Markov;
3. Bayesian;
4. ML; and
5. SVC.

Before 30 evaluated forecasts are available for every component, each receives
exactly 20% weight. Later weights are derived from mean log loss over the
latest 100 evaluated forecasts, with a 5% minimum for every component.
Selecting only Border Group Hybrid still trains and uses its internal SVC;
the standalone `border_group_svc` strategy does not have to be enabled for
display.

## Metrics and interpretation

The Space Groups model collection reports SVC alongside the other Border Group
forecasts. Metrics begin after the existing 100-profile reporting warm-up and
include log loss, Brier score, exact-signature accuracy, group-count accuracy,
and group-count mean absolute error. Strategy-effectiveness views separately
measure the decoded Top-6 number ranking.

These retrospective metrics measure different stages. Better signature log
loss does not guarantee more Top-6 hits, and balanced SVC training does not
make random lottery outcomes predictable.

## Limitations

- The RBF kernel and fixed hyperparameters are engineering choices, not values
  established as optimal on unseen lottery data.
- Rare signatures may be absent from a 500-example window and then receive no
  direct SVC probability.
- Probability estimation adds fitting cost and does not ensure calibration.
- Refitting every 25 labels deliberately leaves the fitted decision boundary
  unchanged between scheduled updates.
- All 82 inputs derive from the same signature and space history and are
  strongly dependent.
- Border threshold selection can materially change both labels and features.
- The decoder is a second modeling stage; signature errors and decoding errors
  can compound.
- Historical evaluation and adaptive Hybrid weights do not establish stable
  future performance.

## Implementation map

Core behavior is implemented in `src/rand_ai/space_groups.py`:

- `_ml_features` constructs the shared 82-value causal vector;
- `_new_svc_model` creates the scaler and configured RBF SVC;
- `_fit_svc_if_due` enforces warm-up, class count, rolling window, and refit
  schedule;
- `observe` settles the pending label only after the actual signature is known;
- `forecast` maps fitted classes to all signatures, masks infeasible classes,
  applies optional group-count conditioning, and includes SVC in Hybrid; and
- `number_scores` performs shared signature-to-number decoding.

`src/rand_ai/strategy_prediction.py` registers and serializes
`border_group_svc`. Electron and renderer catalogs expose it in Settings,
prediction views, strategy families, and color assignments.

Regression coverage in `tests/test_space_groups.py` and
`tests/test_strategy_prediction.py` verifies configuration, deterministic
fitting, delayed labels, 50-example warm-up, 500-example limit, 25-example
refitting, single-class fallback, complete probability mapping, feasibility,
manual conditioning, Hybrid membership, and future-draw independence.
