# Decision Tree Selector

## Introduction

**Decision Tree Selector** is the production strategy with identifier
`decision_tree_selector`. It is a default-disabled member of the
**Ensembles & Coverage** family. Rather than inventing a fourteenth ranking or
averaging several rankings, it estimates which one of 13 established expert
strategies is best suited to the next draw and copies that expert's complete
1–49 ranking.

The selector uses draw-shape context, expert agreement, ranking stability, and
completed walk-forward efficacy. Its first six ranked numbers form the Top-6
prediction consumed by grids, audits, effectiveness histories, comparisons,
portfolios, exports, and Possible Draw.

```{admonition} Selection, not blending
:class: important

The decision tree predicts an expected Top-6 hit count for each expert and
chooses one expert. It does not average number scores, construct a mixed ticket,
or alter the chosen expert's internal model.
```

## Scope

The strategy answers a conditional routing question:

> Given information available after the latest completed draw, which expert's
> current ranking should be used for the next target draw?

Selecting only Decision Tree Selector activates all 13 experts as hidden
dependencies. Only the requested selector is serialized and displayed. The
experts retain their own production calculations, and selecting the router does
not enable their individual output cards.

The strategy is disabled by default. It should be interpreted as an auditable,
retrospective model-selection experiment, not as proof that draw outcomes are
predictable.

## Expert pool

The pool is fixed and ordered. Its order is also the deterministic tie-break
when two experts receive exactly the same estimate.

| Stable order | Strategy ID | Displayed selector label |
|---:|---|---|
| 1 | `freshness` | Freshness |
| 2 | `proximity` | Proximity |
| 3 | `emd` | EMD |
| 4 | `entropy` | Entropy |
| 5 | `markov100` | 100 Markov |
| 6 | `mkfr` | Markov Frequency |
| 7 | `mksp` | Markov Spaces |
| 8 | `bayesian` | Bayesian |
| 9 | `predictive_grid` | Predictive Grid |
| 10 | `co_occurrence` | Co-occurrence |
| 11 | `mixed` | Mixed |
| 12 | `svc` | SVC |
| 13 | `tbl` | TBL |

Every expert must supply a complete permutation of the numbers 1 through 49.
The selector evaluates the expert's first six ranks against the completed
target and later uses the complete selected ranking as its output.

## Prediction target

Let (R_{t,j}) be expert (j)'s ranking prepared for target draw (t), and
let (D_t) be the six numbers observed in that draw. The training response for
expert (j) is its completed Top-6 hit count:

\[
y_{t,j}=\left|\operatorname{Top6}(R_{t,j})\cap D_t\right|,
\qquad y_{t,j}\in\{0,1,\ldots,6\}.
\]

One completed draw therefore contributes one 88-component feature row and a
13-component response vector

\[
\mathbf y_t=(y_{t,1},\ldots,y_{t,13}).
\]

This is multi-output regression. The tree minimizes squared error across the
13 numerical hit targets; it is not a classification tree and does not learn a
probability that an expert will win.

## Feature vector

The current predictor has 88 components:

\[
10\ \text{global draw features}
+13\ \text{experts}\times6\ \text{expert features}=88.
\]

All components are computed from completed history and current expert rankings.
No value from the target draw is present when its prediction is made.

### Global draw features

Let (N) be the number of remembered draws, let the latest completed draw be
(L=\{l_1<\cdots<l_6\}), and let (G_n) be the current zero-based gap for
number (n). The implementation uses population mean and variance for the 49
gap values.

| Feature | Exact production definition | Interpretation |
|---|---|---|
| `history_evidence` | \(\min(N/500,1)\) | Amount of accumulated history, saturated at 500 draws |
| `latest_sum` | \(\sum_{l\in L}l/(6\cdot49)\) | Normalized level of the latest draw |
| `latest_span` | \((l_6-l_1)/48\), or 0 with fewer than two values | Normalized range |
| `latest_odd_share` | Number of odd values in (L), divided by \(\max(|L|,1)\) | Odd/even shape |
| `latest_low_share` | Number of values \(\leq24\), divided by \(\max(|L|,1)\) | Low/high shape |
| `latest_prime_share` | Number of primes from 2 through 47 in (L), divided by \(\max(|L|,1)\) | Prime composition |
| `latest_consecutive_share` | Count of adjacent sorted differences equal to 1, divided by \(\max(|L|-1,1)\) | Consecutive-number density |
| `previous_overlap` | \(|L\cap L_{-1}|/6\) | Overlap of the latest two completed draws |
| `gap_mean` | \(\operatorname{clip}(\operatorname{mean}(G_1,\ldots,G_{49})/40,0,1)\) | Overall recurrence-gap level |
| `gap_deviation` | \(\operatorname{clip}(\sqrt{\operatorname{Var}(G_1,\ldots,G_{49})}/40,0,1)\) | Dispersion of current gaps |

The division guards make empty or very short histories well-defined. Before a
normal six-number history exists, missing shape information becomes zero rather
than generating an invalid value.

### Per-expert features

For each expert (j), let (h_{i,j}) be its completed Top-6 hit count and let
(E) be the number of evaluated target draws. The random-reference mean is

\[
\mu_0=\frac{6\cdot6}{49}=\frac{36}{49}.
\]

The lifetime estimate uses a neutral 24-draw prior:

\[
q_j=\frac{\sum_{i=1}^{E}h_{i,j}+24\mu_0}{E+24}.
\]

Each expert contributes the following six features.

| Feature | Exact production definition | Interpretation |
|---|---|---|
| `smoothed_lifetime_hits` | \(q_j/6\) | Prior-smoothed completed efficacy on a 0–1 scale |
| `recent_10_hits` | Mean of the latest at most 10 completed hits, divided by 6 | Short completed-history efficacy |
| `recent_40_hits` | Mean of the latest at most 40 completed hits, divided by 6 | Broader recent efficacy |
| `recent_momentum` | \((\overline h_{10,j}-\overline h_{40,j})/6\) | Short-minus-broad efficacy change |
| `top_six_consensus` | For the expert's six candidates, mean fraction of the 13 expert Top-6 sets containing each candidate | Current agreement with the pool |
| `top_six_stability` | Top-6 overlap with that expert's preceding pending ranking, divided by 6; initially 0 | Ranking persistence from one target to the next |

When no completed efficacy exists, both recent means use \(\mu_0\). Consensus
is computed from the 13 current causal rankings. Stability compares those
rankings with rankings that were stored for the preceding target.

## Decision-tree mathematics

After warm-up, the estimator is a scikit-learn multi-output
`DecisionTreeRegressor` with fixed configuration:

| Setting | Production value |
|---|---:|
| Split criterion | Squared error |
| Maximum depth | 4 |
| Minimum samples per leaf | 12 completed draws |
| Training window | Latest 500 completed feature/target rows |
| Minimum rows before fitting | 72 |
| Random state | 20260626 |

At a non-leaf node, one feature (x_k) and threshold (a) divide the current
training rows into (x_k\leq a) and (x_k>a). The squared-error criterion
chooses splits that reduce within-node response variance. A terminal leaf stores
the mean 13-component response among its training rows. For a current feature
vector \(\mathbf x_t\), the fitted tree returns

\[
\widehat{\mathbf y}_t=f(\mathbf x_t)
=(\widehat y_{t,1},\ldots,\widehat y_{t,13}).
\]

Each estimate is clipped to the feasible interval \([0,6]\). The selected
expert is

\[
j^*=\arg\max_j\widehat y_{t,j},
\]

with the stable expert order resolving exact ties. The second-ranked estimate
is retained as the displayed runner-up. Once at least 72 completed examples
exist, the tree is refitted after each newly evaluated draw using up to the
latest 500 examples.

```{admonition} What an estimated hit count means
:class: note

The output of a leaf is a conditional sample mean learned under the tree's
partitions. It is not a calibrated forecast, confidence interval, hypothesis
test, or guarantee that the selected expert will obtain that many hits.
```

## Warm-up fallback

Before 72 completed feature/target pairs exist, no decision tree is fitted.
The selector instead ranks experts directly by the smoothed lifetime estimate
(q_j). At absolute cold start, all experts equal the random-reference mean, so
the stable tie order selects Freshness. Completed results can change the
fallback selection before the tree becomes eligible.

This fallback prevents the tree from fitting extremely small histories while
preserving a causal prediction for every target.

## Scoring and ranking

After expert (j^*) is selected, every number receives rank strength from that
expert's complete ranking:

\[
s_t(n)=\frac{49-r_{t,j^*}(n)}{48},
\]

where (r=1) gives strength 1 and (r=49) gives strength 0. Because all 49
ranks are unique, the application's normal score tie-break—larger current gap,
then smaller number—cannot reorder them. Decision Tree Selector therefore
reproduces the selected expert's ranking bit for bit.

The first six numbers become the selector's Top-6. The numeric score shown for
a number is its position-derived rank strength, not the selected expert's raw
score and not a probability of being drawn.

## Causal lifecycle

The sequence for target draw (t) is:

1. After draw (t-1), build all 13 expert rankings and the 88 current features.
2. Store those features and rankings as the pending forecast for (t).
3. When draw (t) occurs, compare it with each pending expert Top-6 and append
   one completed multi-output training row.
4. Update lifetime and recent hit histories, then fit or refit the tree when at
   least 72 training rows exist.
5. Build new causal expert rankings and features for target (t+1), select its
   expert, and store that forecast as the next pending row.

The result of draw (t) can train forecasts only from (t+1) onward. It
cannot alter the expert chosen for its own already-issued prediction. Appending
later draws therefore leaves every earlier prefix prediction unchanged.

## Interpreting prediction details

Each number carries the same selector context plus its rank in the selected
expert:

- **Selected expert** identifies the ranking copied for the target.
- **Decision tree fitted on … completed draws** or **Smoothed efficacy
  fallback (…/72 training draws)** identifies the operating mode.
- **Estimated hits** reports the selected expert's fitted or fallback estimate.
- **Runner-up** reports the next expert and its estimate.
- **Tree path** lists the feature comparisons traversed by the current row;
  warm-up instead reports that no fitted path exists.
- **Selected-expert rank** gives that number's position in the copied ranking.

Tree-path thresholds are displayed to three decimal places for readability.
They describe the route through this fitted tree; they should not be interpreted
as universal domain boundaries.

## Statistical reference and efficacy

For a fixed six-number prediction evaluated against a uniformly random
six-number draw from 49, the Top-6 hit count follows

\[
H\sim\operatorname{Hypergeometric}(49,6,6).
\]

Its mean and variance are

\[
\mathbb E[H]=\frac{36}{49}=0.734694,
\qquad
\operatorname{Var}(H)=0.577572.
\]

A leakage-free replay over the repository's 771 chronological YAML draws
produces 770 target forecasts with the current implementation:

| Slice | Targets | Total Top-6 hits | Mean hits per target | Random expected total |
|---|---:|---:|---:|---:|
| Full replay | 770 | 604 | 0.784416 | 565.714 |
| Validation, target draws 121–520 | 400 | 314 | 0.785000 | 293.878 |
| Holdout, target draws 521–770 | 250 | 199 | 0.796000 | 183.673 |

These figures are retrospective acceptance measurements for the present code
and dataset. The feature set, expert pool, tree settings, and broader strategy
collection were developed while historical data were available. No statistical
significance, stable future advantage, or guaranteed predictability is claimed.

## Core statistical concepts

- **Multi-output regression:** one feature row predicts 13 expert hit means at
  once, allowing shared tree partitions but retaining separate leaf outputs.
- **Squared-error partitioning:** splits seek regions whose historical response
  vectors are more homogeneous; they do not establish causal regimes.
- **Shrinkage:** the 24-draw null prior pulls sparse lifetime expert estimates
  toward \(36/49\).
- **Rolling estimation:** fitted targets are limited to 500 rows, while the
  lifetime fallback feature retains all completed hit totals.
- **Recent summaries:** 10- and 40-draw means expose different horizons, and
  their difference supplies a simple momentum variable.
- **Consensus and stability:** cross-expert agreement and temporal Top-6 overlap
  describe rankings without mixing their scores.
- **Deterministic routing:** fixed seed, stable expert order, causal inputs, and
  exact copied ranks make the replay reproducible.

## Limitations and responsible interpretation

- **Retrospective design:** the expert pool, features, warm-up, and tree
  hyperparameters were chosen in the context of available historical data.
- **Multiple-strategy selection:** choosing among many experts can capitalize on
  historical noise even when all candidates are individually causal.
- **Correlated experts:** several experts use overlapping history signals, so 13
  outputs do not represent 13 independent sources of evidence.
- **Small conditional samples:** depth-four leaves with a minimum of 12 rows can
  still provide noisy 13-dimensional means.
- **Abrupt routing:** a small feature change near a threshold can switch the
  selected expert and the entire ranking.
- **No model averaging:** the runner-up contributes nothing, even when its
  estimate is almost identical to the winner's.
- **Limited extrapolation:** a regression-tree leaf returns historical means and
  cannot express a smooth trend outside observed partitions.
- **Recent-window sensitivity:** 10- and 40-draw summaries may react to random
  streaks rather than persistent behavior.
- **Uncalibrated estimates:** predicted hits are bounded numerical estimates,
  not probabilities or uncertainty intervals.
- **Rank-strength information loss:** the selected expert's original score
  distances are discarded; only rank order survives.
- **Dependency sensitivity:** a future change to any source expert can change
  selector features, targets, choices, and replay results.
- **Dataset dependence:** efficacy varies with the history and evaluation slice.
- **No guaranteed predictability:** historical selection performance can
  disappear or reverse on untouched future draws.

Use Decision Tree Selector to inspect whether a shallow causal router can choose
among stable rankings, not as evidence that past draw structure controls future
lottery outcomes.

## Implementation map

| Responsibility | Production location |
|---|---|
| Constants, expert order, labels, dependencies, and 88 feature names | `src/rand_ai/strategy_prediction.py`, `_DECISION_TREE_*` constants and `_STRATEGY_DEPENDENCIES` |
| State, fixed regressor construction, rolling feature/target buffers, totals, and recent histories | `src/rand_ai/strategy_prediction.py`, `_StrategyState.__init__` |
| Completed-hit targets and causal refitting | `src/rand_ai/strategy_prediction.py`, `_train_decision_tree_selector` |
| Neutral-prior fallback estimate | `src/rand_ai/strategy_prediction.py`, `_decision_tree_smoothed_hits` |
| Global, efficacy, consensus, and stability features | `src/rand_ai/strategy_prediction.py`, `_decision_tree_selector_features` |
| Path explanation, expert selection, rank strength, and details | `src/rand_ai/strategy_prediction.py`, `_decision_tree_selector_path` and `_decision_tree_selector_scores` |
| Learn-before-remember lifecycle and final ranking assembly | `src/rand_ai/strategy_prediction.py`, `_StrategyState.train`, `_StrategyState.remember`, and `_StrategyState.build_strategies` |
| Desktop registration, default-disabled state, and payload serialization | `web/electron/main.cjs` and `src/rand_ai/gui_bridge.py` |
| Settings description, family, color, and displayed details | `web/src/components/SettingsDialog.vue`, `web/src/lib/strategyFamilies.ts`, `web/src/lib/strategyColors.ts`, and `web/src/views/CombinedPredictionGridView.vue` |
| Hidden dependency, fitted-tree, path, and target-leakage tests | `tests/test_strategy_prediction.py` |
| Default-disabled bridge test | `tests/test_gui_bridge.py` |
| Replay dataset | `data/lotto_results_2019.yaml` |

The desktop payload contains all 49 rank-strength scores, ranks, current gaps,
detail strings, the Top-6 numbers, and standard completed efficacy. It does not
expose the hidden experts as separately requested strategies.
