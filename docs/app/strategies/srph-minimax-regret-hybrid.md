(srph-minimax-regret-hybrid)=
# SRPH Minimax Regret Hybrid

## Introduction

**SRPH Minimax Regret Hybrid** is the production strategy with identifier
`srph_minimax_regret_hybrid` and short engine name **SMR**. It is a
default-disabled experimental member of **Ensembles & Coverage**.

SMR applies a finite, deterministic minimax-regret game to five complete
rankings. It asks which guarded blend would have remained closest to the best
available blend in every completed historical block.

```{admonition} What the game represents
:class: important

The lottery is not modeled as an intelligent opponent. The player chooses one
frozen blend; the adversary chooses a completed historical block that exposes
that blend's largest regret.
```

SMR produces a complete ranking of numbers 1–49. Its first six numbers are the
Top-6 consumed by grids, audits, effectiveness histories, comparisons,
portfolios, exports, and Possible Draw.

## Scope

SMR is a shadow strategy for future untouched-draw evaluation. It does not
replace or train any source strategy, and it adds no predictive features of
its own. Its only learned state is the completed-block payoff matrix for a
fixed set of counterfactual blends.

The strategy intentionally has no:

- rolling or overlapping efficacy window;
- adaptive block length or candidate grid;
- prior smoothing;
- randomized Nash sampling;
- safety or promotion gate;
- agreement bonus or source quota beyond the frozen weight bounds; or
- use of the open block during selection.

## Hidden dependencies

Selecting only SMR activates:

```text
srph_minimax_regret_hybrid
├── svc_recurrence_proximity_hybrid
│   ├── svc_recurrence_hybrid
│   │   ├── recurrence_dynamics
│   │   └── svc
│   └── proximity
├── freshness
├── emd
├── bayesian
└── doublet_triplet_markov
```

Dependencies may calculate invisibly. Only SMR is serialized unless users
select the source strategies too.

## Players and actions

The blend sources, in stable order, are:

| Position | Source | Role |
|---:|---|---|
| 0 | SRPH | Guarded base score |
| 1 | Freshness | Residual rank strength |
| 2 | Earth Mover Distance | Residual rank strength |
| 3 | Bayesian | Residual rank strength |
| 4 | Doublet/Triplet Markov | Residual rank strength |

The player's pure actions are all weight vectors on a 5% grid satisfying:

\[
\sum_{j=0}^{4} w_j=1,
\qquad w_0\geq 0.50,
\qquad 0\leq w_j\leq0.20\quad(j=1,\ldots,4).
\]

Equivalently, weights use 20 integer units. SRPH receives at least 10 units
and each residual receives at most four. These constraints generate exactly
**503** candidate mixtures, including 100% SRPH.

## Rank strength and mixture score

For a residual source rank (r_j(n)\in\{1,\ldots,49\}\), rank strength is

\[
q_j(n)=\frac{49-r_j(n)}{48}.
\]

Rank 1 maps to 1, rank 49 maps to 0, and intermediate ranks are evenly
spaced. Let (S_{\mathrm{SRPH}}(n)\) be SRPH's exact score. Candidate mixture
(m\) scores number (n\) as

\[
S_m(n)=w_0S_{\mathrm{SRPH}}(n)+
\sum_{j=1}^{4}w_jq_j(n).
\]

No final min–max scaling is applied. Every mixture is ranked by descending
score, then larger current gap, then smaller number. The 100% SRPH action
copies SRPH's score dictionary and ranking directly.

## Completed-block payoff

Evaluated forecasts are partitioned chronologically into fixed,
non-overlapping blocks of 40 targets. For candidate mixture (m\) and completed
block (b\), define

\[
U_{m,b}=\sum_{t\in b}
\left|\operatorname{Top6}_{m,t}\cap D_t\right|,
\]

where (D_t\) is the six-number target draw. A block payoff can range from 0
to 240 hits.

The retrospective block oracle is

\[
O_b=\max_k U_{k,b}.
\]

It identifies the best of the same 503 frozen mixtures inside that completed
block. It does not inspect future or open-block outcomes.

## Minimax regret selector

Candidate regret in block (b\) is (O_b-U_{m,b}\). Its worst completed-block
regret is

\[
R_m=\max_b\left(O_b-U_{m,b}\right).
\]

Once four blocks are complete, SMR selects the candidate minimizing (R_m\).
Ties resolve by:

1. greater total hits across all completed blocks;
2. greater SRPH weight;
3. greater weights in stable order Freshness, EMD, Bayesian, then
   Doublet/Triplet Markov.

This is deterministic pure-action minimax regret. SMR does not calculate or
sample a randomized mixed equilibrium.

## Warm-up and block updates

The first 160 evaluated targets form four complete blocks. During this
warm-up, SMR reproduces SRPH scores and ranking exactly, while still replaying
all 503 counterfactuals.

The selected mixture can change only after a block closes. Outcomes in the
open block are tracked for diagnostics and future closure but never influence
the current selection.

## Causal lifecycle

For target (t\):

1. Source rankings are built from history through (t-1\).
2. SMR chooses weights using only previously closed blocks.
3. All 503 counterfactual Top-6 sets are built and saved.
4. SMR emits the selected mixture ranking.
5. After draw (t\) occurs, the saved Top-6 sets receive their hit counts.
6. If this is the fortieth result in the open block, the block closes.
7. Only the next prediction may use that completed block.

Therefore the current target cannot select its own weights. Appending later
draws does not change an earlier SMR forecast.

## Interpreting application details

Each number reports:

- warm-up fallback or active minimax status;
- all five effective weights;
- each source rank and source Top-6 membership;
- selected mixture's worst completed-block regret;
- its total counterfactual hits across closed blocks;
- completed block and closed-history counts; and
- open-block progress.

The displayed score is a relative blend score, not a probability. A zero
source weight means that source has no effect on the current score even though
its counterfactuals continue to be tracked.

Standard SMR efficacy measures the ranking actually emitted to users. The
503-action game matrix is separate internal selector evidence. Recurrence's
experimental evidence metadata is not copied; SMR serializes `evidence=None`.

## Statistical reference

For an unrelated six-number prediction and six-number draw from 49, Top-6
overlap follows

\[
H\sim\operatorname{Hypergeometric}(49,6,6),
\qquad \mathbb{E}[H]=\frac{36}{49}\approx0.734694.
\]

This null expectation does not account for trying many strategies, weight
grids, objectives, or reporting slices.

## Fixed replay

The canonical 771-draw dataset provides 770 evaluated targets:

| Slice | SMR hits | SRPH hits | SMR mean | SRPH mean |
|---|---:|---:|---:|---:|
| Full 770 | **635** | 643 | **0.824675** | 0.835065 |
| Validation 121–520 | **328** | 328 | **0.820000** | 0.820000 |
| Holdout 521–770 | **197** | 205 | **0.788000** | 0.820000 |

SMR ties SRPH on the earlier validation slice and loses eight hits on the
nominal holdout. It is not promoted.

The final forecast uses 70% SRPH, 5% Freshness, 5% EMD, 0% Bayesian, and 20%
Doublet/Triplet Markov. That endpoint is descriptive, not a recommendation to
reuse those weights outside the causal selector.

## Core concepts

- **Finite zero-sum game:** the player selects a blend and the adversary
  selects the most unfavorable completed block.
- **Regret:** loss relative to the best frozen blend within the same block.
- **Robust optimization:** selection controls the worst historical deficit
  instead of maximizing only lifetime hits.
- **Counterfactual replay:** every action is evaluated as the ranking it would
  actually have emitted.
- **Pure strategy:** one deterministic candidate blend is deployed.
- **Prefix invariance:** future records cannot alter past outputs.

## Limitations

- The source pool, weight bounds, 5% grid, 40-draw blocks, four-block warm-up,
  and tie hierarchy were designed retrospectively.
- Worst-block robustness did not transfer to the holdout.
- Forty draws provide noisy payoff estimates, while larger blocks would leave
  very few adversary regimes.
- The historical block oracle is not a realizable forward predictor.
- Direct Top-6 payoff is discontinuous: small score changes can replace a
  boundary number and change a block total.
- SRPH already contains SVC, Recurrence, and Proximity, so source information
  is not independent.
- Minimax regret can sacrifice average efficacy to reduce one historical
  worst case.
- Scores are uncalibrated and are not probabilities.
- No historical result establishes predictability or future advantage.

Use SMR only as a frozen shadow audit on future untouched draws.

## Implementation map

| Responsibility | Location |
|---|---|
| Frozen grid, state, counterfactual updates, regret selector, and details | `src/rand_ai/strategy_prediction.py` |
| Cache identity and serialized strategy payload | `src/rand_ai/gui_bridge.py` |
| Desktop registration and default-disabled state | `web/electron/main.cjs` |
| TypeScript identifiers, family, colors, and display names | `web/src/` |
| Causal, formula, selector, serialization, and regression tests | `tests/test_strategy_prediction.py`, `tests/test_gui_bridge.py` |
| Reproducible replay | `scripts/benchmark_srph_minimax_regret_hybrid.py` |
| Frozen result | `reports/srph_minimax_regret_hybrid_benchmark.md` |
