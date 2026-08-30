# SRPH Minimax Regret Hybrid Benchmark

## Fixed protocol

- Dataset: `data/lotto_results_2019.yaml`, 771 draws.
- Evaluation: leakage-free walk-forward Top-6 forecasts for 770 targets.
- Sources: SRPH, Freshness, EMD, Bayesian, and Doublet/Triplet Markov.
- Actions: 503 mixtures on a 5% grid, with SRPH at least 50% and each
  residual source at most 20%.
- Adversary regimes: fixed, non-overlapping, completed 40-draw blocks.
- Warm-up: exact SRPH until four blocks, or 160 outcomes, are complete.
- Selector: minimize worst completed-block regret; break ties by cumulative
  completed-block hits, SRPH weight, then the frozen source order.

The current incomplete block never affects selection. Every counterfactual is
ranked before its target occurs and scored only after that target is observed.

## Replay result

| Slice | Strategy | Targets | Total hits | Mean hits |
|---|---|---:|---:|---:|
| Full | SRPH Minimax Regret Hybrid | 770 | **635** | **0.824675** |
| Full | SRPH | 770 | 643 | 0.835065 |
| Validation 121–520 | SRPH Minimax Regret Hybrid | 400 | **328** | **0.820000** |
| Validation 121–520 | SRPH | 400 | 328 | 0.820000 |
| Holdout 521–770 | SRPH Minimax Regret Hybrid | 250 | **197** | **0.788000** |
| Holdout 521–770 | SRPH | 250 | 205 | 0.820000 |

The theoretical random expectation is (36/49=0.734694) hits per target,
or 565.714 hits over 770 targets. This reference does not correct for model
selection or establish significance.

## Selected mixtures

Weights are ordered as SRPH / Freshness / EMD / Bayesian / Doublet–Triplet.

| Weights (%) | Predictions |
|---|---:|
| 100 / 0 / 0 / 0 / 0 | 160 |
| 95 / 0 / 5 / 0 / 0 | 40 |
| 85 / 5 / 10 / 0 / 0 | 120 |
| 85 / 0 / 5 / 0 / 10 | 120 |
| 70 / 5 / 10 / 0 / 15 | 120 |
| 70 / 5 / 5 / 0 / 20 | 210 |

The final target uses 70% SRPH, 5% Freshness, 5% EMD, 0% Bayesian, and
20% Doublet/Triplet Markov.

## Interpretation

SMR ties SRPH on the earlier validation slice but loses eight hits on both the
full replay and nominal holdout. The result demonstrates that minimizing
historical worst-block regret did not improve out-of-sample Top-6 efficacy.

SMR therefore remains default-disabled. Its grid, source pool, block size,
warm-up, objective, and tie order are frozen solely for future untouched-draw
evaluation. The replay is an acceptance regression, not evidence of lottery
predictability or future advantage.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
uv run python scripts/benchmark_srph_minimax_regret_hybrid.py
```
