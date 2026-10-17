# Experimental LSTM study

Exploratory historical evidence only; this history previously influenced development.
No desktop registration, automatic installation, or claim of confirmed advantage.

Fixed 7 x 49 inputs; one 32-unit LSTM; five-seed probability ensemble.

| Dataset | Mean hits | 3+ rate | Brier | Log loss | Historical gate |
|---|---:|---:|---:|---:|---|
| historical | 0.716000 | 2.000% | 0.107504 | 0.372002 | False |
| synthetic-00 | 0.704000 | 1.600% | 0.107471 | 0.371846 | False |
| synthetic-01 | 0.796000 | 2.000% | 0.107452 | 0.371761 | False |
| synthetic-02 | 0.716000 | 2.400% | 0.107487 | 0.371923 | False |
| synthetic-03 | 0.792000 | 1.200% | 0.107429 | 0.371659 | False |
| synthetic-04 | 0.700000 | 1.200% | 0.107473 | 0.371856 | False |
| synthetic-05 | 0.760000 | 4.000% | 0.107459 | 0.371796 | False |
| synthetic-06 | 0.756000 | 2.800% | 0.107468 | 0.371835 | False |
| synthetic-07 | 0.752000 | 2.000% | 0.107458 | 0.371787 | False |
| synthetic-08 | 0.816000 | 4.000% | 0.107421 | 0.371620 | False |
| synthetic-09 | 0.648000 | 2.000% | 0.107629 | 0.372590 | False |
| synthetic-10 | 0.764000 | 2.400% | 0.107449 | 0.371747 | False |
| synthetic-11 | 0.784000 | 2.000% | 0.107475 | 0.371868 | False |
| synthetic-12 | 0.828000 | 2.800% | 0.107434 | 0.371679 | False |
| synthetic-13 | 0.868000 | 4.400% | 0.107453 | 0.371766 | False |
| synthetic-14 | 0.680000 | 0.400% | 0.107478 | 0.371882 | False |
| synthetic-15 | 0.812000 | 0.800% | 0.107438 | 0.371695 | False |
| synthetic-16 | 0.760000 | 2.800% | 0.107495 | 0.371967 | False |
| synthetic-17 | 0.748000 | 1.600% | 0.107458 | 0.371790 | False |
| synthetic-18 | 0.792000 | 2.000% | 0.107450 | 0.371755 | False |
| synthetic-19 | 0.816000 | 1.600% | 0.107434 | 0.371682 | False |

Synthetic controls completed: 20/20. Seeds are averaged within each history, not counted as independent evidence.

## Historical comparisons

| Baseline | Mean hits | LSTM difference | Paired 95% interval | Holm p |
|---|---:|---:|---|---:|
| random | 0.776000 | -0.060000 | [-0.24, 0.072] | 1.000000 |
| frequency | 0.700000 | 0.016000 | [-0.088, 0.1] | 1.000000 |
| cdm | 0.700000 | 0.016000 | [-0.088, 0.1] | 1.000000 |
| bayesian | 0.744000 | -0.028000 | [-0.136, 0.08] | 1.000000 |

Full metrics, fixed-bin reliability data, individual seed results, training logs,
split boundaries, raw forecasts and fingerprints are in adjacent JSON/JSONL/NPZ files.
CDM has ranking metrics only; its categorical shares are not inclusion probabilities.
Historical gate: positive paired lower bounds and Holm p < .05 against all four
ticket baselines, plus lower Brier and log loss than uniform probabilities.
No tuning or refitting follows inspection of holdout results.

Reproduce with the command recorded in protocol.json.
