# Experimental LSTM study

Exploratory historical evidence only; this history previously influenced development.
No desktop registration, automatic installation, or claim of confirmed advantage.

Fixed 7 x 49 inputs; one 32-unit LSTM; five-seed probability ensemble.

| Dataset | Mean hits | 3+ rate | Brier | Log loss | Historical gate |
|---|---:|---:|---:|---:|---|
| smoke | 0.750000 | 5.000% | 0.107448 | 0.371742 | False |

Full metrics, fixed-bin reliability data, individual seed results, training logs,
split boundaries, raw forecasts and fingerprints are in adjacent JSON/JSONL/NPZ files.
CDM has ranking metrics only; its categorical shares are not inclusion probabilities.
Historical gate: positive paired lower bounds and Holm p < .05 against all four
ticket baselines, plus lower Brier and log loss than uniform probabilities.
No tuning or refitting follows inspection of holdout results.

Reproduce with the command recorded in protocol.json.
