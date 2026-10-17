# Experimental LSTM research

This research-only candidate ranker uses seven completed 49-number presence
vectors to forecast the next draw. It is inspired by the sequence-model idea
in Roi Polanitzer's article, but does not reproduce its in-sample evaluation,
full-history scaling, six-output regression or integer truncation.

## Reproduce

From the repository root:

```powershell
uv sync --all-groups
uv run --group research python scripts/benchmark_lstm.py --stage smoke --output reports/lstm_study/smoke
uv run --group research python scripts/benchmark_lstm.py --stage all --output reports/lstm_study/full
```

The default dataset is `data/lotto_results_2019.yaml`. The full protocol uses
the last 250 eligible targets for holdout, the previous 150 for validation,
and requires at least 200 earlier training targets. The model is a single
32-unit unidirectional LSTM and a 49-output linear layer, trained with
unweighted binary cross-entropy, Adam (0.001), batch size 32 and gradient norm
clipping at 1. Output biases start at the fair-draw log odds. Training stops
after ten epochs without validation log-loss improvement, up to 100 epochs;
the earliest best checkpoint is retained, without refitting. Five fixed CPU
seeds are averaged. Hidden state resets for every window; no scaling is used.

Chronological windows may cross split boundaries using previously completed
draws, but holdout labels never enter optimization or checkpoint selection.
The frozen model observes new completed draws as inference inputs, without
retraining. Rerunning resumes compatible seed checkpoints and completed jobs;
source, data, configuration or runtime changes require a new output directory.
Append-only evaluation must keep the original fitted split; recomputing the
trailing holdout on a longer dataset defines a different experiment.

## Reports and interpretation

Each job records training/checkpoint logs, split boundaries, raw ensemble and
per-seed probabilities, actual numbers, tickets, baseline probabilities and
metrics. `report.md` summarizes historical and twenty independent synthetic
fair-draw histories. `synthetic-aggregate.json` averages by history, not by
training seed. The protocol records data/source fingerprints and runtime
versions. Runtime measurements are diagnostic, not deterministic outputs.

Baselines are uniform probabilities, fixed-seed random tickets, lifetime
frequency, CDM and the existing Bayesian strategy. Frequency and CDM use the
same gap/number tie-break and consequently produce identical selections.
Bayesian probability metrics use its raw estimates, never display scores.
CDM has ticket metrics only: its categorical shares are not inclusion rates.

Reports include average matches, the full 0–6 match distribution, 3+ match
rate, Brier score, log loss and ten fixed calibration bins. Paired mean-hit
comparisons use a non-circular moving-block bootstrap, block length 20,
10,000 replicates and Holm-corrected one-sided centered tests against four
ticket baselines. Historical gate passage requires positive lower 95% bounds,
adjusted p-values below .05, and lower Brier/log loss than uniform probabilities.
Correlated/identical baselines are retained in that conservative fixed family.

This dataset has previously influenced development, so its results remain
exploratory. Passing this gate does not establish a confirmed future advantage.
Negative results must not trigger tuning against the same holdout. Synthetic
controls diagnose false signals; no test requires every control to fail.

## Portable research artifact

Explicit export is available with `--stage historical --artifact PATH.npz`.
Artifacts may be exported after negative results and are labeled exploratory.
They contain five sets of numeric weights and metadata, loaded with
`allow_pickle=False` through `rand_ai.lstm.load_lstm_artifact`. NumPy inference
is checked against PyTorch with an absolute tolerance of 1e-6.

`artifact.predict(completed_draws)` returns 49 raw scores only after the stored
activation reference and when its dated training/validation prefix matches.
`top_six(scores[None])` ranks descending scores with ascending-number ties.
The six selected numbers are distinct; the scores are model estimates, not
calibrated guarantees or a joint distribution over valid tickets.

Benchmark exports are never automatically installed. The desktop explicitly
bundles the reviewed frozen artifact as `lstm_v1.npz` and offers it as
[LSTM (Experimental)](strategies/lstm.md), disabled by default. It uses NumPy
without PyTorch. The historical research gate failed; integration supports
continued observation and does not establish an advantage. Existing fixed
ensemble rosters and saved strategy choices remain unchanged.
