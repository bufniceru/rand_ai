# Statistics and prediction simplification study

Exploratory historical evidence only: this dataset previously influenced strategy selection. The last two blocks are a frozen within-study comparison, not independent confirmation.

Reproduce: `uv run python scripts/study_statistics.py`

Dataset SHA-256: `abc`. Draws: 121.
Targets by block: []. Random expectation: 0.734694 hits/draw.
Protocol and raw rankings, selections, hits, dependencies, export fields, and uncertainty are in the adjacent JSON/JSONL/CSV files.
Moving-block bootstrap: length 20, 10,000 replicates, seed 20261006. One-sided centered bootstrap tests use Holm correction across all baseline, ablation, and merge improvement claims. Equivalence uses paired 95% intervals; it is a simplification screen, not proof of future equivalence.

## Strategies

| Strategy | Hits | Mean | 3+ rate | Five block means | Decision |
|---|---:|---:|---:|---|---|

missing: unevaluated.

## Redundancy

No pair meets the duplicate/equivalence consolidation rule. Retain all strategy controls.

## Input removal experiments

Families are removed at both training and prediction time; each model is retrained from scratch. SVC, TBL and online SVM are covered. Other models' inputs are untested and retained.

| Consumer / removed family | Mean | Difference | Paired 95% interval | Holm p | Decision |
|---|---:|---:|---|---:|---|

## Merge experiments


## Statistics and presentation

Circular-space position charts and their combined heatmap share one expandable detail panel under the aggregate distribution. These are views of the same counts, rather than independent predictive signals. Summary and number/space descriptive tables become expandable details. Charts, exports, settings and diagnostic access remain available.

| Statistic | Role | Fields | Decision | UI |
|---|---|---|---|---|

## Cost and limitations

Shared strategy evaluation took 1.00 seconds on this machine. Ablation runtimes are recorded individually; shared computation prevents attributing the full-run time to individual strategies. Feature masks test specific consumers, not every use of a statistic. No uncertain statistical input is deleted, and no historical win is presented as proven future prediction.
