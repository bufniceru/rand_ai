"""Frozen chronological XGBoost comparison; outputs exploratory evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from rand_ai import load_lotto_results_yaml
from rand_ai.strategy_prediction import _StrategyState, build_prediction_suites

SEED = 20260626
EXPECTED = 36 / 49


def summarize(hits):
    values = np.asarray(hits, dtype=float)
    return {
        "draws": len(values),
        "totalHits": int(values.sum()),
        "meanHits": float(values.mean()),
        "liftOverRandom": float(values.mean() - EXPECTED),
        "hitDistribution": [int(np.sum(values == i)) for i in range(7)],
        "threePlusRate": float(np.mean(values >= 3)),
        "blocks": [
            float(block.mean()) if len(block) else None
            for block in np.array_split(values, 5)
        ],
    }


def paired_interval(difference, replicates=10000):
    values = np.asarray(difference, dtype=float)
    rng = np.random.default_rng(SEED)
    length = min(20, len(values))
    starts = rng.integers(
        0,
        len(values) - length + 1,
        size=(replicates, (len(values) + length - 1) // length),
    )
    indices = (starts[:, :, None] + np.arange(length)).reshape(replicates, -1)[
        :, : len(values)
    ]
    means = values[indices].mean(axis=1)
    return {
        "meanDifference": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def run(dataset: Path, output: Path):
    started = time.perf_counter()
    draws = load_lotto_results_yaml(dataset)
    draws.prepare_predictions()
    suites = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=("xgboost_classifier", "lag_logistic", "randomness"),
    )
    state = _StrategyState(())
    hits = {
        name: []
        for name in (
            "xgboost_classifier",
            "lag_logistic",
            "smoothed_frequency",
            "randomness",
        )
    }
    rows = []
    for index, suite in enumerate(suites):
        drawn = {ball.value for ball in draws.draws[index].balls}
        state.remember(drawn)
        # Draw 121 has 120 completed forecast/target transitions; target 122 is first evaluated.
        if index < 120 or not suite.actual_numbers:
            continue
        predictions = {s.strategy_id: list(s.top_numbers) for s in suite.strategies}
        predictions["smoothed_frequency"] = sorted(
            range(1, 50),
            key=lambda n: (
                -(state.appearances[n] + 24 * 6 / 49) / (state.draw_count + 24),
                n,
            ),
        )[:6]
        actual = set(suite.actual_numbers)
        for strategy, numbers in predictions.items():
            count = len(actual.intersection(numbers))
            hits[strategy].append(count)
            rows.append(
                {
                    "targetDraw": suite.target_draw_number,
                    "strategy": strategy,
                    "numbers": numbers,
                    "actual": sorted(actual),
                    "hits": count,
                }
            )
    if not hits["xgboost_classifier"]:
        raise ValueError("Evaluation needs at least 122 draws")
    report = {
        "dataset": str(dataset),
        "datasetSha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "seed": SEED,
        "initialCompletedTransitions": 120,
        "trainingWindow": 500,
        "retrainInterval": 10,
        "bootstrapReplicates": 10000,
        "bootstrapBlockLength": 20,
        "expectedRandomHits": EXPECTED,
        "interpretation": "Exploratory; history previously used for strategy development.",
        "strategies": {name: summarize(values) for name, values in hits.items()},
        "pairedComparisons": {
            name: paired_interval(np.asarray(hits["xgboost_classifier"]) - values)
            for name, values in hits.items()
            if name != "xgboost_classifier"
        },
        "runtimeSeconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "xgboost_study.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "xgboost_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["targetDraw", "strategy", "numbers", "actual", "hits"]
        )
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# XGBoost chronological study",
        "",
        report["interpretation"],
        "",
        f"Random expectation: {EXPECTED:.6f} hits. Runtime: {report['runtimeSeconds']:.2f}s.",
        "",
        "| Strategy | Draws | Mean hits | 3+ rate |",
        "|---|---:|---:|---:|",
    ]
    for name, result in report["strategies"].items():
        lines.append(
            f"| {name} | {result['draws']} | {result['meanHits']:.6f} | {result['threePlusRate']:.2%} |"
        )
    lines += ["", "## Paired differences (XGBoost minus comparator)", ""]
    for name, result in report["pairedComparisons"].items():
        lines.append(
            f"- {name}: {result['meanDifference']:.6f}; 95% moving-block bootstrap interval {result['ci95']}."
        )
    lines += [
        "",
        "Full hit distributions and five chronological block means are in xgboost_study.json.",
    ]
    (output / "xgboost_study.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/lotto_results_2019.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/xgboost"))
    args = parser.parse_args()
    report = run(args.dataset, args.output)
    print(
        json.dumps(
            {
                "strategies": report["strategies"],
                "runtimeSeconds": report["runtimeSeconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
