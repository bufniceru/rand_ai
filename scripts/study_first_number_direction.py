"""Run the frozen historical and synthetic sorted-position direction study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rand_ai.draw import Draw
from rand_ai.draws import Draws
from rand_ai.first_number_direction import PROTOCOL, analyze_number_directions
from rand_ai.lotto_results import load_lotto_results_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/lotto_results_2019.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/first_number_direction")
    )
    parser.add_argument(
        "--stage", choices=("historical", "synthetic", "all"), default="all"
    )
    args = parser.parse_args()
    draws = load_lotto_results_yaml(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol": PROTOCOL,
        "sourceFingerprint": hashlib.sha256(
            json.dumps(
                [
                    [draw.date, *[ball.value for ball in draw.balls]]
                    for draw in draws.draws
                ]
            ).encode()
        ).hexdigest(),
    }
    manifest_path = args.output / "manifest.json"
    compatible = (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) == identity
    )
    manifest_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    (args.output / "protocol.json").write_text(
        json.dumps(PROTOCOL, indent=2), encoding="utf-8"
    )
    datasets = []
    if args.stage in ("historical", "all"):
        datasets.append(("historical", draws))
    if args.stage in ("synthetic", "all"):
        for index in range(PROTOCOL["syntheticHistories"]):
            rng = np.random.default_rng(PROTOCOL["seed"] + index)
            synthetic = Draws()
            for _ in range(len(draws)):
                synthetic.add(
                    Draw(*map(int, rng.choice(np.arange(1, 50), 6, replace=False)))
                )
            datasets.append((f"synthetic-{index:02}", synthetic))
    summaries = {}
    for name, dataset in datasets:
        result = analyze_number_directions(dataset)
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        position_summaries = []
        for position_report in result["positions"]:
            records = position_report.pop("history")
            position_folder = folder / f"position-{position_report['position']}"
            position_folder.mkdir(exist_ok=True)
            (position_folder / "summary.json").write_text(
                json.dumps(position_report, indent=2, allow_nan=False), encoding="utf-8"
            )
            with (position_folder / "predictions.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for row in records:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            position_summaries.append(
                {
                    "position": position_report["position"],
                    "evaluation": position_report["evaluation"],
                    "learnedPolicyForecasts": sum(
                        row["activeModel"] != "baseline" for row in records
                    ),
                }
            )
        (folder / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        summaries[name] = {
            "datasetFingerprint": result["datasetFingerprint"],
            "positions": position_summaries,
        }
        print(f"{name}: completed all six positions", flush=True)
    aggregate_path = args.output / "aggregate.json"
    if aggregate_path.exists() and compatible:
        existing = json.loads(aggregate_path.read_text(encoding="utf-8"))
        existing.update(summaries)
        summaries = existing
    aggregate_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    lines = [
        "# Number Directions study",
        "",
        str(PROTOCOL["interpretation"]),
        "",
        "Log loss is primary; smaller is better. Holm correction is per position, not simultaneous across six positions or repeated reviews.",
        "",
        "| Dataset | Position | Baseline loss | Gated loss | Gated accuracy | Learned policy forecasts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in summaries.items():
        for item in result["positions"]:
            evaluation = item["evaluation"]
            lines.append(
                f"| {name} | {item['position']} | {evaluation['baseline']['logLoss']} | {evaluation['gated']['logLoss']} | {evaluation['gated']['accuracy']} | {item['learnedPolicyForecasts']} |"
            )
    historical = summaries.get("historical")
    if historical:
        lines.extend(
            [
                "",
                "## Historical candidate comparison",
                "",
                "| Position | Model | Log loss | Brier | Accuracy |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for position in historical["positions"]:
            for name, item in position["evaluation"].items():
                lines.append(
                    f"| {position['position']} | {name} | {item['logLoss']} | {item['brier']} | {item['accuracy']} |"
                )
    synthetic = [
        value for name, value in summaries.items() if name.startswith("synthetic-")
    ]
    for index in range(6):
        activated = sum(
            item["positions"][index]["learnedPolicyForecasts"] > 0 for item in synthetic
        )
        lines.extend(
            [
                "",
                f"Position {index + 1}: synthetic histories with any learned-policy activation: {activated}/{len(synthetic)}.",
            ]
        )
    lines.extend(
        [
            "",
            "Activation on independent synthetic draws is a false-signal diagnostic, not predictive skill.",
        ]
    )
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
