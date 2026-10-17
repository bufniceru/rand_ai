"""Audit, freeze, and prospectively evaluate the fixed XGBoost V2 protocol."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import csv
import time

import numpy as np

from rand_ai import Draw, Draws, load_lotto_results_yaml
from rand_ai.strategy_prediction import _StrategyState
from rand_ai import xgboost_classifier_v2 as v2

STRATEGIES = (
    "xgboost_classifier",
    "raw_v2",
    "xgboost_classifier_v2",
    "lag_logistic",
    "smoothed_frequency",
    "uniform",
)
SEEDS = (20260626, 20260627, 20260628)


def protocol():
    """The freeze contract includes the policy, feature code, and runtime versions."""
    feature_code = "\n".join(
        inspect.getsource(getattr(_StrategyState, name))
        for name in (
            "_history_features",
            "_lag_logistic_pattern",
            "_sklearn_svm_recent_residual",
            "_mean_gap",
        )
    )
    return {
        "formatVersion": 1,
        "candidates": [asdict(candidate) for candidate in v2.CANDIDATES],
        "estimatorParameters": [
            v2.estimator_parameters(candidate) for candidate in v2.CANDIDATES
        ],
        "minimumTransitions": v2.MIN_TRANSITIONS,
        "trainingWindow": v2.TRAINING_WINDOW,
        "selectionInterval": v2.RESELECT_INTERVAL,
        "refitInterval": v2.REFIT_INTERVAL,
        "validationBlocks": [40, 40, 40],
        "gateWindow": v2.GATE_WINDOW,
        "gateRule": "lower95LogLossAdvantage>0 and meanTopSixHitAdvantage>0",
        "bootstrapReplicates": v2.BOOTSTRAP_REPLICATES,
        "bootstrapBlock": v2.BOOTSTRAP_BLOCK,
        "seed": v2.SEED,
        "baselinePrior": 24,
        "lossClipping": [1e-7, 1 - 1e-7],
        "codeSha256": hashlib.sha256(
            Path(v2.__file__).read_bytes() + feature_code.encode()
        ).hexdigest(),
        "versions": {
            name: version(name) for name in ("xgboost", "numpy", "scikit-learn")
        },
    }


def prefix_hash(draws, cutoff):
    payload = [
        [draw.date, *[ball.value for ball in draw.balls]] for draw in draws[:cutoff]
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def freeze(dataset: Path, destination: Path, cutoff: int | None = None):
    draws = load_lotto_results_yaml(dataset).draws
    cutoff = len(draws) if cutoff is None else cutoff
    if not 241 <= cutoff <= len(draws):
        raise ValueError(
            "Freeze cutoff must include at least 241 draws and stay within the dataset"
        )
    manifest = {
        "cutoff": cutoff,
        "prefixSha256": prefix_hash(draws, cutoff),
        "protocol": protocol(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_manifest(draws, manifest):
    if manifest["protocol"] != protocol():
        raise ValueError(
            "Frozen protocol, feature code, or dependency versions changed"
        )
    cutoff = manifest["cutoff"]
    if not isinstance(cutoff, int) or not 241 <= cutoff < len(draws):
        raise ValueError(
            "Prospective evaluation requires draws appended after a valid frozen cutoff"
        )
    if prefix_hash(draws, cutoff) != manifest["prefixSha256"]:
        raise ValueError("The frozen drawing-history prefix changed")
    return cutoff


def synthetic_history(scenario: str, seed: int, count: int = 1000):
    rng = np.random.default_rng(seed)
    previous: set[int] = set()
    draws = Draws()
    for _ in range(count):
        weights = np.ones(49)
        if scenario == "bias":
            weights[:6] = 20
        elif scenario == "persistence":
            for number in previous:
                weights[number - 1] = 30
        elif scenario != "fair":
            raise ValueError(f"Unknown synthetic scenario: {scenario}")
        numbers = [
            int(n)
            for n in rng.choice(
                np.arange(1, 50), 6, replace=False, p=weights / weights.sum()
            )
        ]
        previous = set(numbers)
        draws.add(Draw(*numbers))
    return draws


def feature_transitions(draws):
    """Extract once in chronological order, with targets outside each snapshot."""
    state = _StrategyState(("lag_logistic",))
    rows = []
    for index, draw in enumerate(draws[:-1]):
        state.remember({ball.value for ball in draw.balls}, draw.date)
        target = {ball.value for ball in draws[index + 1].balls}
        rows.append(
            (
                state._history_feature_matrix(),
                np.asarray([int(n in target) for n in range(1, 50)]),
            )
        )
    return rows


def learning_controls():
    """Report every candidate on fixed synthetic test sets; never select on test."""
    results = []
    for scenario in ("fair", "bias", "persistence"):
        for seed in SEEDS:
            rows = feature_transitions(synthetic_history(scenario, seed).draws)
            prefix, testing = rows[:500], rows[500:]
            for candidate in v2.CANDIDATES:
                training = prefix[-candidate.window :]
                model = v2.create_estimator(candidate)
                model.fit(
                    np.concatenate(
                        [x[:, : candidate.feature_count] for x, _ in training]
                    ),
                    np.concatenate([y for _, y in training]),
                )
                train_losses = [
                    v2.forecast_metrics(
                        y, v2.probabilities(model, x[:, : candidate.feature_count])
                    )[0]
                    for x, y in training
                ]
                measured = [
                    v2.forecast_metrics(
                        y, v2.probabilities(model, x[:, : candidate.feature_count])
                    )
                    for x, y in testing
                ]
                uniform_losses = [
                    v2.forecast_metrics(y, np.full(49, 6 / 49))[0] for _, y in testing
                ]
                values = np.asarray(measured)
                interval = v2.bootstrap_interval(
                    np.asarray(uniform_losses) - values[:, 0]
                )
                required = (scenario == "bias" and candidate.feature_count == 14) or (
                    scenario == "persistence" and candidate.feature_count == 4
                )
                recovered = (
                    float(values[:, 0].mean()) < float(np.mean(uniform_losses))
                    and float(values[:, 2].mean()) > 36 / 49
                )
                results.append(
                    dict(
                        scenario=scenario,
                        seed=seed,
                        candidate=candidate.identifier,
                        trainingLogLoss=float(np.mean(train_losses)),
                        logLoss=float(values[:, 0].mean()),
                        brier=float(values[:, 1].mean()),
                        meanHits=float(values[:, 2].mean()),
                        uniformLossAdvantage95=list(interval),
                        apparentImprovement=interval[0] > 0,
                        recoveryRequired=required,
                        recoveredSignal=bool(recovered),
                    )
                )
    return {
        "rows": results,
        "passed": all(
            row["recoveredSignal"] for row in results if row["recoveryRequired"]
        ),
        "fairApparentImprovements": sum(
            row["apparentImprovement"] for row in results if row["scenario"] == "fair"
        ),
        "interpretation": "Fixed 500-transition training / 499-transition test; all eight configurations reported; no test-selected winner.",
    }


def collect(draws, cutoff=0):
    """Compare raw and deployed probabilities on exactly the same target draws."""
    draws.prepare_predictions()
    state = _StrategyState(
        ("xgboost_classifier", "xgboost_classifier_v2", "lag_logistic")
    )
    rows = []
    selections = []
    for index, draw in enumerate(draws.draws):
        state.train({ball.value for ball in draw.balls})
        state.remember({ball.value for ball in draw.balls}, draw.date)
        suite = state.build_strategies(draw.prediction, index)
        model = state.xgboost_classifier_v2
        assert model is not None
        if (
            model.last_selection_transition == model.completed_transitions
            and model.selected is not None
        ):
            selections.append(
                {
                    "referenceDraw": index + 1,
                    "selected": model.selected.identifier,
                    "candidates": model.selection_results,
                }
            )
        if (
            model.raw_probabilities is None
            or index + 1 >= len(draws.draws)
            or index + 2 <= cutoff
        ):
            continue
        scores = {
            s.strategy_id: np.asarray(
                [
                    next(n.score for n in s.numbers if n.number == number)
                    for number in range(1, 50)
                ]
            )
            for s in suite
        }
        scores["raw_v2"] = model.raw_probabilities
        scores["smoothed_frequency"] = model.baseline_probabilities
        scores["uniform"] = np.full(49, 6 / 49)
        actual = {ball.value for ball in draws.draws[index + 1].balls}
        labels = np.asarray([int(n in actual) for n in range(1, 50)])
        for name in STRATEGIES:
            values = scores[name]
            loss, brier, hits = v2.forecast_metrics(labels, values)
            rows.append(
                {
                    "targetDraw": index + 2,
                    "strategy": name,
                    "numbers": (np.lexsort((np.arange(49), -values))[:6] + 1).tolist(),
                    "probabilities": values.tolist(),
                    "actual": sorted(actual),
                    "logLoss": loss,
                    "brier": brier,
                    "hits": hits,
                    "source": model.display_source
                    if name == "xgboost_classifier_v2"
                    else name,
                    "configuration": model.selected.identifier,
                    "trainingLogLoss": model.training_loss,
                    "validationLogLoss": model.validation_loss,
                    "gateSamples": len(model.evidence),
                    "gateInterval": model.gate_interval,
                }
            )
    return rows, selections


def summarize(rows):
    hits = np.asarray([row["hits"] for row in rows])
    losses = np.asarray([row["logLoss"] for row in rows])
    briers = np.asarray([row["brier"] for row in rows])
    blocks = np.array_split(np.arange(len(rows)), 5)
    return {
        "draws": len(rows),
        "meanHits": float(hits.mean()),
        "hitDistribution": [int(np.sum(hits == i)) for i in range(7)],
        "threePlusRate": float(np.mean(hits >= 3)),
        "logLoss": float(losses.mean()),
        "brier": float(briers.mean()),
        "blocks": [
            {
                "draws": len(block),
                "meanHits": float(hits[block].mean()),
                "logLoss": float(losses[block].mean()),
                "brier": float(briers[block].mean()),
            }
            if len(block)
            else None
            for block in blocks
        ],
    }


def run(dataset: Path, output: Path, manifest: dict | None = None, controls=True):
    started = time.perf_counter()
    draws = load_lotto_results_yaml(dataset)
    cutoff = 0 if manifest is None else verify_manifest(draws.draws, manifest)
    rows, selections = collect(draws, cutoff)
    if not rows:
        raise ValueError(
            "Audit needs at least 242 draws with an evaluated fitted V2 forecast"
        )
    grouped = {
        name: [row for row in rows if row["strategy"] == name] for name in STRATEGIES
    }
    comparisons = {}
    for name in ("raw_v2", "xgboost_classifier_v2"):
        for other in STRATEGIES:
            if other == name:
                continue
            loss_adv = np.asarray(
                [
                    b["logLoss"] - a["logLoss"]
                    for a, b in zip(grouped[name], grouped[other], strict=True)
                ]
            )
            hit_adv = np.asarray(
                [
                    a["hits"] - b["hits"]
                    for a, b in zip(grouped[name], grouped[other], strict=True)
                ]
            )
            comparisons[f"{name}_vs_{other}"] = {
                "meanLogLossAdvantage": float(loss_adv.mean()),
                "logLossAdvantage95": v2.bootstrap_interval(loss_adv),
                "meanHitAdvantage": float(hit_adv.mean()),
                "hitAdvantage95": v2.bootstrap_interval(hit_adv),
            }
    deployment = grouped["xgboost_classifier_v2"]
    report = {
        "interpretation": "Exploratory historical evidence; repeated gates and inspected history do not establish future advantage.",
        "evaluation": "appended draws after frozen cutoff"
        if manifest
        else "retrospective",
        "datasetSha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "protocol": protocol(),
        "frozenCutoff": cutoff,
        "expectedRandomHits": 36 / 49,
        "strategies": {name: summarize(records) for name, records in grouped.items()},
        "pairedComparisons": comparisons,
        "fallbackFrequency": sum(
            row["source"] == "smoothed_frequency" for row in deployment
        )
        / len(deployment),
        "configurationForecastCounts": dict(
            Counter(row["configuration"] for row in deployment)
        ),
        "configurationSelectionCounts": dict(
            Counter(row["selected"] for row in selections)
        ),
        "selections": selections,
        "learningControls": learning_controls() if controls else {"status": "not run"},
        "runtimeSeconds": time.perf_counter() - started,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "xgboost_v2_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "xgboost_v2_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value) if isinstance(value, (list, tuple)) else value
                for key, value in row.items()
            }
            for row in rows
        )
    lines = [
        "# XGBoost V2 audit",
        "",
        report["interpretation"],
        "",
        f"Evaluation: {report['evaluation']}. Fallback: {report['fallbackFrequency']:.1%}.",
        "",
        "| Strategy | Draws | Mean hits | Log loss | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["strategies"].items():
        lines.append(
            f"| {name} | {result['draws']} | {result['meanHits']:.6f} | {result['logLoss']:.6f} | {result['brier']:.6f} |"
        )
    lines += [
        "",
        "## Learning controls",
        "",
        json.dumps(
            {k: v for k, v in report["learningControls"].items() if k != "rows"}
        ),
        "",
        "Full paired intervals, five chronological blocks, configuration diagnostics, and control results are in xgboost_v2_audit.json.",
        "",
        f"Runtime: {report['runtimeSeconds']:.2f}s. No configuration or gate thresholds were changed based on these results.",
    ]
    (output / "xgboost_v2_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if manifest is None:
        freeze(dataset, output / "xgboost_v2_freeze.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "freeze", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument(
            "--dataset", type=Path, default=Path("data/lotto_results_2019.yaml")
        )
        if name == "freeze":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--cutoff", type=int)
        else:
            command.add_argument(
                "--output", type=Path, default=Path("reports/xgboost_v2")
            )
            command.add_argument("--skip-controls", action="store_true")
            if name == "evaluate":
                command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.dataset, args.manifest, args.cutoff), indent=2))
    else:
        manifest = (
            json.loads(args.manifest.read_text(encoding="utf-8"))
            if args.command == "evaluate"
            else None
        )
        report = run(args.dataset, args.output, manifest, not args.skip_controls)
        print(
            json.dumps(
                {
                    "strategies": report["strategies"],
                    "fallbackFrequency": report["fallbackFrequency"],
                    "learningControlsPassed": report["learningControls"].get("passed"),
                },
                indent=2,
            )
        )
        if report["learningControls"].get("passed") is False:
            raise SystemExit(
                "Planted-signal recovery failed; inspect the audit before accepting this implementation"
            )


if __name__ == "__main__":
    main()
