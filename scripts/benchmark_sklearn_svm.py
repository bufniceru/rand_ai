"""Benchmark Scikit Online SVM against custom SVC and random expectation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rand_ai import PredictionSuite, load_lotto_results_yaml
from rand_ai.strategy_prediction import build_prediction_suites

_STRATEGY_IDS = ("svc", "sklearn_svm")
_EXPECTED_RANDOM_HITS = 36 / 49
_MAXIMUM_ALLOWED_REGRESSION = 0.02


def _evaluated_suites(dataset_path: Path) -> list[PredictionSuite]:
    draws = load_lotto_results_yaml(dataset_path)
    draws.prepare_predictions()
    evaluated: list[PredictionSuite] = []

    def progress(completed: int, total: int) -> None:
        if completed % 100 == 0 or completed == total:
            print(
                f"{dataset_path.name}: {completed}/{total}",
                file=sys.stderr,
                flush=True,
            )

    build_prediction_suites(
        draws.draws,
        history_start=len(draws.draws),
        enabled_strategy_ids=_STRATEGY_IDS,
        progress=progress,
        evaluated_suite=evaluated.append,
    )
    return evaluated


def _hits(suites: Sequence[PredictionSuite], strategy_id: str) -> list[int]:
    values: list[int] = []
    for suite in suites:
        strategy = next(
            item for item in suite.strategies if item.strategy_id == strategy_id
        )
        values.append(len(set(suite.actual_numbers).intersection(strategy.top_numbers)))
    return values


def _scope(suites: Sequence[PredictionSuite]) -> dict[str, Any]:
    strategy_hits = {
        strategy_id: _hits(suites, strategy_id) for strategy_id in _STRATEGY_IDS
    }
    summaries = {
        strategy_id: {
            "totalHits": sum(hits),
            "averageHitsPerDraw": sum(hits) / len(hits) if hits else 0.0,
        }
        for strategy_id, hits in strategy_hits.items()
    }
    difference = (
        summaries["sklearn_svm"]["averageHitsPerDraw"]
        - summaries["svc"]["averageHitsPerDraw"]
    )
    return {
        "evaluatedDraws": len(suites),
        "expectedRandomHitsPerDraw": _EXPECTED_RANDOM_HITS,
        "strategies": summaries,
        "scikitMinusCustomHitsPerDraw": difference,
        "aboveRandom": (
            summaries["sklearn_svm"]["averageHitsPerDraw"]
            > _EXPECTED_RANDOM_HITS
        ),
        "competitive": difference >= -_MAXIMUM_ALLOWED_REGRESSION,
        "beatsCustomSvc": difference > 0,
    }


def _rankings(suites: Sequence[PredictionSuite]) -> list[list[int]]:
    return [
        list(
            next(
                strategy
                for strategy in suite.strategies
                if strategy.strategy_id == "sklearn_svm"
            ).top_numbers
        )
        for suite in suites
    ]


def benchmark(dataset_path: Path, verify_determinism: bool) -> dict[str, Any]:
    suites = _evaluated_suites(dataset_path)
    scopes = {"wholeHistory": _scope(suites)}
    if len(suites) >= 500:
        scopes["latest500"] = _scope(suites[-500:])
    if len(suites) >= 250:
        scopes["latest250"] = _scope(suites[-250:])
    deterministic: bool | None = None
    if verify_determinism:
        deterministic = _rankings(suites) == _rankings(
            _evaluated_suites(dataset_path)
        )
    return {
        "dataset": str(dataset_path.resolve()),
        "drawCount": len(suites) + 1 if suites else 0,
        "scopes": scopes,
        "deterministic": deterministic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        dest="datasets",
        help="YAML draw history; may be supplied more than once.",
    )
    parser.add_argument("--skip-determinism", action="store_true")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    datasets = options.datasets or [
        Path("data/lotto_results_2019.yaml"),
        Path("data/lotto_results.yaml"),
    ]
    reports = [
        benchmark(dataset, not options.skip_determinism) for dataset in datasets
    ]
    scopes = [scope for report in reports for scope in report["scopes"].values()]
    acceptance = {
        "aboveRandomInEveryScope": all(scope["aboveRandom"] for scope in scopes),
        "competitiveInEveryScope": all(scope["competitive"] for scope in scopes),
        "beatsCustomSvcInAtLeastOneScope": any(
            scope["beatsCustomSvc"] for scope in scopes
        ),
        "deterministic": all(
            report["deterministic"] is True for report in reports
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    rendered = json.dumps(
        {
            "datasets": reports,
            "acceptance": acceptance,
            "rule": (
                "Scikit Online SVM must beat 36/49 in every scope, trail custom "
                "SVC by no more than 0.02 hits/draw in any scope, win at least "
                "one scope, and replay deterministically."
            ),
        },
        indent=2,
    )
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
