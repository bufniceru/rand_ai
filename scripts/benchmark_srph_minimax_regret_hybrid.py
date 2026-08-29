"""Benchmark the frozen SRPH pure minimax-regret shadow strategy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from rand_ai import PredictionSuite, load_lotto_results_yaml
from rand_ai.strategy_prediction import build_prediction_suites

_STRATEGY_IDS = (
    "svc_recurrence_proximity_hybrid",
    "srph_minimax_regret_hybrid",
)
_WEIGHT_PATTERN = re.compile(r" weight ([0-9.]+)%")


def _strategy(suite: PredictionSuite, strategy_id: str):
    return next(
        item for item in suite.strategies if item.strategy_id == strategy_id
    )


def _hit_count(suite: PredictionSuite, strategy_id: str) -> int:
    strategy = _strategy(suite, strategy_id)
    return len(set(suite.actual_numbers).intersection(strategy.top_numbers))


def _summary(
    suites: Sequence[PredictionSuite],
    strategy_id: str,
) -> dict[str, int | float]:
    hits = [_hit_count(suite, strategy_id) for suite in suites]
    return {
        "evaluatedDraws": len(hits),
        "totalHits": sum(hits),
        "averageHitsPerDraw": mean(hits) if hits else 0.0,
        "liftOverTheoreticalRandom": sum(hits) - len(hits) * 36 / 49,
    }


def _scope(suites: Sequence[PredictionSuite]) -> dict[str, Any]:
    return {
        strategy_id: _summary(suites, strategy_id)
        for strategy_id in _STRATEGY_IDS
    }


def _weight_key(suite: PredictionSuite) -> str:
    details = _strategy(suite, "srph_minimax_regret_hybrid").numbers[0].details
    weights = []
    for detail in details[1:6]:
        match = _WEIGHT_PATTERN.search(detail)
        if match is None:
            raise ValueError(f"Missing SMR weight detail: {detail}")
        weights.append(f"{float(match.group(1)):g}")
    return "/".join(weights)


def benchmark(dataset_path: Path) -> dict[str, Any]:
    draws = load_lotto_results_yaml(dataset_path)
    draws.prepare_predictions()
    evaluated: list[PredictionSuite] = []

    def report_progress(completed: int, total: int) -> None:
        if completed % 100 == 0 or completed == total:
            print(
                f"Walk-forward benchmark {completed}/{total}",
                file=sys.stderr,
                flush=True,
            )

    build_prediction_suites(
        draws.draws,
        history_start=len(draws.draws),
        enabled_strategy_ids=_STRATEGY_IDS,
        evaluated_suite=evaluated.append,
        progress=report_progress,
    )
    validation = [
        suite for suite in evaluated if 121 <= suite.target_draw_number <= 520
    ]
    holdout = [
        suite for suite in evaluated if 521 <= suite.target_draw_number <= 770
    ]
    selections = Counter(_weight_key(suite) for suite in evaluated)
    return {
        "dataset": str(dataset_path.resolve()),
        "drawCount": len(draws.draws),
        "sourceOrder": [
            "SRPH",
            "Freshness",
            "EMD",
            "Bayesian",
            "Doublet/Triplet Markov",
        ],
        "candidateMixtures": 503,
        "weightStep": 0.05,
        "minimumSrphWeight": 0.50,
        "maximumPerResidualWeight": 0.20,
        "blockSize": 40,
        "minimumCompletedBlocks": 4,
        "theoreticalRandomHitsPerDraw": 36 / 49,
        "fullHistory": _scope(evaluated),
        "validation121To520": _scope(validation),
        "holdout521To770": _scope(holdout),
        "selectionCounts": dict(sorted(selections.items())),
        "finalWeights": _weight_key(evaluated[-1]),
        "promotion": {
            "passed": False,
            "rule": (
                "Remain a default-disabled shadow strategy because pure "
                "minimax regret trails SRPH on the full replay and holdout."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/lotto_results_2019.yaml"),
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = benchmark(options.dataset)
    rendered = json.dumps(report, indent=2)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
