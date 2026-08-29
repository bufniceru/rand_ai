"""Benchmark the guarded SRPH residual-diversity shadow strategy."""

from __future__ import annotations

import argparse
import json
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
    "srph_residual_diversity_hybrid",
)
_SELECTION_PREFIX = "Selector selected "


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


def _selector_counts(suites: Sequence[PredictionSuite]) -> dict[str, Any]:
    selections: Counter[str] = Counter()
    fallback_count = 0
    for suite in suites:
        strategy = _strategy(suite, "srph_residual_diversity_hybrid")
        status = strategy.numbers[0].details[0]
        if status == "Selector fallback to SRPH":
            fallback_count += 1
        elif status.startswith(_SELECTION_PREFIX):
            selections[status.removeprefix(_SELECTION_PREFIX)] += 1
    return {
        "fallback": fallback_count,
        "selected": {
            "Freshness": selections["Freshness"],
            "EMD": selections["EMD"],
            "Bayesian": selections["Bayesian"],
            "Doublet/Triplet Markov": selections["Doublet/Triplet Markov"],
        },
    }


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
    return {
        "dataset": str(dataset_path.resolve()),
        "drawCount": len(draws.draws),
        "priorDraws": 24,
        "residualWeight": 0.10,
        "candidateOrder": [
            "Freshness",
            "EMD",
            "Bayesian",
            "Doublet/Triplet Markov",
        ],
        "theoreticalRandomHitsPerDraw": 36 / 49,
        "fullHistory": _scope(evaluated),
        "validation121To520": _scope(validation),
        "holdout521To770": _scope(holdout),
        "selector": _selector_counts(evaluated),
        "promotion": {
            "passed": False,
            "rule": (
                "Remain a default-disabled shadow strategy because the guarded "
                "selector trails SRPH on the nominal holdout."
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
