"""Benchmark MKRD against fixed walk-forward baselines and a rank blend."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from rand_ai import PredictionSuite, load_lotto_results_yaml
from rand_ai.strategy_prediction import build_prediction_suites

_STRATEGY_IDS = ("randomness", "mksp", "mknp", "mkrd")
_HOLDOUT_DRAWS = 250


def _hit_count(suite: PredictionSuite, strategy_id: str) -> int:
    actual = set(suite.actual_numbers)
    strategy = next(
        item for item in suite.strategies if item.strategy_id == strategy_id
    )
    return len(actual.intersection(strategy.top_numbers))


def _rank_blend_hits(suite: PredictionSuite) -> int:
    strategies = {item.strategy_id: item for item in suite.strategies}
    scores = {number: 0.0 for number in range(1, 50)}
    for strategy_id in ("mksp", "mkrd"):
        for prediction in strategies[strategy_id].numbers:
            scores[prediction.number] += (49 - prediction.rank) / 48
    blended = set(
        sorted(scores, key=lambda number: (-scores[number], number))[:6]
    )
    return len(blended.intersection(suite.actual_numbers))


def _random_tail_probability(total_hits: int, draw_count: int) -> float:
    expected_per_draw = 36 / 49
    variance_per_draw = (
        6 * (6 / 49) * (43 / 49) * ((49 - 6) / (49 - 1))
    )
    deviation = total_hits - 0.5 - draw_count * expected_per_draw
    z_score = deviation / math.sqrt(draw_count * variance_per_draw)
    return 0.5 * math.erfc(z_score / math.sqrt(2))


def _summary(hits: Sequence[int]) -> dict[str, Any]:
    distribution = Counter(hits)
    total = sum(hits)
    return {
        "evaluatedDraws": len(hits),
        "totalHits": total,
        "averageHitsPerDraw": mean(hits),
        "zeroHits": distribution[0],
        "oneHit": distribution[1],
        "twoHits": distribution[2],
        "threeOrMoreHits": sum(
            count for hit_count, count in distribution.items() if hit_count >= 3
        ),
        "liftOverTheoreticalRandom": total - len(hits) * 36 / 49,
        "approximateRandomTailProbability": _random_tail_probability(
            total,
            len(hits),
        ),
    }


def _paired_summary(
    suites: Sequence[PredictionSuite],
    candidate_id: str,
    baseline_id: str,
) -> dict[str, Any]:
    differences = []
    overlaps = []
    correlations = []
    candidate_unique_hits = 0
    baseline_unique_hits = 0
    identical = 0
    for suite in suites:
        strategies = {item.strategy_id: item for item in suite.strategies}
        candidate = strategies[candidate_id]
        baseline = strategies[baseline_id]
        actual = set(suite.actual_numbers)
        candidate_top = set(candidate.top_numbers)
        baseline_top = set(baseline.top_numbers)
        differences.append(
            len(actual.intersection(candidate_top))
            - len(actual.intersection(baseline_top))
        )
        overlaps.append(len(candidate_top.intersection(baseline_top)))
        identical += candidate_top == baseline_top
        candidate_unique_hits += len(actual.intersection(candidate_top - baseline_top))
        baseline_unique_hits += len(actual.intersection(baseline_top - candidate_top))
        candidate_ranks = {item.number: item.rank for item in candidate.numbers}
        baseline_ranks = {item.number: item.rank for item in baseline.numbers}
        squared_difference = sum(
            (candidate_ranks[number] - baseline_ranks[number]) ** 2
            for number in range(1, 50)
        )
        correlations.append(
            1 - 6 * squared_difference / (49 * (49 * 49 - 1))
        )

    average_difference = mean(differences)
    standard_error = (
        stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0
    )
    return {
        "candidate": candidate_id,
        "baseline": baseline_id,
        "candidateWins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "baselineWins": sum(value < 0 for value in differences),
        "meanHitDifference": average_difference,
        "meanDifference95Interval": [
            average_difference - 1.96 * standard_error,
            average_difference + 1.96 * standard_error,
        ],
        "averageTop6Overlap": mean(overlaps),
        "identicalTop6": identical,
        "averageScoreRankCorrelation": mean(correlations),
        "candidateUniqueRealizedHits": candidate_unique_hits,
        "baselineUniqueRealizedHits": baseline_unique_hits,
    }


def _scope(suites: Sequence[PredictionSuite]) -> dict[str, Any]:
    strategy_hits = {
        strategy_id: [_hit_count(suite, strategy_id) for suite in suites]
        for strategy_id in _STRATEGY_IDS
    }
    blend_hits = [_rank_blend_hits(suite) for suite in suites]
    return {
        "strategies": {
            strategy_id: _summary(hits)
            for strategy_id, hits in strategy_hits.items()
        },
        "mkspMkrdRankBlend": _summary(blend_hits),
        "pairedComparisons": {
            baseline_id: _paired_summary(suites, "mkrd", baseline_id)
            for baseline_id in ("mksp", "mknp", "randomness")
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
    development = evaluated[:-_HOLDOUT_DRAWS]
    holdout = evaluated[-_HOLDOUT_DRAWS:]
    development_report = _scope(development)
    holdout_report = _scope(holdout)
    development_mkrd = development_report["strategies"]["mkrd"]["totalHits"]
    development_mksp = development_report["strategies"]["mksp"]["totalHits"]
    holdout_mkrd = holdout_report["strategies"]["mkrd"]["totalHits"]
    holdout_mksp = holdout_report["strategies"]["mksp"]["totalHits"]
    development_blend = development_report["mkspMkrdRankBlend"]["totalHits"]
    holdout_blend = holdout_report["mkspMkrdRankBlend"]["totalHits"]
    return {
        "dataset": str(dataset_path.resolve()),
        "drawCount": len(draws.draws),
        "featureWeights": {
            "relativeShape": 0.50,
            "coverage": 0.20,
            "uniformity": 0.10,
            "entropy": 0.10,
            "centerBalance": 0.10,
        },
        "development": development_report,
        "holdout": holdout_report,
        "fullHistory": _scope(evaluated),
        "promotion": {
            "standalonePassed": (
                development_mkrd > development_mksp
                and holdout_mkrd >= holdout_mksp
            ),
            "rankBlendAddedValue": (
                development_blend > development_mksp
                and holdout_blend >= holdout_mksp
            ),
            "rule": (
                "Development hits must exceed MKSP and latest-250 "
                "holdout hits must not trail MKSP."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/lotto_results.yaml"),
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
