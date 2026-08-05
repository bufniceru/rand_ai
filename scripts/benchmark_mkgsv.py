"""Benchmark Markov Gap-Space Vector against related fixed baselines."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from rand_ai import load_lotto_results_yaml
from rand_ai.draw import Draw
from rand_ai.mkgsv import (
    BASE_HIT_RATE,
    NUMBER_COUNT,
    MkgsvConfig,
    MkgsvModel,
    mkgsv_configurations,
)
from rand_ai.strategy_prediction import build_prediction_suites

WARM_UP_DRAWS = 320
VALIDATION_DRAWS = 200
HOLDOUT_DRAWS = 250
RELATED_STRATEGY_IDS = ("markov100", "freshness", "proximity")


@dataclass(frozen=True, slots=True)
class ConfigRun:
    config: MkgsvConfig
    hits: tuple[int, ...]
    brier_scores: tuple[float, ...]
    model: MkgsvModel


def hit_summary(hits: Sequence[int]) -> dict[str, int | float]:
    distribution = Counter(hits)
    return {
        "evaluatedDraws": len(hits),
        "totalHits": sum(hits),
        "averageHitsPerDraw": mean(hits) if hits else 0.0,
        "zeroHits": distribution[0],
        "oneHit": distribution[1],
        "twoOrMoreHits": sum(
            count for value, count in distribution.items() if value >= 2
        ),
    }


def paired_summary(
    candidate: Sequence[int],
    baseline: Sequence[int],
) -> dict[str, Any]:
    differences = [left - right for left, right in zip(candidate, baseline)]
    average = mean(differences) if differences else 0.0
    standard_error = (
        stdev(differences) / len(differences) ** 0.5
        if len(differences) > 1
        else 0.0
    )
    return {
        "candidateWins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "baselineWins": sum(value < 0 for value in differences),
        "meanHitDifference": average,
        "meanDifference95Interval": [
            average - 1.96 * standard_error,
            average + 1.96 * standard_error,
        ],
    }


def scope_slice[T](values: Sequence[T], scope: str) -> Sequence[T]:
    holdout_start = WARM_UP_DRAWS + VALIDATION_DRAWS
    if scope == "validation":
        return values[WARM_UP_DRAWS:holdout_start]
    if scope == "holdout":
        return values[holdout_start:]
    raise ValueError(f"Unknown benchmark scope: {scope}")


def run_configuration(draws: Sequence[Any], config: MkgsvConfig) -> ConfigRun:
    model = MkgsvModel(config)
    hits: list[int] = []
    brier_scores: list[float] = []
    for index, draw in enumerate(draws):
        drawn = {ball.value for ball in draw.balls}
        model.train(drawn)
        model.remember(drawn)
        if index + 1 >= len(draws):
            continue
        scores = model.scores()
        ranking = sorted(
            scores,
            key=lambda number: (
                -scores[number].probability,
                -scores[number].gap,
                number,
            ),
        )
        actual = {ball.value for ball in draws[index + 1].balls}
        hits.append(len(actual.intersection(ranking[:6])))
        brier_scores.append(
            sum(
                (
                    scores[number].probability
                    - float(number in actual)
                )
                ** 2
                for number in range(1, NUMBER_COUNT + 1)
            )
            / NUMBER_COUNT
        )
    return ConfigRun(config, tuple(hits), tuple(brier_scores), model)


def combined_top_numbers(draw: Draw) -> tuple[int, ...]:
    prediction = draw.prediction
    return () if prediction is None else prediction.top_numbers


def select_configuration(runs: Sequence[ConfigRun]) -> ConfigRun:
    """Select on validation hits, calibration, smoothing, then grid order."""
    return max(
        enumerate(runs),
        key=lambda item: (
            sum(scope_slice(item[1].hits, "validation")),
            -mean(scope_slice(item[1].brier_scores, "validation")),
            sum(asdict(item[1].config).values()),
            -item[0],
        ),
    )[1]


def passes_promotion(
    validation_candidate: int,
    validation_baseline: int,
    holdout_candidate: int,
    holdout_baseline: int,
) -> bool:
    """Apply the fixed validation-win and holdout-noninferiority gate."""
    return (
        validation_candidate > validation_baseline
        and holdout_candidate >= holdout_baseline
    )


def benchmark(dataset_path: Path) -> dict[str, Any]:
    draws = load_lotto_results_yaml(dataset_path)
    draws.prepare_predictions()
    evaluated_count = len(draws.draws) - 1
    expected = WARM_UP_DRAWS + VALIDATION_DRAWS + HOLDOUT_DRAWS
    if evaluated_count != expected:
        raise ValueError(
            f"MKGSV benchmark requires {expected} evaluated draws; "
            f"received {evaluated_count}"
        )
    runs = tuple(
        run_configuration(draws.draws, config)
        for config in mkgsv_configurations()
    )
    selected = select_configuration(runs)

    suites = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=RELATED_STRATEGY_IDS,
    )[:-1]
    baseline_hits: dict[str, tuple[int, ...]] = {}
    for strategy_id in RELATED_STRATEGY_IDS:
        baseline_hits[strategy_id] = tuple(
            len(
                set(suite.actual_numbers).intersection(
                    next(
                        strategy.top_numbers
                        for strategy in suite.strategies
                        if strategy.strategy_id == strategy_id
                    )
                )
            )
            for suite in suites
        )
    baseline_hits["combined"] = tuple(
        len(
            set(suite.actual_numbers).intersection(
                combined_top_numbers(draws.draws[index])
            )
        )
        for index, suite in enumerate(suites)
    )

    scopes: dict[str, Any] = {}
    strongest_by_scope: dict[str, tuple[str, Sequence[int]]] = {}
    for scope in ("validation", "holdout"):
        candidate_hits = scope_slice(selected.hits, scope)
        scoped_baselines = {
            strategy_id: scope_slice(hits, scope)
            for strategy_id, hits in baseline_hits.items()
        }
        strongest_id, strongest_hits = max(
            scoped_baselines.items(),
            key=lambda item: (sum(item[1]), item[0]),
        )
        strongest_by_scope[scope] = strongest_id, strongest_hits
        scopes[scope] = {
            "mkgsv": {
                **hit_summary(candidate_hits),
                "brierScore": mean(scope_slice(selected.brier_scores, scope)),
            },
            "strongestRelatedBaseline": {
                "id": strongest_id,
                **hit_summary(strongest_hits),
            },
            "relatedBaselines": {
                strategy_id: hit_summary(hits)
                for strategy_id, hits in scoped_baselines.items()
            },
            "pairedVsStrongest": paired_summary(candidate_hits, strongest_hits),
            "theoreticalRandomHits": len(candidate_hits) * 6 * BASE_HIT_RATE,
        }

    validation_candidate = scopes["validation"]["mkgsv"]["totalHits"]
    validation_baseline = scopes["validation"]["strongestRelatedBaseline"][
        "totalHits"
    ]
    holdout_candidate = scopes["holdout"]["mkgsv"]["totalHits"]
    holdout_baseline = scopes["holdout"]["strongestRelatedBaseline"][
        "totalHits"
    ]
    passed = passes_promotion(
        validation_candidate,
        validation_baseline,
        holdout_candidate,
        holdout_baseline,
    )
    return {
        "dataset": dataset_path.as_posix(),
        "evaluatedDraws": evaluated_count,
        "split": {
            "warmUp": WARM_UP_DRAWS,
            "validation": VALIDATION_DRAWS,
            "holdout": HOLDOUT_DRAWS,
        },
        "selectedConfig": asdict(selected.config),
        "stateSupport": selected.model.state_support_distribution(),
        "scopes": scopes,
        "promotion": {
            "passed": passed,
            "defaultEnabled": passed,
            "rule": (
                "MKGSV must beat the strongest related baseline on validation "
                "and trail it by no hits on holdout."
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
