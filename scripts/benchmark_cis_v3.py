"""Benchmark champion-preserving CIS v3 without exposing holdout outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from rand_ai import PredictionSuite, load_lotto_results_yaml
from rand_ai.cis import (
    CIS_V3_EXPERT_IDS,
    ChampionCis,
    CisConfig,
    cis_configurations,
    ranking_correlations,
)
from rand_ai.strategy_prediction import build_prediction_suites

_WARM_UP_DRAWS = 320
_VALIDATION_DRAWS = 200
_HOLDOUT_DRAWS = 250


@dataclass(frozen=True, slots=True)
class _Run:
    config: CisConfig
    hits: tuple[int, ...]
    champion_hits: tuple[int, ...]
    overlaps: tuple[int, ...]
    correction_counts: tuple[int, ...]
    shadow_deltas: tuple[int, ...]


def _summary(hits: Sequence[int]) -> dict[str, Any]:
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


def _paired(candidate: Sequence[int], baseline: Sequence[int]) -> dict[str, Any]:
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


def _strategies(suite: PredictionSuite) -> dict[str, tuple[int, ...]]:
    return {
        strategy.strategy_id: tuple(item.number for item in strategy.numbers)
        for strategy in suite.strategies
    }


def _run_config(
    frames: Sequence[dict[str, tuple[int, ...]]],
    actuals: Sequence[set[int]],
    correlations: Sequence[dict[tuple[str, str], float]],
    config: CisConfig,
) -> _Run:
    engine = ChampionCis(config)
    hits: list[int] = []
    champion_hits: list[int] = []
    overlaps: list[int] = []
    correction_counts: list[int] = []
    shadow_deltas: list[int] = []
    for rankings, actual, frame_correlations in zip(
        frames,
        actuals,
        correlations,
    ):
        prediction = engine.predict(rankings)
        top = set(prediction.top_numbers)
        champion_top = set(prediction.champion_top)
        shadow_top = set(prediction.shadow_top)
        hits.append(len(actual.intersection(top)))
        champion_hits.append(len(actual.intersection(champion_top)))
        overlaps.append(len(top.intersection(champion_top)))
        correction_counts.append(prediction.applied_correction_count)
        if prediction.corrections:
            shadow_deltas.append(
                len(actual.intersection(shadow_top))
                - len(actual.intersection(champion_top))
            )
        engine.observe(actual, frame_correlations)
    return _Run(
        config=config,
        hits=tuple(hits),
        champion_hits=tuple(champion_hits),
        overlaps=tuple(overlaps),
        correction_counts=tuple(correction_counts),
        shadow_deltas=tuple(shadow_deltas),
    )


def _slice(values: Sequence[int], scope: str) -> Sequence[int]:
    validation_start = _WARM_UP_DRAWS
    holdout_start = validation_start + _VALIDATION_DRAWS
    if scope == "validation":
        return values[validation_start:holdout_start]
    if scope == "holdout":
        return values[holdout_start:]
    raise ValueError(f"Unknown benchmark scope: {scope}")


def _select_candidate(runs: Sequence[_Run]) -> _Run:
    return max(
        enumerate(runs),
        key=lambda item: (
            sum(_slice(item[1].hits, "validation")),
            -sum(_slice(item[1].correction_counts, "validation")),
            sum(_slice(item[1].overlaps, "validation")),
            -item[0],
        ),
    )[1]


def _select_champion_config(runs: Sequence[_Run]) -> _Run:
    return max(
        enumerate(runs),
        key=lambda item: (
            sum(_slice(item[1].champion_hits, "validation")),
            -item[0],
        ),
    )[1]


def benchmark(dataset_path: Path) -> dict[str, Any]:
    draws = load_lotto_results_yaml(dataset_path)
    draws.prepare_predictions()
    strategy_ids = (*CIS_V3_EXPERT_IDS, "cis")
    suites = [
        suite
        for suite in build_prediction_suites(
            draws.draws,
            enabled_strategy_ids=strategy_ids,
        )
        if suite.actual_numbers
    ]
    expected = _WARM_UP_DRAWS + _VALIDATION_DRAWS + _HOLDOUT_DRAWS
    if len(suites) != expected:
        raise ValueError(
            f"CIS v3 benchmark requires {expected} evaluated draws; "
            f"received {len(suites)}"
        )
    all_rankings = [_strategies(suite) for suite in suites]
    frames = [
        {
            strategy_id: rankings[strategy_id]
            for strategy_id in CIS_V3_EXPERT_IDS
        }
        for rankings in all_rankings
    ]
    actuals = [set(suite.actual_numbers) for suite in suites]
    correlations = [ranking_correlations(frame) for frame in frames]
    runs = tuple(
        _run_config(frames, actuals, correlations, config)
        for config in cis_configurations()
    )
    candidate = _select_candidate(runs)
    champion = _select_champion_config(runs)
    fixed_hits = {
        strategy_id: tuple(
            len(actual.intersection(rankings[strategy_id][:6]))
            for rankings, actual in zip(all_rankings, actuals)
        )
        for strategy_id in CIS_V3_EXPERT_IDS
    }
    legacy_hits = tuple(
        len(actual.intersection(rankings["cis"][:6]))
        for rankings, actual in zip(all_rankings, actuals)
    )

    scopes: dict[str, Any] = {}
    strongest: dict[str, tuple[str, Sequence[int]]] = {}
    for scope in ("validation", "holdout"):
        fixed_scope = {
            strategy_id: _slice(hits, scope)
            for strategy_id, hits in fixed_hits.items()
        }
        strongest_id, strongest_hits = max(
            fixed_scope.items(),
            key=lambda item: (sum(item[1]), -CIS_V3_EXPERT_IDS.index(item[0])),
        )
        strongest[scope] = (strongest_id, strongest_hits)
        candidate_hits = _slice(candidate.hits, scope)
        champion_hits = _slice(champion.champion_hits, scope)
        scopes[scope] = {
            "cisV3": _summary(candidate_hits),
            "adaptiveChampion": _summary(champion_hits),
            "strongestFixedExpert": {
                "id": strongest_id,
                **_summary(strongest_hits),
            },
            "currentCis": _summary(_slice(legacy_hits, scope)),
            "fixedExperts": {
                strategy_id: _summary(hits)
                for strategy_id, hits in fixed_scope.items()
            },
            "pairedVsAdaptiveChampion": _paired(candidate_hits, champion_hits),
            "pairedVsStrongestFixed": _paired(candidate_hits, strongest_hits),
            "averageChampionOverlap": mean(_slice(candidate.overlaps, scope)),
            "appliedCorrections": sum(
                _slice(candidate.correction_counts, scope)
            ),
        }

    validation_candidate = scopes["validation"]["cisV3"]["totalHits"]
    validation_champion = scopes["validation"]["adaptiveChampion"]["totalHits"]
    validation_fixed = scopes["validation"]["strongestFixedExpert"]["totalHits"]
    holdout_candidate = scopes["holdout"]["cisV3"]["totalHits"]
    holdout_champion = scopes["holdout"]["adaptiveChampion"]["totalHits"]
    holdout_fixed = scopes["holdout"]["strongestFixedExpert"]["totalHits"]
    passed = (
        validation_candidate > validation_champion
        and validation_candidate > validation_fixed
        and holdout_candidate >= holdout_champion
        and holdout_candidate >= holdout_fixed
    )
    return {
        "dataset": str(dataset_path.resolve()),
        "evaluatedDraws": len(suites),
        "split": {
            "warmUp": _WARM_UP_DRAWS,
            "validation": _VALIDATION_DRAWS,
            "holdout": _HOLDOUT_DRAWS,
        },
        "selectedConfig": asdict(candidate.config),
        "adaptiveChampionConfig": {
            "recentWindow": champion.config.recent_window,
            "recentWeight": champion.config.recent_weight,
        },
        "scopes": scopes,
        "shadow": {
            "evaluatedCorrections": len(candidate.shadow_deltas),
            "netGain": sum(candidate.shadow_deltas),
        },
        "promotion": {
            "passed": passed,
            "rule": (
                "CIS v3 must beat the adaptive champion and strongest fixed "
                "expert on validation, then trail neither on holdout."
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
