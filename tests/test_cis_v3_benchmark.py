"""Test deterministic CIS v3 benchmark selection and summaries."""

from __future__ import annotations

import pytest

from rand_ai.cis import CisConfig, ranking_correlations
from scripts.benchmark_cis_v3 import (
    _Run,
    _paired,
    _run_config,
    _select_candidate,
    _select_champion_config,
    _slice,
    _summary,
)


def _config(replacements: int) -> CisConfig:
    return CisConfig(40, 0.8, 0.8, 0.2, 0.05, replacements)


def _scope(validation: list[int], holdout: list[int] | None = None) -> tuple[int, ...]:
    return tuple([0] * 320 + validation + (holdout or [0] * 250))


def _run(
    config: CisConfig,
    validation_hits: list[int],
    *,
    validation_corrections: list[int] | None = None,
    validation_overlaps: list[int] | None = None,
    champion_hits: list[int] | None = None,
) -> _Run:
    zeros = [0] * 200
    return _Run(
        config=config,
        hits=_scope(validation_hits),
        champion_hits=_scope(champion_hits or zeros),
        overlaps=_scope(validation_overlaps or zeros),
        correction_counts=_scope(validation_corrections or zeros),
        shadow_deltas=(),
    )


def test_summaries_and_paired_comparison_report_hit_behavior() -> None:
    assert _summary([0, 1, 2, 3]) == {
        "evaluatedDraws": 4,
        "totalHits": 6,
        "averageHitsPerDraw": 1.5,
        "zeroHits": 1,
        "oneHit": 1,
        "twoOrMoreHits": 2,
    }
    assert _summary([])["averageHitsPerDraw"] == 0
    paired = _paired([2, 1, 0], [1, 1, 1])
    assert paired["candidateWins"] == 1
    assert paired["ties"] == 1
    assert paired["baselineWins"] == 1
    assert _paired([], [])["meanHitDifference"] == 0


def test_selection_uses_hits_then_corrections_overlap_and_grid_order() -> None:
    one = _run(
        _config(1),
        [1] * 200,
        validation_corrections=[1] * 200,
        validation_overlaps=[5] * 200,
        champion_hits=[0] * 200,
    )
    two = _run(
        _config(2),
        [1] * 200,
        validation_corrections=[0] * 200,
        validation_overlaps=[6] * 200,
        champion_hits=[1] * 200,
    )

    assert _select_candidate((one, two)) is two
    assert _select_champion_config((one, two)) is two
    assert len(_slice(two.hits, "validation")) == 200
    assert len(_slice(two.hits, "holdout")) == 250
    with pytest.raises(ValueError, match="Unknown benchmark scope"):
        _slice(two.hits, "future")


def test_run_config_uses_the_same_engine_for_candidate_and_champion() -> None:
    ascending = tuple(range(1, 50))
    descending = tuple(reversed(ascending))
    frames = [{"freshness": ascending, "proximity": descending}]
    actuals = [{1, 2, 3, 4, 5, 6}]
    correlations = [ranking_correlations(frames[0])]

    run = _run_config(frames, actuals, correlations, _config(1))

    assert run.hits == (6,)
    assert run.champion_hits == (6,)
    assert run.overlaps == (6,)
    assert run.correction_counts == (0,)

