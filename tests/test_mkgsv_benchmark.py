"""Test Markov Gap-Space Vector benchmark mechanics."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.benchmark_mkgsv as benchmark_module
from rand_ai import Draw, Draws
from rand_ai.mkgsv import MkgsvConfig, MkgsvModel
from scripts.benchmark_mkgsv import (
    ConfigRun,
    benchmark,
    combined_top_numbers,
    hit_summary,
    paired_summary,
    passes_promotion,
    run_configuration,
    scope_slice,
    select_configuration,
)


def _scope(validation: list[int], holdout: list[int] | None = None) -> tuple[int, ...]:
    return tuple([0] * 320 + validation + (holdout or [0] * 250))


def _run(
    config: MkgsvConfig,
    validation_hits: list[int],
    validation_brier: list[float],
) -> ConfigRun:
    return ConfigRun(
        config=config,
        hits=_scope(validation_hits),
        brier_scores=tuple([1.0] * 320 + validation_brier + [1.0] * 250),
        model=MkgsvModel(config),
    )


def test_hit_and_paired_summaries_cover_empty_and_populated_inputs() -> None:
    assert hit_summary([0, 1, 2, 3]) == {
        "evaluatedDraws": 4,
        "totalHits": 6,
        "averageHitsPerDraw": 1.5,
        "zeroHits": 1,
        "oneHit": 1,
        "twoOrMoreHits": 2,
    }
    assert hit_summary([])["averageHitsPerDraw"] == 0
    paired = paired_summary([2, 1, 0], [1, 1, 1])
    assert paired["candidateWins"] == 1
    assert paired["ties"] == 1
    assert paired["baselineWins"] == 1
    assert paired_summary([], [])["meanHitDifference"] == 0


def test_configuration_selection_uses_hits_brier_smoothing_and_grid_order() -> None:
    weaker = _run(MkgsvConfig(8, 4, 2), [0] * 200, [0.1] * 200)
    less_calibrated = _run(MkgsvConfig(8, 4, 2), [1] * 200, [0.2] * 200)
    less_smoothed = _run(MkgsvConfig(8, 4, 2), [1] * 200, [0.1] * 200)
    selected = _run(MkgsvConfig(64, 32, 16), [1] * 200, [0.1] * 200)
    identical_later = _run(MkgsvConfig(64, 32, 16), [1] * 200, [0.1] * 200)

    assert select_configuration(
        (weaker, less_calibrated, less_smoothed, selected, identical_later)
    ) is selected
    assert len(scope_slice(selected.hits, "validation")) == 200
    assert len(scope_slice(selected.hits, "holdout")) == 250
    with pytest.raises(ValueError, match="Unknown benchmark scope"):
        scope_slice(selected.hits, "future")


def test_promotion_requires_a_validation_win_and_holdout_noninferiority() -> None:
    assert passes_promotion(101, 100, 100, 100)
    assert not passes_promotion(100, 100, 101, 100)
    assert not passes_promotion(101, 100, 99, 100)


def test_run_configuration_scores_each_next_draw_and_combined_fallback() -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 3, 4, 5, 6))
    draws.add(Draw(2, 3, 4, 5, 6, 7))
    run = run_configuration(draws.draws, MkgsvConfig(8, 4, 2))

    assert len(run.hits) == 1
    assert len(run.brier_scores) == 1
    assert 0 <= run.brier_scores[0] <= 1
    assert combined_top_numbers(draws.draws[0]) == ()
    draws.prepare_predictions()
    assert len(combined_top_numbers(draws.draws[0])) == 6


def test_benchmark_rejects_a_dataset_without_the_fixed_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = Draws()
    draws.add(Draw())
    draws.add(Draw(2, 3, 4, 5, 6, 7))
    monkeypatch.setattr(
        benchmark_module,
        "load_lotto_results_yaml",
        lambda _path: draws,
    )

    with pytest.raises(ValueError, match="requires 770 evaluated draws"):
        benchmark(Path("unused.yaml"))
