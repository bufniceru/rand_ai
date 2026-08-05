"""Test guarded MKGSV v2 benchmark mechanics."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.benchmark_mkgsv as benchmark_module
from rand_ai import Draw, Draws
from rand_ai.mkgsv import MkgsvConfig, MkgsvModel
from scripts.benchmark_mkgsv import (
    ConfigRun,
    Markov100Anchor,
    benchmark,
    combined_top_numbers,
    hit_summary,
    paired_summary,
    passes_promotion,
    render_markdown,
    run_configuration,
    scope_slice,
    select_configuration,
)


def _scope[T](validation: list[T], holdout: list[T] | None = None) -> tuple[T, ...]:
    fallback = [validation[0]] * 250 if holdout is None else holdout
    return tuple([validation[0]] * 320 + validation + fallback)


def _run(
    config: MkgsvConfig,
    validation_hits: list[int],
    validation_brier: list[float],
    *,
    proposals: list[bool] | None = None,
    overlap: list[int] | None = None,
) -> ConfigRun:
    hits = _scope(validation_hits)
    brier = _scope(validation_brier)
    return ConfigRun(
        config=config,
        gated_hits=hits,
        shadow_hits=hits,
        base_hits=tuple(0 for _ in hits),
        gated_brier_scores=brier,
        shadow_brier_scores=brier,
        base_brier_scores=brier,
        champion_overlaps=_scope(overlap or [6] * 200),
        proposals=_scope(proposals or [False] * 200),
        active_predictions=tuple(False for _ in hits),
        replacements=tuple(None for _ in hits),
        model=MkgsvModel(config),
    )


def test_summaries_and_scope_validation() -> None:
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
    assert (paired["candidateWins"], paired["ties"], paired["baselineWins"]) == (
        1,
        1,
        1,
    )
    assert paired_summary([], [])["meanHitDifference"] == 0
    values = _scope([1] * 200, [2] * 250)
    assert set(scope_slice(values, "validation")) == {1}
    assert set(scope_slice(values, "holdout")) == {2}
    with pytest.raises(ValueError, match="Unknown benchmark scope"):
        scope_slice(values, "future")


def test_configuration_selection_uses_every_tie_break_in_order() -> None:
    weak = _run(MkgsvConfig(32, 128, 256, "fresh", 0.0025), [0] * 200, [0.1] * 200)
    poor_brier = _run(MkgsvConfig(32, 128, 256, "fresh", 0.0025), [1] * 200, [0.2] * 200)
    many_proposals = _run(
        MkgsvConfig(32, 128, 256, "fresh", 0.0025),
        [1] * 200,
        [0.1] * 200,
        proposals=[True] * 200,
    )
    low_overlap = _run(
        MkgsvConfig(32, 128, 256, "fresh", 0.0025),
        [1] * 200,
        [0.1] * 200,
        overlap=[5] * 200,
    )
    selected = _run(
        MkgsvConfig(64, 256, 512, "combined", 0.005),
        [1] * 200,
        [0.1] * 200,
    )
    identical_later = _run(
        MkgsvConfig(64, 256, 512, "combined", 0.005),
        [1] * 200,
        [0.1] * 200,
    )
    assert select_configuration(
        (weak, poor_brier, many_proposals, low_overlap, selected, identical_later)
    ) is selected


def test_promotion_requires_all_four_conditions() -> None:
    assert passes_promotion(101, 100, 100, 100, 1, 0)
    assert not passes_promotion(100, 100, 101, 100, 1, 1)
    assert not passes_promotion(101, 100, 99, 100, 1, 0)
    assert not passes_promotion(101, 100, 100, 100, 0, 0)
    assert not passes_promotion(101, 100, 100, 100, 1, -1)


def test_anchor_and_configuration_match_production_markov() -> None:
    draws = Draws()
    for index in range(5):
        draws.add(Draw(*range(index + 1, index + 7)))
    draws.prepare_predictions()
    run = run_configuration(
        draws.draws,
        MkgsvConfig(32, 128, 256, "fresh", 0.0025),
    )
    suites = benchmark_module.build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=("markov100",),
    )[:-1]
    expected = tuple(
        len(
            set(suite.actual_numbers).intersection(
                suite.strategies[0].top_numbers
            )
        )
        for suite in suites
    )
    assert run.base_hits == expected
    assert len(run.gated_hits) == 4
    assert all(0 <= value <= 1 for value in run.gated_brier_scores)

    anchor = Markov100Anchor()
    assert anchor.gap(1) == 0
    anchor.train({1, 2, 3, 4, 5, 6})
    anchor.remember({1, 2, 3, 4, 5, 6})
    assert anchor.gap(1) == 0
    assert len(anchor.ranking(anchor.probabilities())) == 49


def test_combined_fallback_and_markdown_rendering() -> None:
    draw = Draw(1, 2, 3, 4, 5, 6)
    assert combined_top_numbers(draw) == ()
    draws = Draws()
    draws.add(draw)
    draws.add(Draw(2, 3, 4, 5, 6, 7))
    draws.prepare_predictions()
    assert len(combined_top_numbers(draws.draws[0])) == 6
    report = {
        "selectedConfig": {
            "single_strength": 64,
            "pair_strength": 256,
            "triple_strength": 512,
            "evidence_variant": "combined",
            "replacement_margin": 0.005,
        },
        "scopes": {
            scope: {
                "mkgsvGated": {"totalHits": 1, "brierScore": 0.1},
                "mkgsvRawShadow": {"totalHits": 2},
                "markov100Champion": {"totalHits": 1, "brierScore": 0.1},
                "gatedCorrectionNetGain": 0,
                "proposalCount": 1,
                "activePredictionCount": 0,
            }
            for scope in ("validation", "holdout")
        },
        "promotion": {"passed": False},
    }
    assert "**Failed.**" in render_markdown(report)
    report["promotion"]["passed"] = True
    assert "**Passed.**" in render_markdown(report)


def test_benchmark_rejects_a_dataset_without_the_fixed_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = Draws()
    draws.add(Draw())
    draws.add(Draw(2, 3, 4, 5, 6, 7))
    monkeypatch.setattr(benchmark_module, "load_lotto_results_yaml", lambda _path: draws)
    with pytest.raises(ValueError, match="requires 770 evaluated draws"):
        benchmark(Path("unused.yaml"))
