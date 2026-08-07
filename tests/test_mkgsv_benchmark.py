"""Test MKGSV v3 benchmark mechanics."""

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
    raw_validation: list[int],
    gated_validation: list[int],
    base_validation: list[int],
    brier: list[float],
    *,
    proposals: list[bool] | None = None,
    overlap: list[int] | None = None,
) -> ConfigRun:
    raw = _scope(raw_validation)
    gated = _scope(gated_validation)
    base = _scope(base_validation)
    briers = _scope(brier)
    return ConfigRun(
        config=config,
        gated_hits=gated,
        raw_hits=raw,
        base_hits=base,
        gated_brier_scores=briers,
        raw_brier_scores=briers,
        base_brier_scores=briers,
        champion_overlaps=_scope(overlap or [6] * 200),
        proposals=_scope(proposals or [False] * 200),
        active_predictions=tuple(False for _ in raw),
        replacements=tuple(None for _ in raw),
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


def test_configuration_selection_rejects_nonpositive_gain_and_uses_ties() -> None:
    config = MkgsvConfig(8, "singles", 0.05)
    rejected = _run(config, [1] * 200, [1] * 200, [1] * 200, [0.1] * 200)
    assert select_configuration((rejected,)) is None

    weak = _run(config, [2] + [1] * 199, [1] * 200, [1] * 200, [0.2] * 200)
    better_gated = _run(
        MkgsvConfig(24, "singles+doubles", 0.1),
        [2] + [1] * 199,
        [2] + [1] * 199,
        [1] * 200,
        [0.2] * 200,
    )
    better_brier = _run(
        MkgsvConfig(64, "all-with-transitions", 0.2),
        [2] + [1] * 199,
        [2] + [1] * 199,
        [1] * 200,
        [0.1] * 200,
    )
    identical_later = _run(
        better_brier.config,
        [2] + [1] * 199,
        [2] + [1] * 199,
        [1] * 200,
        [0.1] * 200,
    )
    assert select_configuration(
        (rejected, weak, better_gated, better_brier, identical_later)
    ) is better_brier


def test_promotion_requires_every_condition() -> None:
    passing = (101, 101, 100, 100, 100, 100, 1, 0, 30)
    assert passes_promotion(*passing)
    for index in range(len(passing)):
        values = list(passing)
        if index in {0, 1}:
            values[index] = 100
        elif index in {3, 4}:
            values[index] = 99
        elif index == 6:
            values[index] = 0
        elif index == 7:
            values[index] = -1
        elif index == 8:
            values[index] = 29
        else:
            continue
        assert not passes_promotion(*values)


def test_anchor_and_configuration_match_production_markov() -> None:
    draws = Draws()
    for index in range(5):
        draws.add(Draw(*range(index + 1, index + 7)))
    draws.prepare_predictions()
    run = run_configuration(
        draws.draws,
        MkgsvConfig(8, "singles", 0.05),
    )
    suites = benchmark_module.build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=("markov100",),
    )[:-1]
    expected = tuple(
        len(set(suite.actual_numbers).intersection(suite.strategies[0].top_numbers))
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
    scope = {
        "mkgsvGated": {"totalHits": 1, "brierScore": 0.1},
        "mkgsvRaw": {"totalHits": 2, "brierScore": 0.1},
        "markov100Champion": {"totalHits": 1, "brierScore": 0.1},
        "gatedCorrectionNetGain": 0,
        "rawCorrectionNetGain": 1,
        "proposalCount": 1,
        "activePredictionCount": 0,
    }
    report = {
        "selectedConfig": None,
        "scopes": {"validation": scope, "holdout": scope},
        "promotion": {"passed": False},
    }
    markdown = render_markdown(report)
    assert "Correction off" in markdown
    assert "**Failed.**" in markdown
    report["selectedConfig"] = {
        "prior_strength": 64,
        "motif_variant": "all-with-transitions",
        "influence": 0.2,
    }
    report["promotion"]["passed"] = True
    assert "**Passed.**" in render_markdown(report)


def test_benchmark_rejects_dataset_without_fixed_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = Draws()
    draws.add(Draw())
    draws.add(Draw(2, 3, 4, 5, 6, 7))
    monkeypatch.setattr(benchmark_module, "load_lotto_results_yaml", lambda _path: draws)
    with pytest.raises(ValueError, match="requires 770 evaluated draws"):
        benchmark(Path("unused.yaml"))
