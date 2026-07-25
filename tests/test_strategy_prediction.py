"""Test the Python-calculated named prediction strategy suite."""

import pytest

from rand_ai import Draw, Draws, PredictionSuite, StrategyPrediction
from rand_ai.strategy_prediction import (
    _StrategyState,
    _proximity_bucket,
    build_prediction_suites,
)


def test_builds_nine_named_rankings_and_reports_progress() -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.add(Draw(3, 6, 12, 22, 36, 47))
    draws.add(Draw(1, 9, 18, 27, 38, 45))
    draws.prepare_predictions()
    progress: list[tuple[int, int]] = []

    suites = build_prediction_suites(
        draws.draws,
        history_start=1,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert len(suites) == 2
    assert isinstance(suites[0], PredictionSuite)
    assert suites[0].actual_numbers == (1, 9, 18, 27, 38, 45)
    assert [strategy.name for strategy in suites[-1].strategies] == [
        "Prox",
        "Fresh",
        "EMD",
        "Rand",
        "Entr",
        "Mark",
        "Baye",
        "SVC",
        "TBL",
    ]
    assert all(
        isinstance(strategy, StrategyPrediction)
        and len(strategy.numbers) == 49
        and tuple(item.rank for item in strategy.numbers) == tuple(range(1, 50))
        and strategy.top_numbers
        == tuple(item.number for item in strategy.numbers[:6])
        for strategy in suites[-1].strategies
    )
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_builds_only_selected_strategy_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.add(Draw(3, 6, 12, 22, 36, 47))
    draws.prepare_predictions()

    def unexpected_emd(
        _state: _StrategyState,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raise AssertionError("disabled EMD strategy was calculated")

    monkeypatch.setattr(_StrategyState, "_earth_mover_scores", unexpected_emd)
    suites = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=("freshness", "entropy"),
    )

    assert [strategy.strategy_id for strategy in suites[-1].strategies] == [
        "freshness",
        "entropy",
    ]


def test_allows_every_strategy_plugin_to_be_disabled() -> None:
    suites = build_prediction_suites(
        (Draw(),),
        enabled_strategy_ids=(),
    )

    assert suites[0].strategies == ()


@pytest.mark.parametrize(
    ("distance", "bucket"),
    ((1, 0), (2, 1), (4, 2), (7, 3), (12, 4), (20, 5)),
)
def test_maps_every_proximity_distance_band(distance: int, bucket: int) -> None:
    assert _proximity_bucket(distance) == bucket


@pytest.mark.parametrize(
    ("distance", "label"),
    (
        (0, "Overlap"),
        (2, "Near"),
        (4, "Close"),
        (7, "Middle"),
        (10, "Far"),
        (14, "Distant"),
    ),
)
def test_maps_every_earth_mover_distance_band(
    distance: float,
    label: str,
) -> None:
    assert _StrategyState._earth_mover_bucket(distance) == label


def test_handles_empty_earth_mover_history_and_vectors() -> None:
    state = _StrategyState()

    scores, details = state._earth_mover_scores()

    assert scores == {number: 0 for number in range(1, 50)}
    assert details == {}
    assert state._earth_mover_distance((), ()) == 0


def test_defensive_missing_ranking_and_unprepared_draw_errors() -> None:
    state = _StrategyState()
    state.prior_rankings["freshness"] = []
    assert state._rank_score("freshness", 49) == 0

    with pytest.raises(ValueError, match="must be prepared"):
        build_prediction_suites((Draw(),))
