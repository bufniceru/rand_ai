"""Test the Python-calculated named prediction strategy suite."""

import pytest

from rand_ai import Draw, Draws, PredictionSuite, StrategyPrediction
from rand_ai.strategy_prediction import (
    _BASE_PROBABILITY,
    _MKFR_MAX_ORDER,
    _MKFR_MIN_CONTEXT_SUPPORT,
    _MKFR_PRIOR_STRENGTH,
    _StrategyState,
    _proximity_bucket,
    build_prediction_suites,
)


def test_builds_ten_named_rankings_and_reports_progress() -> None:
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
        "MKFR",
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

    def unexpected_mkfr(
        _state: _StrategyState,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raise AssertionError("disabled MKFR strategy was calculated")

    monkeypatch.setattr(_StrategyState, "_earth_mover_scores", unexpected_emd)
    monkeypatch.setattr(_StrategyState, "_mkfr_scores", unexpected_mkfr)
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


def test_rejects_unknown_strategy_plugin() -> None:
    with pytest.raises(ValueError, match="Unknown prediction strategy"):
        build_prediction_suites(
            (Draw(),),
            enabled_strategy_ids=("unknown",),
        )


def test_mkfr_learns_joint_binary_context_transitions_for_each_order() -> None:
    state = _StrategyState(("mkfr",))
    for outcome in (1, 0, 1):
        drawn = {1} if outcome else set()
        state.train(drawn)
        state.remember(drawn)

    assert state.mkfr_transitions[1][0][1] == [1, 0]
    assert state.mkfr_transitions[1][0][0] == [0, 1]
    assert state.mkfr_transitions[1][1][0b10] == [0, 1]
    assert list(state.mkfr_histories[1]) == [1, 0, 1]
    assert list(state.mkfr_histories[2]) == [0, 0, 0]


def test_mkfr_backs_off_from_an_unsupported_longer_context() -> None:
    state = _StrategyState(("mkfr",))
    state.draw_count = 100
    state.appearances[1] = 20
    state.mkfr_histories[1].extend((1, 0))
    state.mkfr_transitions[1][0][0] = [8, 2]
    state.mkfr_transitions[1][1][0b10] = [
        _MKFR_MIN_CONTEXT_SUPPORT - 1,
        0,
    ]
    baseline = (
        20 + _MKFR_PRIOR_STRENGTH * _BASE_PROBABILITY
    ) / (100 + _MKFR_PRIOR_STRENGTH)
    expected_order_one = (
        2 + _MKFR_PRIOR_STRENGTH * baseline
    ) / (10 + _MKFR_PRIOR_STRENGTH)

    probability, support, selected_order = state._mkfr_probability(1)

    assert probability == pytest.approx(expected_order_one)
    assert support == 10
    assert selected_order == 1

    state.mkfr_transitions[1][1][0b10] = [6, 2]
    expected_order_two = (
        2 + _MKFR_PRIOR_STRENGTH * expected_order_one
    ) / (8 + _MKFR_PRIOR_STRENGTH)

    probability, support, selected_order = state._mkfr_probability(1)

    assert probability == pytest.approx(expected_order_two)
    assert support == 8
    assert selected_order == 2


def test_mkfr_supports_joint_contexts_through_order_twenty() -> None:
    state = _StrategyState(("mkfr",))
    state.mkfr_histories[1].extend([1] * _MKFR_MAX_ORDER)
    for order in range(1, _MKFR_MAX_ORDER + 1):
        context = (1 << order) - 1
        state.mkfr_transitions[1][order - 1][context] = [
            0,
            _MKFR_MIN_CONTEXT_SUPPORT,
        ]

    probability, support, selected_order = state._mkfr_probability(1)
    scores, details = state._mkfr_scores()

    assert probability > _BASE_PROBABILITY
    assert support == _MKFR_MIN_CONTEXT_SUPPORT
    assert selected_order == _MKFR_MAX_ORDER
    assert scores[1] > scores[2]
    assert details[1][3] == (
        f"Order {_MKFR_MAX_ORDER}/{_MKFR_MAX_ORDER}: "
        f"{'1' * _MKFR_MAX_ORDER}"
    )
    assert details[1][4] == (
        f"Context support {_MKFR_MIN_CONTEXT_SUPPORT}"
    )


def test_mkfr_ranks_transition_lift_above_each_numbers_baseline() -> None:
    state = _StrategyState(("mkfr",))
    state.draw_count = 100
    state.appearances[1] = 20
    state.appearances[2] = 10
    state.mkfr_histories[1].append(0)
    state.mkfr_histories[2].append(0)
    state.mkfr_transitions[1][0][0] = [79, 21]
    state.mkfr_transitions[2][0][0] = [85, 15]

    scores, details = state._mkfr_scores()

    assert state._mkfr_probability(1)[0] > state._mkfr_probability(2)[0]
    assert scores[2] > scores[1]
    assert details[1][2].startswith("Transition lift +")
    assert details[2][2].startswith("Transition lift +")


def test_mkfr_uses_prior_for_empty_history_and_truncates_to_twenty_draws() -> None:
    state = _StrategyState(("mkfr",))

    assert state._mkfr_probability(1) == (_BASE_PROBABILITY, 0, 0)

    outcomes = tuple(index % 2 for index in range(25))
    for outcome in outcomes:
        drawn = {1} if outcome else set()
        state.train(drawn)
        state.remember(drawn)

    assert list(state.mkfr_histories[1]) == list(outcomes[-_MKFR_MAX_ORDER:])


def test_mkfr_prediction_does_not_learn_from_a_future_draw() -> None:
    prefix = (
        Draw(1, 2, 8, 17, 31, 49),
        Draw(3, 6, 12, 22, 36, 47),
        Draw(1, 9, 18, 27, 38, 45),
    )
    without_future = Draws()
    with_future = Draws()
    for draw in prefix:
        without_future.add(draw)
        with_future.add(
            Draw(*(ball.value for ball in draw.balls))
        )
    with_future.add(Draw(4, 11, 20, 29, 37, 48))
    without_future.prepare_predictions()
    with_future.prepare_predictions()

    prefix_strategy = build_prediction_suites(
        without_future.draws,
        enabled_strategy_ids=("mkfr",),
    )[-1].strategies[0]
    future_strategy = build_prediction_suites(
        with_future.draws,
        enabled_strategy_ids=("mkfr",),
    )[-2].strategies[0]

    assert prefix_strategy.top_numbers == future_strategy.top_numbers
    assert prefix_strategy.numbers == future_strategy.numbers


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
