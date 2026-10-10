"""Verify causal adaptation, weight bounds, and independent source forecasts."""

import pytest

from rand_ai import Draw, Draws
from rand_ai.adaptive_shape_recurrence import (
    SOURCE_IDS,
    RANDOM_HITS,
    AdaptiveShapeRecurrenceModel,
    bounded_weights,
)
from rand_ai.gui_bridge import DEFAULT_STRATEGY_IDS
from rand_ai.positional_shape_successor_v2 import PositionalShapeSuccessorV2Model
from rand_ai.strategy_prediction import (
    _STRATEGY_DEPENDENCIES,
    _StrategyState,
    _ranking_from_scores,
    build_prediction_suites,
)

ID = "adaptive_shape_recurrence_blend"


def rankings() -> dict[str, list[int]]:
    """Return complete deterministic rankings for all sources."""
    return {source: list(range(1, 50)) for source in SOURCE_IDS}


def test_priors_rank_conversion_and_confidence() -> None:
    model = AdaptiveShapeRecurrenceModel()
    assert model.effectiveness() == {
        source: (RANDOM_HITS, RANDOM_HITS) for source in SOURCE_IDS
    }
    assert model.weights(1) == pytest.approx(dict.fromkeys(SOURCE_IDS, 0.25))
    weights = model.weights(0)
    assert weights[SOURCE_IDS[-1]] == pytest.approx(1 / 7)
    assert weights["svc"] == pytest.approx(2 / 7)
    assert model.weights(0, use_confidence=False) == pytest.approx(
        dict.fromkeys(SOURCE_IDS, 0.25)
    )
    scores, details = model.predict(rankings(), 1)
    assert scores[1] == pytest.approx(1)
    assert scores[25] == pytest.approx(0.5)
    assert scores[49] == 0
    assert "weight 25.00%" in details[1][0]
    assert "0 completed forecasts" in details[1][5]
    assert "multiplier 1.000" in details[1][4]
    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            model.weights(invalid)


def test_completed_outcomes_are_consumed_once_and_recent_window_evicts() -> None:
    model = AdaptiveShapeRecurrenceModel()
    model.observe_completed(range(1, 7))
    assert model.evaluated_forecasts == 0
    model.predict(rankings(), 1)
    model.observe_completed(range(1, 7))
    model.observe_completed(range(1, 7))
    assert model.evaluated_forecasts == 1
    for source in SOURCE_IDS:
        assert model.effectiveness()[source] == pytest.approx(
            ((6 + 24 * RANDOM_HITS) / 25, (6 + 24 * RANDOM_HITS) / 25)
        )
    for _ in range(40):
        model.predict(rankings(), 1)
        model.observe_completed(range(7, 13))
    assert model.evaluated_forecasts == 41
    for source in SOURCE_IDS:
        assert len(model.recent_hits[source]) == 40
        assert sum(model.recent_hits[source]) == 0
        assert model.effectiveness()[source] == pytest.approx(
            ((6 + 24 * RANDOM_HITS) / 65, 24 * RANDOM_HITS / 64)
        )


def test_lifetime_recent_mix_and_pending_copy() -> None:
    model = AdaptiveShapeRecurrenceModel()
    model.evaluated_forecasts = 40
    for source, total in zip(SOURCE_IDS, (80, 40, 20, 10), strict=True):
        model.total_hits[source] = total
        model.recent_hits[source].extend([total // 40] * 40)
    effectiveness = model.effectiveness()
    expected = bounded_weights(
        {
            source: 0.75 * lifetime + 0.25 * recent
            for source, (lifetime, recent) in effectiveness.items()
        }
    )
    assert model.weights(1) == pytest.approx(expected)
    inputs = rankings()
    model.predict(inputs, 1)
    inputs["svc"].reverse()
    model.observe_completed(range(1, 7))
    assert model.total_hits["svc"] == 86


@pytest.mark.parametrize(
    "values",
    [(1, 1, 1, 1), (1000, 1, 1, 1), (0, 0, 0, 0), (1, 0, 0, 0), (1, 2, 1000, 900)],
)
def test_bounded_weights_redistribute_proportionally(values: tuple[int, ...]) -> None:
    qualities = dict(zip(SOURCE_IDS, values, strict=True))
    weights = bounded_weights(qualities)
    assert sum(weights.values()) == pytest.approx(1, abs=1e-10)
    assert all(0.1 <= weight <= 0.5 for weight in weights.values())
    free = [
        source
        for source in SOURCE_IDS
        if 0.100001 < weights[source] < 0.499999 and qualities[source] > 0
    ]
    if len(free) >= 2:
        assert weights[free[0]] / weights[free[1]] == pytest.approx(
            qualities[free[0]] / qualities[free[1]]
        )
    assert bounded_weights(qualities) == weights
    if values == (1000, 1, 1, 1):
        assert weights["svc"] == 0.5
        assert weights["emd"] == pytest.approx(1 / 6)
    with pytest.raises(ValueError):
        bounded_weights(dict.fromkeys(SOURCE_IDS, -1))


def test_rank_ties_use_existing_gap_and_number_order() -> None:
    inputs = rankings()
    inputs["svc"].reverse()
    inputs["emd"].reverse()
    scores, _ = AdaptiveShapeRecurrenceModel().predict(inputs, 1)
    assert scores == pytest.approx(dict.fromkeys(range(1, 50), 0.5))
    gaps = dict.fromkeys(range(1, 50), 0)
    gaps[49] = 10
    assert _ranking_from_scores(scores, gaps)[:6] == [49, 1, 2, 3, 4, 5]


def test_v2_confidence_is_read_only_and_updates_after_prediction() -> None:
    model = PositionalShapeSuccessorV2Model()
    assert model.last_confidence == 0
    for _ in range(9):
        model.observe((1, 8, 20, 30, 40, 49))
    model.predict()
    assert model.last_confidence == pytest.approx(0.5)
    with pytest.raises(AttributeError):
        setattr(model, "last_confidence", 1)


def test_hidden_dependencies_and_source_preservation() -> None:
    state = _StrategyState((ID,))
    assert state.enabled_strategy_ids == {ID, *SOURCE_IDS}
    assert ID not in DEFAULT_STRATEGY_IDS
    assert all(ID not in sources for sources in _STRATEGY_DEPENDENCIES.values())
    assert _StrategyState(("emd",)).adaptive_shape_recurrence is None
    draws = Draws()
    for numbers in (
        (1, 8, 20, 30, 40, 49),
        (2, 9, 21, 31, 41, 48),
        (3, 10, 22, 32, 42, 47),
    ):
        draws.add(Draw(*numbers))
    draws.prepare_predictions()
    existing = (
        "svc",
        "recurrence_dynamics",
        "emd",
        "svc_recurrence_hybrid",
        "emd_positional_shape_v2_hybrid",
        "positional_shape_successor",
    )
    before = build_prediction_suites(draws.draws, enabled_strategy_ids=existing)
    after = build_prediction_suites(draws.draws, enabled_strategy_ids=(*existing, ID))
    only = build_prediction_suites(draws.draws, enabled_strategy_ids=(ID,))
    for previous, combined, single in zip(before, after, only, strict=True):
        preserved = {
            s.strategy_id: s for s in combined.strategies if s.strategy_id != ID
        }
        assert preserved == {s.strategy_id: s for s in previous.strategies}
        assert [s.strategy_id for s in single.strategies] == [ID]
        assert single.strategies[0] == next(
            s for s in combined.strategies if s.strategy_id == ID
        )
    draws.add(Draw(4, 11, 23, 33, 43, 46))
    draws.prepare_predictions()
    future = build_prediction_suites(draws.draws, enabled_strategy_ids=(ID,))
    for prefix, extended in zip(only, future):
        assert prefix.strategies[0].numbers == extended.strategies[0].numbers
        assert prefix.strategies[0].top_numbers == extended.strategies[0].top_numbers
    latest = future[-1].strategies[0]
    assert len(latest.numbers) == len({p.number for p in latest.numbers}) == 49
    assert latest.top_numbers == tuple(p.number for p in latest.numbers[:6])
    assert latest.efficacy is not None and latest.efficacy.evaluated_draws == 3
    assert "3 completed forecasts" in latest.numbers[0].details[5]
