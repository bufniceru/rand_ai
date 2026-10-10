"""Verify joint shapes, successor support, and causal strategy integration."""

import numpy as np
import pytest

from rand_ai import Draw, Draws
from rand_ai.gui_bridge import DEFAULT_STRATEGY_IDS
from rand_ai.positional_shape_successor import (
    CENTERS,
    PositionalShapeSuccessorModel,
    shape_features,
)
from rand_ai.strategy_prediction import (
    _STRATEGY_DEPENDENCIES,
    _StrategyState,
    build_prediction_suites,
)

ID = "positional_shape_successor"
A = (1, 8, 20, 30, 40, 49)
B = (2, 9, 21, 31, 41, 48)


def test_feature_formulas_and_validation() -> None:
    values = shape_features(A[::-1])
    assert values[:6] == pytest.approx(A)
    assert values[6:11] == pytest.approx([4.5, 14, 25, 35, 44.5])
    assert values[11] == pytest.approx(29 / 3)
    assert values[-1] == pytest.approx(148 / 6)
    assert CENTERS[:6] == pytest.approx([i * 50 / 7 for i in range(1, 7)])
    assert CENTERS[6] == pytest.approx(75 / 7)
    assert CENTERS[-1] == 25
    for invalid in [(1, 2), (1, 1, 2, 3, 4, 5), (0, 2, 3, 4, 5, 6)]:
        with pytest.raises(ValueError):
            shape_features(invalid)


def test_empty_short_and_constant_history() -> None:
    model = PositionalShapeSuccessorModel()
    assert set(model.predict()[0].values()) == {6 / 49}
    model.observe(A)
    assert set(model.predict()[0].values()) == {6 / 49}
    model.observe(A)
    scores, details = model.predict()
    assert scores[1] == pytest.approx((1 + 8 * 6 / 49) / 9)
    assert scores[2] == pytest.approx((8 * 6 / 49) / 9)
    assert "1 of 1 neighbors" in details[1][0]
    assert sum(scores.values()) == pytest.approx(6)


def test_population_spread_family_weights_and_successors() -> None:
    model = PositionalShapeSuccessorModel()
    for draw in (A, B, A):
        model.observe(draw)
    features = np.stack(model.features)
    spreads = np.std(features, axis=0, ddof=0)
    assert spreads[0] == pytest.approx(np.sqrt(2) / 3)
    squared = (
        np.divide(
            features[1] - features[2], spreads, out=np.zeros(21), where=spreads > 0
        )
        ** 2
    )
    distance = (
        0.5 * squared[:6][spreads[:6] > 0].mean()
        + 0.5 * squared[6:][spreads[6:] > 0].mean()
    )
    weight = 1 / (1 + distance)
    scores, _ = model.predict()
    # The exact-match A is followed by B. The latest A is never a source.
    assert scores[2] == pytest.approx((1 + 8 * 6 / 49) / (9 + weight))
    assert scores[1] == pytest.approx((weight + 8 * 6 / 49) / (9 + weight))
    assert scores[2] > scores[1]


def test_neighbor_limit_stable_ties_and_partially_constant_features() -> None:
    model = PositionalShapeSuccessorModel()
    for _ in range(40):
        model.observe(A)
    assert "32 of 32 neighbors" in model.predict()[1][1][0]
    model.observe((1, 8, 20, 30, 40, 48))
    assert all(np.isfinite(score) for score in model.predict()[0].values())
    assert model.predict() == model.predict()


def test_integration_order_efficacy_and_future_invariance() -> None:
    draws = Draws()
    for numbers in (A, B, A):
        draws.add(Draw(*numbers))
    draws.prepare_predictions()
    prefix = build_prediction_suites(draws.draws, enabled_strategy_ids=(ID,))
    draws.add(Draw(3, 10, 22, 32, 42, 47))
    draws.prepare_predictions()
    extended = build_prediction_suites(draws.draws, enabled_strategy_ids=(ID,))
    for index, suite in enumerate(prefix):
        before = suite.strategies[0]
        after = extended[index].strategies[0]
        assert before.numbers == after.numbers
        assert before.top_numbers == after.top_numbers
        assert (
            len(before.numbers) == len({item.number for item in before.numbers}) == 49
        )
        assert before.top_numbers == tuple(item.number for item in before.numbers[:6])
        assert [item.score for item in before.numbers] == sorted(
            (item.score for item in before.numbers), reverse=True
        )
    efficacy = extended[-1].strategies[0].efficacy
    assert efficacy is not None
    assert efficacy.evaluated_draws == 3
    assert ID not in DEFAULT_STRATEGY_IDS
    assert all(
        ID not in dependencies for dependencies in _STRATEGY_DEPENDENCIES.values()
    )
    assert _StrategyState(("freshness",)).positional_shape_successor is None
