"""Tests for the positional ARIMA baseline and causal strategy integration."""

import numpy as np
import pytest

from rand_ai import Draw, Draws
from rand_ai.arima import ArimaModel
from rand_ai.strategy_prediction import build_prediction_suites


def test_scores_are_finite_deterministic_and_bounded_in_memory():
    model = ArimaModel()
    scores, details = model.predict()
    assert set(scores) == set(range(1, 50))
    assert sum(scores.values()) == pytest.approx(1)
    assert "fallback" in details[1][1]
    rng = np.random.default_rng(71)
    for _ in range(260):
        model.observe([int(n) for n in rng.choice(49, 6, replace=False) + 1])
    assert len(model.history) == 250
    first = model.predict()
    assert first == model.predict()
    assert all(np.isfinite(s) and 0 <= s <= 1 for s in first[0].values())
    assert "Conditional least-squares" in first[1][1][1]


def test_constant_history_and_solver_failure(monkeypatch):
    model = ArimaModel()
    for _ in range(24):
        model.observe([1, 9, 17, 25, 33, 49])
    assert all(np.isfinite(v) for v in model.predict()[0].values())
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("test")
    monkeypatch.setattr(np.linalg, "lstsq", fail)
    assert "fallback" in model.predict()[1][1][1]
    with pytest.raises(ValueError):
        model.observe([1, 1, 2, 3, 4, 50])


def test_forecasts_do_not_use_future_draws():
    rng = np.random.default_rng(82)
    prefix = [Draw(*(int(n) for n in rng.choice(49, 6, replace=False) + 1)) for _ in range(30)]
    def replay(last):
        draws = Draws()
        for draw in [*prefix, last]:
            draws.add(draw)
        draws.prepare_predictions()
        return build_prediction_suites(draws.draws, enabled_strategy_ids=("arima",))
    first = replay(Draw(1, 2, 3, 4, 5, 6))
    changed = replay(Draw(44, 45, 46, 47, 48, 49))
    assert first[29].strategies[0].numbers == changed[29].strategies[0].numbers
    strategy = first[-1].strategies[0]
    assert strategy.strategy_id == "arima"
    assert len(strategy.numbers) == 49
    assert len(set(strategy.top_numbers)) == 6
