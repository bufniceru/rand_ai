"""Verify V2 selection chronology, independent evidence, and app registration."""

import builtins
import json
from unittest.mock import Mock

import numpy as np
import pytest

from rand_ai import Draw, Draws
from rand_ai import xgboost_classifier_v2 as v2
from rand_ai.strategy_prediction import _StrategyState, build_prediction_suites


class FlatEstimator:
    def fit(self, features, labels):
        self.features = features.copy()
        self.labels = labels.copy()
        return self

    def predict_proba(self, features):
        return np.tile([43 / 49, 6 / 49], (len(features), 1))


def fake_model():
    model = v2.XGBoostV2Model()
    calls = []

    def factory(candidate):
        estimator = FlatEstimator()
        calls.append((candidate, estimator))
        return estimator

    model.estimator_factory = factory
    return model, calls


def transitions(count):
    return [
        (np.full((49, 14), index, dtype=float), np.asarray([1] * 6 + [0] * 43))
        for index in range(count)
    ]


def test_candidate_grid_parameters_and_immutable_pending():
    assert len(v2.CANDIDATES) == 8
    assert v2.CANDIDATES[0].identifier == "f4-d1-w250"
    assert v2.estimator_parameters(v2.CANDIDATES[-1])["max_depth"] == 2
    model, calls = fake_model()
    model.observe({1, 2, 3, 4, 5, 6})
    assert model.completed_transitions == 0
    features = np.zeros((49, 14))
    scores, details = model.predict(features, [0] * 50, 0)
    features[:] = 9
    assert list(scores.values()) == pytest.approx([6 / 49] * 49)
    assert "smoothed_frequency" in details[1][1]
    model.observe({1, 2, 3, 4, 5, 6})
    assert model.pending is None
    np.testing.assert_array_equal(model.transitions[0][0], np.zeros((49, 14)))
    np.testing.assert_array_equal(model.transitions[0][1], [1] * 6 + [0] * 43)
    model.observe({49})
    assert model.completed_transitions == 1
    assert calls == []


def test_selection_uses_preceding_whole_draws_and_simplicity_ties():
    model, calls = fake_model()
    model.transitions.extend(transitions(240))
    model.completed_transitions = 240
    model._select()
    assert model.selected == v2.CANDIDATES[0]
    assert len(calls) == 24
    for index in range(3):
        estimator = calls[index][1]
        assert estimator.features.shape == ((120 + 40 * index) * 49, 4)
        assert estimator.features[:, 0].max() == 119 + 40 * index
        assert estimator.labels.sum() == (120 + 40 * index) * 6
    assert len(model.selection_results) == 8
    assert len(model.selection_results[0]["fold_validation_losses"]) == 3
    model.transitions.clear()
    model.transitions.extend(transitions(500))
    model.completed_transitions = 500
    calls.clear()
    model._select()
    assert calls[0][1].features[:, 0].min() == 130
    assert calls[0][1].features.shape == (250 * 49, 4)
    assert calls[3][1].features[:, 0].min() == 0
    assert calls[3][1].features.shape == (380 * 49, 4)


def test_observation_schedule_window_and_evidence_do_not_reset_on_selection():
    model, calls = fake_model()
    features = np.zeros((49, 14))
    for count in range(1, 602):
        model.predict(features, [0] * 50, count)
        model.observe({1, 2, 3, 4, 5, 6})
        if count == 239:
            assert model.model is None
        if count == 240:
            assert len(calls) == 25
            assert model.last_fit_transition == model.last_selection_transition == 240
        if count == 249:
            assert model.last_fit_transition == 240
        if count == 250:
            assert model.last_fit_transition == 250
            assert len(model.evidence) == 10
            assert not model.gate_active
        if count == 280:
            assert model.last_selection_transition == 280
            assert len(model.evidence) == 40
        if count == 360:
            assert len(model.evidence) == 120
            assert model.last_gate_transition == 360
    assert len(model.transitions) == 500
    assert model.completed_transitions == 601
    assert model.last_fit_transition == 600
    assert len(model.evidence) == 120
    assert model.gate_interval is not None


def test_conservative_gate_activation_reversal_and_raw_forecasts_during_fallback():
    model, _ = fake_model()
    model.model = FlatEstimator()
    model.selected = v2.CANDIDATES[0]
    model.completed_transitions = 360
    model.evidence.extend([(0.02, 1)] * 120)
    model._evaluate_gate()
    assert model.gate_active
    assert model.gate_interval == pytest.approx((0.02, 0.02))
    scores, details = model.predict(np.zeros((49, 14)), [0] * 50, 360)
    assert model.display_source == "xgboost"
    assert list(scores.values()) == pytest.approx([6 / 49] * 49)
    assert "gate passed" in details[1][1]
    model.evidence.clear()
    model.evidence.extend([(0.02, 0)] * 120)
    model._evaluate_gate()
    assert not model.gate_active
    scores, _ = model.predict(np.zeros((49, 14)), [0] * 50, 360)
    assert model.display_source == "smoothed_frequency"
    assert model.raw_probabilities is not None
    assert model.pending[1] is not None
    model.evidence.clear()
    model.evidence.extend([(-0.02, 1)] * 120)
    model._evaluate_gate()
    assert not model.gate_active
    model.evidence.clear()
    model._evaluate_gate()
    assert not model.gate_active


def test_probability_metrics_bootstrap_and_native_failures(monkeypatch):
    labels = np.asarray([1] * 6 + [0] * 43)
    perfect = labels.astype(float)
    loss, brier, hits = v2.forecast_metrics(labels, perfect)
    assert 0 < loss < 1e-6
    assert brier == 0 and hits == 6
    np.testing.assert_array_equal(perfect, labels)
    assert v2.bootstrap_interval(np.ones(120)) == (1.0, 1.0)
    assert v2.bootstrap_interval(np.zeros(1)) == (0.0, 0.0)
    with pytest.raises(ValueError, match="completed forecasts"):
        v2.bootstrap_interval(np.asarray([]))
    for values in (
        np.full((49, 2), np.nan),
        np.full((49, 2), -1),
        np.full((49, 2), 2),
        np.zeros((48, 2)),
    ):
        with pytest.raises(ValueError, match="finite probabilities"):
            v2.probabilities(
                Mock(predict_proba=Mock(return_value=values)), np.zeros((49, 4))
            )
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("missing native runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert _StrategyState(("freshness",)).xgboost_classifier_v2 is None
    assert _StrategyState(("residual_coverage",)).xgboost_classifier_v2 is None
    with pytest.raises(ImportError, match="native runtime"):
        v2.create_estimator(v2.CANDIDATES[0])
    model = v2.XGBoostV2Model()
    model.estimator_factory = Mock(
        return_value=Mock(fit=Mock(side_effect=ValueError("fit failed")))
    )
    with pytest.raises(ValueError, match="fit failed"):
        model._fit(v2.CANDIDATES[0], transitions(1))


def history(count):
    rng = np.random.default_rng(20260626)
    draws = Draws()
    for _ in range(count):
        draws.add(
            Draw(*(int(n) for n in rng.choice(np.arange(1, 50), 6, replace=False)))
        )
    draws.prepare_predictions()
    return draws


def test_real_v2_prefix_replay_future_changes_and_v1_preservation():
    draws = history(244)
    ids = ("xgboost_classifier", "xgboost_classifier_v2")
    full = build_prediction_suites(draws.draws, enabled_strategy_ids=ids)
    prefix = build_prediction_suites(
        draws.draws[:241], enabled_strategy_ids=("xgboost_classifier_v2",)
    )
    retained = build_prediction_suites(
        draws.draws, history_start=240, enabled_strategy_ids=("xgboost_classifier_v2",)
    )
    assert (
        full[240].strategies[1].numbers
        == prefix[-1].strategies[0].numbers
        == retained[0].strategies[0].numbers
    )
    changed = Draws()
    for draw in draws.draws[:241]:
        changed.add(draw)
    changed.add(Draw(1, 2, 3, 4, 5, 6))
    changed.prepare_predictions()
    altered = build_prediction_suites(
        changed.draws, enabled_strategy_ids=("xgboost_classifier_v2",)
    )
    assert altered[240].strategies[0].numbers == full[240].strategies[1].numbers
    original = build_prediction_suites(
        draws.draws, enabled_strategy_ids=("xgboost_classifier",)
    )
    assert original[-1].strategies[0] == full[-1].strategies[0]
    state = _StrategyState(("xgboost_classifier_v2",))
    assert state.enabled_strategy_ids == {"xgboost_classifier_v2"}
    assert state.xgboost_classifier is None
    assert state.lag_logistic is None


def test_v2_bridge_export_cache_and_details(tmp_path, monkeypatch):
    from zipfile import ZipFile
    from rand_ai import gui_bridge

    draws = history(3)
    source = tmp_path / "draws.pkl"
    draws.save_pickle(source)
    options = dict(
        enabled_strategies=("xgboost_classifier_v2",),
        enabled_reports=("predictions", "draw-portfolio", "strategy-effectiveness"),
    )
    assert "xgboost_classifier_v2" not in gui_bridge.DEFAULT_STRATEGY_IDS
    assert gui_bridge.STRATEGY_CACHE_SCHEMA_VERSION == 27
    assert gui_bridge.parse_strategy_ids("xgboost_classifier_v2") == (
        "xgboost_classifier_v2",
    )
    counted = Mock(wraps=gui_bridge._build_strategy_analysis)
    monkeypatch.setattr(gui_bridge, "_build_strategy_analysis", counted)
    first = gui_bridge.analyze_file(
        source, strategy_cache_dir=tmp_path / "cache", **options
    )
    second = gui_bridge.analyze_file(
        source, strategy_cache_dir=tmp_path / "cache", **options
    )
    assert counted.call_count == 1
    assert first["predictionSuites"] == second["predictionSuites"]
    result = first["predictionSuites"][-1]["strategies"][0]
    assert result["id"] == "xgboost_classifier_v2"
    assert len(result["numbers"]) == 49 and len(result["topNumbers"]) == 6
    assert "smoothed_frequency" in result["numbers"][0]["details"][1]
    json.dumps(first, allow_nan=False)
    monkeypatch.setattr(gui_bridge, "STRATEGY_CACHE_SCHEMA_VERSION", 28)
    gui_bridge.analyze_file(source, strategy_cache_dir=tmp_path / "cache", **options)
    assert counted.call_count == 2
    archive_path = tmp_path / "v2.zip"
    gui_bridge.write_export_archive(source, archive_path, **options)
    with ZipFile(archive_path) as archive:
        assert json.loads(archive.read("metadata.json"))["enabledStrategies"] == [
            "xgboost_classifier_v2"
        ]
