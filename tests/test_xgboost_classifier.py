"""Exercise delayed labels, bounded batch fitting, and public integration."""

import builtins
import json
from unittest.mock import Mock

import numpy as np
import pytest

from rand_ai import Draw, Draws
from rand_ai.strategy_prediction import (
    _StrategyState,
    _strategy,
    build_prediction_suites,
)
from rand_ai.xgboost_classifier import XGBoostClassifierModel, BASE_PROBABILITY


def history(length):
    rng = np.random.default_rng(20260626)
    draws = Draws()
    for _ in range(length):
        draws.add(
            Draw(*(int(n) for n in rng.choice(np.arange(1, 50), 6, replace=False)))
        )
    draws.prepare_predictions()
    return draws


def test_delayed_labels_fit_boundaries_and_bounded_buffer():
    model = XGBoostClassifierModel()
    estimator = Mock()
    factory = Mock(return_value=estimator)
    model.estimator_type = factory
    features = np.arange(49 * 14, dtype=float).reshape(49, 14)
    original = features.copy()
    model.observe(set(range(1, 7)))
    assert model.completed_transitions == 0
    scores, details = model.predict(features, [0] * 50, 0)
    features[:] = -1
    assert list(scores.values()) == pytest.approx([BASE_PROBABILITY] * 49)
    assert "Warm-up 0/120" in details[1][1]
    model.observe(set(range(1, 7)))
    x, y = model.transitions[0]
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(y, [1] * 6 + [0] * 43)
    model.observe({49})  # The same snapshot cannot be labelled twice.
    assert model.completed_transitions == 1
    for transition in range(2, 512):
        model.pending_features = np.full((49, 14), transition, dtype=float)
        model.observe(set(range(1, 7)))
        if transition == 119:
            assert model.model is None
        if transition == 120:
            assert factory.call_count == 1
            assert estimator.fit.call_args.args[0].shape == (120 * 49, 14)
        if transition == 129:
            assert factory.call_count == 1
        if transition == 130:
            assert factory.call_count == 2
    assert model.completed_transitions == 511
    assert model.last_retraining_transition == 510
    assert len(model.transitions) == 500
    assert model.transitions[0][0][0, 0] == 12
    assert estimator.fit.call_args.args[0].shape == (500 * 49, 14)
    assert factory.call_args.kwargs == dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        n_jobs=1,
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=10,
        reg_lambda=10,
        reg_alpha=0,
        subsample=1,
        colsample_bytree=1,
        scale_pos_weight=1,
        base_score=6 / 49,
        random_state=20260626,
    )
    estimator.predict_proba.return_value = np.tile([0.8, 0.2], (49, 1))
    scores, details = model.predict(original, [0] * 50, 0)
    assert set(scores.values()) == {0.2}
    assert "Fitted XGBoost" in details[1][1]
    estimator.predict_proba.return_value = np.full((49, 2), np.nan)
    with pytest.raises(ValueError, match="49 finite"):
        model.predict(original, [0] * 50, 0)
    estimator.predict_proba.return_value = np.zeros((48, 2))
    with pytest.raises(ValueError, match="49 finite"):
        model.predict(original, [0] * 50, 0)


def test_shared_features_and_isolated_history_state():
    standalone = _StrategyState(("xgboost_classifier",))
    logistic = _StrategyState(("lag_logistic",))
    for drawn in ({1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}, {1, 13, 14, 15, 16, 17}):
        standalone.remember(drawn)
        logistic.remember(drawn)
        np.testing.assert_array_equal(
            standalone._history_feature_matrix(),
            logistic._lag_logistic_feature_matrix(),
        )
    assert standalone.lag_logistic is None
    assert standalone._history_feature_matrix().shape == (49, 14)
    assert standalone._history_features(1)[:3] == (1, 0, 1)
    assert standalone.enabled_strategy_ids == {"xgboost_classifier"}


def test_disabled_strategy_does_not_import_xgboost_or_allocate_model(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("native dependency unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert _StrategyState(("freshness",)).xgboost_classifier is None
    assert _StrategyState(("residual_coverage",)).xgboost_classifier is None
    with pytest.raises(ImportError, match="native dependency unavailable"):
        _StrategyState(("xgboost_classifier",))


def test_fit_errors_propagate():
    model = XGBoostClassifierModel()
    model.completed_transitions = 119
    model.pending_features = np.zeros((49, 14))
    model.estimator_type = Mock(
        return_value=Mock(fit=Mock(side_effect=ValueError("fit failed")))
    )
    with pytest.raises(ValueError, match="fit failed"):
        model.observe({1, 2, 3, 4, 5, 6})
    assert model.model is None


def test_ties_use_number_order_and_preserve_actual_gaps():
    scores = {n: 0.2 for n in range(1, 50)}
    gaps = {n: n for n in range(1, 50)}
    result = _strategy("xgboost_classifier", "XGBoost Classifier", "", scores, gaps)
    assert result.top_numbers == (1, 2, 3, 4, 5, 6)
    assert result.numbers[0].gap == 1


def test_real_estimator_replay_is_deterministic_and_future_independent():
    draws = history(133)

    def replay(values, start=0):
        return build_prediction_suites(
            values, history_start=start, enabled_strategy_ids=("xgboost_classifier",)
        )

    full = replay(draws.draws)
    prefix = replay(draws.draws[:131])
    repeated = replay(draws.draws, 130)
    assert (
        full[130].strategies[0].numbers
        == prefix[-1].strategies[0].numbers
        == repeated[0].strategies[0].numbers
    )
    changed = [*draws.draws[:131], Draw(1, 2, 3, 4, 5, 6)]
    changed_history = Draws()
    for draw in changed:
        changed_history.add(draw)
    changed_history.prepare_predictions()
    assert (
        replay(changed_history.draws)[130].strategies[0].numbers
        == full[130].strategies[0].numbers
    )
    prediction = full[130].strategies[0]
    assert len(prediction.numbers) == 49
    assert len(set(prediction.top_numbers)) == 6
    assert all(np.isfinite(n.score) and 0 <= n.score <= 1 for n in prediction.numbers)
    assert "Last retraining transition 130" in prediction.numbers[0].details
    assert (
        build_prediction_suites([], enabled_strategy_ids=("xgboost_classifier",)) == ()
    )


def test_bridge_serialization_cache_and_portfolio(tmp_path):
    from rand_ai.gui_bridge import (
        DEFAULT_STRATEGY_IDS,
        STRATEGY_CACHE_SCHEMA_VERSION,
        build_analysis_payload,
        parse_strategy_ids,
    )

    assert STRATEGY_CACHE_SCHEMA_VERSION == 26
    assert "xgboost_classifier" not in DEFAULT_STRATEGY_IDS
    assert parse_strategy_ids("xgboost_classifier") == ("xgboost_classifier",)
    draws = history(3)
    draws.save_pickle(tmp_path / "draws.pkl")
    payload = build_analysis_payload(
        draws,
        tmp_path / "draws.pkl",
        enabled_strategies=("xgboost_classifier",),
        enabled_reports=(
            "predictions",
            "draw-portfolio",
            "strategy-effectiveness",
            "strategy-hit-statistics",
        ),
    )
    assert payload["options"]["enabledStrategies"] == ["xgboost_classifier"]
    strategy = payload["predictionSuites"][-1]["strategies"][0]
    assert strategy["id"] == "xgboost_classifier"
    assert len(strategy["numbers"]) == 49
    assert len(strategy["topNumbers"]) == 6
    json.dumps(payload, allow_nan=False)


def test_xgboost_cache_reuse_schema_change_and_export(tmp_path, monkeypatch):
    from zipfile import ZipFile
    from rand_ai import gui_bridge

    draws = history(3)
    source = tmp_path / "draws.pkl"
    draws.save_pickle(source)
    options = dict(
        enabled_strategies=("xgboost_classifier",), enabled_reports=("predictions",)
    )
    original = gui_bridge._build_strategy_analysis
    counted = Mock(wraps=original)
    monkeypatch.setattr(gui_bridge, "_build_strategy_analysis", counted)
    first = gui_bridge.analyze_file(
        source, strategy_cache_dir=tmp_path / "cache", **options
    )
    second = gui_bridge.analyze_file(
        source, strategy_cache_dir=tmp_path / "cache", **options
    )
    assert counted.call_count == 1
    assert first["predictionSuites"] == second["predictionSuites"]
    monkeypatch.setattr(gui_bridge, "STRATEGY_CACHE_SCHEMA_VERSION", 27)
    gui_bridge.analyze_file(source, strategy_cache_dir=tmp_path / "cache", **options)
    assert counted.call_count == 2
    archive_path = tmp_path / "export.zip"
    gui_bridge.write_export_archive(source, archive_path, **options)
    with ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["enabledStrategies"] == ["xgboost_classifier"]
