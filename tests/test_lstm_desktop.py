"""Verify optional LSTM desktop activation, payloads and frozen packaging."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rand_ai import Draw, Draws
from rand_ai.gui_bridge import (
    DEFAULT_STRATEGY_IDS,
    build_analysis_payload,
    parse_strategy_ids,
    portfolio_backtest_data,
    write_export_archive,
)
from rand_ai.lstm import LstmArtifact, SHAPES, load_packaged_lstm, prefix_fingerprint
from rand_ai.strategy_prediction import _StrategyState, build_prediction_suites
from zipfile import ZipFile


def draws_fixture():
    draws = Draws()
    for _ in range(6):
        draws.add(Draw())
    draws.add(Draw(40, 41, 42, 43, 44, 45))
    for _ in range(3):
        draws.add(Draw(7, 8, 9, 10, 11, 12))
    draws.prepare_predictions()
    return draws


def zero_artifact(draws):
    models = tuple(
        {key: np.zeros(shape) for key, shape in SHAPES.items()} for _ in range(5)
    )
    return LstmArtifact(7, prefix_fingerprint(draws.draws[:7]), False, models)


def test_packaged_model_and_torch_free_runtime():
    artifact = load_packaged_lstm()
    assert artifact.activation_reference_draw == 521
    assert not artifact.historical_gate_passed
    assert len(artifact.models) == 5
    assert load_packaged_lstm() is artifact
    root = Path(__file__).resolve().parents[1]
    packaged = root / "src/rand_ai/lstm_v1.npz"
    assert (
        hashlib.sha256(packaged.read_bytes()).hexdigest()
        == "07ba94df4b60a1e7781c6294961e25d8a5358a572d9966d8d3b1456e6addba6d"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from rand_ai.lstm import load_packaged_lstm; "
            "a=load_packaged_lstm(); assert a.activation_reference_draw==521; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_default_disabled_and_no_hidden_ensemble_dependency(monkeypatch):
    assert "lstm" not in DEFAULT_STRATEGY_IDS
    assert parse_strategy_ids("lstm,lstm") == ("lstm",)

    def unexpected_load():
        raise AssertionError("disabled LSTM must not load")

    monkeypatch.setattr(
        "rand_ai.strategy_prediction.load_packaged_lstm", unexpected_load
    )
    for selected in (("bayesian",), ("residual_coverage",), ("chained",)):
        state = _StrategyState(selected)
        assert "lstm" not in state.enabled_strategy_ids
        assert state.lstm_history == []


def test_activation_ties_and_efficacy_exclude_unavailable_targets(monkeypatch):
    draws = draws_fixture()
    monkeypatch.setattr(
        "rand_ai.strategy_prediction.load_packaged_lstm", lambda: zero_artifact(draws)
    )
    suites = build_prediction_suites(draws.draws, enabled_strategy_ids=("lstm",))
    assert all(suite.strategies == () for suite in suites[:6])
    for suite in suites[6:]:
        strategy = suite.strategies[0]
        assert strategy.name == "LSTM (Experimental)"
        assert strategy.top_numbers == (1, 2, 3, 4, 5, 6)
        assert len(strategy.numbers) == 49
        assert all(item.score == 0.5 for item in strategy.numbers)
        assert "gate failed" in strategy.description
        assert "gate failed" in strategy.numbers[0].details[3]
    efficacy = suites[-1].strategies[0].efficacy
    assert efficacy is not None and efficacy.evaluated_draws == 3


def test_prefix_rejection_append_and_future_causality(monkeypatch):
    draws = draws_fixture()
    artifact = zero_artifact(draws)
    monkeypatch.setattr(
        "rand_ai.strategy_prediction.load_packaged_lstm", lambda: artifact
    )
    original = build_prediction_suites(draws.draws, enabled_strategy_ids=("lstm",))
    extended = Draws()
    for draw in draws.draws:
        extended.add(Draw(*[ball.value for ball in draw.balls], date=draw.date))
    extended.add(Draw())
    extended.prepare_predictions()
    later = build_prediction_suites(extended.draws, enabled_strategy_ids=("lstm",))
    assert [suite.strategies[0].numbers for suite in original[6:]] == [
        suite.strategies[0].numbers for suite in later[6:10]
    ]
    changed = Draws()
    changed.add(Draw(2, 3, 4, 5, 6, 7))
    for draw in draws.draws[1:]:
        changed.add(draw)
    changed.prepare_predictions()
    rejected = build_prediction_suites(changed.draws, enabled_strategy_ids=("lstm",))
    assert all(suite.strategies == () for suite in rejected)
    state = _StrategyState(("lstm",))
    for draw in changed.draws:
        numbers = {ball.value for ball in draw.balls}
        state.train(numbers)
        state.remember(numbers, draw.date)
    assert state._lstm_scores() is None
    assert state.lstm_invalid


def test_lstm_bridge_comparisons_portfolios_and_export(monkeypatch, tmp_path):
    draws = draws_fixture()
    monkeypatch.setattr(
        "rand_ai.strategy_prediction.load_packaged_lstm", lambda: zero_artifact(draws)
    )
    source = tmp_path / "draws.pkl"
    draws.save_pickle(source)
    payload = build_analysis_payload(
        draws,
        source,
        enabled_strategies=("lstm",),
        enabled_reports=(
            "predictions",
            "draw-comparison",
            "strategy-effectiveness",
            "possible-draw",
        ),
        strategy_cache_dir=tmp_path / "cache",
    )
    latest = payload["predictionSuites"][-1]["strategies"][0]
    assert latest["id"] == "lstm" and latest["name"] == "LSTM (Experimental)"
    assert latest["topNumbers"] == [1, 2, 3, 4, 5, 6]
    assert len(latest["numbers"]) == 49
    assert latest["efficacy"]["evaluatedDraws"] == 3
    assert payload["drawComparisonHistory"][-1]["strategies"][0]["id"] == "lstm"
    compact = portfolio_backtest_data(
        source, enabled_strategies=("lstm",), strategy_cache_dir=tmp_path / "cache"
    )
    assert all(row["strategies"] == [] for row in compact["records"][:6])
    assert compact["records"][-1]["strategies"][0]["id"] == "lstm"
    output = tmp_path / "export.zip"
    write_export_archive(
        source, output, enabled_reports=(), enabled_strategies=("lstm",)
    )
    with ZipFile(output) as archive:
        assert json.loads(archive.read("metadata.json"))["enabledStrategies"] == [
            "lstm"
        ]


def test_lstm_does_not_change_bayesian(monkeypatch):
    draws = draws_fixture()
    monkeypatch.setattr(
        "rand_ai.strategy_prediction.load_packaged_lstm", lambda: zero_artifact(draws)
    )
    before = build_prediction_suites(draws.draws, enabled_strategy_ids=("bayesian",))
    after = build_prediction_suites(
        draws.draws, enabled_strategy_ids=("bayesian", "lstm")
    )
    for first, second in zip(before, after, strict=True):
        assert first.strategies[0] == second.strategies[0]


def test_packaged_loader_rejects_corrupt_resource(monkeypatch, tmp_path):
    import rand_ai.lstm as runtime

    (tmp_path / "lstm_v1.npz").write_bytes(b"invalid artifact")
    load_packaged_lstm.cache_clear()
    monkeypatch.setattr(runtime.resources, "files", lambda _name: tmp_path)
    try:
        with pytest.raises(ValueError):
            load_packaged_lstm()
    finally:
        load_packaged_lstm.cache_clear()
