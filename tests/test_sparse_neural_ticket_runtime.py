"""Test the packaged NumPy Sparse Neural Ticket runtime."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import rand_ai.sparse_neural_ticket as sparse_runtime
from rand_ai import Draw, Draws
from rand_ai.sparse_neural_ticket import (
    SparseNeuralTicketArtifact,
    history_fingerprint,
    load_sparse_neural_ticket,
)
from rand_ai.strategy_prediction import _StrategyState


def _zero_artifact(prefix_fingerprint: str) -> SparseNeuralTicketArtifact:
    layers = (
        (np.zeros((32, 36)), np.zeros(32)),
        (np.zeros((16, 32)), np.zeros(16)),
        (np.zeros((1, 16)), np.zeros(1)),
    )
    return SparseNeuralTicketArtifact(
        activation_reference_draw=1,
        prefix_fingerprint=prefix_fingerprint,
        active_weights=860,
        total_weights=1680,
        selected_round=3,
        promotion_passed=False,
        feature_mean=np.zeros(36),
        feature_scale=np.ones(36),
        models=(layers, layers, layers, layers, layers),
    )


def test_packaged_artifact_is_finite_and_predicts_probabilities() -> None:
    artifact = load_sparse_neural_ticket()
    probabilities = artifact.predict(np.zeros((49, 36)))

    assert artifact.activation_reference_draw == 521
    assert artifact.selected_round == 3
    assert artifact.active_weights == 860
    assert artifact.total_weights == 1680
    assert artifact.active_fraction == pytest.approx(860 / 1680)
    assert not artifact.promotion_passed
    assert probabilities.shape == (49,)
    assert np.isfinite(probabilities).all()
    assert ((0 < probabilities) & (probabilities < 1)).all()
    with pytest.raises(ValueError, match="requires 36"):
        artifact.predict(np.zeros((49, 35)))
    invalid = np.zeros((49, 36))
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        artifact.predict(invalid)


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("schema", "Unsupported"),
        ("seeds", "five seeds"),
        ("mean", "feature_mean"),
        ("scale", "scales must be positive"),
        ("fingerprint", "prefix fingerprint"),
    ),
)
def test_rejects_corrupt_packaged_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    message: str,
) -> None:
    source_path = (
        Path(__file__).parents[1]
        / "src"
        / "rand_ai"
        / "sparse_neural_ticket_v1.npz"
    )
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    metadata = json.loads(str(arrays["metadata"].item()))
    if corruption == "schema":
        metadata["schemaVersion"] = 0
    elif corruption == "seeds":
        metadata["seeds"] = metadata["seeds"][:-1]
    elif corruption == "mean":
        arrays["feature_mean"] = arrays["feature_mean"][:-1]
    elif corruption == "scale":
        arrays["feature_scale"][0] = 0
    else:
        metadata["prefixFingerprint"] = "short"
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    artifact_path = tmp_path / "sparse_neural_ticket_v1.npz"
    with artifact_path.open("wb") as artifact_file:
        np.savez_compressed(artifact_file, **arrays)
    monkeypatch.setattr(sparse_runtime.resources, "files", lambda _name: tmp_path)
    sparse_runtime.load_sparse_neural_ticket.cache_clear()

    with pytest.raises(ValueError, match=message):
        sparse_runtime.load_sparse_neural_ticket()

    sparse_runtime.load_sparse_neural_ticket.cache_clear()


def test_strategy_activates_only_for_matching_history_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draw = Draw(1, 2, 8, 17, 31, 49)
    draws = Draws()
    draws.add(draw)
    draws.prepare_predictions()
    prefix = [[None, 1, 2, 8, 17, 31, 49]]
    artifact = _zero_artifact(history_fingerprint(prefix))
    monkeypatch.setattr(
        "rand_ai.strategy_prediction.load_sparse_neural_ticket",
        lambda: artifact,
    )
    state = _StrategyState(("sparse_neural_ticket",), total_draw_count=2)
    drawn = {1, 2, 8, 17, 31, 49}
    state.train(drawn)
    state.remember(drawn)
    combined = draws.draws[0].prediction
    assert combined is not None

    strategies = state.build_strategies(combined, 0)

    assert [strategy.strategy_id for strategy in strategies] == [
        "sparse_neural_ticket"
    ]
    assert strategies[0].name == "Sparse Neural Ticket (Experimental)"
    assert "promotion gate failed" in strategies[0].description


def test_strategy_rejects_a_changed_training_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.prepare_predictions()
    artifact = _zero_artifact(history_fingerprint([[None, 2, 3, 4, 5, 6, 7]]))
    monkeypatch.setattr(
        "rand_ai.strategy_prediction.load_sparse_neural_ticket",
        lambda: artifact,
    )
    state = _StrategyState(("sparse_neural_ticket",), total_draw_count=2)
    drawn = {1, 2, 8, 17, 31, 49}
    state.train(drawn)
    state.remember(drawn)
    combined = draws.draws[0].prediction
    assert combined is not None

    assert state.build_strategies(combined, 0) == ()
    assert state.sparse_neural_ticket_invalid
    assert state._sparse_neural_ticket_scores() is None


def test_strategy_runtime_defensive_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _zero_artifact("0" * 64)
    monkeypatch.setattr(
        "rand_ai.strategy_prediction.load_sparse_neural_ticket",
        lambda: artifact,
    )
    state = _StrategyState(("sparse_neural_ticket",), total_draw_count=2)
    state.draw_count = 2
    assert state._sparse_neural_ticket_scores() is None
    assert state.sparse_neural_ticket_invalid

    state.sparse_neural_ticket_invalid = False
    state.sparse_neural_ticket_ready = True
    state.lag_logistic_pending_features = None
    with pytest.raises(AssertionError, match="lag features"):
        state._sparse_neural_ticket_scores()

    state.sparse_neural_ticket = None
    assert state._sparse_neural_ticket_scores() is None
