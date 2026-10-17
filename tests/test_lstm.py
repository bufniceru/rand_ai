"""LSTM chronology, portable inference, artifacts and research protocol tests."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rand_ai import Draw
from rand_ai.lstm import (
    SEEDS,
    SHAPES,
    encode_draws,
    load_lstm_artifact,
    model_probabilities,
    prefix_fingerprint,
    top_six,
)
from scripts import benchmark_lstm as study


def history(count=50):
    rng = np.random.default_rng(17)
    return [
        Draw(*map(int, rng.choice(np.arange(1, 50), 6, replace=False)))
        for _ in range(count)
    ]


def config():
    return study.Config(
        validation=8,
        holdout=10,
        minimum_train=10,
        max_epochs=3,
        patience=2,
        bootstrap=100,
    )


def weights():
    return {name: np.zeros(shape) for name, shape in SHAPES.items()}


def test_window_alignment_and_chronological_split():
    draws = history()
    x, y, train_end, val_end = study.build_dataset(draws, config())
    encoded = encode_draws(draws)
    assert x.shape == (43, 7, 49)
    assert y.shape == (43, 49)
    np.testing.assert_array_equal(x[0], encoded[:7])
    np.testing.assert_array_equal(y[0], encoded[7])
    np.testing.assert_array_equal(x[train_end], encoded[train_end : train_end + 7])
    assert train_end == 25 and val_end == 33
    assert (y.sum(axis=1) == 6).all()
    with pytest.raises(ValueError, match="Insufficient"):
        study.build_dataset(draws[:20], config())
    with pytest.raises(ValueError, match="positive"):
        replace(config(), holdout=0)


def test_metrics_and_deterministic_distinct_tickets():
    values = np.full((2, 49), 6 / 49)
    tickets = top_six(values)
    np.testing.assert_array_equal(tickets, [[1, 2, 3, 4, 5, 6]] * 2)
    labels = encode_draws([Draw(), Draw(7, 8, 9, 10, 11, 12)])
    metrics = study.metrics(labels, tickets, values)
    assert metrics["meanHits"] == 3
    assert metrics["matchCounts"] == [1, 0, 0, 0, 0, 0, 1]
    assert metrics["atLeastThreeRate"] == 0.5
    assert metrics["brier"] == pytest.approx((6 / 49) * (43 / 49))
    assert metrics["logLoss"] == pytest.approx(
        -6 / 49 * np.log(6 / 49) - 43 / 49 * np.log(43 / 49)
    )
    assert sum(row["count"] for row in metrics["calibration"]) == 98
    for invalid in (np.full((2, 49), np.nan), np.full((2, 49), 1.1), np.zeros((2, 48))):
        with pytest.raises(ValueError):
            top_six(invalid)


def test_independent_numpy_windows_and_shapes():
    draws = history()
    x, *_ = study.build_dataset(draws, config())
    result = model_probabilities(weights(), x)
    np.testing.assert_array_equal(result, np.full((len(x), 49), 0.5))
    np.testing.assert_array_equal(result[1:2], model_probabilities(weights(), x[1:2]))
    with pytest.raises(ValueError, match="finite windows"):
        model_probabilities(weights(), np.zeros((1, 8, 49)))


def test_artifact_validation_and_prefix_enforcement(tmp_path):
    draws = history()
    path = tmp_path / "lstm.npz"
    models = [weights() for _ in SEEDS]
    study.write_artifact(path, models, draws, 40, False)
    artifact = load_lstm_artifact(path)
    assert not artifact.historical_gate_passed
    np.testing.assert_array_equal(artifact.predict(draws), np.full(49, 0.5))
    assert prefix_fingerprint(draws[:40]) == artifact.fingerprint
    with pytest.raises(ValueError, match="activation"):
        artifact.predict(draws[:39])
    with pytest.raises(ValueError, match="prefix"):
        artifact.predict([Draw(), *draws[1:]])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for invalid in (np.zeros((2, 2)), np.full(SHAPES["head.bias"], np.nan)):
        corrupted = {**arrays, "seed0/head.bias": invalid}
        np.savez_compressed(path, allow_pickle=False, **corrupted)
        with pytest.raises(ValueError, match="Invalid LSTM weight"):
            load_lstm_artifact(path)
    metadata = json.loads(str(arrays["metadata"].item()))
    for key, invalid in (
        ("schemaVersion", 2),
        ("seeds", [1]),
        ("activationReferenceDraw", 1),
        ("prefixFingerprint", "z" * 64),
        ("historicalGatePassed", "false"),
    ):
        np.savez_compressed(
            path,
            allow_pickle=False,
            **{**arrays, "metadata": json.dumps({**metadata, key: invalid})},
        )
        with pytest.raises(ValueError):
            load_lstm_artifact(path)


def test_training_is_deterministic_test_blind_and_portable():
    pytest.importorskip("torch")
    draws = history()
    cfg = config()
    x, y, train_end, val_end = study.build_dataset(draws, cfg)
    original, log = study.train_model(x, y, train_end, val_end, SEEDS[0], cfg)
    altered = [*draws[: val_end + 7], *history(20)]
    # Keep the original frozen split when extending/changing test history.
    encoded = encode_draws(altered)
    altered_x = np.stack([encoded[i : i + 7] for i in range(len(altered) - 7)])
    altered_y = encoded[7:]
    repeated, repeated_log = study.train_model(
        altered_x, altered_y, train_end, val_end, SEEDS[0], cfg
    )
    assert log["bestEpoch"] == repeated_log["bestEpoch"]
    assert log["validationLosses"] == repeated_log["validationLosses"]
    assert log["parityMaxAbsError"] <= 1e-6
    for name in SHAPES:
        np.testing.assert_array_equal(original[name], repeated[name])
    np.testing.assert_array_equal(
        model_probabilities(original, x[:1]),
        model_probabilities(repeated, altered_x[:1]),
    )


def test_baselines_are_aligned_raw_and_causal():
    draws = history()
    probabilities, tickets = study.baseline_predictions(draws)
    assert all(rows.shape == (43, 6) for rows in tickets.values())
    assert all(rows.shape == (43, 49) for rows in probabilities.values())
    assert "cdm" not in probabilities
    np.testing.assert_array_equal(tickets["frequency"], tickets["cdm"])
    assert (probabilities["bayesian"] > 0).all()
    assert (probabilities["bayesian"] < 0.5).all()
    changed = [*draws[:-1], Draw()]
    other_p, other_t = study.baseline_predictions(changed)
    for name in probabilities:
        np.testing.assert_array_equal(probabilities[name], other_p[name])
    for name in tickets:
        np.testing.assert_array_equal(tickets[name], other_t[name])


def test_bootstrap_holm_and_reproducibility():
    comparison = study.paired_comparison(np.ones(20), config())
    assert comparison["ci95"] == [1, 1]
    assert comparison == study.paired_comparison(np.ones(20), config())
    comparisons = {"first": {"p": 0.01}, "second": {"p": 0.04}, "third": {"p": 0.8}}
    study.holm(comparisons)
    assert [row["holmP"] for row in comparisons.values()] == pytest.approx(
        [0.03, 0.08, 0.8]
    )


def test_small_end_to_end_report_and_incompatible_resume(tmp_path: Path):
    pytest.importorskip("torch")
    draws = history()
    cfg = config()
    result, artifact = study.run_job(draws, cfg, tmp_path / "historical")
    assert artifact is not None
    assert result["holdout"]["ensemble"]["draws"] == 10
    assert result["validation"]["ensemble"]["draws"] == 8
    records = [
        json.loads(line)
        for line in (tmp_path / "historical/predictions.jsonl").read_text().splitlines()
    ]
    assert len(records) == 18
    assert records[0]["targetDraw"] == 33
    assert records[-1]["targetDraw"] == 50
    assert all(len(record["ensembleProbabilities"]) == 49 for record in records)
    assert all(len(record["seedProbabilities"]) == 5 for record in records)
    repeated, cached = study.run_job(draws, cfg, tmp_path / "historical")
    assert repeated == result and cached is None
    study.write_report(tmp_path, {})
    assert "Historical comparisons" in (tmp_path / "report.md").read_text()
    with pytest.raises(ValueError, match="incompatible"):
        study.run_job(draws, replace(cfg, max_epochs=4), tmp_path / "historical")
