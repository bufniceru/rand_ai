from __future__ import annotations

import numpy as np
import pytest

import rand_ai.nonlinear_dynamics as nonlinear_dynamics
from rand_ai.nonlinear_dynamics import (
    BASE_PROBABILITY,
    EXPECTED_RANDOM_HITS,
    MAX_ANALOGUES,
    RecurrenceDynamicsModel,
    _inverse_distance_weights,
    _lower_confidence_bound,
    _recurrence_matrix,
    _rqa_metrics,
    _run_lengths,
    classify_evidence,
    classify_forecast_evidence,
    delay_embeddings,
    draw_features,
    feature_history,
    forecast_delay_embeddings,
    forecast_feature_history,
    forecast_features,
    nonlinear_dynamics_analysis,
)


_PATTERN = [
    (1, 7, 13, 21, 34, 48),
    (2, 8, 14, 22, 35, 49),
    (3, 9, 15, 23, 36, 47),
    (4, 10, 16, 24, 37, 46),
]


def test_draw_features_are_order_independent_and_validate_draws() -> None:
    previous = (1, 2, 3, 20, 30, 40)
    previous_previous = (4, 5, 6, 21, 31, 41)
    forward = draw_features((1, 7, 13, 21, 34, 48), previous, previous_previous)
    reverse = draw_features((48, 34, 21, 13, 7, 1), previous, previous_previous)

    assert forward.shape == (20,)
    assert np.allclose(forward, reverse)
    assert np.all(np.isfinite(forward))
    assert feature_history([previous, previous_previous]).shape == (2, 20)
    with pytest.raises(ValueError, match="six unique"):
        draw_features((1, 1, 2, 3, 4, 5))
    with pytest.raises(ValueError, match="between 1 and 49"):
        draw_features((0, 1, 2, 3, 4, 5))


def test_delay_embeddings_use_three_weighted_draws() -> None:
    indexes, empty = delay_embeddings(np.zeros((2, 20)))
    assert indexes.tolist() == []
    assert empty.shape == (0, 60)

    features = np.vstack(
        (np.ones(20), np.full(20, 2.0), np.full(20, 3.0), np.full(20, 4.0))
    )
    indexes, embedded = delay_embeddings(features)
    assert indexes.tolist() == [2, 3]
    assert embedded.shape == (2, 60)
    assert embedded[0, [0, 20, 40]].tolist() == pytest.approx([0.5, 1.5, 3.0])


def test_forecast_features_use_six_values_and_eighteen_value_embeddings() -> None:
    forward = forecast_features((1, 7, 13, 21, 34, 48))
    reverse = forecast_features((48, 34, 21, 13, 7, 1))
    assert forward.shape == (6,)
    assert forward.tolist() == pytest.approx(reverse.tolist())
    assert forecast_feature_history(_PATTERN[:2]).shape == (2, 6)

    indexes, empty = forecast_delay_embeddings(np.zeros((2, 6)))
    assert indexes.tolist() == []
    assert empty.shape == (0, 18)

    features = np.vstack(
        (np.ones(6), np.full(6, 2.0), np.full(6, 3.0), np.full(6, 4.0))
    )
    indexes, embedded = forecast_delay_embeddings(features)
    assert indexes.tolist() == [2, 3]
    assert embedded.shape == (2, 18)
    assert embedded[0, [0, 6, 12]].tolist() == pytest.approx([0.5, 1.5, 3.0])


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"evaluated_forecasts": 99, "analogue_count": 24, "average_hits": 2.0, "lower_bound": 2.0, "surrogate_p_value": 0.01}, "insufficient"),
        ({"evaluated_forecasts": 100, "analogue_count": 7, "average_hits": 2.0, "lower_bound": 2.0, "surrogate_p_value": 0.01}, "insufficient"),
        ({"evaluated_forecasts": 100, "analogue_count": 8, "average_hits": EXPECTED_RANDOM_HITS, "lower_bound": 0.0, "surrogate_p_value": None}, "weak"),
        ({"evaluated_forecasts": 100, "analogue_count": 8, "average_hits": 1.0, "lower_bound": 0.8, "surrogate_p_value": None}, "weak"),
        ({"evaluated_forecasts": 100, "analogue_count": 8, "average_hits": 1.0, "lower_bound": 0.0, "surrogate_p_value": 0.06}, "weak"),
        ({"evaluated_forecasts": 100, "analogue_count": 8, "average_hits": 1.0, "lower_bound": 0.7, "surrogate_p_value": 0.05}, "suggestive"),
        ({"evaluated_forecasts": 100, "analogue_count": 8, "average_hits": 1.0, "lower_bound": 0.8, "surrogate_p_value": 0.01}, "supported"),
    ],
)
def test_evidence_classification_is_fixed(arguments: dict[str, float | int | None], expected: str) -> None:
    assert classify_evidence(**arguments) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"evaluated_forecasts": 99, "analogue_count": 8, "average_hits": 1.0, "lower_bound": 1.0}, "insufficient"),
        ({"evaluated_forecasts": 100, "analogue_count": 7, "average_hits": 1.0, "lower_bound": 1.0}, "insufficient"),
        ({"evaluated_forecasts": 100, "analogue_count": 8, "average_hits": EXPECTED_RANDOM_HITS, "lower_bound": 0.0}, "weak"),
        ({"evaluated_forecasts": 100, "analogue_count": 8, "average_hits": 1.0, "lower_bound": 0.7}, "suggestive"),
        ({"evaluated_forecasts": 100, "analogue_count": 8, "average_hits": 1.0, "lower_bound": 0.8}, "supported"),
    ],
)
def test_forecast_evidence_is_separate_from_dynamical_evidence(
    arguments: dict[str, float | int], expected: str
) -> None:
    assert classify_forecast_evidence(**arguments) == expected  # type: ignore[arg-type]


def test_inverse_distance_weights_are_finite_normalized_and_zero_safe() -> None:
    assert _inverse_distance_weights(np.asarray([])).tolist() == []
    weights = _inverse_distance_weights(np.asarray([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(weights))
    assert weights.sum() == pytest.approx(3.0)
    assert weights[0] > weights[1] > weights[2]


def test_recurrence_model_is_causal_and_returns_complete_scores() -> None:
    model = RecurrenceDynamicsModel()
    model.train(set(_PATTERN[0]))
    first = model.predict()
    assert first.evidence.status == "insufficient"
    assert set(first.scores) == set(range(1, 50))
    assert set(first.scores.values()) == {BASE_PROBABILITY}

    model.set_pending_top_numbers((1, 2, 3, 4, 5, 6))
    model.train({1, 2, 10, 20, 30, 40})
    assert model.forecast_hits == [2]

    prediction = first
    for index in range(16):
        draw = _PATTERN[index % len(_PATTERN)]
        model.train(set(draw))
        model.observe(set(draw))
        prediction = model.predict()

    assert prediction.evidence.analogue_count == MAX_ANALOGUES
    assert 0 < prediction.evidence.effective_neighbors <= MAX_ANALOGUES
    assert 0 <= prediction.evidence.distance_percentile <= 1
    assert 0 <= prediction.evidence.score <= 1
    assert len(prediction.details) == 49
    assert len(model.pending_top_numbers or ()) == 6
    assert all(0 < score < 1 for score in prediction.scores.values())


def test_recurrence_model_excludes_the_three_draw_temporal_neighborhood(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RecurrenceDynamicsModel()
    model.draws = [
        tuple(range(start, start + 6))
        for start in (1, 7, 13, 19, 25, 31, 37)
    ]
    indexes = np.asarray([2, 3, 4, 5, 6])
    embeddings = np.asarray([[2.0], [1.0], [0.1], [0.05], [0.0]])
    monkeypatch.setattr(
        nonlinear_dynamics,
        "forecast_delay_embeddings",
        lambda _features: (indexes, embeddings),
    )

    prediction = model.predict()

    assert prediction.evidence.analogue_count == 2
    assert prediction.scores[19] > prediction.scores[31]
    assert prediction.scores[25] > prediction.scores[31]


def test_recurrence_helpers_cover_empty_and_structured_matrices() -> None:
    assert _lower_confidence_bound([]) == 0.0
    assert _lower_confidence_bound([1, 1, 1]) == 1.0
    assert _run_lengths(np.asarray([False, True, True, False, True])) == [2, 1]
    assert _run_lengths(np.asarray([False, False])) == []

    recurrence, eligible, threshold = _recurrence_matrix(np.zeros((1, 60)))
    assert recurrence.shape == (1, 1)
    assert eligible.shape == (1, 1)
    assert threshold == 0.0

    recurrence, eligible, threshold = _recurrence_matrix(np.zeros((2, 60)))
    assert threshold == 0.0
    assert not recurrence.any()
    empty_metrics = _rqa_metrics(recurrence, eligible)
    assert set(empty_metrics.values()) == {0.0}

    points = np.arange(8, dtype=float)[:, None] * np.ones((1, 60))
    recurrence, eligible, threshold = _recurrence_matrix(points)
    assert threshold > 0
    assert eligible.any()
    assert recurrence.any()

    structured = np.zeros((10, 10), dtype=bool)
    for index in range(4):
        structured[index, index + 5] = True
        structured[index + 5, index] = True
    structured[0:3, 9] = True
    metrics = _rqa_metrics(structured, np.ones_like(structured, dtype=bool))
    assert metrics["determinism"] > 0
    assert metrics["maximumDiagonalLength"] == 4
    assert metrics["laminarity"] > 0
    assert metrics["trappingTime"] >= 2


def test_nonlinear_analysis_handles_empty_and_periodic_histories() -> None:
    empty, empty_tables = nonlinear_dynamics_analysis([], surrogate_count=-1)
    assert empty["status"] == "insufficient"
    assert empty["embeddingCount"] == 0
    assert empty["surrogate"] == {
        "count": 0,
        "meanDeterminism": 0.0,
        "standardDeviation": 0.0,
        "pValue": 1.0,
    }
    assert set(empty_tables) == {
        "nonlinear_dynamics_metrics",
        "nonlinear_dynamics_forecast",
    }

    periodic = [_PATTERN[index % len(_PATTERN)] for index in range(24)]
    result, tables = nonlinear_dynamics_analysis(periodic, surrogate_count=2)
    assert result["drawCount"] == 24
    assert result["embeddingCount"] == 22
    assert result["surrogate"]["count"] == 2  # type: ignore[index]
    assert 0 < result["surrogate"]["pValue"] <= 1  # type: ignore[index,operator]
    assert result["plot"]["size"] == 22  # type: ignore[index]
    assert result["latest"]["analogueCount"] > 0  # type: ignore[index,operator]
    assert len(tables["nonlinear_dynamics_metrics"]) == 8
    assert len(tables["nonlinear_dynamics_forecast"]) == 1


def test_nonlinear_analysis_keeps_the_twenty_feature_diagnostic_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_widths: list[int] = []
    original = nonlinear_dynamics.delay_embeddings

    def tracked(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feature_widths.append(features.shape[1])
        return original(features)

    monkeypatch.setattr(nonlinear_dynamics, "delay_embeddings", tracked)
    nonlinear_dynamics_analysis(_PATTERN * 6, surrogate_count=2)

    assert feature_widths == [20, 20, 20]


def test_recurrence_prediction_is_prefix_invariant() -> None:
    prefix = [_PATTERN[index % len(_PATTERN)] for index in range(18)]

    def latest(history: list[tuple[int, ...]]) -> tuple[dict[int, float], object]:
        model = RecurrenceDynamicsModel()
        prediction = None
        for draw in history:
            model.train(set(draw))
            model.observe(set(draw))
            prediction = model.predict()
        assert prediction is not None
        return prediction.scores, prediction.evidence

    prefix_scores, prefix_evidence = latest(prefix)
    extended_model = RecurrenceDynamicsModel()
    snapshot: tuple[dict[int, float], object] | None = None
    for index, draw in enumerate([*prefix, _PATTERN[1], _PATTERN[2]]):
        extended_model.train(set(draw))
        extended_model.observe(set(draw))
        prediction = extended_model.predict()
        if index == len(prefix) - 1:
            snapshot = (prediction.scores, prediction.evidence)

    assert snapshot == (prefix_scores, prefix_evidence)
