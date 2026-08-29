"""Nonlinear recurrence diagnostics and leakage-free analogue forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

NUMBER_COUNT = 49
NUMBERS_PER_DRAW = 6
BASE_PROBABILITY = NUMBERS_PER_DRAW / NUMBER_COUNT
EXPECTED_RANDOM_HITS = NUMBERS_PER_DRAW * BASE_PROBABILITY
EMBEDDING_DIMENSION = 3
LAG_WEIGHTS = (0.5, 0.75, 1.0)
MAX_ANALOGUES = 24
THEILER_WINDOW = 3
PRIOR_STRENGTH = 8.0
MINIMUM_EVIDENCE_FORECASTS = 100
MINIMUM_CURRENT_ANALOGUES = 8
DIAGNOSTIC_WINDOW = 750
RECURRENCE_PLOT_WINDOW = 250
RECURRENCE_RATE = 0.10
SURROGATE_COUNT = 99
SURROGATE_SEED = 20260829

EvidenceStatus = Literal["insufficient", "weak", "suggestive", "supported"]

_PRIMES = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47})


@dataclass(frozen=True, slots=True)
class RecurrenceEvidence:
    """Describe causal support for one recurrence forecast."""

    status: EvidenceStatus
    score: float
    summary: str
    evaluated_forecasts: int
    analogue_count: int
    effective_neighbors: float
    distance_percentile: float
    average_hits_per_draw: float


@dataclass(frozen=True, slots=True)
class RecurrencePrediction:
    """Return number probabilities and evidence for the next draw."""

    scores: dict[int, float]
    details: dict[int, tuple[str, ...]]
    evidence: RecurrenceEvidence


def _circular_spaces(numbers: tuple[int, ...]) -> tuple[int, ...]:
    return (
        (numbers[0] - 1) + (NUMBER_COUNT - numbers[-1]),
        *(right - left - 1 for left, right in zip(numbers, numbers[1:])),
    )


def draw_features(
    numbers: tuple[int, ...],
    previous: tuple[int, ...] | None = None,
    previous_previous: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return the fixed, order-independent 20-value draw representation."""
    ordered = tuple(sorted(numbers))
    if len(ordered) != NUMBERS_PER_DRAW or len(set(ordered)) != NUMBERS_PER_DRAW:
        raise ValueError("Nonlinear dynamics requires six unique numbers")
    if ordered[0] < 1 or ordered[-1] > NUMBER_COUNT:
        raise ValueError("Draw numbers must be between 1 and 49")

    number_block = np.asarray([(number - 1) / 48 for number in ordered]) / math.sqrt(6)
    space_block = np.asarray(_circular_spaces(ordered), dtype=float) / (43 * math.sqrt(6))
    consecutive_pairs = sum(
        right - left == 1 for left, right in zip(ordered, ordered[1:])
    )
    shape_block = np.asarray(
        (
            (sum(ordered) - 21) / 258,
            (ordered[-1] - ordered[0]) / 48,
            sum(number % 2 == 1 for number in ordered) / 6,
            sum(number <= 24 for number in ordered) / 6,
            sum(number in _PRIMES for number in ordered) / 6,
            consecutive_pairs / 5,
        ),
        dtype=float,
    ) / math.sqrt(6)
    current = set(ordered)
    overlap_block = np.asarray(
        (
            len(current.intersection(previous or ())) / 6,
            len(current.intersection(previous_previous or ())) / 6,
        ),
        dtype=float,
    ) / math.sqrt(2)
    return np.concatenate((number_block, space_block, shape_block, overlap_block))


def feature_history(draws: list[tuple[int, ...]]) -> np.ndarray:
    """Build features for a chronological draw history."""
    return np.asarray(
        [
            draw_features(
                draw,
                draws[index - 1] if index >= 1 else None,
                draws[index - 2] if index >= 2 else None,
            )
            for index, draw in enumerate(draws)
        ],
        dtype=float,
    )


def delay_embeddings(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return embedding end indexes and three-draw lag vectors."""
    if len(features) < EMBEDDING_DIMENSION:
        return np.asarray([], dtype=int), np.empty((0, 60), dtype=float)
    embedded = [
        np.concatenate(
            tuple(
                features[end - (EMBEDDING_DIMENSION - 1) + offset] * LAG_WEIGHTS[offset]
                for offset in range(EMBEDDING_DIMENSION)
            )
        )
        for end in range(EMBEDDING_DIMENSION - 1, len(features))
    ]
    return (
        np.arange(EMBEDDING_DIMENSION - 1, len(features), dtype=int),
        np.asarray(embedded, dtype=float),
    )


def _lower_confidence_bound(hits: list[int]) -> float:
    if len(hits) < 2:
        return 0.0
    return fmean(hits) - 1.96 * stdev(hits) / math.sqrt(len(hits))


def classify_evidence(
    *,
    evaluated_forecasts: int,
    analogue_count: int,
    average_hits: float,
    lower_bound: float,
    surrogate_p_value: float | None,
) -> EvidenceStatus:
    """Apply the fixed v1 evidence policy."""
    if (
        evaluated_forecasts < MINIMUM_EVIDENCE_FORECASTS
        or analogue_count < MINIMUM_CURRENT_ANALOGUES
    ):
        return "insufficient"
    if (
        average_hits <= EXPECTED_RANDOM_HITS
        or (surrogate_p_value is not None and surrogate_p_value > 0.05)
    ):
        return "weak"
    if surrogate_p_value is None:
        return "suggestive"
    if lower_bound > EXPECTED_RANDOM_HITS:
        return "supported"
    return "suggestive"


def _evidence_summary(status: EvidenceStatus) -> str:
    summaries = {
        "insufficient": "Too little causal recurrence history for an evidence claim.",
        "weak": "The observed recurrence forecast does not clear the fixed evidence gates.",
        "suggestive": "Recurrence and mean walk-forward performance are suggestive, not conclusive.",
        "supported": "Recurrence and walk-forward performance clear the fixed evidence gates.",
    }
    return summaries[status]


class RecurrenceDynamicsModel:
    """Maintain causal draw history and produce local analogue forecasts."""

    def __init__(self) -> None:
        self.draws: list[tuple[int, ...]] = []
        self.forecast_hits: list[int] = []
        self.nearest_distance_history: list[float] = []
        self.pending_top_numbers: tuple[int, ...] | None = None

    def train(self, drawn: set[int]) -> None:
        """Evaluate the forecast made immediately before this draw."""
        if self.pending_top_numbers is not None:
            self.forecast_hits.append(len(drawn.intersection(self.pending_top_numbers)))
        self.pending_top_numbers = None

    def observe(self, drawn: set[int]) -> None:
        """Append the current draw after its pending forecast was evaluated."""
        self.draws.append(tuple(sorted(drawn)))

    def set_pending_top_numbers(self, numbers: tuple[int, ...]) -> None:
        """Align causal efficacy tracking with the caller's final tie-breaking."""
        self.pending_top_numbers = numbers

    def _neutral_prediction(self) -> RecurrencePrediction:
        average = fmean(self.forecast_hits) if self.forecast_hits else 0.0
        evidence = RecurrenceEvidence(
            status="insufficient",
            score=0.0,
            summary=_evidence_summary("insufficient"),
            evaluated_forecasts=len(self.forecast_hits),
            analogue_count=0,
            effective_neighbors=0.0,
            distance_percentile=1.0,
            average_hits_per_draw=average,
        )
        scores = {number: BASE_PROBABILITY for number in range(1, NUMBER_COUNT + 1)}
        details: dict[int, tuple[str, ...]] = {
            number: (
                "Neutral 6/49 prior",
                "Insufficient embedded history",
            )
            for number in scores
        }
        self.pending_top_numbers = tuple(range(1, NUMBERS_PER_DRAW + 1))
        return RecurrencePrediction(scores, details, evidence)

    def predict(self) -> RecurrencePrediction:
        """Rank the next draw from historical successors of nearby states."""
        indexes, embeddings = delay_embeddings(feature_history(self.draws))
        if len(embeddings) == 0:
            return self._neutral_prediction()
        current_end = int(indexes[-1])
        eligible_mask = indexes <= current_end - THEILER_WINDOW
        candidate_indexes = indexes[eligible_mask]
        candidates = embeddings[eligible_mask]
        if len(candidates) == 0:
            return self._neutral_prediction()

        distances = np.linalg.norm(candidates - embeddings[-1], axis=1)
        order = np.argsort(distances, kind="stable")[:MAX_ANALOGUES]
        selected_distances = distances[order]
        selected_indexes = candidate_indexes[order]
        scale = max(float(selected_distances[0]), 1e-12)
        raw_weights = np.exp(-selected_distances / scale)
        weights = raw_weights * (len(raw_weights) / float(raw_weights.sum()))
        effective_neighbors = float(weights.sum() ** 2 / np.square(weights).sum())

        nearest_distance = float(selected_distances[0])
        comparison = [*self.nearest_distance_history, nearest_distance]
        distance_percentile = sum(
            value <= nearest_distance for value in comparison
        ) / len(comparison)
        self.nearest_distance_history.append(nearest_distance)
        evidence_score = min(1.0, effective_neighbors / 12) * (1 - distance_percentile)

        weighted_hits = {number: 0.0 for number in range(1, NUMBER_COUNT + 1)}
        for end_index, weight in zip(selected_indexes, weights, strict=True):
            for number in self.draws[int(end_index) + 1]:
                weighted_hits[number] += float(weight)
        denominator = float(weights.sum()) + PRIOR_STRENGTH
        scores = {
            number: (weighted_hits[number] + PRIOR_STRENGTH * BASE_PROBABILITY)
            / denominator
            for number in weighted_hits
        }
        ranking = sorted(scores, key=lambda number: (-scores[number], number))
        self.pending_top_numbers = tuple(ranking[:NUMBERS_PER_DRAW])

        average = fmean(self.forecast_hits) if self.forecast_hits else 0.0
        status = classify_evidence(
            evaluated_forecasts=len(self.forecast_hits),
            analogue_count=len(selected_indexes),
            average_hits=average,
            lower_bound=_lower_confidence_bound(self.forecast_hits),
            surrogate_p_value=None,
        )
        evidence = RecurrenceEvidence(
            status=status,
            score=evidence_score,
            summary=_evidence_summary(status),
            evaluated_forecasts=len(self.forecast_hits),
            analogue_count=len(selected_indexes),
            effective_neighbors=effective_neighbors,
            distance_percentile=distance_percentile,
            average_hits_per_draw=average,
        )
        details: dict[int, tuple[str, ...]] = {
            number: (
                f"Posterior occurrence {scores[number]:.2%}",
                f"{len(selected_indexes)} causal analogues",
                f"Effective neighbors {effective_neighbors:.1f}",
                f"Nearest-distance percentile {distance_percentile:.1%}",
                f"Evidence {status}",
            )
            for number in scores
        }
        return RecurrencePrediction(scores, details, evidence)


def _run_lengths(values: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in values:
        if bool(value):
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _rqa_metrics(recurrence: np.ndarray, eligible: np.ndarray) -> dict[str, float]:
    recurrence_points = int(np.count_nonzero(recurrence))
    eligible_points = int(np.count_nonzero(eligible))
    diagonal_lengths = [
        length
        for offset in range(-len(recurrence) + 1, len(recurrence))
        if abs(offset) > THEILER_WINDOW
        for length in _run_lengths(np.diagonal(recurrence, offset=offset))
        if length >= 2
    ]
    vertical_lengths = [
        length
        for column in range(len(recurrence))
        for length in _run_lengths(recurrence[:, column])
        if length >= 2
    ]
    diagonal_points = sum(diagonal_lengths)
    vertical_points = sum(vertical_lengths)
    return {
        "recurrenceRate": recurrence_points / eligible_points if eligible_points else 0.0,
        "determinism": diagonal_points / recurrence_points if recurrence_points else 0.0,
        "meanDiagonalLength": fmean(diagonal_lengths) if diagonal_lengths else 0.0,
        "maximumDiagonalLength": float(max(diagonal_lengths, default=0)),
        "laminarity": vertical_points / recurrence_points if recurrence_points else 0.0,
        "trappingTime": fmean(vertical_lengths) if vertical_lengths else 0.0,
    }


def _recurrence_matrix(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    count = len(embeddings)
    if count < 2:
        empty = np.zeros((count, count), dtype=bool)
        return empty, empty, 0.0
    distances = squareform(pdist(embeddings, metric="euclidean"))
    positions = np.arange(count)
    eligible = np.abs(positions[:, None] - positions[None, :]) > THEILER_WINDOW
    eligible_distances = distances[eligible]
    threshold = (
        float(np.quantile(eligible_distances, RECURRENCE_RATE))
        if len(eligible_distances)
        else 0.0
    )
    recurrence = (distances <= threshold) & eligible
    return recurrence, eligible, threshold


def _forecast_diagnostic(draws: list[tuple[int, ...]]) -> tuple[RecurrenceDynamicsModel, RecurrencePrediction]:
    model = RecurrenceDynamicsModel()
    prediction: RecurrencePrediction | None = None
    for draw in draws:
        model.train(set(draw))
        model.observe(set(draw))
        prediction = model.predict()
    if prediction is None:
        prediction = model.predict()
    return model, prediction


def nonlinear_dynamics_analysis(
    draws: list[tuple[int, ...]],
    *,
    surrogate_count: int = SURROGATE_COUNT,
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """Build the display payload and export tables for nonlinear dynamics."""
    window = draws[-DIAGNOSTIC_WINDOW:]
    indexes, embeddings = delay_embeddings(feature_history(window))
    recurrence, eligible, threshold = _recurrence_matrix(embeddings)
    metrics = _rqa_metrics(recurrence, eligible)

    rng = np.random.default_rng(SURROGATE_SEED)
    surrogate_determinism: list[float] = []
    if len(window) >= EMBEDDING_DIMENSION:
        draw_array = np.asarray(window, dtype=int)
        for _ in range(max(surrogate_count, 0)):
            shuffled = [
                tuple(int(value) for value in row)
                for row in draw_array[rng.permutation(len(draw_array))]
            ]
            _surrogate_indexes, surrogate_embeddings = delay_embeddings(
                feature_history(shuffled)
            )
            surrogate_recurrence, surrogate_eligible, _surrogate_threshold = (
                _recurrence_matrix(surrogate_embeddings)
            )
            surrogate_determinism.append(
                _rqa_metrics(surrogate_recurrence, surrogate_eligible)["determinism"]
            )
    observed_determinism = metrics["determinism"]
    surrogate_p_value = (
        (1 + sum(value >= observed_determinism for value in surrogate_determinism))
        / (1 + len(surrogate_determinism))
        if surrogate_determinism
        else 1.0
    )

    model, latest_prediction = _forecast_diagnostic(window)
    average_hits = fmean(model.forecast_hits) if model.forecast_hits else 0.0
    lower_bound = _lower_confidence_bound(model.forecast_hits)
    status = classify_evidence(
        evaluated_forecasts=len(model.forecast_hits),
        analogue_count=latest_prediction.evidence.analogue_count,
        average_hits=average_hits,
        lower_bound=lower_bound,
        surrogate_p_value=surrogate_p_value,
    )

    plot_start = max(0, len(recurrence) - RECURRENCE_PLOT_WINDOW)
    plot = recurrence[plot_start:, plot_start:]
    plot_points = [
        {"x": int(column), "y": int(row)}
        for row, column in np.argwhere(plot)
    ]
    surrogate_mean = fmean(surrogate_determinism) if surrogate_determinism else 0.0
    surrogate_std = stdev(surrogate_determinism) if len(surrogate_determinism) > 1 else 0.0
    payload: dict[str, object] = {
        "status": status,
        "summary": _evidence_summary(status),
        "caveat": (
            "Recurrence describes repeated historical states; it does not prove "
            "that lottery draws are chaotic or predictable."
        ),
        "drawCount": len(window),
        "embeddingCount": len(indexes),
        "embeddingDimension": EMBEDDING_DIMENSION,
        "recurrenceThreshold": threshold,
        "metrics": metrics,
        "surrogate": {
            "count": len(surrogate_determinism),
            "meanDeterminism": surrogate_mean,
            "standardDeviation": surrogate_std,
            "pValue": surrogate_p_value,
        },
        "forecast": {
            "evaluatedDraws": len(model.forecast_hits),
            "averageHitsPerDraw": average_hits,
            "lowerConfidenceBound": lower_bound,
            "expectedRandomHitsPerDraw": EXPECTED_RANDOM_HITS,
        },
        "latest": {
            "analogueCount": latest_prediction.evidence.analogue_count,
            "effectiveNeighbors": latest_prediction.evidence.effective_neighbors,
            "distancePercentile": latest_prediction.evidence.distance_percentile,
            "evidenceScore": latest_prediction.evidence.score,
            "topNumbers": sorted(
                latest_prediction.scores,
                key=lambda number: (-latest_prediction.scores[number], number),
            )[:NUMBERS_PER_DRAW],
        },
        "plot": {
            "size": len(plot),
            "points": plot_points,
        },
    }
    metrics_table = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in metrics.items()
        ]
        + [
            {"metric": "surrogate_p_value", "value": surrogate_p_value},
            {"metric": "surrogate_mean_determinism", "value": surrogate_mean},
        ]
    )
    forecast_table = pd.DataFrame(
        [
            {
                "status": status,
                "evaluated_draws": len(model.forecast_hits),
                "average_hits_per_draw": average_hits,
                "lower_95_bound": lower_bound,
                "expected_random_hits_per_draw": EXPECTED_RANDOM_HITS,
                "analogue_count": latest_prediction.evidence.analogue_count,
                "effective_neighbors": latest_prediction.evidence.effective_neighbors,
                "distance_percentile": latest_prediction.evidence.distance_percentile,
            }
        ]
    )
    return payload, {
        "nonlinear_dynamics_metrics": metrics_table,
        "nonlinear_dynamics_forecast": forecast_table,
    }
