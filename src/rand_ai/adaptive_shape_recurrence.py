"""Blend four source rankings using completed-forecast effectiveness."""

import math
from collections import deque
from collections.abc import Collection, Mapping

SOURCE_IDS = ("svc", "recurrence_dynamics", "emd", "positional_shape_successor_v2")
RANDOM_HITS = 36 / 49
PRIOR_FORECASTS = 24
RECENT_WINDOW = 40
MINIMUM_WEIGHT = 0.10
MAXIMUM_WEIGHT = 0.50


def bounded_weights(qualities: Mapping[str, float]) -> dict[str, float]:
    """Allocate proportional weights with 10–50% bounds and total weight one.

    Solve sum(clip(scale * quality, lower, upper)) = 1 by deterministic
    bisection. Unconstrained sources retain their relative quality proportions.
    """
    values = {source: qualities[source] for source in SOURCE_IDS}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("Source qualities must be finite and nonnegative")
    values = {source: max(value, 1e-12) for source, value in values.items()}
    total = sum(values.values())
    values = {source: value / total for source, value in values.items()}
    low, high = 0.0, 1 / min(values.values())
    for _ in range(80):
        scale = (low + high) / 2
        allocated = sum(
            min(MAXIMUM_WEIGHT, max(MINIMUM_WEIGHT, scale * value))
            for value in values.values()
        )
        if allocated < 1:
            low = scale
        else:
            high = scale
    return {
        source: min(MAXIMUM_WEIGHT, max(MINIMUM_WEIGHT, (low + high) / 2 * value))
        for source, value in values.items()
    }


class AdaptiveShapeRecurrenceModel:
    """Track independent source forecasts and build adaptive rank scores."""

    def __init__(self) -> None:
        """Initialize lifetime totals, recent hits, and pending predictions."""
        self.evaluated_forecasts = 0
        self.total_hits = dict.fromkeys(SOURCE_IDS, 0)
        self.recent_hits: dict[str, deque[int]] = {
            source: deque(maxlen=RECENT_WINDOW) for source in SOURCE_IDS
        }
        self.pending: dict[str, tuple[int, ...]] = {}

    def observe_completed(self, drawn: Collection[int]) -> None:
        """Evaluate each pending source exactly once when the target is observed."""
        if not self.pending:
            return
        actual = set(drawn)
        for source in SOURCE_IDS:
            hits = len(actual.intersection(self.pending[source]))
            self.total_hits[source] += hits
            self.recent_hits[source].append(hits)
        self.evaluated_forecasts += 1
        self.pending.clear()

    def effectiveness(self) -> dict[str, tuple[float, float]]:
        """Return independently smoothed lifetime and last-40 hit averages."""
        prior_hits = PRIOR_FORECASTS * RANDOM_HITS
        return {
            source: (
                (self.total_hits[source] + prior_hits)
                / (self.evaluated_forecasts + PRIOR_FORECASTS),
                (sum(self.recent_hits[source]) + prior_hits)
                / (len(self.recent_hits[source]) + PRIOR_FORECASTS),
            )
            for source in SOURCE_IDS
        }

    def weights(
        self, confidence: float, *, use_confidence: bool = True
    ) -> dict[str, float]:
        """Return bounded effectiveness weights, optionally adjusting V2 influence."""
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be finite and between zero and one")
        qualities = {
            source: 0.75 * lifetime + 0.25 * recent
            for source, (lifetime, recent) in self.effectiveness().items()
        }
        if use_confidence:
            qualities["positional_shape_successor_v2"] *= 0.5 + 0.5 * confidence
        return bounded_weights(qualities)

    def predict(
        self,
        rankings: dict[str, list[int]],
        confidence: float,
        *,
        use_confidence: bool = True,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Blend ranks and retain source top-six selections for the next outcome."""
        weights = self.weights(confidence, use_confidence=use_confidence)
        effectiveness = self.effectiveness()
        ranks = {
            source: {number: rank for rank, number in enumerate(rankings[source], 1)}
            for source in SOURCE_IDS
        }
        self.pending = {source: tuple(rankings[source][:6]) for source in SOURCE_IDS}
        scores = {
            number: sum(
                weights[source] * (49 - ranks[source][number]) / 48
                for source in SOURCE_IDS
            )
            for number in range(1, 50)
        }
        multiplier = 0.5 + 0.5 * confidence if use_confidence else 1.0
        details: dict[int, tuple[str, ...]] = {
            number: (
                *(
                    f"{source}: rank #{ranks[source][number]}; weight {weights[source]:.2%}; "
                    f"lifetime {effectiveness[source][0]:.3f}, "
                    f"recent {effectiveness[source][1]:.3f} hits/draw"
                    for source in SOURCE_IDS
                ),
                f"V2 similarity confidence {confidence:.1%}; influence multiplier {multiplier:.3f}",
                f"Effectiveness history: {self.evaluated_forecasts} completed forecasts; "
                "recent window 40; prior 24; lifetime/recent blend 75%/25%",
                "Adaptive rank score; not a calibrated probability",
            )
            for number in range(1, 50)
        }
        return scores, details
