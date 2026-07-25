"""Build walk-forward predictions from exact gap and spacing values."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rand_ai.draw import Draw

_NUMBER_COUNT = 49
_NUMBERS_PER_DRAW = 6
_BASE_HIT_RATE = _NUMBERS_PER_DRAW / _NUMBER_COUNT
_PRIOR_STRENGTH = 2.0


@dataclass(frozen=True, slots=True)
class NumberPrediction:
    """Keep one candidate's exact inputs, component scores, and combined rank."""

    number: int
    rank: int
    score: float
    freshness_score: float
    proximity_score: float
    gap: int
    left_space: int | None
    right_space: int | None
    freshness_support: int
    proximity_support: int


@dataclass(frozen=True, slots=True)
class CombinedPrediction:
    """Store a display-ready 49-number ranking after one reference draw."""

    reference_draw_number: int
    target_draw_number: int
    actual_numbers: tuple[int, ...]
    numbers: tuple[NumberPrediction, ...]
    top_numbers: tuple[int, ...]


class _PredictionEngine:
    """Learn exact-state hit rates without grouping values into buckets."""

    def __init__(self) -> None:
        self.last_seen: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.last_left_space: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.last_right_space: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.freshness_counts: dict[int, list[int]] = {}
        self.proximity_counts: dict[tuple[int, int], list[int]] = {}

    @staticmethod
    def _smoothed_rate(counts: list[int] | None) -> tuple[float, int]:
        hits, exposures = counts if counts is not None else (0, 0)
        rate = (hits + _BASE_HIT_RATE * _PRIOR_STRENGTH) / (
            exposures + _PRIOR_STRENGTH
        )
        return rate, exposures

    def _gap_before_draw(self, number: int, draw_index: int) -> int:
        seen_at = self.last_seen[number]
        return draw_index if seen_at is None else draw_index - seen_at - 1

    def learn_draw(self, draw: Draw, draw_index: int) -> None:
        """Update exact-state outcomes using a draw not used by its prior prediction."""
        drawn_numbers = {ball.value for ball in draw.balls}
        for number in range(1, _NUMBER_COUNT + 1):
            hit = int(number in drawn_numbers)
            gap = self._gap_before_draw(number, draw_index)
            freshness = self.freshness_counts.setdefault(gap, [0, 0])
            freshness[0] += hit
            freshness[1] += 1

            left_space = self.last_left_space[number]
            right_space = self.last_right_space[number]
            if left_space is not None and right_space is not None:
                proximity = self.proximity_counts.setdefault(
                    (left_space, right_space), [0, 0]
                )
                proximity[0] += hit
                proximity[1] += 1

    def observe_draw(self, draw: Draw, draw_index: int) -> None:
        """Advance every candidate's last-seen and exact spacing state."""
        for ball in draw.balls:
            self.last_seen[ball.value] = draw_index
            self.last_left_space[ball.value] = ball.left_dist
            self.last_right_space[ball.value] = ball.right_dist

    def build_prediction(
        self,
        reference_draw_number: int,
        actual_numbers: tuple[int, ...],
    ) -> CombinedPrediction:
        """Rank all numbers for the draw following the current reference."""
        reference_index = reference_draw_number - 1
        scored: list[NumberPrediction] = []
        for number in range(1, _NUMBER_COUNT + 1):
            seen_at = self.last_seen[number]
            gap = reference_draw_number if seen_at is None else reference_index - seen_at
            left_space = self.last_left_space[number]
            right_space = self.last_right_space[number]
            freshness_score, freshness_support = self._smoothed_rate(
                self.freshness_counts.get(gap)
            )
            proximity_key = (
                None
                if left_space is None or right_space is None
                else (left_space, right_space)
            )
            proximity_score, proximity_support = self._smoothed_rate(
                None
                if proximity_key is None
                else self.proximity_counts.get(proximity_key)
            )
            scored.append(
                NumberPrediction(
                    number=number,
                    rank=0,
                    score=(freshness_score + proximity_score) / 2,
                    freshness_score=freshness_score,
                    proximity_score=proximity_score,
                    gap=gap,
                    left_space=left_space,
                    right_space=right_space,
                    freshness_support=freshness_support,
                    proximity_support=proximity_support,
                )
            )

        ranked = sorted(
            scored,
            key=lambda item: (
                -item.score,
                -item.freshness_score,
                -item.proximity_score,
                -item.gap,
                item.number,
            ),
        )
        numbers = tuple(
            NumberPrediction(
                number=item.number,
                rank=rank,
                score=item.score,
                freshness_score=item.freshness_score,
                proximity_score=item.proximity_score,
                gap=item.gap,
                left_space=item.left_space,
                right_space=item.right_space,
                freshness_support=item.freshness_support,
                proximity_support=item.proximity_support,
            )
            for rank, item in enumerate(ranked, start=1)
        )
        return CombinedPrediction(
            reference_draw_number=reference_draw_number,
            target_draw_number=reference_draw_number + 1,
            actual_numbers=actual_numbers,
            numbers=numbers,
            top_numbers=tuple(item.number for item in numbers[:_NUMBERS_PER_DRAW]),
        )


def attach_combined_predictions(
    draws: Sequence[Draw],
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Calculate and attach leakage-free predictions for every reference draw."""
    engine = _PredictionEngine()
    for draw_index, draw in enumerate(draws):
        if draw_index > 0:
            engine.learn_draw(draw, draw_index)
        engine.observe_draw(draw, draw_index)
        actual_numbers = (
            tuple(ball.value for ball in draws[draw_index + 1].balls)
            if draw_index + 1 < len(draws)
            else ()
        )
        draw._set_prediction(
            engine.build_prediction(draw_index + 1, actual_numbers)
        )
        if progress is not None:
            progress(draw_index + 1, len(draws))
