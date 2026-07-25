"""Define a validated six-ball draw and its circular distances."""

from __future__ import annotations

from datetime import date as calendar_date
from typing import TYPE_CHECKING

from rand_ai.ball import Ball

if TYPE_CHECKING:
    from rand_ai.prediction import CombinedPrediction


class Draw:
    """Store six unique, sorted Balls with values from 1 through 49."""

    __slots__ = (
        "_num1",
        "_num2",
        "_num3",
        "_num4",
        "_num5",
        "_num6",
        "_date",
        "_prediction",
    )

    def __init__(
        self,
        num1: int = 1,
        num2: int = 2,
        num3: int = 3,
        num4: int = 4,
        num5: int = 5,
        num6: int = 6,
        *,
        date: str | None = None,
    ) -> None:
        """Initialize the draw after validating and sorting its numbers."""
        numbers = self._sort_numbers(num1, num2, num3, num4, num5, num6)
        self._date = self._validate_date(date)
        self._prediction: CombinedPrediction | None = None
        left_distances = (
            (numbers[0] - 1) + (49 - numbers[5]),
            *(right - left - 1 for left, right in zip(numbers, numbers[1:])),
        )
        right_distances = (*left_distances[1:], left_distances[0])
        (
            self._num1,
            self._num2,
            self._num3,
            self._num4,
            self._num5,
            self._num6,
        ) = tuple(
            Ball(value, left_dist, right_dist)
            for value, left_dist, right_dist in zip(
                numbers, left_distances, right_distances, strict=True
            )
        )

    @staticmethod
    def _validate_date(value: str | None) -> str | None:
        """Return an ISO calendar date or preserve a missing legacy date."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("Date must be an ISO date string")
        try:
            return calendar_date.fromisoformat(value).isoformat()
        except ValueError as error:
            raise ValueError("Date must use YYYY-MM-DD") from error

    @staticmethod
    def _require_integer(value: int) -> int:
        """Return a valid integer from 1 through 49."""
        if type(value) is not int:
            raise TypeError("Value must be an integer")
        if not 1 <= value <= 49:
            raise ValueError("Value must be between 1 and 49")
        return value

    @classmethod
    def _sort_numbers(cls, *numbers: int) -> tuple[int, ...]:
        """Validate unique numbers and return them in ascending order."""
        validated_numbers = tuple(cls._require_integer(number) for number in numbers)
        if len(set(validated_numbers)) != len(validated_numbers):
            raise ValueError("Numbers must be unique")
        return tuple(sorted(validated_numbers))

    @property
    def num1(self) -> Ball:
        """Return the first ball."""
        return self._num1

    @property
    def num2(self) -> Ball:
        """Return the second ball."""
        return self._num2

    @property
    def num3(self) -> Ball:
        """Return the third ball."""
        return self._num3

    @property
    def num4(self) -> Ball:
        """Return the fourth ball."""
        return self._num4

    @property
    def num5(self) -> Ball:
        """Return the fifth ball."""
        return self._num5

    @property
    def num6(self) -> Ball:
        """Return the sixth ball."""
        return self._num6

    @property
    def dist1(self) -> int:
        """Return the wraparound gap outside the first and sixth numbers."""
        return self.num1.left_dist

    @property
    def dist2(self) -> int:
        """Return the count of values between the first and second numbers."""
        return self.num1.right_dist

    @property
    def dist3(self) -> int:
        """Return the count of values between the second and third numbers."""
        return self.num2.right_dist

    @property
    def dist4(self) -> int:
        """Return the count of values between the third and fourth numbers."""
        return self.num3.right_dist

    @property
    def dist5(self) -> int:
        """Return the count of values between the fourth and fifth numbers."""
        return self.num4.right_dist

    @property
    def dist6(self) -> int:
        """Return the count of values between the fifth and sixth numbers."""
        return self.num5.right_dist

    @property
    def balls(self) -> tuple[Ball, ...]:
        """Return all six balls in ascending value order."""
        return (
            self.num1,
            self.num2,
            self.num3,
            self.num4,
            self.num5,
            self.num6,
        )

    @property
    def date(self) -> str | None:
        """Return the draw date, or None for a legacy undated pickle."""
        return getattr(self, "_date", None)

    def _set_gaps(self, gaps: tuple[int, ...]) -> None:
        """Populate each ball's historical gap when this draw is inserted."""
        if len(gaps) != 6:
            raise ValueError("A Draw requires exactly six gap values")
        (
            self._num1,
            self._num2,
            self._num3,
            self._num4,
            self._num5,
            self._num6,
        ) = tuple(
            ball._with_gap(gap)
            for ball, gap in zip(self.balls, gaps, strict=True)
        )

    @property
    def prediction(self) -> CombinedPrediction | None:
        """Return the precomputed raw freshness/proximity prediction."""
        return self._prediction

    def _set_prediction(self, prediction: CombinedPrediction) -> None:
        """Attach display-ready prediction information during dataset import."""
        self._prediction = prediction
