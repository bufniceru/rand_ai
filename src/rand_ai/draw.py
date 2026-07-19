"""Define a validated six-number draw and its gap distances."""


class Draw:
    """Store six unique, sorted integers from 1 through 49."""

    __slots__ = ("_num1", "_num2", "_num3", "_num4", "_num5", "_num6")

    def __init__(
        self,
        num1: int = 1,
        num2: int = 2,
        num3: int = 3,
        num4: int = 4,
        num5: int = 5,
        num6: int = 6,
    ) -> None:
        """Initialize the draw after validating and sorting its numbers."""
        numbers = self._sort_numbers(num1, num2, num3, num4, num5, num6)
        (
            self._num1,
            self._num2,
            self._num3,
            self._num4,
            self._num5,
            self._num6,
        ) = numbers

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
    def num1(self) -> int:
        """Return the first number."""
        return self._num1

    @property
    def num2(self) -> int:
        """Return the second number."""
        return self._num2

    @property
    def num3(self) -> int:
        """Return the third number."""
        return self._num3

    @property
    def num4(self) -> int:
        """Return the fourth number."""
        return self._num4

    @property
    def num5(self) -> int:
        """Return the fifth number."""
        return self._num5

    @property
    def num6(self) -> int:
        """Return the sixth number."""
        return self._num6

    @property
    def dist1(self) -> int:
        """Return the wraparound gap outside the first and sixth numbers."""
        return (self.num1 - 1) + (49 - self.num6)

    @property
    def dist2(self) -> int:
        """Return the count of values between the first and second numbers."""
        return self.num2 - self.num1 - 1

    @property
    def dist3(self) -> int:
        """Return the count of values between the second and third numbers."""
        return self.num3 - self.num2 - 1

    @property
    def dist4(self) -> int:
        """Return the count of values between the third and fourth numbers."""
        return self.num4 - self.num3 - 1

    @property
    def dist5(self) -> int:
        """Return the count of values between the fourth and fifth numbers."""
        return self.num5 - self.num4 - 1

    @property
    def dist6(self) -> int:
        """Return the count of values between the fifth and sixth numbers."""
        return self.num6 - self.num5 - 1
