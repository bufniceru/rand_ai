"""Define one lottery ball with spatial and historical metadata."""


class Ball:
    """Store a number value and its distances within a populated draw."""

    __slots__ = ("_value", "_left_dist", "_right_dist", "_gap")

    def __init__(
        self,
        value: int,
        left_dist: int = 0,
        right_dist: int = 0,
        gap: int = 0,
    ) -> None:
        """Initialize a validated ball."""
        if type(value) is not int:
            raise TypeError("Ball value must be an integer")
        if not 1 <= value <= 49:
            raise ValueError("Ball value must be between 1 and 49")
        self._value = value
        self._left_dist = self._require_non_negative_integer(
            left_dist, "left_dist"
        )
        self._right_dist = self._require_non_negative_integer(
            right_dist, "right_dist"
        )
        self._gap = self._require_non_negative_integer(gap, "gap")

    @staticmethod
    def _require_non_negative_integer(value: int, name: str) -> int:
        """Return a non-negative integer attribute value."""
        if type(value) is not int:
            raise TypeError(f"Ball {name} must be an integer")
        if value < 0:
            raise ValueError(f"Ball {name} cannot be negative")
        return value

    @property
    def value(self) -> int:
        """Return the number previously stored directly by Draw."""
        return self._value

    @property
    def left_dist(self) -> int:
        """Return the count of unused values before this ball."""
        return self._left_dist

    @property
    def right_dist(self) -> int:
        """Return the count of unused values after this ball."""
        return self._right_dist

    @property
    def gap(self) -> int:
        """Return the number of intervening draws since this value appeared."""
        return self._gap

    def _with_gap(self, gap: int) -> "Ball":
        """Return this ball's spatial metadata with a populated draw gap."""
        return Ball(self.value, self.left_dist, self.right_dist, gap)

    def __repr__(self) -> str:
        """Return all ball attributes for logs and interactive output."""
        return (
            f"Ball(value={self.value}, left_dist={self.left_dist}, "
            f"right_dist={self.right_dist}, gap={self.gap})"
        )
