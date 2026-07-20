"""Test Ball validation, metadata, representation, and restricted attributes."""

from typing import cast

import pytest

from rand_ai import Ball


class TestBall:
    """Test the value and metadata stored by one Ball."""

    def test_stores_all_attributes_and_represents_them(self) -> None:
        """Verify all four read-only attributes and the diagnostic representation."""
        ball = Ball(17, left_dist=4, right_dist=8, gap=12)

        assert (
            ball.value,
            ball.left_dist,
            ball.right_dist,
            ball.gap,
        ) == (17, 4, 8, 12)
        assert repr(ball) == (
            "Ball(value=17, left_dist=4, right_dist=8, gap=12)"
        )

    def test_defaults_metadata_to_zero(self) -> None:
        """Verify a standalone Ball has neutral metadata defaults."""
        ball = Ball(1)

        assert (ball.left_dist, ball.right_dist, ball.gap) == (0, 0, 0)

    @pytest.mark.parametrize("invalid_value", ("1", 1.5, True, None, object()))
    def test_rejects_non_integer_value(self, invalid_value: object) -> None:
        """Verify the lottery value must be an integer."""
        with pytest.raises(TypeError, match="Ball value must be an integer"):
            Ball(cast(int, invalid_value))

    @pytest.mark.parametrize("invalid_value", (0, 50))
    def test_rejects_value_outside_lottery_range(self, invalid_value: int) -> None:
        """Verify the lottery value remains between 1 and 49."""
        with pytest.raises(ValueError, match="Ball value must be between 1 and 49"):
            Ball(invalid_value)

    @pytest.mark.parametrize("attribute", ("left_dist", "right_dist", "gap"))
    def test_rejects_non_integer_metadata(self, attribute: str) -> None:
        """Verify every metadata attribute must be an integer."""
        arguments = {"value": 1, attribute: 1.5}

        with pytest.raises(TypeError, match=f"Ball {attribute} must be an integer"):
            Ball(**cast(dict[str, int], arguments))

    @pytest.mark.parametrize("attribute", ("left_dist", "right_dist", "gap"))
    def test_rejects_negative_metadata(self, attribute: str) -> None:
        """Verify every metadata attribute must be non-negative."""
        arguments = {"value": 1, attribute: -1}

        with pytest.raises(ValueError, match=f"Ball {attribute} cannot be negative"):
            Ball(**arguments)

    @pytest.mark.parametrize(
        "attribute", ("value", "left_dist", "right_dist", "gap")
    )
    def test_attributes_are_read_only(self, attribute: str) -> None:
        """Verify public Ball attributes cannot be reassigned."""
        ball = Ball(1)

        with pytest.raises(AttributeError):
            setattr(ball, attribute, 2)

    def test_slots_prevent_undeclared_attributes(self) -> None:
        """Verify Ball instances cannot gain arbitrary attributes."""
        ball = Ball(1)

        assert not hasattr(ball, "__dict__")
        with pytest.raises(AttributeError):
            ball.extra_value = 42
