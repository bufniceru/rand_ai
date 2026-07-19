"""Test construction, validation, properties, slots, and distances for Draw."""

from typing import cast

import pytest

from rand_ai.draw import Draw


PROPERTY_NAMES = ("num1", "num2", "num3", "num4", "num5", "num6")
DISTANCE_NAMES = ("dist1", "dist2", "dist3", "dist4", "dist5", "dist6")


class TestDrawInitialization:
    """Test construction and normalization of Draw instances."""

    def test_default_values_are_one_through_six(self) -> None:
        """Verify that constructor defaults are the first six integers."""
        draw = Draw()

        assert tuple(getattr(draw, name) for name in PROPERTY_NAMES) == (
            1,
            2,
            3,
            4,
            5,
            6,
        )

    def test_constructor_sets_all_values(self) -> None:
        """Verify that valid constructor values are stored."""
        draw = Draw(1, 2, 10, 20, 30, 49)

        assert tuple(getattr(draw, name) for name in PROPERTY_NAMES) == (
            1,
            2,
            10,
            20,
            30,
            49,
        )

    def test_constructor_sorts_numbers_in_ascending_order(self) -> None:
        """Verify that unordered constructor values are sorted."""
        draw = Draw(49, 1, 30, 10, 40, 20)

        assert tuple(getattr(draw, name) for name in PROPERTY_NAMES) == (
            1,
            10,
            20,
            30,
            40,
            49,
        )

    @pytest.mark.parametrize(
        "numbers",
        (
            (1, 1, 2, 3, 4, 5),
            (1, 2, 3, 4, 5, 5),
            (10, 20, 10, 30, 40, 49),
        ),
    )
    def test_constructor_rejects_duplicate_numbers(
        self, numbers: tuple[int, ...]
    ) -> None:
        """Verify that constructor values must be unique."""
        with pytest.raises(ValueError, match="Numbers must be unique"):
            Draw(*numbers)

    @pytest.mark.parametrize("position", range(6))
    @pytest.mark.parametrize(
        "invalid_value", ("invalid", 1.5, True, None, [], {}, object())
    )
    def test_constructor_rejects_non_integer_values(
        self, position: int, invalid_value: object
    ) -> None:
        """Verify that each constructor argument must be an integer."""
        arguments: list[object] = [1, 2, 3, 4, 5, 6]
        arguments[position] = invalid_value

        with pytest.raises(TypeError, match="Value must be an integer"):
            Draw(*cast(tuple[int, ...], tuple(arguments)))

    @pytest.mark.parametrize("position", range(6))
    @pytest.mark.parametrize("invalid_value", (-1, 0, 50, 100))
    def test_constructor_rejects_values_outside_range(
        self, position: int, invalid_value: int
    ) -> None:
        """Verify that each constructor argument must be from 1 through 49."""
        arguments = [1, 2, 3, 4, 5, 6]
        arguments[position] = invalid_value

        with pytest.raises(ValueError, match="Value must be between 1 and 49"):
            Draw(*arguments)


class TestDrawSlots:
    """Test the restricted attribute layout provided by slots."""

    def test_slots_prevent_undeclared_attributes(self) -> None:
        """Verify that Draw instances cannot gain arbitrary attributes."""
        draw = Draw()

        assert not hasattr(draw, "__dict__")
        with pytest.raises(AttributeError):
            draw.extra_value = 42


class TestDrawProperties:
    """Test the read-only number properties."""

    @pytest.mark.parametrize("property_name", PROPERTY_NAMES)
    def test_number_properties_are_read_only(self, property_name: str) -> None:
        """Verify that initialized number properties cannot be reassigned."""
        draw = Draw()

        with pytest.raises(AttributeError):
            setattr(draw, property_name, 10)


class TestDrawDistances:
    """Test calculated gaps between the draw numbers."""

    @pytest.mark.parametrize(
        ("numbers", "expected_distances"),
        (
            ((1, 2, 3, 4, 5, 6), (43, 0, 0, 0, 0, 0)),
            ((1, 10, 20, 30, 40, 49), (0, 8, 9, 9, 9, 8)),
            ((5, 12, 19, 27, 36, 45), (8, 6, 6, 7, 8, 8)),
        ),
    )
    def test_distances(
        self, numbers: tuple[int, ...], expected_distances: tuple[int, ...]
    ) -> None:
        """Verify every calculated distance for representative draws."""
        draw = Draw(*numbers)

        assert (
            tuple(getattr(draw, name) for name in DISTANCE_NAMES) == expected_distances
        )

    @pytest.mark.parametrize(
        "numbers",
        (
            (1, 2, 3, 4, 5, 6),
            (1, 10, 20, 30, 40, 49),
            (5, 12, 19, 27, 36, 45),
        ),
    )
    def test_distances_cover_all_49_values(self, numbers: tuple[int, ...]) -> None:
        """Verify that six numbers and their gaps account for all 49 values."""
        draw = Draw(*numbers)
        distances = (
            draw.dist1,
            draw.dist2,
            draw.dist3,
            draw.dist4,
            draw.dist5,
            draw.dist6,
        )

        assert sum(distances) + 6 == 49

    @pytest.mark.parametrize("distance_name", DISTANCE_NAMES)
    def test_distance_properties_are_read_only(self, distance_name: str) -> None:
        """Verify that calculated distance properties cannot be assigned."""
        draw = Draw()

        with pytest.raises(AttributeError):
            setattr(draw, distance_name, 10)
