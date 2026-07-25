"""Test initialization, insertion, and collection behavior for Draws."""

import logging
import pickle
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest

from rand_ai import Ball, Draw, Draws


class TestDrawsInitialization:
    """Test creation and storage constraints for Draws."""

    def test_collection_starts_empty(self) -> None:
        """Verify that a new collection contains no draws."""
        draws = Draws()

        assert len(draws) == 0
        assert draws.draws == ()

    def test_slots_prevent_undeclared_attributes(self) -> None:
        """Verify that Draws instances cannot gain arbitrary attributes."""
        draws = Draws()

        assert not hasattr(draws, "__dict__")
        with pytest.raises(AttributeError):
            draws.extra_value = 42


class TestDrawsAdd:
    """Test adding Draw instances to the collection."""

    def test_adds_one_draw(self) -> None:
        """Verify that one draw can be added."""
        draws = Draws()
        draw = Draw(1, 5, 10, 20, 30, 49)

        draws.add(draw)

        assert len(draws) == 1
        assert draws.draws == (draw,)

    def test_preserves_insertion_order(self) -> None:
        """Verify that draws remain in the order in which they were added."""
        draws = Draws()
        first = Draw(1, 2, 3, 4, 5, 6)
        second = Draw(10, 20, 30, 40, 45, 49)

        draws.add(first)
        draws.add(second)

        assert draws.draws == (first, second)

    def test_populates_ball_gaps_from_prior_draws(self) -> None:
        """Verify gaps count intervening draws and elapsed initial history."""
        draws = Draws()
        first = Draw(1, 2, 3, 4, 5, 6)
        second = Draw(1, 7, 8, 9, 10, 11)
        third = Draw(2, 12, 13, 14, 15, 16)

        draws.add(first)
        draws.add(second)
        draws.add(third)

        assert tuple(ball.gap for ball in first.balls) == (0, 0, 0, 0, 0, 0)
        assert second.num1.gap == 0
        assert tuple(ball.gap for ball in second.balls[1:]) == (1, 1, 1, 1, 1)
        assert third.num1.gap == 1
        assert tuple(ball.gap for ball in third.balls[1:]) == (2, 2, 2, 2, 2)

    @pytest.mark.parametrize("invalid_value", (None, 1, "draw", [], object()))
    def test_rejects_values_that_are_not_draws(self, invalid_value: object) -> None:
        """Verify that only Draw instances can be added."""
        draws = Draws()

        with pytest.raises(TypeError, match="Value must be a Draw instance"):
            draws.add(cast(Draw, invalid_value))

        assert len(draws) == 0


class TestDrawsCollectionBehavior:
    """Test read-only access and standard collection operations."""

    def test_iterates_in_insertion_order(self) -> None:
        """Verify iteration yields each stored draw in insertion order."""
        draws = Draws()
        first = Draw()
        second = Draw(7, 8, 9, 10, 11, 12)
        draws.add(first)
        draws.add(second)

        assert list(draws) == [first, second]

    def test_returns_draw_by_index(self) -> None:
        """Verify integer indexing returns the expected draw."""
        draws = Draws()
        first = Draw()
        second = Draw(7, 8, 9, 10, 11, 12)
        draws.add(first)
        draws.add(second)

        assert draws[0] is first
        assert draws[-1] is second

    def test_draws_property_is_read_only(self) -> None:
        """Verify that the public collection snapshot cannot be replaced."""
        draws = Draws()

        with pytest.raises(AttributeError):
            setattr(draws, "draws", ())


class TestDrawsRandomGeneration:
    """Test secure random draw generation and its input validation."""

    def test_generates_requested_number_of_valid_draws(self) -> None:
        """Verify that random generation appends valid Draw instances."""
        draws = Draws()

        draws.generate_random(25)

        assert len(draws) == 25
        for draw in draws:
            assert all(isinstance(ball, Ball) for ball in draw.balls)
            numbers = tuple(ball.value for ball in draw.balls)
            assert tuple(sorted(numbers)) == numbers
            assert len(set(numbers)) == 6
            assert all(1 <= number <= 49 for number in numbers)

    @pytest.mark.parametrize("invalid_value", (None, 1.5, True, "10", object()))
    def test_rejects_non_integer_draw_count(self, invalid_value: object) -> None:
        """Verify that the requested draw count must be an integer."""
        draws = Draws()

        with pytest.raises(TypeError, match="Number of draws must be an integer"):
            draws.generate_random(cast(int, invalid_value))

    def test_rejects_negative_draw_count(self) -> None:
        """Verify that the requested draw count cannot be negative."""
        draws = Draws()

        with pytest.raises(ValueError, match="Number of draws cannot be negative"):
            draws.generate_random(-1)


class TestDrawsPersistenceAndLogging:
    """Test large collection persistence and logging without print."""

    def test_save_pickle_creates_parent_directories(self) -> None:
        """Verify persistence creates a missing destination directory tree."""
        draws = Draws()
        draws.add(Draw())

        with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            pickle_path = Path(temporary_directory) / "nested" / "data" / "draws.pkl"
            draws.save_pickle(pickle_path)

            with pickle_path.open("rb") as pickle_file:
                restored = Draws.load_trusted_pickle(pickle_file)
        assert len(restored) == 1

    def test_loads_trusted_pickle_stream(self) -> None:
        """Verify trusted binary streams restore Draws instances."""
        draws = Draws()
        draws.add(Draw())

        restored = Draws.load_trusted_pickle(BytesIO(pickle.dumps(draws)))

        assert isinstance(restored, Draws)
        assert len(restored) == 1

    def test_rejects_trusted_pickle_with_wrong_object_type(self) -> None:
        """Verify trusted pickle content must contain a Draws instance."""
        payload = BytesIO(pickle.dumps({"not": "draws"}))

        with pytest.raises(TypeError, match="Pickle must contain a Draws instance"):
            Draws.load_trusted_pickle(payload)

    def test_generates_pickles_and_logs_10000_draws(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Generate, persist, reload, and log ten thousand draws."""
        draws = Draws()
        draws.generate_random(10_000)
        pickle_path = tmp_path / "draws.pkl"
        draws.save_pickle(pickle_path)

        with pickle_path.open("rb") as pickle_file:
            restored_draws = pickle.load(pickle_file)

        assert isinstance(restored_draws, Draws)
        assert len(restored_draws) == 10_000
        assert pickle_path.is_file()

        with caplog.at_level(logging.INFO, logger="rand_ai.draws"):
            restored_draws.log_draws()

        assert len(caplog.records) == 10_000
        assert caplog.records[0].getMessage().startswith("Draw 1: ")
        assert caplog.records[-1].getMessage().startswith("Draw 10000: ")
        assert all(
            record.getMessage().count("Ball(value=") == 6
            for record in caplog.records
        )
