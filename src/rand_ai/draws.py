"""Define a typed collection of Draw instances."""

import logging
import pickle
from collections.abc import Iterator
from pathlib import Path
from random import SystemRandom
from typing import BinaryIO

from rand_ai.draw import Draw

_LOGGER = logging.getLogger(__name__)
_RANDOM = SystemRandom()


class Draws:
    """Store an ordered collection of draws added one at a time."""

    __slots__ = ("_draws", "_last_drawn")

    _draws: list[Draw]
    _last_drawn: list[int | None]

    def __init__(self) -> None:
        """Initialize an empty collection of draws."""
        self._draws = []
        self._last_drawn = [None] * 50

    def add(self, draw: Draw) -> None:
        """Add one Draw instance to the end of the collection."""
        if not isinstance(draw, Draw):
            raise TypeError("Value must be a Draw instance")
        draw_index = len(self._draws)
        gaps: list[int] = []
        for ball in draw.balls:
            last_drawn = self._last_drawn[ball.value]
            gaps.append(
                draw_index
                if last_drawn is None
                else draw_index - last_drawn - 1
            )
        draw._set_gaps(tuple(gaps))
        self._draws.append(draw)
        for ball in draw.balls:
            self._last_drawn[ball.value] = draw_index

    def generate_random(self, number_of_draws: int) -> None:
        """Append the requested number of securely randomized draws."""
        if type(number_of_draws) is not int:
            raise TypeError("Number of draws must be an integer")
        if number_of_draws < 0:
            raise ValueError("Number of draws cannot be negative")

        for _ in range(number_of_draws):
            numbers = _RANDOM.sample(range(1, 50), 6)
            self.add(Draw(*numbers))

    def save_pickle(self, file_path: str | Path) -> None:
        """Serialize this collection to a pickle file on disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load_trusted_pickle(cls, source: BinaryIO) -> "Draws":
        """Load a Draws object from a trusted pickle binary stream.

        Pickle can execute arbitrary code during loading. Never pass data from
        an unknown or untrusted source to this method.
        """
        loaded_object = pickle.load(source)
        if not isinstance(loaded_object, cls):
            raise TypeError("Pickle must contain a Draws instance")
        return loaded_object

    def log_draws(self) -> None:
        """Log every stored draw at INFO level in insertion order."""
        for index, draw in enumerate(self._draws, start=1):
            _LOGGER.info("Draw %d: %s", index, draw.balls)

    @property
    def draws(self) -> tuple[Draw, ...]:
        """Return an immutable snapshot of the stored draws."""
        return tuple(self._draws)

    def __len__(self) -> int:
        """Return the number of stored draws."""
        return len(self._draws)

    def __iter__(self) -> Iterator[Draw]:
        """Iterate over the stored draws in insertion order."""
        return iter(self._draws)

    def __getitem__(self, index: int) -> Draw:
        """Return the draw stored at an integer index."""
        return self._draws[index]
