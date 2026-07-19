"""Generate a persistent randomized Draws pickle dataset."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from rand_ai import Draws

_LOGGER = logging.getLogger(__name__)
_DEFAULT_DRAW_COUNT = 10_000
_DEFAULT_OUTPUT = Path("data/draws.pkl")


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate randomized draws and save them as a trusted pickle."
    )
    parser.add_argument(
        "--count",
        default=_DEFAULT_DRAW_COUNT,
        type=int,
        help=f"Number of draws to generate (default: {_DEFAULT_DRAW_COUNT}).",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        type=Path,
        help=f"Destination pickle path (default: {_DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(arguments)


def generate_draws_pickle(number_of_draws: int, output: Path) -> Draws:
    """Generate randomized draws and persist them at the requested path."""
    draws = Draws()
    draws.generate_random(number_of_draws)
    draws.save_pickle(output)
    return draws


def main(arguments: Sequence[str] | None = None) -> None:
    """Generate and save a dataset using command-line arguments."""
    options = _parse_arguments(arguments)
    draws = generate_draws_pickle(options.count, options.output)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _LOGGER.info(
        "Saved %d draws to %s",
        len(draws),
        options.output.resolve(),
    )


if __name__ == "__main__":
    main()
