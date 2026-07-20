"""Test importing the real lottery-results YAML into a Draws pickle."""

from pathlib import Path

import yaml

from rand_ai import Draw, Draws, create_lotto_results_pickle

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOTTO_RESULTS_PATH = _PROJECT_ROOT / "data" / "lotto_results.yaml"
_LOTTO_RESULTS_PICKLE_PATH = _PROJECT_ROOT / "data" / "lotto_results.pkl"


def _numbers(draw: Draw) -> tuple[int, ...]:
    """Return the six numbers from one draw."""
    return tuple(getattr(draw, f"num{position}") for position in range(1, 7))


def test_lotto_results_yaml_creates_draws_pickle() -> None:
    """Load the real YAML into a new Draws and verify its generated pickle."""
    with _LOTTO_RESULTS_PATH.open(encoding="utf-8") as yaml_file:
        yaml_results = yaml.safe_load(yaml_file)["lotto_results"]
    expected_numbers = [
        tuple(yaml_draw["numbers"]) for yaml_draw in yaml_results["draws"]
    ]

    imported_draws = create_lotto_results_pickle(
        _LOTTO_RESULTS_PATH,
        _LOTTO_RESULTS_PICKLE_PATH,
    )

    assert len(imported_draws) == yaml_results["total_draws"] == len(expected_numbers)
    assert [_numbers(draw) for draw in imported_draws] == expected_numbers
    assert _LOTTO_RESULTS_PICKLE_PATH.is_file()

    with _LOTTO_RESULTS_PICKLE_PATH.open("rb") as pickle_file:
        restored_draws = Draws.load_trusted_pickle(pickle_file)

    assert len(restored_draws) == len(imported_draws)
    assert [_numbers(draw) for draw in restored_draws] == expected_numbers
