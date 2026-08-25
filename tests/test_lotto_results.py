"""Test importing the real lottery-results YAML into a Draws pickle."""

import logging
from pathlib import Path

import pytest
import yaml

from rand_ai import (
    Draw,
    Draws,
    create_lotto_results_pickle,
    lotto_results_editor_payload,
    resolve_lotto_results_yaml,
    upsert_lotto_result,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOTTO_RESULTS_PATH = _PROJECT_ROOT / "data" / "lotto_results.yaml"
_LOTTO_RESULTS_PICKLE_PATH = _PROJECT_ROOT / "data" / "lotto_results.pkl"


def _numbers(draw: Draw) -> tuple[int, ...]:
    """Return the six numbers from one draw."""
    return tuple(ball.value for ball in draw.balls)


def test_lotto_results_yaml_creates_draws_pickle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Load, persist, restore, and log every real YAML draw."""
    if not _LOTTO_RESULTS_PATH.is_file():
        pytest.skip("the private historical lottery dataset is not installed")
    with _LOTTO_RESULTS_PATH.open(encoding="utf-8") as yaml_file:
        yaml_results = yaml.safe_load(yaml_file)["lotto_results"]
    expected_numbers = [
        tuple(yaml_draw["numbers"]) for yaml_draw in yaml_results["draws"]
    ]
    expected_dates = [yaml_draw["date"] for yaml_draw in yaml_results["draws"]]

    imported_draws = create_lotto_results_pickle(
        _LOTTO_RESULTS_PATH,
        _LOTTO_RESULTS_PICKLE_PATH,
    )

    assert len(imported_draws) == yaml_results["total_draws"] == len(expected_numbers)
    assert [_numbers(draw) for draw in imported_draws] == expected_numbers
    assert [draw.date for draw in imported_draws] == expected_dates
    assert _LOTTO_RESULTS_PICKLE_PATH.is_file()

    with _LOTTO_RESULTS_PICKLE_PATH.open("rb") as pickle_file:
        restored_draws = Draws.load_trusted_pickle(pickle_file)

    assert len(restored_draws) == len(imported_draws)
    assert [_numbers(draw) for draw in restored_draws] == expected_numbers
    assert [draw.date for draw in restored_draws] == expected_dates

    with caplog.at_level(logging.INFO, logger="rand_ai.draws"):
        restored_draws.log_draws()

    assert len(caplog.records) == len(restored_draws)
    assert caplog.records[0].getMessage().startswith("Draw 1: ")
    assert caplog.records[-1].getMessage().startswith(
        f"Draw {len(restored_draws)}: "
    )
    assert all(
        record.getMessage().count("Ball(value=") == 6
        for record in caplog.records
    )


def test_yaml_first_add_and_edit_rebuilds_equivalent_pickle(
    tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "managed.yaml"
    pickle_path = tmp_path / "managed.pkl"
    yaml_path.write_text(
        "lotto_results:\n"
        "  total_draws: 1\n"
        "  first_draw: '2026-07-20'\n"
        "  last_draw: '2026-07-20'\n"
        "  draws:\n"
        "  - date: '2026-07-20'\n"
        "    numbers: [1, 2, 3, 4, 5, 6]\n",
        encoding="utf-8",
    )

    upsert_lotto_result(
        yaml_path,
        pickle_path,
        draw_date="2026-07-27",
        numbers=[7, 8, 9, 10, 11, 12],
    )
    upsert_lotto_result(
        yaml_path,
        pickle_path,
        draw_date="2026-07-28",
        original_date="2026-07-27",
        numbers=[13, 14, 15, 16, 17, 18],
    )

    payload = lotto_results_editor_payload(pickle_path)
    assert payload["yamlPath"] == str(yaml_path.resolve())
    assert payload["draws"] == [
        {"index": 0, "date": "2026-07-20", "numbers": [1, 2, 3, 4, 5, 6]},
        {
            "index": 1,
            "date": "2026-07-28",
            "numbers": [13, 14, 15, 16, 17, 18],
        },
    ]
    with pickle_path.open("rb") as pickle_file:
        restored = Draws.load_trusted_pickle(pickle_file, prepare_predictions=False)
    assert [draw.date for draw in restored] == ["2026-07-20", "2026-07-28"]


def test_yaml_editor_rejects_missing_or_conflicting_sources(
    tmp_path: Path,
) -> None:
    pickle_path = tmp_path / "managed.pkl"
    with pytest.raises(FileNotFoundError, match="No paired YAML"):
        resolve_lotto_results_yaml(pickle_path)

    yaml_path = tmp_path / "managed.yaml"
    yaml_path.write_text(
        "lotto_results:\n"
        "  draws:\n"
        "  - {date: '2026-07-20', numbers: [1, 2, 3, 4, 5, 6]}\n"
        "  - {date: '2026-07-27', numbers: [7, 8, 9, 10, 11, 12]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="already exists"):
        upsert_lotto_result(
            yaml_path,
            pickle_path,
            draw_date="2026-07-27",
            original_date="2026-07-20",
            numbers=[13, 14, 15, 16, 17, 18],
        )
    with pytest.raises(ValueError, match="no longer exists"):
        upsert_lotto_result(
            yaml_path,
            pickle_path,
            draw_date="2026-07-28",
            original_date="2026-07-21",
            numbers=[13, 14, 15, 16, 17, 18],
        )


def test_resolves_paired_yml_source(tmp_path: Path) -> None:
    pickle_path = tmp_path / "managed.pkl"
    yml_path = tmp_path / "managed.yml"
    yml_path.write_text("lotto_results:\n  draws: []\n", encoding="utf-8")

    assert resolve_lotto_results_yaml(pickle_path) == yml_path.resolve()


@pytest.mark.parametrize("content", ("[]\n", "other: value\n"))
def test_yaml_editor_rejects_invalid_documents(
    tmp_path: Path,
    content: str,
) -> None:
    yaml_path = tmp_path / "managed.yaml"
    yaml_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="YAML"):
        upsert_lotto_result(
            yaml_path,
            tmp_path / "managed.pkl",
            draw_date="2026-07-28",
            numbers=[1, 2, 3, 4, 5, 6],
        )
