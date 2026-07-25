"""Test the trusted analysis bridge consumed by the Electron application."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from rand_ai import Draw, Draws, create_lotto_results_pickle
from rand_ai.gui_bridge import (
    DEFAULT_REPORT_IDS,
    DEFAULT_STRATEGY_IDS,
    analyze_file,
    build_analysis_payload,
    main,
    parse_report_ids,
    parse_selected_numbers,
    parse_strategy_ids,
    write_export_archive,
)


def _draws() -> Draws:
    """Return a compact deterministic desktop dataset."""
    draws = Draws()
    draws.add(Draw())
    draws.add(Draw(1, 10, 20, 30, 40, 49))
    draws.add(Draw(5, 12, 19, 27, 36, 45))
    return draws


def _pickle_path(tmp_path: Path) -> Path:
    """Persist and return one trusted test pickle."""
    pickle_path = tmp_path / "draws.pkl"
    _draws().save_pickle(pickle_path)
    return pickle_path


def test_parses_unique_selected_numbers() -> None:
    """Verify Electron filter text becomes a sorted unique tuple."""
    assert parse_selected_numbers("5, 1,5, 49") == (1, 5, 49)


@pytest.mark.parametrize("value", ("", "0", "50", "one,two"))
def test_rejects_invalid_selected_numbers(value: str) -> None:
    """Verify malformed and out-of-range filters fail clearly."""
    with pytest.raises(argparse.ArgumentTypeError):
        parse_selected_numbers(value)


def test_parses_report_plugin_selection_in_stable_order() -> None:
    assert parse_report_ids("randomness,overview") == ("overview", "randomness")
    assert parse_report_ids("") == ()
    with pytest.raises(argparse.ArgumentTypeError, match="unknown report"):
        parse_report_ids("overview,unknown")


def test_parses_strategy_plugin_selection_in_stable_order() -> None:
    assert parse_strategy_ids("entropy,proximity") == ("proximity", "entropy")
    assert parse_strategy_ids("") == ()
    with pytest.raises(argparse.ArgumentTypeError, match="unknown prediction strategy"):
        parse_strategy_ids("freshness,unknown")


def test_builds_complete_analysis_payload(tmp_path: Path) -> None:
    """Verify tables, history, dataset metadata, and options are serialized."""
    source_path = _pickle_path(tmp_path)
    payload = build_analysis_payload(
        _draws(),
        source_path,
        selected_numbers=(1, 5),
        trend_bins=2,
        correlation_method="spearman",
    )

    assert payload["dataset"]["drawCount"] == 3
    assert payload["dataset"]["numberObservations"] == 18
    assert payload["dataset"]["historyWindowStart"] == 1
    assert payload["options"] == {
        "selectedNumbers": [1, 5],
        "trendBins": 2,
        "correlationMethod": "spearman",
        "enabledReports": list(DEFAULT_REPORT_IDS),
        "enabledStrategies": list(DEFAULT_STRATEGY_IDS),
    }
    assert len(payload["history"]) == 3
    assert payload["history"][1]["numbers"][0] == {
        "value": 1,
        "gap": 0,
        "leftSpace": 0,
        "rightSpace": 8,
    }
    assert len(payload["combinedPredictions"]) == 3
    assert payload["combinedPredictions"][0]["actualNumbers"] == [
        1,
        10,
        20,
        30,
        40,
        49,
    ]
    assert payload["combinedPredictions"][-1]["actualNumbers"] == []
    assert len(payload["combinedPredictions"][-1]["numbers"]) == 49
    assert payload["combinedPredictions"][-1]["topNumbers"] == [
        item["number"]
        for item in payload["combinedPredictions"][-1]["numbers"][:6]
    ]
    latest_suite = payload["predictionSuites"][-1]
    assert latest_suite["referenceDrawNumber"] == 3
    assert latest_suite["actualNumbers"] == []
    assert [strategy["name"] for strategy in latest_suite["strategies"]] == [
        "Prox",
        "Fresh",
        "EMD",
        "Rand",
        "Entr",
        "Mark",
        "MKFR",
        "Baye",
        "SVC",
        "TBL",
    ]
    assert all(
        len(strategy["numbers"]) == 49
        and len(strategy["topNumbers"]) == 6
        for strategy in latest_suite["strategies"]
    )
    assert payload["possibleDraw"]["lastDrawNumbers"] == [5, 12, 19, 27, 36, 45]
    assert len(payload["possibleDraw"]["lastSeenRows"]) == 49
    assert len(payload["possibleDraw"]["relationshipEdges"]) == 1176
    assert "sampled_spaces" in payload["tables"]
    assert payload["tables"]["freshness_gap_distribution"]["columns"] == [
        "gap",
        "hits",
        "opportunities",
        "hit_rate",
        "hit_percentage",
    ]
    assert sum(
        row["hits"]
        for row in payload["tables"]["freshness_gap_distribution"]["rows"]
    ) == 18
    assert payload["tables"]["number_trends"]["columns"] == [
        "bin",
        "start_draw",
        "end_draw",
        "number",
        "count",
        "appearance_rate",
    ]
    assert payload["tables"]["number_correlations_spearman"]["columns"][0] == "row"


def test_disabled_report_plugins_are_not_calculated_or_returned(
    tmp_path: Path,
) -> None:
    source_path = _pickle_path(tmp_path)
    payload = build_analysis_payload(
        _draws(),
        source_path,
        enabled_reports=("overview",),
    )

    assert payload["options"]["enabledReports"] == ["overview"]
    assert set(payload["tables"]) == {
        "summary",
        "number_frequencies",
        "draw_structure_distributions",
    }
    assert payload["history"] == []
    assert payload["combinedPredictions"] == []
    assert payload["predictionSuites"] == []
    assert payload["possibleDraw"]["relationshipEdges"] == []


def test_numbers_report_builds_shared_frequency_table_without_overview(
    tmp_path: Path,
) -> None:
    """Verify the Numbers plugin can calculate its shared table independently."""
    payload = build_analysis_payload(
        _draws(),
        _pickle_path(tmp_path),
        enabled_reports=("numbers",),
    )

    assert "number_frequencies" in payload["tables"]


def test_possible_draw_prepares_its_prediction_dependency(
    tmp_path: Path,
) -> None:
    source_path = _pickle_path(tmp_path)
    draws = _draws()
    assert draws.draws[-1].prediction is None

    payload = build_analysis_payload(
        draws,
        source_path,
        enabled_reports=("possible-draw",),
    )

    assert payload["options"]["enabledReports"] == ["possible-draw"]
    assert payload["combinedPredictions"] == []
    assert payload["predictionSuites"]
    assert payload["possibleDraw"]["relationshipEdges"]


def test_analysis_emits_only_enabled_strategy_plugins(tmp_path: Path) -> None:
    source_path = _pickle_path(tmp_path)
    payload = build_analysis_payload(
        _draws(),
        source_path,
        enabled_reports=("predictions",),
        enabled_strategies=("freshness", "entropy"),
    )

    assert payload["options"]["enabledStrategies"] == ["freshness", "entropy"]
    assert [
        strategy["id"]
        for strategy in payload["predictionSuites"][-1]["strategies"]
    ] == ["freshness", "entropy"]


def test_analyzes_and_exports_trusted_file(tmp_path: Path) -> None:
    """Verify direct bridge helpers load pickles and write the CSV archive."""
    source_path = _pickle_path(tmp_path)
    progress_events: list[tuple[int, str]] = []
    payload = analyze_file(
        source_path,
        trend_bins=2,
        progress=lambda percent, message: progress_events.append((percent, message)),
    )
    output_path = tmp_path / "exports" / "statistics.zip"

    write_export_archive(
        source_path,
        output_path,
        selected_numbers=(1, 5),
        trend_bins=2,
        correlation_method="spearman",
    )

    assert payload["dataset"]["name"] == "draws.pkl"
    percents = [percent for percent, _message in progress_events]
    assert percents[0:3] == [4, 7, 8]
    assert percents[-7:] == [61, 62, 72, 78, 82, 85, 93]
    assert percents == sorted(percents)
    with ZipFile(output_path) as archive:
        names = set(archive.namelist())
        metadata = json.loads(archive.read("metadata.json"))
    assert "tables/summary.csv" in names
    assert "tables/sampled_spaces.csv" in names
    assert metadata["selectedNumbers"] == [1, 5]
    assert metadata["correlationMethod"] == "spearman"
    assert metadata["enabledReports"] == list(DEFAULT_REPORT_IDS)
    assert metadata["enabledStrategies"] == list(DEFAULT_STRATEGY_IDS)


def test_cli_analyze_and_export_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify both subprocess commands used by Electron."""
    source_path = _pickle_path(tmp_path)
    output_path = tmp_path / "statistics.zip"

    main(
        [
            "analyze",
            "--input",
            str(source_path),
            "--selected-numbers",
            "1,5",
            "--trend-bins",
            "2",
            "--correlation-method",
            "pearson",
        ]
    )
    analyze_output = capsys.readouterr()
    payload = json.loads(analyze_output.out)
    main(
        [
            "export",
            "--input",
            str(source_path),
            "--output",
            str(output_path),
            "--selected-numbers",
            "1,5",
            "--trend-bins",
            "2",
            "--correlation-method",
            "spearman",
        ]
    )

    assert payload["dataset"]["drawCount"] == 3
    assert "RAND_AI_PROGRESS" in analyze_output.err
    assert output_path.is_file()


def test_cli_draw_editor_reads_and_saves_yaml_first(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    yaml_path = tmp_path / "managed.yaml"
    pickle_path = tmp_path / "managed.pkl"
    yaml_path.write_text(
        "lotto_results:\n"
        "  draws:\n"
        "  - {date: '2026-07-20', numbers: [1, 2, 3, 4, 5, 6]}\n",
        encoding="utf-8",
    )
    create_lotto_results_pickle(yaml_path, pickle_path)

    main(["draw-editor", "--input", str(pickle_path)])
    initial = json.loads(capsys.readouterr().out)
    main(
        [
            "draw-save",
            "--input",
            str(pickle_path),
            "--date",
            "2026-07-27",
            "--numbers",
            "7,8,9,10,11,12",
        ]
    )
    saved = json.loads(capsys.readouterr().out)

    assert initial["draws"][0]["date"] == "2026-07-20"
    assert saved["draws"][-1]["date"] == "2026-07-27"
    assert yaml_path.stat().st_mtime_ns <= pickle_path.stat().st_mtime_ns


def test_cli_draw_save_requires_exactly_six_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly six"):
        main(
            [
                "draw-save",
                "--input",
                str(tmp_path / "managed.pkl"),
                "--date",
                "2026-07-27",
                "--numbers",
                "1,2,3",
            ]
        )
