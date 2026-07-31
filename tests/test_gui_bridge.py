"""Test the trusted analysis bridge consumed by the Electron application."""

import argparse
import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from zipfile import ZipFile

import pytest

import rand_ai.gui_bridge as gui_bridge
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
    portfolio_backtest_data,
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
    assert parse_report_ids("draw-portfolio,predictions") == (
        "predictions",
        "draw-portfolio",
    )
    assert parse_report_ids("") == ()
    with pytest.raises(argparse.ArgumentTypeError, match="unknown report"):
        parse_report_ids("overview,unknown")


def test_parses_strategy_plugin_selection_in_stable_order() -> None:
    assert parse_strategy_ids("entropy,proximity") == ("proximity", "entropy")
    assert parse_strategy_ids("mkrd,mknp,mksp") == ("mksp", "mknp", "mkrd")
    assert "mkrd" not in DEFAULT_STRATEGY_IDS
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
    assert payload["analysisHistory"] == [
        {"date": None, "numbers": [1, 2, 3, 4, 5, 6]},
        {"date": None, "numbers": [1, 10, 20, 30, 40, 49]},
        {"date": None, "numbers": [5, 12, 19, 27, 36, 45]},
    ]
    assert payload["history"][1]["numbers"][0] == {
        "value": 1,
        "gap": 0,
        "leftSpace": 0,
        "rightSpace": 8,
    }
    assert [
        number["rightSpace"] for number in payload["history"][1]["numbers"][:-1]
    ] == [8, 9, 9, 9, 8]
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
        item["number"] for item in payload["combinedPredictions"][-1]["numbers"][:6]
    ]
    latest_suite = payload["predictionSuites"][-1]
    assert latest_suite["referenceDrawNumber"] == 3
    assert latest_suite["actualNumbers"] == []
    assert [strategy["name"] for strategy in latest_suite["strategies"]] == [
        "Prox",
        "Fresh",
        "EMD",
        "Rand",
        "FRnd",
        "Chi²",
        "Entr",
        "Mark",
        "MKFR",
        "MKSP",
        "MKNP",
        "Baye",
        "Grid",
        "CoOc",
        "Doublet & Triplet Markov",
        "Mix",
        "SVC",
        "TBL",
        "CIS",
        "RCOV",
        "Chained Strategy",
    ]
    assert all(
        len(strategy["numbers"]) == 49
        and len(strategy["topNumbers"]) == 6
        and strategy["efficacy"]["evaluatedDraws"] == 2
        and strategy["efficacy"]["expectedRandomHits"] == pytest.approx(72 / 49)
        for strategy in latest_suite["strategies"]
    )
    random_efficacy = latest_suite["strategies"][3]["efficacy"]
    assert random_efficacy["strategyHits"] == random_efficacy["randomHits"]
    assert random_efficacy["hitDifference"] == 0
    efficacy_history = payload["strategyEfficacyHistory"]
    assert len(efficacy_history) == 2
    assert efficacy_history[0]["referenceDrawNumber"] == 1
    assert efficacy_history[0]["targetDrawNumber"] == 2
    assert efficacy_history[0]["actualNumbers"] == [1, 10, 20, 30, 40, 49]
    assert set(efficacy_history[0]["strategyHits"]) == {
        strategy["id"] for strategy in latest_suite["strategies"]
    }
    audit_history = payload["predictionAuditHistory"]
    assert len(audit_history) == 2
    assert audit_history[0]["targetDrawNumber"] == 2
    assert audit_history[0]["date"] is None
    assert [item["number"] for item in audit_history[0]["numbers"]] == [
        1,
        10,
        20,
        30,
        40,
        49,
    ]
    assert all(
        set(strategy) == {"id", "name"}
        for item in audit_history[0]["numbers"]
        for strategy in item["strategies"]
    )
    comparison = payload["latestDrawComparison"]
    assert len(payload["drawComparisonHistory"]) == 2
    assert payload["drawComparisonHistory"][-1] == comparison
    assert comparison["referenceDrawNumber"] == 2
    assert comparison["targetDrawNumber"] == 3
    assert comparison["date"] is None
    assert comparison["actualNumbers"] == [5, 12, 19, 27, 36, 45]
    assert len(comparison["strategies"]) == len(DEFAULT_STRATEGY_IDS)
    assert all(
        len(strategy["predictedNumbers"]) == 6
        and strategy["hitCount"] == len(strategy["matchedNumbers"])
        and set(strategy["matchedNumbers"]).issubset(comparison["actualNumbers"])
        and set(strategy["missedPredictions"]).isdisjoint(
            comparison["actualNumbers"]
        )
        and strategy["efficacy"]["evaluatedDraws"] == 2
        for strategy in comparison["strategies"]
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
    assert (
        sum(
            row["hits"]
            for row in payload["tables"]["freshness_gap_distribution"]["rows"]
        )
        == 18
    )
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
    assert payload["analysisHistory"] == []
    assert payload["combinedPredictions"] == []
    assert payload["predictionSuites"] == []
    assert payload["strategyEfficacyHistory"] == []
    assert payload["predictionAuditHistory"] == []
    assert payload["drawComparisonHistory"] == []
    assert payload["latestDrawComparison"] is None
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


def test_co_occurrence_report_receives_complete_history_without_autocorrelation(
    tmp_path: Path,
) -> None:
    payload = build_analysis_payload(
        _draws(),
        _pickle_path(tmp_path),
        enabled_reports=("co-occurrence",),
        enabled_strategies=(),
    )

    assert len(payload["analysisHistory"]) == 3
    assert payload["options"]["enabledReports"] == ["co-occurrence"]


def test_last_seen_space_report_receives_history(tmp_path: Path) -> None:
    payload = build_analysis_payload(
        _draws(),
        _pickle_path(tmp_path),
        enabled_reports=("last-seen-space",),
        enabled_strategies=(),
    )

    assert len(payload["history"]) == 3
    assert payload["options"]["enabledReports"] == ["last-seen-space"]


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


def test_draw_portfolio_prepares_predictions_and_relationships(
    tmp_path: Path,
) -> None:
    source_path = _pickle_path(tmp_path)
    draws = _draws()
    assert draws.draws[-1].prediction is None

    payload = build_analysis_payload(
        draws,
        source_path,
        enabled_reports=("draw-portfolio",),
    )

    assert payload["options"]["enabledReports"] == ["draw-portfolio"]
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
        strategy["id"] for strategy in payload["predictionSuites"][-1]["strategies"]
    ] == ["freshness", "entropy"]


def test_full_history_prediction_reports_use_compact_audit_records(
    tmp_path: Path,
) -> None:
    payload = build_analysis_payload(
        _draws(),
        _pickle_path(tmp_path),
        enabled_reports=("prediction-audit", "strategy-effectiveness"),
        enabled_strategies=("freshness", "entropy"),
    )

    assert payload["predictionSuites"] == []
    assert len(payload["predictionAuditHistory"]) == 2
    assert len(payload["strategyEfficacyHistory"]) == 2
    assert all(
        {
            strategy["id"]
            for item in record["numbers"]
            for strategy in item["strategies"]
        }.issubset({"freshness", "entropy"})
        for record in payload["predictionAuditHistory"]
    )


def test_draw_comparison_returns_only_the_latest_compact_result(
    tmp_path: Path,
) -> None:
    payload = build_analysis_payload(
        _draws(),
        _pickle_path(tmp_path),
        enabled_reports=("draw-comparison",),
        enabled_strategies=("freshness", "chained"),
    )

    comparison = payload["latestDrawComparison"]
    assert payload["predictionSuites"] == []
    assert payload["predictionAuditHistory"] == []
    assert payload["options"]["enabledReports"] == ["draw-comparison"]
    assert len(payload["drawComparisonHistory"]) == 2
    assert comparison["targetDrawNumber"] == 3
    assert [strategy["id"] for strategy in comparison["strategies"]] == [
        "freshness",
        "chained",
    ]
    assert all(
        set(strategy) == {
            "id",
            "name",
            "description",
            "predictedNumbers",
            "matchedNumbers",
            "missedPredictions",
            "missedActualNumbers",
            "hitCount",
            "efficacy",
        }
        for strategy in comparison["strategies"]
    )


def test_prediction_audit_keeps_drawn_numbers_when_all_strategies_are_disabled(
    tmp_path: Path,
) -> None:
    payload = build_analysis_payload(
        _draws(),
        _pickle_path(tmp_path),
        enabled_reports=("prediction-audit",),
        enabled_strategies=(),
    )

    assert len(payload["predictionAuditHistory"]) == 2
    assert all(
        len(record["numbers"]) == 6
        and all(item["strategies"] == [] for item in record["numbers"])
        for record in payload["predictionAuditHistory"]
    )


def test_strategy_cache_reuses_report_independent_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _pickle_path(tmp_path)
    cache_dir = tmp_path / "analysis-cache"
    calls = 0
    original = gui_bridge._build_strategy_analysis

    def counted_build(
        draws: Draws,
        strategy_ids: Sequence[str],
        history_start: int,
        progress: gui_bridge.ProgressCallback | None,
    ) -> gui_bridge.StrategyAnalysisArtifacts:
        nonlocal calls
        calls += 1
        return original(draws, strategy_ids, history_start, progress)

    monkeypatch.setattr(gui_bridge, "_build_strategy_analysis", counted_build)
    first_progress: list[tuple[int, str]] = []
    first = analyze_file(
        source_path,
        enabled_reports=("predictions",),
        enabled_strategies=("freshness",),
        strategy_cache_dir=cache_dir,
        progress=lambda percent, message: first_progress.append((percent, message)),
    )
    second_progress: list[tuple[int, str]] = []
    second = analyze_file(
        source_path,
        selected_numbers=(5, 12),
        trend_bins=2,
        correlation_method="spearman",
        enabled_reports=("prediction-audit", "draw-comparison"),
        enabled_strategies=("freshness",),
        strategy_cache_dir=cache_dir,
        progress=lambda percent, message: second_progress.append((percent, message)),
    )

    assert calls == 1
    assert first["predictionSuites"]
    assert second["predictionSuites"] == []
    assert second["predictionAuditHistory"]
    assert second["drawComparisonHistory"]
    assert any("No compatible cache" in message for _, message in first_progress)
    assert (31, "Loading cached strategy analysis") in second_progress
    assert (60, "Cached strategy analysis is ready") in second_progress
    cache_files = list(cache_dir.glob("*.json.gz"))
    assert len(cache_files) == 1
    with gzip.open(cache_files[0], "rt", encoding="utf-8") as cache_file:
        cached = json.load(cache_file)
    assert cached["identity"] == {
        "schemaVersion": gui_bridge.STRATEGY_CACHE_SCHEMA_VERSION,
        "datasetFingerprint": gui_bridge._dataset_fingerprint(_draws()),
        "strategyIds": ["freshness"],
        "historyLimit": gui_bridge.MAX_HISTORY_WINDOW,
    }
    assert cached["createdAt"].endswith("+00:00")
    assert len(cached["analysis"]["portfolioBacktestHistory"]) == 2
    compact = cached["analysis"]["portfolioBacktestHistory"][0]
    assert compact["targetDrawNumber"] == 2
    assert compact["strategies"][0]["id"] == "freshness"
    assert len(compact["strategies"][0]["ranking"]) == 49


def test_portfolio_backtest_data_builds_and_reuses_compact_history(
    tmp_path: Path,
) -> None:
    source_path = _pickle_path(tmp_path)
    cache_dir = tmp_path / "analysis-cache"
    first_progress: list[tuple[int, str]] = []
    first = portfolio_backtest_data(
        source_path,
        enabled_strategies=("freshness", "entropy"),
        strategy_cache_dir=cache_dir,
        progress=lambda percent, message: first_progress.append((percent, message)),
    )
    second_progress: list[tuple[int, str]] = []
    second = portfolio_backtest_data(
        source_path,
        enabled_strategies=("freshness", "entropy"),
        strategy_cache_dir=cache_dir,
        progress=lambda percent, message: second_progress.append((percent, message)),
    )

    assert len(first["cacheKey"]) == 64
    assert first["strategyIds"] == ["freshness", "entropy"]
    assert len(first["draws"]) == 3
    assert len(first["records"]) == 2
    assert first["records"][0]["actualNumbers"] == [1, 10, 20, 30, 40, 49]
    assert len(first["records"][0]["strategies"][0]["ranking"]) == 49
    assert second == first
    assert any("Preparing full-history" in message for _, message in first_progress)
    assert (60, "Loaded compact full-history strategy rankings") in second_progress


def test_strategy_cache_refresh_and_inputs_invalidate_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _pickle_path(tmp_path)
    cache_dir = tmp_path / "analysis-cache"
    calls = 0
    original = gui_bridge._build_strategy_analysis

    def counted_build(
        draws: Draws,
        strategy_ids: Sequence[str],
        history_start: int,
        progress: gui_bridge.ProgressCallback | None,
    ) -> gui_bridge.StrategyAnalysisArtifacts:
        nonlocal calls
        calls += 1
        return original(draws, strategy_ids, history_start, progress)

    monkeypatch.setattr(gui_bridge, "_build_strategy_analysis", counted_build)

    def analyze(*, refresh: bool = False, strategies: tuple[str, ...] = ("freshness",)) -> None:
        analyze_file(
            source_path,
            enabled_reports=("predictions",),
            enabled_strategies=strategies,
            strategy_cache_dir=cache_dir,
            refresh_strategy_cache=refresh,
        )

    analyze()
    analyze(refresh=True)
    analyze(strategies=("freshness", "entropy"))
    changed = _draws()
    changed.add(Draw(7, 14, 21, 28, 35, 42))
    changed.save_pickle(source_path)
    analyze()
    monkeypatch.setattr(
        gui_bridge,
        "STRATEGY_CACHE_SCHEMA_VERSION",
        gui_bridge.STRATEGY_CACHE_SCHEMA_VERSION + 1,
    )
    analyze()

    assert calls == 5
    assert len(list(cache_dir.glob("*.json.gz"))) == 4


def test_strategy_cache_recovers_from_corrupt_and_invalid_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _pickle_path(tmp_path)
    cache_dir = tmp_path / "analysis-cache"
    calls = 0
    original = gui_bridge._build_strategy_analysis

    def counted_build(
        draws: Draws,
        strategy_ids: Sequence[str],
        history_start: int,
        progress: gui_bridge.ProgressCallback | None,
    ) -> gui_bridge.StrategyAnalysisArtifacts:
        nonlocal calls
        calls += 1
        return original(draws, strategy_ids, history_start, progress)

    monkeypatch.setattr(gui_bridge, "_build_strategy_analysis", counted_build)

    def analyze() -> None:
        analyze_file(
            source_path,
            enabled_reports=("predictions",),
            enabled_strategies=("freshness",),
            strategy_cache_dir=cache_dir,
        )

    analyze()
    cache_path = next(cache_dir.glob("*.json.gz"))
    cache_path.write_bytes(b"not gzip")
    analyze()
    with gzip.open(cache_path, "wt", encoding="utf-8") as cache_file:
        json.dump({"identity": {}, "analysis": {}}, cache_file)
    analyze()

    assert calls == 3
    with gzip.open(cache_path, "rt", encoding="utf-8") as cache_file:
        assert gui_bridge._valid_strategy_artifacts(json.load(cache_file)["analysis"])


def test_strategy_cache_write_failure_is_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _pickle_path(tmp_path)
    unusable_cache_dir = tmp_path / "cache-file"
    unusable_cache_dir.write_text("not a directory", encoding="utf-8")

    payload = analyze_file(
        source_path,
        enabled_reports=("predictions",),
        enabled_strategies=("freshness",),
        strategy_cache_dir=unusable_cache_dir,
    )

    assert payload["predictionSuites"]
    assert unusable_cache_dir.is_file()

    writable_cache_dir = tmp_path / "writable-cache"
    monkeypatch.setattr(
        gui_bridge.gzip,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )
    gui_bridge._write_strategy_cache(
        writable_cache_dir,
        gui_bridge._strategy_cache_identity(_draws(), ("freshness",)),
        {
            "predictionSuites": [],
            "strategyEfficacyHistory": [],
            "predictionAuditHistory": [],
            "drawComparisonHistory": [],
        },
    )
    assert list(writable_cache_dir.iterdir()) == []


def test_strategy_cache_prunes_old_entries_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "analysis-cache"
    analysis = {
        "predictionSuites": [],
        "strategyEfficacyHistory": [],
        "predictionAuditHistory": [],
        "drawComparisonHistory": [],
    }
    monkeypatch.setattr(gui_bridge, "STRATEGY_CACHE_MAX_ENTRIES", 2)
    for index in range(3):
        gui_bridge._write_strategy_cache(
            cache_dir,
            {
                "schemaVersion": 1,
                "datasetFingerprint": str(index),
                "strategyIds": ["freshness"],
                "historyLimit": 250,
            },
            analysis,
        )

    assert len(list(cache_dir.glob("*.json.gz"))) == 2
    assert list(cache_dir.glob("*.tmp")) == []
    monkeypatch.setattr(gui_bridge, "STRATEGY_CACHE_MAX_BYTES", 1)
    gui_bridge._prune_strategy_cache(cache_dir)
    assert list(cache_dir.glob("*.json.gz")) == []


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
            "--cache-dir",
            str(tmp_path / "analysis-cache"),
            "--refresh-cache",
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


def test_cli_returns_portfolio_backtest_data(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = _pickle_path(tmp_path)
    main(
        [
            "portfolio-backtest-data",
            "--input",
            str(source_path),
            "--strategies",
            "freshness,entropy",
            "--cache-dir",
            str(tmp_path / "analysis-cache"),
        ]
    )

    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert len(payload["cacheKey"]) == 64
    assert payload["strategyIds"] == ["freshness", "entropy"]
    assert len(payload["records"]) == 2
    assert "RAND_AI_PROGRESS" in output.err


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


def test_cli_yaml_import_generates_managed_pickle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    yaml_path = tmp_path / "managed.yaml"
    pickle_path = tmp_path / "managed.pkl"
    yaml_path.write_text(
        "lotto_results:\n"
        "  draws:\n"
        "  - {date: '2026-07-20', numbers: [1, 2, 3, 4, 5, 6]}\n"
        "  - {date: '2026-07-27', numbers: [7, 8, 9, 10, 11, 12]}\n",
        encoding="utf-8",
    )

    main(
        [
            "yaml-import",
            "--input",
            str(yaml_path),
            "--output",
            str(pickle_path),
        ]
    )
    output = capsys.readouterr()
    result = json.loads(output.out)

    assert result == {
        "picklePath": str(pickle_path.resolve()),
        "drawCount": 2,
    }
    assert "RAND_AI_PROGRESS" in output.err
    assert pickle_path.is_file()
    with pickle_path.open("rb") as pickle_file:
        restored = Draws.load_trusted_pickle(
            pickle_file,
            prepare_predictions=False,
        )
    assert [draw.date for draw in restored] == ["2026-07-20", "2026-07-27"]


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
