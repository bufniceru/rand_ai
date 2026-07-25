"""Provide the trusted Python analysis bridge used by the Electron desktop UI."""

import argparse
import json
import math
from collections import Counter
import sys
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import Any, cast
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from rand_ai.draws import Draws
from rand_ai.lotto_results import (
    lotto_results_editor_payload,
    resolve_lotto_results_yaml,
    upsert_lotto_result,
)
from rand_ai.statistics import CorrelationMethod, DrawsStatistics
from rand_ai.strategy_prediction import (
    PredictionSuite,
    STRATEGY_IDS,
    StrategyPrediction,
    build_prediction_suites,
)

DEFAULT_SELECTED_NUMBERS = (1, 2, 3, 4, 5, 6)
REPORT_IDS = (
    "overview",
    "numbers",
    "spaces",
    "relationships",
    "randomness",
    "gaps",
    "last-seen",
    "last-seen-gap",
    "predictions",
    "possible-draw",
)
DEFAULT_REPORT_IDS = REPORT_IDS
DEFAULT_STRATEGY_IDS = STRATEGY_IDS
MAX_HISTORY_WINDOW = 250
PROGRESS_PREFIX = "RAND_AI_PROGRESS "
ProgressCallback = Callable[[int, str], None]


def _report_progress(
    progress: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    """Send one analysis milestone when a progress consumer is available."""
    if progress is not None:
        progress(percent, message)


def _write_progress(percent: int, message: str) -> None:
    """Write one machine-readable progress event for the Electron parent."""
    payload = json.dumps({"percent": percent, "message": message}, separators=(",", ":"))
    print(f"{PROGRESS_PREFIX}{payload}", file=sys.stderr, flush=True)


def parse_selected_numbers(value: str) -> tuple[int, ...]:
    """Parse a comma-delimited, unique selection of lottery numbers."""
    try:
        numbers = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "selected numbers must be comma-delimited integers"
        ) from error
    if not numbers or any(number < 1 or number > 49 for number in numbers):
        raise argparse.ArgumentTypeError(
            "selected numbers must contain values from 1 through 49"
        )
    return numbers


def parse_report_ids(value: str) -> tuple[str, ...]:
    """Parse a comma-delimited set of supported report plugin identifiers."""
    requested = {item.strip() for item in value.split(",") if item.strip()}
    unknown = requested.difference(REPORT_IDS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown report plugin(s): {', '.join(sorted(unknown))}"
        )
    return tuple(report_id for report_id in REPORT_IDS if report_id in requested)


def parse_strategy_ids(value: str) -> tuple[str, ...]:
    """Parse a comma-delimited set of prediction strategy plugin identifiers."""
    requested = {item.strip() for item in value.split(",") if item.strip()}
    unknown = requested.difference(STRATEGY_IDS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown prediction strategy plugin(s): {', '.join(sorted(unknown))}"
        )
    return tuple(
        strategy_id for strategy_id in STRATEGY_IDS if strategy_id in requested
    )


def _table_payload(table: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-safe table while preserving meaningful indexes."""
    display = table.copy()
    if not isinstance(display.index, pd.RangeIndex):
        display.insert(0, "row", [str(value) for value in display.index])
    serialized = display.to_json(orient="records", double_precision=15)
    rows = json.loads(serialized)
    return {
        "columns": [str(column) for column in display.columns],
        "rows": rows,
    }


def _analysis_tables(
    statistics: DrawsStatistics,
    selected_numbers: Sequence[int],
    trend_bins: int,
    correlation_method: CorrelationMethod,
    enabled_reports: Collection[str],
    progress: ProgressCallback | None = None,
) -> dict[str, pd.DataFrame]:
    """Build only the tables required by enabled report plugins."""
    _report_progress(progress, 62, "Building statistical summary tables")
    reports = set(enabled_reports)
    tables: dict[str, pd.DataFrame] = {}
    if "overview" in reports:
        tables["summary"] = statistics.summary()
        tables["number_frequencies"] = statistics.number_frequencies()
        tables["draw_structure_distributions"] = (
            statistics.draw_structure_distributions()
        )
    if "numbers" in reports:
        if "number_frequencies" not in tables:
            tables["number_frequencies"] = statistics.number_frequencies()
        tables["position_frequencies"] = statistics.position_frequencies()
        tables["number_descriptive"] = statistics.number_descriptive()
        tables["pair_cooccurrence"] = statistics.pair_cooccurrence()
        _report_progress(progress, 72, "Calculating selected-number trend series")
        tables["number_trends"] = statistics.trend(
            selected_numbers, bins=trend_bins
        )
    if "spaces" in reports:
        tables["space_frequencies"] = statistics.space_frequencies()
        tables["distance_frequencies"] = statistics.distance_frequencies()
        tables["space_descriptive"] = statistics.space_descriptive()
        tables["space_extreme_distributions"] = (
            statistics.space_extreme_distributions()
        )
        _report_progress(progress, 78, "Sampling number-space combinations")
        tables["sampled_spaces"] = statistics.sampled_spaces()
    if "relationships" in reports:
        correlations = statistics.correlations(correlation_method)
        tables[f"number_correlations_{correlation_method}"] = correlations["numbers"]
        tables[f"space_correlations_{correlation_method}"] = correlations["spaces"]
        tables[f"number_space_correlations_{correlation_method}"] = correlations[
            "number_space"
        ]
    if "randomness" in reports:
        tables["randomness_diagnostics"] = statistics.randomness_diagnostics()
    if "gaps" in reports:
        tables["freshness_gap_distribution"] = (
            statistics.freshness_gap_distribution()
        )
    _report_progress(progress, 82, "Statistical tables are complete")
    return tables


def _prediction_progress(
    progress: ProgressCallback | None,
    start: int,
    end: int,
    message: str,
) -> Callable[[int, int], None]:
    """Map per-draw prediction work onto a visible percentage range."""
    last_percent = start - 1

    def report(completed: int, total: int) -> None:
        nonlocal last_percent
        percent = end if total == 0 else start + ((end - start) * completed // total)
        if percent != last_percent:
            last_percent = percent
            _report_progress(progress, percent, message)

    return report


def _strategy_payload(strategy: StrategyPrediction) -> dict[str, Any]:
    return {
        "id": strategy.strategy_id,
        "name": strategy.name,
        "description": strategy.description,
        "topNumbers": list(strategy.top_numbers),
        "numbers": [
            {
                "number": item.number,
                "rank": item.rank,
                "score": item.score,
                "gap": item.gap,
                "details": list(item.details),
            }
            for item in strategy.numbers
        ],
    }


def _suite_payload(suite: PredictionSuite) -> dict[str, Any]:
    return {
        "referenceDrawNumber": suite.reference_draw_number,
        "targetDrawNumber": suite.target_draw_number,
        "actualNumbers": list(suite.actual_numbers),
        "strategies": [
            _strategy_payload(strategy) for strategy in suite.strategies
        ],
    }


def _possible_draw_payload(draws: Draws) -> dict[str, Any]:
    """Build the all-history state needed by the Possible Draw workspace."""
    pair_counts: Counter[tuple[int, int]] = Counter()
    last_seen: dict[int, int | None] = {
        number: None for number in range(1, 50)
    }
    for draw_index, draw in enumerate(draws.draws):
        numbers = sorted(ball.value for ball in draw.balls)
        for number in numbers:
            last_seen[number] = draw_index
        for left_index, left in enumerate(numbers[:-1]):
            for right in numbers[left_index + 1 :]:
                pair_counts[(left, right)] += 1

    draw_count = len(draws)
    pair_universe = 49 * 48 / 2
    expected = draw_count * 15 / pair_universe if draw_count else 0.0
    pair_probability = 15 / pair_universe
    deviation = math.sqrt(
        max(draw_count * pair_probability * (1 - pair_probability), 0)
    )
    relationship_edges = []
    for left in range(1, 49):
        for right in range(left + 1, 50):
            count = pair_counts[(left, right)]
            relationship_edges.append(
                {
                    "left": left,
                    "right": right,
                    "count": count,
                    "expected": expected,
                    "lift": count / expected if expected else 0.0,
                    "residual": (
                        (count - expected) / deviation if deviation else 0.0
                    ),
                }
            )

    latest = draws.draws[-1] if draws.draws else None
    return {
        "lastDrawNumbers": (
            [ball.value for ball in latest.balls] if latest is not None else []
        ),
        "lastSeenRows": sorted(
            (
                {
                    "number": number,
                    "gap": (
                        draw_count
                        if draw_index is None
                        else draw_count - 1 - draw_index
                    ),
                }
                for number, draw_index in last_seen.items()
            ),
            key=lambda row: (-row["gap"], row["number"]),
        ),
        "relationshipEdges": relationship_edges,
    }


def build_analysis_payload(
    draws: Draws,
    source_path: Path,
    *,
    selected_numbers: Sequence[int] = DEFAULT_SELECTED_NUMBERS,
    trend_bins: int = 100,
    correlation_method: CorrelationMethod = "pearson",
    enabled_reports: Collection[str] = DEFAULT_REPORT_IDS,
    enabled_strategies: Collection[str] = DEFAULT_STRATEGY_IDS,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build the complete serializable payload consumed by the Vue dashboard."""
    report_ids = tuple(
        report_id for report_id in REPORT_IDS if report_id in enabled_reports
    )
    report_set = set(report_ids)
    strategy_ids = tuple(
        strategy_id
        for strategy_id in STRATEGY_IDS
        if strategy_id in enabled_strategies
    )
    history_start = max(0, len(draws) - MAX_HISTORY_WINDOW)
    prediction_suites_required = bool(
        report_set.intersection({"predictions", "possible-draw"})
    )
    prediction_prerequisites_required = (
        "predictions" in report_set
        or (prediction_suites_required and bool(strategy_ids))
    )
    if prediction_prerequisites_required:
        _report_progress(progress, 8, "Preparing prediction prerequisites")
        if draws.draws and draws.draws[-1].prediction is None:
            draws.prepare_predictions(
                _prediction_progress(
                    progress,
                    8,
                    30,
                    "Calculating prediction prerequisites draw by draw",
                )
            )
    prediction_suites: Sequence[PredictionSuite] = ()
    if prediction_suites_required:
        _report_progress(progress, 31, "Preparing named PyLotto strategy models")
        prediction_suites = build_prediction_suites(
            draws.draws,
            history_start=history_start,
            enabled_strategy_ids=strategy_ids,
            progress=_prediction_progress(
                progress,
                31,
                60,
                f"Calculating {len(strategy_ids)} enabled prediction strategy plugins",
            ),
        )
    _report_progress(progress, 61, "Validating draw history and analysis options")
    statistics = DrawsStatistics(draws, trend_bins=trend_bins)
    tables = _analysis_tables(
        statistics,
        selected_numbers,
        trend_bins,
        correlation_method,
        report_ids,
        progress,
    )
    _report_progress(progress, 85, "Preparing recent history for highlight views")
    history_required = bool(
        report_set.intersection({"last-seen", "last-seen-gap"})
    )
    history = (
        [
            {
                "drawNumber": draw_index + 1,
                "date": draw.date,
                "numbers": [
                    {
                        "value": ball.value,
                        "gap": ball.gap,
                        "leftSpace": ball.left_dist,
                        "rightSpace": ball.right_dist,
                    }
                    for ball in draw.balls
                ],
            }
            for draw_index, draw in enumerate(
                draws.draws[history_start:], history_start
            )
        ]
        if history_required
        else []
    )
    prediction_history = []
    if "predictions" in report_set:
        for draw in draws.draws[history_start:]:
            prediction = draw.prediction
            assert prediction is not None
            prediction_history.append(
                {
                    "referenceDrawNumber": prediction.reference_draw_number,
                    "targetDrawNumber": prediction.target_draw_number,
                    "actualNumbers": list(prediction.actual_numbers),
                    "topNumbers": list(prediction.top_numbers),
                    "numbers": [
                        {
                            "number": item.number,
                            "rank": item.rank,
                            "score": item.score,
                            "gap": item.gap,
                            "leftSpace": item.left_space,
                            "rightSpace": item.right_space,
                        }
                        for item in prediction.numbers
                    ],
                }
            )
    _report_progress(progress, 93, "Serializing analysis for the Electron renderer")
    return {
        "dataset": {
            "name": source_path.name,
            "path": str(source_path.resolve()),
            "sizeBytes": source_path.stat().st_size,
            "drawCount": statistics.draw_count,
            "numberObservations": statistics.draw_count * 6,
            "sampleSize": statistics.sample_size,
            "historyWindowStart": history_start + 1,
        },
        "options": {
            "selectedNumbers": list(selected_numbers),
            "trendBins": min(trend_bins, statistics.draw_count),
            "correlationMethod": correlation_method,
            "enabledReports": list(report_ids),
            "enabledStrategies": list(strategy_ids),
        },
        "tables": {
            name: _table_payload(table) for name, table in tables.items()
        },
        "history": history,
        "combinedPredictions": prediction_history,
        "predictionSuites": [
            _suite_payload(suite) for suite in prediction_suites
        ],
        "possibleDraw": (
            _possible_draw_payload(draws)
            if "possible-draw" in report_set
            else {
                "lastDrawNumbers": [],
                "lastSeenRows": [],
                "relationshipEdges": [],
            }
        ),
    }


def load_trusted_draws(
    source_path: Path,
    *,
    prepare_predictions: bool = True,
) -> Draws:
    """Load a user-confirmed trusted pickle file."""
    with source_path.open("rb") as pickle_file:
        return Draws.load_trusted_pickle(
            pickle_file,
            prepare_predictions=prepare_predictions,
        )


def analyze_file(
    source_path: Path,
    *,
    selected_numbers: Sequence[int] = DEFAULT_SELECTED_NUMBERS,
    trend_bins: int = 100,
    correlation_method: CorrelationMethod = "pearson",
    enabled_reports: Collection[str] = DEFAULT_REPORT_IDS,
    enabled_strategies: Collection[str] = DEFAULT_STRATEGY_IDS,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Load a trusted dataset and return its desktop-analysis payload."""
    _report_progress(progress, 4, "Opening the trusted pickle file")
    draws = load_trusted_draws(source_path, prepare_predictions=False)
    _report_progress(progress, 7, "Dataset loaded; checking draw records")
    return build_analysis_payload(
        draws,
        source_path,
        selected_numbers=selected_numbers,
        trend_bins=trend_bins,
        correlation_method=correlation_method,
        enabled_reports=enabled_reports,
        enabled_strategies=enabled_strategies,
        progress=progress,
    )


def write_export_archive(
    source_path: Path,
    output_path: Path,
    *,
    selected_numbers: Sequence[int] = DEFAULT_SELECTED_NUMBERS,
    trend_bins: int = 100,
    correlation_method: CorrelationMethod = "pearson",
    enabled_reports: Collection[str] = DEFAULT_REPORT_IDS,
    enabled_strategies: Collection[str] = DEFAULT_STRATEGY_IDS,
) -> None:
    """Write CSV tables and auditable metadata for the active analysis."""
    draws = load_trusted_draws(source_path)
    statistics = DrawsStatistics(draws, trend_bins=trend_bins)
    report_ids = tuple(
        report_id for report_id in REPORT_IDS if report_id in enabled_reports
    )
    tables = _analysis_tables(
        statistics,
        selected_numbers,
        trend_bins,
        correlation_method,
        report_ids,
    )
    metadata = {
        "source": str(source_path.resolve()),
        "drawCount": statistics.draw_count,
        "selectedNumbers": list(selected_numbers),
        "trendBins": min(trend_bins, statistics.draw_count),
        "correlationMethod": correlation_method,
        "enabledReports": list(report_ids),
        "enabledStrategies": [
            strategy_id
            for strategy_id in STRATEGY_IDS
            if strategy_id in enabled_strategies
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        for name, table in tables.items():
            include_index = not isinstance(table.index, pd.RangeIndex)
            archive.writestr(
                f"tables/{name}.csv",
                table.to_csv(index=include_index, index_label="row"),
            )


def _argument_parser() -> argparse.ArgumentParser:
    """Create the Electron bridge command-line parser."""
    parser = argparse.ArgumentParser(description="Rand AI Electron analysis bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "export"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--input", required=True, type=Path)
        command_parser.add_argument(
            "--selected-numbers",
            default=",".join(str(number) for number in DEFAULT_SELECTED_NUMBERS),
            type=parse_selected_numbers,
        )
        command_parser.add_argument("--trend-bins", default=100, type=int)
        command_parser.add_argument(
            "--correlation-method",
            choices=("pearson", "spearman"),
            default="pearson",
        )
        command_parser.add_argument(
            "--reports",
            default=",".join(DEFAULT_REPORT_IDS),
            type=parse_report_ids,
        )
        command_parser.add_argument(
            "--strategies",
            default=",".join(DEFAULT_STRATEGY_IDS),
            type=parse_strategy_ids,
        )
        if command == "export":
            command_parser.add_argument("--output", required=True, type=Path)
    editor_parser = subparsers.add_parser("draw-editor")
    editor_parser.add_argument("--input", required=True, type=Path)
    save_parser = subparsers.add_parser("draw-save")
    save_parser.add_argument("--input", required=True, type=Path)
    save_parser.add_argument("--date", required=True)
    save_parser.add_argument("--numbers", required=True, type=parse_selected_numbers)
    save_parser.add_argument("--original-date")
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    """Run one bridge request from Electron."""
    options = _argument_parser().parse_args(arguments)
    if options.command == "draw-editor":
        json.dump(
            lotto_results_editor_payload(options.input),
            sys.stdout,
            separators=(",", ":"),
        )
        return
    if options.command == "draw-save":
        numbers = list(cast(tuple[int, ...], options.numbers))
        if len(numbers) != 6:
            raise ValueError("A draw requires exactly six unique numbers")
        yaml_path = resolve_lotto_results_yaml(options.input)
        upsert_lotto_result(
            yaml_path,
            options.input,
            draw_date=options.date,
            numbers=numbers,
            original_date=options.original_date,
        )
        json.dump(
            lotto_results_editor_payload(options.input),
            sys.stdout,
            separators=(",", ":"),
        )
        return
    selected_numbers = cast(tuple[int, ...], options.selected_numbers)
    correlation_method = cast(CorrelationMethod, options.correlation_method)
    enabled_reports = cast(tuple[str, ...], options.reports)
    enabled_strategies = cast(tuple[str, ...], options.strategies)
    if options.command == "analyze":
        payload = analyze_file(
            options.input,
            selected_numbers=selected_numbers,
            trend_bins=options.trend_bins,
            correlation_method=correlation_method,
            enabled_reports=enabled_reports,
            enabled_strategies=enabled_strategies,
            progress=_write_progress,
        )
        _write_progress(97, "Transferring the completed analysis to Rand AI")
        json.dump(payload, sys.stdout, separators=(",", ":"))
        return
    write_export_archive(
        options.input,
        options.output,
        selected_numbers=selected_numbers,
        trend_bins=options.trend_bins,
        correlation_method=correlation_method,
        enabled_reports=enabled_reports,
        enabled_strategies=enabled_strategies,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
