"""Provide the trusted Python analysis bridge used by the Electron desktop UI."""

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from rand_ai.draws import Draws
from rand_ai.lotto_results import (
    create_lotto_results_pickle,
    lotto_results_editor_payload,
    resolve_lotto_results_yaml,
    upsert_lotto_result,
)
from rand_ai.nonlinear_dynamics import nonlinear_dynamics_analysis
from rand_ai.statistics import CorrelationMethod, DrawsStatistics
from rand_ai.space_groups import (
    DEFAULT_BORDER_SPACE,
    validate_border_space,
    validate_target_group_count,
)
from rand_ai.strategy_prediction import (
    PredictionSuite,
    STRATEGY_IDS,
    StrategyEfficacyRecord,
    StrategyPrediction,
    build_prediction_suites,
)

DEFAULT_SELECTED_NUMBERS = (1, 2, 3, 4, 5, 6)
REPORT_IDS = (
    "overview",
    "numbers",
    "spaces",
    "space-groups",
    "relationships",
    "randomness",
    "nonlinear-dynamics",
    "autocorrelation",
    "co-occurrence",
    "prediction-audit",
    "draw-comparison",
    "strategy-effectiveness",
    "gaps",
    "last-seen",
    "last-seen-gap",
    "last-seen-space",
    "predictions",
    "draw-portfolio",
    "possible-draw",
)
DEFAULT_REPORT_IDS = REPORT_IDS
DEFAULT_STRATEGY_IDS = tuple(
    strategy_id
    for strategy_id in STRATEGY_IDS
    if strategy_id
    not in {
        "mkrd",
        "mkgsv",
        "sklearn_svm",
        "lag_logistic",
        "sparse_neural_ticket",
        "decision_tree_selector",
        "recurrence_dynamics",
        "svc_recurrence_hybrid",
        "svc_recurrence_proximity_hybrid",
        "srph_residual_diversity_hybrid",
        "srph_minimax_regret_hybrid",
    }
)
MAX_HISTORY_WINDOW = 250
STATISTICS_COMMAND_IDS = (
    "statistics.number-frequency",
    "statistics.group-frequency",
)
STRATEGY_CACHE_SCHEMA_VERSION = 20
STRATEGY_CACHE_MAX_ENTRIES = 20
STRATEGY_CACHE_MAX_BYTES = 1024 * 1024 * 1024
PROGRESS_PREFIX = "RAND_AI_PROGRESS "
ProgressCallback = Callable[[int, str], None]
StrategyAnalysisArtifacts = dict[str, list[dict[str, Any]]]


def _dataset_fingerprint(draws: Draws) -> str:
    """Return a stable digest of dates and sorted draw values."""
    history = [
        [draw.date, *sorted(ball.value for ball in draw.balls)]
        for draw in draws.draws
    ]
    encoded = json.dumps(history, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strategy_cache_identity(
    draws: Draws,
    strategy_ids: Sequence[str],
    border_space: int = DEFAULT_BORDER_SPACE,
    target_group_count: int | None = None,
) -> dict[str, Any]:
    """Describe every input that changes cached walk-forward results."""
    return {
        "schemaVersion": STRATEGY_CACHE_SCHEMA_VERSION,
        "datasetFingerprint": _dataset_fingerprint(draws),
        "strategyIds": list(strategy_ids),
        "borderSpace": validate_border_space(border_space),
        "targetGroupCount": validate_target_group_count(target_group_count),
        "historyLimit": MAX_HISTORY_WINDOW,
    }


def _strategy_cache_path(
    cache_dir: Path,
    identity: dict[str, Any],
) -> Path:
    """Return the content-addressed compressed cache path."""
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()
    return cache_dir / f"{key}.json.gz"


def _valid_strategy_artifacts(value: object) -> bool:
    """Return whether a decoded cache has all required artifact arrays."""
    required = (
        "predictionSuites",
        "strategyEfficacyHistory",
        "predictionAuditHistory",
        "drawComparisonHistory",
        "portfolioBacktestHistory",
    )
    return isinstance(value, dict) and all(
        isinstance(value.get(name), list) for name in required
    )


def _discard_cache_file(path: Path) -> None:
    """Remove one invalid internal cache entry on a best-effort basis."""
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - a concurrently locked cache is harmless
        pass


def _read_strategy_cache(
    cache_dir: Path,
    identity: dict[str, Any],
) -> StrategyAnalysisArtifacts | None:
    """Load one valid compressed strategy artifact and refresh its age."""
    path = _strategy_cache_path(cache_dir, identity)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        _discard_cache_file(path)
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("identity") != identity
        or not _valid_strategy_artifacts(payload.get("analysis"))
    ):
        _discard_cache_file(path)
        return None
    try:
        path.touch()
    except OSError:  # pragma: no cover - cache aging is best effort
        pass
    return cast(StrategyAnalysisArtifacts, payload["analysis"])


def _prune_strategy_cache(cache_dir: Path) -> None:
    """Keep only the newest bounded set of completed cache entries."""
    try:
        entries = [
            (path, path.stat()) for path in cache_dir.glob("*.json.gz")
        ]
    except OSError:  # pragma: no cover - pruning must not break analysis
        return
    entries.sort(key=lambda item: item[1].st_mtime_ns, reverse=True)
    total_bytes = sum(stat.st_size for _path, stat in entries)
    for index, (path, stat) in enumerate(entries):
        if (
            index < STRATEGY_CACHE_MAX_ENTRIES
            and total_bytes <= STRATEGY_CACHE_MAX_BYTES
        ):
            continue
        try:
            path.unlink()
            total_bytes -= stat.st_size
        except OSError:  # pragma: no cover - another process may own the file
            pass


def _write_strategy_cache(
    cache_dir: Path,
    identity: dict[str, Any],
    analysis: StrategyAnalysisArtifacts,
) -> None:
    """Atomically store one safe compressed JSON cache entry."""
    temporary_path: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = _strategy_cache_path(cache_dir, identity)
        with NamedTemporaryFile(
            dir=cache_dir,
            prefix=".strategy-analysis-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        with gzip.open(temporary_path, "wt", encoding="utf-8") as cache_file:
            json.dump(
                {
                    "identity": identity,
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "analysis": analysis,
                },
                cache_file,
                separators=(",", ":"),
            )
        temporary_path.replace(destination)
        _prune_strategy_cache(cache_dir)
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            _discard_cache_file(temporary_path)


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
        tables["draw_structure_distributions"] = (
            statistics.draw_structure_distributions()
        )
    if "numbers" in reports:
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
    efficacy = strategy.efficacy
    evidence = strategy.evidence
    return {
        "id": strategy.strategy_id,
        "name": strategy.name,
        "description": strategy.description,
        "topNumbers": list(strategy.top_numbers),
        "efficacy": (
            None
            if efficacy is None
            else {
                "evaluatedDraws": efficacy.evaluated_draws,
                "strategyHits": efficacy.strategy_hits,
                "randomHits": efficacy.random_hits,
                "expectedRandomHits": efficacy.expected_random_hits,
                "averageHitsPerDraw": efficacy.average_hits_per_draw,
                "randomAverageHitsPerDraw": (
                    efficacy.random_average_hits_per_draw
                ),
                "hitDifference": efficacy.hit_difference,
            }
        ),
        "evidence": (
            None
            if evidence is None
            else {
                "status": evidence.status,
                "score": evidence.score,
                "summary": evidence.summary,
                "evaluatedForecasts": evidence.evaluated_forecasts,
                "analogueCount": evidence.analogue_count,
                "effectiveNeighbors": evidence.effective_neighbors,
                "distancePercentile": evidence.distance_percentile,
                "averageHitsPerDraw": evidence.average_hits_per_draw,
            }
        ),
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


def _draw_comparison_payload(
    suite: PredictionSuite,
    target_date: str | None,
) -> dict[str, Any]:
    """Return a compact comparison of one completed draw and its prior forecast."""
    actual = set(suite.actual_numbers)
    return {
        "referenceDrawNumber": suite.reference_draw_number,
        "targetDrawNumber": suite.target_draw_number,
        "date": target_date,
        "actualNumbers": list(suite.actual_numbers),
        "strategies": [
            {
                "id": strategy.strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "predictedNumbers": list(strategy.top_numbers),
                "matchedNumbers": [
                    number for number in strategy.top_numbers if number in actual
                ],
                "missedPredictions": [
                    number for number in strategy.top_numbers if number not in actual
                ],
                "missedActualNumbers": [
                    number
                    for number in suite.actual_numbers
                    if number not in strategy.top_numbers
                ],
                "hitCount": len(actual.intersection(strategy.top_numbers)),
                "efficacy": (
                    None
                    if strategy.efficacy is None
                    else {
                        "evaluatedDraws": strategy.efficacy.evaluated_draws,
                        "strategyHits": strategy.efficacy.strategy_hits,
                        "randomHits": strategy.efficacy.random_hits,
                        "expectedRandomHits": (
                            strategy.efficacy.expected_random_hits
                        ),
                        "averageHitsPerDraw": (
                            strategy.efficacy.average_hits_per_draw
                        ),
                        "randomAverageHitsPerDraw": (
                            strategy.efficacy.random_average_hits_per_draw
                        ),
                        "hitDifference": strategy.efficacy.hit_difference,
                    }
                ),
            }
            for strategy in suite.strategies
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


def _efficacy_record_payload(
    record: StrategyEfficacyRecord,
) -> dict[str, Any]:
    return {
        "referenceDrawNumber": record.reference_draw_number,
        "targetDrawNumber": record.target_draw_number,
        "actualNumbers": list(record.actual_numbers),
        "randomHits": record.random_hits,
        "strategyHits": dict(record.strategy_hits),
    }


def _prediction_audit_record_payload(
    suite: PredictionSuite,
    target_date: str | None,
) -> dict[str, Any]:
    """Return the strategies that correctly implied each drawn number."""
    return {
        "referenceDrawNumber": suite.reference_draw_number,
        "targetDrawNumber": suite.target_draw_number,
        "date": target_date,
        "numbers": [
            {
                "number": number,
                "strategies": [
                    {
                        "id": strategy.strategy_id,
                        "name": strategy.name,
                    }
                    for strategy in suite.strategies
                    if number in strategy.top_numbers
                ],
            }
            for number in suite.actual_numbers
        ],
    }


def _build_strategy_analysis(
    draws: Draws,
    strategy_ids: Sequence[str],
    history_start: int,
    progress: ProgressCallback | None,
    border_space: int = DEFAULT_BORDER_SPACE,
    target_group_count: int | None = None,
) -> StrategyAnalysisArtifacts:
    """Calculate the report-independent walk-forward strategy artifacts."""
    efficacy_records: list[StrategyEfficacyRecord] = []
    prediction_audit_history: list[dict[str, Any]] = []
    draw_comparison_history: list[dict[str, Any]] = []
    portfolio_backtest_history: list[dict[str, Any]] = []

    def record_evaluated_suite(suite: PredictionSuite) -> None:
        target_index = suite.target_draw_number - 1
        target_date = (
            draws.draws[target_index].date
            if 0 <= target_index < len(draws.draws)
            else None
        )
        prediction_audit_history.append(
            _prediction_audit_record_payload(suite, target_date)
        )
        draw_comparison_history.append(
            _draw_comparison_payload(suite, target_date)
        )
        portfolio_backtest_history.append(
            {
                "referenceDrawNumber": suite.reference_draw_number,
                "targetDrawNumber": suite.target_draw_number,
                "date": target_date,
                "actualNumbers": list(suite.actual_numbers),
                "strategies": [
                    {
                        "id": strategy.strategy_id,
                        "ranking": [item.number for item in strategy.numbers],
                    }
                    for strategy in suite.strategies
                ],
            }
        )

    prediction_suites = build_prediction_suites(
        draws.draws,
        history_start=history_start,
        enabled_strategy_ids=strategy_ids,
        border_space=border_space,
        target_group_count=target_group_count,
        progress=_prediction_progress(
            progress,
            31,
            60,
            f"Calculating {len(strategy_ids)} enabled prediction strategy plugins",
        ),
        efficacy_record=efficacy_records.append,
        evaluated_suite=record_evaluated_suite,
    )
    return {
        "predictionSuites": [
            _suite_payload(suite) for suite in prediction_suites
        ],
        "strategyEfficacyHistory": [
            _efficacy_record_payload(record) for record in efficacy_records
        ],
        "predictionAuditHistory": prediction_audit_history,
        "drawComparisonHistory": draw_comparison_history,
        "portfolioBacktestHistory": portfolio_backtest_history,
    }


def _strategy_analysis(
    draws: Draws,
    strategy_ids: Sequence[str],
    history_start: int,
    progress: ProgressCallback | None,
    cache_dir: Path | None,
    refresh_cache: bool,
    border_space: int = DEFAULT_BORDER_SPACE,
    target_group_count: int | None = None,
) -> StrategyAnalysisArtifacts:
    """Load cached strategy artifacts or calculate and persist them."""
    identity = _strategy_cache_identity(
        draws, strategy_ids, border_space, target_group_count
    )
    cached = (
        None
        if cache_dir is None or refresh_cache or not strategy_ids
        else _read_strategy_cache(cache_dir, identity)
    )
    if cached is not None:
        _report_progress(progress, 31, "Loading cached strategy analysis")
        _report_progress(progress, 60, "Cached strategy analysis is ready")
        return cached
    if cache_dir is not None and strategy_ids:
        _report_progress(
            progress,
            31,
            "No compatible cache; calculating prediction strategies",
        )
    analysis = _build_strategy_analysis(
        draws,
        strategy_ids,
        history_start,
        progress,
        border_space,
        target_group_count,
    )
    if cache_dir is not None and strategy_ids:
        _write_strategy_cache(cache_dir, identity, analysis)
    return analysis


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
    border_space: int = DEFAULT_BORDER_SPACE,
    target_group_count: int | None = None,
    enabled_reports: Collection[str] = DEFAULT_REPORT_IDS,
    enabled_strategies: Collection[str] = DEFAULT_STRATEGY_IDS,
    progress: ProgressCallback | None = None,
    strategy_cache_dir: Path | None = None,
    refresh_strategy_cache: bool = False,
) -> dict[str, Any]:
    """Build the complete serializable payload consumed by the Vue dashboard."""
    border_space = validate_border_space(border_space)
    target_group_count = validate_target_group_count(target_group_count)
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
        report_set.intersection(
            {
                "predictions",
                "draw-portfolio",
                "possible-draw",
                "prediction-audit",
                "draw-comparison",
                "strategy-effectiveness",
            }
        )
    )
    display_prediction_suites_required = bool(
        report_set.intersection({"predictions", "draw-portfolio", "possible-draw"})
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
    strategy_analysis: StrategyAnalysisArtifacts = {
        "predictionSuites": [],
        "strategyEfficacyHistory": [],
        "predictionAuditHistory": [],
        "drawComparisonHistory": [],
    }
    if prediction_suites_required:
        _report_progress(progress, 31, "Preparing named PyLotto strategy models")
        strategy_analysis = _strategy_analysis(
            draws,
            strategy_ids,
            history_start,
            progress,
            strategy_cache_dir,
            refresh_strategy_cache,
            border_space,
            target_group_count,
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
    nonlinear_dynamics_payload: dict[str, object] | None = None
    if "nonlinear-dynamics" in report_set:
        nonlinear_dynamics_payload, nonlinear_tables = nonlinear_dynamics_analysis(
            [tuple(ball.value for ball in draw.balls) for draw in draws.draws]
        )
        tables.update(nonlinear_tables)
    space_group_payload: dict[str, object] | None = None
    if "space-groups" in report_set:
        _report_progress(progress, 83, "Analyzing border-space groups")
        group_tables, space_group_payload = statistics.space_group_analysis(
            border_space, target_group_count
        )
        tables.update(group_tables)
    _report_progress(progress, 85, "Preparing recent history for highlight views")
    history_required = bool(
        report_set.intersection(
            {"last-seen", "last-seen-gap", "last-seen-space"}
        )
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
            "borderSpace": border_space,
            "targetGroupCount": target_group_count,
            "enabledReports": list(report_ids),
            "enabledStrategies": list(strategy_ids),
        },
        "tables": {
            name: _table_payload(table) for name, table in tables.items()
        },
        "spaceGroups": space_group_payload,
        "nonlinearDynamics": nonlinear_dynamics_payload,
        "history": history,
        "analysisHistory": (
            [
                {
                    "date": draw.date,
                    "numbers": [ball.value for ball in draw.balls],
                }
                for draw in draws.draws
            ]
            if report_set.intersection({"autocorrelation", "co-occurrence"})
            else []
        ),
        "combinedPredictions": prediction_history,
        "predictionSuites": (
            strategy_analysis["predictionSuites"]
            if display_prediction_suites_required
            else []
        ),
        "strategyEfficacyHistory": strategy_analysis[
            "strategyEfficacyHistory"
        ],
        "predictionAuditHistory": (
            strategy_analysis["predictionAuditHistory"]
            if "prediction-audit" in report_set
            else []
        ),
        "drawComparisonHistory": (
            strategy_analysis["drawComparisonHistory"]
            if "draw-comparison" in report_set
            else []
        ),
        "latestDrawComparison": (
            strategy_analysis["drawComparisonHistory"][-1]
            if "draw-comparison" in report_set
            and strategy_analysis["drawComparisonHistory"]
            else None
        ),
        "possibleDraw": (
            _possible_draw_payload(draws)
            if report_set.intersection({"draw-portfolio", "possible-draw"})
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


def statistics_command_data(
    source_path: Path,
    command_id: str,
    *,
    border_space: int = DEFAULT_BORDER_SPACE,
) -> dict[str, Any]:
    """Calculate one whitelisted statistic against the complete trusted dataset."""
    if command_id not in STATISTICS_COMMAND_IDS:
        raise ValueError(f"Unknown statistics command: {command_id}")
    draws = load_trusted_draws(source_path, prepare_predictions=False)
    statistics = DrawsStatistics(draws)
    if command_id == "statistics.number-frequency":
        table = statistics.number_frequencies()
        validated_border = None
    elif command_id == "statistics.group-frequency":
        validated_border = validate_border_space(border_space)
        table = statistics.group_signature_frequencies(validated_border)
    else:  # pragma: no cover - guarded by the whitelist above
        raise ValueError(f"Unsupported statistics command: {command_id}")
    payload: dict[str, Any] = {
        "id": command_id,
        "datasetName": source_path.name,
        "drawCount": statistics.draw_count,
        "table": _table_payload(table),
    }
    if validated_border is not None:
        payload["borderSpace"] = validated_border
    return payload


def analyze_file(
    source_path: Path,
    *,
    selected_numbers: Sequence[int] = DEFAULT_SELECTED_NUMBERS,
    trend_bins: int = 100,
    correlation_method: CorrelationMethod = "pearson",
    border_space: int = DEFAULT_BORDER_SPACE,
    target_group_count: int | None = None,
    enabled_reports: Collection[str] = DEFAULT_REPORT_IDS,
    enabled_strategies: Collection[str] = DEFAULT_STRATEGY_IDS,
    progress: ProgressCallback | None = None,
    strategy_cache_dir: Path | None = None,
    refresh_strategy_cache: bool = False,
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
        border_space=border_space,
        target_group_count=target_group_count,
        enabled_reports=enabled_reports,
        enabled_strategies=enabled_strategies,
        progress=progress,
        strategy_cache_dir=strategy_cache_dir,
        refresh_strategy_cache=refresh_strategy_cache,
    )


def portfolio_backtest_data(
    source_path: Path,
    *,
    enabled_strategies: Collection[str] = DEFAULT_STRATEGY_IDS,
    border_space: int = DEFAULT_BORDER_SPACE,
    target_group_count: int | None = None,
    progress: ProgressCallback | None = None,
    strategy_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Return compact full-history rankings for an on-demand portfolio backtest."""
    _report_progress(progress, 4, "Opening the trusted portfolio history dataset")
    draws = load_trusted_draws(source_path, prepare_predictions=False)
    strategy_ids = tuple(
        strategy_id
        for strategy_id in STRATEGY_IDS
        if strategy_id in enabled_strategies
    )
    border_space = validate_border_space(border_space)
    target_group_count = validate_target_group_count(target_group_count)
    identity = _strategy_cache_identity(
        draws, strategy_ids, border_space, target_group_count
    )
    cached = (
        _read_strategy_cache(strategy_cache_dir, identity)
        if strategy_cache_dir is not None and strategy_ids
        else None
    )
    if cached is None:
        _report_progress(progress, 8, "Preparing full-history prediction prerequisites")
        if draws.draws and draws.draws[-1].prediction is None:
            draws.prepare_predictions(
                _prediction_progress(
                    progress,
                    8,
                    30,
                    "Calculating portfolio prediction prerequisites",
                )
            )
        cached = _strategy_analysis(
            draws,
            strategy_ids,
            max(0, len(draws) - MAX_HISTORY_WINDOW),
            progress,
            strategy_cache_dir,
            False,
            border_space,
            target_group_count,
        )
    else:
        _report_progress(progress, 60, "Loaded compact full-history strategy rankings")
    cache_name = _strategy_cache_path(Path("."), identity).name
    _report_progress(progress, 92, "Serializing full-history portfolio inputs")
    return {
        "cacheKey": cache_name.removesuffix(".json.gz"),
        "strategyIds": list(strategy_ids),
        "draws": [
            {
                "date": draw.date,
                "numbers": [ball.value for ball in draw.balls],
            }
            for draw in draws.draws
        ],
        "records": cached["portfolioBacktestHistory"],
    }


def write_export_archive(
    source_path: Path,
    output_path: Path,
    *,
    selected_numbers: Sequence[int] = DEFAULT_SELECTED_NUMBERS,
    trend_bins: int = 100,
    correlation_method: CorrelationMethod = "pearson",
    border_space: int = DEFAULT_BORDER_SPACE,
    target_group_count: int | None = None,
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
    border_space = validate_border_space(border_space)
    target_group_count = validate_target_group_count(target_group_count)
    if "space-groups" in report_ids:
        group_tables, _payload = statistics.space_group_analysis(
            border_space, target_group_count
        )
        tables.update(group_tables)
    if "nonlinear-dynamics" in report_ids:
        _payload, nonlinear_tables = nonlinear_dynamics_analysis(
            [tuple(ball.value for ball in draw.balls) for draw in draws.draws]
        )
        tables.update(nonlinear_tables)
    metadata = {
        "source": str(source_path.resolve()),
        "drawCount": statistics.draw_count,
        "selectedNumbers": list(selected_numbers),
        "trendBins": min(trend_bins, statistics.draw_count),
        "correlationMethod": correlation_method,
        "borderSpace": border_space,
        "targetGroupCount": target_group_count,
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
            "--border-space",
            default=DEFAULT_BORDER_SPACE,
            type=int,
            choices=range(44),
        )
        command_parser.add_argument(
            "--target-group-count",
            type=int,
            choices=range(1, 7),
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
        if command == "analyze":
            command_parser.add_argument("--cache-dir", type=Path)
            command_parser.add_argument(
                "--refresh-cache",
                action="store_true",
            )
        else:
            command_parser.add_argument("--output", required=True, type=Path)
    editor_parser = subparsers.add_parser("draw-editor")
    editor_parser.add_argument("--input", required=True, type=Path)
    save_parser = subparsers.add_parser("draw-save")
    save_parser.add_argument("--input", required=True, type=Path)
    save_parser.add_argument("--date", required=True)
    save_parser.add_argument("--numbers", required=True, type=parse_selected_numbers)
    save_parser.add_argument("--original-date")
    yaml_import_parser = subparsers.add_parser("yaml-import")
    yaml_import_parser.add_argument("--input", required=True, type=Path)
    yaml_import_parser.add_argument("--output", required=True, type=Path)
    portfolio_parser = subparsers.add_parser("portfolio-backtest-data")
    portfolio_parser.add_argument("--input", required=True, type=Path)
    portfolio_parser.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGY_IDS),
        type=parse_strategy_ids,
    )
    portfolio_parser.add_argument("--cache-dir", required=True, type=Path)
    portfolio_parser.add_argument(
        "--border-space",
        default=DEFAULT_BORDER_SPACE,
        type=int,
        choices=range(44),
    )
    portfolio_parser.add_argument(
        "--target-group-count",
        type=int,
        choices=range(1, 7),
    )
    statistics_parser = subparsers.add_parser("statistics-command")
    statistics_parser.add_argument("--input", required=True, type=Path)
    statistics_parser.add_argument(
        "--command-id",
        required=True,
        choices=STATISTICS_COMMAND_IDS,
    )
    statistics_parser.add_argument(
        "--border-space",
        default=DEFAULT_BORDER_SPACE,
        type=int,
        choices=range(44),
    )
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
    if options.command == "yaml-import":
        _write_progress(5, "Reading and validating the YAML draw history")
        draws = create_lotto_results_pickle(options.input, options.output)
        _write_progress(100, "Managed pickle generated from YAML")
        json.dump(
            {
                "picklePath": str(options.output.resolve()),
                "drawCount": len(draws),
            },
            sys.stdout,
            separators=(",", ":"),
        )
        return
    if options.command == "portfolio-backtest-data":
        payload = portfolio_backtest_data(
            options.input,
            enabled_strategies=cast(tuple[str, ...], options.strategies),
            progress=_write_progress,
            strategy_cache_dir=options.cache_dir,
            border_space=options.border_space,
            target_group_count=options.target_group_count,
        )
        _write_progress(97, "Transferring full-history portfolio inputs")
        json.dump(payload, sys.stdout, separators=(",", ":"))
        return
    if options.command == "statistics-command":
        json.dump(
            statistics_command_data(
                options.input,
                options.command_id,
                border_space=options.border_space,
            ),
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
            border_space=options.border_space,
            target_group_count=options.target_group_count,
            enabled_reports=enabled_reports,
            enabled_strategies=enabled_strategies,
            progress=_write_progress,
            strategy_cache_dir=options.cache_dir,
            refresh_strategy_cache=options.refresh_cache,
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
        border_space=options.border_space,
        target_group_count=options.target_group_count,
        enabled_reports=enabled_reports,
        enabled_strategies=enabled_strategies,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
