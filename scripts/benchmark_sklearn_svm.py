"""Compile, select, and verify all 216 Scikit Online SVM variants."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rand_ai.sklearn_svm_experiment import (
    CompiledSvmDataset,
    SVM_EXPERIMENT_SCHEMA_VERSION,
    SvmExperimentConfig,
    SvmReplayResult,
    acceptance_report,
    compile_svm_dataset,
    compiled_dataset_matches_source,
    evaluate_svm_configurations,
    load_compiled_svm_dataset,
    replay_svm_configuration,
    result_scopes,
    save_compiled_svm_dataset,
    svm_experiment_configurations,
    validation_leaderboards,
    validation_variant_record,
)

_HOLDOUT_DRAWS = 250


def _progress(label: str) -> Callable[[int, int], None]:
    def report(completed: int, total: int) -> None:
        if completed == 0 or completed == total or completed % 10 == 0:
            print(f"{label}: {completed}/{total}", file=sys.stderr, flush=True)

    return report


def _ensure_compiled(
    dataset_path: Path,
    compiled_path: Path,
    *,
    force: bool,
) -> None:
    if compiled_path.exists() and not force:
        compiled = load_compiled_svm_dataset(compiled_path)
        if compiled_dataset_matches_source(compiled, dataset_path):
            return
    compiled = compile_svm_dataset(
        dataset_path,
        progress=_progress(f"compile {dataset_path.name}"),
    )
    save_compiled_svm_dataset(compiled, compiled_path)


def _selection(
    compiled_path: Path,
    checkpoint_path: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    dataset = load_compiled_svm_dataset(compiled_path)
    if dataset.evaluated_draws <= _HOLDOUT_DRAWS:
        raise ValueError("Primary dataset is too short for the frozen holdout")
    holdout_start = dataset.evaluated_draws - _HOLDOUT_DRAWS
    configurations = svm_experiment_configurations()
    results = evaluate_svm_configurations(
        compiled_path,
        configurations,
        stop=holdout_start,
        checkpoint_path=checkpoint_path,
        workers=workers,
        progress=_progress("primary validation variants"),
    )
    records = [validation_variant_record(result, dataset) for result in results]
    leaderboards = validation_leaderboards(records)
    winner_key = leaderboards["stableMultiWindow"][0]
    winner_record = next(
        record for record in records if record["configKey"] == winner_key
    )
    return {
        "schemaVersion": SVM_EXPERIMENT_SCHEMA_VERSION,
        "dataset": dataset.source_path,
        "datasetFingerprint": dataset.fingerprint,
        "evaluatedDraws": dataset.evaluated_draws,
        "holdout": {
            "draws": _HOLDOUT_DRAWS,
            "start": holdout_start,
            "opened": False,
        },
        "configurationCount": len(configurations),
        "leaderboards": leaderboards,
        "winner": winner_record,
        "variants": records,
    }


def _load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schemaVersion") != SVM_EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("Locked selection schema is stale")
    if value.get("holdout", {}).get("opened") is not False:
        raise ValueError("Selection lock was not created before opening the holdout")
    return value


def _config_from_lock(lock: dict[str, Any]) -> SvmExperimentConfig:
    winner = lock.get("winner")
    if not isinstance(winner, dict) or not isinstance(winner.get("config"), dict):
        raise ValueError("Selection lock has no winning configuration")
    return SvmExperimentConfig.from_dict(winner["config"])


def _variant_scope_record(
    result: SvmReplayResult,
    dataset: CompiledSvmDataset,
) -> dict[str, Any]:
    return {
        "configKey": result.config.key,
        "config": asdict(result.config),
        "featureNames": list(result.feature_names),
        "featureCount": len(result.feature_names),
        "durationSeconds": result.duration_seconds,
        "trainedDraws": result.trained_draws,
        "rankingDigest": result.ranking_digest,
        "correctnessPassed": result.correctness_passed,
        "scopes": result_scopes(result, dataset),
    }


def _final_report(
    lock: dict[str, Any],
    primary_compiled_path: Path,
    primary_checkpoint_path: Path,
    secondary_compiled_path: Path | None,
    secondary_checkpoint_path: Path | None,
    *,
    workers: int,
) -> dict[str, Any]:
    primary = load_compiled_svm_dataset(primary_compiled_path)
    if lock["datasetFingerprint"] != primary.fingerprint:
        raise ValueError("Locked selection does not match the primary dataset")
    if lock["holdout"]["start"] != primary.evaluated_draws - _HOLDOUT_DRAWS:
        raise ValueError("Locked holdout boundary does not match the primary dataset")
    winner = _config_from_lock(lock)
    primary_results = evaluate_svm_configurations(
        primary_compiled_path,
        (winner,),
        stop=primary.evaluated_draws,
        checkpoint_path=primary_checkpoint_path,
        workers=1,
        progress=_progress("locked primary winner"),
    )
    primary_winner = primary_results[0]
    repeated = replay_svm_configuration(primary, winner)
    deterministic = (
        primary_winner.ranking_digest == repeated.ranking_digest
        and primary_winner.hits == repeated.hits
    )
    acceptance_inputs = [(primary_winner, primary)]
    secondary_report: dict[str, Any] | None = None

    if secondary_compiled_path is not None and secondary_checkpoint_path is not None:
        secondary = load_compiled_svm_dataset(secondary_compiled_path)
        secondary_results = evaluate_svm_configurations(
            secondary_compiled_path,
            svm_experiment_configurations(),
            stop=secondary.evaluated_draws,
            checkpoint_path=secondary_checkpoint_path,
            workers=workers,
            progress=_progress("secondary variants"),
        )
        secondary_records = [
            _variant_scope_record(result, secondary) for result in secondary_results
        ]
        secondary_winner = next(
            result for result in secondary_results if result.config.key == winner.key
        )
        secondary_repeated = replay_svm_configuration(secondary, winner)
        deterministic = deterministic and (
            secondary_winner.ranking_digest == secondary_repeated.ranking_digest
            and secondary_winner.hits == secondary_repeated.hits
        )
        acceptance_inputs.append((secondary_winner, secondary))
        secondary_report = {
            "dataset": secondary.source_path,
            "datasetFingerprint": secondary.fingerprint,
            "evaluatedDraws": secondary.evaluated_draws,
            "configurationCount": len(secondary_records),
            "variants": secondary_records,
        }

    acceptance = acceptance_report(acceptance_inputs, deterministic=deterministic)
    return {
        "schemaVersion": SVM_EXPERIMENT_SCHEMA_VERSION,
        "configurationCount": len(svm_experiment_configurations()),
        "selection": lock,
        "lockedWinner": _variant_scope_record(primary_winner, primary),
        "secondary": secondary_report,
        "deterministicReplay": deterministic,
        "acceptance": acceptance,
        "productionAction": (
            "promote_locked_winner"
            if acceptance["promotionAuthorized"]
            else "retain_current_opt_in_strategy"
        ),
    }


def _write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "phase",
        "configKey",
        "historyWindow",
        "averageWeights",
        "objective",
        "centerInputs",
        "inputProfile",
        "featureCount",
        "scope",
        "evaluatedDraws",
        "totalHits",
        "averageHitsPerDraw",
        "customSvcAverageHitsPerDraw",
        "customSvcDifference",
        "aboveRandom",
        "competitive",
    )
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()

        def write_scopes(
            dataset: str,
            phase: str,
            record: dict[str, Any],
            scopes: dict[str, dict[str, Any]],
        ) -> None:
            config = record["config"]
            for scope_name, scope in scopes.items():
                writer.writerow(
                    {
                        "dataset": dataset,
                        "phase": phase,
                        "configKey": record["configKey"],
                        "historyWindow": config["history_window"],
                        "averageWeights": config["average_weights"],
                        "objective": config["objective"],
                        "centerInputs": config["center_inputs"],
                        "inputProfile": config["input_profile"],
                        "featureCount": record["featureCount"],
                        "scope": scope_name,
                        "evaluatedDraws": scope["evaluatedDraws"],
                        "totalHits": scope["totalHits"],
                        "averageHitsPerDraw": scope["averageHitsPerDraw"],
                        "customSvcAverageHitsPerDraw": scope[
                            "customSvcAverageHitsPerDraw"
                        ],
                        "customSvcDifference": scope["customSvcDifference"],
                        "aboveRandom": scope["aboveRandom"],
                        "competitive": scope["competitive"],
                    }
                )

        selection = report.get("selection", report)
        for record in selection.get("variants", []):
            fold_scopes = {
                f"validation{index + 1}": scope
                for index, scope in enumerate(record["folds"])
            }
            fold_scopes["wholePreHoldout"] = record["wholePreHoldout"]
            write_scopes(selection["dataset"], "selection", record, fold_scopes)
        secondary = report.get("secondary")
        if secondary:
            for record in secondary["variants"]:
                write_scopes(
                    secondary["dataset"], "secondary", record, record["scopes"]
                )


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/lotto_results.yaml"))
    parser.add_argument("--secondary-dataset", type=Path)
    parser.add_argument(
        "--phase", choices=("compile", "select", "final", "all"), default="all"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("outputs/sklearn_svm_experiment")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--locked-config", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force-recompile", action="store_true")
    options = parser.parse_args()
    if options.workers < 1:
        parser.error("--workers must be positive")

    work_dir: Path = options.work_dir
    primary_compiled = work_dir / f"{options.dataset.stem}.npz"
    primary_validation_checkpoint = work_dir / "primary-validation.jsonl"
    primary_final_checkpoint = work_dir / "primary-final.jsonl"
    lock_path = options.locked_config or work_dir / "locked-selection.json"
    output_path = options.output or work_dir / "report.json"
    csv_path = options.csv_output or work_dir / "report.csv"

    _ensure_compiled(
        options.dataset,
        primary_compiled,
        force=options.force_recompile,
    )
    secondary_compiled: Path | None = None
    secondary_checkpoint: Path | None = None
    if options.secondary_dataset is not None:
        secondary_compiled = work_dir / f"{options.secondary_dataset.stem}.npz"
        secondary_checkpoint = work_dir / "secondary-variants.jsonl"
        _ensure_compiled(
            options.secondary_dataset,
            secondary_compiled,
            force=options.force_recompile,
        )
    if options.phase == "compile":
        print(primary_compiled)
        if secondary_compiled is not None:
            print(secondary_compiled)
        return

    if options.phase in {"select", "all"}:
        lock = _selection(
            primary_compiled,
            primary_validation_checkpoint,
            workers=options.workers,
        )
        _write_json(lock, lock_path)
    else:
        if options.locked_config is None:
            parser.error("--phase final requires --locked-config")
        lock = _load_lock(options.locked_config)

    if options.phase == "select":
        _write_json(lock, output_path)
        _write_csv(lock, csv_path)
        print(json.dumps({"lock": str(lock_path), "winner": lock["winner"]}, indent=2))
        return

    report = _final_report(
        lock,
        primary_compiled,
        primary_final_checkpoint,
        secondary_compiled,
        secondary_checkpoint,
        workers=options.workers,
    )
    _write_json(report, output_path)
    _write_csv(report, csv_path)
    print(
        json.dumps(
            {
                "winner": report["lockedWinner"]["configKey"],
                "acceptance": report["acceptance"],
                "productionAction": report["productionAction"],
                "report": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
