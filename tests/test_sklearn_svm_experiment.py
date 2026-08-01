"""Test the exhaustive, leakage-safe Scikit Online SVM experiment engine."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from rand_ai import Draw, Draws
from rand_ai.sklearn_svm_experiment import (
    SVM_INPUT_PROFILES,
    CompiledSvmDataset,
    SvmExperimentConfig,
    SvmReplayResult,
    _initialize_worker,
    _new_estimator,
    _recent_sort_key,
    _stable_sort_key,
    _training_batch,
    _whole_sort_key,
    _worker_replay,
    acceptance_report,
    compile_svm_dataset,
    compiled_dataset_matches_source,
    evaluate_svm_configurations,
    feature_names_for_profile,
    features_for_config,
    load_replay_checkpoint,
    load_compiled_svm_dataset,
    replay_svm_configuration,
    save_compiled_svm_dataset,
    svm_experiment_configurations,
    validation_fold_ranges,
    validation_leaderboards,
    validation_variant_record,
    result_scopes,
)


def _compiled_dataset(draw_count: int = 30) -> CompiledSvmDataset:
    generator = np.random.default_rng(20260626)
    base = generator.uniform(0.0, 1.0, size=(draw_count, 49, 32))
    base[:, :, 8:15] = generator.uniform(-1.0, 1.0, size=(draw_count, 49, 7))
    compact = np.concatenate(
        (
            base[:, :, :15],
            base[:, :, (15, 17, 19, 21, 23, 25)],
            base[:, :, 31:32],
            generator.uniform(0.0, 1.0, size=(draw_count, 49, 4)),
            generator.uniform(-1.0, 1.0, size=(draw_count, 49, 1)),
        ),
        axis=2,
    )
    labels = np.zeros((draw_count, 49), dtype=np.uint8)
    svc_rankings = np.empty((draw_count, 49), dtype=np.uint8)
    for draw_index in range(draw_count):
        labels[draw_index, [(draw_index + offset * 7) % 49 for offset in range(6)]] = 1
        svc_rankings[draw_index] = np.roll(
            np.arange(1, 50, dtype=np.uint8), draw_index
        )
    expert_rankings = np.tile(
        np.arange(1, 50, dtype=np.uint8), (draw_count, 6, 1)
    )
    return CompiledSvmDataset(
        source_path="synthetic.yaml",
        fingerprint="synthetic",
        reference_draw_numbers=np.arange(1, draw_count + 1, dtype=np.int32),
        base_features=base,
        compact_features=compact,
        labels=labels,
        gaps=np.tile(np.arange(49, dtype=np.uint16), (draw_count, 1)),
        expert_weights=np.ones((draw_count, 6), dtype=float),
        expert_rankings=expert_rankings,
        svc_rankings=svc_rankings,
    )


def _config(**changes: object) -> SvmExperimentConfig:
    original = SvmExperimentConfig(
        history_window="unlimited",
        average_weights=True,
        objective="classification",
        center_inputs=False,
        input_profile="full_hybrid",
    )
    return replace(original, **changes)


def test_generates_all_216_unique_configuration_variants() -> None:
    configurations = svm_experiment_configurations()

    assert len(configurations) == 216
    assert len({config.key for config in configurations}) == 216
    assert {config.input_profile for config in configurations} == set(
        SVM_INPUT_PROFILES
    )
    assert {config.objective for config in configurations} == {
        "classification",
        "hard_pairwise",
        "all_pairwise",
    }
    assert {config.history_window for config in configurations} == {
        "unlimited",
        "500",
        "250",
    }


@pytest.mark.parametrize(
    ("profile", "count"),
    (
        ("full_hybrid", 32),
        ("historical_only", 15),
        ("expert_only", 17),
        ("no_static", 28),
        ("no_efficacy_interactions", 26),
        ("compact_hybrid", 27),
    ),
)
def test_selects_exact_input_profiles(profile: str, count: int) -> None:
    dataset = _compiled_dataset(2)
    config = _config(input_profile=profile)

    selected = features_for_config(dataset, config)

    assert selected.shape == (2, 49, count)
    assert len(feature_names_for_profile(config.input_profile)) == count


def test_centers_unsigned_inputs_but_preserves_signed_residuals() -> None:
    dataset = _compiled_dataset(1)
    dataset.base_features[0, 0, 0] = 0.25
    dataset.base_features[0, 0, 8] = -0.25

    centered = features_for_config(dataset, _config(center_inputs=True))

    assert centered[0, 0, 0] == pytest.approx(-0.5)
    assert centered[0, 0, 8] == pytest.approx(-0.25)
    assert np.all(centered >= -1)
    assert np.all(centered <= 1)


def test_builds_classification_and_pairwise_training_batches() -> None:
    dataset = _compiled_dataset(1)
    features = dataset.base_features[0]
    labels = dataset.labels[0]
    scores = np.arange(49, dtype=float)
    gaps = dataset.gaps[0]

    classification = _training_batch(
        _config(), features, labels, scores, gaps
    )
    hard = _training_batch(
        _config(objective="hard_pairwise"), features, labels, scores, gaps
    )
    all_negative = _training_batch(
        _config(objective="all_pairwise"), features, labels, scores, gaps
    )

    assert classification[0].shape == (49, 32)
    assert classification[2][labels == 1] == pytest.approx([43 / 6] * 6)
    assert classification[2][labels == 0] == pytest.approx([1.0] * 43)
    assert hard[0].shape == (6 * 12 * 2, 32)
    assert all_negative[0].shape == (6 * 43 * 2, 32)
    assert set(hard[1]) == {0, 1}
    assert sum(hard[2]) == pytest.approx(1.0)
    assert sum(all_negative[2]) == pytest.approx(1.0)
    assert _new_estimator(_config()).fit_intercept
    assert not _new_estimator(_config(objective="hard_pairwise")).fit_intercept


@pytest.mark.parametrize(
    "objective", ("classification", "hard_pairwise", "all_pairwise")
)
def test_replay_is_deterministic_and_produces_valid_rankings(objective: str) -> None:
    dataset = _compiled_dataset()
    config = _config(objective=objective, average_weights=False)

    first = replay_svm_configuration(dataset, config)
    repeated = replay_svm_configuration(dataset, config)

    assert first.hits == repeated.hits
    assert first.ranking_digest == repeated.ranking_digest
    assert first.correctness_passed
    assert first.trained_draws == dataset.evaluated_draws


def test_replay_stop_does_not_read_future_labels() -> None:
    dataset = _compiled_dataset(20)
    changed_labels = dataset.labels.copy()
    changed_labels[10:] = np.flip(changed_labels[10:], axis=1)
    changed = replace(dataset, labels=changed_labels)
    config = _config(average_weights=False)

    original_result = replay_svm_configuration(dataset, config, stop=10)
    changed_result = replay_svm_configuration(changed, config, stop=10)

    assert original_result.hits == changed_result.hits
    assert original_result.ranking_digest == changed_result.ranking_digest


def test_checkpoint_resume_does_not_repeat_completed_variants(tmp_path: Path) -> None:
    dataset = _compiled_dataset(5)
    compiled_path = tmp_path / "compiled.npz"
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    save_compiled_svm_dataset(dataset, compiled_path)
    configs = (_config(), _config(average_weights=False))

    first = evaluate_svm_configurations(
        compiled_path,
        configs,
        stop=5,
        checkpoint_path=checkpoint_path,
    )
    line_count = len(checkpoint_path.read_text(encoding="utf-8").splitlines())
    repeated = evaluate_svm_configurations(
        compiled_path,
        configs,
        stop=5,
        checkpoint_path=checkpoint_path,
    )

    assert [result.ranking_digest for result in first] == [
        result.ranking_digest for result in repeated
    ]
    assert len(checkpoint_path.read_text(encoding="utf-8").splitlines()) == line_count
    assert len(
        load_replay_checkpoint(
            checkpoint_path,
            dataset_fingerprint=dataset.fingerprint,
            stop=5,
        )
    ) == 2


def test_checkpoint_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="Invalid checkpoint configuration"):
        SvmReplayResult.from_checkpoint({"config": "invalid"})


def test_compilation_labels_only_the_subsequent_target_draw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = Draws()
    values = (
        (1, 2, 8, 17, 31, 49),
        (3, 6, 12, 22, 36, 47),
        (1, 9, 18, 27, 38, 45),
        (4, 11, 20, 29, 37, 46),
    )
    for numbers in values:
        draws.add(Draw(*numbers))
    source = tmp_path / "draws.yaml"
    source.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        "rand_ai.sklearn_svm_experiment.load_lotto_results_yaml",
        lambda _path: draws,
    )
    progress: list[tuple[int, int]] = []

    compiled = compile_svm_dataset(source, progress=lambda done, total: progress.append((done, total)))

    assert compiled.evaluated_draws == 3
    assert set(np.flatnonzero(compiled.labels[0]) + 1) == set(values[1])
    assert set(np.flatnonzero(compiled.labels[1]) + 1) == set(values[2])
    assert compiled.base_features.shape == (3, 49, 32)
    assert compiled.compact_features.shape == (3, 49, 27)
    assert np.all(compiled.expert_weights >= 0.5)
    assert np.all(compiled.expert_weights <= 1.5)
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_compilation_rejects_unprepared_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.add(Draw(3, 6, 12, 22, 36, 47))
    source = tmp_path / "draws.yaml"
    source.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(
        "rand_ai.sklearn_svm_experiment.load_lotto_results_yaml",
        lambda _path: draws,
    )
    monkeypatch.setattr(Draws, "prepare_predictions", lambda _self: None)

    with pytest.raises(ValueError, match="Combined predictions"):
        compile_svm_dataset(source)


def test_archive_schema_and_source_fingerprint_validation(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text("original", encoding="utf-8")
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    dataset = replace(
        _compiled_dataset(2),
        source_path=str(source.resolve()),
        fingerprint=fingerprint,
    )
    archive = tmp_path / "compiled.npz"
    save_compiled_svm_dataset(dataset, archive)

    loaded = load_compiled_svm_dataset(archive)

    assert compiled_dataset_matches_source(loaded, source)
    source.write_text("changed", encoding="utf-8")
    assert not compiled_dataset_matches_source(loaded, source)
    stale = tmp_path / "stale.npz"
    np.savez_compressed(
        stale,
        metadata=np.asarray(
            json.dumps(
                {
                    "schemaVersion": 0,
                    "sourcePath": str(source),
                    "fingerprint": "stale",
                }
            )
        ),
    )
    with pytest.raises(ValueError, match="schema is stale"):
        load_compiled_svm_dataset(stale)


def test_validation_ranges_and_all_three_leaderboards() -> None:
    assert validation_fold_ranges(1000) == ((0, 250), (250, 500), (500, 750))
    dataset = _compiled_dataset(1000)
    first = replay_svm_configuration(dataset, _config(average_weights=False), stop=750)
    second = replay_svm_configuration(
        dataset,
        _config(average_weights=False, input_profile="historical_only"),
        stop=750,
    )
    records = [
        validation_variant_record(first, dataset),
        validation_variant_record(second, dataset),
    ]

    leaderboards = validation_leaderboards(records)

    assert set(leaderboards) == {
        "stableMultiWindow",
        "wholeHistory",
        "recentHistory",
    }
    assert all(len(ranking) == 2 for ranking in leaderboards.values())


def test_rolling_replay_rebuilds_and_validates_stop_boundaries() -> None:
    dataset = _compiled_dataset(275)
    rolling = _config(
        history_window="250",
        average_weights=False,
        refit_interval=25,
    )

    result = replay_svm_configuration(dataset, rolling)

    assert result.trained_draws == 275
    with pytest.raises(ValueError, match="outside the compiled dataset"):
        replay_svm_configuration(dataset, rolling, stop=276)


def test_validation_and_scope_defensive_boundaries() -> None:
    with pytest.raises(ValueError, match="too short"):
        validation_fold_ranges(999)
    dataset = _compiled_dataset(1000)
    short_result = replay_svm_configuration(dataset, _config(), stop=749)
    with pytest.raises(ValueError, match="stop exactly"):
        validation_variant_record(short_result, dataset)
    with pytest.raises(ValueError, match="full replay"):
        result_scopes(short_result, dataset)
    with pytest.raises(ValueError, match="Invalid validation configuration"):
        _stable_sort_key({"config": "invalid"})
    with pytest.raises(ValueError, match="Invalid whole-history metrics"):
        _whole_sort_key({"wholePreHoldout": "invalid"})
    with pytest.raises(ValueError, match="Invalid recent-history metrics"):
        _recent_sort_key({"recentValidation": "invalid"})


def test_scope_and_all_acceptance_interpretations() -> None:
    dataset = _compiled_dataset(500)
    result = replay_svm_configuration(dataset, _config(average_weights=False))

    scopes = result_scopes(result, dataset)
    acceptance = acceptance_report(((result, dataset),), deterministic=True)

    assert set(scopes) == {"wholeHistory", "latest500", "latest250"}
    assert set(acceptance) == {
        "datasets",
        "stableCompetitive",
        "majorityWinner",
        "correctnessOnly",
        "promotionAuthorized",
    }
    assert acceptance["correctnessOnly"]["passed"]


def test_worker_initialization_replay_and_parallel_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = _compiled_dataset(5)
    compiled_path = tmp_path / "compiled.npz"
    save_compiled_svm_dataset(dataset, compiled_path)
    config = _config(average_weights=False)
    import rand_ai.sklearn_svm_experiment as experiment

    monkeypatch.setattr(experiment, "_WORKER_DATASET", None)
    with pytest.raises(RuntimeError, match="was not initialized"):
        _worker_replay({}, 5)
    _initialize_worker(str(compiled_path))
    record = _worker_replay(
        {
            "history_window": config.history_window,
            "average_weights": config.average_weights,
            "objective": config.objective,
            "center_inputs": config.center_inputs,
            "input_profile": config.input_profile,
            "refit_interval": config.refit_interval,
        },
        5,
    )
    assert record["configKey"] == config.key

    checkpoint = tmp_path / "parallel.jsonl"
    progress: list[tuple[int, int]] = []
    configurations = (config, _config(input_profile="historical_only"))
    results = evaluate_svm_configurations(
        compiled_path,
        configurations,
        stop=5,
        checkpoint_path=checkpoint,
        workers=2,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert len(results) == 2
    assert progress[0] == (0, 2)
    assert progress[-1] == (2, 2)
    checkpoint.write_text(
        checkpoint.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert len(
        load_replay_checkpoint(
            checkpoint,
            dataset_fingerprint=dataset.fingerprint,
            stop=5,
        )
    ) == 2
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_svm_configurations(
            compiled_path,
            configurations,
            stop=5,
            checkpoint_path=checkpoint,
            workers=0,
        )
