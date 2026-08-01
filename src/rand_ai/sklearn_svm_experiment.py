"""Leakage-safe exhaustive experiments for the Scikit Online SVM strategy."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from sklearn.linear_model import SGDClassifier

from rand_ai.lotto_results import load_lotto_results_yaml
from rand_ai.strategy_prediction import (
    _NUMBER_COUNT,
    _NUMBERS_PER_DRAW,
    _RANDOM_SEED,
    _SKLEARN_SVM_EXPERT_IDS,
    _SKLEARN_SVM_FEATURE_NAMES,
    _StrategyState,
)

SVM_EXPERIMENT_SCHEMA_VERSION = 1
SVM_REFIT_INTERVAL = 25
SVM_RANDOM_EXPECTATION = 36 / 49
SVM_MAXIMUM_REGRESSION = 0.02

SvmObjective = Literal["classification", "hard_pairwise", "all_pairwise"]
SvmHistoryWindow = Literal["unlimited", "500", "250"]
SvmInputProfile = Literal[
    "full_hybrid",
    "historical_only",
    "expert_only",
    "no_static",
    "no_efficacy_interactions",
    "compact_hybrid",
]

SVM_OBJECTIVES: tuple[SvmObjective, ...] = (
    "classification",
    "hard_pairwise",
    "all_pairwise",
)
SVM_HISTORY_WINDOWS: tuple[SvmHistoryWindow, ...] = ("unlimited", "500", "250")
SVM_INPUT_PROFILES: tuple[SvmInputProfile, ...] = (
    "full_hybrid",
    "historical_only",
    "expert_only",
    "no_static",
    "no_efficacy_interactions",
    "compact_hybrid",
)

_HISTORICAL_FEATURE_COUNT = 15
_RAW_EXPERT_INDICES = (15, 17, 19, 21, 23, 25)
_EFFICACY_INTERACTION_INDICES = frozenset((16, 18, 20, 22, 24, 26))
_STATIC_FEATURE_INDICES = frozenset((0, 1, 2, 3))
_SIGNED_BASE_FEATURES = frozenset(
    {
        "overdue_ratio",
        "lifetime_frequency_residual",
        "recent_frequency_residual_5",
        "recent_frequency_residual_20",
        "recent_frequency_residual_100",
        "recent_5_vs_20_trend",
        "latest_draw_compatibility",
    }
)
_COMPACT_EXPERT_FEATURE_NAMES = (
    *(f"{strategy_id}_strength" for strategy_id in _SKLEARN_SVM_EXPERT_IDS),
    "expert_rank_variance",
    "efficacy_weighted_top_six_support",
    "efficacy_weighted_top_quarter_support",
    "long_horizon_consensus",
    "recent_pattern_consensus",
    "recent_minus_long_consensus",
)
_COMPACT_FEATURE_NAMES = (
    *_SKLEARN_SVM_FEATURE_NAMES[:_HISTORICAL_FEATURE_COUNT],
    *_COMPACT_EXPERT_FEATURE_NAMES,
)
_SIGNED_COMPACT_FEATURES = frozenset(
    {*_SIGNED_BASE_FEATURES, "recent_minus_long_consensus"}
)


@dataclass(frozen=True, slots=True)
class SvmExperimentConfig:
    """Identify one deterministic member of the exhaustive model matrix."""

    history_window: SvmHistoryWindow
    average_weights: bool
    objective: SvmObjective
    center_inputs: bool
    input_profile: SvmInputProfile
    refit_interval: int = SVM_REFIT_INTERVAL

    @property
    def key(self) -> str:
        """Return a stable, sortable serialized configuration key."""
        return (
            f"history={self.history_window}__average={int(self.average_weights)}"
            f"__objective={self.objective}__center={int(self.center_inputs)}"
            f"__profile={self.input_profile}__refit={self.refit_interval}"
        )

    @property
    def window_draws(self) -> int | None:
        """Return the integer rolling window or None for unlimited history."""
        return None if self.history_window == "unlimited" else int(self.history_window)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SvmExperimentConfig:
        """Recreate a configuration from its JSON-compatible representation."""
        return cls(
            history_window=cast(SvmHistoryWindow, str(value["history_window"])),
            average_weights=bool(value["average_weights"]),
            objective=cast(SvmObjective, str(value["objective"])),
            center_inputs=bool(value["center_inputs"]),
            input_profile=cast(SvmInputProfile, str(value["input_profile"])),
            refit_interval=int(value.get("refit_interval", SVM_REFIT_INTERVAL)),
        )


@dataclass(frozen=True, slots=True)
class CompiledSvmDataset:
    """Store immutable pre-result features and subsequent supervised labels."""

    source_path: str
    fingerprint: str
    reference_draw_numbers: np.ndarray
    base_features: np.ndarray
    compact_features: np.ndarray
    labels: np.ndarray
    gaps: np.ndarray
    expert_weights: np.ndarray
    expert_rankings: np.ndarray
    svc_rankings: np.ndarray

    @property
    def evaluated_draws(self) -> int:
        """Return the number of feature batches with known target results."""
        return int(self.labels.shape[0])

    @property
    def svc_hits(self) -> np.ndarray:
        """Return custom-SVC Top-6 hits for every evaluated target draw."""
        hits = np.zeros(self.evaluated_draws, dtype=np.uint8)
        for draw_index in range(self.evaluated_draws):
            drawn = self.labels[draw_index].astype(bool)
            top_indices = self.svc_rankings[draw_index, :_NUMBERS_PER_DRAW] - 1
            hits[draw_index] = int(np.count_nonzero(drawn[top_indices]))
        return hits


@dataclass(frozen=True, slots=True)
class SvmReplayResult:
    """Store a complete chronological replay result for one configuration."""

    config: SvmExperimentConfig
    feature_names: tuple[str, ...]
    hits: tuple[int, ...]
    ranking_digest: str
    trained_draws: int
    finite_scores: bool
    unique_rankings: bool
    valid_top_six: bool
    duration_seconds: float

    @property
    def correctness_passed(self) -> bool:
        """Return whether every produced ranking satisfies structural checks."""
        return self.finite_scores and self.unique_rankings and self.valid_top_six

    def checkpoint_record(self, dataset_fingerprint: str, stop: int) -> dict[str, Any]:
        """Return a JSON-compatible checkpoint entry."""
        return {
            "datasetFingerprint": dataset_fingerprint,
            "stop": stop,
            "config": asdict(self.config),
            "configKey": self.config.key,
            "featureNames": list(self.feature_names),
            "hits": list(self.hits),
            "rankingDigest": self.ranking_digest,
            "trainedDraws": self.trained_draws,
            "finiteScores": self.finite_scores,
            "uniqueRankings": self.unique_rankings,
            "validTopSix": self.valid_top_six,
            "durationSeconds": self.duration_seconds,
        }

    @classmethod
    def from_checkpoint(cls, value: dict[str, Any]) -> SvmReplayResult:
        """Load one replay result from a validated checkpoint entry."""
        config_value = value["config"]
        if not isinstance(config_value, dict):
            raise ValueError("Invalid checkpoint configuration")
        return cls(
            config=SvmExperimentConfig.from_dict(config_value),
            feature_names=tuple(str(item) for item in value["featureNames"]),
            hits=tuple(int(item) for item in value["hits"]),
            ranking_digest=str(value["rankingDigest"]),
            trained_draws=int(value["trainedDraws"]),
            finite_scores=bool(value["finiteScores"]),
            unique_rankings=bool(value["uniqueRankings"]),
            valid_top_six=bool(value["validTopSix"]),
            duration_seconds=float(value["durationSeconds"]),
        )


ProgressCallback = Callable[[int, int], None]


def svm_experiment_configurations() -> tuple[SvmExperimentConfig, ...]:
    """Return all 216 configurations in a stable order."""
    return tuple(
        SvmExperimentConfig(history, average, objective, centered, profile)
        for history, average, objective, centered, profile in product(
            SVM_HISTORY_WINDOWS,
            (True, False),
            SVM_OBJECTIVES,
            (False, True),
            SVM_INPUT_PROFILES,
        )
    )


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compiled_dataset_matches_source(
    dataset: CompiledSvmDataset,
    source_path: Path,
) -> bool:
    """Return whether a compiled archive still represents the source file."""
    resolved = source_path.resolve()
    return (
        dataset.source_path == str(resolved)
        and dataset.fingerprint == _file_fingerprint(resolved)
    )


def _compact_feature_row(
    base_row: np.ndarray,
    number: int,
    rankings: Sequence[Sequence[int]],
    weights: np.ndarray,
) -> np.ndarray:
    strengths = np.asarray([base_row[index] for index in _RAW_EXPERT_INDICES])
    weight_total = float(np.sum(weights)) or 1.0
    top_quarter_limit = int(np.ceil(_NUMBER_COUNT * 0.25))
    top_six = sum(
        float(weights[index])
        for index, ranking in enumerate(rankings)
        if number in ranking[:_NUMBERS_PER_DRAW]
    ) / weight_total
    top_quarter = sum(
        float(weights[index])
        for index, ranking in enumerate(rankings)
        if number in ranking[:top_quarter_limit]
    ) / weight_total
    long_horizon = float(np.mean(strengths[[0, 2, 4]]))
    recent_pattern = float(np.mean(strengths[[1, 3, 5]]))
    expert_values = np.asarray(
        [
            *strengths,
            base_row[31],
            top_six,
            top_quarter,
            long_horizon,
            recent_pattern,
            recent_pattern - long_horizon,
        ],
        dtype=float,
    )
    return np.concatenate((base_row[:_HISTORICAL_FEATURE_COUNT], expert_values))


def compile_svm_dataset(
    dataset_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> CompiledSvmDataset:
    """Compile one draw history into immutable leakage-safe feature batches."""
    resolved = dataset_path.resolve()
    draws = load_lotto_results_yaml(resolved)
    draws.prepare_predictions()
    evaluated_draws = max(len(draws.draws) - 1, 0)
    base_features = np.empty(
        (evaluated_draws, _NUMBER_COUNT, len(_SKLEARN_SVM_FEATURE_NAMES)),
        dtype=float,
    )
    compact_features = np.empty(
        (evaluated_draws, _NUMBER_COUNT, len(_COMPACT_FEATURE_NAMES)),
        dtype=float,
    )
    labels = np.zeros((evaluated_draws, _NUMBER_COUNT), dtype=np.uint8)
    gaps = np.zeros((evaluated_draws, _NUMBER_COUNT), dtype=np.uint16)
    expert_weights = np.empty(
        (evaluated_draws, len(_SKLEARN_SVM_EXPERT_IDS)), dtype=float
    )
    expert_rankings = np.empty(
        (evaluated_draws, len(_SKLEARN_SVM_EXPERT_IDS), _NUMBER_COUNT),
        dtype=np.uint8,
    )
    svc_rankings = np.empty((evaluated_draws, _NUMBER_COUNT), dtype=np.uint8)
    state = _StrategyState(("svc", "sklearn_svm"), total_draw_count=len(draws.draws))

    for draw_index, draw in enumerate(draws.draws[:-1]):
        drawn = {ball.value for ball in draw.balls}
        state.train(drawn)
        state.remember(drawn, draw.date)
        combined = draw.prediction
        if combined is None:
            raise ValueError("Combined predictions must be prepared first")
        strategies = state.build_strategies(combined, draw_index)
        svc = next(item for item in strategies if item.strategy_id == "svc")
        ranking_rows = [
            state.sklearn_svm_pending_rankings[strategy_id]
            for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        ]
        weights = np.asarray(
            [
                state._sklearn_svm_expert_weight(strategy_id)
                for strategy_id in _SKLEARN_SVM_EXPERT_IDS
            ],
            dtype=float,
        )
        expert_weights[draw_index] = weights
        expert_rankings[draw_index] = np.asarray(ranking_rows, dtype=np.uint8)
        svc_rankings[draw_index] = np.asarray(svc.top_numbers + tuple(
            item.number for item in svc.numbers[_NUMBERS_PER_DRAW:]
        ), dtype=np.uint8)
        current_gaps = state.current_gaps()
        target = {ball.value for ball in draws.draws[draw_index + 1].balls}
        for number in range(1, _NUMBER_COUNT + 1):
            base_row = np.asarray(state.sklearn_svm_pending_features[number])
            base_features[draw_index, number - 1] = base_row
            compact_features[draw_index, number - 1] = _compact_feature_row(
                base_row,
                number,
                ranking_rows,
                weights,
            )
            labels[draw_index, number - 1] = int(number in target)
            gaps[draw_index, number - 1] = current_gaps[number]
        if progress is not None:
            progress(draw_index + 1, evaluated_draws)

    return CompiledSvmDataset(
        source_path=str(resolved),
        fingerprint=_file_fingerprint(resolved),
        reference_draw_numbers=np.arange(1, evaluated_draws + 1, dtype=np.int32),
        base_features=base_features,
        compact_features=compact_features,
        labels=labels,
        gaps=gaps,
        expert_weights=expert_weights,
        expert_rankings=expert_rankings,
        svc_rankings=svc_rankings,
    )


def save_compiled_svm_dataset(dataset: CompiledSvmDataset, path: Path) -> None:
    """Persist a compiled dataset as a compressed, non-pickle NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "schemaVersion": SVM_EXPERIMENT_SCHEMA_VERSION,
            "sourcePath": dataset.source_path,
            "fingerprint": dataset.fingerprint,
        },
        sort_keys=True,
    )
    np.savez_compressed(
        path,
        metadata=np.asarray(metadata),
        reference_draw_numbers=dataset.reference_draw_numbers,
        base_features=dataset.base_features,
        compact_features=dataset.compact_features,
        labels=dataset.labels,
        gaps=dataset.gaps,
        expert_weights=dataset.expert_weights,
        expert_rankings=dataset.expert_rankings,
        svc_rankings=dataset.svc_rankings,
    )


def load_compiled_svm_dataset(path: Path) -> CompiledSvmDataset:
    """Load and validate a compiled feature archive without pickle support."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schemaVersion") != SVM_EXPERIMENT_SCHEMA_VERSION:
            raise ValueError("Compiled Scikit SVM dataset schema is stale")
        return CompiledSvmDataset(
            source_path=str(metadata["sourcePath"]),
            fingerprint=str(metadata["fingerprint"]),
            reference_draw_numbers=archive["reference_draw_numbers"].copy(),
            base_features=archive["base_features"].copy(),
            compact_features=archive["compact_features"].copy(),
            labels=archive["labels"].copy(),
            gaps=archive["gaps"].copy(),
            expert_weights=archive["expert_weights"].copy(),
            expert_rankings=archive["expert_rankings"].copy(),
            svc_rankings=archive["svc_rankings"].copy(),
        )


def feature_names_for_profile(profile: SvmInputProfile) -> tuple[str, ...]:
    """Return the ordered input schema for one profile."""
    if profile == "full_hybrid":
        return _SKLEARN_SVM_FEATURE_NAMES
    if profile == "historical_only":
        return _SKLEARN_SVM_FEATURE_NAMES[:_HISTORICAL_FEATURE_COUNT]
    if profile == "expert_only":
        return _SKLEARN_SVM_FEATURE_NAMES[_HISTORICAL_FEATURE_COUNT:]
    if profile == "no_static":
        return tuple(
            name
            for index, name in enumerate(_SKLEARN_SVM_FEATURE_NAMES)
            if index not in _STATIC_FEATURE_INDICES
        )
    if profile == "no_efficacy_interactions":
        return tuple(
            name
            for index, name in enumerate(_SKLEARN_SVM_FEATURE_NAMES)
            if index not in _EFFICACY_INTERACTION_INDICES
        )
    return _COMPACT_FEATURE_NAMES


def features_for_config(
    dataset: CompiledSvmDataset,
    config: SvmExperimentConfig,
) -> np.ndarray:
    """Select and analytically transform one configuration's feature tensor."""
    if config.input_profile == "compact_hybrid":
        selected = dataset.compact_features
        signed = _SIGNED_COMPACT_FEATURES
    else:
        if config.input_profile == "full_hybrid":
            indices = tuple(range(len(_SKLEARN_SVM_FEATURE_NAMES)))
        elif config.input_profile == "historical_only":
            indices = tuple(range(_HISTORICAL_FEATURE_COUNT))
        elif config.input_profile == "expert_only":
            indices = tuple(
                range(_HISTORICAL_FEATURE_COUNT, len(_SKLEARN_SVM_FEATURE_NAMES))
            )
        elif config.input_profile == "no_static":
            indices = tuple(
                index
                for index in range(len(_SKLEARN_SVM_FEATURE_NAMES))
                if index not in _STATIC_FEATURE_INDICES
            )
        else:
            indices = tuple(
                index
                for index in range(len(_SKLEARN_SVM_FEATURE_NAMES))
                if index not in _EFFICACY_INTERACTION_INDICES
            )
        selected = dataset.base_features[:, :, indices]
        signed = _SIGNED_BASE_FEATURES
    if not config.center_inputs:
        return selected
    centered = selected.copy()
    for index, name in enumerate(feature_names_for_profile(config.input_profile)):
        if name not in signed:
            centered[:, :, index] = centered[:, :, index] * 2 - 1
    return centered


def _new_estimator(config: SvmExperimentConfig) -> SGDClassifier:
    return SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=0.0001,
        learning_rate="optimal",
        fit_intercept=config.objective == "classification",
        average=config.average_weights,
        random_state=_RANDOM_SEED,
    )


def _cold_start_scores(dataset: CompiledSvmDataset, draw_index: int) -> np.ndarray:
    strengths = dataset.base_features[draw_index][:, _RAW_EXPERT_INDICES]
    weights = dataset.expert_weights[draw_index]
    weight_total = float(np.sum(weights)) or 1.0
    return np.sum(strengths * weights, axis=1) / weight_total


def _ranking(scores: np.ndarray, gaps: np.ndarray) -> tuple[int, ...]:
    return tuple(
        index + 1
        for index in sorted(
            range(_NUMBER_COUNT),
            key=lambda item: (-float(scores[item]), -int(gaps[item]), item + 1),
        )
    )


def _training_batch(
    config: SvmExperimentConfig,
    features: np.ndarray,
    labels: np.ndarray,
    prediction_scores: np.ndarray,
    gaps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.objective == "classification":
        positive_weight = (_NUMBER_COUNT - _NUMBERS_PER_DRAW) / _NUMBERS_PER_DRAW
        weights = np.where(labels == 1, positive_weight, 1.0)
        return features, labels.astype(np.uint8), weights

    positives = np.flatnonzero(labels == 1)
    negatives = list(np.flatnonzero(labels == 0))
    if config.objective == "hard_pairwise":
        negatives.sort(
            key=lambda item: (
                -float(prediction_scores[item]),
                -int(gaps[item]),
                item,
            )
        )
        negatives = negatives[:12]
    negative_indices = np.asarray(negatives, dtype=int)
    forward = np.vstack(
        [features[positive] - features[negative_indices] for positive in positives]
    )
    pair_features = np.vstack((forward, -forward))
    pair_labels = np.concatenate(
        (np.ones(len(forward), dtype=np.uint8), np.zeros(len(forward), dtype=np.uint8))
    )
    pair_weights = np.full(len(pair_labels), 1 / len(pair_labels), dtype=float)
    return pair_features, pair_labels, pair_weights


def _fit_batch(
    estimator: SGDClassifier,
    fitted: bool,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> bool:
    features, labels, weights = batch
    if fitted:
        estimator.partial_fit(features, labels, sample_weight=weights)
    else:
        estimator.partial_fit(
            features,
            labels,
            classes=np.asarray([0, 1]),
            sample_weight=weights,
        )
    return True


def replay_svm_configuration(
    dataset: CompiledSvmDataset,
    config: SvmExperimentConfig,
    *,
    stop: int | None = None,
) -> SvmReplayResult:
    """Chronologically replay one configuration without target leakage."""
    start_time = time.perf_counter()
    limit = dataset.evaluated_draws if stop is None else stop
    if not 0 <= limit <= dataset.evaluated_draws:
        raise ValueError("Replay stop is outside the compiled dataset")
    all_features = features_for_config(dataset, config)
    estimator = _new_estimator(config)
    fitted = False
    history: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
        maxlen=config.window_draws
    )
    hits: list[int] = []
    digest = hashlib.sha256()
    finite_scores = True
    unique_rankings = True
    valid_top_six = True

    for draw_index in range(limit):
        features = all_features[draw_index]
        if fitted:
            scores = np.asarray(estimator.decision_function(features), dtype=float)
        else:
            scores = _cold_start_scores(dataset, draw_index)
        finite_scores = finite_scores and bool(np.all(np.isfinite(scores)))
        ranking = _ranking(scores, dataset.gaps[draw_index])
        unique_rankings = unique_rankings and len(set(ranking)) == _NUMBER_COUNT
        valid_top_six = valid_top_six and len(set(ranking[:_NUMBERS_PER_DRAW])) == 6
        digest.update(np.asarray(ranking, dtype=np.uint8).tobytes())
        labels = dataset.labels[draw_index]
        hits.append(sum(int(labels[number - 1]) for number in ranking[:6]))

        batch = _training_batch(
            config,
            features,
            labels,
            scores,
            dataset.gaps[draw_index],
        )
        history.append(batch)
        fitted = _fit_batch(estimator, fitted, batch)
        trained_draws = draw_index + 1
        if (
            config.window_draws is not None
            and trained_draws > config.window_draws
            and trained_draws % config.refit_interval == 0
        ):
            estimator = _new_estimator(config)
            fitted = False
            for retained_batch in history:
                fitted = _fit_batch(estimator, fitted, retained_batch)

    return SvmReplayResult(
        config=config,
        feature_names=feature_names_for_profile(config.input_profile),
        hits=tuple(hits),
        ranking_digest=digest.hexdigest(),
        trained_draws=limit,
        finite_scores=finite_scores,
        unique_rankings=unique_rankings,
        valid_top_six=valid_top_six,
        duration_seconds=time.perf_counter() - start_time,
    )


def validation_fold_ranges(
    evaluated_draws: int,
    *,
    holdout_draws: int = 250,
    fold_draws: int = 250,
    fold_count: int = 3,
) -> tuple[tuple[int, int], ...]:
    """Return the three pre-holdout temporal validation fold ranges."""
    holdout_start = evaluated_draws - holdout_draws
    first_fold = holdout_start - fold_draws * fold_count
    if first_fold < 0:
        raise ValueError("Dataset is too short for temporal validation")
    return tuple(
        (first_fold + index * fold_draws, first_fold + (index + 1) * fold_draws)
        for index in range(fold_count)
    )


def _scope_summary(
    hits: Sequence[int],
    svc_hits: Sequence[int],
    start: int,
    stop: int,
) -> dict[str, Any]:
    selected_hits = hits[start:stop]
    selected_svc_hits = svc_hits[start:stop]
    evaluated = len(selected_hits)
    average = sum(selected_hits) / evaluated if evaluated else 0.0
    svc_average = sum(selected_svc_hits) / evaluated if evaluated else 0.0
    difference = average - svc_average
    above_random = average > SVM_RANDOM_EXPECTATION
    competitive = difference >= -SVM_MAXIMUM_REGRESSION - 1e-12
    return {
        "start": start,
        "stop": stop,
        "evaluatedDraws": evaluated,
        "totalHits": sum(selected_hits),
        "averageHitsPerDraw": average,
        "customSvcTotalHits": sum(selected_svc_hits),
        "customSvcAverageHitsPerDraw": svc_average,
        "expectedRandomHitsPerDraw": SVM_RANDOM_EXPECTATION,
        "randomDifference": average - SVM_RANDOM_EXPECTATION,
        "customSvcDifference": difference,
        "aboveRandom": above_random,
        "competitive": competitive,
        "beatsCustomSvc": difference > 0,
        "failed": not above_random or not competitive,
    }


def validation_variant_record(
    result: SvmReplayResult,
    dataset: CompiledSvmDataset,
    *,
    holdout_draws: int = 250,
) -> dict[str, Any]:
    """Build fold and leaderboard metrics for one pre-holdout replay."""
    holdout_start = dataset.evaluated_draws - holdout_draws
    if len(result.hits) != holdout_start:
        raise ValueError("Validation replay must stop exactly before the holdout")
    svc_hits = dataset.svc_hits.tolist()
    folds = [
        _scope_summary(result.hits, svc_hits, start, stop)
        for start, stop in validation_fold_ranges(dataset.evaluated_draws)
    ]
    whole = _scope_summary(result.hits, svc_hits, 0, holdout_start)
    differences = [float(fold["customSvcDifference"]) for fold in folds]
    return {
        "configKey": result.config.key,
        "config": asdict(result.config),
        "featureNames": list(result.feature_names),
        "featureCount": len(result.feature_names),
        "durationSeconds": result.duration_seconds,
        "trainedDraws": result.trained_draws,
        "rankingDigest": result.ranking_digest,
        "correctnessPassed": result.correctness_passed,
        "folds": folds,
        "foldFailures": sum(bool(fold["failed"]) for fold in folds),
        "worstFoldDifference": min(differences),
        "meanFoldDifference": sum(differences) / len(differences),
        "validationTotalHits": sum(int(fold["totalHits"]) for fold in folds),
        "wholePreHoldout": whole,
        "recentValidation": folds[-1],
    }


def _stable_sort_key(record: dict[str, Any]) -> tuple[object, ...]:
    config = record["config"]
    if not isinstance(config, dict):
        raise ValueError("Invalid validation configuration")
    return (
        int(record["foldFailures"]),
        -float(record["worstFoldDifference"]),
        -float(record["meanFoldDifference"]),
        -int(record["validationTotalHits"]),
        int(record["featureCount"]),
        0 if config["history_window"] == "unlimited" else 1,
        0 if config["objective"] == "classification" else 1,
        str(record["configKey"]),
    )


def _whole_sort_key(record: dict[str, Any]) -> tuple[object, ...]:
    whole = record["wholePreHoldout"]
    if not isinstance(whole, dict):
        raise ValueError("Invalid whole-history metrics")
    return (
        -float(whole["averageHitsPerDraw"]),
        int(record["featureCount"]),
        str(record["configKey"]),
    )


def _recent_sort_key(record: dict[str, Any]) -> tuple[object, ...]:
    recent = record["recentValidation"]
    if not isinstance(recent, dict):
        raise ValueError("Invalid recent-history metrics")
    return (
        -float(recent["averageHitsPerDraw"]),
        int(record["featureCount"]),
        str(record["configKey"]),
    )


def validation_leaderboards(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[str]]:
    """Rank every configuration under all three declared policies."""
    return {
        "stableMultiWindow": [
            str(record["configKey"]) for record in sorted(records, key=_stable_sort_key)
        ],
        "wholeHistory": [
            str(record["configKey"]) for record in sorted(records, key=_whole_sort_key)
        ],
        "recentHistory": [
            str(record["configKey"]) for record in sorted(records, key=_recent_sort_key)
        ],
    }


def result_scopes(
    result: SvmReplayResult,
    dataset: CompiledSvmDataset,
) -> dict[str, dict[str, Any]]:
    """Summarize whole/latest-500/latest-250 result scopes."""
    if len(result.hits) != dataset.evaluated_draws:
        raise ValueError("Scope reporting requires a full replay")
    svc_hits = dataset.svc_hits.tolist()
    scopes = {
        "wholeHistory": _scope_summary(
            result.hits, svc_hits, 0, dataset.evaluated_draws
        )
    }
    if dataset.evaluated_draws >= 500:
        scopes["latest500"] = _scope_summary(
            result.hits,
            svc_hits,
            dataset.evaluated_draws - 500,
            dataset.evaluated_draws,
        )
    if dataset.evaluated_draws >= 250:
        scopes["latest250"] = _scope_summary(
            result.hits,
            svc_hits,
            dataset.evaluated_draws - 250,
            dataset.evaluated_draws,
        )
    return scopes


def acceptance_report(
    results: Sequence[tuple[SvmReplayResult, CompiledSvmDataset]],
    *,
    deterministic: bool,
) -> dict[str, Any]:
    """Report all acceptance interpretations; stable competitive controls release."""
    dataset_reports: list[dict[str, Any]] = [
        {
            "dataset": dataset.source_path,
            "fingerprint": dataset.fingerprint,
            "scopes": result_scopes(result, dataset),
            "correctnessPassed": result.correctness_passed,
        }
        for result, dataset in results
    ]
    scopes = [
        scope
        for dataset_report in dataset_reports
        for scope in dataset_report["scopes"].values()
    ]
    stable = {
        "aboveRandomInEveryScope": all(bool(scope["aboveRandom"]) for scope in scopes),
        "competitiveInEveryScope": all(bool(scope["competitive"]) for scope in scopes),
        "beatsCustomSvcInAtLeastOneScope": any(
            bool(scope["beatsCustomSvc"]) for scope in scopes
        ),
        "deterministic": deterministic,
    }
    stable["passed"] = all(stable.values())
    winning_scopes = sum(bool(scope["beatsCustomSvc"]) for scope in scopes)
    whole_scopes = [
        report["scopes"]["wholeHistory"]
        for report in dataset_reports
    ]
    majority = {
        "winningScopes": winning_scopes,
        "eligibleScopes": len(scopes),
        "winsMoreThanHalf": winning_scopes > len(scopes) / 2,
        "aboveRandomOverall": all(
            float(scope["averageHitsPerDraw"]) > SVM_RANDOM_EXPECTATION
            for scope in whole_scopes
        ),
    }
    majority["passed"] = bool(
        majority["winsMoreThanHalf"] and majority["aboveRandomOverall"]
    )
    correctness = {
        "finiteScores": all(result.finite_scores for result, _dataset in results),
        "uniqueRankings": all(result.unique_rankings for result, _dataset in results),
        "validTopSix": all(result.valid_top_six for result, _dataset in results),
        "deterministic": deterministic,
    }
    correctness["passed"] = all(correctness.values())
    return {
        "datasets": dataset_reports,
        "stableCompetitive": stable,
        "majorityWinner": majority,
        "correctnessOnly": correctness,
        "promotionAuthorized": stable["passed"],
    }


_WORKER_DATASET: CompiledSvmDataset | None = None


def _initialize_worker(dataset_path: str) -> None:
    global _WORKER_DATASET
    _WORKER_DATASET = load_compiled_svm_dataset(Path(dataset_path))


def _worker_replay(
    config_value: dict[str, Any],
    stop: int,
) -> dict[str, Any]:
    if _WORKER_DATASET is None:
        raise RuntimeError("SVM experiment worker was not initialized")
    result = replay_svm_configuration(
        _WORKER_DATASET,
        SvmExperimentConfig.from_dict(config_value),
        stop=stop,
    )
    return result.checkpoint_record(_WORKER_DATASET.fingerprint, stop)


def load_replay_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_fingerprint: str,
    stop: int,
) -> dict[str, SvmReplayResult]:
    """Load completed matching variants from an append-only JSONL checkpoint."""
    completed: dict[str, SvmReplayResult] = {}
    if not checkpoint_path.exists():
        return completed
    for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if (
            value.get("datasetFingerprint") == dataset_fingerprint
            and value.get("stop") == stop
        ):
            result = SvmReplayResult.from_checkpoint(value)
            completed[result.config.key] = result
    return completed


def evaluate_svm_configurations(
    compiled_path: Path,
    configurations: Sequence[SvmExperimentConfig],
    *,
    stop: int,
    checkpoint_path: Path,
    workers: int = 1,
    progress: ProgressCallback | None = None,
) -> tuple[SvmReplayResult, ...]:
    """Replay all requested variants with resumable per-variant checkpoints."""
    if workers < 1:
        raise ValueError("Worker count must be positive")
    dataset = load_compiled_svm_dataset(compiled_path)
    completed = load_replay_checkpoint(
        checkpoint_path,
        dataset_fingerprint=dataset.fingerprint,
        stop=stop,
    )
    requested = {config.key: config for config in configurations}
    completed = {key: value for key, value in completed.items() if key in requested}
    pending = [config for key, config in requested.items() if key not in completed]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(requested)
    if progress is not None:
        progress(len(completed), total)

    def record(value: dict[str, Any]) -> None:
        result = SvmReplayResult.from_checkpoint(value)
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
            checkpoint.write(json.dumps(value, sort_keys=True) + "\n")
        completed[result.config.key] = result
        if progress is not None:
            progress(len(completed), total)

    if workers == 1:
        for config in pending:
            result = replay_svm_configuration(dataset, config, stop=stop)
            record(result.checkpoint_record(dataset.fingerprint, stop))
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(str(compiled_path.resolve()),),
        ) as executor:
            futures = {
                executor.submit(_worker_replay, asdict(config), stop): config
                for config in pending
            }
            for future in as_completed(futures):
                record(future.result())
    return tuple(completed[key] for key in sorted(requested))
