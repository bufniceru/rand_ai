"""Benchmark a leakage-safe Lottery Ticket neural candidate ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from rand_ai import Draw, load_lotto_results_yaml
from rand_ai.strategy_prediction import (
    _LAG_LOGISTIC_FEATURE_NAMES,
    _SKLEARN_SVM_FEATURE_NAMES,
    _StrategyState,
)

_NUMBER_COUNT = 49
_NUMBERS_PER_DRAW = 6
_BASELINE_IDS = ("randomness", "sklearn_svm", "lag_logistic")
_LAG_FEATURE_COUNT = 4
_FEATURE_NAMES = (
    *_SKLEARN_SVM_FEATURE_NAMES,
    *_LAG_LOGISTIC_FEATURE_NAMES[:_LAG_FEATURE_COUNT],
)
_DEFAULT_SEEDS = tuple(range(20260626, 20260631))
_PRUNABLE_LAYER_NAMES = ("0.weight", "2.weight", "4.weight")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Store deterministic experiment and chronological split settings."""

    warmup_draws: int = 100
    validation_draws: int = 150
    holdout_draws: int = 250
    minimum_train_draws: int = 200
    seeds: tuple[int, ...] = _DEFAULT_SEEDS
    pruning_rounds: int = 7
    prune_fraction: float = 0.20
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 256
    learning_rate: float = 0.001


@dataclass(frozen=True, slots=True)
class CandidateDataset:
    """Keep draw-grouped candidate features, targets, and baseline tickets."""

    target_draw_numbers: np.ndarray
    features: np.ndarray
    targets: np.ndarray
    baseline_top_numbers: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Identify chronological draw rows used by each benchmark stage."""

    train: slice
    validation: slice
    holdout: slice


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Store training-only feature scaling parameters."""

    mean: np.ndarray
    scale: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.scale


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """Store one fitted network without retaining a live torch module."""

    state: Mapping[str, Any]
    masks: Mapping[str, Any]
    epochs: int
    training_seconds: float


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Store model metrics and draw-aligned Top-6 hit counts."""

    hits: np.ndarray
    brier_score: float
    log_loss: float
    probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class RoundResult:
    """Store one seed's trained network at a pruning level."""

    round_index: int
    active_weights: int
    total_weights: int
    trained: TrainedModel
    validation: Evaluation


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise RuntimeError(
            "PyTorch is required for this research command. Run it with "
            "`uv run --group research python scripts/benchmark_sparse_neural_ticket.py`."
        ) from error
    return torch, nn


def build_candidate_dataset(draws: Sequence[Draw]) -> CandidateDataset:
    """Build next-draw samples while advancing state in production order."""

    if len(draws) < 2:
        raise ValueError("At least two draws are required to build candidate targets")
    state = _StrategyState(_BASELINE_IDS, total_draw_count=len(draws))
    target_draw_numbers: list[int] = []
    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    baseline_tickets: dict[str, list[tuple[int, ...]]] = {
        strategy_id: [] for strategy_id in _BASELINE_IDS
    }

    for draw_index, draw in enumerate(draws[:-1]):
        drawn = {ball.value for ball in draw.balls}
        state.train(drawn)
        state.remember(drawn, draw.date)
        combined = draw.prediction
        if combined is None:
            raise ValueError("Combined predictions must be prepared first")
        strategies = {
            strategy.strategy_id: strategy
            for strategy in state.build_strategies(combined, draw_index)
        }
        svm_features = np.asarray(
            [
                state.sklearn_svm_pending_features[number]
                for number in range(1, _NUMBER_COUNT + 1)
            ],
            dtype=np.float64,
        )
        lag_features = np.asarray(
            state.lag_logistic_pending_features,
            dtype=np.float64,
        )[:, :_LAG_FEATURE_COUNT]
        features = np.concatenate((svm_features, lag_features), axis=1)
        if features.shape != (_NUMBER_COUNT, len(_FEATURE_NAMES)):
            raise AssertionError("Unexpected Sparse Neural Ticket feature shape")
        if not np.isfinite(features).all():
            raise ValueError("Candidate features contain non-finite values")

        target_numbers = {
            ball.value for ball in draws[draw_index + 1].balls
        }
        targets = np.asarray(
            [
                float(number in target_numbers)
                for number in range(1, _NUMBER_COUNT + 1)
            ],
            dtype=np.float64,
        )
        target_draw_numbers.append(draw_index + 2)
        feature_batches.append(features)
        target_batches.append(targets)
        for strategy_id in _BASELINE_IDS:
            baseline_tickets[strategy_id].append(
                strategies[strategy_id].top_numbers
            )

    return CandidateDataset(
        target_draw_numbers=np.asarray(target_draw_numbers, dtype=np.int64),
        features=np.stack(feature_batches),
        targets=np.stack(target_batches),
        baseline_top_numbers={
            strategy_id: np.asarray(tickets, dtype=np.int64)
            for strategy_id, tickets in baseline_tickets.items()
        },
    )


def chronological_split(
    row_count: int,
    config: BenchmarkConfig,
    *,
    minimum_validation_draws: int = 50,
    minimum_holdout_draws: int = 50,
) -> DatasetSplit:
    """Return draw-aligned train, validation, and untouched holdout slices."""

    if config.warmup_draws < 0:
        raise ValueError("Warm-up draws cannot be negative")
    if config.validation_draws < minimum_validation_draws:
        raise ValueError(
            f"Validation requires at least {minimum_validation_draws} target draws"
        )
    if config.holdout_draws < minimum_holdout_draws:
        raise ValueError(
            f"Holdout requires at least {minimum_holdout_draws} target draws"
        )
    holdout_start = row_count - config.holdout_draws
    validation_start = holdout_start - config.validation_draws
    train_start = config.warmup_draws
    train_draws = validation_start - train_start
    if train_draws < config.minimum_train_draws:
        required = (
            config.warmup_draws
            + config.minimum_train_draws
            + config.validation_draws
            + config.holdout_draws
        )
        raise ValueError(
            f"Dataset supplies {row_count} target draws; at least {required} are "
            "required by the requested warm-up, training, validation, and holdout split"
        )
    return DatasetSplit(
        train=slice(train_start, validation_start),
        validation=slice(validation_start, holdout_start),
        holdout=slice(holdout_start, row_count),
    )


def fit_standardizer(training_features: np.ndarray) -> Standardizer:
    """Fit per-feature scaling exclusively from training candidate rows."""

    flattened = training_features.reshape(-1, training_features.shape[-1])
    feature_mean = flattened.mean(axis=0)
    feature_scale = flattened.std(axis=0)
    feature_scale = np.where(feature_scale > 0, feature_scale, 1.0)
    return Standardizer(mean=feature_mean, scale=feature_scale)


def _configure_torch(seed: int) -> Any:
    torch, _nn = _torch_modules()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch


def _new_model(input_count: int, seed: int) -> Any:
    torch = _configure_torch(seed)
    _torch, nn = _torch_modules()
    model = nn.Sequential(
        nn.Linear(input_count, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    return model


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: tensor.detach().clone() for name, tensor in state.items()}


def _initial_state(input_count: int, seed: int) -> dict[str, Any]:
    return _clone_state(_new_model(input_count, seed).state_dict())


def _full_masks(initial_state: Mapping[str, Any]) -> dict[str, Any]:
    torch, _nn = _torch_modules()
    return {
        name: torch.ones_like(initial_state[name])
        for name in _PRUNABLE_LAYER_NAMES
    }


def _clone_masks(masks: Mapping[str, Any]) -> dict[str, Any]:
    return {name: mask.detach().clone() for name, mask in masks.items()}


def _apply_masks(model: Any, masks: Mapping[str, Any]) -> None:
    torch, _nn = _torch_modules()
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            parameters[name].mul_(mask)


def _model_from_state(
    input_count: int,
    state: Mapping[str, Any],
    masks: Mapping[str, Any],
    seed: int,
) -> Any:
    model = _new_model(input_count, seed)
    model.load_state_dict(state)
    _apply_masks(model, masks)
    return model


def train_model(
    initial_state: Mapping[str, Any],
    masks: Mapping[str, Any],
    training_features: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    config: BenchmarkConfig,
) -> TrainedModel:
    """Rewind, train with fixed masks, and retain minimum validation loss."""

    if config.max_epochs < 1 or config.patience < 1:
        raise ValueError("Epoch and patience settings must be positive")
    torch = _configure_torch(seed)
    _torch, nn = _torch_modules()
    input_count = training_features.shape[-1]
    model = _model_from_state(input_count, initial_state, masks, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([43 / 6], dtype=torch.float32)
    )
    flat_training_features = torch.as_tensor(
        training_features.reshape(-1, input_count), dtype=torch.float32
    )
    flat_training_targets = torch.as_tensor(
        training_targets.reshape(-1, 1), dtype=torch.float32
    )
    flat_validation_features = torch.as_tensor(
        validation_features.reshape(-1, input_count), dtype=torch.float32
    )
    flat_validation_targets = torch.as_tensor(
        validation_targets.reshape(-1, 1), dtype=torch.float32
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    parameter_map = dict(model.named_parameters())
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    stale_epochs = 0
    epochs_completed = 0
    started = time.perf_counter()

    for epoch in range(config.max_epochs):
        model.train()
        order = torch.randperm(
            flat_training_features.shape[0], generator=generator
        )
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(flat_training_features[indices])
            loss = criterion(logits, flat_training_targets[indices])
            loss.backward()
            for name, mask in masks.items():
                gradient = parameter_map[name].grad
                if gradient is not None:
                    gradient.mul_(mask)
            optimizer.step()
            _apply_masks(model, masks)

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                criterion(
                    model(flat_validation_features), flat_validation_targets
                ).item()
            )
        epochs_completed = epoch + 1
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_state = _clone_state(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:  # pragma: no cover - one epoch always records a state
        raise AssertionError("Training did not produce a model state")
    model.load_state_dict(best_state)
    _apply_masks(model, masks)
    return TrainedModel(
        state=_clone_state(model.state_dict()),
        masks=_clone_masks(masks),
        epochs=epochs_completed,
        training_seconds=time.perf_counter() - started,
    )


def predict_probabilities(
    trained: TrainedModel,
    features: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Return draw-grouped candidate probabilities for a frozen network."""

    torch = _configure_torch(seed)
    input_count = features.shape[-1]
    model = _model_from_state(input_count, trained.state, trained.masks, seed)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.as_tensor(
                features.reshape(-1, input_count), dtype=torch.float32
            )
        )
        probabilities = torch.sigmoid(logits).reshape(
            features.shape[0], _NUMBER_COUNT
        )
    return probabilities.detach().cpu().numpy().astype(np.float64)


def _top_six_hits(probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
    candidate_numbers = np.arange(_NUMBER_COUNT)
    hits: list[int] = []
    for probability_row, target_row in zip(probabilities, targets):
        ranking = np.lexsort((candidate_numbers, -probability_row))
        hits.append(int(target_row[ranking[:_NUMBERS_PER_DRAW]].sum()))
    return np.asarray(hits, dtype=np.int64)


def evaluate_model(
    trained: TrainedModel,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> Evaluation:
    probabilities = predict_probabilities(trained, features, seed=seed)
    clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
    return Evaluation(
        hits=_top_six_hits(probabilities, targets),
        brier_score=float(np.mean((probabilities - targets) ** 2)),
        log_loss=float(
            -np.mean(
                targets * np.log(clipped)
                + (1 - targets) * np.log(1 - clipped)
            )
        ),
        probabilities=probabilities,
    )


def _active_weight_count(masks: Mapping[str, Any]) -> int:
    return sum(int(mask.sum().item()) for mask in masks.values())


def _total_weight_count(masks: Mapping[str, Any]) -> int:
    return sum(int(mask.numel()) for mask in masks.values())


def prune_global_magnitude(
    trained_state: Mapping[str, Any],
    masks: Mapping[str, Any],
    prune_fraction: float,
) -> dict[str, Any]:
    """Prune an exact fraction of active weights with deterministic tie breaks."""

    if not 0 < prune_fraction < 1:
        raise ValueError("Prune fraction must be strictly between zero and one")
    updated = _clone_masks(masks)
    positions: list[tuple[float, int, str, int]] = []
    ordinal = 0
    for name in _PRUNABLE_LAYER_NAMES:
        weights = trained_state[name].detach().cpu().numpy().reshape(-1)
        active = masks[name].detach().cpu().numpy().reshape(-1)
        for flat_index in np.flatnonzero(active):
            positions.append(
                (abs(float(weights[flat_index])), ordinal, name, int(flat_index))
            )
            ordinal += 1
    active_count = len(positions)
    remaining_count = round(active_count * (1 - prune_fraction))
    prune_count = active_count - remaining_count
    for _magnitude, _ordinal, name, flat_index in sorted(positions)[:prune_count]:
        updated[name].view(-1)[flat_index] = 0
    return updated


def random_mask_like(
    masks: Mapping[str, Any],
    active_count: int,
    *,
    seed: int,
) -> dict[str, Any]:
    """Create a random global mask with the requested exact active count."""

    total_count = _total_weight_count(masks)
    if not 0 <= active_count <= total_count:
        raise ValueError("Random-mask active count is outside the model size")
    rng = np.random.default_rng(seed)
    active_positions = set(
        int(index)
        for index in rng.choice(total_count, size=active_count, replace=False)
    )
    torch, _nn = _torch_modules()
    result: dict[str, Any] = {}
    offset = 0
    for name in _PRUNABLE_LAYER_NAMES:
        count = int(masks[name].numel())
        values = [
            1.0 if offset + index in active_positions else 0.0
            for index in range(count)
        ]
        result[name] = torch.as_tensor(
            values, dtype=masks[name].dtype
        ).reshape(masks[name].shape)
        offset += count
    return result


def run_pruning_seed(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    config: BenchmarkConfig,
) -> tuple[RoundResult, ...]:
    """Train and evaluate every iterative pruning level for one seed."""

    initial_state = _initial_state(training_features.shape[-1], seed)
    masks = _full_masks(initial_state)
    total_weights = _total_weight_count(masks)
    results: list[RoundResult] = []
    for round_index in range(config.pruning_rounds + 1):
        trained = train_model(
            initial_state,
            masks,
            training_features,
            training_targets,
            validation_features,
            validation_targets,
            seed=seed,
            config=config,
        )
        validation = evaluate_model(
            trained, validation_features, validation_targets, seed=seed
        )
        results.append(
            RoundResult(
                round_index=round_index,
                active_weights=_active_weight_count(masks),
                total_weights=total_weights,
                trained=trained,
                validation=validation,
            )
        )
        if round_index < config.pruning_rounds:
            masks = prune_global_magnitude(
                trained.state, masks, config.prune_fraction
            )
    return tuple(results)


def select_pruning_round(results: Sequence[Sequence[RoundResult]]) -> int:
    """Select by validation hits, Brier score, then smaller active model."""

    if not results or not results[0]:
        raise ValueError("At least one pruning result is required")
    round_count = len(results[0])
    if any(len(seed_results) != round_count for seed_results in results):
        raise ValueError("Every seed must evaluate the same pruning rounds")
    return min(
        range(round_count),
        key=lambda round_index: (
            -mean(
                float(seed_results[round_index].validation.hits.mean())
                for seed_results in results
            ),
            mean(
                seed_results[round_index].validation.brier_score
                for seed_results in results
            ),
            results[0][round_index].active_weights,
        ),
    )


def _hit_summary(
    hits: Sequence[int | float] | np.ndarray,
) -> dict[str, Any]:
    values = [float(value) for value in hits]
    distribution = Counter(int(value) for value in values)
    return {
        "evaluatedDraws": len(values),
        "totalHits": sum(values),
        "averageHitsPerDraw": mean(values),
        "hitDistribution": {
            str(hit_count): distribution[hit_count]
            for hit_count in range(_NUMBERS_PER_DRAW + 1)
        },
    }


def _evaluation_payload(evaluation: Evaluation) -> dict[str, Any]:
    return {
        **_hit_summary(evaluation.hits),
        "brierScore": evaluation.brier_score,
        "logLoss": evaluation.log_loss,
    }


def aggregate_evaluations(
    evaluations: Sequence[tuple[int, Evaluation]],
) -> tuple[dict[str, Any], np.ndarray]:
    """Aggregate seed metrics without disguising individual ticket results."""

    if not evaluations:
        raise ValueError("At least one evaluation is required")
    mean_hits = np.mean(
        np.stack([evaluation.hits for _seed, evaluation in evaluations]),
        axis=0,
    )
    mean_distribution = {
        str(hit_count): mean(
            Counter(int(hit) for hit in evaluation.hits)[hit_count]
            for _seed, evaluation in evaluations
        )
        for hit_count in range(_NUMBERS_PER_DRAW + 1)
    }
    payload = {
        "perSeed": [
            {"seed": seed, **_evaluation_payload(evaluation)}
            for seed, evaluation in evaluations
        ],
        "meanAcrossSeeds": {
            "evaluatedDraws": len(mean_hits),
            "totalHits": float(mean_hits.sum()),
            "averageHitsPerDraw": float(mean_hits.mean()),
            "meanHitDistributionAcrossSeeds": mean_distribution,
            "brierScore": mean(
                evaluation.brier_score for _seed, evaluation in evaluations
            ),
            "logLoss": mean(
                evaluation.log_loss for _seed, evaluation in evaluations
            ),
        },
    }
    return payload, mean_hits


def _baseline_hits(
    tickets: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    hits = []
    for ticket, target_row in zip(tickets, targets):
        hits.append(int(target_row[np.asarray(ticket) - 1].sum()))
    return np.asarray(hits, dtype=np.int64)


def _paired_summary(
    candidate_hits: Sequence[int | float] | np.ndarray,
    baseline_hits: Sequence[int | float] | np.ndarray,
) -> dict[str, Any]:
    differences = [
        float(candidate) - float(baseline)
        for candidate, baseline in zip(candidate_hits, baseline_hits)
    ]
    average = mean(differences)
    standard_error = (
        stdev(differences) / math.sqrt(len(differences))
        if len(differences) > 1
        else 0.0
    )
    return {
        "meanHitDifference": average,
        "meanDifference95Interval": [
            average - 1.96 * standard_error,
            average + 1.96 * standard_error,
        ],
        "candidateWins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "baselineWins": sum(value < 0 for value in differences),
    }


def _feature_survival(
    selected_results: Sequence[RoundResult],
) -> list[dict[str, Any]]:
    per_seed_masks = [
        result.trained.masks["0.weight"].detach().cpu().numpy()
        for result in selected_results
    ]
    return [
        {
            "name": feature_name,
            "meanSurvival": float(
                np.mean([mask[:, index].mean() for mask in per_seed_masks])
            ),
            "perSeedSurvival": [
                float(mask[:, index].mean()) for mask in per_seed_masks
            ],
        }
        for index, feature_name in enumerate(_FEATURE_NAMES)
    ]


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def build_promotion(
    *,
    active_fraction: float,
    ticket_payload: Mapping[str, Any],
    dense_payload: Mapping[str, Any],
    ticket_hits: np.ndarray,
    dense_hits: np.ndarray,
    baseline_hits: Mapping[str, np.ndarray],
    deterministic: bool,
    finite: bool,
) -> dict[str, Any]:
    """Evaluate every fixed promotion criterion with diagnostic values."""

    compact_ids = ("sklearn_svm", "lag_logistic")
    best_compact_id = max(
        compact_ids,
        key=lambda strategy_id: float(baseline_hits[strategy_id].mean()),
    )
    dense_difference = float(ticket_hits.mean() - dense_hits.mean())
    compact_comparison = _paired_summary(
        ticket_hits, baseline_hits[best_compact_id]
    )
    random_comparison = _paired_summary(ticket_hits, baseline_hits["randomness"])
    ticket_brier = float(
        ticket_payload["meanAcrossSeeds"]["brierScore"]  # type: ignore[index]
    )
    dense_brier = float(
        dense_payload["meanAcrossSeeds"]["brierScore"]  # type: ignore[index]
    )
    checks = [
        {
            "name": "atLeastHalfWeightsRemoved",
            "passed": active_fraction <= 0.50,
            "value": active_fraction,
            "threshold": "<= 0.50 active fraction",
        },
        {
            "name": "denseHitNonInferiority",
            "passed": dense_difference >= -0.02,
            "value": dense_difference,
            "threshold": ">= -0.02 hits/draw",
        },
        {
            "name": "compactBaselineNonInferiority",
            "passed": compact_comparison["meanDifference95Interval"][0] >= -0.05,
            "value": compact_comparison["meanDifference95Interval"][0],
            "threshold": ">= -0.05 lower 95% bound",
            "baseline": best_compact_id,
        },
        {
            "name": "randomBaselineSuperiority",
            "passed": random_comparison["meanDifference95Interval"][0] > 0,
            "value": random_comparison["meanDifference95Interval"][0],
            "threshold": "> 0 lower 95% bound",
        },
        {
            "name": "denseBrierNonInferiority",
            "passed": ticket_brier - dense_brier <= 0.005,
            "value": ticket_brier - dense_brier,
            "threshold": "<= 0.005 Brier-score increase",
        },
        {
            "name": "finiteDeterministicOutputs",
            "passed": finite and deterministic,
            "value": {"finite": finite, "deterministic": deterministic},
            "threshold": "both true",
        },
    ]
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "bestCompactBaseline": best_compact_id,
        "pairedComparisons": {
            best_compact_id: compact_comparison,
            "randomness": random_comparison,
            "dense": _paired_summary(ticket_hits, dense_hits),
        },
    }


def _split_payload(
    dataset: CandidateDataset,
    split: DatasetSplit,
) -> dict[str, Any]:
    def describe(name: str, rows: slice) -> dict[str, Any]:
        target_numbers = dataset.target_draw_numbers[rows]
        return {
            "name": name,
            "draws": len(target_numbers),
            "firstTargetDraw": int(target_numbers[0]),
            "lastTargetDraw": int(target_numbers[-1]),
        }

    return {
        "train": describe("train", split.train),
        "validation": describe("validation", split.validation),
        "holdout": describe("holdout", split.holdout),
    }


def _config_payload(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "warmupDraws": config.warmup_draws,
        "validationDraws": config.validation_draws,
        "holdoutDraws": config.holdout_draws,
        "minimumTrainDraws": config.minimum_train_draws,
        "seeds": list(config.seeds),
        "pruningRounds": config.pruning_rounds,
        "pruneFraction": config.prune_fraction,
        "maxEpochs": config.max_epochs,
        "patience": config.patience,
        "batchSize": config.batch_size,
        "learningRate": config.learning_rate,
        "architecture": [len(_FEATURE_NAMES), 32, 16, 1],
        "positiveClassWeight": 43 / 6,
    }


def write_runtime_artifact(
    artifact_path: Path,
    *,
    draws: Sequence[Draw],
    split: DatasetSplit,
    standardizer: Standardizer,
    config: BenchmarkConfig,
    selected_round: int,
    selected_results: Sequence[RoundResult],
    promotion_passed: bool,
) -> None:
    """Persist the frozen ticket ensemble for leakage-safe NumPy inference."""

    if split.holdout.start is None:
        raise ValueError("Runtime artifact requires a bounded holdout split")
    activation_reference_draw = split.holdout.start + 1
    prefix = [
        [draw.date, *sorted(ball.value for ball in draw.balls)]
        for draw in draws[:activation_reference_draw]
    ]
    prefix_fingerprint = hashlib.sha256(
        json.dumps(prefix, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    example = selected_results[0]
    metadata = {
        "schemaVersion": 1,
        "strategyId": "sparse_neural_ticket",
        "activationReferenceDraw": activation_reference_draw,
        "prefixFingerprint": prefix_fingerprint,
        "selectedRound": selected_round,
        "activeWeights": example.active_weights,
        "totalWeights": example.total_weights,
        "promotionPassed": promotion_passed,
        "seeds": list(config.seeds),
        "featureNames": list(_FEATURE_NAMES),
    }
    arrays: dict[str, Any] = {
        "metadata": np.asarray(json.dumps(metadata, separators=(",", ":"))),
        "feature_mean": standardizer.mean.astype(np.float64),
        "feature_scale": standardizer.scale.astype(np.float64),
    }
    layer_names = ("0", "2", "4")
    for seed, result in zip(config.seeds, selected_results):
        for layer_index, layer_name in enumerate(layer_names):
            for parameter_name in ("weight", "bias"):
                state_name = f"{layer_name}.{parameter_name}"
                arrays[
                    f"seed_{seed}_layer_{layer_index}_{parameter_name}"
                ] = (
                    result.trained.state[state_name]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as artifact_file:
        np.savez_compressed(artifact_file, **arrays)


def benchmark(
    dataset_path: Path,
    *,
    config: BenchmarkConfig = BenchmarkConfig(),
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Run the complete research gate and return a JSON-safe report."""

    draws = load_lotto_results_yaml(dataset_path)
    draws.prepare_predictions()
    dataset = build_candidate_dataset(draws.draws)
    split = chronological_split(len(dataset.target_draw_numbers), config)
    standardizer = fit_standardizer(dataset.features[split.train])
    scaled_features = standardizer.transform(dataset.features)
    train_features = scaled_features[split.train]
    train_targets = dataset.targets[split.train]
    validation_features = scaled_features[split.validation]
    validation_targets = dataset.targets[split.validation]
    holdout_features = scaled_features[split.holdout]
    holdout_targets = dataset.targets[split.holdout]

    seed_results: list[tuple[RoundResult, ...]] = []
    for seed_index, seed in enumerate(config.seeds, start=1):
        print(
            f"Sparse ticket seed {seed_index}/{len(config.seeds)}: {seed}",
            file=sys.stderr,
            flush=True,
        )
        seed_results.append(
            run_pruning_seed(
                train_features,
                train_targets,
                validation_features,
                validation_targets,
                seed=seed,
                config=config,
            )
        )
    selected_round = select_pruning_round(seed_results)
    selected_results = [results[selected_round] for results in seed_results]
    dense_results = [results[0] for results in seed_results]

    validation_ticket_evaluations = [
        (seed, result.validation)
        for seed, result in zip(config.seeds, selected_results)
    ]
    validation_dense_evaluations = [
        (seed, result.validation)
        for seed, result in zip(config.seeds, dense_results)
    ]
    holdout_ticket_evaluations = [
        (
            seed,
            evaluate_model(
                result.trained, holdout_features, holdout_targets, seed=seed
            ),
        )
        for seed, result in zip(config.seeds, selected_results)
    ]
    holdout_dense_evaluations = [
        (
            seed,
            evaluate_model(
                result.trained, holdout_features, holdout_targets, seed=seed
            ),
        )
        for seed, result in zip(config.seeds, dense_results)
    ]
    validation_ticket, _validation_ticket_hits = aggregate_evaluations(
        validation_ticket_evaluations
    )
    validation_dense, _validation_dense_hits = aggregate_evaluations(
        validation_dense_evaluations
    )
    holdout_ticket, holdout_ticket_hits = aggregate_evaluations(
        holdout_ticket_evaluations
    )
    holdout_dense, holdout_dense_hits = aggregate_evaluations(
        holdout_dense_evaluations
    )

    validation_baseline_hits = {
        strategy_id: _baseline_hits(
            dataset.baseline_top_numbers[strategy_id][split.validation],
            validation_targets,
        )
        for strategy_id in _BASELINE_IDS
    }
    holdout_baseline_hits = {
        strategy_id: _baseline_hits(
            dataset.baseline_top_numbers[strategy_id][split.holdout],
            holdout_targets,
        )
        for strategy_id in _BASELINE_IDS
    }
    validation_baselines = {
        strategy_id: _hit_summary(hits)
        for strategy_id, hits in validation_baseline_hits.items()
    }
    holdout_baselines = {
        strategy_id: _hit_summary(hits)
        for strategy_id, hits in holdout_baseline_hits.items()
    }

    control_evaluations: dict[str, dict[str, list[tuple[int, Evaluation]]]] = {
        "randomReinitialization": {"validation": [], "holdout": []},
        "randomMask": {"validation": [], "holdout": []},
    }
    control_training: dict[str, list[tuple[int, TrainedModel]]] = {
        "randomReinitialization": [],
        "randomMask": [],
    }
    for seed, selected in zip(config.seeds, selected_results):
        reinitialized_state = _initial_state(len(_FEATURE_NAMES), seed + 1_000_000)
        reinitialized = train_model(
            reinitialized_state,
            selected.trained.masks,
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            seed=seed + 1_000_000,
            config=config,
        )
        random_masks = random_mask_like(
            selected.trained.masks,
            selected.active_weights,
            seed=seed + 2_000_000,
        )
        original_state = _initial_state(len(_FEATURE_NAMES), seed)
        random_mask_model = train_model(
            original_state,
            random_masks,
            train_features,
            train_targets,
            validation_features,
            validation_targets,
            seed=seed,
            config=config,
        )
        for control_name, trained, evaluation_seed in (
            ("randomReinitialization", reinitialized, seed + 1_000_000),
            ("randomMask", random_mask_model, seed),
        ):
            control_training[control_name].append((seed, trained))
            control_evaluations[control_name]["validation"].append(
                (
                    seed,
                    evaluate_model(
                        trained,
                        validation_features,
                        validation_targets,
                        seed=evaluation_seed,
                    ),
                )
            )
            control_evaluations[control_name]["holdout"].append(
                (
                    seed,
                    evaluate_model(
                        trained,
                        holdout_features,
                        holdout_targets,
                        seed=evaluation_seed,
                    ),
                )
            )

    controls: dict[str, Any] = {}
    for control_name, scopes in control_evaluations.items():
        validation_payload, _validation_hits = aggregate_evaluations(
            scopes["validation"]
        )
        holdout_payload, _holdout_hits = aggregate_evaluations(scopes["holdout"])
        controls[control_name] = {
            "validation": validation_payload,
            "holdout": holdout_payload,
            "training": {
                "perSeed": [
                    {
                        "seed": seed,
                        "epochs": trained.epochs,
                        "seconds": trained.training_seconds,
                    }
                    for seed, trained in control_training[control_name]
                ],
                "meanEpochs": mean(
                    trained.epochs
                    for _seed, trained in control_training[control_name]
                ),
                "meanSeconds": mean(
                    trained.training_seconds
                    for _seed, trained in control_training[control_name]
                ),
            },
        }

    pruning_rounds = []
    for round_index in range(config.pruning_rounds + 1):
        round_evaluations = [
            (seed, results[round_index].validation)
            for seed, results in zip(config.seeds, seed_results)
        ]
        validation_payload, _hits = aggregate_evaluations(round_evaluations)
        example = seed_results[0][round_index]
        pruning_rounds.append(
            {
                "round": round_index,
                "activeWeights": example.active_weights,
                "totalWeights": example.total_weights,
                "activeFraction": example.active_weights / example.total_weights,
                "validation": validation_payload,
                "meanTrainingSeconds": mean(
                    results[round_index].trained.training_seconds
                    for results in seed_results
                ),
                "meanEpochs": mean(
                    results[round_index].trained.epochs
                    for results in seed_results
                ),
            }
        )

    selected_example = selected_results[0]
    report: dict[str, Any] = {
        "config": _config_payload(config),
        "dataset": {
            "path": str(dataset_path.resolve()),
            "drawCount": len(draws.draws),
            "targetRows": len(dataset.target_draw_numbers),
        },
        "splits": _split_payload(dataset, split),
        "featureSchema": {
            "count": len(_FEATURE_NAMES),
            "names": list(_FEATURE_NAMES),
            "standardization": "Training rows only; frozen for validation and holdout",
        },
        "pruningRounds": pruning_rounds,
        "selectedSparsity": {
            "round": selected_round,
            "activeWeights": selected_example.active_weights,
            "totalWeights": selected_example.total_weights,
            "activeFraction": (
                selected_example.active_weights / selected_example.total_weights
            ),
            "selectionRule": (
                "Highest mean validation Top-6 hits; then lower Brier score; "
                "then fewer active weights"
            ),
        },
        "validation": {
            "ticket": validation_ticket,
            "dense": validation_dense,
            "baselines": validation_baselines,
        },
        "holdout": {
            "ticket": holdout_ticket,
            "dense": holdout_dense,
            "baselines": holdout_baselines,
            "pairedComparisons": {
                strategy_id: _paired_summary(holdout_ticket_hits, hits)
                for strategy_id, hits in holdout_baseline_hits.items()
            },
        },
        "controls": controls,
        "featureSurvival": _feature_survival(selected_results),
    }
    torch, _nn = _torch_modules()
    deterministic = bool(
        len(set(config.seeds)) == len(config.seeds)
        and len(config.seeds) > 0
        and torch.are_deterministic_algorithms_enabled()
    )
    finite_before_promotion = _all_finite(report)
    report["promotion"] = build_promotion(
        active_fraction=report["selectedSparsity"]["activeFraction"],
        ticket_payload=holdout_ticket,
        dense_payload=holdout_dense,
        ticket_hits=holdout_ticket_hits,
        dense_hits=holdout_dense_hits,
        baseline_hits=holdout_baseline_hits,
        deterministic=deterministic,
        finite=finite_before_promotion,
    )
    if artifact_path is not None:
        write_runtime_artifact(
            artifact_path,
            draws=draws.draws,
            split=split,
            standardizer=standardizer,
            config=config,
            selected_round=selected_round,
            selected_results=selected_results,
            promotion_passed=bool(report["promotion"]["passed"]),
        )
    return report


def _at_least_fifty(value: str) -> int:
    parsed = int(value)
    if parsed < 50:
        raise argparse.ArgumentTypeError("value must be at least 50")
    return parsed


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--warmup-draws", type=_non_negative, default=100)
    parser.add_argument("--validation-draws", type=_at_least_fifty, default=150)
    parser.add_argument("--holdout-draws", type=_at_least_fifty, default=250)
    options = parser.parse_args()
    config = BenchmarkConfig(
        warmup_draws=options.warmup_draws,
        validation_draws=options.validation_draws,
        holdout_draws=options.holdout_draws,
    )
    try:
        report = benchmark(
            options.dataset,
            config=config,
            artifact_path=options.artifact,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
