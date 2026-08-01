"""Load and evaluate the frozen experimental Sparse Neural Ticket ensemble."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

_ARTIFACT_NAME = "sparse_neural_ticket_v1.npz"
_EXPECTED_FEATURE_COUNT = 36
_EXPECTED_SEED_COUNT = 5
_LAYER_SHAPES = (
    ((32, _EXPECTED_FEATURE_COUNT), (32,)),
    ((16, 32), (16,)),
    ((1, 16), (1,)),
)


@dataclass(frozen=True, slots=True)
class SparseNeuralTicketArtifact:
    """Store a validated, NumPy-ready frozen ticket ensemble."""

    activation_reference_draw: int
    prefix_fingerprint: str
    active_weights: int
    total_weights: int
    selected_round: int
    promotion_passed: bool
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    models: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]

    @property
    def active_fraction(self) -> float:
        return self.active_weights / self.total_weights

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return mean candidate probabilities across the frozen seed models."""

        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != _EXPECTED_FEATURE_COUNT:
            raise ValueError("Sparse Neural Ticket requires 36 candidate features")
        if not np.isfinite(values).all():
            raise ValueError("Sparse Neural Ticket features must be finite")
        standardized = (values - self.feature_mean) / self.feature_scale
        probabilities: list[np.ndarray] = []
        for layers in self.models:
            hidden = standardized
            for weight, bias in layers[:-1]:
                hidden = np.maximum(hidden @ weight.T + bias, 0.0)
            output_weight, output_bias = layers[-1]
            logits = (hidden @ output_weight.T + output_bias).reshape(-1)
            probabilities.append(1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35))))
        return np.mean(np.stack(probabilities), axis=0)


def history_fingerprint(
    history: Sequence[Sequence[str | int | None]],
) -> str:
    """Return the stable prefix digest stored in the research artifact."""

    encoded = json.dumps(history, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array(archive: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(archive[name], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Invalid Sparse Neural Ticket artifact array: {name}")
    value.setflags(write=False)
    return value


@lru_cache(maxsize=1)
def load_sparse_neural_ticket() -> SparseNeuralTicketArtifact:
    """Load the packaged ticket artifact and validate its contract."""

    artifact = resources.files("rand_ai").joinpath(_ARTIFACT_NAME)
    with resources.as_file(artifact) as artifact_path, np.load(
        artifact_path,
        allow_pickle=False,
    ) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("schemaVersion") != 1:
            raise ValueError("Unsupported Sparse Neural Ticket artifact schema")
        seeds = tuple(int(seed) for seed in metadata["seeds"])
        if len(seeds) != _EXPECTED_SEED_COUNT or len(set(seeds)) != len(seeds):
            raise ValueError("Sparse Neural Ticket artifact must contain five seeds")
        feature_mean = _array(
            archive,
            "feature_mean",
            (_EXPECTED_FEATURE_COUNT,),
        )
        feature_scale = _array(
            archive,
            "feature_scale",
            (_EXPECTED_FEATURE_COUNT,),
        )
        if np.any(feature_scale <= 0):
            raise ValueError("Sparse Neural Ticket feature scales must be positive")
        models: list[tuple[tuple[np.ndarray, np.ndarray], ...]] = []
        for seed in seeds:
            layers: list[tuple[np.ndarray, np.ndarray]] = []
            for layer_index, (weight_shape, bias_shape) in enumerate(_LAYER_SHAPES):
                layers.append(
                    (
                        _array(
                            archive,
                            f"seed_{seed}_layer_{layer_index}_weight",
                            weight_shape,
                        ),
                        _array(
                            archive,
                            f"seed_{seed}_layer_{layer_index}_bias",
                            bias_shape,
                        ),
                    )
                )
            models.append(tuple(layers))

    prefix_fingerprint = str(metadata["prefixFingerprint"])
    if len(prefix_fingerprint) != 64:
        raise ValueError("Invalid Sparse Neural Ticket prefix fingerprint")
    return SparseNeuralTicketArtifact(
        activation_reference_draw=int(metadata["activationReferenceDraw"]),
        prefix_fingerprint=prefix_fingerprint,
        active_weights=int(metadata["activeWeights"]),
        total_weights=int(metadata["totalWeights"]),
        selected_round=int(metadata["selectedRound"]),
        promotion_passed=bool(metadata["promotionPassed"]),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        models=tuple(models),
    )
