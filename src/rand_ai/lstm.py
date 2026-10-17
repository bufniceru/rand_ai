"""Validated NumPy-only inference for experimental frozen LSTM ensembles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from importlib import resources

import numpy as np

from rand_ai.draw import Draw

WINDOW = 7
HIDDEN = 32
SEEDS = tuple(range(20260626, 20260631))
SHAPES = {
    "lstm.weight_ih_l0": (128, 49),
    "lstm.weight_hh_l0": (128, 32),
    "lstm.bias_ih_l0": (128,),
    "lstm.bias_hh_l0": (128,),
    "head.weight": (49, 32),
    "head.bias": (49,),
}


def encode_draws(draws: Sequence[Draw]) -> np.ndarray:
    """Encode completed, validated six-number draws without scaling."""
    values = np.zeros((len(draws), 49), dtype=np.float64)
    for index, draw in enumerate(draws):
        values[index, [ball.value - 1 for ball in draw.balls]] = 1
    return values


def prefix_fingerprint(draws: Sequence[Draw]) -> str:
    """Bind model eligibility to exact dated training/validation history."""
    rows = [[draw.date, *[ball.value for ball in draw.balls]] for draw in draws]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Stable sigmoid without clipping useful finite logits."""
    return np.exp(-np.logaddexp(0, -values))


def model_probabilities(
    weights: dict[str, np.ndarray], windows: np.ndarray
) -> np.ndarray:
    """Evaluate independent windows with zero initial hidden/cell states."""
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (WINDOW, 49) or not np.isfinite(x).all():
        raise ValueError("LSTM requires finite windows shaped (batch, 7, 49)")
    hidden = np.zeros((len(x), HIDDEN))
    cell = np.zeros_like(hidden)
    for step in range(WINDOW):
        gates = (
            x[:, step] @ weights["lstm.weight_ih_l0"].T
            + hidden @ weights["lstm.weight_hh_l0"].T
            + weights["lstm.bias_ih_l0"]
            + weights["lstm.bias_hh_l0"]
        )
        input_gate, forget, candidate, output = np.split(gates, 4, axis=1)
        cell = sigmoid(forget) * cell + sigmoid(input_gate) * np.tanh(candidate)
        hidden = sigmoid(output) * np.tanh(cell)
    return sigmoid(hidden @ weights["head.weight"].T + weights["head.bias"])


def top_six(probabilities: np.ndarray) -> np.ndarray:
    """Select six distinct numbers; equal scores prefer smaller numbers."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 49:
        raise ValueError("Expected probability rows with 49 candidates")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be finite and between zero and one")
    return np.argsort(-values, axis=1, kind="stable")[:, :6] + 1


@dataclass(frozen=True)
class LstmArtifact:
    """A frozen research model, eligible only after its fitted prefix."""

    activation_reference_draw: int
    fingerprint: str
    historical_gate_passed: bool
    models: tuple[dict[str, np.ndarray], ...]

    def predict(self, history: Sequence[Draw]) -> np.ndarray:
        if len(history) < self.activation_reference_draw:
            raise ValueError("History precedes LSTM activation reference")
        if (
            prefix_fingerprint(history[: self.activation_reference_draw])
            != self.fingerprint
        ):
            raise ValueError("History does not match LSTM fitted prefix")
        window = encode_draws(history[-WINDOW:])[None]
        return np.mean(
            [model_probabilities(model, window)[0] for model in self.models], axis=0
        )


def load_lstm_artifact(path: str | Path) -> LstmArtifact:
    """Load numeric weights only; reject corrupt or incompatible artifacts."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if (
            metadata.get("schemaVersion") != 1
            or metadata.get("model") != "lstm-7x49-h32"
        ):
            raise ValueError("Unsupported LSTM artifact schema/model")
        if metadata.get("seeds") != list(SEEDS):
            raise ValueError("LSTM artifact requires the five fixed research seeds")
        activation = metadata.get("activationReferenceDraw")
        fingerprint = metadata.get("prefixFingerprint")
        gate = metadata.get("historicalGatePassed")
        if type(activation) is not int or activation < WINDOW:
            raise ValueError("Invalid LSTM activation reference")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Invalid LSTM prefix fingerprint")
        try:
            int(fingerprint, 16)
        except ValueError as exc:
            raise ValueError("Invalid LSTM prefix fingerprint") from exc
        if type(gate) is not bool:
            raise ValueError("Invalid LSTM historical gate status")
        models = []
        for index in range(len(SEEDS)):
            weights = {}
            for name, shape in SHAPES.items():
                values = np.asarray(archive[f"seed{index}/{name}"], dtype=np.float64)
                if values.shape != shape or not np.isfinite(values).all():
                    raise ValueError(f"Invalid LSTM weight {name}")
                values.setflags(write=False)
                weights[name] = values
            models.append(weights)
    return LstmArtifact(activation, fingerprint, gate, tuple(models))


@lru_cache(maxsize=1)
def load_packaged_lstm() -> LstmArtifact:
    """Load the desktop's explicitly frozen experimental artifact without torch."""
    artifact = resources.files("rand_ai").joinpath("lstm_v1.npz")
    with resources.as_file(artifact) as path:
        return load_lstm_artifact(path)
