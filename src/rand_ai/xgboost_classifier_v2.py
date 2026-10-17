"""Causal XGBoost configuration selection with an independent evidence gate."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

MIN_TRANSITIONS = 240
TRAINING_WINDOW = 500
RESELECT_INTERVAL = 40
REFIT_INTERVAL = 10
VALIDATION_BLOCK = 40
GATE_WINDOW = 120
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK = 10
SEED = 20260626
BASE_PROBABILITY = 6 / 49


@dataclass(frozen=True, order=True)
class Candidate:
    """Order also defines the deterministic simplicity tie-break."""

    feature_count: int
    depth: int
    window: int

    @property
    def identifier(self) -> str:
        return f"f{self.feature_count}-d{self.depth}-w{self.window}"


CANDIDATES = tuple(
    Candidate(*values) for values in product((4, 14), (1, 2), (250, 500))
)


def estimator_parameters(candidate: Candidate) -> dict[str, Any]:
    """Freeze all non-search parameters to the V1 values."""
    return dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        n_jobs=1,
        n_estimators=100,
        max_depth=candidate.depth,
        learning_rate=0.05,
        min_child_weight=10,
        reg_lambda=10,
        reg_alpha=0,
        subsample=1,
        colsample_bytree=1,
        scale_pos_weight=1,
        base_score=BASE_PROBABILITY,
        random_state=SEED,
    )


def create_estimator(candidate: Candidate) -> Any:
    """Load the native dependency only when fitting an enabled strategy."""
    from xgboost import XGBClassifier

    return XGBClassifier(**estimator_parameters(candidate))


def probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """Validate unmodified marginal scores before display or evidence use."""
    values = np.asarray(model.predict_proba(features), dtype=float)[:, 1]
    if (
        values.shape != (len(features),)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise ValueError("XGBoost V2 must return 49 finite probabilities in [0, 1]")
    return values


def forecast_metrics(
    labels: np.ndarray, values: np.ndarray
) -> tuple[float, float, int]:
    """Return draw-level log loss, Brier score, and number-ascending Top-6 hits."""
    clipped = np.clip(values, 1e-7, 1 - 1e-7)
    loss = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    top = np.lexsort((np.arange(49), -values))[:6]
    return float(loss), float(np.mean((values - labels) ** 2)), int(labels[top].sum())


def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    """Paired non-circular moving-block interval with a fixed random seed."""
    if not len(values):
        raise ValueError("A bootstrap interval needs completed forecasts")
    length = min(BOOTSTRAP_BLOCK, len(values))
    starts = np.random.default_rng(SEED).integers(
        0,
        len(values) - length + 1,
        size=(BOOTSTRAP_REPLICATES, (len(values) + length - 1) // length),
    )
    indices = (starts[:, :, None] + np.arange(length)).reshape(
        BOOTSTRAP_REPLICATES, -1
    )[:, : len(values)]
    limits = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return float(limits[0]), float(limits[1])


class XGBoostV2Model:
    """Select on old folds, gate on later recorded forecasts, and fail closed."""

    def __init__(self) -> None:
        self.estimator_factory = create_estimator
        self.transitions: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=TRAINING_WINDOW
        )
        self.evidence: deque[tuple[float, int]] = deque(maxlen=GATE_WINDOW)
        self.pending: tuple[np.ndarray, np.ndarray | None, np.ndarray] | None = None
        self.completed_transitions = 0
        self.last_selection_transition = 0
        self.last_fit_transition = 0
        self.last_gate_transition = 0
        self.selected: Candidate | None = None
        self.model: Any = None
        self.validation_loss: float | None = None
        self.training_loss: float | None = None
        self.selection_results: list[dict[str, Any]] = []
        self.gate_active = False
        self.gate_interval: tuple[float, float] | None = None
        self.gate_hit_advantage: float | None = None
        self.raw_probabilities: np.ndarray | None = None
        self.baseline_probabilities: np.ndarray | None = None
        self.display_source = "smoothed_frequency"

    def _fit(
        self, candidate: Candidate, transitions: list[tuple[np.ndarray, np.ndarray]]
    ) -> Any:
        rows = transitions[-candidate.window :]
        model = self.estimator_factory(candidate)
        model.fit(
            np.concatenate([x[:, : candidate.feature_count] for x, _ in rows]),
            np.concatenate([y for _, y in rows]),
        )
        return model

    def _loss(
        self,
        model: Any,
        candidate: Candidate,
        rows: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        features = np.concatenate([x[:, : candidate.feature_count] for x, _ in rows])
        labels = np.concatenate([y for _, y in rows])
        values = np.clip(probabilities(model, features), 1e-7, 1 - 1e-7)
        return float(
            -np.mean(labels * np.log(values) + (1 - labels) * np.log1p(-values))
        )

    def _select(self) -> None:
        history = list(self.transitions)
        results: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            validation_losses = []
            training_losses = []
            for start in range(len(history) - 120, len(history), VALIDATION_BLOCK):
                training = history[max(0, start - candidate.window) : start]
                model = self._fit(candidate, training)
                training_losses.append(self._loss(model, candidate, training))
                validation_losses.append(
                    self._loss(
                        model, candidate, history[start : start + VALIDATION_BLOCK]
                    )
                )
            results.append(
                dict(
                    candidate=asdict(candidate),
                    identifier=candidate.identifier,
                    validation_loss=float(np.mean(validation_losses)),
                    training_loss=float(np.mean(training_losses)),
                    fold_validation_losses=validation_losses,
                )
            )
        best = min(
            range(len(CANDIDATES)),
            key=lambda i: (results[i]["validation_loss"], CANDIDATES[i]),
        )
        self.selected = CANDIDATES[best]
        self.validation_loss = results[best]["validation_loss"]
        self.selection_results = results
        self.last_selection_transition = self.completed_transitions

    def _evaluate_gate(self) -> None:
        self.last_gate_transition = self.completed_transitions
        if len(self.evidence) < GATE_WINDOW:
            self.gate_active = False
            return
        values = np.asarray(self.evidence, dtype=float)
        self.gate_interval = bootstrap_interval(values[:, 0])
        self.gate_hit_advantage = float(values[:, 1].mean())
        self.gate_active = self.gate_interval[0] > 0 and self.gate_hit_advantage > 0

    def observe(self, drawn: set[int]) -> None:
        """Consume one pending snapshot and its previously unknown target."""
        if self.pending is None:
            return
        features, raw, baseline = self.pending
        self.pending = None
        labels = np.asarray([int(n in drawn) for n in range(1, 50)], dtype=np.int32)
        self.transitions.append((features, labels))
        self.completed_transitions += 1
        if raw is not None:
            raw_loss, _, raw_hits = forecast_metrics(labels, raw)
            baseline_loss, _, baseline_hits = forecast_metrics(labels, baseline)
            self.evidence.append((baseline_loss - raw_loss, raw_hits - baseline_hits))
        if self.completed_transitions < MIN_TRANSITIONS:
            return
        if (
            self.completed_transitions - self.last_selection_transition
            >= RESELECT_INTERVAL
        ):
            self._select()
        if self.completed_transitions - self.last_fit_transition >= REFIT_INTERVAL:
            # Selection always runs before the first fit at transition 240.
            candidate = self.selected
            assert candidate is not None
            rows = list(self.transitions)[-candidate.window :]
            self.model = self._fit(candidate, rows)
            self.training_loss = self._loss(self.model, candidate, rows)
            self.last_fit_transition = self.completed_transitions
        if self.completed_transitions % REFIT_INTERVAL == 0:
            self._evaluate_gate()

    def predict(
        self, features: np.ndarray, appearances: list[int], draw_count: int
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Record raw scores even while the evidence gate displays fallback."""
        baseline = np.asarray(
            [
                (appearances[n] + 24 * BASE_PROBABILITY) / (draw_count + 24)
                for n in range(1, 50)
            ]
        )
        raw = None
        if self.model is not None:
            assert self.selected is not None
            raw = probabilities(self.model, features[:, : self.selected.feature_count])
        self.pending = (
            features.copy(),
            None if raw is None else raw.copy(),
            baseline.copy(),
        )
        self.raw_probabilities = raw
        self.baseline_probabilities = baseline
        self.display_source = (
            "xgboost" if self.gate_active and raw is not None else "smoothed_frequency"
        )
        values = raw if self.gate_active and raw is not None else baseline
        config = "not selected" if self.selected is None else self.selected.identifier
        diagnostics = (
            f"Source {self.display_source}; conservative evidence gate {'passed' if self.gate_active else 'not passed'}",
            f"Configuration {config}; completed transitions {self.completed_transitions}",
            f"Training log loss {self.training_loss}; validation log loss {self.validation_loss}",
            f"Gate forecasts {len(self.evidence)}/{GATE_WINDOW}; loss-advantage 95% interval {self.gate_interval}",
            f"Gate mean hit advantage {self.gate_hit_advantage}; last gate transition {self.last_gate_transition}",
            f"Last selection {self.last_selection_transition}; last fit {self.last_fit_transition}; buffer {len(self.transitions)}/{TRAINING_WINDOW}",
            "Experimental adaptive policy; repeated historical gate tests do not establish future advantage",
        )
        scores = {n: float(values[n - 1]) for n in range(1, 50)}
        details: dict[int, tuple[str, ...]] = {
            n: (f"Estimated marginal score {scores[n]:.2%}", *diagnostics)
            for n in range(1, 50)
        }
        return scores, details
