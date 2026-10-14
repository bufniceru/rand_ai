"""Bounded, causally labelled XGBoost candidate classifier."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

MIN_TRANSITIONS = 120
TRAINING_WINDOW = 500
RETRAIN_INTERVAL = 10
PRIOR_STRENGTH = 24
BASE_PROBABILITY = 6 / 49


class XGBoostClassifierModel:
    """Keep completed transitions separate from the next pending forecast."""

    def __init__(self) -> None:
        # Loading native libraries is deliberately restricted to enabled strategies.
        from xgboost import XGBClassifier

        self.estimator_type = XGBClassifier
        self.model: Any = None
        self.transitions: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=TRAINING_WINDOW
        )
        self.pending_features: np.ndarray | None = None
        self.completed_transitions = 0
        self.last_retraining_transition = 0

    def observe(self, drawn: set[int]) -> None:
        """Label the previous forecast only after its target has arrived."""
        if self.pending_features is None:
            return
        labels = np.asarray([int(n in drawn) for n in range(1, 50)], dtype=np.int32)
        self.transitions.append((self.pending_features, labels))
        self.pending_features = None
        self.completed_transitions += 1
        if self.completed_transitions < MIN_TRANSITIONS:
            return
        if (
            self.model is not None
            and self.completed_transitions - self.last_retraining_transition
            < RETRAIN_INTERVAL
        ):
            return
        model = self.estimator_type(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=10,
            reg_lambda=10,
            reg_alpha=0,
            subsample=1,
            colsample_bytree=1,
            scale_pos_weight=1,
            base_score=BASE_PROBABILITY,
            random_state=20260626,
        )
        model.fit(
            np.concatenate([features for features, _ in self.transitions]),
            np.concatenate([target for _, target in self.transitions]),
        )
        self.model = model
        self.last_retraining_transition = self.completed_transitions

    def predict(
        self, features: np.ndarray, appearances: list[int], draw_count: int
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Stage an immutable feature snapshot and rank marginal hit estimates."""
        self.pending_features = features.copy()
        if self.model is None:
            probabilities = np.asarray(
                [
                    (appearances[n] + PRIOR_STRENGTH * BASE_PROBABILITY)
                    / (draw_count + PRIOR_STRENGTH)
                    for n in range(1, 50)
                ]
            )
            status = f"Warm-up {self.completed_transitions}/{MIN_TRANSITIONS}; smoothed-frequency fallback"
        else:
            probabilities = np.asarray(self.model.predict_proba(features))[:, 1]
            status = "Fitted XGBoost; uncalibrated marginal estimates"
        if probabilities.shape != (49,) or not np.isfinite(probabilities).all():
            raise ValueError("XGBoost must return 49 finite candidate scores")
        scores = {n: float(probabilities[n - 1]) for n in range(1, 50)}
        details: dict[int, tuple[str, ...]] = {
            n: (
                f"Estimated hit score {scores[n]:.2%}",
                status,
                f"Completed transitions {self.completed_transitions}",
                f"Training window {len(self.transitions)}/{TRAINING_WINDOW}",
                f"Last retraining transition {self.last_retraining_transition}",
            )
            for n in range(1, 50)
        }
        return scores, details
