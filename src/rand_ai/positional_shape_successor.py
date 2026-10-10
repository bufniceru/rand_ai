"""Rank numbers from successors of similar joint positional draw shapes."""

from collections.abc import Collection

import numpy as np

GROUPS = tuple((start, size) for size in range(2, 7) for start in range(1, 8 - size))
CENTERS = np.array(
    [position * 50 / 7 for position in range(1, 7)]
    + [25 * (2 * start + size - 1) / 7 for start, size in GROUPS]
)


def shape_features(numbers: Collection[int]) -> np.ndarray:
    """Return six sorted values and fifteen adjacent-group averages.

    Args:
        numbers: Six distinct integer balls from 1 through 49.

    Returns:
        The 21 positional and group features in statistics-report order.
    """
    ordered = sorted(numbers)
    if (
        len(ordered) != 6
        or len(set(ordered)) != 6
        or any(type(number) is not int or not 1 <= number <= 49 for number in ordered)
    ):
        raise ValueError("Expected six distinct integers from 1 to 49")
    return np.array(
        ordered
        + [sum(ordered[start - 1 : start - 1 + size]) / size for start, size in GROUPS],
        dtype=float,
    )


class PositionalShapeSuccessorModel:
    """Maintain chronological history and forecast from known successors."""

    def __init__(self) -> None:
        """Initialize empty draw and feature histories."""
        self.history: list[frozenset[int]] = []
        self.features: list[np.ndarray] = []

    def observe(self, numbers: Collection[int]) -> None:
        """Append one validated draw after its outcome is available."""
        features = shape_features(numbers)
        self.history.append(frozenset(numbers))
        self.features.append(features)

    def predict(self) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Return smoothed successor-support scores and explanatory details.

        Only draws preceding the latest draw have known successors. Population
        spreads are recomputed from observed history; constant features are ignored.
        """
        support = np.zeros(49)
        counts = np.zeros(49, dtype=int)
        neighbors = 0
        total_weight = 0.0
        if len(self.features) >= 2:
            features = np.stack(self.features)
            spreads = np.std(features, axis=0, ddof=0)
            active = spreads > 0
            deviations = np.divide(
                features - CENTERS,
                spreads,
                out=np.zeros_like(features),
                where=active,
            )
            differences = (deviations[:-1] - deviations[-1]) ** 2
            distances = np.zeros(len(features) - 1)
            for family in (slice(0, 6), slice(6, 21)):
                mask = active[family]
                if np.any(mask):
                    distances += 0.5 * np.mean(differences[:, family][:, mask], axis=1)
            indices = np.argsort(distances, kind="stable")[:32]
            neighbors = len(indices)
            for index in indices:
                weight = 1 / (1 + float(distances[index]))
                total_weight += weight
                for number in self.history[index + 1]:
                    support[number - 1] += weight
                    counts[number - 1] += 1
        scores = {
            number: float((support[number - 1] + 8 * 6 / 49) / (total_weight + 8))
            for number in range(1, 50)
        }
        details: dict[int, tuple[str, ...]] = {
            number: (
                f"Successor support {counts[number - 1]} of {neighbors} neighbors",
                f"Weighted support {support[number - 1]:.4f} / {total_weight:.4f}; prior 8",
                "Joint positional/group shape model score; not a calibrated probability",
            )
            for number in range(1, 50)
        }
        return scores, details
