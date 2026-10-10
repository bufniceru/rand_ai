"""Extend positional successors with joint center/tail profiles and confidence."""

import numpy as np

from rand_ai.positional_shape_successor import CENTERS, PositionalShapeSuccessorModel

CENTRAL_BAND = 0.5
PROFILE_WEIGHT = 0.5
CONFIDENCE_SUPPORT = 8.0


def center_tail_profile(deviations: np.ndarray) -> np.ndarray:
    """Classify standardized deviations as below (-1), central (0), or above (1).

    The central band includes both boundaries at half a historical population SD.
    Constant features are masked out by the caller rather than treated as evidence.
    """
    return np.where(
        deviations < -CENTRAL_BAND, -1, np.where(deviations > CENTRAL_BAND, 1, 0)
    )


def shape_distances(
    features: np.ndarray, *, use_profile: bool = True
) -> tuple[np.ndarray, tuple[int, int]]:
    """Compare eligible source draws with the latest observed draw.

    Each family contributes half of the distance. Within a family the profile
    distance averages label mismatch and squared difference in the central share.
    Continuous and profile distances have equal influence when profiles are enabled.
    """
    spreads = np.std(features, axis=0, ddof=0)
    # Constant fractional averages can acquire roundoff-sized nonzero SDs.
    active = (spreads > 0) & (np.ptp(features, axis=0) > 0)
    deviations = np.divide(
        features - CENTERS, spreads, out=np.zeros_like(features), where=active
    )
    profiles = center_tail_profile(deviations)
    squared = (deviations[:-1] - deviations[-1]) ** 2
    distances = np.zeros(len(features) - 1)
    central_counts = []
    for family in (slice(0, 6), slice(6, 21)):
        mask = active[family]
        latest = profiles[-1, family][mask]
        central_counts.append(int(np.count_nonzero(latest == 0)))
        if not np.any(mask):
            continue
        continuous = np.mean(squared[:, family][:, mask], axis=1)
        if use_profile:
            historical = profiles[:-1, family][:, mask]
            mismatch = np.mean(historical != latest, axis=1)
            central_share = np.mean(historical == 0, axis=1)
            share_difference = (central_share - np.mean(latest == 0)) ** 2
            profile_distance = 0.5 * (mismatch + share_difference)
            distances += 0.5 * (
                (1 - PROFILE_WEIGHT) * continuous + PROFILE_WEIGHT * profile_distance
            )
        else:
            distances += 0.5 * continuous
    return distances, (central_counts[0], central_counts[1])


class PositionalShapeSuccessorV2Model(PositionalShapeSuccessorModel):
    """Forecast from joint center/tail analogues with similarity-based shrinkage."""

    def __init__(
        self, *, use_profile: bool = True, use_confidence: bool = True
    ) -> None:
        """Initialize V2; switches support controlled evaluation of each addition."""
        super().__init__()
        self.use_profile = use_profile
        self.use_confidence = use_confidence

    def predict(self) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Return successor scores shrunk toward uniform when evidence is weak.

        Confidence is mean neighbor similarity times effective neighbor support
        divided by support plus eight. It changes score strength, not score order,
        because every number shares the same confidence and uniform baseline.
        """
        support = np.zeros(49)
        counts = np.zeros(49, dtype=int)
        neighbors = 0
        total_weight = 0.0
        squared_weight = 0.0
        central_counts = (0, 0)
        if len(self.features) >= 2:
            distances, central_counts = shape_distances(
                np.stack(self.features), use_profile=self.use_profile
            )
            indices = np.argsort(distances, kind="stable")[:32]
            neighbors = len(indices)
            for index in indices:
                weight = 1 / (1 + float(distances[index]))
                total_weight += weight
                squared_weight += weight * weight
                for number in self.history[index + 1]:
                    support[number - 1] += weight
                    counts[number - 1] += 1
        effective_neighbors = (
            total_weight * total_weight / squared_weight if squared_weight else 0.0
        )
        similarity = total_weight / neighbors if neighbors else 0.0
        confidence = (
            similarity
            * effective_neighbors
            / (effective_neighbors + CONFIDENCE_SUPPORT)
        )
        blend = confidence if self.use_confidence else 1.0
        baseline = 6 / 49
        scores = {
            number: float(
                baseline
                + blend
                * ((support[number - 1] + 8 * baseline) / (total_weight + 8) - baseline)
            )
            for number in range(1, 50)
        }
        details: dict[int, tuple[str, ...]] = {
            number: (
                f"Successor support {counts[number - 1]} of {neighbors} neighbors",
                f"Weighted support {support[number - 1]:.4f} / {total_weight:.4f}; prior 8",
                f"Central active features: {central_counts[0]} positions, "
                f"{central_counts[1]} groups; band +/-{CENTRAL_BAND:g} SD",
                f"Similarity {similarity:.1%}; effective neighbors {effective_neighbors:.2f}; "
                f"confidence {confidence:.1%}",
                "V2 center/tail successor model score; not a calibrated probability",
            )
            for number in range(1, 50)
        }
        return scores, details
