"""Champion-preserving collective-intelligence ranking."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

NUMBER_COUNT = 49
NUMBERS_PER_DRAW = 6
EXPECTED_RANDOM_HITS = NUMBERS_PER_DRAW * NUMBERS_PER_DRAW / NUMBER_COUNT
EXPERT_PRIOR_DRAWS = 24.0
CORRELATION_WINDOW = 40
SHADOW_WINDOW = 120
SHADOW_MINIMUM_DRAWS = 80
SHADOW_ACTIVATION_GAIN = 3

CIS_V3_EXPERT_IDS = (
    "freshness",
    "proximity",
    "emd",
    "chi_square",
    "entropy",
    "markov100",
    "mkfr",
    "mksp",
    "mknp",
    "bayesian",
    "co_occurrence",
    "doublet_triplet_markov",
    "svc",
    "tbl",
)


@dataclass(frozen=True, slots=True, order=True)
class CisConfig:
    """Hyperparameters selected on the development period only."""

    recent_window: int
    recent_weight: float
    correlation_threshold: float
    minimum_support: float
    minimum_gain: float
    maximum_replacements: int


@dataclass(frozen=True, slots=True)
class CisCorrection:
    """One proposed replacement of a champion Top-6 number."""

    removed: int
    added: int
    gain: float
    support: float


@dataclass(frozen=True, slots=True)
class CisResult:
    """One champion and its evidence-gated corrected ranking."""

    ranking: tuple[int, ...]
    champion_id: str
    champion_quality: float
    champion_top: tuple[int, ...]
    shadow_top: tuple[int, ...]
    corrections: tuple[CisCorrection, ...]
    corrections_active: bool
    peer_scores: Mapping[int, float]

    @property
    def top_numbers(self) -> tuple[int, ...]:
        return self.ranking[:NUMBERS_PER_DRAW]

    @property
    def applied_correction_count(self) -> int:
        return len(self.corrections) if self.corrections_active else 0


@dataclass(slots=True)
class _PendingPrediction:
    rankings: dict[str, tuple[int, ...]]
    champion_top: frozenset[int]
    shadow_top: frozenset[int]
    has_shadow_correction: bool


def cis_configurations() -> tuple[CisConfig, ...]:
    """Return the fixed, deterministic development-search grid."""
    return tuple(
        CisConfig(window, weight, correlation, support, gain, replacements)
        for window in (20, 40, 80)
        for weight in (0.6, 0.8)
        for correlation in (0.8, 0.9)
        for support in (0.2, 0.3)
        for gain in (0.05, 0.10)
        for replacements in (1, 2)
    )


def _rank_map(ranking: Sequence[int]) -> dict[int, int]:
    return {number: rank for rank, number in enumerate(ranking, start=1)}


def spearman_rank_correlation(
    left: Sequence[int],
    right: Sequence[int],
) -> float:
    """Calculate Spearman correlation between two complete number rankings."""
    left_ranks = _rank_map(left)
    right_ranks = _rank_map(right)
    squared_difference = sum(
        (left_ranks[number] - right_ranks[number]) ** 2
        for number in range(1, NUMBER_COUNT + 1)
    )
    return 1 - 6 * squared_difference / (
        NUMBER_COUNT * (NUMBER_COUNT * NUMBER_COUNT - 1)
    )


def ranking_correlations(
    rankings: Mapping[str, Sequence[int]],
) -> dict[tuple[str, str], float]:
    """Calculate canonical pair correlations for one prediction frame."""
    available = [
        strategy_id for strategy_id in CIS_V3_EXPERT_IDS if strategy_id in rankings
    ]
    return {
        (left, right): spearman_rank_correlation(rankings[left], rankings[right])
        for left_index, left in enumerate(available)
        for right in available[left_index + 1 :]
    }


class ChampionCis:
    """Preserve a proven expert and activate only proven peer corrections."""

    def __init__(self, config: CisConfig) -> None:
        self.config = config
        self.total_hits = {strategy_id: 0 for strategy_id in CIS_V3_EXPERT_IDS}
        self.evaluated_draws = {
            strategy_id: 0 for strategy_id in CIS_V3_EXPERT_IDS
        }
        self.recent_hits = {
            strategy_id: deque(maxlen=config.recent_window)
            for strategy_id in CIS_V3_EXPERT_IDS
        }
        self.correlations = {
            (left, right): deque(maxlen=CORRELATION_WINDOW)
            for left_index, left in enumerate(CIS_V3_EXPERT_IDS)
            for right in CIS_V3_EXPERT_IDS[left_index + 1 :]
        }
        self.shadow_deltas: deque[int] = deque(maxlen=SHADOW_WINDOW)
        self.corrections_active = False
        self.pending: _PendingPrediction | None = None

    @staticmethod
    def _smoothed_hits(hits: int, draws: int) -> float:
        return (hits + EXPERT_PRIOR_DRAWS * EXPECTED_RANDOM_HITS) / (
            draws + EXPERT_PRIOR_DRAWS
        )

    def expert_quality(self, strategy_id: str) -> float:
        recent = self.recent_hits[strategy_id]
        recent_rate = self._smoothed_hits(sum(recent), len(recent))
        lifetime_rate = self._smoothed_hits(
            self.total_hits[strategy_id],
            self.evaluated_draws[strategy_id],
        )
        return (
            self.config.recent_weight * recent_rate
            + (1 - self.config.recent_weight) * lifetime_rate
        )

    def _champion(self, rankings: Mapping[str, Sequence[int]]) -> str:
        available = [
            strategy_id
            for strategy_id in CIS_V3_EXPERT_IDS
            if strategy_id in rankings
        ]
        if not available:
            raise ValueError("CIS v3 requires at least one expert ranking")
        return max(
            available,
            key=lambda strategy_id: (
                self.expert_quality(strategy_id),
                -CIS_V3_EXPERT_IDS.index(strategy_id),
            ),
        )

    def _mean_correlation(self, left: str, right: str) -> float:
        left_index = CIS_V3_EXPERT_IDS.index(left)
        right_index = CIS_V3_EXPERT_IDS.index(right)
        pair = (left, right) if left_index < right_index else (right, left)
        history = self.correlations.get(pair)
        return 0.0 if not history else sum(history) / len(history)

    def _peer_weights(
        self,
        rankings: Mapping[str, Sequence[int]],
        champion_id: str,
    ) -> dict[str, float]:
        peers = [
            strategy_id
            for strategy_id in CIS_V3_EXPERT_IDS
            if strategy_id != champion_id and strategy_id in rankings
        ]
        weights: dict[str, float] = {}
        threshold = self.config.correlation_threshold
        for strategy_id in peers:
            redundancy = 1.0 + sum(
                max(
                    0.0,
                    (self._mean_correlation(strategy_id, other) - threshold)
                    / (1 - threshold),
                )
                for other in peers
                if other != strategy_id
            )
            uplift = max(
                self.expert_quality(strategy_id) - EXPECTED_RANDOM_HITS,
                0.0,
            )
            weights[strategy_id] = uplift / redundancy
        return weights

    def _peer_evidence(
        self,
        rankings: Mapping[str, Sequence[int]],
        champion_id: str,
    ) -> tuple[dict[int, float], dict[int, float]]:
        weights = self._peer_weights(rankings, champion_id)
        total_weight = sum(weights.values())
        if total_weight <= 0:
            neutral = {number: 0.0 for number in range(1, NUMBER_COUNT + 1)}
            return neutral, dict(neutral)
        rank_maps = {
            strategy_id: _rank_map(rankings[strategy_id]) for strategy_id in weights
        }
        strength: dict[int, float] = {}
        support: dict[int, float] = {}
        for number in range(1, NUMBER_COUNT + 1):
            strength[number] = sum(
                weight
                * (NUMBER_COUNT - rank_maps[strategy_id][number])
                / (NUMBER_COUNT - 1)
                for strategy_id, weight in weights.items()
            ) / total_weight
            support[number] = sum(
                weight
                for strategy_id, weight in weights.items()
                if rank_maps[strategy_id][number] <= NUMBERS_PER_DRAW
            ) / total_weight
        return (
            {
                number: 0.8 * strength[number] + 0.2 * support[number]
                for number in strength
            },
            support,
        )

    def _shadow_ticket(
        self,
        champion_ranking: Sequence[int],
        peer_scores: Mapping[int, float],
        peer_support: Mapping[int, float],
    ) -> tuple[tuple[int, ...], tuple[CisCorrection, ...]]:
        shadow = list(champion_ranking[:NUMBERS_PER_DRAW])
        original_champion = set(shadow)
        corrections: list[CisCorrection] = []
        for _step in range(self.config.maximum_replacements):
            removable = [
                number
                for number in shadow
                if number in original_champion
                and all(item.removed != number for item in corrections)
            ]
            outsiders = [
                number
                for number in champion_ranking[NUMBERS_PER_DRAW:]
                if number not in shadow
            ]
            if not removable or not outsiders:
                break
            removed = min(
                removable,
                key=lambda number: (peer_scores[number], -number),
            )
            added = max(
                outsiders,
                key=lambda number: (peer_scores[number], -number),
            )
            gain = peer_scores[added] - peer_scores[removed]
            if (
                peer_support[added] < self.config.minimum_support
                or gain < self.config.minimum_gain
            ):
                break
            position = shadow.index(removed)
            shadow[position] = added
            corrections.append(
                CisCorrection(
                    removed=removed,
                    added=added,
                    gain=gain,
                    support=peer_support[added],
                )
            )
        return tuple(shadow), tuple(corrections)

    def predict(self, rankings: Mapping[str, Sequence[int]]) -> CisResult:
        """Build a ranking using state learned strictly before this target draw."""
        normalized = {
            strategy_id: tuple(rankings[strategy_id])
            for strategy_id in CIS_V3_EXPERT_IDS
            if strategy_id in rankings
        }
        champion_id = self._champion(normalized)
        champion_ranking = normalized[champion_id]
        peer_scores, peer_support = self._peer_evidence(normalized, champion_id)
        shadow_top, corrections = self._shadow_ticket(
            champion_ranking,
            peer_scores,
            peer_support,
        )
        apply_corrections = self.corrections_active and bool(corrections)
        selected_top = shadow_top if apply_corrections else tuple(
            champion_ranking[:NUMBERS_PER_DRAW]
        )
        selected = set(selected_top)
        ranking = (*selected_top, *(n for n in champion_ranking if n not in selected))
        self.pending = _PendingPrediction(
            rankings=normalized,
            champion_top=frozenset(champion_ranking[:NUMBERS_PER_DRAW]),
            shadow_top=frozenset(shadow_top),
            has_shadow_correction=bool(corrections),
        )
        return CisResult(
            ranking=ranking,
            champion_id=champion_id,
            champion_quality=self.expert_quality(champion_id),
            champion_top=tuple(champion_ranking[:NUMBERS_PER_DRAW]),
            shadow_top=shadow_top,
            corrections=corrections,
            corrections_active=apply_corrections,
            peer_scores=peer_scores,
        )

    def observe(
        self,
        drawn: set[int],
        correlations: Mapping[tuple[str, str], float] | None = None,
    ) -> None:
        """Learn a completed target draw after its pending prediction."""
        if self.pending is None:
            return
        for strategy_id, ranking in self.pending.rankings.items():
            hits = len(drawn.intersection(ranking[:NUMBERS_PER_DRAW]))
            self.total_hits[strategy_id] += hits
            self.evaluated_draws[strategy_id] += 1
            self.recent_hits[strategy_id].append(hits)
        frame_correlations = (
            ranking_correlations(self.pending.rankings)
            if correlations is None
            else correlations
        )
        for pair, correlation in frame_correlations.items():
            history = self.correlations.get(pair)
            if history is not None:
                history.append(correlation)
        if self.pending.has_shadow_correction:
            champion_hits = len(drawn.intersection(self.pending.champion_top))
            shadow_hits = len(drawn.intersection(self.pending.shadow_top))
            self.shadow_deltas.append(shadow_hits - champion_hits)
            trailing_gain = sum(self.shadow_deltas)
            if (
                not self.corrections_active
                and len(self.shadow_deltas) >= SHADOW_MINIMUM_DRAWS
                and trailing_gain >= SHADOW_ACTIVATION_GAIN
            ):
                self.corrections_active = True
            elif self.corrections_active and trailing_gain <= 0:
                self.corrections_active = False
        self.pending = None
