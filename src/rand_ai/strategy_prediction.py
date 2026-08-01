"""Build display-ready PyLotto-inspired strategy prediction histories."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from heapq import nlargest
from itertools import combinations

import numpy as np
from sklearn.linear_model import SGDClassifier

from rand_ai.draw import Draw
from rand_ai.prediction import CombinedPrediction

_NUMBER_COUNT = 49
_NUMBERS_PER_DRAW = 6
_BASE_PROBABILITY = _NUMBERS_PER_DRAW / _NUMBER_COUNT
_EXPECTED_RANDOM_HITS_PER_DRAW = _NUMBERS_PER_DRAW * _NUMBERS_PER_DRAW / _NUMBER_COUNT
_MAX_GAP_BUCKET = 35
_MARKOV_PRIOR_STRENGTH = 8.0
_MARKOV_DECAY = 0.5 ** (1 / 500)
_BAYESIAN_GAP_PRIOR_STRENGTH = 1024.0
_BAYESIAN_GAP_RECENCY_HALF_LIFE = 1000
_BAYESIAN_GAP_DECAY = 0.5 ** (1 / _BAYESIAN_GAP_RECENCY_HALF_LIFE)
_BAYESIAN_RECENT_GAP_WEIGHT = 0.15
_BAYESIAN_NUMBER_PRIOR_STRENGTH = 64.0
_BAYESIAN_NUMBER_RECENCY_HALF_LIFE = 100
_BAYESIAN_NUMBER_DECAY = 0.5 ** (1 / _BAYESIAN_NUMBER_RECENCY_HALF_LIFE)
_BAYESIAN_RECENT_NUMBER_WEIGHT = 0.35
_BAYESIAN_LIFETIME_GAP_WEIGHT = (
    1 - _BAYESIAN_RECENT_GAP_WEIGHT - _BAYESIAN_RECENT_NUMBER_WEIGHT
)
_PREDICTIVE_GRID_EMD_WEIGHT = 0.30
_CO_OCCURRENCE_ADJUSTED_WEIGHT = 0.10
_CO_OCCURRENCE_PRIOR_STRENGTH = 4.0
_CO_OCCURRENCE_RECENT_WEIGHT = 0.10
_CO_OCCURRENCE_RECENT_WINDOW = 100
_DOUBLET_TRIPLET_MARKOV_PRIOR_STRENGTH = 8.0
_DOUBLET_TRIPLET_MARKOV_RECENT_WINDOW = 120
_MKFR_MAX_ORDER = 20
_MKFR_PRIOR_STRENGTH = 8.0
_MKFR_MIN_CONTEXT_SUPPORT = 8
_MKSP_MAX_ORDER = 20
_MKSP_PRIOR_STRENGTH = 8.0
_MKSP_MIN_CONTEXT_SUPPORT = 8
_MKSP_VALUE_COUNT = 44
_MKSP_ANALOGUE_LIMIT = 512
_MKSP_ANALOGUE_PRIOR_STRENGTH = 4.0
_MKSP_ANALOGUE_BLEND = 0.70
_MKSP_CONTEXT_DECAY = 0.86
_MKSP_RECENCY_HALF_LIFE = 800
_MKSP_SIMILARITY_SHARPNESS = 10.0
_MKSP_BEAM_WIDTH = 8
_MKNP_POSITION_COUNT = _NUMBERS_PER_DRAW - 1
_MKRD_SHAPE_WEIGHT = 0.50
_MKRD_COVERAGE_WEIGHT = 0.20
_MKRD_UNIFORMITY_WEIGHT = 0.10
_MKRD_ENTROPY_WEIGHT = 0.10
_MKRD_CENTER_WEIGHT = 0.10
_RANDOM_SEED = 20260626
_FRESH_RANDOM_SEED_OFFSET = 7919
_FRESH_RANDOM_INFLUENCE = 0.35
_CHAIN_EFFECTIVENESS_PRIOR_DRAWS = 24.0
_CHAIN_EFFECTIVENESS_MINIMUM = 0.50
_CHAIN_EFFECTIVENESS_MAXIMUM = 1.50
_CIS_MINIMUM_TRAINING_DRAWS = 72
_CIS_EXPERT_PRIOR_DRAWS = 24.0
_CIS_RECENT_WINDOW = 40
_CIS_LEARNER_MAX_BLEND = 0.15
_CIS_LEARNER_EVIDENCE_DRAWS = 24.0
_CIS_LEARNER_MIN_ADVANTAGE = 0.10
_SKLEARN_SVM_EXPERT_IDS = (
    "mksp",
    "doublet_triplet_markov",
    "bayesian",
    "tbl",
    "mknp",
    "emd",
)
_SKLEARN_SVM_RECENT_WINDOW = 40
_SKLEARN_SVM_EFFECTIVENESS_PRIOR_DRAWS = 24.0
_SKLEARN_SVM_FEATURE_COUNT = 32
_PROXIMITY_BUCKETS = ("paired", "tight", "near", "balanced", "wide", "isolated")
_EARTH_MOVER_BUCKETS = ("Overlap", "Near", "Close", "Middle", "Far", "Distant")
_PRIMES = {
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
}

PredictionProgress = Callable[[int, int], None]
_BASE_STRATEGY_IDS = (
    "proximity",
    "freshness",
    "emd",
    "randomness",
    "fresh_random",
    "chi_square",
    "entropy",
    "markov100",
    "mkfr",
    "mksp",
    "mknp",
    "mkrd",
    "bayesian",
    "predictive_grid",
    "co_occurrence",
    "doublet_triplet_markov",
    "mixed",
    "svc",
    "tbl",
    "sklearn_svm",
    "cis",
)
_CHAIN_EXPERT_IDS = (
    "freshness",
    "proximity",
    "emd",
    "chi_square",
    "entropy",
    "markov100",
    "mkfr",
    "mksp",
    "bayesian",
    "predictive_grid",
    "co_occurrence",
    "doublet_triplet_markov",
    "mixed",
    "svc",
    "tbl",
    "cis",
)
STRATEGY_IDS = (*_BASE_STRATEGY_IDS, "residual_coverage", "chained")
_CIS_EXPERTS = (
    ("freshness", "Freshness", 0.06),
    ("proximity", "Proximity", 0.05),
    ("emd", "EMD", 0.12),
    ("entropy", "Entropy", 0.08),
    ("markov100", "100 Markov", 0.05),
    ("mkfr", "Markov Frequency", 0.10),
    ("mksp", "Markov Spaces", 0.14),
    ("bayesian", "Bayesian", 0.07),
    ("predictive_grid", "Predictive Grid", 0.08),
    ("co_occurrence", "Co-occurrence", 0.08),
    ("mixed", "Mixed", 0.07),
    ("svc", "SVC", 0.05),
    ("tbl", "TBL", 0.05),
)
_STRATEGY_DEPENDENCIES = {
    "tbl": {"freshness", "proximity", "randomness"},
    "fresh_random": {"freshness", "randomness"},
    "mixed": {"freshness", "proximity", "emd", "bayesian"},
    "predictive_grid": {"emd", "markov100"},
    "cis": {
        "freshness",
        "proximity",
        "emd",
        "entropy",
        "bayesian",
        "markov100",
        "mkfr",
        "mksp",
        "predictive_grid",
        "co_occurrence",
        "mixed",
        "svc",
        "tbl",
    },
    "sklearn_svm": set(_SKLEARN_SVM_EXPERT_IDS),
    "residual_coverage": set(_BASE_STRATEGY_IDS).difference(
        {"mknp", "mkrd", "sklearn_svm"}
    ),
    "chained": set(_CHAIN_EXPERT_IDS),
}


@dataclass(frozen=True, slots=True)
class StrategyNumberPrediction:
    """Store one candidate's rank and score for a named strategy."""

    number: int
    rank: int
    score: float
    gap: int
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyEfficacy:
    """Compare walk-forward Top-6 hits with equal-size random selections."""

    evaluated_draws: int
    strategy_hits: int
    random_hits: int
    expected_random_hits: float
    average_hits_per_draw: float
    random_average_hits_per_draw: float
    hit_difference: int


@dataclass(frozen=True, slots=True)
class StrategyPrediction:
    """Store one named 49-number strategy ranking."""

    strategy_id: str
    name: str
    description: str
    numbers: tuple[StrategyNumberPrediction, ...]
    top_numbers: tuple[int, ...]
    efficacy: StrategyEfficacy | None = None


@dataclass(frozen=True, slots=True)
class StrategyEfficacyRecord:
    """Store one completed draw's hits for selectable-range comparisons."""

    reference_draw_number: int
    target_draw_number: int
    actual_numbers: tuple[int, ...]
    random_hits: int
    strategy_hits: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PredictionSuite:
    """Store all prediction strategies after one reference draw."""

    reference_draw_number: int
    target_draw_number: int
    actual_numbers: tuple[int, ...]
    strategies: tuple[StrategyPrediction, ...]


EfficacyRecordCallback = Callable[[StrategyEfficacyRecord], None]
PredictionSuiteCallback = Callable[[PredictionSuite], None]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-_clamp(value, -35, 35)))


def _scale_scores(scores: dict[int, float]) -> dict[int, float]:
    minimum = min(scores.values(), default=0.0)
    maximum = max(scores.values(), default=0.0)
    spread = maximum - minimum
    if spread <= 0:
        return {number: 0.0 for number in scores}
    return {number: (score - minimum) / spread for number, score in scores.items()}


def _strategy(
    strategy_id: str,
    name: str,
    description: str,
    scores: dict[int, float],
    gaps: dict[int, int],
    details: dict[int, tuple[str, ...]] | None = None,
) -> StrategyPrediction:
    ranked = _ranking_from_scores(scores, gaps)
    predictions = tuple(
        StrategyNumberPrediction(
            number=number,
            rank=rank,
            score=scores[number],
            gap=gaps[number],
            details=() if details is None else details.get(number, ()),
        )
        for rank, number in enumerate(ranked, start=1)
    )
    return StrategyPrediction(
        strategy_id=strategy_id,
        name=name,
        description=description,
        numbers=predictions,
        top_numbers=tuple(ranked[:_NUMBERS_PER_DRAW]),
    )


def _random_ranking(draw_index: int, seed: int = _RANDOM_SEED) -> list[int]:
    """Return the deterministic LCG/Fisher-Yates baseline used by PyLotto."""
    state = seed ^ (((draw_index + 1) * 2654435761) & 0xFFFFFFFF)
    state &= 0xFFFFFFFF
    numbers = list(range(1, _NUMBER_COUNT + 1))
    for index in range(len(numbers) - 1, 0, -1):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        swap_index = state % (index + 1)
        numbers[index], numbers[swap_index] = numbers[swap_index], numbers[index]
    return numbers


def _ranking_from_scores(
    scores: dict[int, float],
    gaps: dict[int, int],
) -> list[int]:
    return sorted(
        scores,
        key=lambda number: (-scores[number], -gaps[number], number),
    )


def _rank_strength(ranking: Sequence[int], number: int) -> float:
    try:
        rank = ranking.index(number) + 1
    except ValueError:
        rank = _NUMBER_COUNT
    return (_NUMBER_COUNT - rank) / (_NUMBER_COUNT - 1)


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _average(values)
    return _average([(value - mean) ** 2 for value in values])


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


class _EfficacyTracker:
    """Accumulate walk-forward comparisons without retaining every ranking."""

    def __init__(self) -> None:
        self.evaluated_draws = 0
        self.random_hits = 0
        self.strategy_hits: dict[str, int] = {}

    def compare(
        self,
        suite: PredictionSuite,
    ) -> tuple[PredictionSuite, StrategyEfficacyRecord | None]:
        """Attach metrics available through this suite's target draw."""
        record: StrategyEfficacyRecord | None = None
        if suite.actual_numbers:
            self.evaluated_draws += 1
            actual = set(suite.actual_numbers)
            random_top = _random_ranking(suite.reference_draw_number)[
                :_NUMBERS_PER_DRAW
            ]
            current_random_hits = len(actual.intersection(random_top))
            self.random_hits += current_random_hits
            current_strategy_hits = tuple(
                (
                    strategy.strategy_id,
                    len(actual.intersection(strategy.top_numbers)),
                )
                for strategy in suite.strategies
            )
            for strategy_id, hits in current_strategy_hits:
                self.strategy_hits[strategy_id] = (
                    self.strategy_hits.get(strategy_id, 0) + hits
                )
            record = StrategyEfficacyRecord(
                reference_draw_number=suite.reference_draw_number,
                target_draw_number=suite.target_draw_number,
                actual_numbers=suite.actual_numbers,
                random_hits=current_random_hits,
                strategy_hits=current_strategy_hits,
            )

        expected_random_hits = self.evaluated_draws * _EXPECTED_RANDOM_HITS_PER_DRAW
        updated_strategies = tuple(
            replace(
                strategy,
                efficacy=StrategyEfficacy(
                    evaluated_draws=self.evaluated_draws,
                    strategy_hits=self.strategy_hits.get(strategy.strategy_id, 0),
                    random_hits=self.random_hits,
                    expected_random_hits=expected_random_hits,
                    average_hits_per_draw=(
                        self.strategy_hits.get(strategy.strategy_id, 0)
                        / self.evaluated_draws
                        if self.evaluated_draws
                        else 0.0
                    ),
                    random_average_hits_per_draw=(
                        self.random_hits / self.evaluated_draws
                        if self.evaluated_draws
                        else 0.0
                    ),
                    hit_difference=(
                        self.strategy_hits.get(strategy.strategy_id, 0)
                        - self.random_hits
                    ),
                ),
            )
            for strategy in suite.strategies
        )
        return replace(suite, strategies=updated_strategies), record


def _proximity_bucket(distance: int) -> int:
    if distance <= 1:
        return 0
    if distance <= 3:
        return 1
    if distance <= 6:
        return 2
    if distance <= 10:
        return 3
    if distance <= 15:
        return 4
    return 5


def _gap_entropy_percent(numbers: Sequence[int]) -> float:
    ordered = sorted(numbers)
    circular_gaps = [
        ordered[index + 1] - ordered[index] for index in range(len(ordered) - 1)
    ]
    circular_gaps.append((_NUMBER_COUNT + ordered[0]) - ordered[-1])
    total = sum(circular_gaps)
    entropy = -sum(
        (gap / total) * math.log2(gap / total) for gap in circular_gaps if gap > 0
    )
    return entropy / math.log2(_NUMBERS_PER_DRAW) * 100


def _spaces_for_numbers(numbers: Collection[int]) -> tuple[int, ...]:
    ordered = sorted(numbers)
    if len(ordered) != _NUMBERS_PER_DRAW:
        raise ValueError("Space states require exactly six numbers")
    return (
        (ordered[0] - 1) + (_NUMBER_COUNT - ordered[-1]),
        *(
            right - left - 1
            for left, right in zip(ordered, ordered[1:])
        ),
    )


def _normalized_positions_for_numbers(
    numbers: Collection[int],
) -> tuple[int, ...]:
    """Translate a six-number draw so its first number occupies position one."""
    ordered = sorted(numbers)
    if len(ordered) != _NUMBERS_PER_DRAW:
        raise ValueError("Normalized position states require exactly six numbers")
    first = ordered[0]
    return tuple(number - first + 1 for number in ordered)


@dataclass(frozen=True, slots=True)
class _RelativeDispersionProfile:
    """Describe a draw's scale-independent shape and dispersion summaries."""

    span: int
    coverage: float
    relative_positions: tuple[float, ...]
    gap_shares: tuple[float, ...]
    uniformity: float
    entropy: float
    center_balance: float


def _relative_dispersion_profile(
    numbers: Collection[int],
) -> _RelativeDispersionProfile:
    """Return the fixed Markov Relative Dispersion feature profile."""
    ordered = sorted(numbers)
    if len(ordered) != _NUMBERS_PER_DRAW:
        raise ValueError("Relative dispersion states require exactly six numbers")
    if any(left >= right for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Relative dispersion states must be strictly increasing")
    if ordered[0] < 1 or ordered[-1] > _NUMBER_COUNT:
        raise ValueError("Relative dispersion states must be within 1–49")

    extent = ordered[-1] - ordered[0]
    span = extent + 1
    relative_positions = tuple(
        (number - ordered[0]) / extent for number in ordered
    )
    gap_shares = tuple(
        (right - left) / extent
        for left, right in zip(ordered, ordered[1:])
    )
    ideal_share = 1 / (_NUMBERS_PER_DRAW - 1)
    gap_share_std = math.sqrt(
        sum((share - ideal_share) ** 2 for share in gap_shares)
        / len(gap_shares)
    )
    # The maximum population standard deviation for five shares is 0.4.
    uniformity = _clamp(gap_share_std / 0.4, 0.0, 1.0)
    entropy = -sum(
        share * math.log(share) for share in gap_shares if share > 0
    ) / math.log(len(gap_shares))
    relative_mean = sum(relative_positions) / len(relative_positions)
    # Strictly ordered points bound the mean offset from 0.5 by one third.
    center_balance = _clamp(
        0.5 + 1.5 * (relative_mean - 0.5),
        0.0,
        1.0,
    )
    return _RelativeDispersionProfile(
        span=span,
        coverage=span / _NUMBER_COUNT,
        relative_positions=relative_positions,
        gap_shares=gap_shares,
        uniformity=uniformity,
        entropy=entropy,
        center_balance=center_balance,
    )


class _StrategyState:
    """Maintain incremental state for the enabled prediction strategy plugins."""

    def __init__(
        self,
        enabled_strategy_ids: Collection[str] = STRATEGY_IDS,
        total_draw_count: int = 1,
    ) -> None:
        self.requested_strategy_ids = frozenset(enabled_strategy_ids)
        active_strategy_ids = set(self.requested_strategy_ids)
        unresolved_strategy_ids = list(active_strategy_ids)
        while unresolved_strategy_ids:
            strategy_id = unresolved_strategy_ids.pop()
            for dependency_id in _STRATEGY_DEPENDENCIES.get(strategy_id, ()):
                if dependency_id not in active_strategy_ids:
                    active_strategy_ids.add(dependency_id)
                    unresolved_strategy_ids.append(dependency_id)
        self.enabled_strategy_ids = frozenset(active_strategy_ids)
        self.total_draw_count = max(total_draw_count, 1)
        self.draw_count = 0
        self.appearances = [0] * (_NUMBER_COUNT + 1)
        self.last_seen: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.occurrences: list[list[int]] = [[] for _ in range(_NUMBER_COUNT + 1)]
        self.recent_draws: deque[set[int]] = deque(maxlen=100)
        self.pair_counts: dict[tuple[int, int], int] = {}
        doublet_triplet_markov_enabled = (
            "doublet_triplet_markov" in self.enabled_strategy_ids
        )
        self.doublet_markov_counts = (
            [0] * (_NUMBER_COUNT + 1)
            if doublet_triplet_markov_enabled
            else []
        )
        self.triplet_markov_counts = (
            [0] * (_NUMBER_COUNT + 1)
            if doublet_triplet_markov_enabled
            else []
        )
        self.doublet_markov_transitions = (
            [
                [0] * (_NUMBER_COUNT + 1)
                for _ in range(_NUMBER_COUNT + 1)
            ]
            if doublet_triplet_markov_enabled
            else []
        )
        self.triplet_markov_transitions = (
            [
                [0] * (_NUMBER_COUNT + 1)
                for _ in range(_NUMBER_COUNT + 1)
            ]
            if doublet_triplet_markov_enabled
            else []
        )
        self.doublet_triplet_transition_totals = (
            [0] * (_NUMBER_COUNT + 1)
            if doublet_triplet_markov_enabled
            else []
        )
        self.doublet_triplet_shape_counts = (
            [0, 0, 0] if doublet_triplet_markov_enabled else []
        )
        self.doublet_triplet_shape_transitions = (
            [[0, 0, 0] for _ in range(3)]
            if doublet_triplet_markov_enabled
            else []
        )
        self.doublet_triplet_shape_transition_totals = (
            [0, 0, 0] if doublet_triplet_markov_enabled else []
        )
        self.doublet_triplet_recent_groups: deque[
            tuple[frozenset[int], frozenset[int]]
        ] = deque(maxlen=_DOUBLET_TRIPLET_MARKOV_RECENT_WINDOW)
        self.previous_draw: set[int] = set()
        self.previous_previous_draw: set[int] = set()
        self.current_month = 0
        self.transition_counts = [
            [0] * (_NUMBER_COUNT + 1) for _ in range(_NUMBER_COUNT + 1)
        ]
        self.transition_totals = [0] * (_NUMBER_COUNT + 1)
        self.proximity_counts = [
            [0] * len(_PROXIMITY_BUCKETS) for _ in range(_NUMBER_COUNT + 1)
        ]
        self.proximity_totals = [0] * len(_PROXIMITY_BUCKETS)
        self.entropy_totals = [0.0] * (_NUMBER_COUNT + 1)
        self.high_entropy_hits = [0] * (_NUMBER_COUNT + 1)
        self.markov_opportunities = [0.0] * (_MAX_GAP_BUCKET + 1)
        self.markov_hits = [0.0] * (_MAX_GAP_BUCKET + 1)
        self.bayesian_opportunities = [0] * (_MAX_GAP_BUCKET + 1)
        self.bayesian_hits = [0] * (_MAX_GAP_BUCKET + 1)
        self.bayesian_recent_opportunities = [0.0] * (_MAX_GAP_BUCKET + 1)
        self.bayesian_recent_hits = [0.0] * (_MAX_GAP_BUCKET + 1)
        self.bayesian_recent_number_hits = [0.0] * (_NUMBER_COUNT + 1)
        self.mkfr_histories: list[deque[int]] = (
            [deque(maxlen=_MKFR_MAX_ORDER) for _ in range(_NUMBER_COUNT + 1)]
            if "mkfr" in self.enabled_strategy_ids
            else []
        )
        self.mkfr_transitions: list[list[dict[int, list[int]]]] = (
            [[{} for _ in range(_MKFR_MAX_ORDER)] for _ in range(_NUMBER_COUNT + 1)]
            if "mkfr" in self.enabled_strategy_ids
            else []
        )
        self.mksp_histories: list[deque[int]] = (
            [deque(maxlen=_MKSP_MAX_ORDER) for _ in range(_NUMBERS_PER_DRAW)]
            if "mksp" in self.enabled_strategy_ids
            else []
        )
        self.mksp_transitions: list[
            list[dict[tuple[int, ...], dict[int, int]]]
        ] = (
            [
                [{} for _ in range(_MKSP_MAX_ORDER)]
                for _ in range(_NUMBERS_PER_DRAW)
            ]
            if "mksp" in self.enabled_strategy_ids
            else []
        )
        self.mksp_value_counts: list[list[int]] = (
            [[0] * _MKSP_VALUE_COUNT for _ in range(_NUMBERS_PER_DRAW)]
            if "mksp" in self.enabled_strategy_ids
            else []
        )
        self.mksp_anchor_counts: list[int] = (
            [0] * _MKSP_VALUE_COUNT
            if "mksp" in self.enabled_strategy_ids
            else []
        )
        self.mksp_observations: list[tuple[tuple[int, ...], int]] = []
        self.mknp_histories: list[deque[int]] = (
            [deque(maxlen=_MKSP_MAX_ORDER) for _ in range(_MKNP_POSITION_COUNT)]
            if "mknp" in self.enabled_strategy_ids
            else []
        )
        self.mknp_transitions: list[
            list[dict[tuple[int, ...], dict[int, int]]]
        ] = (
            [
                [{} for _ in range(_MKSP_MAX_ORDER)]
                for _ in range(_MKNP_POSITION_COUNT)
            ]
            if "mknp" in self.enabled_strategy_ids
            else []
        )
        self.mknp_value_counts: list[list[int]] = (
            [[0] * (_NUMBER_COUNT + 1) for _ in range(_MKNP_POSITION_COUNT)]
            if "mknp" in self.enabled_strategy_ids
            else []
        )
        self.mknp_anchor_counts: list[int] = (
            [0] * _MKSP_VALUE_COUNT
            if "mknp" in self.enabled_strategy_ids
            else []
        )
        self.mknp_observations: list[tuple[tuple[int, ...], int]] = []
        self.mkrd_histories: list[deque[int]] = (
            [deque(maxlen=_MKSP_MAX_ORDER) for _ in range(_MKNP_POSITION_COUNT)]
            if "mkrd" in self.enabled_strategy_ids
            else []
        )
        self.mkrd_transitions: list[
            list[dict[tuple[int, ...], dict[int, int]]]
        ] = (
            [
                [{} for _ in range(_MKSP_MAX_ORDER)]
                for _ in range(_MKNP_POSITION_COUNT)
            ]
            if "mkrd" in self.enabled_strategy_ids
            else []
        )
        self.mkrd_value_counts: list[list[int]] = (
            [[0] * (_NUMBER_COUNT + 1) for _ in range(_MKNP_POSITION_COUNT)]
            if "mkrd" in self.enabled_strategy_ids
            else []
        )
        self.mkrd_anchor_counts: list[int] = (
            [0] * _MKSP_VALUE_COUNT
            if "mkrd" in self.enabled_strategy_ids
            else []
        )
        self.mkrd_observations: list[
            tuple[tuple[int, ...], int, _RelativeDispersionProfile]
        ] = []
        self.svc_weights = [0.0] * 11
        self.tbl_weights = [0.0] * 14
        self.sklearn_svm = (
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=0.0001,
                learning_rate="optimal",
                fit_intercept=True,
                average=True,
                random_state=_RANDOM_SEED,
            )
            if "sklearn_svm" in self.enabled_strategy_ids
            else None
        )
        self.sklearn_svm_fitted = False
        self.sklearn_svm_trained_draws = 0
        self.sklearn_svm_pending_features: dict[int, tuple[float, ...]] = {}
        self.sklearn_svm_pending_rankings: dict[str, list[int]] = {}
        self.sklearn_svm_expert_total_hits = {
            strategy_id: 0 for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        }
        self.sklearn_svm_expert_evaluated_draws = {
            strategy_id: 0 for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        }
        self.sklearn_svm_expert_recent_hits = {
            strategy_id: deque(maxlen=_SKLEARN_SVM_RECENT_WINDOW)
            for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        }
        self.cis_weights = [0.0] * (22 + len(_CIS_EXPERTS) * 4)
        self.cis_draw_count = 0
        self.cis_total_hits = {
            strategy_id: 0 for strategy_id, _label, _weight in _CIS_EXPERTS
        }
        self.cis_evaluated_draws = {
            strategy_id: 0 for strategy_id, _label, _weight in _CIS_EXPERTS
        }
        self.cis_recent_hits = {
            strategy_id: deque(maxlen=_CIS_RECENT_WINDOW)
            for strategy_id, _label, _weight in _CIS_EXPERTS
        }
        self.cis_recent_ensemble_hits: deque[int] = deque(
            maxlen=_CIS_RECENT_WINDOW
        )
        self.cis_recent_learner_hits: deque[int] = deque(
            maxlen=_CIS_RECENT_WINDOW
        )
        self.cis_prior_rankings: dict[str, list[int]] = {}
        self.cis_pending_rankings: dict[str, list[int]] = {}
        self.cis_pending_features: dict[int, tuple[float, ...]] = {}
        self.cis_pending_ensemble_scores: dict[int, float] = {}
        self.cis_pending_learner_scores: dict[int, float] = {}
        self.chain_evaluated_draws = 0
        self.chain_total_hits = {
            strategy_id: 0 for strategy_id in _CHAIN_EXPERT_IDS
        }
        self.chain_recent_hits = {
            strategy_id: deque(maxlen=40) for strategy_id in _CHAIN_EXPERT_IDS
        }
        self.chain_pending_rankings: dict[str, list[int]] = {}
        self.prior_rankings = {
            "freshness": list(range(1, _NUMBER_COUNT + 1)),
            "proximity": list(range(1, _NUMBER_COUNT + 1)),
            "randomness": list(range(1, _NUMBER_COUNT + 1)),
        }
        self.draw_vectors: list[tuple[int, ...]] = []

    def _gap_before_current_draw(self, number: int) -> int:
        seen_at = self.last_seen[number]
        return self.draw_count if seen_at is None else self.draw_count - seen_at - 1

    def current_gaps(self) -> dict[int, int]:
        gaps: dict[int, int] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            seen_at = self.last_seen[number]
            gaps[number] = (
                self.draw_count if seen_at is None else self.draw_count - 1 - seen_at
            )
        return gaps

    def _recent_count(self, number: int, window: int) -> int:
        return sum(number in draw for draw in list(self.recent_draws)[-window:])

    def _svc_features(self, number: int) -> list[float]:
        draw_count = max(self.draw_count, 1)
        seen_at = self.last_seen[number]
        gap = draw_count + 1 if seen_at is None else self.draw_count - seen_at
        expected = max(self.draw_count * _BASE_PROBABILITY, 1)
        recent8_length = min(len(self.recent_draws), 8)
        recent24_length = min(len(self.recent_draws), 24)
        recent8_expected = max(recent8_length * _BASE_PROBABILITY, 1)
        recent24_expected = max(recent24_length * _BASE_PROBABILITY, 1)
        return [
            1,
            number / _NUMBER_COUNT,
            float(number <= 24),
            float(number % 2 == 1),
            min(gap / 40, 1),
            float(gap == 1),
            float(2 <= gap <= 4),
            float(gap >= 12),
            _clamp((expected - self.appearances[number]) / expected, -1, 1),
            _clamp(
                (recent8_expected - self._recent_count(number, 8)) / recent8_expected,
                -1,
                1,
            ),
            _clamp(
                (recent24_expected - self._recent_count(number, 24))
                / recent24_expected,
                -1,
                1,
            ),
        ]

    def _rank_score(self, strategy: str, number: int) -> float:
        ranking = self.prior_rankings[strategy]
        try:
            rank = ranking.index(number) + 1
        except ValueError:
            rank = _NUMBER_COUNT
        return (_NUMBER_COUNT - rank) / (_NUMBER_COUNT - 1)

    def _mean_gap(self, number: int) -> float:
        occurrences = self.occurrences[number]
        if len(occurrences) < 2:
            return 0.0
        gaps = [right - left for left, right in zip(occurrences, occurrences[1:])]
        return sum(gaps) / len(gaps)

    def _previous_compatibility(self, number: int) -> float:
        if not self.previous_draw:
            return 0.0
        count = sum(
            self.pair_counts.get(tuple(sorted((previous, number))), 0)
            for previous in self.previous_draw
            if previous != number
        )
        return count / max(len(self.previous_draw) * self.draw_count, 1)

    def _tbl_features(self, number: int) -> list[float]:
        draw_count = max(self.draw_count, 1)
        gap = self.current_gaps()[number]
        mean_gap = self._mean_gap(number)
        overdue = 0.0 if mean_gap <= 0 else _clamp((gap - mean_gap) / mean_gap, -1, 1)
        recent5 = self._recent_count(number, 5) / max(min(len(self.recent_draws), 5), 1)
        recent20 = self._recent_count(number, 20) / max(
            min(len(self.recent_draws), 20), 1
        )
        return [
            1,
            number / _NUMBER_COUNT,
            float(number <= 24),
            float(number % 2 == 1),
            float(number in _PRIMES),
            min(gap / 60, 1),
            overdue,
            self.appearances[number] / draw_count,
            recent5,
            recent20,
            recent5 - recent20,
            self._previous_compatibility(number),
            self._rank_score("freshness", number),
            (
                self._rank_score("proximity", number)
                + self._rank_score("randomness", number)
            )
            / 2,
        ]

    def _sklearn_svm_recent_residual(self, number: int, window: int) -> float:
        recent = list(self.recent_draws)[-window:]
        if not recent:
            return 0.0
        observed_rate = sum(number in draw for draw in recent) / len(recent)
        return _clamp(
            (observed_rate - _BASE_PROBABILITY) / _BASE_PROBABILITY,
            -1,
            1,
        )

    def _sklearn_svm_relationship_residual(self, number: int) -> float:
        if not self.previous_draw:
            return 0.0
        conditional_rates = [
            self.pair_counts.get(tuple(sorted((previous, number))), 0)
            / max(self.appearances[previous], 1)
            for previous in self.previous_draw
            if previous != number
        ]
        if not conditional_rates:
            return 0.0
        return _clamp(
            (_average(conditional_rates) - _BASE_PROBABILITY)
            / _BASE_PROBABILITY,
            -1,
            1,
        )

    def _sklearn_svm_expert_weight(self, strategy_id: str) -> float:
        evaluated = self.sklearn_svm_expert_evaluated_draws[strategy_id]
        prior_draws = _SKLEARN_SVM_EFFECTIVENESS_PRIOR_DRAWS
        long_term_hits = (
            self.sklearn_svm_expert_total_hits[strategy_id]
            + prior_draws * _EXPECTED_RANDOM_HITS_PER_DRAW
        ) / (evaluated + prior_draws)
        recent = self.sklearn_svm_expert_recent_hits[strategy_id]
        recent_hits = (
            sum(recent) + prior_draws * _EXPECTED_RANDOM_HITS_PER_DRAW
        ) / (len(recent) + prior_draws)
        confidence = evaluated / (evaluated + prior_draws)
        blended_hits = (
            long_term_hits * (1 - 0.70 * confidence)
            + recent_hits * 0.70 * confidence
        )
        return _clamp(
            blended_hits / _EXPECTED_RANDOM_HITS_PER_DRAW,
            0.5,
            1.5,
        )

    def _sklearn_svm_features(
        self,
        number: int,
        rankings: dict[str, list[int]],
    ) -> tuple[float, ...]:
        gap = self.current_gaps()[number]
        mean_gap = self._mean_gap(number)
        overdue = (
            0.0
            if mean_gap <= 0
            else _clamp((gap - mean_gap) / mean_gap, -1, 1)
        )
        lifetime_rate = self.appearances[number] / max(self.draw_count, 1)
        lifetime_residual = _clamp(
            (lifetime_rate - _BASE_PROBABILITY) / _BASE_PROBABILITY,
            -1,
            1,
        )
        recent5 = self._sklearn_svm_recent_residual(number, 5)
        recent20 = self._sklearn_svm_recent_residual(number, 20)
        recent100 = self._sklearn_svm_recent_residual(number, 100)
        expert_strengths = [
            _rank_strength(rankings[strategy_id], number)
            for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        ]
        expert_weights = [
            self._sklearn_svm_expert_weight(strategy_id)
            for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        ]
        top_six_support = sum(
            rankings[strategy_id].index(number) < _NUMBERS_PER_DRAW
            for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        )
        top_quarter_limit = math.ceil(_NUMBER_COUNT * 0.25)
        top_quarter_support = sum(
            rankings[strategy_id].index(number) < top_quarter_limit
            for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        )
        features = [
            number / _NUMBER_COUNT,
            float(number <= 24),
            float(number % 2 == 1),
            float(number in _PRIMES),
            min(gap / 60, 1),
            float(gap == 1),
            float(2 <= gap <= 4),
            float(gap >= 12),
            overdue,
            lifetime_residual,
            recent5,
            recent20,
            recent100,
            _clamp(recent5 - recent20, -1, 1),
            self._sklearn_svm_relationship_residual(number),
        ]
        for strength, weight in zip(expert_strengths, expert_weights):
            features.extend((strength, strength * weight / 1.5))
        features.extend(
            (
                _average(expert_strengths),
                _median(expert_strengths),
                top_six_support / len(_SKLEARN_SVM_EXPERT_IDS),
                top_quarter_support / len(_SKLEARN_SVM_EXPERT_IDS),
                _variance(expert_strengths),
            )
        )
        if len(features) != _SKLEARN_SVM_FEATURE_COUNT:
            raise AssertionError(  # pragma: no cover - fixed schema invariant
                "Unexpected Scikit Online SVM feature count"
            )
        return tuple(features)

    def _train_sklearn_svm(self, drawn: set[int]) -> None:
        if (
            self.sklearn_svm is None
            or not self.sklearn_svm_pending_features
        ):
            return
        numbers = list(range(1, _NUMBER_COUNT + 1))
        features = np.asarray(
            [self.sklearn_svm_pending_features[number] for number in numbers],
            dtype=float,
        )
        labels = np.asarray([int(number in drawn) for number in numbers])
        positive_weight = (_NUMBER_COUNT - _NUMBERS_PER_DRAW) / _NUMBERS_PER_DRAW
        sample_weights = np.asarray(
            [positive_weight if label else 1.0 for label in labels],
            dtype=float,
        )
        if self.sklearn_svm_fitted:
            self.sklearn_svm.partial_fit(
                features,
                labels,
                sample_weight=sample_weights,
            )
        else:
            self.sklearn_svm.partial_fit(
                features,
                labels,
                classes=np.asarray([0, 1]),
                sample_weight=sample_weights,
            )
            self.sklearn_svm_fitted = True
        self.sklearn_svm_trained_draws += 1
        for strategy_id, ranking in self.sklearn_svm_pending_rankings.items():
            hits = len(drawn.intersection(ranking[:_NUMBERS_PER_DRAW]))
            self.sklearn_svm_expert_total_hits[strategy_id] += hits
            self.sklearn_svm_expert_evaluated_draws[strategy_id] += 1
            self.sklearn_svm_expert_recent_hits[strategy_id].append(hits)

    def _sklearn_svm_scores(
        self,
        rankings: dict[str, list[int]],
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        self.sklearn_svm_pending_rankings = {
            strategy_id: list(rankings[strategy_id])
            for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        }
        self.sklearn_svm_pending_features = {
            number: self._sklearn_svm_features(number, rankings)
            for number in range(1, _NUMBER_COUNT + 1)
        }
        expert_weights = {
            strategy_id: self._sklearn_svm_expert_weight(strategy_id)
            for strategy_id in _SKLEARN_SVM_EXPERT_IDS
        }
        if self.sklearn_svm is not None and self.sklearn_svm_fitted:
            feature_rows = np.asarray(
                [
                    self.sklearn_svm_pending_features[number]
                    for number in range(1, _NUMBER_COUNT + 1)
                ],
                dtype=float,
            )
            decision_values = self.sklearn_svm.decision_function(feature_rows)
            margins = {
                number: float(decision_values[number - 1])
                for number in range(1, _NUMBER_COUNT + 1)
            }
            score_label = "Margin"
        else:
            weight_total = sum(expert_weights.values()) or 1.0
            margins = {
                number: sum(
                    _rank_strength(rankings[strategy_id], number)
                    * expert_weights[strategy_id]
                    for strategy_id in _SKLEARN_SVM_EXPERT_IDS
                )
                / weight_total
                for number in range(1, _NUMBER_COUNT + 1)
            }
            score_label = "Cold-start consensus"
        scores = _scale_scores(margins)
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            strongest = sorted(
                _SKLEARN_SVM_EXPERT_IDS,
                key=lambda strategy_id: (
                    -_rank_strength(rankings[strategy_id], number)
                    * expert_weights[strategy_id],
                    strategy_id,
                ),
            )[:3]
            details[number] = (
                f"{score_label} {margins[number]:.3f}",
                f"Trained draws {self.sklearn_svm_trained_draws}",
                f"Strongest expert inputs: {', '.join(strongest)}",
            )
        return scores, details

    def _cis_expert_accuracy(
        self,
        strategy_id: str,
        window_size: int,
    ) -> float:
        recent = list(self.cis_recent_hits[strategy_id])[-window_size:]
        smoothed_hits = (
            sum(recent)
            + _CIS_EXPERT_PRIOR_DRAWS * _EXPECTED_RANDOM_HITS_PER_DRAW
        ) / (len(recent) + _CIS_EXPERT_PRIOR_DRAWS)
        return smoothed_hits / _NUMBERS_PER_DRAW

    def _cis_expert_weight(
        self,
        strategy_id: str,
        base_weight: float,
    ) -> float:
        evaluated = self.cis_evaluated_draws[strategy_id]
        long_term_hits = (
            self.cis_total_hits[strategy_id]
            + _CIS_EXPERT_PRIOR_DRAWS * _EXPECTED_RANDOM_HITS_PER_DRAW
        ) / (evaluated + _CIS_EXPERT_PRIOR_DRAWS)
        recent = list(self.cis_recent_hits[strategy_id])
        recent_hits = (
            sum(recent)
            + _CIS_EXPERT_PRIOR_DRAWS * _EXPECTED_RANDOM_HITS_PER_DRAW
        ) / (len(recent) + _CIS_EXPERT_PRIOR_DRAWS)
        confidence = evaluated / (evaluated + _CIS_EXPERT_PRIOR_DRAWS)
        excess_hits = confidence * (
            0.70 * (recent_hits - _EXPECTED_RANDOM_HITS_PER_DRAW)
            + 0.30 * (long_term_hits - _EXPECTED_RANDOM_HITS_PER_DRAW)
        )
        return base_weight * math.exp(
            _clamp(excess_hits * 3.0, -1.5, 1.5)
        )

    def _cis_learner_blend(self) -> float:
        evaluated = min(
            len(self.cis_recent_ensemble_hits),
            len(self.cis_recent_learner_hits),
        )
        if self.cis_draw_count < _CIS_MINIMUM_TRAINING_DRAWS or evaluated == 0:
            return 0.0
        advantage = (
            sum(self.cis_recent_learner_hits)
            - sum(self.cis_recent_ensemble_hits)
        ) / (evaluated + _CIS_LEARNER_EVIDENCE_DRAWS)
        confidence = evaluated / (evaluated + _CIS_LEARNER_EVIDENCE_DRAWS)
        return min(
            _CIS_LEARNER_MAX_BLEND,
            max(0.0, advantage - _CIS_LEARNER_MIN_ADVANTAGE)
            * 0.5
            * confidence,
        )

    def _cis_previous_draw_features(self) -> tuple[float, float, float]:
        if not self.previous_draw:
            return 0.0, 0.0, 0.0
        values = sorted(self.previous_draw)
        gaps = [right - left for left, right in zip(values, values[1:])]
        entropy = min(_variance(gaps) / 100, 1)
        low_count = sum(value <= 24 for value in values)
        odd_count = sum(value % 2 == 1 for value in values)
        balance = 1 - (abs(low_count - 3) + abs(odd_count - 3)) / _NUMBERS_PER_DRAW
        volatility = (
            0.0
            if not self.previous_previous_draw
            else abs(sum(self.previous_draw) - sum(self.previous_previous_draw)) / 294
        )
        return entropy, balance, volatility

    def _cis_features(
        self,
        number: int,
        rankings: dict[str, list[int]],
        rank_maps: dict[str, dict[int, int]],
        prior_rank_maps: dict[str, dict[int, int]],
        dynamic_weights: dict[str, float],
        recent_accuracies: dict[str, float],
        long_term_accuracies: dict[str, float],
    ) -> tuple[float, ...]:
        ranks = [
            rank_maps[strategy_id][number]
            for strategy_id, _label, _weight in _CIS_EXPERTS
            if strategy_id in rankings
        ]
        strengths = [(_NUMBER_COUNT - rank) / (_NUMBER_COUNT - 1) for rank in ranks]
        consensus_count = sum(rank <= math.ceil(_NUMBER_COUNT * 0.25) for rank in ranks)
        top_six_count = sum(rank <= _NUMBERS_PER_DRAW for rank in ranks)
        opposition_count = sum(
            rank > math.floor(_NUMBER_COUNT * 0.75) for rank in ranks
        )
        expert_count = max(len(_CIS_EXPERTS), 1)
        disagreement = _variance(strengths)
        normalized_rank_variance = _variance([float(rank) for rank in ranks]) / (
            _NUMBER_COUNT**2
        )
        entropy_inputs = [
            top_six_count / expert_count,
            consensus_count / expert_count,
            opposition_count / expert_count,
        ]
        recommendation_entropy = -sum(
            value * math.log2(value) for value in entropy_inputs if value > 0
        ) / math.log2(3)
        weighted_strength_total = 0.0
        dynamic_weight_total = 0.0
        for strategy_id, _label, _base_weight in _CIS_EXPERTS:
            rank = rank_maps.get(strategy_id, {}).get(number)
            strength = (
                (_NUMBER_COUNT - rank) / (_NUMBER_COUNT - 1)
                if rank is not None
                else 0.0
            )
            dynamic_weight = dynamic_weights[strategy_id]
            weighted_strength_total += strength * dynamic_weight
            dynamic_weight_total += dynamic_weight
        weighted_agreement = weighted_strength_total / max(
            dynamic_weight_total,
            1e-12,
        )
        previous_entropy, previous_balance, volatility = (
            self._cis_previous_draw_features()
        )
        month = self.current_month
        season = math.floor((month % 12) / 3) / 3 if month else 0.0
        features = [
            1.0,
            _average(strengths),
            _median(strengths),
            max([*strengths, 0.0]),
            min([*strengths, 0.0]),
            disagreement,
            math.sqrt(disagreement),
            consensus_count / expert_count,
            top_six_count / expert_count,
            opposition_count / expert_count,
            consensus_count / expert_count,
            min(disagreement * 4, 1),
            min(normalized_rank_variance, 1),
            recommendation_entropy if math.isfinite(recommendation_entropy) else 0.0,
            weighted_agreement,
            weighted_agreement * (1 - min(disagreement * 2, 1)),
            min(self.draw_count / 500, 1),
            (month - 1) / 11 if month else 0.0,
            season,
            previous_entropy,
            previous_balance,
            volatility,
        ]
        for strategy_id, _label, _base_weight in _CIS_EXPERTS:
            current_rank = rank_maps.get(strategy_id, {}).get(number)
            previous_rank = prior_rank_maps.get(strategy_id, {}).get(number)
            momentum = (
                0.0
                if previous_rank is None or current_rank is None
                else _clamp(
                    (previous_rank - current_rank) / _NUMBER_COUNT,
                    -1,
                    1,
                )
            )
            features.extend(
                (
                    (
                        0.0
                        if current_rank is None
                        else (_NUMBER_COUNT - current_rank) / (_NUMBER_COUNT - 1)
                    ),
                    recent_accuracies[strategy_id],
                    long_term_accuracies[strategy_id],
                    momentum,
                )
            )
        return tuple(features)

    @staticmethod
    def _cis_nonlinear_score(features: Sequence[float]) -> float:
        return (
            math.tanh((features[14] - 0.3) * 3) * 0.22
            + math.tanh((features[7] - features[9]) * 2.5) * 0.18
            - math.tanh(features[12] * 2) * 0.08
        )

    def _cis_probability(self, features: Sequence[float]) -> float:
        return _sigmoid(
            self._dot(self.cis_weights, features) + self._cis_nonlinear_score(features)
        )

    def _train_cis(self, drawn: set[int]) -> None:
        if self.cis_pending_ensemble_scores and self.cis_pending_learner_scores:
            ensemble_top = set(
                nlargest(
                    _NUMBERS_PER_DRAW,
                    self.cis_pending_ensemble_scores,
                    key=self.cis_pending_ensemble_scores.__getitem__,
                )
            )
            learner_top = set(
                nlargest(
                    _NUMBERS_PER_DRAW,
                    self.cis_pending_learner_scores,
                    key=self.cis_pending_learner_scores.__getitem__,
                )
            )
            self.cis_recent_ensemble_hits.append(len(drawn & ensemble_top))
            self.cis_recent_learner_hits.append(len(drawn & learner_top))

        learning_rate = 0.018 / math.sqrt(1 + self.cis_draw_count / 250)
        decay = 1 - learning_rate * 0.0008
        self.cis_weights = [weight * decay for weight in self.cis_weights]
        positives = [
            number for number in drawn if number in self.cis_pending_features
        ]
        hard_negatives = nlargest(
            12,
            (
                number
                for number in self.cis_pending_features
                if number not in drawn
            ),
            key=lambda number: self.cis_pending_learner_scores.get(number, 0.0),
        )
        pair_count = max(len(positives) * len(hard_negatives), 1)
        pair_rate = learning_rate / pair_count
        for positive in positives:
            positive_features = self.cis_pending_features[positive]
            for negative in hard_negatives:
                negative_features = self.cis_pending_features[negative]
                differences = tuple(
                    positive_feature - negative_feature
                    for positive_feature, negative_feature in zip(
                        positive_features,
                        negative_features,
                    )
                )
                gradient = _sigmoid(-self._dot(self.cis_weights, differences))
                for index, difference in enumerate(differences):
                    self.cis_weights[index] += pair_rate * gradient * difference

        for strategy_id, ranking in self.cis_pending_rankings.items():
            hits = len(drawn.intersection(ranking[:_NUMBERS_PER_DRAW]))
            self.cis_total_hits[strategy_id] += hits
            self.cis_evaluated_draws[strategy_id] += 1
            self.cis_recent_hits[strategy_id].append(hits)
            self.cis_prior_rankings[strategy_id] = list(ranking)
        self.cis_draw_count += 1

    def _train_chained_effectiveness(self, drawn: set[int]) -> None:
        if not self.chain_pending_rankings:
            return
        for strategy_id, ranking in self.chain_pending_rankings.items():
            hits = len(drawn.intersection(ranking[:_NUMBERS_PER_DRAW]))
            self.chain_total_hits[strategy_id] += hits
            self.chain_recent_hits[strategy_id].append(hits)
        self.chain_evaluated_draws += 1

    def _chain_expert_weight(self, strategy_id: str) -> float:
        evaluated = self.chain_evaluated_draws
        long_term_hits = self.chain_total_hits[strategy_id]
        smoothed_hits = (
            long_term_hits
            + _CHAIN_EFFECTIVENESS_PRIOR_DRAWS
            * _EXPECTED_RANDOM_HITS_PER_DRAW
        ) / (evaluated + _CHAIN_EFFECTIVENESS_PRIOR_DRAWS)
        recent = self.chain_recent_hits[strategy_id]
        recent_hits = _average(recent) if recent else _EXPECTED_RANDOM_HITS_PER_DRAW
        evidence = min(evaluated / _CHAIN_EFFECTIVENESS_PRIOR_DRAWS, 1)
        blended_hits = (
            smoothed_hits * (1 - 0.25 * evidence)
            + recent_hits * 0.25 * evidence
        )
        return _clamp(
            blended_hits / _EXPECTED_RANDOM_HITS_PER_DRAW,
            _CHAIN_EFFECTIVENESS_MINIMUM,
            _CHAIN_EFFECTIVENESS_MAXIMUM,
        )

    @staticmethod
    def _consecutive_group_starts(
        numbers: Collection[int],
        size: int,
    ) -> frozenset[int]:
        drawn = set(numbers)
        return frozenset(
            start
            for start in range(1, _NUMBER_COUNT - size + 2)
            if all(start + offset in drawn for offset in range(size))
        )

    @classmethod
    def _doublet_triplet_shape_state(
        cls,
        numbers: Collection[int],
    ) -> int:
        if cls._consecutive_group_starts(numbers, 3):
            return 2
        if cls._consecutive_group_starts(numbers, 2):
            return 1
        return 0

    @staticmethod
    def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
        return sum(weight * feature for weight, feature in zip(weights, features))

    def train(self, drawn: set[int]) -> None:
        """Learn the current draw using only the state available before it."""
        if (
            "doublet_triplet_markov" in self.enabled_strategy_ids
            and self.previous_draw
        ):
            doublet_starts = self._consecutive_group_starts(drawn, 2)
            triplet_starts = self._consecutive_group_starts(drawn, 3)
            previous_state = self._doublet_triplet_shape_state(
                self.previous_draw
            )
            current_state = self._doublet_triplet_shape_state(drawn)
            self.doublet_triplet_shape_transitions[previous_state][
                current_state
            ] += 1
            self.doublet_triplet_shape_transition_totals[previous_state] += 1
            for previous_number in self.previous_draw:
                self.doublet_triplet_transition_totals[previous_number] += 1
                for start in doublet_starts:
                    self.doublet_markov_transitions[previous_number][start] += 1
                for start in triplet_starts:
                    self.triplet_markov_transitions[previous_number][start] += 1

        if "chained" in self.enabled_strategy_ids:
            self._train_chained_effectiveness(drawn)

        if "cis" in self.enabled_strategy_ids:
            self._train_cis(drawn)

        if "sklearn_svm" in self.enabled_strategy_ids:
            self._train_sklearn_svm(drawn)

        if "mkfr" in self.enabled_strategy_ids:
            for number in range(1, _NUMBER_COUNT + 1):
                target = int(number in drawn)
                history = self.mkfr_histories[number]
                context = 0
                for order in range(
                    1,
                    min(len(history), _MKFR_MAX_ORDER) + 1,
                ):
                    context |= history[-order] << (order - 1)
                    counts = self.mkfr_transitions[number][order - 1].setdefault(
                        context,
                        [0, 0],
                    )
                    counts[target] += 1

        if "mksp" in self.enabled_strategy_ids:
            for position, target in enumerate(_spaces_for_numbers(drawn)):
                history_values = tuple(self.mksp_histories[position])
                for order in range(
                    1,
                    min(len(history_values), _MKSP_MAX_ORDER) + 1,
                ):
                    context = history_values[-order:]
                    counts = self.mksp_transitions[position][
                        order - 1
                    ].setdefault(context, {})
                    counts[target] = counts.get(target, 0) + 1

        if "mknp" in self.enabled_strategy_ids:
            normalized = _normalized_positions_for_numbers(drawn)[1:]
            for position, target in enumerate(normalized):
                history_values = tuple(self.mknp_histories[position])
                for order in range(
                    1,
                    min(len(history_values), _MKSP_MAX_ORDER) + 1,
                ):
                    context = history_values[-order:]
                    counts = self.mknp_transitions[position][
                        order - 1
                    ].setdefault(context, {})
                    counts[target] = counts.get(target, 0) + 1

        if "mkrd" in self.enabled_strategy_ids:
            normalized = _normalized_positions_for_numbers(drawn)[1:]
            for position, target in enumerate(normalized):
                history_values = tuple(self.mkrd_histories[position])
                for order in range(
                    1,
                    min(len(history_values), _MKSP_MAX_ORDER) + 1,
                ):
                    context = history_values[-order:]
                    counts = self.mkrd_transitions[position][
                        order - 1
                    ].setdefault(context, {})
                    counts[target] = counts.get(target, 0) + 1

        gap_models_enabled = bool(
            self.enabled_strategy_ids.intersection({"markov100", "bayesian"})
        )
        if self.draw_count > 0 and "bayesian" in self.enabled_strategy_ids:
            for bucket in range(_MAX_GAP_BUCKET + 1):
                self.bayesian_recent_opportunities[bucket] *= _BAYESIAN_GAP_DECAY
                self.bayesian_recent_hits[bucket] *= _BAYESIAN_GAP_DECAY
            for number in range(1, _NUMBER_COUNT + 1):
                self.bayesian_recent_number_hits[
                    number
                ] *= _BAYESIAN_NUMBER_DECAY
        if self.draw_count > 0 and gap_models_enabled:
            for bucket in range(_MAX_GAP_BUCKET + 1):
                if "markov100" in self.enabled_strategy_ids:
                    self.markov_opportunities[bucket] *= _MARKOV_DECAY
                    self.markov_hits[bucket] *= _MARKOV_DECAY
            for number in range(1, _NUMBER_COUNT + 1):
                bucket = min(self._gap_before_current_draw(number), _MAX_GAP_BUCKET)
                if "markov100" in self.enabled_strategy_ids:
                    self.markov_opportunities[bucket] += 1
                if "bayesian" in self.enabled_strategy_ids:
                    self.bayesian_opportunities[bucket] += 1
                    self.bayesian_recent_opportunities[bucket] += 1
                if number in drawn:
                    if "markov100" in self.enabled_strategy_ids:
                        self.markov_hits[bucket] += 1
                    if "bayesian" in self.enabled_strategy_ids:
                        self.bayesian_hits[bucket] += 1
                        self.bayesian_recent_hits[bucket] += 1
        if "bayesian" in self.enabled_strategy_ids:
            for number in drawn:
                self.bayesian_recent_number_hits[number] += 1

        positive_weight = (_NUMBER_COUNT - _NUMBERS_PER_DRAW) / _NUMBERS_PER_DRAW
        if "svc" in self.enabled_strategy_ids:
            svc_rate = 0.08 / math.sqrt(self.draw_count + 1)
            for number in range(1, _NUMBER_COUNT + 1):
                label = 1 if number in drawn else -1
                features = self._svc_features(number)
                margin = label * self._dot(self.svc_weights, features)
                class_weight = positive_weight if label == 1 else 1
                for index, feature in enumerate(features):
                    self.svc_weights[index] *= 1 - svc_rate * 0.0008
                    if margin < 1:
                        self.svc_weights[index] += (
                            svc_rate * class_weight * label * feature
                        )

        if "tbl" in self.enabled_strategy_ids:
            tbl_rate = 0.09 / math.sqrt(self.draw_count + 1)
            for number in range(1, _NUMBER_COUNT + 1):
                features = self._tbl_features(number)
                target = float(number in drawn)
                predicted = _sigmoid(self._dot(self.tbl_weights, features))
                class_weight = positive_weight if target else 1
                error = (target - predicted) * class_weight
                for index, feature in enumerate(features):
                    self.tbl_weights[index] = (
                        self.tbl_weights[index] * (1 - tbl_rate * 0.0006)
                        + tbl_rate * error * feature
                    )

    def remember(self, drawn: set[int], draw_date: str | None = None) -> None:
        entropy_enabled = "entropy" in self.enabled_strategy_ids
        proximity_enabled = bool(
            self.enabled_strategy_ids.intersection({"proximity", "tbl"})
        )
        entropy_percent = _gap_entropy_percent(tuple(drawn)) if entropy_enabled else 0.0
        ordered = sorted(drawn)
        for index, number in enumerate(ordered):
            if proximity_enabled:
                distances = []
                if index > 0:
                    distances.append(number - ordered[index - 1])
                if index + 1 < len(ordered):
                    distances.append(ordered[index + 1] - number)
                bucket = _proximity_bucket(min(distances))
                self.proximity_counts[number][bucket] += 1
                self.proximity_totals[bucket] += 1
            self.appearances[number] += 1
            self.last_seen[number] = self.draw_count
            self.occurrences[number].append(self.draw_count)
            if entropy_enabled:
                self.entropy_totals[number] += entropy_percent
                if entropy_percent >= 92:
                    self.high_entropy_hits[number] += 1

        pair_history_enabled = bool(
            self.enabled_strategy_ids.intersection(
                {"co_occurrence", "predictive_grid", "tbl"}
            )
        )
        if pair_history_enabled:
            for left, right in combinations(ordered, 2):
                key = (left, right)
                self.pair_counts[key] = self.pair_counts.get(key, 0) + 1

        if "doublet_triplet_markov" in self.enabled_strategy_ids:
            doublet_starts = self._consecutive_group_starts(drawn, 2)
            triplet_starts = self._consecutive_group_starts(drawn, 3)
            for start in doublet_starts:
                self.doublet_markov_counts[start] += 1
            for start in triplet_starts:
                self.triplet_markov_counts[start] += 1
            self.doublet_triplet_shape_counts[
                self._doublet_triplet_shape_state(drawn)
            ] += 1
            self.doublet_triplet_recent_groups.append(
                (doublet_starts, triplet_starts)
            )

        if "predictive_grid" in self.enabled_strategy_ids and self.previous_draw:
            for previous in self.previous_draw:
                for current in drawn:
                    self.transition_counts[previous][current] += 1
                self.transition_totals[previous] += len(drawn)

        if self.enabled_strategy_ids.intersection(
            {
                "cis",
                "co_occurrence",
                "doublet_triplet_markov",
                "mksp",
                "predictive_grid",
                "tbl",
            }
        ):
            self.previous_previous_draw = self.previous_draw
            self.previous_draw = set(drawn)
        self.current_month = int(draw_date[5:7]) if draw_date else 0
        if self.enabled_strategy_ids.intersection(
            {"co_occurrence", "predictive_grid", "svc", "tbl"}
        ):
            self.recent_draws.append(drawn)
        if "emd" in self.enabled_strategy_ids:
            self.draw_vectors.append(tuple(sorted(drawn)))
        if "mkfr" in self.enabled_strategy_ids:
            for number in range(1, _NUMBER_COUNT + 1):
                self.mkfr_histories[number].append(int(number in drawn))
        if "mksp" in self.enabled_strategy_ids:
            spaces = _spaces_for_numbers(drawn)
            for position, space in enumerate(spaces):
                self.mksp_histories[position].append(space)
                self.mksp_value_counts[position][space] += 1
            anchor = min(drawn) - 1
            self.mksp_anchor_counts[anchor] += 1
            self.mksp_observations.append((spaces, anchor))
        if "mknp" in self.enabled_strategy_ids:
            normalized = _normalized_positions_for_numbers(drawn)
            for position, value in enumerate(normalized[1:]):
                self.mknp_histories[position].append(value)
                self.mknp_value_counts[position][value] += 1
            anchor = min(drawn) - 1
            self.mknp_anchor_counts[anchor] += 1
            self.mknp_observations.append((normalized, anchor))
        if "mkrd" in self.enabled_strategy_ids:
            normalized = _normalized_positions_for_numbers(drawn)
            for position, value in enumerate(normalized[1:]):
                self.mkrd_histories[position].append(value)
                self.mkrd_value_counts[position][value] += 1
            anchor = min(drawn) - 1
            self.mkrd_anchor_counts[anchor] += 1
            self.mkrd_observations.append(
                (normalized, anchor, _relative_dispersion_profile(drawn))
            )
        self.draw_count += 1

    @staticmethod
    def _earth_mover_distance(
        left: Sequence[int],
        right: Sequence[int],
    ) -> float:
        length = min(len(left), len(right))
        if length == 0:
            return 0.0
        return sum(abs(left[index] - right[index]) for index in range(length)) / length

    @staticmethod
    def _earth_mover_bucket(distance: float) -> str:
        if distance <= 1:
            return _EARTH_MOVER_BUCKETS[0]
        if distance <= 3:
            return _EARTH_MOVER_BUCKETS[1]
        if distance <= 5:
            return _EARTH_MOVER_BUCKETS[2]
        if distance <= 8:
            return _EARTH_MOVER_BUCKETS[3]
        if distance <= 12:
            return _EARTH_MOVER_BUCKETS[4]
        return _EARTH_MOVER_BUCKETS[5]

    def _earth_mover_scores(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        weighted_hits = {number: 0.0 for number in range(1, _NUMBER_COUNT + 1)}
        weighted_distances = {number: 0.0 for number in range(1, _NUMBER_COUNT + 1)}
        support = {number: 0 for number in range(1, _NUMBER_COUNT + 1)}
        if not self.draw_vectors:
            return weighted_hits, {}

        target = self.draw_vectors[-1]
        for index in range(len(self.draw_vectors) - 1):
            following = self.draw_vectors[index + 1]
            distance = self._earth_mover_distance(
                target,
                self.draw_vectors[index],
            )
            weight = 1 / (1 + distance)
            for number in following:
                weighted_hits[number] += weight
                weighted_distances[number] += distance * weight
                support[number] += 1

        maximum = max(weighted_hits.values(), default=0.0)
        scores = {
            number: (weighted_hits[number] / maximum if maximum > 0 else 0.0)
            for number in weighted_hits
        }
        details: dict[int, tuple[str, ...]] = {}
        for number in scores:
            average_distance = (
                weighted_distances[number] / weighted_hits[number]
                if weighted_hits[number] > 0
                else 0.0
            )
            label = (
                self._earth_mover_bucket(average_distance)
                if support[number] > 0
                else _EARTH_MOVER_BUCKETS[-1]
            )
            details[number] = (
                label,
                f"Average distance {average_distance:.2f}",
                f"Support draws {support[number]}",
            )
        return scores, details

    def _proximity_scores(self) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        total_numbers = max(self.draw_count * _NUMBERS_PER_DRAW, 1)
        shares = [count / total_numbers for count in self.proximity_totals]
        scores: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            counts = self.proximity_counts[number]
            scores[number] = sum(
                count * share for count, share in zip(counts, shares)
            ) / max(self.draw_count, 1)
            top_bucket = max(
                range(len(counts)), key=lambda index: (counts[index], -index)
            )
            details[number] = (
                _PROXIMITY_BUCKETS[top_bucket].title(),
                f"{self.appearances[number]} appearances",
            )
        return _scale_scores(scores), details

    def _entropy_scores(
        self, gaps: dict[int, int]
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raw: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            appearances = self.appearances[number]
            average = (
                50.0 if appearances == 0 else self.entropy_totals[number] / appearances
            )
            high_share = (
                0.0
                if appearances == 0
                else self.high_entropy_hits[number] / appearances
            )
            raw[number] = (
                average * 0.55
                + high_share * 100 * 0.30
                + _clamp(gaps[number] / 28, 0, 1) * 100 * 0.15
            )
            details[number] = (
                f"Average entropy {average:.1f}%",
                f"High-entropy share {high_share:.1%}",
            )
        return _scale_scores(raw), details

    def _chi_square_scores(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        expected = self.draw_count * _NUMBERS_PER_DRAW / _NUMBER_COUNT
        residuals: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            observed = self.appearances[number]
            difference = observed - expected
            residual = difference / math.sqrt(expected) if expected > 0 else 0.0
            contribution = difference**2 / expected if expected > 0 else 0.0
            if residual <= -2:
                band = "Strong under"
            elif residual <= -1:
                band = "Mild under"
            elif residual >= 2:
                band = "Strong over"
            elif residual >= 1:
                band = "Mild over"
            else:
                band = "Near expected"
            residuals[number] = residual
            details[number] = (
                band,
                f"Observed {observed} vs expected {expected:.2f}",
                f"Signed Pearson residual {residual:+.3f}",
                f"Chi-square contribution {contribution:.3f}",
            )
        return _scale_scores(residuals), details

    def _gap_model_scores(
        self,
        gaps: dict[int, int],
        *,
        weighted: bool,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        if weighted:
            probabilities = [
                (hit + _MARKOV_PRIOR_STRENGTH * _BASE_PROBABILITY)
                / (opportunity + _MARKOV_PRIOR_STRENGTH)
                for hit, opportunity in zip(
                    self.markov_hits,
                    self.markov_opportunities,
                )
            ]
            raw = {
                number: probabilities[min(gaps[number], _MAX_GAP_BUCKET)]
                for number in range(1, _NUMBER_COUNT + 1)
            }
            details: dict[int, tuple[str, ...]] = {
                number: (
                    f"Gap bucket {min(gaps[number], _MAX_GAP_BUCKET)}",
                    f"Posterior probability {raw[number]:.2%}",
                )
                for number in raw
            }
            return _scale_scores(raw), details

        lifetime_gap_probabilities = [
            (hit + _BAYESIAN_GAP_PRIOR_STRENGTH * _BASE_PROBABILITY)
            / (opportunity + _BAYESIAN_GAP_PRIOR_STRENGTH)
            for hit, opportunity in zip(
                self.bayesian_hits,
                self.bayesian_opportunities,
            )
        ]
        recent_gap_probabilities = [
            (hit + _BAYESIAN_GAP_PRIOR_STRENGTH * _BASE_PROBABILITY)
            / (opportunity + _BAYESIAN_GAP_PRIOR_STRENGTH)
            for hit, opportunity in zip(
                self.bayesian_recent_hits,
                self.bayesian_recent_opportunities,
            )
        ]
        effective_recent_draws = (
            sum(self.bayesian_recent_number_hits) / _NUMBERS_PER_DRAW
        )
        recent_number_probabilities = [
            (
                self.bayesian_recent_number_hits[number]
                + _BAYESIAN_NUMBER_PRIOR_STRENGTH * _BASE_PROBABILITY
            )
            / (effective_recent_draws + _BAYESIAN_NUMBER_PRIOR_STRENGTH)
            for number in range(_NUMBER_COUNT + 1)
        ]
        raw: dict[int, float] = {}
        details = {}
        for number in range(1, _NUMBER_COUNT + 1):
            bucket = min(gaps[number], _MAX_GAP_BUCKET)
            lifetime_gap = lifetime_gap_probabilities[bucket]
            recent_gap = recent_gap_probabilities[bucket]
            recent_number = recent_number_probabilities[number]
            raw[number] = (
                _BAYESIAN_LIFETIME_GAP_WEIGHT * lifetime_gap
                + _BAYESIAN_RECENT_GAP_WEIGHT * recent_gap
                + _BAYESIAN_RECENT_NUMBER_WEIGHT * recent_number
            )
            details[number] = (
                f"Model-averaged probability {raw[number]:.2%}",
                f"Gap bucket {bucket}",
                (
                    f"Lifetime gap {lifetime_gap:.2%} "
                    f"({_BAYESIAN_LIFETIME_GAP_WEIGHT:.0%} weight)"
                ),
                (
                    f"Recent gap {recent_gap:.2%} "
                    f"({_BAYESIAN_RECENT_GAP_WEIGHT:.0%} weight, "
                    f"{_BAYESIAN_GAP_RECENCY_HALF_LIFE}-draw half-life)"
                ),
                (
                    f"Recent number {recent_number:.2%} "
                    f"({_BAYESIAN_RECENT_NUMBER_WEIGHT:.0%} weight, "
                    f"{_BAYESIAN_NUMBER_RECENCY_HALF_LIFE}-draw half-life)"
                ),
                (
                    f"Hierarchical prior strengths "
                    f"{_BAYESIAN_GAP_PRIOR_STRENGTH:.0f} gap / "
                    f"{_BAYESIAN_NUMBER_PRIOR_STRENGTH:.0f} number"
                ),
            )
        return _scale_scores(raw), details

    def _mkfr_baseline_probability(self, number: int) -> float:
        return (self.appearances[number] + _MKFR_PRIOR_STRENGTH * _BASE_PROBABILITY) / (
            self.draw_count + _MKFR_PRIOR_STRENGTH
        )

    def _mkfr_probability(self, number: int) -> tuple[float, int, int]:
        history = self.mkfr_histories[number]
        active_orders = min(len(history), _MKFR_MAX_ORDER)
        probability = self._mkfr_baseline_probability(number)
        selected_support = 0
        selected_order = 0
        context = 0
        for order in range(1, active_orders + 1):
            context |= history[-order] << (order - 1)
            failures, hits = self.mkfr_transitions[number][order - 1].get(
                context,
                (0, 0),
            )
            opportunities = failures + hits
            if opportunities < _MKFR_MIN_CONTEXT_SUPPORT:
                continue
            probability = (hits + _MKFR_PRIOR_STRENGTH * probability) / (
                opportunities + _MKFR_PRIOR_STRENGTH
            )
            selected_support = opportunities
            selected_order = order
        return probability, selected_support, selected_order

    def _mkfr_scores(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raw: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            probability, support, selected_order = self._mkfr_probability(number)
            baseline = self._mkfr_baseline_probability(number)
            lift = probability - baseline
            context = "".join(str(outcome) for outcome in self.mkfr_histories[number])
            selected_context = context[-selected_order:] if selected_order > 0 else "—"
            raw[number] = lift
            details[number] = (
                f"Context probability {probability:.2%}",
                f"Baseline probability {baseline:.2%}",
                f"Transition lift {lift * 100:+.2f} pp",
                f"Order {selected_order}/{_MKFR_MAX_ORDER}: {selected_context}",
                f"Context support {support}",
            )
        return _scale_scores(raw), details

    def _mksp_baseline_probability(self, position: int, value: int) -> float:
        return (
            self.mksp_value_counts[position][value]
            + _MKSP_PRIOR_STRENGTH / _MKSP_VALUE_COUNT
        ) / (self.draw_count + _MKSP_PRIOR_STRENGTH)

    def _mksp_probability(
        self,
        position: int,
        value: int,
    ) -> tuple[float, int, int]:
        history_values = tuple(self.mksp_histories[position])
        active_orders = min(len(history_values), _MKSP_MAX_ORDER)
        probability = self._mksp_baseline_probability(position, value)
        selected_support = 0
        selected_order = 0
        for order in range(1, active_orders + 1):
            context = history_values[-order:]
            counts = self.mksp_transitions[position][order - 1].get(
                context,
                {},
            )
            opportunities = sum(counts.values())
            if opportunities < _MKSP_MIN_CONTEXT_SUPPORT:
                continue
            probability = (
                counts.get(value, 0) + _MKSP_PRIOR_STRENGTH * probability
            ) / (opportunities + _MKSP_PRIOR_STRENGTH)
            selected_support = opportunities
            selected_order = order
        return probability, selected_support, selected_order

    def _mksp_analogue_weights(
        self,
    ) -> list[tuple[tuple[int, ...], int, float]]:
        observation_count = len(self.mksp_observations)
        if observation_count < 2:
            return []

        weighted_analogues = []
        first_target = max(1, observation_count - _MKSP_ANALOGUE_LIMIT)
        maximum_space_distance = (
            _NUMBERS_PER_DRAW * (_MKSP_VALUE_COUNT - 1)
        )
        for target_index in range(first_target, observation_count):
            order = min(_MKSP_MAX_ORDER, target_index)
            weighted_distance = 0.0
            context_weight = 0.0
            for lag in range(1, order + 1):
                lag_weight = _MKSP_CONTEXT_DECAY ** (lag - 1)
                current_spaces = self.mksp_observations[-lag][0]
                historical_spaces = self.mksp_observations[target_index - lag][0]
                lag_distance = sum(
                    abs(current - historical)
                    for current, historical in zip(
                        current_spaces,
                        historical_spaces,
                    )
                ) / maximum_space_distance
                weighted_distance += lag_weight * lag_distance
                context_weight += lag_weight

            normalized_distance = weighted_distance / context_weight
            similarity = math.exp(
                -_MKSP_SIMILARITY_SHARPNESS * normalized_distance
            )
            age = observation_count - target_index
            recency = 0.5 ** (age / _MKSP_RECENCY_HALF_LIFE)
            length_confidence = 0.35 + 0.65 * order / _MKSP_MAX_ORDER
            spaces, anchor = self.mksp_observations[target_index]
            weighted_analogues.append(
                (spaces, anchor, similarity * recency * length_confidence)
            )
        return weighted_analogues

    @staticmethod
    def _mksp_normalize(values: Sequence[float]) -> tuple[float, ...]:
        total = sum(values)
        if total <= 0:
            return tuple(1 / len(values) for _value in values)
        return tuple(value / total for value in values)

    def _mksp_distributions(
        self,
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[float, ...],
        float,
        int,
        tuple[int, ...],
        tuple[int, ...],
    ]:
        analogues = self._mksp_analogue_weights()
        analogue_weight = sum(weight for _spaces, _anchor, weight in analogues)
        squared_weight = sum(
            weight * weight for _spaces, _anchor, weight in analogues
        )
        effective_support = (
            analogue_weight * analogue_weight / squared_weight
            if squared_weight > 0
            else 0.0
        )

        weighted_space_counts = [
            [0.0] * _MKSP_VALUE_COUNT for _position in range(_NUMBERS_PER_DRAW)
        ]
        weighted_anchor_counts = [0.0] * _MKSP_VALUE_COUNT
        for spaces, anchor, weight in analogues:
            for position, value in enumerate(spaces):
                weighted_space_counts[position][value] += weight
            weighted_anchor_counts[anchor] += weight

        space_distributions = []
        selected_orders = []
        selected_supports = []
        for position in range(_NUMBERS_PER_DRAW):
            baseline = self._mksp_normalize(
                tuple(
                    self._mksp_baseline_probability(position, value)
                    for value in range(_MKSP_VALUE_COUNT)
                )
            )
            analogue_distribution = self._mksp_normalize(
                tuple(
                    weighted_space_counts[position][value]
                    + _MKSP_ANALOGUE_PRIOR_STRENGTH * baseline[value]
                    for value in range(_MKSP_VALUE_COUNT)
                )
            )
            exact_distribution = self._mksp_normalize(
                tuple(
                    self._mksp_probability(position, value)[0]
                    for value in range(_MKSP_VALUE_COUNT)
                )
            )
            distribution = self._mksp_normalize(
                tuple(
                    _MKSP_ANALOGUE_BLEND * analogue_distribution[value]
                    + (1 - _MKSP_ANALOGUE_BLEND) * exact_distribution[value]
                    for value in range(_MKSP_VALUE_COUNT)
                )
            )
            selected_value = max(
                range(_MKSP_VALUE_COUNT),
                key=distribution.__getitem__,
            )
            _probability, support, order = self._mksp_probability(
                position,
                selected_value,
            )
            space_distributions.append(distribution)
            selected_orders.append(order)
            selected_supports.append(support)

        anchor_total = sum(self.mksp_anchor_counts)
        anchor_baseline = tuple(
            (
                self.mksp_anchor_counts[value]
                + _MKSP_ANALOGUE_PRIOR_STRENGTH / _MKSP_VALUE_COUNT
            )
            / (anchor_total + _MKSP_ANALOGUE_PRIOR_STRENGTH)
            for value in range(_MKSP_VALUE_COUNT)
        )
        anchor_distribution = self._mksp_normalize(
            tuple(
                weighted_anchor_counts[value]
                + _MKSP_ANALOGUE_PRIOR_STRENGTH * anchor_baseline[value]
                for value in range(_MKSP_VALUE_COUNT)
            )
        )
        return (
            tuple(space_distributions),
            anchor_distribution,
            effective_support,
            len(analogues),
            tuple(selected_orders),
            tuple(selected_supports),
        )

    @staticmethod
    def _mksp_internal_beam(
        distributions: Sequence[Sequence[float]],
    ) -> dict[int, list[tuple[float, tuple[int, ...]]]]:
        states: dict[int, list[tuple[float, tuple[int, ...]]]] = {
            0: [(0.0, ())]
        }
        for position in range(1, _NUMBERS_PER_DRAW):
            candidates: dict[int, list[tuple[float, tuple[int, ...]]]] = {}
            for total, paths in states.items():
                for log_probability, prefix in paths:
                    for value, probability in enumerate(distributions[position]):
                        next_total = total + value
                        if next_total > _MKSP_VALUE_COUNT - 1:
                            break
                        candidates.setdefault(next_total, []).append(
                            (
                                log_probability + math.log(probability),
                                (*prefix, value),
                            )
                        )
            states = {
                total: nlargest(
                    _MKSP_BEAM_WIDTH,
                    paths,
                    key=lambda path: (path[0], path[1]),
                )
                for total, paths in candidates.items()
            }
        return states

    def _mksp_scores(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        (
            distributions,
            anchor_distribution,
            effective_support,
            analogue_count,
            selected_orders,
            selected_supports,
        ) = self._mksp_distributions()
        internal_beam = self._mksp_internal_beam(distributions)
        generated: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for internal_total, paths in internal_beam.items():
            outer_space = (_MKSP_VALUE_COUNT - 1) - internal_total
            outer_probability = distributions[0][outer_space]
            for internal_log_probability, internal_spaces in paths:
                spaces = (outer_space, *internal_spaces)
                for anchor in range(outer_space + 1):
                    first_number = anchor + 1
                    numbers = [first_number]
                    for space in internal_spaces:
                        numbers.append(numbers[-1] + space + 1)
                    generated.append(
                        (
                            math.log(outer_probability)
                            + internal_log_probability
                            + math.log(anchor_distribution[anchor]),
                            tuple(numbers),
                            spaces,
                        )
                    )

        maximum_log_probability = max(
            log_probability for log_probability, _numbers, _spaces in generated
        )
        weighted_generated = [
            (
                math.exp(log_probability - maximum_log_probability),
                numbers,
                spaces,
            )
            for log_probability, numbers, spaces in generated
        ]
        total_weight = sum(
            weight for weight, _numbers, _spaces in weighted_generated
        )
        marginals = {
            number: 0.0 for number in range(1, _NUMBER_COUNT + 1)
        }
        best_contribution: dict[
            int,
            tuple[float, tuple[int, ...], tuple[int, ...]],
        ] = {}
        for weight, numbers, spaces in weighted_generated:
            for number in numbers:
                marginals[number] += weight
                previous_best = best_contribution.get(number)
                if previous_best is None or weight > previous_best[0]:
                    best_contribution[number] = (weight, numbers, spaces)

        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            marginal = marginals[number] / total_weight
            _weight, best_draw, best_spaces = best_contribution[number]
            details[number] = (
                f"Marginal probability {marginal:.2%}",
                f"Random baseline {_BASE_PROBABILITY:.2%}",
                f"Best generated draw {','.join(str(value) for value in best_draw)}",
                (
                    "Best six-space state "
                    f"{','.join(str(value) for value in best_spaces)} "
                    f"(sum {sum(best_spaces)})"
                ),
                (
                    f"Analogue support {effective_support:.1f} effective "
                    f"/ {analogue_count} candidates"
                ),
                (
                    f"Exact orders /{_MKSP_MAX_ORDER}: "
                    f"{','.join(str(order) for order in selected_orders)}; "
                    f"support {min(selected_supports)}–{max(selected_supports)}"
                ),
                f"Valid-draw beam width {_MKSP_BEAM_WIDTH}",
            )
        return _scale_scores(marginals), details

    @staticmethod
    def _mknp_valid_values(position: int) -> range:
        """Return valid normalized values for one of positions two through six."""
        ordinal = position + 2
        remaining_positions = _NUMBERS_PER_DRAW - ordinal
        return range(ordinal, _NUMBER_COUNT - remaining_positions + 1)

    def _mknp_baseline_probability(self, position: int, value: int) -> float:
        valid_values = self._mknp_valid_values(position)
        if value not in valid_values:
            return 0.0
        return (
            self.mknp_value_counts[position][value]
            + _MKSP_PRIOR_STRENGTH / len(valid_values)
        ) / (self.draw_count + _MKSP_PRIOR_STRENGTH)

    def _mknp_probability(
        self,
        position: int,
        value: int,
    ) -> tuple[float, int, int]:
        history_values = tuple(self.mknp_histories[position])
        active_orders = min(len(history_values), _MKSP_MAX_ORDER)
        probability = self._mknp_baseline_probability(position, value)
        selected_support = 0
        selected_order = 0
        for order in range(1, active_orders + 1):
            context = history_values[-order:]
            counts = self.mknp_transitions[position][order - 1].get(
                context,
                {},
            )
            opportunities = sum(counts.values())
            if opportunities < _MKSP_MIN_CONTEXT_SUPPORT:
                continue
            probability = (
                counts.get(value, 0) + _MKSP_PRIOR_STRENGTH * probability
            ) / (opportunities + _MKSP_PRIOR_STRENGTH)
            selected_support = opportunities
            selected_order = order
        return probability, selected_support, selected_order

    def _mknp_analogue_weights(
        self,
    ) -> list[tuple[tuple[int, ...], int, float]]:
        observation_count = len(self.mknp_observations)
        if observation_count < 2:
            return []

        weighted_analogues = []
        first_target = max(1, observation_count - _MKSP_ANALOGUE_LIMIT)
        maximum_position_distance = (
            _MKNP_POSITION_COUNT * (_NUMBER_COUNT - 1)
        )
        for target_index in range(first_target, observation_count):
            order = min(_MKSP_MAX_ORDER, target_index)
            weighted_distance = 0.0
            context_weight = 0.0
            for lag in range(1, order + 1):
                lag_weight = _MKSP_CONTEXT_DECAY ** (lag - 1)
                current_positions = self.mknp_observations[-lag][0]
                historical_positions = self.mknp_observations[
                    target_index - lag
                ][0]
                lag_distance = sum(
                    abs(current - historical)
                    for current, historical in zip(
                        current_positions[1:],
                        historical_positions[1:],
                    )
                ) / maximum_position_distance
                weighted_distance += lag_weight * lag_distance
                context_weight += lag_weight

            normalized_distance = weighted_distance / context_weight
            similarity = math.exp(
                -_MKSP_SIMILARITY_SHARPNESS * normalized_distance
            )
            age = observation_count - target_index
            recency = 0.5 ** (age / _MKSP_RECENCY_HALF_LIFE)
            length_confidence = 0.35 + 0.65 * order / _MKSP_MAX_ORDER
            positions, anchor = self.mknp_observations[target_index]
            weighted_analogues.append(
                (positions, anchor, similarity * recency * length_confidence)
            )
        return weighted_analogues

    def _mknp_distributions(
        self,
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[float, ...],
        float,
        int,
        tuple[int, ...],
        tuple[int, ...],
    ]:
        analogues = self._mknp_analogue_weights()
        analogue_weight = sum(weight for _positions, _anchor, weight in analogues)
        squared_weight = sum(
            weight * weight for _positions, _anchor, weight in analogues
        )
        effective_support = (
            analogue_weight * analogue_weight / squared_weight
            if squared_weight > 0
            else 0.0
        )

        weighted_position_counts = [
            [0.0] * (_NUMBER_COUNT + 1)
            for _position in range(_MKNP_POSITION_COUNT)
        ]
        weighted_anchor_counts = [0.0] * _MKSP_VALUE_COUNT
        for positions, anchor, weight in analogues:
            for position, value in enumerate(positions[1:]):
                weighted_position_counts[position][value] += weight
            weighted_anchor_counts[anchor] += weight

        position_distributions = []
        selected_orders = []
        selected_supports = []
        for position in range(_MKNP_POSITION_COUNT):
            baseline = self._mksp_normalize(
                tuple(
                    self._mknp_baseline_probability(position, value)
                    for value in range(_NUMBER_COUNT + 1)
                )
            )
            analogue_distribution = self._mksp_normalize(
                tuple(
                    weighted_position_counts[position][value]
                    + _MKSP_ANALOGUE_PRIOR_STRENGTH * baseline[value]
                    for value in range(_NUMBER_COUNT + 1)
                )
            )
            exact_distribution = self._mksp_normalize(
                tuple(
                    self._mknp_probability(position, value)[0]
                    for value in range(_NUMBER_COUNT + 1)
                )
            )
            distribution = self._mksp_normalize(
                tuple(
                    _MKSP_ANALOGUE_BLEND * analogue_distribution[value]
                    + (1 - _MKSP_ANALOGUE_BLEND) * exact_distribution[value]
                    for value in range(_NUMBER_COUNT + 1)
                )
            )
            selected_value = max(
                self._mknp_valid_values(position),
                key=distribution.__getitem__,
            )
            _probability, support, order = self._mknp_probability(
                position,
                selected_value,
            )
            position_distributions.append(distribution)
            selected_orders.append(order)
            selected_supports.append(support)

        anchor_total = sum(self.mknp_anchor_counts)
        anchor_baseline = tuple(
            (
                self.mknp_anchor_counts[value]
                + _MKSP_ANALOGUE_PRIOR_STRENGTH / _MKSP_VALUE_COUNT
            )
            / (anchor_total + _MKSP_ANALOGUE_PRIOR_STRENGTH)
            for value in range(_MKSP_VALUE_COUNT)
        )
        anchor_distribution = self._mksp_normalize(
            tuple(
                weighted_anchor_counts[value]
                + _MKSP_ANALOGUE_PRIOR_STRENGTH * anchor_baseline[value]
                for value in range(_MKSP_VALUE_COUNT)
            )
        )
        return (
            tuple(position_distributions),
            anchor_distribution,
            effective_support,
            len(analogues),
            tuple(selected_orders),
            tuple(selected_supports),
        )

    @classmethod
    def _mknp_shape_beam(
        cls,
        distributions: Sequence[Sequence[float]],
    ) -> dict[int, list[tuple[float, tuple[int, ...]]]]:
        states: dict[int, list[tuple[float, tuple[int, ...]]]] = {
            1: [(0.0, (1,))]
        }
        for position, distribution in enumerate(distributions):
            candidates: dict[int, list[tuple[float, tuple[int, ...]]]] = {}
            for previous_value, paths in states.items():
                for log_probability, prefix in paths:
                    for value in cls._mknp_valid_values(position):
                        if value <= previous_value:
                            continue
                        candidates.setdefault(value, []).append(
                            (
                                log_probability + math.log(distribution[value]),
                                (*prefix, value),
                            )
                        )
            states = {
                value: nlargest(
                    _MKSP_BEAM_WIDTH,
                    paths,
                    key=lambda path: (path[0], path[1]),
                )
                for value, paths in candidates.items()
            }
        return states

    def _mknp_scores(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        (
            distributions,
            anchor_distribution,
            effective_support,
            analogue_count,
            selected_orders,
            selected_supports,
        ) = self._mknp_distributions()
        shape_beam = self._mknp_shape_beam(distributions)
        generated: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for spread, paths in shape_beam.items():
            maximum_anchor = _NUMBER_COUNT - spread
            for shape_log_probability, positions in paths:
                for anchor in range(maximum_anchor + 1):
                    numbers = tuple(anchor + position for position in positions)
                    generated.append(
                        (
                            shape_log_probability
                            + math.log(anchor_distribution[anchor]),
                            numbers,
                            positions,
                        )
                    )

        maximum_log_probability = max(
            log_probability for log_probability, _numbers, _positions in generated
        )
        weighted_generated = [
            (
                math.exp(log_probability - maximum_log_probability),
                numbers,
                positions,
            )
            for log_probability, numbers, positions in generated
        ]
        total_weight = sum(
            weight for weight, _numbers, _positions in weighted_generated
        )
        marginals = {number: 0.0 for number in range(1, _NUMBER_COUNT + 1)}
        best_contribution: dict[
            int,
            tuple[float, tuple[int, ...], tuple[int, ...]],
        ] = {}
        for weight, numbers, positions in weighted_generated:
            for number in numbers:
                marginals[number] += weight
                previous_best = best_contribution.get(number)
                if previous_best is None or weight > previous_best[0]:
                    best_contribution[number] = (weight, numbers, positions)

        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            marginal = marginals[number] / total_weight
            _weight, best_draw, best_positions = best_contribution[number]
            spread = best_positions[-1]
            details[number] = (
                f"Marginal probability {marginal:.2%}",
                f"Random baseline {_BASE_PROBABILITY:.2%}",
                f"Best generated draw {','.join(str(value) for value in best_draw)}",
                (
                    "Best normalized positions "
                    f"{','.join(str(value) for value in best_positions)}"
                ),
                f"Spread {spread}; wraparound space {_NUMBER_COUNT - spread}",
                f"First-number anchor {best_draw[0]}",
                (
                    f"Analogue support {effective_support:.1f} effective "
                    f"/ {analogue_count} candidates"
                ),
                (
                    f"Exact orders /{_MKSP_MAX_ORDER}: "
                    f"{','.join(str(order) for order in selected_orders)}; "
                    f"support {min(selected_supports)}–{max(selected_supports)}"
                ),
                f"Valid-shape beam width {_MKSP_BEAM_WIDTH}",
            )
        return _scale_scores(marginals), details

    def _mkrd_baseline_probability(self, position: int, value: int) -> float:
        valid_values = self._mknp_valid_values(position)
        if value not in valid_values:
            return 0.0
        return (
            self.mkrd_value_counts[position][value]
            + _MKSP_PRIOR_STRENGTH / len(valid_values)
        ) / (self.draw_count + _MKSP_PRIOR_STRENGTH)

    def _mkrd_probability(
        self,
        position: int,
        value: int,
    ) -> tuple[float, int, int]:
        history_values = tuple(self.mkrd_histories[position])
        active_orders = min(len(history_values), _MKSP_MAX_ORDER)
        probability = self._mkrd_baseline_probability(position, value)
        selected_support = 0
        selected_order = 0
        for order in range(1, active_orders + 1):
            context = history_values[-order:]
            counts = self.mkrd_transitions[position][order - 1].get(
                context,
                {},
            )
            opportunities = sum(counts.values())
            if opportunities < _MKSP_MIN_CONTEXT_SUPPORT:
                continue
            probability = (
                counts.get(value, 0) + _MKSP_PRIOR_STRENGTH * probability
            ) / (opportunities + _MKSP_PRIOR_STRENGTH)
            selected_support = opportunities
            selected_order = order
        return probability, selected_support, selected_order

    @staticmethod
    def _mkrd_profile_distance(
        left: _RelativeDispersionProfile,
        right: _RelativeDispersionProfile,
    ) -> float:
        shape_distance = sum(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left.relative_positions[1:-1],
                right.relative_positions[1:-1],
            )
        ) / (_NUMBERS_PER_DRAW - 2)
        return (
            _MKRD_SHAPE_WEIGHT * shape_distance
            + _MKRD_COVERAGE_WEIGHT * abs(left.coverage - right.coverage)
            + _MKRD_UNIFORMITY_WEIGHT
            * abs(left.uniformity - right.uniformity)
            + _MKRD_ENTROPY_WEIGHT * abs(left.entropy - right.entropy)
            + _MKRD_CENTER_WEIGHT
            * abs(left.center_balance - right.center_balance)
        )

    def _mkrd_analogue_weights(
        self,
    ) -> list[
        tuple[tuple[int, ...], int, _RelativeDispersionProfile, float]
    ]:
        observation_count = len(self.mkrd_observations)
        if observation_count < 2:
            return []

        weighted_analogues = []
        first_target = max(1, observation_count - _MKSP_ANALOGUE_LIMIT)
        for target_index in range(first_target, observation_count):
            order = min(_MKSP_MAX_ORDER, target_index)
            weighted_distance = 0.0
            context_weight = 0.0
            for lag in range(1, order + 1):
                lag_weight = _MKSP_CONTEXT_DECAY ** (lag - 1)
                current_profile = self.mkrd_observations[-lag][2]
                historical_profile = self.mkrd_observations[
                    target_index - lag
                ][2]
                weighted_distance += lag_weight * self._mkrd_profile_distance(
                    current_profile,
                    historical_profile,
                )
                context_weight += lag_weight

            normalized_distance = weighted_distance / context_weight
            similarity = math.exp(
                -_MKSP_SIMILARITY_SHARPNESS * normalized_distance
            )
            age = observation_count - target_index
            recency = 0.5 ** (age / _MKSP_RECENCY_HALF_LIFE)
            length_confidence = 0.35 + 0.65 * order / _MKSP_MAX_ORDER
            positions, anchor, profile = self.mkrd_observations[target_index]
            weighted_analogues.append(
                (
                    positions,
                    anchor,
                    profile,
                    similarity * recency * length_confidence,
                )
            )
        return weighted_analogues

    def _mkrd_distributions(
        self,
    ) -> tuple[
        tuple[tuple[float, ...], ...],
        tuple[float, ...],
        float,
        int,
        tuple[int, ...],
        tuple[int, ...],
    ]:
        analogues = self._mkrd_analogue_weights()
        analogue_weight = sum(
            weight for _positions, _anchor, _profile, weight in analogues
        )
        squared_weight = sum(
            weight * weight
            for _positions, _anchor, _profile, weight in analogues
        )
        effective_support = (
            analogue_weight * analogue_weight / squared_weight
            if squared_weight > 0
            else 0.0
        )

        weighted_position_counts = [
            [0.0] * (_NUMBER_COUNT + 1)
            for _position in range(_MKNP_POSITION_COUNT)
        ]
        weighted_anchor_counts = [0.0] * _MKSP_VALUE_COUNT
        for positions, anchor, _profile, weight in analogues:
            for position, value in enumerate(positions[1:]):
                weighted_position_counts[position][value] += weight
            weighted_anchor_counts[anchor] += weight

        position_distributions = []
        selected_orders = []
        selected_supports = []
        for position in range(_MKNP_POSITION_COUNT):
            baseline = self._mksp_normalize(
                tuple(
                    self._mkrd_baseline_probability(position, value)
                    for value in range(_NUMBER_COUNT + 1)
                )
            )
            analogue_distribution = self._mksp_normalize(
                tuple(
                    weighted_position_counts[position][value]
                    + _MKSP_ANALOGUE_PRIOR_STRENGTH * baseline[value]
                    for value in range(_NUMBER_COUNT + 1)
                )
            )
            exact_distribution = self._mksp_normalize(
                tuple(
                    self._mkrd_probability(position, value)[0]
                    for value in range(_NUMBER_COUNT + 1)
                )
            )
            distribution = self._mksp_normalize(
                tuple(
                    _MKSP_ANALOGUE_BLEND * analogue_distribution[value]
                    + (1 - _MKSP_ANALOGUE_BLEND) * exact_distribution[value]
                    for value in range(_NUMBER_COUNT + 1)
                )
            )
            selected_value = max(
                self._mknp_valid_values(position),
                key=distribution.__getitem__,
            )
            _probability, support, order = self._mkrd_probability(
                position,
                selected_value,
            )
            position_distributions.append(distribution)
            selected_orders.append(order)
            selected_supports.append(support)

        anchor_total = sum(self.mkrd_anchor_counts)
        anchor_baseline = tuple(
            (
                self.mkrd_anchor_counts[value]
                + _MKSP_ANALOGUE_PRIOR_STRENGTH / _MKSP_VALUE_COUNT
            )
            / (anchor_total + _MKSP_ANALOGUE_PRIOR_STRENGTH)
            for value in range(_MKSP_VALUE_COUNT)
        )
        anchor_distribution = self._mksp_normalize(
            tuple(
                weighted_anchor_counts[value]
                + _MKSP_ANALOGUE_PRIOR_STRENGTH * anchor_baseline[value]
                for value in range(_MKSP_VALUE_COUNT)
            )
        )
        return (
            tuple(position_distributions),
            anchor_distribution,
            effective_support,
            len(analogues),
            tuple(selected_orders),
            tuple(selected_supports),
        )

    def _mkrd_scores(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        (
            distributions,
            anchor_distribution,
            effective_support,
            analogue_count,
            selected_orders,
            selected_supports,
        ) = self._mkrd_distributions()
        shape_beam = self._mknp_shape_beam(distributions)
        generated: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for spread, paths in shape_beam.items():
            maximum_anchor = _NUMBER_COUNT - spread
            for shape_log_probability, positions in paths:
                for anchor in range(maximum_anchor + 1):
                    numbers = tuple(anchor + position for position in positions)
                    generated.append(
                        (
                            shape_log_probability
                            + math.log(anchor_distribution[anchor]),
                            numbers,
                            positions,
                        )
                    )

        maximum_log_probability = max(
            log_probability for log_probability, _numbers, _positions in generated
        )
        weighted_generated = [
            (
                math.exp(log_probability - maximum_log_probability),
                numbers,
                positions,
            )
            for log_probability, numbers, positions in generated
        ]
        total_weight = sum(
            weight for weight, _numbers, _positions in weighted_generated
        )
        marginals = {number: 0.0 for number in range(1, _NUMBER_COUNT + 1)}
        best_contribution: dict[
            int,
            tuple[float, tuple[int, ...], tuple[int, ...]],
        ] = {}
        for weight, numbers, positions in weighted_generated:
            for number in numbers:
                marginals[number] += weight
                previous_best = best_contribution.get(number)
                if previous_best is None or weight > previous_best[0]:
                    best_contribution[number] = (weight, numbers, positions)

        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            marginal = marginals[number] / total_weight
            _weight, best_draw, _best_positions = best_contribution[number]
            profile = _relative_dispersion_profile(best_draw)
            details[number] = (
                f"Marginal probability {marginal:.2%}",
                f"Random baseline {_BASE_PROBABILITY:.2%}",
                f"Best generated draw {','.join(str(value) for value in best_draw)}",
                (
                    "Relative positions "
                    + ",".join(
                        f"{value:.3f}" for value in profile.relative_positions
                    )
                ),
                f"Span {profile.span}; coverage {profile.coverage:.2%}",
                (
                    "Gap shares "
                    + ",".join(f"{value:.3f}" for value in profile.gap_shares)
                ),
                f"Uniformity deviation {profile.uniformity:.3f}",
                f"Internal-gap entropy {profile.entropy:.3f}",
                f"Center balance {profile.center_balance:.3f}",
                f"First-number anchor {best_draw[0]}",
                (
                    f"Analogue support {effective_support:.1f} effective "
                    f"/ {analogue_count} candidates"
                ),
                (
                    f"Exact orders /{_MKSP_MAX_ORDER}: "
                    f"{','.join(str(order) for order in selected_orders)}; "
                    f"support {min(selected_supports)}–{max(selected_supports)}"
                ),
                f"Valid-shape beam width {_MKSP_BEAM_WIDTH}",
            )
        return _scale_scores(marginals), details

    @staticmethod
    def _combine_rankings(
        sources: Sequence[tuple[Sequence[int], float]],
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        weight_total = sum(weight for _ranking, weight in sources) or 1
        source_count = max(len(sources), 1)
        scores: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            ranks = [ranking.index(number) + 1 for ranking, _weight in sources]
            weighted_strength = sum(
                _rank_strength(ranking, number) * weight for ranking, weight in sources
            )
            agreement_count = sum(
                rank <= math.ceil(_NUMBER_COUNT * 0.25) for rank in ranks
            )
            top_six_count = sum(rank <= _NUMBERS_PER_DRAW for rank in ranks)
            agreement_bonus = agreement_count / source_count * 0.07
            top_six_bonus = top_six_count / source_count * 0.05
            scores[number] = (
                weighted_strength / weight_total + agreement_bonus + top_six_bonus
            )
            details[number] = (
                f"Top-quarter agreement {agreement_count}/{source_count}",
                f"Top-6 agreement {top_six_count}/{source_count}",
            )
        return scores, details

    @staticmethod
    def _residual_coverage_scores(
        rankings: dict[str, list[int]],
        gaps: dict[int, int],
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Rank numbers outside every base Top-6, preferring overdue candidates."""
        source_count = max(len(rankings), 1)
        maximum_gap = max(gaps.values(), default=1) or 1
        scores: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            ranks = [
                ranking.index(number) + 1
                for ranking in rankings.values()
            ]
            support = sum(rank <= _NUMBERS_PER_DRAW for rank in ranks)
            mean_rank = _average(ranks) if ranks else float(_NUMBER_COUNT)
            near_miss_strength = _average(
                [_rank_strength(ranking, number) for ranking in rankings.values()]
            )
            if support == 0:
                scores[number] = (
                    0.55
                    + 0.4498 * gaps[number] / maximum_gap
                    + 0.0001 * near_miss_strength
                )
                coverage_label = "Outside every base Top-6"
            else:
                scores[number] = (
                    0.43 * (1 - support / source_count)
                    + 0.01 * gaps[number] / maximum_gap
                )
                coverage_label = "Already covered by the base portfolio"
            details[number] = (
                f"Base Top-6 support {support}/{source_count}",
                coverage_label,
                f"Current gap {gaps[number]}",
                f"Average base rank {mean_rank:.1f}",
            )
        return scores, details

    def _chain_conditional_score(
        self,
        selected: Sequence[int],
        number: int,
    ) -> tuple[float, float]:
        if not selected:
            return 0.5, 1.0
        prior_strength = _CO_OCCURRENCE_PRIOR_STRENGTH
        baseline = (
            self.appearances[number] + prior_strength * _BASE_PROBABILITY
        ) / (self.draw_count + prior_strength)
        log_lifts: list[float] = []
        for seed in selected:
            pair_count = self.pair_counts.get(tuple(sorted((seed, number))), 0)
            conditional = (
                pair_count + prior_strength * baseline
            ) / (self.appearances[seed] + prior_strength)
            log_lifts.append(
                math.log(max(conditional, 1e-12) / max(baseline, 1e-12))
            )
        average_log_lift = _average(log_lifts)
        return _sigmoid(average_log_lift), math.exp(average_log_lift)

    @staticmethod
    def _chain_shape_score(
        selected: Sequence[int],
        number: int,
    ) -> float:
        proposed = tuple((*selected, number))
        count = len(proposed)
        if count <= 1:
            return 0.5
        ordered = sorted(proposed)
        odd_count = sum(value % 2 for value in proposed)
        parity_difference = abs(odd_count - (count - odd_count))
        parity_score = 1 - max(parity_difference - count % 2, 0) / count
        span = ordered[-1] - ordered[0]
        expected_span = 48 * (count - 1) / (count + 1)
        span_score = math.exp(-abs(span - expected_span) / max(expected_span, 1))
        minimum_gap = min(
            right - left for left, right in zip(ordered, ordered[1:])
        )
        spacing_score = min(minimum_gap / 5, 1)
        decade_count = len({(value - 1) // 10 for value in proposed})
        coverage_score = decade_count / min(count, 5)
        entropy_score = (
            _gap_entropy_percent(proposed) / 100
            if count == _NUMBERS_PER_DRAW
            else 0.5
        )
        return _clamp(
            0.25 * parity_score
            + 0.25 * span_score
            + 0.20 * spacing_score
            + 0.15 * coverage_score
            + 0.15 * entropy_score,
            0,
            1,
        )

    def _chained_scores(
        self,
        rankings: dict[str, list[int]],
        gaps: dict[int, int],
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Greedily chain consensus, relationships, shape, and residual coverage."""
        expert_rankings = {
            strategy_id: rankings[strategy_id]
            for strategy_id in _CHAIN_EXPERT_IDS
            if strategy_id in rankings
        }
        self.chain_pending_rankings = {
            strategy_id: list(ranking)
            for strategy_id, ranking in expert_rankings.items()
        }
        if not expert_rankings:
            neutral = {number: 0.0 for number in range(1, _NUMBER_COUNT + 1)}
            return neutral, {}

        rank_maps = {
            strategy_id: {
                number: rank for rank, number in enumerate(ranking, start=1)
            }
            for strategy_id, ranking in expert_rankings.items()
        }
        weights = {
            strategy_id: self._chain_expert_weight(strategy_id)
            for strategy_id in expert_rankings
        }
        total_weight = sum(weights.values()) or 1.0
        source_count = len(expert_rankings)
        maximum_gap = max(gaps.values(), default=1) or 1

        def components(
            number: int,
            selected: Sequence[int],
        ) -> tuple[float, float, float, float, int, str, float]:
            strengths = {
                strategy_id: (
                    _NUMBER_COUNT - rank_maps[strategy_id][number]
                ) / (_NUMBER_COUNT - 1)
                for strategy_id in expert_rankings
            }
            top_six_support = sum(
                rank_maps[strategy_id][number] <= _NUMBERS_PER_DRAW
                for strategy_id in expert_rankings
            )
            weighted_strength = sum(
                strengths[strategy_id] * weights[strategy_id]
                for strategy_id in expert_rankings
            ) / total_weight
            consensus = (
                0.92 * weighted_strength
                + 0.08 * top_six_support / source_count
            )
            conditional, relationship_lift = self._chain_conditional_score(
                selected,
                number,
            )
            shape = (
                0.5
                if len(selected) >= _NUMBERS_PER_DRAW
                else self._chain_shape_score(selected, number)
            )
            residual = (
                0.70 * (1 - top_six_support / source_count)
                + 0.30 * gaps[number] / maximum_gap
            )
            strongest_expert = max(
                expert_rankings,
                key=lambda strategy_id: (
                    strengths[strategy_id] * weights[strategy_id],
                    strategy_id,
                ),
            )
            return (
                consensus,
                conditional,
                shape,
                residual,
                top_six_support,
                strongest_expert,
                relationship_lift,
            )

        stage_weights = (
            (0.78, 0.07, 0.12, 0.03),
            (0.62, 0.18, 0.15, 0.05),
            (0.62, 0.18, 0.15, 0.05),
            (0.58, 0.19, 0.15, 0.08),
            (0.52, 0.20, 0.13, 0.15),
            (0.44, 0.20, 0.12, 0.24),
        )
        ranking: list[int] = []
        raw_scores: dict[int, float] = {}
        component_rows: dict[
            int,
            tuple[float, float, float, float, int, str, float],
        ] = {}
        available = set(range(1, _NUMBER_COUNT + 1))
        for step, weights_for_step in enumerate(stage_weights, start=1):
            candidate_scores: dict[int, float] = {}
            for number in available:
                row = components(number, ranking)
                candidate_scores[number] = sum(
                    component * weight
                    for component, weight in zip(row[:4], weights_for_step)
                )
            selected_number = max(
                candidate_scores,
                key=lambda candidate: (
                    candidate_scores[candidate],
                    gaps[candidate],
                    -candidate,
                ),
            )
            ranking.append(selected_number)
            available.remove(selected_number)
            raw_scores[selected_number] = candidate_scores[selected_number]
            component_rows[selected_number] = components(
                selected_number,
                ranking[:-1],
            )

        reserve_scores: dict[int, float] = {}
        for number in available:
            row = components(number, ranking)
            reserve_scores[number] = (
                0.65 * row[0] + 0.20 * row[1] + 0.15 * row[3]
            )
            raw_scores[number] = reserve_scores[number]
            component_rows[number] = row
        ranking.extend(
            sorted(
                available,
                key=lambda number: (
                    -reserve_scores[number],
                    -gaps[number],
                    number,
                ),
            )
        )

        scores = {
            number: (_NUMBER_COUNT - rank) / (_NUMBER_COUNT - 1)
            for rank, number in enumerate(ranking, start=1)
        }
        details: dict[int, tuple[str, ...]] = {}
        for rank, number in enumerate(ranking, start=1):
            (
                consensus,
                conditional,
                shape,
                residual,
                top_six_support,
                strongest_expert,
                relationship_lift,
            ) = component_rows[number]
            details[number] = (
                (
                    f"Chain pick {rank}/{_NUMBERS_PER_DRAW}"
                    if rank <= _NUMBERS_PER_DRAW
                    else f"Reserve rank {rank}"
                ),
                f"Effectiveness-weighted consensus {consensus:.1%}",
                (
                    f"Conditional relationship {conditional:.1%} "
                    f"({relationship_lift:.2f}× lift)"
                ),
                f"Draw-shape fit {shape:.1%}",
                f"Residual coverage {residual:.1%}",
                f"Base Top-6 support {top_six_support}/{source_count}",
                (
                    f"Strongest expert {strongest_expert} "
                    f"({weights[strongest_expert]:.2f}× weight)"
                ),
                f"Stage score {raw_scores[number]:.1%}",
                f"Effectiveness history {self.chain_evaluated_draws} draws",
            )
        return scores, details

    def _predictive_grid_scores(
        self,
        gaps: dict[int, int],
        earth_mover_scores: dict[int, float],
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        gap_probabilities = [
            (hit + _MARKOV_PRIOR_STRENGTH * _BASE_PROBABILITY)
            / (opportunity + _MARKOV_PRIOR_STRENGTH)
            for hit, opportunity in zip(
                self.markov_hits,
                self.markov_opportunities,
            )
        ]
        markov_raw: dict[int, float] = {}
        transition_raw: dict[int, float] = {}
        frequency_raw: dict[int, float] = {}
        recent_raw: dict[int, float] = {}
        gap_raw: dict[int, float] = {}
        pair_raw: dict[int, float] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            bucket = min(gaps[number], _MAX_GAP_BUCKET)
            markov_raw[number] = gap_probabilities[bucket]
            transition_values = [
                self.transition_counts[previous][number]
                / self.transition_totals[previous]
                if self.transition_totals[previous]
                else 0.0
                for previous in self.previous_draw
            ]
            transition_raw[number] = _average(transition_values)
            frequency_raw[number] = self.appearances[number] / max(
                self.draw_count,
                1,
            )
            recent_raw[number] = float(self._recent_count(number, 20))
            gap_raw[number] = float(gaps[number])
            affinities = [
                float(
                    self.pair_counts.get(
                        tuple(sorted((number, previous))),
                        0,
                    )
                )
                for previous in self.previous_draw
                if previous != number
            ]
            pair_raw[number] = _average(affinities)

        components = {
            "markov": _scale_scores(markov_raw),
            "transition": _scale_scores(transition_raw),
            "frequency": _scale_scores(frequency_raw),
            "recent": _scale_scores(recent_raw),
            "gap": _scale_scores(gap_raw),
            "pair": _scale_scores(pair_raw),
        }
        history_weight = 1 - _PREDICTIVE_GRID_EMD_WEIGHT
        scores = {
            number: (
                history_weight
                * (
                    0.35 * components["markov"][number]
                    + 0.20 * components["transition"][number]
                    + 0.15 * components["frequency"][number]
                    + 0.15 * components["recent"][number]
                    + 0.10 * components["gap"][number]
                    + 0.05 * components["pair"][number]
                )
                + _PREDICTIVE_GRID_EMD_WEIGHT * earth_mover_scores[number]
            )
            for number in range(1, _NUMBER_COUNT + 1)
        }
        details: dict[int, tuple[str, ...]] = {
            number: (
                f"Gap-state Markov {components['markov'][number]:.1%}",
                f"Last-draw transition {components['transition'][number]:.1%}",
                f"Lifetime frequency {components['frequency'][number]:.1%}",
                f"Recent-20 activity {components['recent'][number]:.1%}",
                f"Current gap {components['gap'][number]:.1%}",
                f"Pair affinity {components['pair'][number]:.1%}",
                f"Earth-mover similarity {earth_mover_scores[number]:.1%}",
            )
            for number in range(1, _NUMBER_COUNT + 1)
        }
        return scores, details

    def _doublet_triplet_markov_scores(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Rank numbers through consecutive-group recurrence and Markov transitions."""
        prior_strength = _DOUBLET_TRIPLET_MARKOV_PRIOR_STRENGTH
        recent_length = len(self.doublet_triplet_recent_groups)

        def group_components(
            size: int,
            lifetime_counts: Sequence[int],
            transitions: Sequence[Sequence[int]],
            recent_index: int,
        ) -> tuple[
            dict[int, float],
            dict[int, float],
            dict[int, float],
            dict[int, float],
        ]:
            starts = range(1, _NUMBER_COUNT - size + 2)
            baseline_probability = (
                math.comb(_NUMBER_COUNT - size, _NUMBERS_PER_DRAW - size)
                / math.comb(_NUMBER_COUNT, _NUMBERS_PER_DRAW)
            )
            lifetime = {
                start: (
                    lifetime_counts[start]
                    + prior_strength * baseline_probability
                )
                / (self.draw_count + prior_strength)
                for start in starts
            }
            recent = {
                start: (
                    sum(
                        start in groups[recent_index]
                        for groups in self.doublet_triplet_recent_groups
                    )
                    + prior_strength * lifetime[start]
                )
                / (recent_length + prior_strength)
                for start in starts
            }
            conditional: dict[int, float] = {}
            for start in starts:
                if self.previous_draw:
                    conditional[start] = _average(
                        [
                            (
                                transitions[previous][start]
                                + prior_strength * lifetime[start]
                            )
                            / (
                                self.doublet_triplet_transition_totals[previous]
                                + prior_strength
                            )
                            for previous in self.previous_draw
                        ]
                    )
                else:
                    conditional[start] = lifetime[start]
            lifetime_scaled = _scale_scores(lifetime)
            recent_scaled = _scale_scores(recent)
            conditional_scaled = _scale_scores(conditional)
            combined = {
                start: (
                    0.30 * lifetime_scaled[start]
                    + 0.22 * recent_scaled[start]
                    + 0.48 * conditional_scaled[start]
                )
                for start in starts
            }
            return combined, lifetime, recent, conditional

        (
            doublet_scores,
            doublet_lifetime,
            doublet_recent,
            doublet_conditional,
        ) = group_components(
            2,
            self.doublet_markov_counts,
            self.doublet_markov_transitions,
            0,
        )
        (
            triplet_scores,
            triplet_lifetime,
            triplet_recent,
            triplet_conditional,
        ) = group_components(
            3,
            self.triplet_markov_counts,
            self.triplet_markov_transitions,
            1,
        )

        previous_state = self._doublet_triplet_shape_state(self.previous_draw)
        shape_total = sum(self.doublet_triplet_shape_counts)
        shape_prior = [
            (
                self.doublet_triplet_shape_counts[state] / shape_total
                if shape_total
                else (0.65, 0.30, 0.05)[state]
            )
            for state in range(3)
        ]
        transition_total = (
            self.doublet_triplet_shape_transition_totals[previous_state]
            if self.previous_draw
            else 0
        )
        next_shape_probabilities = [
            (
                self.doublet_triplet_shape_transitions[previous_state][state]
                + prior_strength * shape_prior[state]
            )
            / (transition_total + prior_strength)
            for state in range(3)
        ]
        next_doublet_probability = (
            next_shape_probabilities[1] + next_shape_probabilities[2]
        )
        next_triplet_probability = next_shape_probabilities[2]
        doublet_shape_weight = 0.55 + 0.45 * next_doublet_probability
        triplet_shape_weight = 0.45 + 0.55 * next_triplet_probability

        raw_scores: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            containing_doublets = [
                start
                for start in doublet_scores
                if start <= number < start + 2
            ]
            containing_triplets = [
                start
                for start in triplet_scores
                if start <= number < start + 3
            ]
            strongest_doublet = max(
                containing_doublets,
                key=lambda start: (doublet_scores[start], -start),
                default=0,
            )
            strongest_triplet = max(
                containing_triplets,
                key=lambda start: (triplet_scores[start], -start),
                default=0,
            )
            doublet_support = (
                0.70
                * max(
                    (doublet_scores[start] for start in containing_doublets),
                    default=0.0,
                )
                + 0.30
                * _average(
                    [doublet_scores[start] for start in containing_doublets]
                )
            )
            triplet_support = (
                0.68
                * max(
                    (triplet_scores[start] for start in containing_triplets),
                    default=0.0,
                )
                + 0.32
                * _average(
                    [triplet_scores[start] for start in containing_triplets]
                )
            )
            raw_scores[number] = (
                0.46 * doublet_shape_weight * doublet_support
                + 0.54 * triplet_shape_weight * triplet_support
            )
            details[number] = (
                (
                    f"Strongest doublet {strongest_doublet}-"
                    f"{strongest_doublet + 1}: "
                    f"{doublet_scores[strongest_doublet]:.1%} group score, "
                    f"{doublet_conditional[strongest_doublet]:.2%} Markov"
                    if strongest_doublet
                    else "No consecutive doublet support"
                ),
                (
                    f"Strongest triplet {strongest_triplet}-"
                    f"{strongest_triplet + 1}-"
                    f"{strongest_triplet + 2}: "
                    f"{triplet_scores[strongest_triplet]:.1%} group score, "
                    f"{triplet_conditional[strongest_triplet]:.2%} Markov"
                    if strongest_triplet
                    else "No consecutive triplet support"
                ),
                (
                    f"Next shape: doublet {next_doublet_probability:.1%}, "
                    f"triplet {next_triplet_probability:.1%}"
                ),
                (
                    f"Recent window {recent_length}; "
                    f"conditioned on {len(self.previous_draw)} prior numbers"
                ),
                (
                    f"Lifetime/recent best: "
                    f"{doublet_lifetime.get(strongest_doublet, 0):.2%}/"
                    f"{doublet_recent.get(strongest_doublet, 0):.2%} doublet, "
                    f"{triplet_lifetime.get(strongest_triplet, 0):.2%}/"
                    f"{triplet_recent.get(strongest_triplet, 0):.2%} triplet"
                ),
            )

        candidate_groups = [
            (
                score * triplet_shape_weight,
                3,
                start,
            )
            for start, score in triplet_scores.items()
        ] + [
            (
                score * doublet_shape_weight,
                2,
                start,
            )
            for start, score in doublet_scores.items()
        ]
        selected: set[int] = set()
        for group_score, size, start in sorted(
            candidate_groups,
            key=lambda item: (-item[0], -item[1], item[2]),
        ):
            if group_score <= 0:
                break
            members = set(range(start, start + size))
            new_members = sorted(
                members.difference(selected),
                key=lambda member: (-raw_scores[member], member),
            )
            admitted = set(
                new_members[: _NUMBERS_PER_DRAW - len(selected)]
            )
            for member in members.intersection(selected).union(admitted):
                raw_scores[member] += 0.30 * group_score
            selected.update(admitted)
            if len(selected) == _NUMBERS_PER_DRAW:
                break

        return _scale_scores(raw_scores), details

    def _co_occurrence_scores(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        recent_draws = tuple(
            list(self.recent_draws)[-_CO_OCCURRENCE_RECENT_WINDOW:]
        )
        recent_length = len(recent_draws)
        recent_appearances = {
            number: sum(number in draw for draw in recent_draws)
            for number in range(1, _NUMBER_COUNT + 1)
        }
        expected_pair_count = (
            self.draw_count
            * math.comb(_NUMBERS_PER_DRAW, 2)
            / math.comb(_NUMBER_COUNT, 2)
        )
        raw_scores: dict[int, float] = {}
        legacy_scores: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            baseline = (
                self.appearances[number]
                + _CO_OCCURRENCE_PRIOR_STRENGTH * _BASE_PROBABILITY
            ) / (self.draw_count + _CO_OCCURRENCE_PRIOR_STRENGTH)
            recent_baseline = (
                recent_appearances[number]
                + _CO_OCCURRENCE_PRIOR_STRENGTH * baseline
            ) / (recent_length + _CO_OCCURRENCE_PRIOR_STRENGTH)
            partner_evidence: list[tuple[int, float, float, int, int]] = []
            for partner in sorted(
                previous for previous in self.previous_draw if previous != number
            ):
                pair_count = self.pair_counts.get(
                    tuple(sorted((number, partner))),
                    0,
                )
                lifetime_conditional = (
                    pair_count
                    + _CO_OCCURRENCE_PRIOR_STRENGTH * baseline
                ) / (
                    self.appearances[partner]
                    + _CO_OCCURRENCE_PRIOR_STRENGTH
                )
                lifetime_log_lift = math.log(
                    max(lifetime_conditional, 1e-12)
                    / max(baseline, 1e-12)
                )
                recent_pair_count = sum(
                    number in draw and partner in draw
                    for draw in recent_draws
                )
                recent_conditional = (
                    recent_pair_count
                    + _CO_OCCURRENCE_PRIOR_STRENGTH * recent_baseline
                ) / (
                    recent_appearances[partner]
                    + _CO_OCCURRENCE_PRIOR_STRENGTH
                )
                recent_log_lift = math.log(
                    max(recent_conditional, 1e-12)
                    / max(recent_baseline, 1e-12)
                )
                blended_log_lift = (
                    (1 - _CO_OCCURRENCE_RECENT_WEIGHT)
                    * lifetime_log_lift
                    + _CO_OCCURRENCE_RECENT_WEIGHT
                    * recent_log_lift
                )
                partner_evidence.append(
                    (
                        partner,
                        blended_log_lift,
                        lifetime_log_lift,
                        pair_count,
                        recent_pair_count,
                    )
                )

            average_log_lift = _average(
                [evidence[1] for evidence in partner_evidence]
            )
            total_pair_count = sum(
                evidence[3] for evidence in partner_evidence
            )
            legacy_average_lift = _average(
                [
                    evidence[3] / expected_pair_count
                    if expected_pair_count
                    else 0.0
                    for evidence in partner_evidence
                ]
            )
            positive_partners = sum(
                evidence[1] > 0 for evidence in partner_evidence
            )
            (
                strongest_partner,
                strongest_log_lift,
                _strongest_lifetime_log_lift,
                strongest_count,
                strongest_recent_count,
            ) = max(
                partner_evidence,
                key=lambda evidence: (evidence[1], -evidence[0]),
                default=(0, 0.0, 0.0, 0, 0),
            )
            raw_scores[number] = average_log_lift
            legacy_scores[number] = (
                legacy_average_lift * 70
                + total_pair_count / max(self.draw_count, 1) * 30
            )
            details[number] = (
                f"Adjusted average lift {math.exp(average_log_lift):.2f}×",
                (
                    f"Positive latest-draw partners "
                    f"{positive_partners}/{len(partner_evidence)}"
                ),
                f"Recent window {recent_length} draws",
                (
                    f"Strongest partner {strongest_partner}: "
                    f"{math.exp(strongest_log_lift):.2f}× lift, "
                    f"{strongest_count} lifetime / "
                    f"{strongest_recent_count} recent pairs"
                    if strongest_partner
                    else "No latest-draw partner history yet"
                ),
                "Bayesian-smoothed conditional association",
            )
        adjusted = _scale_scores(raw_scores)
        legacy = _scale_scores(legacy_scores)
        scores = {
            number: (
                _CO_OCCURRENCE_ADJUSTED_WEIGHT * adjusted[number]
                + (1 - _CO_OCCURRENCE_ADJUSTED_WEIGHT) * legacy[number]
            )
            for number in range(1, _NUMBER_COUNT + 1)
        }
        return scores, details

    def _cis_scores(
        self,
        rankings: dict[str, list[int]],
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        rank_maps = {
            strategy_id: {number: rank for rank, number in enumerate(ranking, start=1)}
            for strategy_id, ranking in rankings.items()
        }
        prior_rank_maps = {
            strategy_id: {number: rank for rank, number in enumerate(ranking, start=1)}
            for strategy_id, ranking in self.cis_prior_rankings.items()
        }
        dynamic_weights = {
            strategy_id: self._cis_expert_weight(
                strategy_id,
                base_weight,
            )
            for strategy_id, _label, base_weight in _CIS_EXPERTS
        }
        recent_accuracies = {
            strategy_id: self._cis_expert_accuracy(strategy_id, 20)
            for strategy_id, _label, _weight in _CIS_EXPERTS
        }
        long_term_accuracies = {
            strategy_id: (
                0.0
                if self.cis_evaluated_draws[strategy_id] == 0
                else self.cis_total_hits[strategy_id]
                / (self.cis_evaluated_draws[strategy_id] * _NUMBERS_PER_DRAW)
            )
            for strategy_id, _label, _weight in _CIS_EXPERTS
        }
        self.cis_pending_rankings = {
            strategy_id: list(rankings[strategy_id])
            for strategy_id, _label, _weight in _CIS_EXPERTS
            if strategy_id in rankings
        }
        self.cis_pending_features = {
            number: self._cis_features(
                number,
                rankings,
                rank_maps,
                prior_rank_maps,
                dynamic_weights,
                recent_accuracies,
                long_term_accuracies,
            )
            for number in range(1, _NUMBER_COUNT + 1)
        }
        self.cis_pending_ensemble_scores = {}
        self.cis_pending_learner_scores = {
            number: self._cis_probability(features)
            for number, features in self.cis_pending_features.items()
        }
        learner_scores = _scale_scores(self.cis_pending_learner_scores)
        learner_blend = self._cis_learner_blend()
        scores: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number, features in self.cis_pending_features.items():
            contributions = [
                (
                    strategy_id,
                    (_NUMBER_COUNT - rank_maps[strategy_id][number])
                    / (_NUMBER_COUNT - 1)
                    * dynamic_weights[strategy_id],
                )
                for strategy_id, _label, _base_weight in _CIS_EXPERTS
                if strategy_id in rankings
            ]
            contribution_total = sum(
                contribution for _strategy_id, contribution in contributions
            )
            dynamic_weight_total = sum(
                dynamic_weights[strategy_id]
                for strategy_id, _label, _base_weight in _CIS_EXPERTS
                if strategy_id in rankings
            )
            ensemble_score = contribution_total / max(
                dynamic_weight_total,
                1e-12,
            )
            self.cis_pending_ensemble_scores[number] = ensemble_score
            probability = self.cis_pending_learner_scores[number]
            scores[number] = (
                (1 - learner_blend) * ensemble_score
                + learner_blend * learner_scores[number]
            )
            supporters = sorted(
                contributions,
                key=lambda item: (-item[1], item[0]),
            )[:3]
            supporter_text = ", ".join(
                strategy_id for strategy_id, _contribution in supporters
            )
            details[number] = (
                f"Adaptive ensemble {1 - learner_blend:.0%}",
                f"Guarded ranking learner {learner_blend:.0%} ({probability:.2%})",
                f"Strongest experts: {supporter_text}",
                f"Consensus {features[7]:.0%}",
                f"Opposition {features[9]:.0%}",
            )
        return _scale_scores(scores), details

    def build_strategies(
        self,
        combined: CombinedPrediction,
        draw_index: int,
    ) -> tuple[StrategyPrediction, ...]:
        gaps = self.current_gaps()
        enabled = self.enabled_strategy_ids
        requested = self.requested_strategy_ids
        built: dict[str, StrategyPrediction] = {}
        rankings: dict[str, list[int]] = {}

        freshness_scores: dict[int, float] = {}
        if "freshness" in enabled:
            by_number = {item.number: item for item in combined.numbers}
            freshness_scores = _scale_scores(
                {
                    number: by_number[number].freshness_score
                    for number in range(1, _NUMBER_COUNT + 1)
                }
            )
            rankings["freshness"] = _ranking_from_scores(
                freshness_scores,
                gaps,
            )
            if "freshness" in requested:
                freshness_details: dict[int, tuple[str, ...]] = {
                    number: (
                        f"Gap {gaps[number]}",
                        f"Hit probability {by_number[number].freshness_score:.2%}",
                    )
                    for number in range(1, _NUMBER_COUNT + 1)
                }
                built["freshness"] = _strategy(
                    "freshness",
                    "Fresh",
                    "PyLotto gap recency and historical hit-rate model.",
                    freshness_scores,
                    gaps,
                    freshness_details,
                )

        proximity_scores: dict[int, float] = {}
        if "proximity" in enabled:
            proximity_scores, proximity_details = self._proximity_scores()
            rankings["proximity"] = _ranking_from_scores(
                proximity_scores,
                gaps,
            )
            if "proximity" in requested:
                built["proximity"] = _strategy(
                    "proximity",
                    "Prox",
                    "PyLotto nearest-neighbor spacing profile.",
                    proximity_scores,
                    gaps,
                    proximity_details,
                )

        earth_mover_scores: dict[int, float] = {}
        if "emd" in enabled:
            earth_mover_scores, earth_mover_details = self._earth_mover_scores()
            rankings["emd"] = _ranking_from_scores(
                earth_mover_scores,
                gaps,
            )
            if "emd" in requested:
                built["emd"] = _strategy(
                    "emd",
                    "EMD",
                    "Earth-mover analogue ranking from historical draw vectors.",
                    earth_mover_scores,
                    gaps,
                    earth_mover_details,
                )

        random_ranking: list[int] = []
        if "randomness" in enabled:
            random_ranking = _random_ranking(draw_index + 1)
            randomness_scores = {
                number: (_NUMBER_COUNT - rank) / (_NUMBER_COUNT - 1)
                for rank, number in enumerate(random_ranking, start=1)
            }
            rankings["randomness"] = random_ranking
            if "randomness" in requested:
                randomness_details: dict[int, tuple[str, ...]] = {
                    number: ("Deterministic PyLotto baseline",)
                    for number in randomness_scores
                }
                built["randomness"] = _strategy(
                    "randomness",
                    "Rand",
                    "Deterministic random baseline used for comparison.",
                    randomness_scores,
                    gaps,
                    randomness_details,
                )

        if "entropy" in enabled:
            entropy_scores, entropy_details = self._entropy_scores(gaps)
            rankings["entropy"] = _ranking_from_scores(
                entropy_scores,
                gaps,
            )
            if "entropy" in requested:
                built["entropy"] = _strategy(
                    "entropy",
                    "Entr",
                    "Structural gap-entropy history with overdue adjustment.",
                    entropy_scores,
                    gaps,
                    entropy_details,
                )

        if "chi_square" in enabled:
            chi_square_scores, chi_square_details = self._chi_square_scores()
            rankings["chi_square"] = _ranking_from_scores(
                chi_square_scores,
                gaps,
            )
            if "chi_square" in requested:
                built["chi_square"] = _strategy(
                    "chi_square",
                    "Chi²",
                    "Signed Pearson residual from the uniform 6/49 frequency expectation.",
                    chi_square_scores,
                    gaps,
                    chi_square_details,
                )

        if "markov100" in enabled:
            markov_scores, markov_details = self._gap_model_scores(gaps, weighted=True)
            rankings["markov100"] = _ranking_from_scores(
                markov_scores,
                gaps,
            )
            if "markov100" in requested:
                built["markov100"] = _strategy(
                    "markov100",
                    "Mark",
                    "Recency-weighted gap-state Markov model on a 0–100 scale.",
                    markov_scores,
                    gaps,
                    markov_details,
                )

        if "mkfr" in enabled:
            mkfr_scores, mkfr_details = self._mkfr_scores()
            rankings["mkfr"] = _ranking_from_scores(
                mkfr_scores,
                gaps,
            )
            if "mkfr" in requested:
                built["mkfr"] = _strategy(
                    "mkfr",
                    "MKFR",
                    "Per-number variable-order D/!D context model ranked by transition lift.",
                    mkfr_scores,
                    gaps,
                    mkfr_details,
                )

        if "mksp" in enabled:
            mksp_scores, mksp_details = self._mksp_scores()
            rankings["mksp"] = _ranking_from_scores(
                mksp_scores,
                gaps,
            )
            if "mksp" in requested:
                built["mksp"] = _strategy(
                    "mksp",
                    "MKSP",
                    (
                        "Analogue-weighted six-position Markov spaces through "
                        "order 20, decoded into complete valid draws."
                    ),
                    mksp_scores,
                    gaps,
                    mksp_details,
                )

        if "mknp" in enabled:
            mknp_scores, mknp_details = self._mknp_scores()
            rankings["mknp"] = _ranking_from_scores(
                mknp_scores,
                gaps,
            )
            if "mknp" in requested:
                built["mknp"] = _strategy(
                    "mknp",
                    "MKNP",
                    (
                        "Order-20 normalized-position analogues decoded into "
                        "valid translated draws."
                    ),
                    mknp_scores,
                    gaps,
                    mknp_details,
                )

        if "mkrd" in enabled:
            mkrd_scores, mkrd_details = self._mkrd_scores()
            rankings["mkrd"] = _ranking_from_scores(
                mkrd_scores,
                gaps,
            )
            if "mkrd" in requested:
                built["mkrd"] = _strategy(
                    "mkrd",
                    "MKRD",
                    (
                        "Order-20 relative-shape and dispersion analogues "
                        "decoded into valid translated draws."
                    ),
                    mkrd_scores,
                    gaps,
                    mkrd_details,
                )

        if "bayesian" in enabled:
            bayesian_scores, bayesian_details = self._gap_model_scores(
                gaps, weighted=False
            )
            rankings["bayesian"] = _ranking_from_scores(
                bayesian_scores,
                gaps,
            )
            if "bayesian" in requested:
                built["bayesian"] = _strategy(
                    "bayesian",
                    "Baye",
                    (
                        "Hierarchically shrunk Bayesian model average of "
                        "lifetime gap, recent gap, and recent number posteriors."
                    ),
                    bayesian_scores,
                    gaps,
                    bayesian_details,
                )

        if "svc" in enabled:
            svc_margins = {
                number: self._dot(self.svc_weights, self._svc_features(number))
                for number in range(1, _NUMBER_COUNT + 1)
            }
            svc_scores = _scale_scores(svc_margins)
            svc_details: dict[int, tuple[str, ...]] = {
                number: (
                    f"Margin {svc_margins[number]:.3f}",
                    f"Recent 8: {self._recent_count(number, 8)}",
                    f"Recent 24: {self._recent_count(number, 24)}",
                )
                for number in svc_scores
            }
            rankings["svc"] = _ranking_from_scores(
                svc_scores,
                gaps,
            )
            if "svc" in requested:
                built["svc"] = _strategy(
                    "svc",
                    "SVC",
                    "Online linear support-vector classifier inspired by PyLotto.",
                    svc_scores,
                    gaps,
                    svc_details,
                )

        if "tbl" in enabled:
            self.prior_rankings = {
                "freshness": rankings["freshness"],
                "proximity": rankings["proximity"],
                "randomness": random_ranking,
            }
            tbl_raw: dict[int, float] = {}
            tbl_details: dict[int, tuple[str, ...]] = {}
            for number in range(1, _NUMBER_COUNT + 1):
                features = self._tbl_features(number)
                linear = self._dot(self.tbl_weights, features)
                nonlinear = (
                    math.tanh(features[6] * 1.4) * 0.2
                    + math.tanh(features[10] * 4) * 0.16
                    + math.tanh(features[11] * 30) * 0.12
                    + math.tanh((features[12] + features[13] - 1) * 1.5) * 0.16
                )
                probability = _sigmoid(linear + nonlinear)
                tbl_raw[number] = probability
                tbl_details[number] = (
                    f"Probability {probability:.2%}",
                    f"Lifetime frequency {self.appearances[number] / max(self.draw_count, 1):.2%}",
                    f"Recent 20: {self._recent_count(number, 20)}",
                )
            tbl_scores = _scale_scores(tbl_raw)
            rankings["tbl"] = _ranking_from_scores(
                tbl_scores,
                gaps,
            )
            if "tbl" in requested:
                built["tbl"] = _strategy(
                    "tbl",
                    "TBL",
                    "Temporal Behavior Learning with recency, frequency, and strategy features.",
                    tbl_scores,
                    gaps,
                    tbl_details,
                )

        if "fresh_random" in enabled:
            fresh_random_ranking = _random_ranking(
                draw_index + 1,
                _RANDOM_SEED + _FRESH_RANDOM_SEED_OFFSET,
            )
            random_rank = {
                number: rank
                for rank, number in enumerate(
                    fresh_random_ranking,
                    start=1,
                )
            }
            freshness_rank = {
                number: rank
                for rank, number in enumerate(
                    rankings["freshness"],
                    start=1,
                )
            }
            fresh_random_scores = {
                number: (
                    (_NUMBER_COUNT - random_rank[number])
                    / (_NUMBER_COUNT - 1)
                    * (1 - _FRESH_RANDOM_INFLUENCE)
                    + (_NUMBER_COUNT - freshness_rank[number])
                    / (_NUMBER_COUNT - 1)
                    * _FRESH_RANDOM_INFLUENCE
                )
                for number in range(1, _NUMBER_COUNT + 1)
            }
            rankings["fresh_random"] = sorted(
                fresh_random_scores,
                key=lambda number: (
                    -fresh_random_scores[number],
                    freshness_rank[number],
                    number,
                ),
            )
            if "fresh_random" in requested:
                fresh_random_details: dict[int, tuple[str, ...]] = {
                    number: (
                        f"Random rank {random_rank[number]}",
                        f"Freshness rank {freshness_rank[number]}",
                        "Blend 65% random / 35% freshness",
                    )
                    for number in range(1, _NUMBER_COUNT + 1)
                }
                built["fresh_random"] = _strategy(
                    "fresh_random",
                    "FRnd",
                    "Seeded random ranking softly guided by freshness.",
                    fresh_random_scores,
                    gaps,
                    fresh_random_details,
                )

        if "mixed" in enabled:
            mixed_scores, mixed_details = self._combine_rankings(
                (
                    (rankings["freshness"], 0.30),
                    (rankings["proximity"], 0.24),
                    (rankings["emd"], 0.14),
                    (rankings["bayesian"], 0.32),
                )
            )
            rankings["mixed"] = _ranking_from_scores(mixed_scores, gaps)
            if "mixed" in requested:
                built["mixed"] = _strategy(
                    "mixed",
                    "Mix",
                    "Weighted PyLotto consensus of Freshness, Proximity, EMD, and Bayesian.",
                    mixed_scores,
                    gaps,
                    mixed_details,
                )

        if "predictive_grid" in enabled:
            grid_scores, grid_details = self._predictive_grid_scores(
                gaps,
                earth_mover_scores,
            )
            rankings["predictive_grid"] = _ranking_from_scores(
                grid_scores,
                gaps,
            )
            if "predictive_grid" in requested:
                built["predictive_grid"] = _strategy(
                    "predictive_grid",
                    "Grid",
                    (
                        "Seven-component score grid blending historical "
                        "signals with earth-mover draw similarity."
                    ),
                    grid_scores,
                    gaps,
                    grid_details,
                )

        if "co_occurrence" in enabled:
            co_occurrence_scores, co_occurrence_details = (
                self._co_occurrence_scores()
            )
            rankings["co_occurrence"] = _ranking_from_scores(
                co_occurrence_scores,
                gaps,
            )
            if "co_occurrence" in requested:
                built["co_occurrence"] = _strategy(
                    "co_occurrence",
                    "CoOc",
                    (
                        "Pair-count ranking stabilized with candidate-adjusted, "
                        "Bayesian-smoothed lifetime and recent lift."
                    ),
                    co_occurrence_scores,
                    gaps,
                    co_occurrence_details,
                )

        if "doublet_triplet_markov" in enabled:
            (
                doublet_triplet_markov_scores,
                doublet_triplet_markov_details,
            ) = self._doublet_triplet_markov_scores()
            rankings["doublet_triplet_markov"] = _ranking_from_scores(
                doublet_triplet_markov_scores,
                gaps,
            )
            if "doublet_triplet_markov" in requested:
                built["doublet_triplet_markov"] = _strategy(
                    "doublet_triplet_markov",
                    "Doublet & Triplet Markov",
                    (
                        "First-order Markov model of consecutive doublets and "
                        "triplets, blended with lifetime and recent recurrence."
                    ),
                    doublet_triplet_markov_scores,
                    gaps,
                    doublet_triplet_markov_details,
                )

        if "sklearn_svm" in enabled:
            sklearn_svm_scores, sklearn_svm_details = self._sklearn_svm_scores(
                rankings
            )
            rankings["sklearn_svm"] = _ranking_from_scores(
                sklearn_svm_scores,
                gaps,
            )
            if "sklearn_svm" in requested:
                built["sklearn_svm"] = _strategy(
                    "sklearn_svm",
                    "Scikit Online SVM",
                    (
                        "Scikit-learn online linear SVM using temporal, frequency, "
                        "expert-rank, and leakage-safe efficacy inputs."
                    ),
                    sklearn_svm_scores,
                    gaps,
                    sklearn_svm_details,
                )

        if "cis" in enabled:
            cis_scores, cis_details = self._cis_scores(rankings)
            rankings["cis"] = _ranking_from_scores(cis_scores, gaps)
            if "cis" in requested:
                built["cis"] = _strategy(
                    "cis",
                    "CIS",
                    (
                        "Collective Intelligence Strategy v2: a leakage-free, "
                        "performance-weighted expert portfolio with a guarded "
                        "online ranking correction."
                    ),
                    cis_scores,
                    gaps,
                    cis_details,
                )

        if "residual_coverage" in enabled:
            displayed_rankings = {
                strategy_id: (
                    [item.number for item in built[strategy_id].numbers]
                    if strategy_id in built
                    else ranking
                )
                for strategy_id, ranking in rankings.items()
                if strategy_id not in {"mknp", "mkrd"}
            }
            residual_scores, residual_details = self._residual_coverage_scores(
                displayed_rankings,
                gaps,
            )
            rankings["residual_coverage"] = _ranking_from_scores(
                residual_scores,
                gaps,
            )
            if "residual_coverage" in requested:
                built["residual_coverage"] = _strategy(
                    "residual_coverage",
                    "RCOV",
                    (
                        "Diversity-first ensemble complement selecting numbers "
                        "outside every base Top-6, with longest current gaps first."
                    ),
                    residual_scores,
                    gaps,
                    residual_details,
                )

        if "chained" in enabled:
            chained_scores, chained_details = self._chained_scores(
                rankings,
                gaps,
            )
            rankings["chained"] = _ranking_from_scores(chained_scores, gaps)
            if "chained" in requested:
                built["chained"] = _strategy(
                    "chained",
                    "Chained Strategy",
                    (
                        "Sequential leakage-safe chain of effectiveness-weighted "
                        "consensus, conditional relationships, draw-shape fit, "
                        "and residual coverage."
                    ),
                    chained_scores,
                    gaps,
                    chained_details,
                )

        return tuple(
            built[strategy_id] for strategy_id in STRATEGY_IDS if strategy_id in built
        )


def build_prediction_suites(
    draws: Sequence[Draw],
    *,
    history_start: int = 0,
    enabled_strategy_ids: Collection[str] = STRATEGY_IDS,
    progress: PredictionProgress | None = None,
    efficacy_record: EfficacyRecordCallback | None = None,
    evaluated_suite: PredictionSuiteCallback | None = None,
) -> tuple[PredictionSuite, ...]:
    """Evaluate all draws and retain only the requested display history."""
    requested = set(enabled_strategy_ids)
    unknown = requested.difference(STRATEGY_IDS)
    if unknown:
        raise ValueError(
            f"Unknown prediction strategy plugin(s): {', '.join(sorted(unknown))}"
        )
    selected = tuple(
        strategy_id for strategy_id in STRATEGY_IDS if strategy_id in requested
    )
    state = _StrategyState(selected, total_draw_count=len(draws))
    efficacy = _EfficacyTracker()
    suites: list[PredictionSuite] = []
    total = len(draws)
    for draw_index, draw in enumerate(draws):
        drawn = {ball.value for ball in draw.balls}
        if selected:
            state.train(drawn)
            state.remember(drawn, draw.date)
            combined = draw.prediction
            if combined is None:
                raise ValueError("Combined predictions must be prepared first")
            actual = (
                tuple(ball.value for ball in draws[draw_index + 1].balls)
                if draw_index + 1 < total
                else ()
            )
            compared, record = efficacy.compare(
                PredictionSuite(
                    reference_draw_number=draw_index + 1,
                    target_draw_number=draw_index + 2,
                    actual_numbers=actual,
                    strategies=state.build_strategies(combined, draw_index),
                )
            )
            if record is not None and efficacy_record is not None:
                efficacy_record(record)
            if record is not None and evaluated_suite is not None:
                evaluated_suite(compared)
            if draw_index >= history_start:
                suites.append(compared)
        else:
            actual = (
                tuple(ball.value for ball in draws[draw_index + 1].balls)
                if draw_index + 1 < total
                else ()
            )
            empty_suite = PredictionSuite(
                reference_draw_number=draw_index + 1,
                target_draw_number=draw_index + 2,
                actual_numbers=actual,
                strategies=(),
            )
            if actual and evaluated_suite is not None:
                evaluated_suite(empty_suite)
            if draw_index >= history_start:
                suites.append(empty_suite)
        if progress is not None:
            progress(draw_index + 1, total)
    return tuple(suites)
