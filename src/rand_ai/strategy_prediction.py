"""Build display-ready PyLotto-inspired strategy prediction histories."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from itertools import combinations

from rand_ai.draw import Draw
from rand_ai.prediction import CombinedPrediction

_NUMBER_COUNT = 49
_NUMBERS_PER_DRAW = 6
_BASE_PROBABILITY = _NUMBERS_PER_DRAW / _NUMBER_COUNT
_EXPECTED_RANDOM_HITS_PER_DRAW = _NUMBERS_PER_DRAW * _NUMBERS_PER_DRAW / _NUMBER_COUNT
_MAX_GAP_BUCKET = 35
_MARKOV_PRIOR_STRENGTH = 8.0
_MARKOV_DECAY = 0.5 ** (1 / 500)
_MKFR_MAX_ORDER = 20
_MKFR_PRIOR_STRENGTH = 8.0
_MKFR_MIN_CONTEXT_SUPPORT = 8
_RANDOM_SEED = 20260626
_FRESH_RANDOM_SEED_OFFSET = 7919
_FRESH_RANDOM_INFLUENCE = 0.35
_CIS_MINIMUM_TRAINING_DRAWS = 36
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
STRATEGY_IDS = (
    "proximity",
    "freshness",
    "emd",
    "randomness",
    "fresh_random",
    "chi_square",
    "entropy",
    "markov100",
    "mkfr",
    "bayesian",
    "predictive_grid",
    "mixed",
    "svc",
    "tbl",
    "cis",
)
_CIS_EXPERTS = (
    ("freshness", "Freshness", 0.15),
    ("proximity", "Proximity", 0.12),
    ("emd", "EMD", 0.10),
    ("bayesian", "Bayesian", 0.15),
    ("markov100", "100 Markov", 0.10),
    ("mixed", "Mixed", 0.12),
    ("randomness", "Randomness", 0.04),
    ("fresh_random", "Fresh Random", 0.05),
    ("svc", "SVC", 0.08),
    ("tbl", "TBL", 0.09),
)
_STRATEGY_DEPENDENCIES = {
    "fresh_random": {"freshness", "randomness"},
    "mixed": {"freshness", "proximity", "emd", "bayesian"},
    "predictive_grid": {"markov100"},
    "cis": {
        "freshness",
        "proximity",
        "emd",
        "bayesian",
        "markov100",
        "mixed",
        "randomness",
        "fresh_random",
        "svc",
        "tbl",
    },
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


class _StrategyState:
    """Maintain incremental state for the enabled prediction strategy plugins."""

    def __init__(
        self,
        enabled_strategy_ids: Collection[str] = STRATEGY_IDS,
        total_draw_count: int = 1,
    ) -> None:
        self.requested_strategy_ids = frozenset(enabled_strategy_ids)
        active_strategy_ids = set(self.requested_strategy_ids)
        for strategy_id in tuple(active_strategy_ids):
            active_strategy_ids.update(_STRATEGY_DEPENDENCIES.get(strategy_id, ()))
        self.enabled_strategy_ids = frozenset(active_strategy_ids)
        self.total_draw_count = max(total_draw_count, 1)
        self.draw_count = 0
        self.appearances = [0] * (_NUMBER_COUNT + 1)
        self.last_seen: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.occurrences: list[list[int]] = [[] for _ in range(_NUMBER_COUNT + 1)]
        self.recent_draws: deque[set[int]] = deque(maxlen=100)
        self.pair_counts: dict[tuple[int, int], int] = {}
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
        self.svc_weights = [0.0] * 11
        self.tbl_weights = [0.0] * 14
        self.cis_weights = [0.0] * (22 + len(_CIS_EXPERTS) * 4)
        self.cis_draw_count = 0
        self.cis_total_hits = {
            strategy_id: 0 for strategy_id, _label, _weight in _CIS_EXPERTS
        }
        self.cis_evaluated_draws = {
            strategy_id: 0 for strategy_id, _label, _weight in _CIS_EXPERTS
        }
        self.cis_recent_hits = {
            strategy_id: deque(maxlen=100)
            for strategy_id, _label, _weight in _CIS_EXPERTS
        }
        self.cis_prior_rankings: dict[str, list[int]] = {}
        self.cis_pending_rankings: dict[str, list[int]] = {}
        self.cis_pending_features: dict[int, tuple[float, ...]] = {}
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

    def _cis_expert_accuracy(
        self,
        strategy_id: str,
        window_size: int,
    ) -> float:
        recent = list(self.cis_recent_hits[strategy_id])[-window_size:]
        return _average(recent) / _NUMBERS_PER_DRAW if recent else 0.0

    def _cis_expert_weight(
        self,
        strategy_id: str,
        base_weight: float,
    ) -> float:
        evaluated = self.cis_evaluated_draws[strategy_id]
        long_term = (
            _BASE_PROBABILITY
            if evaluated == 0
            else self.cis_total_hits[strategy_id] / (evaluated * _NUMBERS_PER_DRAW)
        )
        return base_weight * (
            0.45
            + self._cis_expert_accuracy(strategy_id, 20) * 1.4
            + self._cis_expert_accuracy(strategy_id, 50) * 0.8
            + long_term * 0.9
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
            self.draw_count / self.total_draw_count,
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
        learning_rate = 0.08 / math.sqrt(self.cis_draw_count + 1)
        positive_weight = (_NUMBER_COUNT - _NUMBERS_PER_DRAW) / _NUMBERS_PER_DRAW
        for number, features in self.cis_pending_features.items():
            target = float(number in drawn)
            predicted = self._cis_probability(features)
            error = (target - predicted) * (positive_weight if target else 1)
            for index, feature in enumerate(features):
                self.cis_weights[index] = (
                    self.cis_weights[index] * (1 - learning_rate * 0.0006)
                    + learning_rate * error * feature
                )

        for strategy_id, ranking in self.cis_pending_rankings.items():
            hits = len(drawn.intersection(ranking[:_NUMBERS_PER_DRAW]))
            self.cis_total_hits[strategy_id] += hits
            self.cis_evaluated_draws[strategy_id] += 1
            self.cis_recent_hits[strategy_id].append(hits)
            self.cis_prior_rankings[strategy_id] = list(ranking)
        self.cis_draw_count += 1

    @staticmethod
    def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
        return sum(weight * feature for weight, feature in zip(weights, features))

    def train(self, drawn: set[int]) -> None:
        """Learn the current draw using only the state available before it."""
        if "cis" in self.enabled_strategy_ids:
            self._train_cis(drawn)

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

        gap_models_enabled = bool(
            self.enabled_strategy_ids.intersection({"markov100", "bayesian"})
        )
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
                if number in drawn:
                    if "markov100" in self.enabled_strategy_ids:
                        self.markov_hits[bucket] += 1
                    if "bayesian" in self.enabled_strategy_ids:
                        self.bayesian_hits[bucket] += 1

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

        grid_or_tbl_enabled = bool(
            self.enabled_strategy_ids.intersection({"predictive_grid", "tbl"})
        )
        if grid_or_tbl_enabled:
            for left, right in combinations(ordered, 2):
                key = (left, right)
                self.pair_counts[key] = self.pair_counts.get(key, 0) + 1

        if "predictive_grid" in self.enabled_strategy_ids and self.previous_draw:
            for previous in self.previous_draw:
                for current in drawn:
                    self.transition_counts[previous][current] += 1
                self.transition_totals[previous] += len(drawn)

        if self.enabled_strategy_ids.intersection({"cis", "predictive_grid", "tbl"}):
            self.previous_previous_draw = self.previous_draw
            self.previous_draw = set(drawn)
        self.current_month = int(draw_date[5:7]) if draw_date else 0
        if self.enabled_strategy_ids.intersection({"predictive_grid", "svc", "tbl"}):
            self.recent_draws.append(drawn)
        if "emd" in self.enabled_strategy_ids:
            self.draw_vectors.append(tuple(sorted(drawn)))
        if "mkfr" in self.enabled_strategy_ids:
            for number in range(1, _NUMBER_COUNT + 1):
                self.mkfr_histories[number].append(int(number in drawn))
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
        opportunities = (
            self.markov_opportunities if weighted else self.bayesian_opportunities
        )
        hits = self.markov_hits if weighted else self.bayesian_hits
        probabilities = [
            (hit + _MARKOV_PRIOR_STRENGTH * _BASE_PROBABILITY)
            / (opportunity + _MARKOV_PRIOR_STRENGTH)
            for hit, opportunity in zip(hits, opportunities)
        ]
        raw = {
            number: probabilities[min(gaps[number], _MAX_GAP_BUCKET)]
            for number in range(1, _NUMBER_COUNT + 1)
        }
        scaled = _scale_scores(raw)
        details: dict[int, tuple[str, ...]] = {
            number: (
                f"Gap bucket {min(gaps[number], _MAX_GAP_BUCKET)}",
                f"Posterior probability {raw[number]:.2%}",
            )
            for number in raw
        }
        return scaled, details

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

    def _predictive_grid_scores(
        self,
        gaps: dict[int, int],
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
            frequency_raw[number] = self.appearances[number] / max(self.draw_count, 1)
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
        scores = {
            number: (
                0.35 * components["markov"][number]
                + 0.20 * components["transition"][number]
                + 0.15 * components["frequency"][number]
                + 0.15 * components["recent"][number]
                + 0.10 * components["gap"][number]
                + 0.05 * components["pair"][number]
            )
            for number in range(1, _NUMBER_COUNT + 1)
        }
        details = {
            number: (
                f"Gap-state Markov {components['markov'][number]:.1%}",
                f"Last-draw transition {components['transition'][number]:.1%}",
                f"Lifetime frequency {components['frequency'][number]:.1%}",
                f"Recent-20 activity {components['recent'][number]:.1%}",
                f"Current gap {components['gap'][number]:.1%}",
                f"Pair affinity {components['pair'][number]:.1%}",
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
        warm_up = self.cis_draw_count < _CIS_MINIMUM_TRAINING_DRAWS
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
            probability = self._cis_probability(features)
            if warm_up:
                contribution_total = sum(
                    contribution for _strategy_id, contribution in contributions
                )
                dynamic_weight_total = sum(
                    dynamic_weights[strategy_id]
                    for strategy_id, _label, _base_weight in _CIS_EXPERTS
                    if strategy_id in rankings
                )
                scores[number] = contribution_total / max(
                    dynamic_weight_total,
                    1e-12,
                )
            else:
                scores[number] = probability
            supporters = sorted(
                contributions,
                key=lambda item: (-item[1], item[0]),
            )[:3]
            supporter_text = ", ".join(
                strategy_id for strategy_id, _contribution in supporters
            )
            details[number] = (
                (
                    f"Warm-up ensemble {self.cis_draw_count}/"
                    f"{_CIS_MINIMUM_TRAINING_DRAWS} draws"
                    if warm_up
                    else f"Learned probability {probability:.2%}"
                ),
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
                    "Beta-smoothed Bayesian gap-state posterior ranking.",
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
                fresh_random_details = {
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
            grid_scores, grid_details = self._predictive_grid_scores(gaps)
            rankings["predictive_grid"] = _ranking_from_scores(
                grid_scores,
                gaps,
            )
            if "predictive_grid" in requested:
                built["predictive_grid"] = _strategy(
                    "predictive_grid",
                    "Grid",
                    "Six-component predictive score grid from PyLotto.",
                    grid_scores,
                    gaps,
                    grid_details,
                )

        if "cis" in enabled:
            cis_scores, cis_details = self._cis_scores(rankings)
            rankings["cis"] = _ranking_from_scores(cis_scores, gaps)
            if "cis" in requested:
                built["cis"] = _strategy(
                    "cis",
                    "CIS",
                    "Collective Intelligence Strategy learning from ten expert rankings.",
                    cis_scores,
                    gaps,
                    cis_details,
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
            if draw_index >= history_start:
                suites.append(compared)
        elif draw_index >= history_start:
            actual = (
                tuple(ball.value for ball in draws[draw_index + 1].balls)
                if draw_index + 1 < total
                else ()
            )
            suites.append(
                PredictionSuite(
                    reference_draw_number=draw_index + 1,
                    target_draw_number=draw_index + 2,
                    actual_numbers=actual,
                    strategies=(),
                )
            )
        if progress is not None:
            progress(draw_index + 1, total)
    return tuple(suites)
