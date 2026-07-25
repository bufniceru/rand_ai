"""Build display-ready PyLotto-inspired strategy prediction histories."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from itertools import combinations

from rand_ai.draw import Draw
from rand_ai.prediction import CombinedPrediction

_NUMBER_COUNT = 49
_NUMBERS_PER_DRAW = 6
_BASE_PROBABILITY = _NUMBERS_PER_DRAW / _NUMBER_COUNT
_MAX_GAP_BUCKET = 35
_MARKOV_PRIOR_STRENGTH = 8.0
_MARKOV_DECAY = 0.5 ** (1 / 500)
_MKFR_MAX_ORDER = 20
_MKFR_PRIOR_STRENGTH = 8.0
_MKFR_MIN_CONTEXT_SUPPORT = 8
_RANDOM_SEED = 20260626
_PROXIMITY_BUCKETS = ("paired", "tight", "near", "balanced", "wide", "isolated")
_EARTH_MOVER_BUCKETS = ("Overlap", "Near", "Close", "Middle", "Far", "Distant")
_PRIMES = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
}

PredictionProgress = Callable[[int, int], None]
STRATEGY_IDS = (
    "proximity",
    "freshness",
    "emd",
    "randomness",
    "entropy",
    "markov100",
    "mkfr",
    "bayesian",
    "svc",
    "tbl",
)


@dataclass(frozen=True, slots=True)
class StrategyNumberPrediction:
    """Store one candidate's rank and score for a named strategy."""

    number: int
    rank: int
    score: float
    gap: int
    details: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyPrediction:
    """Store one named 49-number strategy ranking."""

    strategy_id: str
    name: str
    description: str
    numbers: tuple[StrategyNumberPrediction, ...]
    top_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PredictionSuite:
    """Store all prediction strategies after one reference draw."""

    reference_draw_number: int
    target_draw_number: int
    actual_numbers: tuple[int, ...]
    strategies: tuple[StrategyPrediction, ...]


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
    return {
        number: (score - minimum) / spread
        for number, score in scores.items()
    }


def _strategy(
    strategy_id: str,
    name: str,
    description: str,
    scores: dict[int, float],
    gaps: dict[int, int],
    details: dict[int, tuple[str, ...]] | None = None,
) -> StrategyPrediction:
    ranked = sorted(
        scores,
        key=lambda number: (-scores[number], -gaps[number], number),
    )
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


def _random_ranking(draw_index: int) -> list[int]:
    """Return the deterministic LCG/Fisher-Yates baseline used by PyLotto."""
    state = (_RANDOM_SEED ^ (((draw_index + 1) * 2654435761) & 0xFFFFFFFF))
    state &= 0xFFFFFFFF
    numbers = list(range(1, _NUMBER_COUNT + 1))
    for index in range(len(numbers) - 1, 0, -1):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        swap_index = state % (index + 1)
        numbers[index], numbers[swap_index] = numbers[swap_index], numbers[index]
    return numbers


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
        ordered[index + 1] - ordered[index]
        for index in range(len(ordered) - 1)
    ]
    circular_gaps.append((_NUMBER_COUNT + ordered[0]) - ordered[-1])
    total = sum(circular_gaps)
    entropy = -sum(
        (gap / total) * math.log2(gap / total)
        for gap in circular_gaps
        if gap > 0
    )
    return entropy / math.log2(_NUMBERS_PER_DRAW) * 100


class _StrategyState:
    """Maintain incremental state for the enabled prediction strategy plugins."""

    def __init__(
        self,
        enabled_strategy_ids: Collection[str] = STRATEGY_IDS,
    ) -> None:
        self.enabled_strategy_ids = frozenset(enabled_strategy_ids)
        self.draw_count = 0
        self.appearances = [0] * (_NUMBER_COUNT + 1)
        self.last_seen: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.occurrences: list[list[int]] = [
            [] for _ in range(_NUMBER_COUNT + 1)
        ]
        self.recent_draws: deque[set[int]] = deque(maxlen=100)
        self.pair_counts: dict[tuple[int, int], int] = {}
        self.previous_draw: set[int] = set()
        self.proximity_counts = [
            [0] * len(_PROXIMITY_BUCKETS)
            for _ in range(_NUMBER_COUNT + 1)
        ]
        self.proximity_totals = [0] * len(_PROXIMITY_BUCKETS)
        self.entropy_totals = [0.0] * (_NUMBER_COUNT + 1)
        self.high_entropy_hits = [0] * (_NUMBER_COUNT + 1)
        self.markov_opportunities = [0.0] * (_MAX_GAP_BUCKET + 1)
        self.markov_hits = [0.0] * (_MAX_GAP_BUCKET + 1)
        self.bayesian_opportunities = [0] * (_MAX_GAP_BUCKET + 1)
        self.bayesian_hits = [0] * (_MAX_GAP_BUCKET + 1)
        self.mkfr_histories: list[deque[int]] = (
            [
                deque(maxlen=_MKFR_MAX_ORDER)
                for _ in range(_NUMBER_COUNT + 1)
            ]
            if "mkfr" in self.enabled_strategy_ids
            else []
        )
        self.mkfr_transitions: list[list[dict[int, list[int]]]] = (
            [
                [
                    {}
                    for _ in range(_MKFR_MAX_ORDER)
                ]
                for _ in range(_NUMBER_COUNT + 1)
            ]
            if "mkfr" in self.enabled_strategy_ids
            else []
        )
        self.svc_weights = [0.0] * 11
        self.tbl_weights = [0.0] * 14
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
                self.draw_count
                if seen_at is None
                else self.draw_count - 1 - seen_at
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
                (recent8_expected - self._recent_count(number, 8))
                / recent8_expected,
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
        gaps = [
            right - left
            for left, right in zip(occurrences, occurrences[1:])
        ]
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
        overdue = (
            0.0
            if mean_gap <= 0
            else _clamp((gap - mean_gap) / mean_gap, -1, 1)
        )
        recent5 = self._recent_count(number, 5) / max(
            min(len(self.recent_draws), 5), 1
        )
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

    @staticmethod
    def _dot(weights: Sequence[float], features: Sequence[float]) -> float:
        return sum(weight * feature for weight, feature in zip(weights, features))

    def train(self, drawn: set[int]) -> None:
        """Learn the current draw using only the state available before it."""
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

    def remember(self, drawn: set[int]) -> None:
        entropy_enabled = "entropy" in self.enabled_strategy_ids
        proximity_enabled = bool(
            self.enabled_strategy_ids.intersection({"proximity", "tbl"})
        )
        entropy_percent = (
            _gap_entropy_percent(tuple(drawn)) if entropy_enabled else 0.0
        )
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

        if "tbl" in self.enabled_strategy_ids:
            for left, right in combinations(ordered, 2):
                key = (left, right)
                self.pair_counts[key] = self.pair_counts.get(key, 0) + 1

        if "tbl" in self.enabled_strategy_ids:
            self.previous_draw = drawn
        if self.enabled_strategy_ids.intersection({"svc", "tbl"}):
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
        return sum(
            abs(left[index] - right[index]) for index in range(length)
        ) / length

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
        weighted_hits = {
            number: 0.0 for number in range(1, _NUMBER_COUNT + 1)
        }
        weighted_distances = {
            number: 0.0 for number in range(1, _NUMBER_COUNT + 1)
        }
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
            number: (
                weighted_hits[number] / maximum if maximum > 0 else 0.0
            )
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
            scores[number] = (
                sum(count * share for count, share in zip(counts, shares))
                / max(self.draw_count, 1)
            )
            top_bucket = max(range(len(counts)), key=lambda index: (counts[index], -index))
            details[number] = (
                _PROXIMITY_BUCKETS[top_bucket].title(),
                f"{self.appearances[number]} appearances",
            )
        return _scale_scores(scores), details

    def _entropy_scores(self, gaps: dict[int, int]) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raw: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            appearances = self.appearances[number]
            average = (
                50.0
                if appearances == 0
                else self.entropy_totals[number] / appearances
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
        return (
            self.appearances[number]
            + _MKFR_PRIOR_STRENGTH * _BASE_PROBABILITY
        ) / (self.draw_count + _MKFR_PRIOR_STRENGTH)

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
            probability = (
                hits + _MKFR_PRIOR_STRENGTH * probability
            ) / (opportunities + _MKFR_PRIOR_STRENGTH)
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
            selected_context = (
                context[-selected_order:]
                if selected_order > 0
                else "—"
            )
            raw[number] = lift
            details[number] = (
                f"Context probability {probability:.2%}",
                f"Baseline probability {baseline:.2%}",
                f"Transition lift {lift * 100:+.2f} pp",
                f"Order {selected_order}/{_MKFR_MAX_ORDER}: {selected_context}",
                f"Context support {support}",
            )
        return _scale_scores(raw), details

    def build_strategies(
        self,
        combined: CombinedPrediction,
        draw_index: int,
    ) -> tuple[StrategyPrediction, ...]:
        gaps = self.current_gaps()
        enabled = self.enabled_strategy_ids
        built: dict[str, StrategyPrediction] = {}
        tbl_enabled = "tbl" in enabled

        freshness_scores: dict[int, float] = {}
        if "freshness" in enabled or tbl_enabled:
            by_number = {item.number: item for item in combined.numbers}
            freshness_scores = _scale_scores(
                {
                    number: by_number[number].freshness_score
                    for number in range(1, _NUMBER_COUNT + 1)
                }
            )
            if "freshness" in enabled:
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
        if "proximity" in enabled or tbl_enabled:
            proximity_scores, proximity_details = self._proximity_scores()
            if "proximity" in enabled:
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
            built["emd"] = _strategy(
                "emd",
                "EMD",
                "Earth-mover analogue ranking from historical draw vectors.",
                earth_mover_scores,
                gaps,
                earth_mover_details,
            )

        random_ranking: list[int] = []
        if "randomness" in enabled or tbl_enabled:
            random_ranking = _random_ranking(draw_index + 1)
            randomness_scores = {
                number: (_NUMBER_COUNT - rank) / (_NUMBER_COUNT - 1)
                for rank, number in enumerate(random_ranking, start=1)
            }
            if "randomness" in enabled:
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
            built["entropy"] = _strategy(
                "entropy",
                "Entr",
                "Structural gap-entropy history with overdue adjustment.",
                entropy_scores,
                gaps,
                entropy_details,
            )

        if "markov100" in enabled:
            markov_scores, markov_details = self._gap_model_scores(
                gaps, weighted=True
            )
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
            built["svc"] = _strategy(
                "svc",
                "SVC",
                "Online linear support-vector classifier inspired by PyLotto.",
                svc_scores,
                gaps,
                svc_details,
            )

        if tbl_enabled:
            self.prior_rankings = {
                "freshness": sorted(
                    freshness_scores,
                    key=lambda number: (-freshness_scores[number], number),
                ),
                "proximity": sorted(
                    proximity_scores,
                    key=lambda number: (-proximity_scores[number], number),
                ),
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
            built["tbl"] = _strategy(
                "tbl",
                "TBL",
                "Temporal Behavior Learning with recency, frequency, and strategy features.",
                _scale_scores(tbl_raw),
                gaps,
                tbl_details,
            )

        return tuple(
            built[strategy_id]
            for strategy_id in STRATEGY_IDS
            if strategy_id in built
        )


def build_prediction_suites(
    draws: Sequence[Draw],
    *,
    history_start: int = 0,
    enabled_strategy_ids: Collection[str] = STRATEGY_IDS,
    progress: PredictionProgress | None = None,
) -> tuple[PredictionSuite, ...]:
    """Calculate enabled strategy plugins and retain the requested history."""
    requested = set(enabled_strategy_ids)
    unknown = requested.difference(STRATEGY_IDS)
    if unknown:
        raise ValueError(
            f"Unknown prediction strategy plugin(s): {', '.join(sorted(unknown))}"
        )
    selected = tuple(
        strategy_id for strategy_id in STRATEGY_IDS if strategy_id in requested
    )
    state = _StrategyState(selected)
    suites: list[PredictionSuite] = []
    total = len(draws)
    for draw_index, draw in enumerate(draws):
        drawn = {ball.value for ball in draw.balls}
        if selected:
            state.train(drawn)
            state.remember(drawn)
        if draw_index >= history_start:
            combined = draw.prediction
            if selected and combined is None:
                raise ValueError("Combined predictions must be prepared first")
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
                    strategies=(
                        ()
                        if combined is None
                        else state.build_strategies(combined, draw_index)
                    ),
                )
            )
        if progress is not None:
            progress(draw_index + 1, total)
    return tuple(suites)
