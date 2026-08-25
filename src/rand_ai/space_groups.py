"""Circular border-space grouping, diagnostics, and online forecasts."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import exp, log, sqrt
from typing import TypedDict

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import SGDClassifier

NUMBER_COUNT = 49
NUMBERS_PER_DRAW = 6
SPACE_TOTAL = NUMBER_COUNT - NUMBERS_PER_DRAW
DEFAULT_BORDER_SPACE = 7
MIN_BORDER_SPACE = 0
MAX_BORDER_SPACE = SPACE_TOTAL
MODEL_EVALUATION_WARMUP = 100
ML_WARMUP = 50

SIGNATURES: tuple[tuple[int, ...], ...] = (
    (6,),
    (5, 1),
    (4, 2),
    (4, 1, 1),
    (3, 3),
    (3, 2, 1),
    (3, 1, 1, 1),
    (2, 2, 2),
    (2, 2, 1, 1),
    (2, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1),
)
SIGNATURE_INDEX: dict[tuple[int, ...], int] = {
    signature: index for index, signature in enumerate(SIGNATURES)
}
MODEL_IDS = (
    "border_group_statistical",
    "border_group_markov",
    "border_group_bayesian",
    "border_group_ml",
    "border_group_hybrid",
)
COMPONENT_MODEL_IDS = MODEL_IDS[:-1]
MODEL_NAMES = {
    "border_group_statistical": "Border Group Statistical",
    "border_group_markov": "Border Group Markov",
    "border_group_bayesian": "Border Group Bayesian",
    "border_group_ml": "Border Group ML",
    "border_group_hybrid": "Border Group Hybrid",
    "random_null": "Exact random 6/49 null",
}


def validate_border_space(value: int) -> int:
    """Return a valid inclusive small-space border."""
    if type(value) is not int or not MIN_BORDER_SPACE <= value <= MAX_BORDER_SPACE:
        raise ValueError(
            f"border_space must be between {MIN_BORDER_SPACE} and {MAX_BORDER_SPACE}"
        )
    return value


def spaces_for_numbers(numbers: Collection[int]) -> tuple[int, ...]:
    """Return the six empty-number spaces around a sorted 6/49 draw."""
    ordered = tuple(sorted(numbers))
    if (
        len(ordered) != NUMBERS_PER_DRAW
        or len(set(ordered)) != NUMBERS_PER_DRAW
        or ordered[0] < 1
        or ordered[-1] > NUMBER_COUNT
    ):
        raise ValueError("Space groups require six unique numbers from 1 through 49")
    return (
        (ordered[0] - 1) + (NUMBER_COUNT - ordered[-1]),
        *(right - left - 1 for left, right in zip(ordered, ordered[1:])),
    )


def _ordered_group_sizes(
    spaces: Sequence[int], numbers: Sequence[int] | None = None
) -> tuple[int, ...]:
    ordered_starts = _ordered_separator_indices(spaces, numbers)
    if not ordered_starts:
        return (NUMBERS_PER_DRAW,)
    sizes = []
    for index, start in enumerate(ordered_starts):
        following = ordered_starts[(index + 1) % len(ordered_starts)]
        sizes.append((following - start) % NUMBERS_PER_DRAW or NUMBERS_PER_DRAW)
    return tuple(sizes)


def _ordered_separator_indices(
    spaces: Sequence[int], numbers: Sequence[int] | None = None
) -> tuple[int, ...]:
    """Order separators from the largest gap and deterministic following number."""
    large = [index for index, value in enumerate(spaces) if value > 0]
    if not large:
        return ()
    maximum = max(spaces[index] for index in large)
    candidates = [index for index in large if spaces[index] == maximum]
    anchor = (
        min(candidates)
        if numbers is None
        else min(candidates, key=lambda index: numbers[index])
    )
    return tuple(
        sorted(large, key=lambda index: (index - anchor) % NUMBERS_PER_DRAW)
    )


@dataclass(frozen=True, slots=True)
class SpaceGroupProfile:
    """Describe one draw under an inclusive border-space threshold."""

    spaces: tuple[int, ...]
    large_spaces: tuple[int, ...]
    separator_indices: tuple[int, ...]
    ordered_separator_indices: tuple[int, ...]
    separator_count: int
    group_count: int
    ordered_groups: tuple[tuple[int, ...], ...]
    ordered_group_sizes: tuple[int, ...]
    signature: tuple[int, ...]
    maximum_space: int
    anchor: int | None

    @property
    def signature_text(self) -> str:
        return "+".join(str(value) for value in self.signature)


def profile_from_spaces(
    spaces: Sequence[int],
    border_space: int = DEFAULT_BORDER_SPACE,
    *,
    numbers: Sequence[int] | None = None,
) -> SpaceGroupProfile:
    """Classify a six-space circle into maximal small-space-connected groups."""
    border = validate_border_space(border_space)
    values = tuple(spaces)
    if (
        len(values) != NUMBERS_PER_DRAW
        or any(type(value) is not int or value < 0 for value in values)
        or sum(values) != SPACE_TOTAL
    ):
        raise ValueError("spaces must contain six non-negative integers summing to 43")
    ordered_numbers = None if numbers is None else tuple(numbers)
    if ordered_numbers is not None and (
        len(ordered_numbers) != NUMBERS_PER_DRAW
        or len(set(ordered_numbers)) != NUMBERS_PER_DRAW
        or any(
            type(number) is not int or not 1 <= number <= NUMBER_COUNT
            for number in ordered_numbers
        )
    ):
        raise ValueError("numbers must contain six unique values from 1 through 49")
    separator_values = tuple(value if value > border else 0 for value in values)
    separator_indices = tuple(
        index for index, value in enumerate(separator_values) if value > 0
    )
    ordered_separators = _ordered_separator_indices(
        separator_values, ordered_numbers
    )
    separator_count = len(separator_indices)
    ordered_sizes = _ordered_group_sizes(separator_values, ordered_numbers)
    if ordered_numbers is None:
        ordered_groups: tuple[tuple[int, ...], ...] = ()
    elif not ordered_separators:
        ordered_groups = (ordered_numbers,)
    else:
        ordered_groups = tuple(
            tuple(
                ordered_numbers[(start + offset) % NUMBERS_PER_DRAW]
                for offset in range(size)
            )
            for start, size in zip(
                ordered_separators, ordered_sizes, strict=True
            )
        )
    signature = tuple(sorted(ordered_sizes, reverse=True))
    return SpaceGroupProfile(
        spaces=values,
        large_spaces=tuple(value for value in values if value > border),
        separator_indices=separator_indices,
        ordered_separator_indices=ordered_separators,
        separator_count=separator_count,
        group_count=max(separator_count, 1),
        ordered_groups=ordered_groups,
        ordered_group_sizes=ordered_sizes,
        signature=signature,
        maximum_space=max(values),
        anchor=None if ordered_numbers is None else min(ordered_numbers) - 1,
    )


def profile_for_numbers(
    numbers: Collection[int], border_space: int = DEFAULT_BORDER_SPACE
) -> SpaceGroupProfile:
    """Return a border-group profile for a six-number draw."""
    ordered = tuple(sorted(numbers))
    return profile_from_spaces(
        spaces_for_numbers(ordered), border_space, numbers=ordered
    )


def _signature_for_mask(mask: int) -> tuple[int, ...]:
    separators = [index for index in range(NUMBERS_PER_DRAW) if mask & (1 << index)]
    if not separators:
        return (NUMBERS_PER_DRAW,)
    sizes = [
        (separators[(index + 1) % len(separators)] - start) % NUMBERS_PER_DRAW
        or NUMBERS_PER_DRAW
        for index, start in enumerate(separators)
    ]
    return tuple(sorted(sizes, reverse=True))


@lru_cache(maxsize=MAX_BORDER_SPACE + 1)
def exact_null_signature_counts(border_space: int) -> tuple[int, ...]:
    """Count rooted circular gap compositions for every group signature."""
    border = validate_border_space(border_space)
    assignment_counts = []
    for large_count in range(NUMBERS_PER_DRAW + 1):
        states = {(0, 0): 1}
        for position in range(NUMBERS_PER_DRAW):
            next_states: dict[tuple[int, int], int] = defaultdict(int)
            must_be_large = position < large_count
            lower = border + 1 if must_be_large else 0
            upper = SPACE_TOTAL if must_be_large else border
            for (used, _placed), count in states.items():
                for value in range(lower, min(upper, SPACE_TOTAL - used) + 1):
                    next_states[(used + value, position + 1)] += count
            states = next_states
        assignment_counts.append(states.get((SPACE_TOTAL, NUMBERS_PER_DRAW), 0))

    counts = [0] * len(SIGNATURES)
    for mask in range(1 << NUMBERS_PER_DRAW):
        signature = _signature_for_mask(mask)
        counts[SIGNATURE_INDEX[signature]] += assignment_counts[mask.bit_count()]
    return tuple(counts)


def exact_null_probabilities(border_space: int) -> tuple[float, ...]:
    """Return exact random 6/49 probabilities for canonical signatures."""
    counts = exact_null_signature_counts(border_space)
    total = sum(counts)
    return tuple(count / total for count in counts)


def exact_null_group_probabilities(border_space: int) -> tuple[float, ...]:
    """Return exact probabilities for group counts one through six."""
    result = [0.0] * NUMBERS_PER_DRAW
    for signature, probability in zip(
        SIGNATURES, exact_null_probabilities(border_space), strict=True
    ):
        result[len(signature) - 1] += probability
    return tuple(result)


def _normalize(values: Sequence[float]) -> tuple[float, ...]:
    total = sum(values)
    if total <= 0:
        return tuple(1 / len(values) for _ in values)
    return tuple(value / total for value in values)


@lru_cache(maxsize=(MAX_BORDER_SPACE + 1) * len(SIGNATURES))
def _fallback_shapes(
    signature_index: int, border_space: int
) -> tuple[tuple[int, ...], ...]:
    """Build deterministic valid-space priors for one signature beam."""
    signature = SIGNATURES[signature_index]
    shapes: set[tuple[int, ...]] = set()
    for mask in range(1 << NUMBERS_PER_DRAW):
        if _signature_for_mask(mask) != signature:
            continue
        minimums = [
            border_space + 1 if mask & (1 << index) else 0
            for index in range(NUMBERS_PER_DRAW)
        ]
        capacities = [
            SPACE_TOTAL - minimums[index]
            if mask & (1 << index)
            else border_space
            for index in range(NUMBERS_PER_DRAW)
        ]
        remaining = SPACE_TOTAL - sum(minimums)
        if remaining < 0 or remaining > sum(capacities):
            continue
        values = minimums[:]
        for index, capacity in enumerate(capacities):
            addition = min(remaining, capacity)
            values[index] += addition
            remaining -= addition
        if remaining == 0:
            shapes.add(tuple(values))
    return tuple(sorted(shapes))


def _space_bucket(value: int) -> int:
    if value <= 7:
        return 0
    if value <= 11:
        return 1
    if value <= 15:
        return 2
    return 3


def _trend_category(profiles: Sequence[SpaceGroupProfile]) -> int:
    recent = profiles[-25:]
    if len(recent) < 4:
        return 1
    midpoint = len(recent) // 2
    left = sum(profile.group_count for profile in recent[:midpoint]) / midpoint
    right_values = recent[midpoint:]
    right = sum(profile.group_count for profile in right_values) / len(right_values)
    if right - left > 0.2:
        return 2
    if left - right > 0.2:
        return 0
    return 1


class SpaceGroupForecaster:
    """Maintain five online, next-signature probability forecasts."""

    def __init__(self, border_space: int = DEFAULT_BORDER_SPACE) -> None:
        self.border_space = validate_border_space(border_space)
        self.feasible_signatures = tuple(
            count > 0 for count in exact_null_signature_counts(self.border_space)
        )
        self.profiles: list[SpaceGroupProfile] = []
        self.signature_counts = [0] * len(SIGNATURES)
        self.transition_counts = [
            [0] * len(SIGNATURES) for _ in range(len(SIGNATURES))
        ]
        self.bayes_counts: list[defaultdict[tuple[int, int], int]] = [
            defaultdict(int) for _ in range(5)
        ]
        self.bayes_cardinalities = (len(SIGNATURES), 6, 4, len(SIGNATURES), 3)
        self.ml = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.001,
            average=True,
            random_state=0,
        )
        self.ml_training_count = 0
        self.pending_features: tuple[float, ...] | None = None
        self.pending_bayes: tuple[int, ...] | None = None
        self.pending_forecasts: dict[str, tuple[float, ...]] | None = None
        self.pending_prior_count = 0
        self.losses = {
            model_id: deque(maxlen=100) for model_id in COMPONENT_MODEL_IDS
        }
        self.recent_shapes: list[deque[tuple[int, ...]]] = [
            deque(maxlen=16) for _ in SIGNATURES
        ]
        self.anchor_counts: list[Counter[int]] = [
            Counter() for _ in SIGNATURES
        ]
        self._candidate_cache: tuple[
            dict[tuple[int, ...], float], ...
        ] | None = None
        self._decoder_cache: tuple[dict[int, float], ...] | None = None

    def _statistical(self) -> tuple[float, ...]:
        return _normalize(
            tuple(
                count + 1.0 if feasible else 0.0
                for count, feasible in zip(
                    self.signature_counts, self.feasible_signatures, strict=True
                )
            )
        )

    def _markov(self, baseline: Sequence[float]) -> tuple[float, ...]:
        if not self.profiles:
            return tuple(baseline)
        previous = SIGNATURE_INDEX[self.profiles[-1].signature]
        row = self.transition_counts[previous]
        total = sum(row)
        return _normalize(tuple(
            (row[index] + 10.0 * baseline[index]) / (total + 10.0)
            for index in range(len(SIGNATURES))
        ))

    def _bayes_features(self) -> tuple[int, ...]:
        current = self.profiles[-1]
        modal = Counter(
            SIGNATURE_INDEX[profile.signature] for profile in self.profiles[-10:]
        ).most_common(1)[0][0]
        return (
            SIGNATURE_INDEX[current.signature],
            current.group_count - 1,
            _space_bucket(current.maximum_space),
            modal,
            _trend_category(self.profiles),
        )

    def _bayesian(
        self, baseline: Sequence[float], features: Sequence[int]
    ) -> tuple[float, ...]:
        log_scores = []
        total = sum(self.signature_counts)
        for target in range(len(SIGNATURES)):
            if not self.feasible_signatures[target]:
                log_scores.append(float("-inf"))
                continue
            score = log((self.signature_counts[target] + 1) / (total + len(SIGNATURES)))
            for feature_index, value in enumerate(features):
                numerator = self.bayes_counts[feature_index][(target, value)] + 1
                denominator = (
                    self.signature_counts[target]
                    + self.bayes_cardinalities[feature_index]
                )
                score += log(numerator / denominator)
            log_scores.append(score)
        maximum = max(log_scores)
        return _normalize(
            tuple(
                0.0 if score == float("-inf") else exp(score - maximum)
                for score in log_scores
            )
        )

    def _ml_features(self) -> tuple[float, ...]:
        features: list[float] = []
        for lag in range(1, 4):
            one_hot = [0.0] * len(SIGNATURES)
            profile = self.profiles[-lag] if len(self.profiles) >= lag else None
            if profile is not None:
                one_hot[SIGNATURE_INDEX[profile.signature]] = 1.0
            features.extend(one_hot)
            features.extend(
                [
                    0.0 if profile is None else profile.group_count / 6,
                    0.0 if profile is None else profile.maximum_space / SPACE_TOTAL,
                ]
            )
        current = self.profiles[-1]
        features.extend(value / SPACE_TOTAL for value in current.spaces)
        for window in (10, 25, 100):
            recent = self.profiles[-window:]
            counts = Counter(SIGNATURE_INDEX[profile.signature] for profile in recent)
            divisor = max(len(recent), 1)
            features.extend(counts[index] / divisor for index in range(len(SIGNATURES)))
            features.append(
                sum(profile.group_count for profile in recent) / divisor / 6
            )
        features.append(_trend_category(self.profiles) / 2)
        return tuple(features)

    def _hybrid_weights(self) -> dict[str, float]:
        if min((len(values) for values in self.losses.values()), default=0) < 30:
            return {model_id: 1 / len(COMPONENT_MODEL_IDS) for model_id in COMPONENT_MODEL_IDS}
        raw = {
            model_id: exp(-sum(values) / len(values))
            for model_id, values in self.losses.items()
        }
        total = sum(raw.values())
        floor = 0.05
        remaining = 1 - floor * len(raw)
        return {
            model_id: floor + remaining * value / total
            for model_id, value in raw.items()
        }

    def observe(
        self, profile: SpaceGroupProfile
    ) -> tuple[int, dict[str, tuple[float, ...]], int] | None:
        """Score the pending forecast, train it, then append one actual draw."""
        actual = SIGNATURE_INDEX[profile.signature]
        evaluation = None
        if self.pending_forecasts is not None:
            evaluation = (
                self.pending_prior_count,
                self.pending_forecasts,
                actual,
            )
            for model_id in COMPONENT_MODEL_IDS:
                probability = max(self.pending_forecasts[model_id][actual], 1e-15)
                self.losses[model_id].append(-log(probability))
        if self.pending_features is not None:
            features = np.asarray([self.pending_features], dtype=np.float64)
            target = np.asarray([actual], dtype=np.int64)
            if self.ml_training_count == 0:
                self.ml.partial_fit(features, target, classes=np.arange(len(SIGNATURES)))
            else:
                self.ml.partial_fit(features, target)
            self.ml_training_count += 1
        if self.pending_bayes is not None:
            for feature_index, value in enumerate(self.pending_bayes):
                self.bayes_counts[feature_index][(actual, value)] += 1
        if self.profiles:
            previous = SIGNATURE_INDEX[self.profiles[-1].signature]
            self.transition_counts[previous][actual] += 1
        self.signature_counts[actual] += 1
        self.profiles.append(profile)
        self.recent_shapes[actual].append(profile.spaces)
        if profile.anchor is not None:
            self.anchor_counts[actual][profile.anchor] += 1
        self._candidate_cache = None
        self._decoder_cache = None
        return evaluation

    def forecast(self) -> dict[str, tuple[float, ...]]:
        """Build and retain all next-draw forecasts from current state only."""
        baseline = self._statistical()
        if not self.profiles:
            forecasts = {model_id: baseline for model_id in MODEL_IDS}
            return forecasts
        bayes_features = self._bayes_features()
        ml_features = self._ml_features()
        forecasts: dict[str, tuple[float, ...]] = {
            "border_group_statistical": baseline,
            "border_group_markov": self._markov(baseline),
            "border_group_bayesian": self._bayesian(baseline, bayes_features),
        }
        if self.ml_training_count >= ML_WARMUP:
            probabilities = self.ml.predict_proba(
                np.asarray([ml_features], dtype=np.float64)
            )[0]
            forecasts["border_group_ml"] = _normalize(
                tuple(
                    float(value) if self.feasible_signatures[index] else 0.0
                    for index, value in enumerate(probabilities)
                )
            )
        else:
            forecasts["border_group_ml"] = baseline
        weights = self._hybrid_weights()
        forecasts["border_group_hybrid"] = tuple(
            sum(weights[model_id] * forecasts[model_id][index] for model_id in COMPONENT_MODEL_IDS)
            for index in range(len(SIGNATURES))
        )
        self.pending_features = ml_features
        self.pending_bayes = bayes_features
        self.pending_forecasts = forecasts
        self.pending_prior_count = len(self.profiles)
        return forecasts

    def hybrid_weights(self) -> dict[str, float]:
        """Return the current normalized hybrid component weights."""
        return self._hybrid_weights()

    def number_scores(
        self, probabilities: Sequence[float]
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Decode signature probabilities through valid historical ticket marginals."""
        signature_marginals = self._signature_marginals()
        marginals = {number: 0.0 for number in range(1, NUMBER_COUNT + 1)}
        for signature_index, distribution in enumerate(signature_marginals):
            for number, probability in distribution.items():
                marginals[number] += probabilities[signature_index] * probability
        if not any(marginals.values()):
            marginals = {number: 1 / NUMBER_COUNT for number in marginals}
        minimum = min(marginals.values())
        maximum = max(marginals.values())
        spread = maximum - minimum
        scores = {
            number: 0.0 if spread <= 0 else (value - minimum) / spread
            for number, value in marginals.items()
        }
        ranked_signatures = sorted(
            range(len(SIGNATURES)), key=lambda index: (-probabilities[index], index)
        )
        leading = ranked_signatures[:3]
        details: dict[int, tuple[str, ...]] = {
            number: (
                f"Border space {self.border_space}",
                f"Decoded marginal {marginals[number]:.2%}",
                "Leading signatures "
                + ", ".join(
                    f"{'+'.join(map(str, SIGNATURES[index]))} {probabilities[index]:.1%}"
                    for index in leading
                ),
            )
            for number in marginals
        }
        return scores, details

    def _signature_marginals(self) -> tuple[dict[int, float], ...]:
        """Return cached valid-ticket number marginals for each signature beam."""
        if self._decoder_cache is not None:
            return self._decoder_cache
        decoded: list[dict[int, float]] = []
        for candidates in self._signature_candidates():
            marginals: dict[int, float] = defaultdict(float)
            for numbers, weight in candidates.items():
                for number in numbers:
                    marginals[number] += weight / NUMBERS_PER_DRAW
            decoded.append(dict(marginals))
        self._decoder_cache = tuple(decoded)
        return self._decoder_cache

    def _signature_candidates(
        self,
    ) -> tuple[dict[tuple[int, ...], float], ...]:
        """Return smoothed, leakage-safe valid-ticket beams per signature."""
        if self._candidate_cache is not None:
            return self._candidate_cache
        beams: list[dict[tuple[int, ...], float]] = []
        for signature_index, shape_history in enumerate(self.recent_shapes):
            shape_counts: Counter[tuple[int, ...]] = Counter(shape_history)
            for shape in _fallback_shapes(signature_index, self.border_space):
                shape_counts[shape] += 1
            shape_total = sum(shape_counts.values())
            candidates: dict[tuple[int, ...], float] = defaultdict(float)
            for spaces, shape_count in shape_counts.items():
                anchors = range(spaces[0] + 1)
                anchor_weights = {
                    anchor: self.anchor_counts[signature_index][anchor] + 1.0
                    for anchor in anchors
                }
                anchor_total = sum(anchor_weights.values())
                shape_weight = shape_count / shape_total
                for anchor, anchor_weight in anchor_weights.items():
                    numbers = [anchor + 1]
                    for space in spaces[1:]:
                        numbers.append(numbers[-1] + space + 1)
                    ticket = tuple(numbers)
                    candidates[ticket] += (
                        shape_weight * anchor_weight / anchor_total
                    )
            beams.append(dict(candidates))
        self._candidate_cache = tuple(beams)
        return self._candidate_cache

    def decoded_tickets(
        self, probabilities: Sequence[float]
    ) -> tuple[tuple[tuple[int, ...], str, float], ...]:
        """Expose the reweighted valid ticket beam used for number marginals."""
        if len(probabilities) != len(SIGNATURES):
            raise ValueError("probabilities must contain all 11 signatures")
        rows = []
        for signature_index, candidates in enumerate(
            self._signature_candidates()
        ):
            signature_text = "+".join(map(str, SIGNATURES[signature_index]))
            for ticket, candidate_weight in candidates.items():
                rows.append(
                    (
                        ticket,
                        signature_text,
                        probabilities[signature_index] * candidate_weight,
                    )
                )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class ModelMetric:
    model_id: str
    name: str
    evaluated_draws: int
    log_loss: float | None
    log_loss_ci_low: float | None
    log_loss_ci_high: float | None
    brier_score: float | None
    signature_accuracy: float | None
    group_count_accuracy: float | None
    group_count_mae: float | None


class WalkForwardResult(TypedDict):
    forecaster: SpaceGroupForecaster
    metrics: list[ModelMetric]
    best_model_id: str | None
    provisional: bool
    latest: dict[str, tuple[float, ...]]
    hybrid_weights: dict[str, float]


def _metrics(
    model_id: str,
    evaluations: Sequence[tuple[Sequence[float], int]],
) -> ModelMetric:
    if not evaluations:
        return ModelMetric(model_id, MODEL_NAMES[model_id], 0, None, None, None, None, None, None, None)
    losses = [-log(max(probabilities[actual], 1e-15)) for probabilities, actual in evaluations]
    briers = [
        sum((probability - int(index == actual)) ** 2 for index, probability in enumerate(probabilities))
        for probabilities, actual in evaluations
    ]
    signature_hits = [int(max(range(len(SIGNATURES)), key=probabilities.__getitem__) == actual) for probabilities, actual in evaluations]
    count_errors = []
    count_hits = []
    for probabilities, actual in evaluations:
        predicted = max(range(len(SIGNATURES)), key=probabilities.__getitem__)
        error = abs(len(SIGNATURES[predicted]) - len(SIGNATURES[actual]))
        count_errors.append(error)
        count_hits.append(int(error == 0))
    mean_loss = sum(losses) / len(losses)
    if len(losses) > 1:
        variance = sum((value - mean_loss) ** 2 for value in losses) / (len(losses) - 1)
        margin = 1.96 * sqrt(variance / len(losses))
    else:
        margin = 0.0
    return ModelMetric(
        model_id=model_id,
        name=MODEL_NAMES[model_id],
        evaluated_draws=len(evaluations),
        log_loss=mean_loss,
        log_loss_ci_low=max(0.0, mean_loss - margin),
        log_loss_ci_high=mean_loss + margin,
        brier_score=sum(briers) / len(briers),
        signature_accuracy=sum(signature_hits) / len(signature_hits),
        group_count_accuracy=sum(count_hits) / len(count_hits),
        group_count_mae=sum(count_errors) / len(count_errors),
    )


def walk_forward_models(
    profiles: Sequence[SpaceGroupProfile], border_space: int
) -> WalkForwardResult:
    """Evaluate all approaches chronologically and forecast the next signature."""
    forecaster = SpaceGroupForecaster(border_space)
    evaluations: dict[str, list[tuple[Sequence[float], int]]] = {
        model_id: [] for model_id in (*MODEL_IDS, "random_null")
    }
    null = exact_null_probabilities(border_space)
    for profile in profiles:
        result = forecaster.observe(profile)
        if result is not None:
            prior_count, forecasts, actual = result
            if prior_count >= MODEL_EVALUATION_WARMUP:
                for model_id in MODEL_IDS:
                    evaluations[model_id].append((forecasts[model_id], actual))
                evaluations["random_null"].append((null, actual))
        forecaster.forecast()
    latest = forecaster.forecast() if profiles else {
        model_id: exact_null_probabilities(border_space) for model_id in MODEL_IDS
    }
    metrics = [_metrics(model_id, evaluations[model_id]) for model_id in (*MODEL_IDS, "random_null")]
    candidates = [metric for metric in metrics if metric.model_id in MODEL_IDS and metric.log_loss is not None]
    best = min(candidates, key=lambda metric: (metric.log_loss, metric.brier_score)) if candidates else None
    return {
        "forecaster": forecaster,
        "metrics": metrics,
        "best_model_id": None if best is None else best.model_id,
        "provisional": best is None,
        "latest": latest,
        "hybrid_weights": forecaster.hybrid_weights(),
    }


def transition_diagnostics(
    profiles: Sequence[SpaceGroupProfile], permutations: int = 200
) -> tuple[list[list[int]], float, float]:
    """Return transition counts, mutual information, and a seeded permutation p-value."""
    size = len(SIGNATURES)
    matrix = [[0] * size for _ in range(size)]
    indexes = [SIGNATURE_INDEX[profile.signature] for profile in profiles]
    for left, right in zip(indexes, indexes[1:]):
        matrix[left][right] += 1

    def mutual_information(sequence: Sequence[int]) -> float:
        pairs = Counter(zip(sequence, sequence[1:]))
        total = sum(pairs.values())
        if total == 0:
            return 0.0
        left_counts = Counter(sequence[:-1])
        right_counts = Counter(sequence[1:])
        return sum(
            count / total
            * log((count * total) / (left_counts[left] * right_counts[right]))
            for (left, right), count in pairs.items()
        )

    observed = mutual_information(indexes)
    if len(indexes) < 3 or permutations <= 0:
        return matrix, observed, 1.0
    rng = np.random.default_rng(0)
    sampled = np.asarray(indexes[-5000:], dtype=np.int64)
    exceedances = 0
    for _ in range(permutations):
        shuffled = rng.permutation(sampled)
        exceedances += mutual_information(shuffled.tolist()) >= observed
    return matrix, observed, (exceedances + 1) / (permutations + 1)


def signature_chi_square(
    profiles: Sequence[SpaceGroupProfile], border_space: int
) -> tuple[float, float]:
    """Compare observed signatures with the exact 6/49 null."""
    observed = Counter(profile.signature for profile in profiles)
    probabilities = exact_null_probabilities(border_space)
    statistic = 0.0
    degrees = 0
    for signature, probability in zip(SIGNATURES, probabilities, strict=True):
        expected = len(profiles) * probability
        if expected > 0:
            statistic += (observed[signature] - expected) ** 2 / expected
            degrees += 1
    return statistic, float(chi2.sf(statistic, max(degrees - 1, 1)))
