"""Exact-state categorical chi-square probability estimates for 6/49 draws."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

_NUMBER_COUNT = 49
_NUMBERS_PER_DRAW = 6
_BASE_PROBABILITY = _NUMBERS_PER_DRAW / _NUMBER_COUNT
_BASELINE_PRIOR_EXPOSURES = 49.0
_BASELINE_PRIOR_HITS = 6.0
_SINGLE_PRIOR_STRENGTH = 12.0
_PAIR_PRIOR_STRENGTH = 24.0
_TRIPLE_PRIOR_STRENGTH = 48.0
_PROBABILITY_EPSILON = 1e-9

CategoryValue: TypeAlias = int | None
Category: TypeAlias = CategoryValue | tuple[CategoryValue, ...]


@dataclass(frozen=True, slots=True)
class ContingencyEvidence:
    """Describe one current cell within a categorical outcome table."""

    probability: float
    support: int
    hits: int
    chi_square: float
    residual: float
    cramers_v: float
    adjustment: float


class ContingencyTable:
    """Maintain a categorical-by-binary table and constant-time statistics."""

    def __init__(self) -> None:
        self.cells: dict[Category, list[int]] = {}
        self.total_hits = 0
        self.total_exposures = 0
        self._quadratic_sum = 0.0

    def observe(self, category: Category, hit: bool) -> None:
        """Add one binary outcome to an exact category."""
        counts = self.cells.setdefault(category, [0, 0])
        old_hits, old_exposures = counts
        if old_exposures:
            self._quadratic_sum -= old_hits**2 / old_exposures
        counts[0] += int(hit)
        counts[1] += 1
        self._quadratic_sum += counts[0] ** 2 / counts[1]
        self.total_hits += int(hit)
        self.total_exposures += 1

    def cell_counts(self, category: Category) -> tuple[int, int]:
        """Return hits and exposures for one category, including unseen cells."""
        counts = self.cells.get(category)
        return (0, 0) if counts is None else (counts[0], counts[1])

    def chi_square(self) -> float:
        """Return Pearson's chi-square for the categorical-by-outcome table."""
        exposures = self.total_exposures
        hits = self.total_hits
        misses = exposures - hits
        if exposures == 0 or hits == 0 or misses == 0:
            return 0.0
        centered = self._quadratic_sum - hits**2 / exposures
        statistic = exposures**2 * centered / (hits * misses)
        return max(statistic, 0.0)

    def corrected_cramers_v(self) -> float:
        """Return bias-corrected Cramer's V for this table."""
        exposures = self.total_exposures
        row_count = len(self.cells)
        if exposures <= 1 or row_count <= 1:
            return 0.0
        column_count = 2
        phi_squared = self.chi_square() / exposures
        correction = (row_count - 1) * (column_count - 1) / (exposures - 1)
        corrected_phi = max(0.0, phi_squared - correction)
        corrected_rows = row_count - (row_count - 1) ** 2 / (exposures - 1)
        corrected_columns = (
            column_count - (column_count - 1) ** 2 / (exposures - 1)
        )
        denominator = min(corrected_rows - 1, corrected_columns - 1)
        if denominator <= 0:
            return 0.0
        return min(math.sqrt(corrected_phi / denominator), 1.0)

    def residual(self, category: Category) -> float:
        """Return the current category's Pearson residual in the hit column."""
        hits, support = self.cell_counts(category)
        if support == 0 or self.total_exposures == 0:
            return 0.0
        expected = support * self.total_hits / self.total_exposures
        if expected <= 0:
            return 0.0
        return (hits - expected) / math.sqrt(expected)

    def evidence(
        self,
        category: Category,
        prior_probability: float,
        prior_strength: float,
    ) -> ContingencyEvidence:
        """Return a smoothed estimate and association-weighted logit adjustment."""
        hits, support = self.cell_counts(category)
        probability = (hits + prior_strength * prior_probability) / (
            support + prior_strength
        )
        residual = self.residual(category)
        cramers_v = self.corrected_cramers_v()
        reliability = support / (support + prior_strength)
        raw_delta = _logit(probability) - _logit(prior_probability)
        signed_delta = (
            0.0
            if residual == 0
            else math.copysign(abs(raw_delta), residual)
        )
        return ContingencyEvidence(
            probability=probability,
            support=support,
            hits=hits,
            chi_square=self.chi_square(),
            residual=residual,
            cramers_v=cramers_v,
            adjustment=signed_delta * cramers_v * reliability,
        )


def _logit(probability: float) -> float:
    bounded = min(max(probability, _PROBABILITY_EPSILON), 1 - _PROBABILITY_EPSILON)
    return math.log(bounded / (1 - bounded))


def _sigmoid(value: float) -> float:
    bounded = min(max(value, -35.0), 35.0)
    return 1 / (1 + math.exp(-bounded))


class CategoricalChiSquareModel:
    """Learn one exact categorical dependency system for each number 1-49."""

    _VIEW_NAMES = (
        "gap",
        "left",
        "right",
        "gap_left",
        "gap_right",
        "left_right",
        "triple",
    )

    def __init__(self) -> None:
        self.draw_count = 0
        self.last_seen: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.last_left_space: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.last_right_space: list[int | None] = [None] * (_NUMBER_COUNT + 1)
        self.number_hits = [0] * (_NUMBER_COUNT + 1)
        self.number_exposures = [0] * (_NUMBER_COUNT + 1)
        self.tables = {
            view: [ContingencyTable() for _ in range(_NUMBER_COUNT + 1)]
            for view in self._VIEW_NAMES
        }

    def _state(self, number: int) -> tuple[int, int | None, int | None]:
        seen_at = self.last_seen[number]
        gap = (
            self.draw_count
            if seen_at is None
            else self.draw_count - seen_at - 1
        )
        return (
            gap,
            self.last_left_space[number],
            self.last_right_space[number],
        )

    @staticmethod
    def _categories(
        state: tuple[int, int | None, int | None],
    ) -> tuple[Category, ...]:
        gap, left, right = state
        return (
            gap,
            left,
            right,
            (gap, left),
            (gap, right),
            (left, right),
            (gap, left, right),
        )

    def learn(self, drawn: set[int]) -> None:
        """Record outcomes against state captured before the completed draw."""
        for number in range(1, _NUMBER_COUNT + 1):
            hit = number in drawn
            self.number_hits[number] += int(hit)
            self.number_exposures[number] += 1
            categories = self._categories(self._state(number))
            for view, category in zip(self._VIEW_NAMES, categories, strict=True):
                self.tables[view][number].observe(category, hit)

    def remember(self, drawn: set[int]) -> None:
        """Advance exact last-seen and circular space state after learning."""
        ordered = sorted(drawn)
        left_spaces = (
            (ordered[0] - 1) + (_NUMBER_COUNT - ordered[-1]),
            *(right - left - 1 for left, right in zip(ordered, ordered[1:])),
        )
        right_spaces = (*left_spaces[1:], left_spaces[0])
        for number, left, right in zip(
            ordered,
            left_spaces,
            right_spaces,
            strict=True,
        ):
            self.last_seen[number] = self.draw_count
            self.last_left_space[number] = left
            self.last_right_space[number] = right
        self.draw_count += 1

    def _baseline(self, number: int) -> float:
        return (
            self.number_hits[number] + _BASELINE_PRIOR_HITS
        ) / (
            self.number_exposures[number] + _BASELINE_PRIOR_EXPOSURES
        )

    def _score_number(
        self,
        number: int,
    ) -> tuple[float, tuple[str, ...]]:
        gap, left, right = self._state(number)
        categories = self._categories((gap, left, right))
        baseline = self._baseline(number)

        singles = tuple(
            self.tables[view][number].evidence(
                category,
                baseline,
                _SINGLE_PRIOR_STRENGTH,
            )
            for view, category in zip(
                self._VIEW_NAMES[:3],
                categories[:3],
                strict=True,
            )
        )
        pair_priors = (
            (singles[0].probability + singles[1].probability) / 2,
            (singles[0].probability + singles[2].probability) / 2,
            (singles[1].probability + singles[2].probability) / 2,
        )
        pairs = tuple(
            self.tables[view][number].evidence(
                category,
                prior,
                _PAIR_PRIOR_STRENGTH,
            )
            for view, category, prior in zip(
                self._VIEW_NAMES[3:6],
                categories[3:6],
                pair_priors,
                strict=True,
            )
        )
        triple_prior = sum(item.probability for item in pairs) / len(pairs)
        triple = self.tables["triple"][number].evidence(
            categories[6],
            triple_prior,
            _TRIPLE_PRIOR_STRENGTH,
        )

        single_adjustment = sum(item.adjustment for item in singles) / len(singles)
        pair_adjustment = sum(item.adjustment for item in pairs) / len(pairs)
        probability = _sigmoid(
            _logit(baseline)
            + single_adjustment
            + pair_adjustment
            + triple.adjustment
        )
        lift = probability / baseline if baseline > 0 else 1.0
        if triple.adjustment:
            backoff = "triple"
        elif any(item.adjustment for item in pairs):
            backoff = "pair"
        elif any(item.adjustment for item in singles):
            backoff = "single"
        else:
            backoff = "baseline"

        def state_value(value: int | None) -> str:
            return "unseen" if value is None else str(value)

        def evidence_line(label: str, item: ContingencyEvidence) -> str:
            return (
                f"{label}: support {item.support}, residual {item.residual:+.3f}, "
                f"corrected V {item.cramers_v:.3f}, chi-square {item.chi_square:.3f}"
            )

        details = (
            (
                f"Exact state gap {gap}, left {state_value(left)}, "
                f"right {state_value(right)}"
            ),
            (
                f"Estimated probability {probability:.2%}; baseline "
                f"{baseline:.2%}; lift {lift:.3f}x"
            ),
            evidence_line("Gap", singles[0]),
            evidence_line("Left", singles[1]),
            evidence_line("Right", singles[2]),
            f"Triple support {triple.support}; effective backoff {backoff}",
        )
        return probability, details

    def scores_and_details(
        self,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        """Return next-draw probabilities and audit details for all candidates."""
        scores: dict[int, float] = {}
        details: dict[int, tuple[str, ...]] = {}
        for number in range(1, _NUMBER_COUNT + 1):
            scores[number], details[number] = self._score_number(number)
        return scores, details
