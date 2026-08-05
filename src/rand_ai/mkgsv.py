"""Hierarchical Markov scoring for per-number gap/space vectors."""

from __future__ import annotations

from dataclasses import dataclass

NUMBER_COUNT = 49
NUMBERS_PER_DRAW = 6
BASE_HIT_RATE = NUMBERS_PER_DRAW / NUMBER_COUNT


@dataclass(frozen=True, slots=True, order=True)
class MkgsvConfig:
    """Bayesian prior strengths for the three hierarchy levels."""

    single_strength: float
    pair_strength: float
    triple_strength: float


# Development-selected configuration preserved even though the promotion gate failed.
SELECTED_MKGSV_CONFIG = MkgsvConfig(64.0, 4.0, 2.0)


@dataclass(slots=True)
class BinaryCounts:
    """Hits and exposures observed for one state."""

    hits: int = 0
    exposures: int = 0

    def observe(self, hit: bool) -> None:
        self.hits += int(hit)
        self.exposures += 1


@dataclass(frozen=True, slots=True)
class MkgsvScore:
    """Posterior and evidence used to rank one candidate number."""

    number: int
    probability: float
    gap: int
    left_space: int | None
    right_space: int | None
    single_probabilities: tuple[float, ...]
    pair_probabilities: tuple[float, ...]
    single_supports: tuple[int, ...]
    pair_supports: tuple[int, ...]
    triple_support: int

    @property
    def backoff_path(self) -> str:
        if self.left_space is None or self.right_space is None:
            return "gap-only"
        if self.triple_support:
            return "triple → pairs → singles → global"
        if any(self.pair_supports):
            return "pairs → singles → global"
        return "singles → global"


def mkgsv_configurations() -> tuple[MkgsvConfig, ...]:
    """Return the fixed development-only hyperparameter grid."""
    return tuple(
        MkgsvConfig(single, pair, triple)
        for single in (8.0, 24.0, 64.0)
        for pair in (4.0, 12.0, 32.0)
        for triple in (2.0, 8.0, 16.0)
    )


class MkgsvModel:
    """Learn next-draw hits from ordered ``(gap, left, right)`` states."""

    def __init__(self, config: MkgsvConfig = SELECTED_MKGSV_CONFIG) -> None:
        self.config = config
        self.draw_count = 0
        self.last_seen: list[int | None] = [None] * (NUMBER_COUNT + 1)
        self.left_spaces: list[int | None] = [None] * (NUMBER_COUNT + 1)
        self.right_spaces: list[int | None] = [None] * (NUMBER_COUNT + 1)
        self.x_counts: dict[int, BinaryCounts] = {}
        self.y_counts: dict[int, BinaryCounts] = {}
        self.z_counts: dict[int, BinaryCounts] = {}
        self.xy_counts: dict[tuple[int, int], BinaryCounts] = {}
        self.xz_counts: dict[tuple[int, int], BinaryCounts] = {}
        self.yz_counts: dict[tuple[int, int], BinaryCounts] = {}
        self.xyz_counts: dict[tuple[int, int, int], BinaryCounts] = {}

    def _gap(self, number: int) -> int:
        seen_at = self.last_seen[number]
        return self.draw_count if seen_at is None else self.draw_count - seen_at - 1

    @staticmethod
    def _counts[K](store: dict[K, BinaryCounts], key: K) -> BinaryCounts:
        return store.setdefault(key, BinaryCounts())

    @staticmethod
    def _posterior(
        counts: BinaryCounts | None,
        prior: float,
        strength: float,
    ) -> float:
        hits = 0 if counts is None else counts.hits
        exposures = 0 if counts is None else counts.exposures
        return (hits + strength * prior) / (exposures + strength)

    def train(self, drawn: set[int]) -> None:
        """Learn a draw from states that existed before its outcome."""
        for number in range(1, NUMBER_COUNT + 1):
            hit = number in drawn
            gap = self._gap(number)
            self._counts(self.x_counts, gap).observe(hit)
            left = self.left_spaces[number]
            right = self.right_spaces[number]
            if left is None or right is None:
                continue
            self._counts(self.y_counts, left).observe(hit)
            self._counts(self.z_counts, right).observe(hit)
            self._counts(self.xy_counts, (gap, left)).observe(hit)
            self._counts(self.xz_counts, (gap, right)).observe(hit)
            self._counts(self.yz_counts, (left, right)).observe(hit)
            self._counts(self.xyz_counts, (gap, left, right)).observe(hit)

    def remember(self, drawn: set[int]) -> None:
        """Advance last-seen and ordered circular-space state."""
        ordered = sorted(drawn)
        if len(ordered) != NUMBERS_PER_DRAW:
            raise ValueError("MKGSV requires exactly six drawn numbers")
        left_spaces = (
            (ordered[0] - 1) + (NUMBER_COUNT - ordered[-1]),
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
            self.left_spaces[number] = left
            self.right_spaces[number] = right
        self.draw_count += 1

    def _single_probability(
        self,
        store: dict[int, BinaryCounts],
        value: int,
    ) -> float:
        return self._posterior(
            store.get(value),
            BASE_HIT_RATE,
            self.config.single_strength,
        )

    def score(self, number: int) -> MkgsvScore:
        """Return the hierarchical next-draw posterior for one number."""
        gap = self._gap(number)
        left = self.left_spaces[number]
        right = self.right_spaces[number]
        gap_probability = self._single_probability(self.x_counts, gap)
        gap_support = self.x_counts.get(gap, BinaryCounts()).exposures
        if left is None or right is None:
            return MkgsvScore(
                number=number,
                probability=gap_probability,
                gap=gap,
                left_space=None,
                right_space=None,
                single_probabilities=(gap_probability,),
                pair_probabilities=(),
                single_supports=(gap_support,),
                pair_supports=(),
                triple_support=0,
            )

        left_probability = self._single_probability(self.y_counts, left)
        right_probability = self._single_probability(self.z_counts, right)
        single_probabilities = (
            gap_probability,
            left_probability,
            right_probability,
        )
        single_supports = (
            gap_support,
            self.y_counts.get(left, BinaryCounts()).exposures,
            self.z_counts.get(right, BinaryCounts()).exposures,
        )
        pair_inputs = (
            (self.xy_counts.get((gap, left)), (gap_probability + left_probability) / 2),
            (
                self.xz_counts.get((gap, right)),
                (gap_probability + right_probability) / 2,
            ),
            (
                self.yz_counts.get((left, right)),
                (left_probability + right_probability) / 2,
            ),
        )
        pair_probabilities = tuple(
            self._posterior(counts, prior, self.config.pair_strength)
            for counts, prior in pair_inputs
        )
        pair_supports = tuple(
            0 if counts is None else counts.exposures for counts, _prior in pair_inputs
        )
        triple = self.xyz_counts.get((gap, left, right))
        probability = self._posterior(
            triple,
            sum(pair_probabilities) / len(pair_probabilities),
            self.config.triple_strength,
        )
        return MkgsvScore(
            number=number,
            probability=probability,
            gap=gap,
            left_space=left,
            right_space=right,
            single_probabilities=single_probabilities,
            pair_probabilities=pair_probabilities,
            single_supports=single_supports,
            pair_supports=pair_supports,
            triple_support=0 if triple is None else triple.exposures,
        )

    def scores(self) -> dict[int, MkgsvScore]:
        """Score all 49 candidate numbers."""
        return {number: self.score(number) for number in range(1, NUMBER_COUNT + 1)}

    def state_support_distribution(self) -> dict[str, int | float]:
        """Summarize exact triple sparsity for benchmark reporting."""
        supports = sorted(counts.exposures for counts in self.xyz_counts.values())
        if not supports:
            return {
                "uniqueTripleStates": 0,
                "tripleExposures": 0,
                "medianTripleSupport": 0.0,
                "singleExposureStates": 0,
                "doubleExposureStates": 0,
                "threeToFiveExposureStates": 0,
                "sixToTenExposureStates": 0,
                "overTenExposureStates": 0,
            }
        middle = len(supports) // 2
        median = (
            float(supports[middle])
            if len(supports) % 2
            else (supports[middle - 1] + supports[middle]) / 2
        )
        return {
            "uniqueTripleStates": len(supports),
            "tripleExposures": sum(supports),
            "medianTripleSupport": median,
            "singleExposureStates": sum(support == 1 for support in supports),
            "doubleExposureStates": sum(support == 2 for support in supports),
            "threeToFiveExposureStates": sum(3 <= support <= 5 for support in supports),
            "sixToTenExposureStates": sum(6 <= support <= 10 for support in supports),
            "overTenExposureStates": sum(support > 10 for support in supports),
        }
