"""Guarded gap-space residual corrections for the Markov 100 strategy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import mean, median
from typing import Literal

NUMBER_COUNT = 49
NUMBERS_PER_DRAW = 6
BASE_HIT_RATE = NUMBERS_PER_DRAW / NUMBER_COUNT
CORRECTION_CAP = 0.02
MINIMUM_PAIR_SUPPORT = 12
SHADOW_ACTIVATION_RESULTS = 100
SHADOW_WINDOW = 120
SHADOW_ACTIVATION_GAIN = 3

EvidenceVariant = Literal["historical", "fresh", "combined"]
GroupName = Literal["historical", "fresh"]


@dataclass(frozen=True, slots=True, order=True)
class MkgsvConfig:
    """Residual strengths and evidence selection fixed on development data."""

    single_strength: float
    pair_strength: float
    triple_strength: float
    evidence_variant: EvidenceVariant
    replacement_margin: float


# Selected on the fixed 200-draw development-validation partition.
SELECTED_MKGSV_CONFIG = MkgsvConfig(32.0, 128.0, 512.0, "historical", 0.0025)


@dataclass(slots=True)
class ResidualCounts:
    """Observed and Markov-expected hits for one categorical state."""

    actual_hits: int = 0
    expected_hits: float = 0.0
    exposures: int = 0

    def observe(self, hit: bool, expected_probability: float) -> None:
        self.actual_hits += int(hit)
        self.expected_hits += expected_probability
        self.exposures += 1

    def correction(self, strength: float) -> float:
        return (self.actual_hits - self.expected_hits) / (
            self.exposures + strength
        )


@dataclass(frozen=True, slots=True)
class FeatureGroupState:
    """One ordered historical or fresh space state."""

    left_space: int
    right_space: int
    left_bucket: int
    right_bucket: int


@dataclass(frozen=True, slots=True)
class CandidateState:
    """Leakage-safe feature state for one candidate."""

    number: int
    gap: int
    gap_bucket: int
    historical: FeatureGroupState | None
    fresh: FeatureGroupState | None


@dataclass(frozen=True, slots=True)
class GroupEvidence:
    """Residual contribution and support for one space feature group."""

    correction: float
    single_corrections: tuple[float, float]
    pair_corrections: tuple[float, float]
    triple_correction: float
    single_supports: tuple[int, int]
    pair_supports: tuple[int, int]
    triple_support: int

    @property
    def minimum_pair_support(self) -> int:
        return min(self.pair_supports)


@dataclass(frozen=True, slots=True)
class MkgsvScore:
    """Champion probability and guarded residual evidence for one number."""

    state: CandidateState
    base_probability: float
    corrected_probability: float
    correction: float
    historical_evidence: GroupEvidence | None
    fresh_evidence: GroupEvidence | None

    @property
    def number(self) -> int:
        return self.state.number


@dataclass(frozen=True, slots=True)
class MkgsvDecision:
    """One champion, shadow, and gated MKGSV prediction."""

    scores: dict[int, MkgsvScore]
    base_ranking: tuple[int, ...]
    shadow_ranking: tuple[int, ...]
    output_ranking: tuple[int, ...]
    ranking_scores: dict[int, float]
    proposed_insider: int | None
    proposed_outsider: int | None
    correction_active: bool
    shadow_results: int
    trailing_shadow_gain: int
    status: str

    @property
    def base_ticket(self) -> tuple[int, ...]:
        return self.base_ranking[:NUMBERS_PER_DRAW]

    @property
    def shadow_ticket(self) -> tuple[int, ...]:
        return self.shadow_ranking[:NUMBERS_PER_DRAW]

    @property
    def output_ticket(self) -> tuple[int, ...]:
        return self.output_ranking[:NUMBERS_PER_DRAW]


@dataclass(frozen=True, slots=True)
class _PendingObservation:
    expected_probability: float
    state: CandidateState


@dataclass(frozen=True, slots=True)
class _PendingPrediction:
    observations: dict[int, _PendingObservation]
    base_ticket: tuple[int, ...]
    shadow_ticket: tuple[int, ...]


def mkgsv_configurations() -> tuple[MkgsvConfig, ...]:
    """Return the fixed v2 development-only configuration grid."""
    return tuple(
        MkgsvConfig(single, pair, triple, variant, margin)
        for single in (32.0, 64.0)
        for pair in (128.0, 256.0)
        for triple in (256.0, 512.0)
        for variant in ("historical", "fresh", "combined")
        for margin in (0.0025, 0.005)
    )


def gap_bucket(gap: int) -> int:
    """Map an ordinal gap into a stable low-cardinality bucket."""
    if gap <= 2:
        return max(gap, 0)
    if gap <= 4:
        return 3
    if gap <= 7:
        return 4
    if gap <= 12:
        return 5
    if gap <= 20:
        return 6
    return 7


def space_bucket(space: int) -> int:
    """Map a circular space into a stable low-cardinality bucket."""
    if space <= 2:
        return max(space, 0)
    if space <= 4:
        return 3
    if space <= 7:
        return 4
    if space <= 11:
        return 5
    return 6


def _ranking_scores(ranking: tuple[int, ...]) -> dict[int, float]:
    denominator = NUMBER_COUNT - 1
    return {
        number: (NUMBER_COUNT - rank) / denominator
        for rank, number in enumerate(ranking, start=1)
    }


class MkgsvModel:
    """Learn conservative residual corrections around a Markov champion."""

    def __init__(self, config: MkgsvConfig = SELECTED_MKGSV_CONFIG) -> None:
        self.config = config
        self.draw_count = 0
        self.last_seen: list[int | None] = [None] * (NUMBER_COUNT + 1)
        self.left_spaces: list[int | None] = [None] * (NUMBER_COUNT + 1)
        self.right_spaces: list[int | None] = [None] * (NUMBER_COUNT + 1)
        self.latest_draw: set[int] = set()
        self.single_counts: dict[tuple[GroupName, str, int], ResidualCounts] = {}
        self.pair_counts: dict[
            tuple[GroupName, str, int, int], ResidualCounts
        ] = {}
        self.triple_counts: dict[
            tuple[GroupName, int, int, int], ResidualCounts
        ] = {}
        self.pending: _PendingPrediction | None = None
        self.shadow_results = 0
        self.shadow_deltas: deque[int] = deque(maxlen=SHADOW_WINDOW)
        self.correction_active = False
        self.proposal_count = 0
        self.activation_count = 0

    def _gap(self, number: int) -> int:
        seen_at = self.last_seen[number]
        return self.draw_count if seen_at is None else self.draw_count - seen_at - 1

    @staticmethod
    def _counts[K](store: dict[K, ResidualCounts], key: K) -> ResidualCounts:
        return store.setdefault(key, ResidualCounts())

    def _fresh_spaces(self, number: int) -> tuple[int, int] | None:
        neighbors = self.latest_draw.difference({number})
        if not neighbors:
            return None
        left = min((number - neighbor - 1) % NUMBER_COUNT for neighbor in neighbors)
        right = min((neighbor - number - 1) % NUMBER_COUNT for neighbor in neighbors)
        return left, right

    @staticmethod
    def _group_state(left: int, right: int) -> FeatureGroupState:
        return FeatureGroupState(
            left_space=left,
            right_space=right,
            left_bucket=space_bucket(left),
            right_bucket=space_bucket(right),
        )

    def state(self, number: int) -> CandidateState:
        """Return the current historical and fresh feature state."""
        gap = self._gap(number)
        historical_left = self.left_spaces[number]
        historical_right = self.right_spaces[number]
        historical = (
            None
            if historical_left is None or historical_right is None
            else self._group_state(historical_left, historical_right)
        )
        fresh_spaces = self._fresh_spaces(number)
        fresh = (
            None
            if fresh_spaces is None
            else self._group_state(fresh_spaces[0], fresh_spaces[1])
        )
        return CandidateState(number, gap, gap_bucket(gap), historical, fresh)

    def _observe_group(
        self,
        group_name: GroupName,
        gap: int,
        group: FeatureGroupState,
        hit: bool,
        expected_probability: float,
    ) -> None:
        for direction, bucket in (
            ("left", group.left_bucket),
            ("right", group.right_bucket),
        ):
            self._counts(
                self.single_counts,
                (group_name, direction, bucket),
            ).observe(hit, expected_probability)
            self._counts(
                self.pair_counts,
                (group_name, direction, gap, bucket),
            ).observe(hit, expected_probability)
        self._counts(
            self.triple_counts,
            (group_name, gap, group.left_bucket, group.right_bucket),
        ).observe(hit, expected_probability)

    def train(self, drawn: set[int]) -> None:
        """Settle the prior prediction after its target outcome is available."""
        pending = self.pending
        if pending is None:
            return
        for number, observation in pending.observations.items():
            hit = number in drawn
            state = observation.state
            if state.historical is not None:
                self._observe_group(
                    "historical",
                    state.gap_bucket,
                    state.historical,
                    hit,
                    observation.expected_probability,
                )
            if state.fresh is not None:
                self._observe_group(
                    "fresh",
                    state.gap_bucket,
                    state.fresh,
                    hit,
                    observation.expected_probability,
                )

        base_hits = len(set(pending.base_ticket).intersection(drawn))
        shadow_hits = len(set(pending.shadow_ticket).intersection(drawn))
        self.shadow_deltas.append(shadow_hits - base_hits)
        self.shadow_results += 1
        trailing_gain = sum(self.shadow_deltas)
        if self.correction_active:
            if trailing_gain <= 0:
                self.correction_active = False
        elif (
            self.shadow_results >= SHADOW_ACTIVATION_RESULTS
            and trailing_gain >= SHADOW_ACTIVATION_GAIN
        ):
            self.correction_active = True
            self.activation_count += 1
        self.pending = None

    def remember(self, drawn: set[int]) -> None:
        """Advance last-seen, historical-space, and latest-draw state."""
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
        self.latest_draw = set(drawn)
        self.draw_count += 1

    def _group_evidence(
        self,
        group_name: GroupName,
        gap: int,
        group: FeatureGroupState | None,
    ) -> GroupEvidence | None:
        if group is None:
            return None
        single_keys = (
            (group_name, "left", group.left_bucket),
            (group_name, "right", group.right_bucket),
        )
        pair_keys = (
            (group_name, "left", gap, group.left_bucket),
            (group_name, "right", gap, group.right_bucket),
        )
        triple_key = (group_name, gap, group.left_bucket, group.right_bucket)
        single_counts = tuple(self.single_counts.get(key) for key in single_keys)
        pair_counts = tuple(self.pair_counts.get(key) for key in pair_keys)
        triple_counts = self.triple_counts.get(triple_key)
        single_corrections = (
            0.0
            if single_counts[0] is None
            else single_counts[0].correction(self.config.single_strength),
            0.0
            if single_counts[1] is None
            else single_counts[1].correction(self.config.single_strength),
        )
        pair_corrections = (
            0.0
            if pair_counts[0] is None
            else pair_counts[0].correction(self.config.pair_strength),
            0.0
            if pair_counts[1] is None
            else pair_counts[1].correction(self.config.pair_strength),
        )
        triple_correction = (
            0.0
            if triple_counts is None
            else triple_counts.correction(self.config.triple_strength)
        )
        correction = (
            0.50 * mean(single_corrections)
            + 0.35 * mean(pair_corrections)
            + 0.15 * triple_correction
        )
        return GroupEvidence(
            correction=correction,
            single_corrections=single_corrections,
            pair_corrections=pair_corrections,
            triple_correction=triple_correction,
            single_supports=(
                0 if single_counts[0] is None else single_counts[0].exposures,
                0 if single_counts[1] is None else single_counts[1].exposures,
            ),
            pair_supports=(
                0 if pair_counts[0] is None else pair_counts[0].exposures,
                0 if pair_counts[1] is None else pair_counts[1].exposures,
            ),
            triple_support=0 if triple_counts is None else triple_counts.exposures,
        )

    def _combined_correction(
        self,
        historical: GroupEvidence | None,
        fresh: GroupEvidence | None,
    ) -> float:
        variant = self.config.evidence_variant
        if variant == "historical":
            return 0.0 if historical is None else historical.correction
        if variant == "fresh":
            return 0.0 if fresh is None else fresh.correction
        available = [
            evidence.correction
            for evidence in (historical, fresh)
            if evidence is not None
        ]
        return mean(available) if available else 0.0

    def scores(self, base_probabilities: dict[int, float]) -> dict[int, MkgsvScore]:
        """Score all candidates with conservative residual corrections."""
        rows: dict[int, MkgsvScore] = {}
        for number in range(1, NUMBER_COUNT + 1):
            state = self.state(number)
            historical = self._group_evidence(
                "historical",
                state.gap_bucket,
                state.historical,
            )
            fresh = self._group_evidence("fresh", state.gap_bucket, state.fresh)
            correction = max(
                -CORRECTION_CAP,
                min(CORRECTION_CAP, self._combined_correction(historical, fresh)),
            )
            base_probability = base_probabilities[number]
            rows[number] = MkgsvScore(
                state=state,
                base_probability=base_probability,
                corrected_probability=max(
                    0.0,
                    min(1.0, base_probability + correction),
                ),
                correction=correction,
                historical_evidence=historical,
                fresh_evidence=fresh,
            )
        return rows

    @staticmethod
    def _supported(evidence: GroupEvidence | None) -> bool:
        return (
            evidence is not None
            and evidence.minimum_pair_support >= MINIMUM_PAIR_SUPPORT
        )

    def _candidate_supported(self, row: MkgsvScore) -> bool:
        variant = self.config.evidence_variant
        if variant == "historical":
            return self._supported(row.historical_evidence)
        if variant == "fresh":
            return self._supported(row.fresh_evidence)
        return self._supported(row.historical_evidence) and self._supported(
            row.fresh_evidence
        )

    def _groups_agree(self, outsider: MkgsvScore, insider: MkgsvScore) -> bool:
        if self.config.evidence_variant != "combined":
            return True
        outsider_historical = outsider.historical_evidence
        outsider_fresh = outsider.fresh_evidence
        insider_historical = insider.historical_evidence
        insider_fresh = insider.fresh_evidence
        if None in (
            outsider_historical,
            outsider_fresh,
            insider_historical,
            insider_fresh,
        ):
            return False
        assert outsider_historical is not None
        assert outsider_fresh is not None
        assert insider_historical is not None
        assert insider_fresh is not None
        return (
            outsider_historical.correction > insider_historical.correction
            and outsider_fresh.correction > insider_fresh.correction
        )

    def _proposal(
        self,
        rows: dict[int, MkgsvScore],
        base_ranking: tuple[int, ...],
    ) -> tuple[int | None, int | None, tuple[int, ...]]:
        insider = base_ranking[NUMBERS_PER_DRAW - 1]
        insider_row = rows[insider]
        if insider_row.correction >= 0 or not self._candidate_supported(insider_row):
            return None, None, base_ranking
        eligible: list[tuple[int, int]] = []
        for rank, outsider in enumerate(
            base_ranking[NUMBERS_PER_DRAW : NUMBERS_PER_DRAW * 2],
            start=NUMBERS_PER_DRAW + 1,
        ):
            outsider_row = rows[outsider]
            if outsider_row.correction <= 0:
                continue
            if not self._candidate_supported(outsider_row):
                continue
            if not self._groups_agree(outsider_row, insider_row):
                continue
            if (
                outsider_row.corrected_probability
                < insider_row.corrected_probability
                + self.config.replacement_margin
            ):
                continue
            eligible.append((rank, outsider))
        if not eligible:
            return None, None, base_ranking
        _rank, outsider = max(
            eligible,
            key=lambda item: (
                rows[item[1]].corrected_probability,
                -item[0],
                -item[1],
            ),
        )
        shadow = list(base_ranking)
        outsider_index = shadow.index(outsider)
        shadow[NUMBERS_PER_DRAW - 1], shadow[outsider_index] = (
            shadow[outsider_index],
            shadow[NUMBERS_PER_DRAW - 1],
        )
        return insider, outsider, tuple(shadow)

    def _status(self, proposal_exists: bool) -> str:
        trailing_gain = sum(self.shadow_deltas)
        if self.correction_active:
            return (
                f"Active after {self.shadow_results} shadow results; "
                f"trailing-{SHADOW_WINDOW} gain {trailing_gain:+d}"
            )
        if self.shadow_results < SHADOW_ACTIVATION_RESULTS:
            return (
                f"Shadow warm-up {self.shadow_results}/"
                f"{SHADOW_ACTIVATION_RESULTS}"
            )
        if not proposal_exists:
            return f"Inactive; no supported boundary replacement (gain {trailing_gain:+d})"
        return (
            f"Inactive; trailing-{SHADOW_WINDOW} gain {trailing_gain:+d} "
            f"is below +{SHADOW_ACTIVATION_GAIN}"
        )

    def predict(
        self,
        base_probabilities: dict[int, float],
        base_ranking: tuple[int, ...],
    ) -> MkgsvDecision:
        """Create and retain the next leakage-safe shadow prediction."""
        if set(base_ranking) != set(range(1, NUMBER_COUNT + 1)):
            raise ValueError("MKGSV requires a complete 1-49 champion ranking")
        rows = self.scores(base_probabilities)
        insider, outsider, shadow_ranking = self._proposal(rows, base_ranking)
        if outsider is not None:
            self.proposal_count += 1
        output_ranking = shadow_ranking if self.correction_active else base_ranking
        self.pending = _PendingPrediction(
            observations={
                number: _PendingObservation(row.base_probability, row.state)
                for number, row in rows.items()
            },
            base_ticket=base_ranking[:NUMBERS_PER_DRAW],
            shadow_ticket=shadow_ranking[:NUMBERS_PER_DRAW],
        )
        return MkgsvDecision(
            scores=rows,
            base_ranking=base_ranking,
            shadow_ranking=shadow_ranking,
            output_ranking=output_ranking,
            ranking_scores=_ranking_scores(output_ranking),
            proposed_insider=insider,
            proposed_outsider=outsider,
            correction_active=self.correction_active,
            shadow_results=self.shadow_results,
            trailing_shadow_gain=sum(self.shadow_deltas),
            status=self._status(outsider is not None),
        )

    def state_support_distribution(self) -> dict[str, int | float]:
        """Summarize binned residual-state support for benchmark reporting."""
        result: dict[str, int | float] = {}
        for group_name in ("historical", "fresh"):
            supports = sorted(
                counts.exposures
                for key, counts in self.triple_counts.items()
                if key[0] == group_name
            )
            prefix = f"{group_name}Triple"
            result[f"{prefix}States"] = len(supports)
            result[f"{prefix}Exposures"] = sum(supports)
            result[f"{prefix}MedianSupport"] = (
                0.0 if not supports else float(median(supports))
            )
        return result
