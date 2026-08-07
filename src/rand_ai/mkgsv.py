"""Ticket-level gap-space motifs guarded by the Markov 100 champion."""

from __future__ import annotations

import math
import random
from collections import Counter, deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from statistics import mean, median
from typing import Literal

NUMBER_COUNT = 49
NUMBERS_PER_DRAW = 6
BASE_HIT_RATE = NUMBERS_PER_DRAW / NUMBER_COUNT
NULL_TICKETS_PER_PREDICTION = 128
NULL_SEED = 20260626
RECENT_WINDOW = 120
SHADOW_ACTIVATION_RESULTS = 100
SHADOW_SHORT_WINDOW = 60
SHADOW_LONG_WINDOW = 120
SHADOW_SHORT_GAIN = 2
SHADOW_LONG_GAIN = 5

MotifVariant = Literal[
    "singles",
    "singles+doubles",
    "singles+doubles+triples",
    "all-with-transitions",
]
FamilyCounts = dict[str, Counter[Hashable]]
TransitionKey = tuple[tuple[int, int], tuple[int, int]]
TransitionCounts = Counter[TransitionKey]


@dataclass(frozen=True, slots=True, order=True)
class MkgsvConfig:
    """Development-only motif hyperparameters."""

    prior_strength: float
    motif_variant: MotifVariant
    influence: float


# Best research configuration on the fixed validation partition. The explicit
# correction-off selection below prevents it from changing production output.
SELECTED_MKGSV_CONFIG = MkgsvConfig(
    24.0, "singles+doubles+triples", 0.20
)
# The strict v3 gate controls production behavior, not runtime history length.
MKGSV_PROMOTED = False


@dataclass(frozen=True, slots=True)
class TicketVector:
    """Prospective gap and ordered circular spaces for one ticket position."""

    number: int
    gap: int
    gap_class: int
    left_space: int
    right_space: int
    left_class: int
    right_class: int
    space_shape: int

    @property
    def token(self) -> tuple[int, int]:
        return self.gap_class, self.space_shape


@dataclass(frozen=True, slots=True)
class TicketEvidence:
    """Smoothed log-lift components for one complete ticket."""

    single: float
    double: float
    triple: float
    transition: float
    total: float
    supports: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class MkgsvScore:
    """Public per-number view of the ticket-level decision."""

    number: int
    champion_rank: int
    base_probability: float
    selected_vector: TicketVector | None
    selected_by_shadow: bool


@dataclass(frozen=True, slots=True)
class MkgsvDecision:
    """Champion, research shadow, and guarded output for one prediction."""

    scores: dict[int, MkgsvScore]
    base_ranking: tuple[int, ...]
    shadow_ranking: tuple[int, ...]
    output_ranking: tuple[int, ...]
    ranking_scores: dict[int, float]
    base_evidence: TicketEvidence
    shadow_evidence: TicketEvidence
    proposed_insider: int | None
    proposed_outsider: int | None
    correction_active: bool
    shadow_results: int
    lifetime_shadow_gain: int
    trailing_60_gain: int
    trailing_120_gain: int
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
class _MotifObservation:
    families: FamilyCounts
    vectors: tuple[TicketVector, ...]


@dataclass(frozen=True, slots=True)
class _PendingPrediction:
    gaps: dict[int, int]
    gap_classes: dict[int, int]
    null_families: FamilyCounts
    previous_tokens: tuple[tuple[int, int], ...]
    base_ticket: tuple[int, ...]
    shadow_ticket: tuple[int, ...]
    proposed: bool


@dataclass(frozen=True, slots=True)
class _RecentObservation:
    actual: FamilyCounts
    null: FamilyCounts
    transitions: TransitionCounts


@dataclass(frozen=True, slots=True)
class _TicketCandidate:
    ticket: tuple[int, ...]
    ranking: tuple[int, ...]
    outsider: int | None
    outsider_rank: int
    markov_score: float
    evidence: TicketEvidence
    total_score: float


_FAMILIES = (
    "single",
    "full_double",
    "gap_double",
    "space_double",
    "gap_triple",
    "space_triple",
)
_CARDINALITIES = {
    "single": 4 * 9,
    "full_double": (4 * 9) ** 2,
    "gap_double": 4**2,
    "space_double": 9**2,
    "gap_triple": 4**3,
    "space_triple": 9**3,
}
_COMPONENT_WEIGHTS = {
    "single": 0.20,
    "double": 0.25,
    "triple": 0.25,
    "transition": 0.30,
}


def mkgsv_configurations() -> tuple[MkgsvConfig, ...]:
    """Return the deterministic v3 development configuration grid."""
    return tuple(
        MkgsvConfig(prior, variant, influence)
        for prior in (8.0, 24.0, 64.0)
        for variant in (
            "singles",
            "singles+doubles",
            "singles+doubles+triples",
            "all-with-transitions",
        )
        for influence in (0.05, 0.10, 0.20)
    )


def gap_class(gap: int) -> int:
    """Map a nonnegative prospective gap to one of four coarse classes."""
    if gap <= 1:
        return 0
    if gap <= 4:
        return 1
    if gap <= 12:
        return 2
    return 3


def space_class(space: int) -> int:
    """Map an ordered circular space to tight, medium, or wide."""
    if space <= 2:
        return 0
    if space <= 7:
        return 1
    return 2


def ticket_vectors(
    ticket: Iterable[int],
    gaps: Mapping[int, int],
) -> tuple[TicketVector, ...]:
    """Construct ordered prospective vectors for a complete valid ticket."""
    ordered = tuple(sorted(ticket))
    if len(ordered) != NUMBERS_PER_DRAW or len(set(ordered)) != NUMBERS_PER_DRAW:
        raise ValueError("MKGSV tickets require six distinct numbers")
    if ordered[0] < 1 or ordered[-1] > NUMBER_COUNT:
        raise ValueError("MKGSV ticket numbers must be between 1 and 49")
    vectors: list[TicketVector] = []
    for index, number in enumerate(ordered):
        previous = ordered[index - 1]
        following = ordered[(index + 1) % NUMBERS_PER_DRAW]
        left = (number - previous - 1) % NUMBER_COUNT
        right = (following - number - 1) % NUMBER_COUNT
        left_bucket = space_class(left)
        right_bucket = space_class(right)
        gap = gaps[number]
        vectors.append(
            TicketVector(
                number=number,
                gap=gap,
                gap_class=gap_class(gap),
                left_space=left,
                right_space=right,
                left_class=left_bucket,
                right_class=right_bucket,
                space_shape=3 * left_bucket + right_bucket,
            )
        )
    return tuple(vectors)


def _families(vectors: Sequence[TicketVector]) -> FamilyCounts:
    tokens = tuple(vector.token for vector in vectors)
    gaps = tuple(vector.gap_class for vector in vectors)
    spaces = tuple(vector.space_shape for vector in vectors)
    result = {family: Counter() for family in _FAMILIES}
    for index in range(NUMBERS_PER_DRAW):
        next_index = (index + 1) % NUMBERS_PER_DRAW
        third_index = (index + 2) % NUMBERS_PER_DRAW
        result["single"][tokens[index]] += 1
        result["full_double"][(tokens[index], tokens[next_index])] += 1
        result["gap_double"][(gaps[index], gaps[next_index])] += 1
        result["space_double"][(spaces[index], spaces[next_index])] += 1
        result["gap_triple"][
            (gaps[index], gaps[next_index], gaps[third_index])
        ] += 1
        result["space_triple"][
            (spaces[index], spaces[next_index], spaces[third_index])
        ] += 1
    return result


def _empty_families() -> FamilyCounts:
    return {family: Counter() for family in _FAMILIES}


@cache
def _cached_null_families(
    draw_count: int,
    gap_values: tuple[int, ...],
) -> tuple[tuple[str, tuple[tuple[Hashable, int], ...]], ...]:
    """Build one immutable deterministic null shared by all configurations."""
    gaps = {
        number: gap_values[number - 1]
        for number in range(1, NUMBER_COUNT + 1)
    }
    generator = random.Random(NULL_SEED + draw_count)
    combined = _empty_families()
    population = range(1, NUMBER_COUNT + 1)
    for _ in range(NULL_TICKETS_PER_PREDICTION):
        ticket = generator.sample(population, NUMBERS_PER_DRAW)
        vectors = ticket_vectors(ticket, gaps)
        _merge_families(combined, _families(vectors))
    return tuple(
        (family, tuple(combined[family].items())) for family in _FAMILIES
    )


def _merge_families(target: FamilyCounts, source: FamilyCounts, sign: int = 1) -> None:
    for family in _FAMILIES:
        target[family].update(
            {key: sign * count for key, count in source[family].items()}
        )
        if sign < 0:
            target[family] += Counter()


def _ranking_scores(ranking: tuple[int, ...]) -> dict[int, float]:
    denominator = NUMBER_COUNT - 1
    return {
        number: (NUMBER_COUNT - rank) / denominator
        for rank, number in enumerate(ranking, start=1)
    }


def _logit(probability: float) -> float:
    bounded = min(1.0 - 1e-9, max(1e-9, probability))
    return math.log(bounded / (1.0 - bounded))


class MkgsvModel:
    """Learn ticket motifs while preserving Markov 100 as the champion."""

    def __init__(
        self,
        config: MkgsvConfig = SELECTED_MKGSV_CONFIG,
        *,
        promotion_enabled: bool = MKGSV_PROMOTED,
    ) -> None:
        self.config = config
        self.promotion_enabled = promotion_enabled
        self.draw_count = 0
        self.settled_draws = 0
        self.last_seen: list[int | None] = [None] * (NUMBER_COUNT + 1)
        self.actual_counts = _empty_families()
        self.null_counts = _empty_families()
        self.recent_actual_counts = _empty_families()
        self.recent_null_counts = _empty_families()
        self.transition_counts: TransitionCounts = Counter()
        self.recent_transition_counts: TransitionCounts = Counter()
        self.transition_source_totals: Counter[tuple[int, int]] = Counter()
        self.recent_transition_source_totals: Counter[tuple[int, int]] = Counter()
        self.recent_observations: deque[_RecentObservation] = deque()
        self.previous_actual_tokens: tuple[tuple[int, int], ...] = ()
        self.pending: _PendingPrediction | None = None
        self.shadow_results = 0
        self.lifetime_shadow_gain = 0
        self.shadow_deltas_60: deque[int] = deque(maxlen=SHADOW_SHORT_WINDOW)
        self.shadow_deltas_120: deque[int] = deque(maxlen=SHADOW_LONG_WINDOW)
        self.correction_active = False
        self.proposal_count = 0
        self.activation_count = 0

    def _gap(self, number: int) -> int:
        seen_at = self.last_seen[number]
        return self.draw_count if seen_at is None else self.draw_count - seen_at - 1

    def gaps(self) -> dict[int, int]:
        return {number: self._gap(number) for number in range(1, NUMBER_COUNT + 1)}

    @staticmethod
    def _observation(ticket: Iterable[int], gaps: Mapping[int, int]) -> _MotifObservation:
        vectors = ticket_vectors(ticket, gaps)
        return _MotifObservation(_families(vectors), vectors)

    def _null_observation(self, gaps: Mapping[int, int]) -> FamilyCounts:
        immutable = _cached_null_families(
            self.draw_count,
            tuple(gaps[number] for number in range(1, NUMBER_COUNT + 1)),
        )
        return {family: Counter(dict(items)) for family, items in immutable}

    @staticmethod
    def _transitions(
        previous_tokens: Sequence[tuple[int, int]],
        target_tokens: Sequence[tuple[int, int]],
    ) -> TransitionCounts:
        return Counter(
            (source, target)
            for source in previous_tokens
            for target in target_tokens
        )

    def _append_recent(
        self,
        actual: FamilyCounts,
        null: FamilyCounts,
        transitions: TransitionCounts,
    ) -> None:
        item = _RecentObservation(actual, null, transitions)
        self.recent_observations.append(item)
        _merge_families(self.recent_actual_counts, actual)
        _merge_families(self.recent_null_counts, null)
        self.recent_transition_counts.update(transitions)
        for (source, _target), count in transitions.items():
            self.recent_transition_source_totals[source] += count
        if len(self.recent_observations) <= RECENT_WINDOW:
            return
        expired = self.recent_observations.popleft()
        _merge_families(self.recent_actual_counts, expired.actual, -1)
        _merge_families(self.recent_null_counts, expired.null, -1)
        self.recent_transition_counts.subtract(expired.transitions)
        self.recent_transition_counts += Counter()
        for (source, _target), count in expired.transitions.items():
            self.recent_transition_source_totals[source] -= count
        self.recent_transition_source_totals += Counter()

    def _update_guard(self, delta: int) -> None:
        self.shadow_results += 1
        self.lifetime_shadow_gain += delta
        self.shadow_deltas_60.append(delta)
        self.shadow_deltas_120.append(delta)
        short_gain = sum(self.shadow_deltas_60)
        long_gain = sum(self.shadow_deltas_120)
        if not self.promotion_enabled:
            self.correction_active = False
            return
        if self.correction_active:
            if self.lifetime_shadow_gain <= 0 or short_gain <= 0 or long_gain <= 0:
                self.correction_active = False
            return
        if (
            self.shadow_results >= SHADOW_ACTIVATION_RESULTS
            and self.lifetime_shadow_gain > 0
            and len(self.shadow_deltas_60) == SHADOW_SHORT_WINDOW
            and short_gain >= SHADOW_SHORT_GAIN
            and long_gain >= SHADOW_LONG_GAIN
        ):
            self.correction_active = True
            self.activation_count += 1

    def train(self, drawn: set[int]) -> None:
        """Settle captured motif and shadow state after the target is known."""
        pending = self.pending
        if pending is None:
            return
        actual = self._observation(drawn, pending.gaps)
        transitions = self._transitions(
            pending.previous_tokens,
            tuple(vector.token for vector in actual.vectors),
        )
        _merge_families(self.actual_counts, actual.families)
        _merge_families(self.null_counts, pending.null_families)
        self.transition_counts.update(transitions)
        for (source, _target), count in transitions.items():
            self.transition_source_totals[source] += count
        self._append_recent(actual.families, pending.null_families, transitions)
        self.previous_actual_tokens = tuple(vector.token for vector in actual.vectors)
        self.settled_draws += 1
        if pending.proposed:
            base_hits = len(set(pending.base_ticket).intersection(drawn))
            shadow_hits = len(set(pending.shadow_ticket).intersection(drawn))
            self._update_guard(shadow_hits - base_hits)
        self.pending = None

    def remember(self, drawn: set[int]) -> None:
        """Advance only the last-seen state after learning the current outcome."""
        if len(drawn) != NUMBERS_PER_DRAW:
            raise ValueError("MKGSV requires exactly six drawn numbers")
        for number in drawn:
            self.last_seen[number] = self.draw_count
        self.draw_count += 1

    @staticmethod
    def _null_probability(
        family: str,
        key: Hashable,
        counts: FamilyCounts,
        total: int | None = None,
    ) -> float:
        resolved_total = sum(counts[family].values()) if total is None else total
        cardinality = _CARDINALITIES[family]
        return (counts[family][key] + 1.0 / cardinality) / (
            resolved_total + 1.0
        )

    def _family_lift(
        self,
        family: str,
        key: Hashable,
        *,
        recent: bool,
    ) -> tuple[float, int]:
        actual = self.recent_actual_counts if recent else self.actual_counts
        null = self.recent_null_counts if recent else self.null_counts
        observed_draws = (
            len(self.recent_observations) if recent else self.settled_draws
        )
        actual_total = observed_draws * NUMBERS_PER_DRAW
        null_total = actual_total * NULL_TICKETS_PER_PREDICTION
        if observed_draws == 0:
            actual_total = sum(actual[family].values())
            null_total = sum(null[family].values())
        null_probability = self._null_probability(
            family, key, null, null_total
        )
        actual_probability = (
            actual[family][key] + self.config.prior_strength * null_probability
        ) / (actual_total + self.config.prior_strength)
        return math.log(actual_probability / null_probability), actual[family][key]

    def _family_score(
        self,
        family: str,
        keys: Sequence[Hashable],
    ) -> tuple[float, int]:
        lifetime = [self._family_lift(family, key, recent=False) for key in keys]
        recent = [self._family_lift(family, key, recent=True) for key in keys]
        lifts = [
            0.75 * lifetime_item[0] + 0.25 * recent_item[0]
            for lifetime_item, recent_item in zip(lifetime, recent, strict=True)
        ]
        supports = [item[1] for item in lifetime]
        return (mean(lifts) if lifts else 0.0, min(supports, default=0))

    def _token_probability(
        self,
        token: tuple[int, int],
        *,
        recent: bool,
    ) -> float:
        counts = self.recent_actual_counts if recent else self.actual_counts
        observed_draws = (
            len(self.recent_observations) if recent else self.settled_draws
        )
        total = observed_draws * NUMBERS_PER_DRAW
        if observed_draws == 0:
            total = sum(counts["single"].values())
        return (counts["single"][token] + 1.0 / 36.0) / (total + 1.0)

    def _transition_lift(
        self,
        source: tuple[int, int],
        target: tuple[int, int],
        *,
        recent: bool,
    ) -> tuple[float, int]:
        counts = self.recent_transition_counts if recent else self.transition_counts
        source_totals = (
            self.recent_transition_source_totals
            if recent
            else self.transition_source_totals
        )
        baseline = self._token_probability(target, recent=recent)
        support = source_totals[source]
        observed = counts[(source, target)]
        conditional = (
            observed + self.config.prior_strength * baseline
        ) / (support + self.config.prior_strength)
        return math.log(conditional / baseline), observed

    def _transition_score(
        self,
        target_tokens: Sequence[tuple[int, int]],
    ) -> tuple[float, int]:
        if not self.previous_actual_tokens:
            return 0.0, 0
        keys = tuple(
            (source, target)
            for source in self.previous_actual_tokens
            for target in target_tokens
        )
        lifetime = [
            self._transition_lift(source, target, recent=False)
            for source, target in keys
        ]
        recent = [
            self._transition_lift(source, target, recent=True)
            for source, target in keys
        ]
        lifts = [
            0.75 * lifetime_item[0] + 0.25 * recent_item[0]
            for lifetime_item, recent_item in zip(lifetime, recent, strict=True)
        ]
        return mean(lifts), min((item[1] for item in lifetime), default=0)

    def ticket_evidence(
        self,
        ticket: Iterable[int],
        gaps: Mapping[int, int] | None = None,
    ) -> TicketEvidence:
        """Score a prospective ticket from smoothed factorized motif lift."""
        resolved_gaps = self.gaps() if gaps is None else gaps
        observation = self._observation(ticket, resolved_gaps)
        family_scores = {
            family: self._family_score(
                family,
                tuple(observation.families[family].elements()),
            )
            for family in _FAMILIES
        }
        single = family_scores["single"][0]
        double = (
            0.50 * family_scores["full_double"][0]
            + 0.25 * family_scores["gap_double"][0]
            + 0.25 * family_scores["space_double"][0]
        )
        triple = (
            0.50 * family_scores["gap_triple"][0]
            + 0.50 * family_scores["space_triple"][0]
        )
        transition, transition_support = self._transition_score(
            tuple(vector.token for vector in observation.vectors)
        )
        components = {
            "single": single,
            "double": double,
            "triple": triple,
            "transition": transition,
        }
        enabled = {
            "singles": ("single",),
            "singles+doubles": ("single", "double"),
            "singles+doubles+triples": ("single", "double", "triple"),
            "all-with-transitions": (
                "single",
                "double",
                "triple",
                "transition",
            ),
        }[self.config.motif_variant]
        denominator = sum(_COMPONENT_WEIGHTS[name] for name in enabled)
        total = sum(
            _COMPONENT_WEIGHTS[name] * components[name] for name in enabled
        ) / denominator
        supports = tuple(
            (family, family_scores[family][1]) for family in _FAMILIES
        ) + (("transition", transition_support),)
        return TicketEvidence(single, double, triple, transition, total, supports)

    @staticmethod
    def _swap_ranking(
        ranking: tuple[int, ...], insider: int, outsider: int
    ) -> tuple[int, ...]:
        changed = list(ranking)
        insider_index = changed.index(insider)
        outsider_index = changed.index(outsider)
        changed[insider_index], changed[outsider_index] = (
            changed[outsider_index],
            changed[insider_index],
        )
        return tuple(changed)

    def _candidate(
        self,
        ticket: tuple[int, ...],
        ranking: tuple[int, ...],
        outsider: int | None,
        outsider_rank: int,
        probabilities: Mapping[int, float],
        gaps: Mapping[int, int],
    ) -> _TicketCandidate:
        markov_score = sum(_logit(probabilities[number]) for number in ticket)
        evidence = self.ticket_evidence(ticket, gaps)
        return _TicketCandidate(
            ticket=ticket,
            ranking=ranking,
            outsider=outsider,
            outsider_rank=outsider_rank,
            markov_score=markov_score,
            evidence=evidence,
            total_score=markov_score + self.config.influence * evidence.total,
        )

    def candidates(
        self,
        base_probabilities: Mapping[int, float],
        base_ranking: tuple[int, ...],
        gaps: Mapping[int, int] | None = None,
    ) -> tuple[_TicketCandidate, ...]:
        """Return the champion plus ten rank-six boundary alternatives."""
        resolved_gaps = self.gaps() if gaps is None else gaps
        insider = base_ranking[NUMBERS_PER_DRAW - 1]
        base = self._candidate(
            tuple(sorted(base_ranking[:NUMBERS_PER_DRAW])),
            base_ranking,
            None,
            NUMBERS_PER_DRAW,
            base_probabilities,
            resolved_gaps,
        )
        alternatives = tuple(
            self._candidate(
                tuple(sorted((*base_ranking[: NUMBERS_PER_DRAW - 1], outsider))),
                self._swap_ranking(base_ranking, insider, outsider),
                outsider,
                rank,
                base_probabilities,
                resolved_gaps,
            )
            for rank, outsider in enumerate(base_ranking[6:16], start=7)
        )
        return (base, *alternatives)

    @staticmethod
    def _select_candidate(
        candidates: Sequence[_TicketCandidate],
    ) -> _TicketCandidate:
        base = candidates[0]
        selected = min(
            candidates,
            key=lambda candidate: (
                -candidate.total_score,
                -candidate.markov_score,
                candidate.outsider_rank,
                candidate.ticket,
            ),
        )
        if selected.total_score <= base.total_score + 1e-12:
            return base
        return selected

    def _status(self, proposal_exists: bool) -> str:
        if not self.promotion_enabled:
            return "Benchmark gate failed; exact Markov 100 fallback"
        if self.correction_active:
            return (
                f"Active; lifetime {self.lifetime_shadow_gain:+d}, "
                f"trailing-60 {sum(self.shadow_deltas_60):+d}, "
                f"trailing-120 {sum(self.shadow_deltas_120):+d}"
            )
        if self.shadow_results < SHADOW_ACTIVATION_RESULTS:
            return f"Shadow warm-up {self.shadow_results}/{SHADOW_ACTIVATION_RESULTS} proposals"
        if not proposal_exists:
            return "Inactive; champion ticket has the strongest guarded score"
        return (
            f"Inactive; gains lifetime {self.lifetime_shadow_gain:+d}, "
            f"trailing-60 {sum(self.shadow_deltas_60):+d}, "
            f"trailing-120 {sum(self.shadow_deltas_120):+d}"
        )

    def predict(
        self,
        base_probabilities: dict[int, float],
        base_ranking: tuple[int, ...],
        base_scores: dict[int, float] | None = None,
    ) -> MkgsvDecision:
        """Capture a leakage-safe ticket prediction and deterministic null."""
        expected = set(range(1, NUMBER_COUNT + 1))
        if set(base_ranking) != expected or set(base_probabilities) != expected:
            raise ValueError("MKGSV requires complete 1-49 Markov inputs")
        gaps = self.gaps()
        candidates = self.candidates(base_probabilities, base_ranking, gaps)
        base = candidates[0]
        shadow = self._select_candidate(candidates)
        proposed = shadow.outsider is not None
        if proposed:
            self.proposal_count += 1
        output = shadow if self.correction_active and proposed else base
        resolved_base_scores = (
            dict(base_probabilities) if base_scores is None else dict(base_scores)
        )
        ranking_scores = (
            resolved_base_scores
            if output is base
            else _ranking_scores(output.ranking)
        )
        selected_vectors = {
            vector.number: vector
            for vector in ticket_vectors(shadow.ticket, gaps)
        }
        scores = {
            number: MkgsvScore(
                number=number,
                champion_rank=base_ranking.index(number) + 1,
                base_probability=base_probabilities[number],
                selected_vector=selected_vectors.get(number),
                selected_by_shadow=number in shadow.ticket,
            )
            for number in range(1, NUMBER_COUNT + 1)
        }
        self.pending = _PendingPrediction(
            gaps=dict(gaps),
            gap_classes={number: gap_class(gap) for number, gap in gaps.items()},
            null_families=self._null_observation(gaps),
            previous_tokens=self.previous_actual_tokens,
            base_ticket=base.ticket,
            shadow_ticket=shadow.ticket,
            proposed=proposed,
        )
        insider = base_ranking[NUMBERS_PER_DRAW - 1] if proposed else None
        return MkgsvDecision(
            scores=scores,
            base_ranking=base_ranking,
            shadow_ranking=shadow.ranking,
            output_ranking=output.ranking,
            ranking_scores=ranking_scores,
            base_evidence=base.evidence,
            shadow_evidence=shadow.evidence,
            proposed_insider=insider,
            proposed_outsider=shadow.outsider,
            correction_active=self.correction_active and proposed,
            shadow_results=self.shadow_results,
            lifetime_shadow_gain=self.lifetime_shadow_gain,
            trailing_60_gain=sum(self.shadow_deltas_60),
            trailing_120_gain=sum(self.shadow_deltas_120),
            status=self._status(proposed),
        )

    def state_support_distribution(self) -> dict[str, int | float]:
        """Summarize actual/null motif support for benchmark reporting."""
        result: dict[str, int | float] = {}
        for family in _FAMILIES:
            actual_supports = sorted(self.actual_counts[family].values())
            null_supports = sorted(self.null_counts[family].values())
            result[f"{family}ActualStates"] = len(actual_supports)
            result[f"{family}ActualMedianSupport"] = (
                0.0 if not actual_supports else float(median(actual_supports))
            )
            result[f"{family}NullStates"] = len(null_supports)
            result[f"{family}NullMedianSupport"] = (
                0.0 if not null_supports else float(median(null_supports))
            )
        result["transitionStates"] = len(self.transition_counts)
        result["transitionMedianSupport"] = (
            0.0
            if not self.transition_counts
            else float(median(self.transition_counts.values()))
        )
        return result
