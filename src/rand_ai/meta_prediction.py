"""Aggregate strategy families and predict the next prevailing family."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rand_ai.strategy_prediction import PredictionSuite


META_DRAW_SCHEMA_VERSION = 1
FAMILY_CATALOG_VERSION = 1
RANDOM_EXPECTED_HITS = 36 / 49
LONG_TERM_PRIOR_DRAWS = 12
RECENT_FORM_HALF_LIFE = 12
RECENT_FORM_ALPHA = 1 - 0.5 ** (1 / RECENT_FORM_HALF_LIFE)
TRANSITION_ALPHA = 1.0
ENSEMBLE_WEIGHTS = {
    "long_term_strength": 0.4,
    "recent_form": 0.4,
    "winner_transition": 0.2,
}


class StrategyFamilyId(StrEnum):
    """Identify one stable family of prediction strategies."""

    FREQUENCY_RECENCY = "frequency-recency"
    SHAPE_SIMILARITY = "shape-similarity"
    MARKOV_SEQUENCE = "markov-sequence"
    RELATIONSHIPS_ML = "relationships-machine-learning"
    ENSEMBLES_COVERAGE = "ensembles-coverage"
    RANDOM_BASELINES = "random-baselines"


@dataclass(frozen=True, slots=True)
class StrategyFamily:
    """Describe the strategies assigned to one family."""

    family_id: StrategyFamilyId
    label: str
    strategy_ids: tuple[str, ...]
    predictive: bool


STRATEGY_FAMILIES = (
    StrategyFamily(
        StrategyFamilyId.FREQUENCY_RECENCY,
        "Frequency & Recency",
        ("freshness", "chi_square", "entropy", "bayesian"),
        True,
    ),
    StrategyFamily(
        StrategyFamilyId.SHAPE_SIMILARITY,
        "Shape & Similarity",
        ("proximity", "emd", "predictive_grid"),
        True,
    ),
    StrategyFamily(
        StrategyFamilyId.MARKOV_SEQUENCE,
        "Markov & Sequence",
        (
            "markov100",
            "mkfr",
            "mksp",
            "mknp",
            "mkrd",
            "doublet_triplet_markov",
        ),
        True,
    ),
    StrategyFamily(
        StrategyFamilyId.RELATIONSHIPS_ML,
        "Relationships & Machine Learning",
        (
            "co_occurrence",
            "svc",
            "tbl",
            "sklearn_svm",
            "lag_logistic",
            "sparse_neural_ticket",
        ),
        True,
    ),
    StrategyFamily(
        StrategyFamilyId.ENSEMBLES_COVERAGE,
        "Ensembles & Coverage",
        ("mixed", "cis", "residual_coverage", "chained"),
        True,
    ),
    StrategyFamily(
        StrategyFamilyId.RANDOM_BASELINES,
        "Random Baselines",
        ("randomness", "fresh_random"),
        False,
    ),
)
STRATEGY_FAMILY_BY_ID = {
    strategy_id: family.family_id
    for family in STRATEGY_FAMILIES
    for strategy_id in family.strategy_ids
}
_STRATEGY_CATALOG_ORDER = tuple(
    strategy_id
    for family in STRATEGY_FAMILIES
    for strategy_id in family.strategy_ids
)
_FAMILY_ORDER = {
    family.family_id: index for index, family in enumerate(STRATEGY_FAMILIES)
}


@dataclass(frozen=True, slots=True)
class FamilyDrawOutcome:
    """Store one family's member hits against a completed real draw."""

    family_id: StrategyFamilyId
    member_hits: tuple[tuple[str, int], ...]
    strategy_count: int
    total_hits: int
    mean_hits_per_strategy: float
    normalized_lift: float
    rank: int
    is_prevailing: bool


@dataclass(frozen=True, slots=True)
class FamilyEfficiencySnapshot:
    """Store family evidence available before the target draw occurs."""

    family_id: StrategyFamilyId
    evaluated_draws: int
    evaluations: int
    cumulative_hits: int
    mean_hits_per_strategy: float
    recent_ewma_hits_per_strategy: float
    normalized_lift: float
    win_share: float
    volatility: float
    draws_since_win: int | None


@dataclass(frozen=True, slots=True)
class FamilyProbability:
    """Store one family's score, normalized probability, and rank."""

    family_id: StrategyFamilyId
    rank: int
    raw_score: float
    probability: float


@dataclass(frozen=True, slots=True)
class MetaStrategyForecast:
    """Store one meta strategy's complete family probability ranking."""

    meta_strategy_id: str
    family_probabilities: tuple[FamilyProbability, ...]
    predicted_family_id: StrategyFamilyId


@dataclass(frozen=True, slots=True)
class MetaForecastEvaluation:
    """Measure one settled meta forecast against prevailing families."""

    meta_strategy_id: str
    top_prediction_hit: bool
    winning_probability_mass: float
    reciprocal_winner_rank: float
    brier_score: float


@dataclass(frozen=True, slots=True)
class MetaDraw:
    """Link pre-draw family forecasts to optional post-draw outcomes."""

    reference_draw_number: int
    target_draw_number: int
    reference_date: str | None
    target_date: str | None
    family_snapshots: tuple[FamilyEfficiencySnapshot, ...]
    forecasts: tuple[MetaStrategyForecast, ...]
    actual_numbers: tuple[int, ...]
    family_outcomes: tuple[FamilyDrawOutcome, ...]
    prevailing_family_ids: tuple[StrategyFamilyId, ...]
    forecast_evaluations: tuple[MetaForecastEvaluation, ...]

    @property
    def is_settled(self) -> bool:
        """Return whether the target real draw is available."""
        return bool(self.actual_numbers)


@dataclass(frozen=True, slots=True)
class MetaDrawHistory:
    """Store a configuration-specific chronological MetaDraw history."""

    schema_version: int
    family_catalog_version: int
    strategy_set_fingerprint: str
    enabled_strategy_ids: tuple[str, ...]
    records: tuple[MetaDraw, ...]

    @property
    def latest_forecast(self) -> MetaDraw | None:
        """Return the final pending next-draw forecast, when present."""
        if self.records and not self.records[-1].is_settled:
            return self.records[-1]
        return None


@dataclass(slots=True)
class _FamilyState:
    evaluated_draws: int = 0
    evaluations: int = 0
    cumulative_hits: int = 0
    recent_ewma: float = RANDOM_EXPECTED_HITS
    win_credit: float = 0.0
    mean: float = 0.0
    mean_square_delta: float = 0.0
    last_win_target: int | None = None

    def update(self, outcome: FamilyDrawOutcome, win_credit: float) -> None:
        """Advance running metrics with one settled family outcome."""
        self.evaluated_draws += 1
        self.evaluations += outcome.strategy_count
        self.cumulative_hits += outcome.total_hits
        value = outcome.mean_hits_per_strategy
        delta = value - self.mean
        self.mean += delta / self.evaluated_draws
        self.mean_square_delta += delta * (value - self.mean)
        self.recent_ewma += RECENT_FORM_ALPHA * (value - self.recent_ewma)
        self.win_credit += win_credit

    @property
    def volatility(self) -> float:
        """Return the population standard deviation of draw-level means."""
        if self.evaluated_draws == 0:
            return 0.0
        return math.sqrt(self.mean_square_delta / self.evaluated_draws)


class MetaHistoryBuilder:
    """Build compact MetaDraw records from chronological prediction suites."""

    def __init__(self, enabled_strategy_ids: tuple[str, ...]) -> None:
        unique_ids = tuple(dict.fromkeys(enabled_strategy_ids))
        unknown = set(unique_ids).difference(STRATEGY_FAMILY_BY_ID)
        if unknown:
            raise ValueError(
                f"Unknown family strategy plugin(s): {', '.join(sorted(unknown))}"
            )
        self.enabled_strategy_ids = unique_ids
        enabled = set(unique_ids)
        self.family_members = {
            family.family_id: tuple(
                strategy_id
                for strategy_id in family.strategy_ids
                if strategy_id in enabled
            )
            for family in STRATEGY_FAMILIES
            if enabled.intersection(family.strategy_ids)
        }
        self.candidate_family_ids = tuple(
            family.family_id
            for family in STRATEGY_FAMILIES
            if family.predictive and family.family_id in self.family_members
        )
        self.states = {
            family_id: _FamilyState() for family_id in self.family_members
        }
        self.transition_counts = {
            source: {destination: 0.0 for destination in self.candidate_family_ids}
            for source in self.candidate_family_ids
        }
        self.previous_prevailing: tuple[StrategyFamilyId, ...] = ()
        self.settled_draws = 0
        self.records: list[MetaDraw] = []

    def _snapshots(
        self, reference_draw_number: int
    ) -> tuple[FamilyEfficiencySnapshot, ...]:
        snapshots = []
        for family in STRATEGY_FAMILIES:
            if family.family_id not in self.family_members:
                continue
            state = self.states[family.family_id]
            mean_hits = (
                state.cumulative_hits / state.evaluations
                if state.evaluations
                else 0.0
            )
            snapshots.append(
                FamilyEfficiencySnapshot(
                    family_id=family.family_id,
                    evaluated_draws=state.evaluated_draws,
                    evaluations=state.evaluations,
                    cumulative_hits=state.cumulative_hits,
                    mean_hits_per_strategy=mean_hits,
                    recent_ewma_hits_per_strategy=state.recent_ewma,
                    normalized_lift=(
                        mean_hits - RANDOM_EXPECTED_HITS
                        if state.evaluations
                        else 0.0
                    ),
                    win_share=(
                        state.win_credit / self.settled_draws
                        if self.settled_draws
                        else 0.0
                    ),
                    volatility=state.volatility,
                    draws_since_win=(
                        None
                        if state.last_win_target is None
                        else reference_draw_number - state.last_win_target
                    ),
                )
            )
        return tuple(snapshots)

    def _ranked_forecast(
        self,
        meta_strategy_id: str,
        raw_scores: dict[StrategyFamilyId, float],
    ) -> MetaStrategyForecast:
        total = sum(raw_scores.values())
        probabilities = (
            {
                family_id: score / total
                for family_id, score in raw_scores.items()
            }
            if total > 0
            else {
                family_id: 1 / len(raw_scores) for family_id in raw_scores
            }
        )
        ordered = sorted(
            raw_scores,
            key=lambda family_id: (
                -probabilities[family_id],
                _FAMILY_ORDER[family_id],
            ),
        )
        family_probabilities = tuple(
            FamilyProbability(
                family_id=family_id,
                rank=rank,
                raw_score=raw_scores[family_id],
                probability=probabilities[family_id],
            )
            for rank, family_id in enumerate(ordered, start=1)
        )
        return MetaStrategyForecast(
            meta_strategy_id=meta_strategy_id,
            family_probabilities=family_probabilities,
            predicted_family_id=family_probabilities[0].family_id,
        )

    def _transition_scores(self) -> dict[StrategyFamilyId, float]:
        family_count = len(self.candidate_family_ids)
        if not self.previous_prevailing:
            return {
                family_id: 1 / family_count
                for family_id in self.candidate_family_ids
            }
        scores = {family_id: 0.0 for family_id in self.candidate_family_ids}
        for source in self.previous_prevailing:
            row = self.transition_counts[source]
            denominator = sum(row.values()) + TRANSITION_ALPHA * family_count
            for destination in self.candidate_family_ids:
                scores[destination] += (
                    row[destination] + TRANSITION_ALPHA
                ) / denominator / len(self.previous_prevailing)
        return scores

    def _forecasts(self) -> tuple[MetaStrategyForecast, ...]:
        if not self.candidate_family_ids:
            return ()
        long_term_scores = {}
        recent_scores = {}
        for family_id in self.candidate_family_ids:
            state = self.states[family_id]
            strategy_count = len(self.family_members[family_id])
            prior_evaluations = LONG_TERM_PRIOR_DRAWS * strategy_count
            long_term_scores[family_id] = (
                state.cumulative_hits
                + prior_evaluations * RANDOM_EXPECTED_HITS
            ) / (state.evaluations + prior_evaluations)
            recent_scores[family_id] = state.recent_ewma
        long_term = self._ranked_forecast(
            "long_term_strength", long_term_scores
        )
        recent = self._ranked_forecast("recent_form", recent_scores)
        transition = self._ranked_forecast(
            "winner_transition", self._transition_scores()
        )
        component_probabilities = {
            forecast.meta_strategy_id: {
                item.family_id: item.probability
                for item in forecast.family_probabilities
            }
            for forecast in (long_term, recent, transition)
        }
        ensemble_scores = {
            family_id: sum(
                ENSEMBLE_WEIGHTS[strategy_id]
                * component_probabilities[strategy_id][family_id]
                for strategy_id in ENSEMBLE_WEIGHTS
            )
            for family_id in self.candidate_family_ids
        }
        ensemble = self._ranked_forecast("family_ensemble", ensemble_scores)
        return long_term, recent, transition, ensemble

    def _outcomes(
        self, suite: PredictionSuite
    ) -> tuple[
        tuple[FamilyDrawOutcome, ...], tuple[StrategyFamilyId, ...]
    ]:
        actual = set(suite.actual_numbers)
        strategies = {
            strategy.strategy_id: strategy for strategy in suite.strategies
        }
        missing = set(self.enabled_strategy_ids).difference(strategies)
        if missing:
            raise ValueError(
                f"Prediction suite is missing strategy plugin(s): "
                f"{', '.join(sorted(missing))}"
            )
        family_values = {}
        family_hits = {}
        for family_id, member_ids in self.family_members.items():
            member_hits = tuple(
                (
                    strategy_id,
                    len(actual.intersection(strategies[strategy_id].top_numbers)),
                )
                for strategy_id in member_ids
            )
            total_hits = sum(hits for _strategy_id, hits in member_hits)
            family_hits[family_id] = member_hits
            family_values[family_id] = total_hits / len(member_ids)
        predictive_values = {
            family_id: family_values[family_id]
            for family_id in self.candidate_family_ids
        }
        maximum = max(predictive_values.values(), default=None)
        prevailing = tuple(
            family_id
            for family_id in self.candidate_family_ids
            if predictive_values[family_id] == maximum
        )
        unique_values = sorted(set(predictive_values.values()), reverse=True)
        ranks = {value: rank for rank, value in enumerate(unique_values, start=1)}
        outcomes = tuple(
            FamilyDrawOutcome(
                family_id=family.family_id,
                member_hits=family_hits[family.family_id],
                strategy_count=len(self.family_members[family.family_id]),
                total_hits=sum(
                    hits for _strategy_id, hits in family_hits[family.family_id]
                ),
                mean_hits_per_strategy=family_values[family.family_id],
                normalized_lift=(
                    family_values[family.family_id] - RANDOM_EXPECTED_HITS
                ),
                rank=(
                    ranks[family_values[family.family_id]]
                    if family.predictive
                    else 0
                ),
                is_prevailing=family.family_id in prevailing,
            )
            for family in STRATEGY_FAMILIES
            if family.family_id in self.family_members
        )
        return outcomes, prevailing

    @staticmethod
    def _evaluations(
        forecasts: tuple[MetaStrategyForecast, ...],
        prevailing: tuple[StrategyFamilyId, ...],
    ) -> tuple[MetaForecastEvaluation, ...]:
        if not prevailing:
            return ()
        winner_target = 1 / len(prevailing)
        winner_set = set(prevailing)
        return tuple(
            MetaForecastEvaluation(
                meta_strategy_id=forecast.meta_strategy_id,
                top_prediction_hit=forecast.predicted_family_id in winner_set,
                winning_probability_mass=sum(
                    item.probability
                    for item in forecast.family_probabilities
                    if item.family_id in winner_set
                ),
                reciprocal_winner_rank=(
                    1
                    / min(
                        item.rank
                        for item in forecast.family_probabilities
                        if item.family_id in winner_set
                    )
                ),
                brier_score=sum(
                    (
                        item.probability
                        - (winner_target if item.family_id in winner_set else 0.0)
                    )
                    ** 2
                    for item in forecast.family_probabilities
                ),
            )
            for forecast in forecasts
        )

    def _update_history(
        self,
        outcomes: tuple[FamilyDrawOutcome, ...],
        prevailing: tuple[StrategyFamilyId, ...],
        target_draw_number: int,
    ) -> None:
        winner_credit = 1 / len(prevailing) if prevailing else 0.0
        for outcome in outcomes:
            state = self.states[outcome.family_id]
            state.update(
                outcome,
                winner_credit if outcome.family_id in prevailing else 0.0,
            )
            if outcome.family_id in prevailing:
                state.last_win_target = target_draw_number
        if self.previous_prevailing and prevailing:
            transition_credit = 1 / (
                len(self.previous_prevailing) * len(prevailing)
            )
            for source in self.previous_prevailing:
                for destination in prevailing:
                    self.transition_counts[source][destination] += transition_credit
        self.previous_prevailing = prevailing
        self.settled_draws += 1

    def record_suite(
        self,
        suite: PredictionSuite,
        *,
        reference_date: str | None = None,
        target_date: str | None = None,
    ) -> MetaDraw:
        """Forecast, optionally settle, and retain one chronological suite."""
        if self.records and not self.records[-1].is_settled:
            raise ValueError("A pending MetaDraw must be the final history record")
        snapshots = self._snapshots(suite.reference_draw_number)
        forecasts = self._forecasts()
        if suite.actual_numbers:
            outcomes, prevailing = self._outcomes(suite)
            evaluations = self._evaluations(forecasts, prevailing)
        else:
            outcomes, prevailing, evaluations = (), (), ()
        record = MetaDraw(
            reference_draw_number=suite.reference_draw_number,
            target_draw_number=suite.target_draw_number,
            reference_date=reference_date,
            target_date=target_date,
            family_snapshots=snapshots,
            forecasts=forecasts,
            actual_numbers=suite.actual_numbers,
            family_outcomes=outcomes,
            prevailing_family_ids=prevailing,
            forecast_evaluations=evaluations,
        )
        self.records.append(record)
        if record.is_settled:
            self._update_history(
                outcomes,
                prevailing,
                suite.target_draw_number,
            )
        return record

    def build(self) -> MetaDrawHistory:
        """Return the immutable history accumulated so far."""
        enabled = set(self.enabled_strategy_ids)
        encoded = json.dumps(
            tuple(
                strategy_id
                for strategy_id in _STRATEGY_CATALOG_ORDER
                if strategy_id in enabled
            ),
            separators=(",", ":"),
        ).encode("utf-8")
        return MetaDrawHistory(
            schema_version=META_DRAW_SCHEMA_VERSION,
            family_catalog_version=FAMILY_CATALOG_VERSION,
            strategy_set_fingerprint=hashlib.sha256(encoded).hexdigest(),
            enabled_strategy_ids=self.enabled_strategy_ids,
            records=tuple(self.records),
        )
