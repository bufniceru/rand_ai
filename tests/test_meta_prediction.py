"""Test family aggregation and leakage-safe MetaDraw forecasting."""

import hashlib
import json
import math

import pytest

from rand_ai import (
    MetaDrawHistory,
    PredictionSuite,
    StrategyFamilyId,
    StrategyPrediction,
)
from rand_ai.meta_prediction import (
    ENSEMBLE_WEIGHTS,
    LONG_TERM_PRIOR_DRAWS,
    RANDOM_EXPECTED_HITS,
    RECENT_FORM_ALPHA,
    STRATEGY_FAMILIES,
    STRATEGY_FAMILY_BY_ID,
    MetaHistoryBuilder,
)
from rand_ai.strategy_prediction import STRATEGY_IDS


def _strategy(strategy_id: str, *top_numbers: int) -> StrategyPrediction:
    return StrategyPrediction(
        strategy_id=strategy_id,
        name=strategy_id,
        description="test strategy",
        numbers=(),
        top_numbers=tuple(top_numbers),
    )


def _suite(
    reference: int,
    actual_numbers: tuple[int, ...],
    *strategies: StrategyPrediction,
) -> PredictionSuite:
    return PredictionSuite(
        reference_draw_number=reference,
        target_draw_number=reference + 1,
        actual_numbers=actual_numbers,
        strategies=strategies,
    )


def _by_family(items: tuple[object, ...]) -> dict[StrategyFamilyId, object]:
    return {getattr(item, "family_id"): item for item in items}


def _forecast_probabilities(record: object, strategy_id: str) -> dict:
    forecast = next(
        forecast
        for forecast in getattr(record, "forecasts")
        if forecast.meta_strategy_id == strategy_id
    )
    return {
        item.family_id: item.probability
        for item in forecast.family_probabilities
    }


def test_family_catalog_covers_every_strategy_once() -> None:
    catalog_ids = [
        strategy_id
        for family in STRATEGY_FAMILIES
        for strategy_id in family.strategy_ids
    ]

    assert set(catalog_ids) == set(STRATEGY_IDS)
    assert len(catalog_ids) == len(set(catalog_ids))
    assert set(STRATEGY_FAMILY_BY_ID) == set(STRATEGY_IDS)
    assert STRATEGY_FAMILIES[-1].family_id is StrategyFamilyId.RANDOM_BASELINES
    assert not STRATEGY_FAMILIES[-1].predictive
    assert all(family.predictive for family in STRATEGY_FAMILIES[:-1])


def test_cold_start_aggregation_co_winners_and_pending_snapshot() -> None:
    enabled = (
        "freshness",
        "proximity",
        "emd",
        "markov100",
        "co_occurrence",
        "mixed",
        "randomness",
    )
    builder = MetaHistoryBuilder(enabled)
    settled = builder.record_suite(
        _suite(
            1,
            (1, 2, 3, 4, 5, 6),
            _strategy("freshness", 1, 2),
            _strategy("proximity", 1, 2),
            _strategy("emd", 3, 4),
            _strategy("markov100", 1),
            _strategy("co_occurrence", 1),
            _strategy("mixed", 1),
            _strategy("randomness", 1, 2, 3, 4, 5, 6),
        ),
        reference_date="2026-01-01",
        target_date="2026-01-02",
    )

    assert settled.is_settled
    assert settled.reference_date == "2026-01-01"
    assert settled.target_date == "2026-01-02"
    assert settled.prevailing_family_ids == (
        StrategyFamilyId.FREQUENCY_RECENCY,
        StrategyFamilyId.SHAPE_SIMILARITY,
    )
    outcomes = _by_family(settled.family_outcomes)
    frequency = outcomes[StrategyFamilyId.FREQUENCY_RECENCY]
    shape = outcomes[StrategyFamilyId.SHAPE_SIMILARITY]
    random = outcomes[StrategyFamilyId.RANDOM_BASELINES]
    assert getattr(frequency, "total_hits") == 2
    assert getattr(frequency, "mean_hits_per_strategy") == 2
    assert getattr(frequency, "rank") == 1
    assert getattr(frequency, "is_prevailing")
    assert getattr(shape, "total_hits") == 4
    assert getattr(shape, "strategy_count") == 2
    assert getattr(shape, "mean_hits_per_strategy") == 2
    assert getattr(shape, "rank") == 1
    assert getattr(random, "mean_hits_per_strategy") == 6
    assert getattr(random, "rank") == 0
    assert not getattr(random, "is_prevailing")
    assert all(
        all(
            math.isfinite(item.probability) and item.probability >= 0
            for item in forecast.family_probabilities
        )
        and sum(item.probability for item in forecast.family_probabilities)
        == pytest.approx(1)
        for forecast in settled.forecasts
    )
    assert all(
        item.probability == pytest.approx(0.2)
        for forecast in settled.forecasts
        for item in forecast.family_probabilities
    )
    assert all(
        evaluation.top_prediction_hit
        and evaluation.winning_probability_mass == pytest.approx(0.4)
        and evaluation.reciprocal_winner_rank == 1
        and evaluation.brier_score == pytest.approx(0.3)
        for evaluation in settled.forecast_evaluations
    )

    pending = builder.record_suite(
        _suite(
            2,
            (),
            _strategy("freshness", 7),
            _strategy("proximity", 7),
            _strategy("emd", 8),
            _strategy("markov100", 9),
            _strategy("co_occurrence", 10),
            _strategy("mixed", 11),
            _strategy("randomness", 12),
        )
    )
    snapshots = _by_family(pending.family_snapshots)
    frequency_snapshot = snapshots[StrategyFamilyId.FREQUENCY_RECENCY]
    shape_snapshot = snapshots[StrategyFamilyId.SHAPE_SIMILARITY]
    assert not pending.is_settled
    assert pending.family_outcomes == ()
    assert pending.forecast_evaluations == ()
    assert getattr(frequency_snapshot, "evaluations") == 1
    assert getattr(frequency_snapshot, "mean_hits_per_strategy") == 2
    assert getattr(frequency_snapshot, "win_share") == 0.5
    assert getattr(frequency_snapshot, "draws_since_win") == 0
    assert getattr(shape_snapshot, "evaluations") == 2
    assert getattr(shape_snapshot, "mean_hits_per_strategy") == 2
    assert getattr(shape_snapshot, "win_share") == 0.5

    history = builder.build()
    encoded = json.dumps(enabled, separators=(",", ":")).encode()
    assert history.strategy_set_fingerprint == hashlib.sha256(encoded).hexdigest()
    assert history.strategy_set_fingerprint == MetaHistoryBuilder(
        tuple(reversed(enabled))
    ).build().strategy_set_fingerprint
    assert history.strategy_set_fingerprint != MetaHistoryBuilder(
        enabled[:-1]
    ).build().strategy_set_fingerprint
    assert history.latest_forecast is pending
    with pytest.raises(ValueError, match="pending MetaDraw"):
        builder.record_suite(_suite(3, (), *pending_suite_strategies(enabled)))


def pending_suite_strategies(enabled: tuple[str, ...]) -> tuple[StrategyPrediction, ...]:
    return tuple(_strategy(strategy_id, 1) for strategy_id in enabled)


def test_history_models_use_only_prior_results_and_fractional_transitions() -> None:
    enabled = ("freshness", "proximity")
    builder = MetaHistoryBuilder(enabled)
    first = builder.record_suite(
        _suite(
            1,
            (1, 2, 3, 4, 5, 6),
            _strategy("freshness", 1, 2),
            _strategy("proximity", 1, 2),
        )
    )
    second = builder.record_suite(
        _suite(
            2,
            (7, 8, 9, 10, 11, 12),
            _strategy("freshness", 7, 8),
            _strategy("proximity", 30),
        )
    )
    third = builder.record_suite(
        _suite(
            3,
            (),
            _strategy("freshness", 20),
            _strategy("proximity", 21),
        )
    )

    assert first.family_snapshots[0].evaluated_draws == 0
    first_second_snapshot = _by_family(second.family_snapshots)[
        StrategyFamilyId.FREQUENCY_RECENCY
    ]
    assert getattr(first_second_snapshot, "evaluated_draws") == 1
    assert getattr(first_second_snapshot, "cumulative_hits") == 2
    third_snapshots = _by_family(third.family_snapshots)
    frequency = third_snapshots[StrategyFamilyId.FREQUENCY_RECENCY]
    assert getattr(frequency, "evaluated_draws") == 2
    assert getattr(frequency, "cumulative_hits") == 4
    assert getattr(frequency, "mean_hits_per_strategy") == 2
    assert getattr(frequency, "volatility") == 0
    assert getattr(frequency, "win_share") == pytest.approx(0.75)
    assert getattr(frequency, "draws_since_win") == 0
    expected_ewma = RANDOM_EXPECTED_HITS
    expected_ewma += RECENT_FORM_ALPHA * (2 - expected_ewma)
    expected_ewma += RECENT_FORM_ALPHA * (2 - expected_ewma)
    assert getattr(frequency, "recent_ewma_hits_per_strategy") == pytest.approx(
        expected_ewma
    )

    transition = _forecast_probabilities(third, "winner_transition")
    assert transition[StrategyFamilyId.FREQUENCY_RECENCY] == pytest.approx(0.6)
    assert transition[StrategyFamilyId.SHAPE_SIMILARITY] == pytest.approx(0.4)
    long_term_forecast = next(
        forecast
        for forecast in third.forecasts
        if forecast.meta_strategy_id == "long_term_strength"
    )
    frequency_raw = next(
        item.raw_score
        for item in long_term_forecast.family_probabilities
        if item.family_id is StrategyFamilyId.FREQUENCY_RECENCY
    )
    assert frequency_raw == pytest.approx(
        (4 + LONG_TERM_PRIOR_DRAWS * RANDOM_EXPECTED_HITS)
        / (2 + LONG_TERM_PRIOR_DRAWS)
    )
    components = {
        strategy_id: _forecast_probabilities(third, strategy_id)
        for strategy_id in ENSEMBLE_WEIGHTS
    }
    ensemble = _forecast_probabilities(third, "family_ensemble")
    for family_id in ensemble:
        assert ensemble[family_id] == pytest.approx(
            sum(
                ENSEMBLE_WEIGHTS[strategy_id]
                * components[strategy_id][family_id]
                for strategy_id in ENSEMBLE_WEIGHTS
            )
        )


def test_volatility_loss_metrics_and_no_winner_paths() -> None:
    builder = MetaHistoryBuilder(("freshness", "proximity"))
    builder.record_suite(
        _suite(
            1,
            (1, 2, 3, 4, 5, 6),
            _strategy("freshness", 1, 2),
            _strategy("proximity", 20),
        )
    )
    losing = builder.record_suite(
        _suite(
            2,
            (7, 8, 9, 10, 11, 12),
            _strategy("freshness", 20),
            _strategy("proximity", 7, 8),
        )
    )
    pending = builder.record_suite(
        _suite(
            3,
            (),
            _strategy("freshness", 1),
            _strategy("proximity", 2),
        )
    )

    assert all(
        not evaluation.top_prediction_hit
        for evaluation in losing.forecast_evaluations
    )
    snapshots = _by_family(pending.family_snapshots)
    assert getattr(
        snapshots[StrategyFamilyId.FREQUENCY_RECENCY], "volatility"
    ) == pytest.approx(1)
    assert getattr(
        snapshots[StrategyFamilyId.FREQUENCY_RECENCY], "draws_since_win"
    ) == 1

    random_builder = MetaHistoryBuilder(("randomness",))
    random_settled = random_builder.record_suite(
        _suite(
            1,
            (1, 2, 3, 4, 5, 6),
            _strategy("randomness", 1),
        )
    )
    assert random_settled.forecasts == ()
    assert random_settled.prevailing_family_ids == ()
    assert random_settled.forecast_evaluations == ()
    assert random_settled.family_outcomes[0].rank == 0
    assert random_builder.build().latest_forecast is None

    empty_history = MetaDrawHistory(1, 1, "fingerprint", (), ())
    assert empty_history.latest_forecast is None


def test_builder_validation_and_zero_score_fallback() -> None:
    with pytest.raises(ValueError, match="Unknown family strategy"):
        MetaHistoryBuilder(("unknown",))

    builder = MetaHistoryBuilder(("freshness", "proximity"))
    with pytest.raises(ValueError, match="missing strategy"):
        builder.record_suite(
            _suite(
                1,
                (1, 2, 3, 4, 5, 6),
                _strategy("freshness", 1),
            )
        )

    forecast = builder._ranked_forecast(
        "zero",
        {
            StrategyFamilyId.FREQUENCY_RECENCY: 0.0,
            StrategyFamilyId.SHAPE_SIMILARITY: 0.0,
        },
    )
    assert forecast.predicted_family_id is StrategyFamilyId.FREQUENCY_RECENCY
    assert [item.probability for item in forecast.family_probabilities] == [0.5, 0.5]
