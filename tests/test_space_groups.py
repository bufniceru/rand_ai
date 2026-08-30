"""Test circular border groups, null statistics, and online forecasts."""

from __future__ import annotations

from math import comb
from typing import cast

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import rand_ai.space_groups as groups
from rand_ai.draw import Draw
from rand_ai.draws import Draws
from rand_ai.space_groups import (
    MODEL_IDS,
    SIGNATURES,
    SpaceGroupForecaster,
    exact_null_group_probabilities,
    exact_null_probabilities,
    exact_null_signature_counts,
    profile_for_numbers,
    profile_from_spaces,
    signature_chi_square,
    spaces_for_numbers,
    transition_diagnostics,
    validate_border_space,
    validate_target_group_count,
    walk_forward_models,
)
from rand_ai.statistics import DrawsStatistics


SPACE_PATTERNS = (
    (18, 5, 5, 5, 5, 5),
    (10, 10, 6, 6, 6, 5),
    (10, 6, 10, 6, 6, 5),
    (10, 6, 6, 10, 6, 5),
)


def profiles(count: int = 8) -> list[groups.SpaceGroupProfile]:
    return [profile_from_spaces(SPACE_PATTERNS[index % 4], 7) for index in range(count)]


def test_validates_border_and_six_number_spaces() -> None:
    assert validate_border_space(0) == 0
    assert validate_border_space(43) == 43
    assert validate_target_group_count(None) is None
    assert validate_target_group_count(3) == 3
    assert spaces_for_numbers({1, 2, 8, 17, 31, 49}) == (0, 0, 5, 8, 13, 17)
    assert profile_for_numbers({1, 2, 8, 17, 31, 49}, 7).spaces == (
        0,
        0,
        5,
        8,
        13,
        17,
    )
    for value in (-1, 44, True, 7.0):
        with pytest.raises(ValueError, match="border_space"):
            validate_border_space(cast(int, value))
    for value in (0, 7, True, 3.0):
        with pytest.raises(ValueError, match="target_group_count"):
            validate_target_group_count(cast(int, value))
    for values in ({1, 2}, {0, 1, 2, 3, 4, 5}, {1, 2, 3, 4, 5, 50}):
        with pytest.raises(ValueError, match="six unique"):
            spaces_for_numbers(values)


def test_profiles_groups_circularly_and_uses_inclusive_border() -> None:
    one = profile_from_spaces((8, 7, 7, 7, 7, 7), 7, numbers=(1, 9, 17, 25, 33, 41))
    assert one.separator_count == 1
    assert one.group_count == 1
    assert one.ordered_group_sizes == (6,)
    assert one.signature == (6,)
    assert one.signature_text == "6"
    assert one.large_spaces == (8,)
    assert one.separator_indices == (0,)
    assert one.ordered_groups == ((1, 9, 17, 25, 33, 41),)

    three = profile_from_spaces((10, 6, 6, 10, 6, 5), 7)
    assert three.separator_count == 2
    assert three.group_count == 2
    assert three.ordered_group_sizes == (3, 3)
    assert three.signature == (3, 3)
    assert three.maximum_space == 10

    no_separator = profile_from_spaces((8, 7, 7, 7, 7, 7), 43)
    assert no_separator.group_count == 1
    assert no_separator.separator_count == 0
    assert no_separator.ordered_group_sizes == (6,)
    no_separator_with_numbers = profile_from_spaces(
        (8, 7, 7, 7, 7, 7),
        43,
        numbers=(1, 9, 17, 25, 33, 41),
    )
    assert no_separator_with_numbers.ordered_groups == (
        (1, 9, 17, 25, 33, 41),
    )

    all_separate = profile_from_spaces((8, 7, 7, 7, 7, 7), 0)
    assert all_separate.group_count == 6
    assert all_separate.signature == (1, 1, 1, 1, 1, 1)


def test_profile_anchor_ties_use_lowest_following_number() -> None:
    profile = profile_from_spaces(
        (10, 6, 6, 10, 6, 5),
        7,
        numbers=(20, 25, 30, 1, 7, 13),
    )
    assert profile.ordered_group_sizes == (3, 3)
    assert profile.ordered_separator_indices == (3, 0)
    assert profile.ordered_groups == ((1, 7, 13), (20, 25, 30))
    rotated = profile_from_spaces(
        (10, 6, 5, 10, 6, 6),
        7,
        numbers=(1, 7, 13, 20, 25, 30),
    )
    assert rotated.signature == profile.signature
    assert rotated.ordered_groups == profile.ordered_groups
    with pytest.raises(ValueError, match="six non-negative"):
        profile_from_spaces((1, 2, 3), 7)
    with pytest.raises(ValueError, match="six non-negative"):
        profile_from_spaces((8, 7, 7, 7, 7, -1), 7)
    with pytest.raises(ValueError, match="numbers must contain"):
        profile_from_spaces((8, 7, 7, 7, 7, 7), 7, numbers=(1, 2))


def test_exact_random_null_matches_rooted_gap_compositions() -> None:
    counts = exact_null_signature_counts(7)
    probabilities = exact_null_probabilities(7)
    group_probabilities = exact_null_group_probabilities(7)
    assert sum(counts) == comb(48, 5)
    assert sum(probabilities) == pytest.approx(1)
    assert sum(group_probabilities) == pytest.approx(1)
    assert sum(group_probabilities[:3]) == pytest.approx(0.9625206739)
    assert group_probabilities[:4] == pytest.approx(
        (0.1148207328, 0.5023407059, 0.3453592353, 0.0372830993)
    )
    assert exact_null_signature_counts(7) is counts
    assert SpaceGroupForecaster(43).forecast()["border_group_statistical"] == (
        1.0,
        *(0.0 for _ in SIGNATURES[1:]),
    )


def test_online_forecaster_normalizes_models_and_decodes_valid_marginals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(groups, "ML_WARMUP", 2)
    forecaster = SpaceGroupForecaster(7)
    empty = forecaster.forecast()
    assert set(empty) == set(MODEL_IDS)
    assert all(sum(probabilities) == pytest.approx(1) for probabilities in empty.values())

    evaluated = []
    for profile in profiles(10):
        result = forecaster.observe(profile)
        if result is not None:
            evaluated.append(result)
        forecast = forecaster.forecast()
        assert all(sum(probabilities) == pytest.approx(1) for probabilities in forecast.values())
    assert evaluated[0][0] == 1
    assert forecaster.ml_training_count == 9

    probabilities = forecaster.forecast()["border_group_hybrid"]
    scores, details = forecaster.number_scores(probabilities)
    assert set(scores) == set(range(1, 50))
    assert all(0 <= score <= 1 for score in scores.values())
    assert "Border space 7" in details[1]
    assert forecaster.number_scores(probabilities) == (scores, details)
    decoded = forecaster.decoded_tickets(probabilities)
    signature_mass: dict[str, float] = {}
    for ticket, signature, weight in decoded:
        assert len(ticket) == len(set(ticket)) == 6
        assert tuple(sorted(ticket)) == ticket
        assert all(1 <= number <= 49 for number in ticket)
        assert sum(spaces_for_numbers(ticket)) == 43
        assert profile_for_numbers(ticket, 7).signature_text == signature
        signature_mass[signature] = signature_mass.get(signature, 0.0) + weight
    for signature, probability in zip(SIGNATURES, probabilities, strict=True):
        signature_text = "+".join(map(str, signature))
        assert signature_mass.get(signature_text, 0.0) == pytest.approx(probability)
    with pytest.raises(ValueError, match="all 11"):
        forecaster.decoded_tickets((1.0,))


def test_svc_configuration_and_bounded_retraining_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = SpaceGroupForecaster(7)
    assert groups.SVC_WARMUP == 50
    assert groups.SVC_REFIT_INTERVAL == 25
    assert configured.svc_training_features.maxlen == 500
    assert configured.svc_training_targets.maxlen == 500
    pipeline = configured._new_svc_model()
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    svc = pipeline.named_steps["svc"]
    assert isinstance(svc, SVC)
    assert svc.kernel == "rbf"
    assert svc.C == 1.0
    assert svc.gamma == "scale"
    assert svc.class_weight == "balanced"
    assert svc.probability is True
    assert svc.random_state == 0

    monkeypatch.setattr(groups, "SVC_WARMUP", 2)
    monkeypatch.setattr(groups, "SVC_REFIT_INTERVAL", 2)
    monkeypatch.setattr(groups, "SVC_TRAINING_WINDOW", 3)
    forecaster = SpaceGroupForecaster(7)
    one_group = profile_from_spaces(SPACE_PATTERNS[0], 7)
    two_groups = profile_from_spaces(SPACE_PATTERNS[3], 7)

    def advance(profile: groups.SpaceGroupProfile) -> dict[str, tuple[float, ...]]:
        forecaster.observe(profile)
        return forecaster.forecast()

    advance(one_group)
    pending = forecaster.pending_features
    assert forecaster.svc_training_count == 0
    advance(one_group)
    assert forecaster.svc_training_features[-1] == pending
    assert forecaster.svc_training_targets[-1] == groups.SIGNATURE_INDEX[(6,)]
    assert forecaster.svc is None
    single_class = advance(one_group)
    assert forecaster.svc_last_fit_count == 2
    assert forecaster.svc is None
    assert single_class["border_group_svc"] == single_class["border_group_statistical"]
    advance(two_groups)
    assert forecaster.svc_training_count == 3
    assert forecaster.svc_last_fit_count == 2
    fitted_forecast = advance(two_groups)
    first_model = forecaster.svc
    assert first_model is not None
    assert forecaster.svc_last_fit_count == 4
    assert len(forecaster.svc_training_features) == 3
    assert len(forecaster.svc_training_targets) == 3
    assert sum(fitted_forecast["border_group_svc"]) == pytest.approx(1)
    single_window = advance(two_groups)
    assert forecaster.svc is first_model
    assert single_window["border_group_svc"] == single_window[
        "border_group_statistical"
    ]
    final_forecast = advance(one_group)
    assert forecaster.svc is not first_model
    assert forecaster.svc_last_fit_count == 6
    assert sum(final_forecast["border_group_svc"]) == pytest.approx(1)

    replica = SpaceGroupForecaster(7)
    replica_forecast: dict[str, tuple[float, ...]] = {}
    for profile in (
        one_group,
        one_group,
        one_group,
        two_groups,
        two_groups,
        two_groups,
        one_group,
    ):
        replica.observe(profile)
        replica_forecast = replica.forecast()
    assert replica_forecast["border_group_svc"] == pytest.approx(
        final_forecast["border_group_svc"]
    )


def test_svc_maps_classes_to_signatures_and_conditions_target() -> None:
    class FakeSVCStep:
        classes_ = np.arange(len(SIGNATURES))

    class FakePipeline:
        named_steps = {"svc": FakeSVCStep()}

        @staticmethod
        def predict_proba(_features: np.ndarray) -> np.ndarray:
            return np.asarray([[float(index + 1) for index in range(len(SIGNATURES))]])

    forecaster = SpaceGroupForecaster(7)
    forecaster.observe(profile_from_spaces(SPACE_PATTERNS[0], 7))
    forecaster.forecast()
    forecaster.svc = cast(groups.Pipeline, FakePipeline())
    probabilities = forecaster.forecast()["border_group_svc"]
    assert len(probabilities) == len(SIGNATURES)
    assert sum(probabilities) == pytest.approx(1)
    assert all(
        probability == 0
        for probability, feasible in zip(
            probabilities, forecaster.feasible_signatures, strict=True
        )
        if not feasible
    )

    forecaster.target_group_count = 3
    conditioned = forecaster.forecast()["border_group_svc"]
    assert sum(conditioned) == pytest.approx(1)
    assert all(
        probability == 0 or len(signature) == 3
        for signature, probability in zip(SIGNATURES, conditioned, strict=True)
    )


def test_hybrid_uses_trailing_losses_and_number_fallback() -> None:
    forecaster = SpaceGroupForecaster(7)
    assert set(forecaster.hybrid_weights().values()) == {0.2}
    assert tuple(forecaster.hybrid_weights()) == groups.COMPONENT_MODEL_IDS
    impossible = (0.0,) * (len(SIGNATURES) - 1) + (1.0,)
    uniform_scores, _details = forecaster.number_scores(impossible)
    assert set(uniform_scores.values()) == {0.0}
    for index, values in enumerate(forecaster.losses.values()):
        values.extend([float(index + 1)] * 30)
    weights = forecaster.hybrid_weights()
    assert sum(weights.values()) == pytest.approx(1)
    assert min(weights.values()) >= 0.05
    assert weights["border_group_statistical"] > weights["border_group_ml"]


def test_manual_target_conditions_every_forecast_and_rejects_impossible_count() -> None:
    forecaster = SpaceGroupForecaster(5, target_group_count=3)
    for profile in profiles(6):
        forecaster.observe(profile_from_spaces(profile.spaces, 5))
        forecasts = forecaster.forecast()
    for probabilities in forecasts.values():
        assert sum(probabilities) == pytest.approx(1)
        assert all(
            probability == 0 or len(signature) == 3
            for signature, probability in zip(
                SIGNATURES, probabilities, strict=True
            )
        )
    conditioned = groups.condition_signature_probabilities((0.0,) * 11, 3)
    assert sum(conditioned) == pytest.approx(1)
    assert all(
        probability == 0 or len(signature) == 3
        for signature, probability in zip(SIGNATURES, conditioned, strict=True)
    )
    with pytest.raises(ValueError, match="impossible"):
        SpaceGroupForecaster(7, target_group_count=6)
    with pytest.raises(ValueError, match="all 11"):
        groups.condition_signature_probabilities((1.0,), 3)


def test_forecast_helpers_cover_sparse_and_trending_history() -> None:
    assert groups._normalize((0.0, 0.0)) == (0.5, 0.5)
    assert [groups._space_bucket(value) for value in (7, 11, 15, 16)] == [0, 1, 2, 3]
    low = profile_from_spaces((8, 7, 7, 7, 7, 7), 43)
    high = profile_from_spaces((8, 7, 7, 7, 7, 7), 0)
    assert groups._trend_category([low, high, high, high]) == 2
    assert groups._trend_category([high, low, low, low]) == 0
    assert groups._trend_category([low, low]) == 1
    forecaster = SpaceGroupForecaster(7)
    uniform = (1 / len(SIGNATURES),) * len(SIGNATURES)
    assert forecaster._markov(uniform) == uniform
    metric = groups._metrics("border_group_statistical", [(uniform, 0)])
    assert metric.evaluated_draws == 1
    assert metric.log_loss_ci_low == metric.log_loss_ci_high
    assert transition_diagnostics([low], permutations=0)[1:] == (0.0, 1.0)


def test_walk_forward_metrics_and_pattern_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(groups, "MODEL_EVALUATION_WARMUP", 4)
    monkeypatch.setattr(groups, "ML_WARMUP", 2)
    history = profiles(18)
    result = walk_forward_models(history, 7)
    assert result["best_model_id"] in MODEL_IDS
    assert result["provisional"] is False
    assert len(result["metrics"]) == 7
    assert all(metric.evaluated_draws == 14 for metric in result["metrics"])
    assert all(metric.log_loss is not None for metric in result["metrics"])

    matrix, information, p_value = transition_diagnostics(history, permutations=8)
    assert sum(map(sum, matrix)) == len(history) - 1
    assert information >= 0
    assert 0 < p_value <= 1
    assert transition_diagnostics(history[:2], permutations=0)[2] == 1
    statistic, chi_square_p = signature_chi_square(history, 7)
    assert statistic >= 0
    assert 0 <= chi_square_p <= 1


def test_short_walk_forward_is_provisional_and_statistics_exports_tables() -> None:
    short = walk_forward_models([], 7)
    assert short["best_model_id"] is None
    assert short["provisional"] is True
    assert all(metric.evaluated_draws == 0 for metric in short["metrics"])

    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49, date="2026-01-01"))
    draws.add(Draw(3, 6, 12, 22, 36, 47, date="2026-01-08"))
    draws.add(Draw(1, 9, 18, 27, 38, 45, date="2026-01-15"))
    tables, payload = DrawsStatistics(draws).space_group_analysis(7)
    assert set(tables) == {
        "space_group_history",
        "space_group_count_distribution",
        "space_group_size_distribution",
        "space_group_signature_distribution",
        "space_group_transitions",
        "space_group_threshold_sensitivity",
        "space_group_model_metrics",
    }
    assert tables["space_group_history"].iloc[0]["date"] == "2026-01-01"
    assert len(tables["space_group_threshold_sensitivity"]) == 44
    assert payload["borderSpace"] == 7
    assert payload["targetGroupCount"] is None
    assert payload["bestModelId"] is None
    assert payload["provisional"] is True
    forecasts = payload["forecasts"]
    assert isinstance(forecasts, list)
    assert len(forecasts) == 6
