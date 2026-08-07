"""Tests for the guarded ticket-level MKGSV v3 model."""

from __future__ import annotations

from collections import Counter

import pytest

from rand_ai import Draw, Draws
from rand_ai.mkgsv import (
    MKGSV_PROMOTED,
    NULL_TICKETS_PER_PREDICTION,
    SELECTED_MKGSV_CONFIG,
    MkgsvConfig,
    MkgsvModel,
    MotifVariant,
    TicketEvidence,
    gap_class,
    mkgsv_configurations,
    space_class,
    ticket_vectors,
)
from rand_ai.strategy_prediction import build_prediction_suites


def _probabilities() -> dict[int, float]:
    return {number: 0.20 - number / 10_000 for number in range(1, 50)}


def _ranking() -> tuple[int, ...]:
    return tuple(range(1, 50))


def _neutral_evidence(total: float = 0.0) -> TicketEvidence:
    return TicketEvidence(total, total, total, total, total, ())


def test_configuration_grid_and_bucket_boundaries() -> None:
    configurations = mkgsv_configurations()
    assert len(configurations) == 36
    assert configurations[0] == MkgsvConfig(8, "singles", 0.05)
    assert configurations[-1] == MkgsvConfig(
        64, "all-with-transitions", 0.20
    )
    assert SELECTED_MKGSV_CONFIG == MkgsvConfig(
        24, "singles+doubles+triples", 0.20
    )
    assert not MKGSV_PROMOTED
    assert [gap_class(value) for value in (0, 1, 2, 4, 5, 12, 13, 100)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert [space_class(value) for value in (0, 2, 3, 7, 8, 40)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]


def test_ticket_vectors_use_ordered_circular_spaces() -> None:
    gaps = {number: number - 1 for number in range(1, 50)}
    vectors = ticket_vectors({1, 10, 20, 30, 40, 49}, gaps)
    assert [vector.number for vector in vectors] == [1, 10, 20, 30, 40, 49]
    assert [(vector.left_space, vector.right_space) for vector in vectors] == [
        (0, 8),
        (8, 9),
        (9, 9),
        (9, 9),
        (9, 8),
        (8, 0),
    ]
    assert vectors[0].token == (0, 2)
    assert vectors[-1].space_shape == 6
    assert vectors[0].left_class != vectors[0].right_class
    with pytest.raises(ValueError, match="six distinct"):
        ticket_vectors((1, 2, 3), gaps)
    with pytest.raises(ValueError, match="between 1 and 49"):
        ticket_vectors((0, 1, 2, 3, 4, 5), {**gaps, 0: 0})


def test_pending_actual_and_null_features_settle_only_after_outcome() -> None:
    model = MkgsvModel()
    model.train({1, 2, 3, 4, 5, 6})
    model.remember({1, 10, 20, 30, 40, 49})
    decision = model.predict(_probabilities(), _ranking())
    assert model.pending is not None
    assert not model.actual_counts["single"]
    assert sum(model.pending.null_families["single"].values()) == (
        NULL_TICKETS_PER_PREDICTION * 6
    )
    model.train({2, 4, 6, 8, 10, 12})
    assert model.pending is None
    assert sum(model.actual_counts["single"].values()) == 6
    assert sum(model.null_counts["single"].values()) == 128 * 6
    assert len(model.previous_actual_tokens) == 6
    assert decision.base_ticket == (1, 2, 3, 4, 5, 6)


def test_null_sampling_is_deterministic_and_changes_by_prediction_index() -> None:
    left = MkgsvModel()
    right = MkgsvModel()
    for model in (left, right):
        model.remember({1, 2, 3, 4, 5, 6})
        model.predict(_probabilities(), _ranking())
    assert left.pending is not None and right.pending is not None
    assert left.pending.null_families == right.pending.null_families
    first = left.pending.null_families
    left.remember({7, 8, 9, 10, 11, 12})
    left.predict(_probabilities(), _ranking())
    assert left.pending is not None
    assert left.pending.null_families != first


def test_lifts_recent_blend_and_unseen_states_are_smoothed() -> None:
    model = MkgsvModel(MkgsvConfig(8, "all-with-transitions", 0.1))
    key = (0, 0)
    model.actual_counts["single"][key] = 6
    model.null_counts["single"][key] = 10
    model.null_counts["single"][(1, 1)] = 90
    lifetime_lift, support = model._family_lift("single", key, recent=False)
    unseen_lift, unseen_support = model._family_lift(
        "single", (3, 8), recent=False
    )
    assert lifetime_lift > 0
    assert support == 6
    assert unseen_support == 0
    assert unseen_lift < 0

    model.recent_actual_counts["single"][key] = 1
    model.recent_null_counts["single"][key] = 100
    score, minimum = model._family_score("single", (key,))
    assert score < lifetime_lift
    assert minimum == 6


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("singles", 1.0),
        ("singles+doubles", (0.2 + 0.25 * 2) / 0.45),
        (
            "singles+doubles+triples",
            (0.2 + 0.25 * 2 + 0.25 * 3) / 0.70,
        ),
        (
            "all-with-transitions",
            0.2 + 0.25 * 2 + 0.25 * 3 + 0.30 * 4,
        ),
    ],
)
def test_ticket_component_variants_renormalize_weights(
    monkeypatch: pytest.MonkeyPatch,
    variant: MotifVariant,
    expected: float,
) -> None:
    model = MkgsvModel(MkgsvConfig(8, variant, 0.1))
    family_values = {
        "single": 1.0,
        "full_double": 2.0,
        "gap_double": 2.0,
        "space_double": 2.0,
        "gap_triple": 3.0,
        "space_triple": 3.0,
    }
    monkeypatch.setattr(
        model,
        "_family_score",
        lambda family, _keys: (family_values[family], 7),
    )
    monkeypatch.setattr(model, "_transition_score", lambda _tokens: (4.0, 5))
    evidence = model.ticket_evidence((1, 2, 3, 4, 5, 6), _model_gaps())
    assert evidence.total == pytest.approx(expected)
    assert dict(evidence.supports)["transition"] == 5
    assert "full_triple" not in dict(evidence.supports)


def _model_gaps() -> dict[int, int]:
    return {number: number % 15 for number in range(1, 50)}


def test_transition_evidence_uses_previous_actual_vectors() -> None:
    model = MkgsvModel(MkgsvConfig(8, "all-with-transitions", 0.1))
    assert model._transition_score(((0, 0),)) == (0.0, 0)
    model.previous_actual_tokens = ((0, 0),)
    model.actual_counts["single"][(1, 1)] = 10
    model.transition_counts[((0, 0), (1, 1))] = 8
    model.transition_source_totals[(0, 0)] = 8
    model.recent_actual_counts["single"][(1, 1)] = 2
    model.recent_transition_counts[((0, 0), (1, 1))] = 2
    model.recent_transition_source_totals[(0, 0)] = 2
    lift, support = model._transition_score(((1, 1),))
    assert lift > 0
    assert support == 8


def test_recent_window_evicts_old_actual_null_and_transition_counts() -> None:
    model = MkgsvModel()
    for index in range(121):
        actual = {family: Counter() for family in model.actual_counts}
        null = {family: Counter() for family in model.null_counts}
        actual["single"][(index, 0)] = 1
        null["single"][(index, 0)] = 2
        transition = Counter({((index, 0), (0, 0)): 1})
        model._append_recent(actual, null, transition)
    assert len(model.recent_observations) == 120
    assert model.recent_actual_counts["single"][(0, 0)] == 0
    assert model.recent_null_counts["single"][(0, 0)] == 0
    assert model.recent_transition_counts[((0, 0), (0, 0))] == 0
    assert model.recent_transition_source_totals[(0, 0)] == 0


def test_candidates_protect_top_five_and_offer_exactly_ten_outsiders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MkgsvModel()
    monkeypatch.setattr(model, "ticket_evidence", lambda _ticket, _gaps: _neutral_evidence())
    candidates = model.candidates(_probabilities(), _ranking(), _model_gaps())
    assert len(candidates) == 11
    assert candidates[0].ticket == (1, 2, 3, 4, 5, 6)
    assert [candidate.outsider for candidate in candidates[1:]] == list(range(7, 17))
    assert all(set(candidate.ticket[:5]) == {1, 2, 3, 4, 5} for candidate in candidates)
    assert candidates[-1].ranking[:6] == (1, 2, 3, 4, 5, 16)


def test_candidate_selection_uses_motif_gain_and_falls_back_on_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MkgsvModel(MkgsvConfig(8, "singles", 0.2))

    def evidence(ticket: tuple[int, ...], _gaps: object) -> TicketEvidence:
        return _neutral_evidence(20.0 if 8 in ticket else 0.0)

    monkeypatch.setattr(model, "ticket_evidence", evidence)
    decision = model.predict(_probabilities(), _ranking())
    assert decision.proposed_insider == 6
    assert decision.proposed_outsider == 8
    assert decision.shadow_ticket == (1, 2, 3, 4, 5, 8)
    assert decision.output_ticket == decision.base_ticket

    candidates = model.candidates(_probabilities(), _ranking())
    tied = tuple(
        type(candidate)(
            candidate.ticket,
            candidate.ranking,
            candidate.outsider,
            candidate.outsider_rank,
            1.0,
            candidate.evidence,
            1.0,
        )
        for candidate in candidates
    )
    assert model._select_candidate(tied).outsider is None


def test_promoted_active_decision_applies_only_the_shadow_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MkgsvModel(
        MkgsvConfig(8, "singles", 0.2), promotion_enabled=True
    )
    model.correction_active = True
    model.lifetime_shadow_gain = 5
    model.shadow_deltas_60.extend([1, 1])
    model.shadow_deltas_120.extend([1, 1, 1, 1, 1])
    monkeypatch.setattr(
        model,
        "ticket_evidence",
        lambda ticket, _gaps: _neutral_evidence(20.0 if 7 in ticket else 0.0),
    )
    base_scores = {number: 100.0 - number for number in range(1, 50)}
    decision = model.predict(_probabilities(), _ranking(), base_scores)
    assert decision.output_ticket == (1, 2, 3, 4, 5, 7)
    assert decision.output_ranking[6] == 6
    assert decision.ranking_scores != base_scores
    assert decision.ranking_scores[7] > decision.ranking_scores[6]
    assert decision.correction_active
    assert decision.status.startswith("Active;")


def test_guard_activation_hysteresis_and_unpromoted_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = MkgsvModel(promotion_enabled=True)
    inactive = MkgsvModel(promotion_enabled=False)
    for _ in range(100):
        active._update_guard(1)
        inactive._update_guard(1)
    assert active.correction_active
    assert active.activation_count == 1
    assert not inactive.correction_active
    active._update_guard(-1)
    assert active.correction_active
    for _ in range(59):
        active._update_guard(-1)
    assert not active.correction_active
    assert "Inactive" in active._status(True)
    assert "champion" in active._status(False)
    assert "Benchmark gate failed" in inactive._status(True)

    warm = MkgsvModel(promotion_enabled=True)
    assert "Shadow warm-up" in warm._status(True)
    warm.shadow_results = 100
    warm.lifetime_shadow_gain = 5
    warm.shadow_deltas_60.extend([0] * 60)
    warm.shadow_deltas_120.extend([0] * 100)
    assert "gains lifetime" in warm._status(True)


def test_training_records_only_proposed_shadow_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MkgsvModel(promotion_enabled=True)
    model.remember({1, 10, 20, 30, 40, 49})
    monkeypatch.setattr(
        model,
        "ticket_evidence",
        lambda ticket, _gaps: _neutral_evidence(50 if 7 in ticket else 0),
    )
    model.predict(_probabilities(), _ranking())
    model.train({1, 2, 3, 4, 5, 7})
    assert model.shadow_results == 1
    assert model.lifetime_shadow_gain == 1
    assert model.previous_actual_tokens

    monkeypatch.setattr(
        model, "ticket_evidence", lambda _ticket, _gaps: _neutral_evidence()
    )
    model.predict(_probabilities(), _ranking())
    model.train({1, 2, 3, 4, 5, 6})
    assert model.shadow_results == 1


def test_prediction_validation_and_state_support() -> None:
    model = MkgsvModel()
    with pytest.raises(ValueError, match="complete 1-49"):
        model.predict({1: 0.1}, (1, 2, 3))
    assert model.state_support_distribution()["singleActualStates"] == 0
    model.actual_counts["single"][(0, 0)] = 3
    model.null_counts["single"][(0, 0)] = 5
    model.transition_counts[((0, 0), (0, 0))] = 2
    support = model.state_support_distribution()
    assert support["singleActualMedianSupport"] == 3.0
    assert support["singleNullMedianSupport"] == 5.0
    assert support["transitionMedianSupport"] == 2.0
    with pytest.raises(ValueError, match="exactly six"):
        model.remember({1, 2, 3})


def _draws(count: int) -> Draws:
    draws = Draws()
    for index in range(count):
        start = index % 44 + 1
        draws.add(Draw(*range(start, start + 6)))
    draws.prepare_predictions()
    return draws


def test_failed_gate_output_exactly_matches_markov_scores_and_ranking() -> None:
    draws = _draws(8)
    suite = build_prediction_suites(
        draws.draws,
        history_start=7,
        enabled_strategy_ids=("markov100", "mkgsv"),
    )[0]
    markov = next(item for item in suite.strategies if item.strategy_id == "markov100")
    mkgsv = next(item for item in suite.strategies if item.strategy_id == "mkgsv")
    assert mkgsv.name == "Markov Gap-Space Vector (Experimental)"
    assert mkgsv.top_numbers == markov.top_numbers
    assert tuple((item.number, item.score) for item in mkgsv.numbers) == tuple(
        (item.number, item.score) for item in markov.numbers
    )
    assert "Champion rank" in mkgsv.numbers[0].details[0]
    assert "exact Markov 100 fallback" in mkgsv.numbers[0].details[-2]


def test_mkgsv_dependency_exclusions_and_future_independence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategies = build_prediction_suites(
        _draws(3).draws,
        history_start=2,
        enabled_strategy_ids=("mkgsv",),
    )[0].strategies
    assert [strategy.strategy_id for strategy in strategies] == ["mkgsv"]

    import rand_ai.strategy_prediction as strategy_prediction

    with monkeypatch.context() as context:
        context.setattr(
            strategy_prediction,
            "MkgsvModel",
            lambda: pytest.fail("CIS or Chained loaded MKGSV"),
        )
        strategies = build_prediction_suites(
            _draws(2).draws,
            history_start=1,
            enabled_strategy_ids=("cis", "chained"),
        )[0].strategies
    assert {strategy.strategy_id for strategy in strategies} == {"cis", "chained"}

    prefix = build_prediction_suites(
        _draws(8).draws,
        history_start=7,
        enabled_strategy_ids=("mkgsv",),
    )[0].strategies[0]
    extended = build_prediction_suites(
        _draws(9).draws,
        history_start=7,
        enabled_strategy_ids=("mkgsv",),
    )[0].strategies[0]
    assert prefix.numbers == extended.numbers
    assert prefix.top_numbers == extended.top_numbers
