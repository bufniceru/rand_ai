"""Test the guarded Markov Gap-Space Vector correction."""

from __future__ import annotations

import pytest

from rand_ai import Draw, Draws
from rand_ai.mkgsv import (
    CORRECTION_CAP,
    MINIMUM_PAIR_SUPPORT,
    SELECTED_MKGSV_CONFIG,
    CandidateState,
    FeatureGroupState,
    GroupName,
    GroupEvidence,
    MkgsvConfig,
    MkgsvModel,
    MkgsvScore,
    ResidualCounts,
    gap_bucket,
    mkgsv_configurations,
    space_bucket,
)
from rand_ai.strategy_prediction import build_prediction_suites


def test_configuration_grid_buckets_and_residual_formula() -> None:
    configurations = mkgsv_configurations()
    counts = ResidualCounts()
    counts.observe(True, 0.25)
    counts.observe(False, 0.25)

    assert len(configurations) == 48
    assert configurations[0] == MkgsvConfig(32, 128, 256, "historical", 0.0025)
    assert configurations[-1] == MkgsvConfig(64, 256, 512, "combined", 0.005)
    assert SELECTED_MKGSV_CONFIG == MkgsvConfig(
        32,
        128,
        512,
        "historical",
        0.0025,
    )
    assert counts == ResidualCounts(actual_hits=1, expected_hits=0.5, exposures=2)
    assert counts.correction(8) == pytest.approx(0.05)
    assert [gap_bucket(value) for value in (-1, 0, 1, 2, 3, 4, 5, 7)] == [
        0,
        0,
        1,
        2,
        3,
        3,
        4,
        4,
    ]
    assert [gap_bucket(value) for value in (8, 12, 13, 20, 21)] == [5, 5, 6, 6, 7]
    assert [space_bucket(value) for value in (-1, 0, 1, 2, 3, 4)] == [
        0,
        0,
        1,
        2,
        3,
        3,
    ]
    assert [space_bucket(value) for value in (5, 7, 8, 11, 12)] == [4, 4, 5, 5, 6]


def test_historical_and_fresh_spaces_are_circular_directional_and_complete() -> None:
    model = MkgsvModel()
    assert model.state(1).historical is None
    assert model.state(1).fresh is None

    drawn = {1, 10, 20, 30, 40, 49}
    model.remember(drawn)

    one = model.state(1)
    two = model.state(2)
    ten = model.state(10)
    forty_nine = model.state(49)
    assert one.gap == 0
    assert two.gap == 1
    assert one.historical is not None
    assert one.fresh is not None
    assert (one.historical.left_space, one.historical.right_space) == (0, 8)
    assert (one.fresh.left_space, one.fresh.right_space) == (0, 8)
    assert two.historical is None
    assert two.fresh is not None
    assert ten.fresh is not None
    assert forty_nine.fresh is not None
    assert (two.fresh.left_space, two.fresh.right_space) == (0, 7)
    assert (ten.fresh.left_space, ten.fresh.right_space) == (8, 9)
    assert (forty_nine.fresh.left_space, forty_nine.fresh.right_space) == (8, 0)
    assert len({model.state(number).number for number in range(1, 50)}) == 49

    with pytest.raises(ValueError, match="exactly six"):
        model.remember({1, 2, 3})


def test_pending_features_learn_only_after_the_target_outcome() -> None:
    model = MkgsvModel(MkgsvConfig(32, 128, 256, "fresh", 0.0025))
    model.remember({1, 10, 20, 30, 40, 49})
    probabilities = {number: 0.1 for number in range(1, 50)}
    ranking = tuple(range(1, 50))

    decision = model.predict(probabilities, ranking)

    assert model.pending is not None
    assert not model.single_counts
    assert decision.base_ticket == (1, 2, 3, 4, 5, 6)
    model.train({1, 2, 3, 4, 5, 6})

    assert model.pending is None
    assert model.shadow_results == 1
    assert sum(count.exposures for count in model.single_counts.values()) == 110
    assert sum(count.exposures for count in model.pair_counts.values()) == 110
    assert sum(count.exposures for count in model.triple_counts.values()) == 55
    assert sum(count.actual_hits for count in model.triple_counts.values()) == 7


def _seed_group(
    model: MkgsvModel,
    number: int,
    group_name: GroupName,
    *,
    positive: bool,
    exposures: int = 100_000,
) -> None:
    state = model.state(number)
    group = state.historical if group_name == "historical" else state.fresh
    assert group is not None
    actual = exposures if positive else 0
    expected = 0.0 if positive else float(exposures)
    for direction, bucket in (
        ("left", group.left_bucket),
        ("right", group.right_bucket),
    ):
        model.single_counts[(group_name, direction, bucket)] = ResidualCounts(
            actual,
            expected,
            exposures,
        )
        model.pair_counts[(
            group_name,
            direction,
            state.gap_bucket,
            bucket,
        )] = ResidualCounts(actual, expected, exposures)
    model.triple_counts[(
        group_name,
        state.gap_bucket,
        group.left_bucket,
        group.right_bucket,
    )] = ResidualCounts(actual, expected, exposures)


def test_residual_evidence_is_weighted_supported_and_capped() -> None:
    model = MkgsvModel(MkgsvConfig(32, 128, 256, "fresh", 0.0025))
    model.remember({1, 10, 20, 30, 40, 49})
    _seed_group(model, 2, "fresh", positive=True)
    row = model.scores({number: 0.1 for number in range(1, 50)})[2]

    assert row.historical_evidence is None
    assert row.fresh_evidence is not None
    assert row.fresh_evidence.minimum_pair_support >= MINIMUM_PAIR_SUPPORT
    assert row.correction == CORRECTION_CAP
    assert row.corrected_probability == pytest.approx(0.12)

    empty = MkgsvModel(MkgsvConfig(32, 128, 256, "historical", 0.0025))
    empty.remember({1, 10, 20, 30, 40, 49})
    unseen = empty.scores({number: 0.1 for number in range(1, 50)})[2]
    assert unseen.correction == 0
    assert unseen.corrected_probability == 0.1


def _evidence(correction: float) -> GroupEvidence:
    return GroupEvidence(
        correction=correction,
        single_corrections=(correction, correction),
        pair_corrections=(correction, correction),
        triple_correction=correction,
        single_supports=(100, 100),
        pair_supports=(100, 100),
        triple_support=100,
    )


def _row(
    number: int,
    correction: float,
    *,
    historical: float | None = None,
    fresh: float | None = None,
) -> MkgsvScore:
    group = FeatureGroupState(1, 2, 1, 2)
    state = CandidateState(number, 1, 1, group, group)
    base_probability = 0.1
    return MkgsvScore(
        state=state,
        base_probability=base_probability,
        corrected_probability=base_probability + correction,
        correction=correction,
        historical_evidence=(
            None if historical is None else _evidence(historical)
        ),
        fresh_evidence=None if fresh is None else _evidence(fresh),
    )


def _proposal_rows(variant: str = "fresh") -> dict[int, MkgsvScore]:
    rows = {
        number: _row(
            number,
            0,
            historical=0 if variant == "combined" else None,
            fresh=0,
        )
        for number in range(1, 50)
    }
    if variant == "combined":
        rows[6] = _row(6, -0.02, historical=-0.02, fresh=-0.02)
        rows[7] = _row(7, 0.02, historical=0.02, fresh=0.02)
        rows[8] = _row(8, 0.02, historical=0.02, fresh=0.02)
        rows[13] = _row(13, 0.05, historical=0.05, fresh=0.05)
    else:
        rows[6] = _row(6, -0.02, fresh=-0.02)
        rows[7] = _row(7, 0.02, fresh=0.02)
        rows[8] = _row(8, 0.02, fresh=0.02)
        rows[13] = _row(13, 0.05, fresh=0.05)
    return rows


def test_guarded_proposal_replaces_only_rank_six_with_nearby_outsider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MkgsvModel(MkgsvConfig(32, 128, 256, "fresh", 0.0025))
    rows = _proposal_rows()
    monkeypatch.setattr(model, "scores", lambda _probabilities: rows)
    probabilities = {number: 0.1 - number / 100_000 for number in range(1, 50)}
    ranking = tuple(range(1, 50))

    decision = model.predict(probabilities, ranking)

    assert decision.proposed_insider == 6
    assert decision.proposed_outsider == 7
    assert decision.shadow_ticket == (1, 2, 3, 4, 5, 7)
    assert decision.output_ranking == ranking
    assert decision.output_ticket == decision.base_ticket
    assert decision.ranking_scores[1] > decision.ranking_scores[49]
    assert 13 not in decision.shadow_ticket
    assert decision.status.startswith("Shadow warm-up")


def test_combined_variant_requires_historical_and_fresh_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MkgsvModel(MkgsvConfig(32, 128, 256, "combined", 0.0025))
    rows = _proposal_rows("combined")
    rows[7] = _row(7, 0.01, historical=0.02, fresh=-0.03)
    rows[8] = _row(8, 0.01, historical=0.02, fresh=-0.03)
    monkeypatch.setattr(model, "scores", lambda _probabilities: rows)
    probabilities = {number: 0.1 - number / 100_000 for number in range(1, 50)}

    decision = model.predict(probabilities, tuple(range(1, 50)))

    assert decision.proposed_outsider is None
    assert decision.shadow_ranking == decision.base_ranking


def test_guard_edges_fall_back_deterministically() -> None:
    model = MkgsvModel(MkgsvConfig(32, 128, 256, "combined", 0.0025))
    rows = _proposal_rows("combined")
    ranking = tuple(range(1, 50))

    assert rows[1].number == 1
    assert model._combined_correction(None, None) == 0
    assert model._combined_correction(None, _evidence(0.01)) == 0.01
    missing_group = _row(7, 0.01, historical=None, fresh=0.01)
    assert not model._groups_agree(missing_group, rows[6])

    unsupported = _evidence(0.02)
    unsupported = GroupEvidence(
        correction=unsupported.correction,
        single_corrections=unsupported.single_corrections,
        pair_corrections=unsupported.pair_corrections,
        triple_correction=unsupported.triple_correction,
        single_supports=unsupported.single_supports,
        pair_supports=(0, 0),
        triple_support=unsupported.triple_support,
    )
    rows[7] = MkgsvScore(
        state=rows[7].state,
        base_probability=0.1,
        corrected_probability=0.12,
        correction=0.02,
        historical_evidence=unsupported,
        fresh_evidence=unsupported,
    )
    rows[8] = _row(8, 0.001, historical=0.001, fresh=0.001)
    rows[8] = MkgsvScore(
        state=rows[8].state,
        base_probability=0.08,
        corrected_probability=0.081,
        correction=0.001,
        historical_evidence=rows[8].historical_evidence,
        fresh_evidence=rows[8].fresh_evidence,
    )
    rows[9] = _row(9, -0.01, historical=-0.01, fresh=-0.01)
    assert model._proposal(rows, ranking)[0] is None

    model.shadow_results = 100
    assert "no supported" in model._status(False)
    assert "below +3" in model._status(True)
    with pytest.raises(ValueError, match="complete 1-49"):
        model.predict({number: 0.1 for number in range(1, 50)}, (1, 2, 3))


def test_shadow_activation_hysteresis_and_disable_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MkgsvModel(MkgsvConfig(32, 128, 256, "fresh", 0.0025))
    rows = _proposal_rows()
    monkeypatch.setattr(model, "scores", lambda _probabilities: rows)
    probabilities = {number: 0.1 - number / 100_000 for number in range(1, 50)}
    ranking = tuple(range(1, 50))
    winning_shadow = {1, 2, 3, 4, 5, 7}
    winning_base = {1, 2, 3, 4, 5, 6}

    for _ in range(100):
        model.predict(probabilities, ranking)
        model.train(winning_shadow)
    assert model.correction_active
    assert model.activation_count == 1

    for _ in range(59):
        model.predict(probabilities, ranking)
        model.train(winning_base)
    assert sum(model.shadow_deltas) == 2
    assert model.correction_active

    model.predict(probabilities, ranking)
    model.train(winning_base)
    assert sum(model.shadow_deltas) == 0
    assert not model.correction_active


def test_support_distribution_covers_empty_and_populated_groups() -> None:
    model = MkgsvModel()
    assert model.state_support_distribution() == {
        "historicalTripleStates": 0,
        "historicalTripleExposures": 0,
        "historicalTripleMedianSupport": 0.0,
        "freshTripleStates": 0,
        "freshTripleExposures": 0,
        "freshTripleMedianSupport": 0.0,
    }
    model.triple_counts[("fresh", 0, 0, 0)] = ResidualCounts(1, 0.5, 3)
    model.triple_counts[("fresh", 1, 1, 1)] = ResidualCounts(1, 0.5, 5)
    assert model.state_support_distribution()["freshTripleMedianSupport"] == 4.0


def _draws(count: int) -> Draws:
    draws = Draws()
    for index in range(count):
        start = index % 44 + 1
        draws.add(Draw(*range(start, start + 6)))
    draws.prepare_predictions()
    return draws


def test_inactive_strategy_exactly_matches_markov_and_explains_guard() -> None:
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
    assert tuple(item.number for item in mkgsv.numbers) == tuple(
        item.number for item in markov.numbers
    )
    assert len(mkgsv.numbers) == 49
    assert "Champion rank" in mkgsv.numbers[0].details[0]
    assert "Shadow warm-up" in mkgsv.numbers[0].details[-2]


def test_mkgsv_only_loads_markov_dependency_without_returning_it() -> None:
    draws = _draws(3)
    strategies = build_prediction_suites(
        draws.draws,
        history_start=2,
        enabled_strategy_ids=("mkgsv",),
    )[0].strategies
    assert [strategy.strategy_id for strategy in strategies] == ["mkgsv"]


def test_cis_and_chained_do_not_load_mkgsv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rand_ai.strategy_prediction as strategy_prediction

    monkeypatch.setattr(
        strategy_prediction,
        "MkgsvModel",
        lambda: pytest.fail("CIS or Chained loaded MKGSV"),
    )
    draws = _draws(2)
    strategies = build_prediction_suites(
        draws.draws,
        history_start=1,
        enabled_strategy_ids=("cis", "chained"),
    )[0].strategies
    assert {strategy.strategy_id for strategy in strategies} == {"cis", "chained"}


def test_strategy_prediction_is_independent_of_future_dataset_length() -> None:
    prefix = _draws(8)
    extended = _draws(9)
    prefix_strategy = build_prediction_suites(
        prefix.draws,
        history_start=7,
        enabled_strategy_ids=("mkgsv",),
    )[0].strategies[0]
    extended_strategy = build_prediction_suites(
        extended.draws,
        history_start=7,
        enabled_strategy_ids=("mkgsv",),
    )[0].strategies[0]
    assert prefix_strategy.numbers == extended_strategy.numbers
    assert prefix_strategy.top_numbers == extended_strategy.top_numbers
