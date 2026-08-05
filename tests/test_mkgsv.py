"""Test the hierarchical Markov Gap-Space Vector strategy."""

from __future__ import annotations

import pytest

from rand_ai import Draw, Draws
from rand_ai.mkgsv import (
    BASE_HIT_RATE,
    SELECTED_MKGSV_CONFIG,
    BinaryCounts,
    MkgsvConfig,
    MkgsvModel,
    mkgsv_configurations,
)
from rand_ai.strategy_prediction import build_prediction_suites


def test_configuration_grid_and_binary_counts_are_deterministic() -> None:
    configurations = mkgsv_configurations()
    counts = BinaryCounts()
    counts.observe(True)
    counts.observe(False)

    assert len(configurations) == 27
    assert configurations[0] == MkgsvConfig(8.0, 4.0, 2.0)
    assert configurations[-1] == MkgsvConfig(64.0, 32.0, 16.0)
    assert SELECTED_MKGSV_CONFIG == MkgsvConfig(64.0, 4.0, 2.0)
    assert counts == BinaryCounts(hits=1, exposures=2)


def test_remember_preserves_ordered_circular_spaces_and_gaps() -> None:
    model = MkgsvModel()
    drawn = {1, 10, 20, 30, 40, 49}

    model.train(drawn)
    model.remember(drawn)

    assert model.draw_count == 1
    assert model._gap(1) == 0
    assert model._gap(2) == 1
    assert (model.left_spaces[1], model.right_spaces[1]) == (0, 8)
    assert (model.left_spaces[10], model.right_spaces[10]) == (8, 9)
    assert (model.left_spaces[49], model.right_spaces[49]) == (8, 0)
    assert model.x_counts[0] == BinaryCounts(hits=6, exposures=49)
    assert not model.y_counts

    with pytest.raises(ValueError, match="exactly six"):
        model.remember({1, 2, 3})


def test_training_records_all_hierarchy_levels_without_future_state() -> None:
    model = MkgsvModel()
    first = {1, 10, 20, 30, 40, 49}
    second = {1, 2, 3, 4, 5, 6}
    model.train(first)
    model.remember(first)

    model.train(second)

    assert sum(count.exposures for count in model.x_counts.values()) == 98
    assert sum(count.exposures for count in model.y_counts.values()) == 6
    assert sum(count.exposures for count in model.xy_counts.values()) == 6
    assert sum(count.exposures for count in model.xyz_counts.values()) == 6
    assert model.xyz_counts[(0, 0, 8)] == BinaryCounts(hits=1, exposures=1)
    assert model.left_spaces[2] is None

    model.remember(second)
    assert model.left_spaces[1] == 43
    assert model.left_spaces[2] == 0
    assert model.right_spaces[2] == 0


def test_gap_only_and_hierarchical_posteriors_back_off_as_documented() -> None:
    config = MkgsvConfig(8.0, 4.0, 2.0)
    model = MkgsvModel(config)
    model.draw_count = 1
    model.x_counts[1] = BinaryCounts(2, 4)

    gap_only = model.score(2)
    expected_gap = (2 + 8 * BASE_HIT_RATE) / 12
    assert gap_only.probability == pytest.approx(expected_gap)
    assert gap_only.backoff_path == "gap-only"
    assert gap_only.single_supports == (4,)

    model.last_seen[1] = 0
    model.left_spaces[1] = 2
    model.right_spaces[1] = 3
    model.x_counts[0] = BinaryCounts(2, 4)
    model.y_counts[2] = BinaryCounts(3, 5)
    model.z_counts[3] = BinaryCounts(1, 5)
    singles_only = model.score(1)
    assert singles_only.backoff_path == "singles → global"
    assert singles_only.pair_supports == (0, 0, 0)

    model.xy_counts[(0, 2)] = BinaryCounts(2, 3)
    pair_backoff = model.score(1)
    assert pair_backoff.backoff_path == "pairs → singles → global"
    assert pair_backoff.pair_supports == (3, 0, 0)

    model.xyz_counts[(0, 2, 3)] = BinaryCounts(1, 2)
    triple = model.score(1)
    triple_prior = sum(triple.pair_probabilities) / 3
    assert triple.probability == pytest.approx((1 + 2 * triple_prior) / 4)
    assert triple.backoff_path == "triple → pairs → singles → global"
    assert triple.triple_support == 2


def test_left_and_right_are_directional_and_all_numbers_are_scored() -> None:
    model = MkgsvModel(MkgsvConfig(8.0, 4.0, 2.0))
    model.draw_count = 1
    for number, left, right in ((1, 2, 3), (2, 3, 2)):
        model.last_seen[number] = 0
        model.left_spaces[number] = left
        model.right_spaces[number] = right
    model.y_counts[2] = BinaryCounts(8, 10)
    model.y_counts[3] = BinaryCounts(1, 10)
    model.z_counts[2] = BinaryCounts(2, 10)
    model.z_counts[3] = BinaryCounts(7, 10)
    model.xyz_counts[(0, 2, 3)] = BinaryCounts(3, 4)
    model.xyz_counts[(0, 3, 2)] = BinaryCounts(0, 4)

    scores = model.scores()

    assert len(scores) == 49
    assert scores[1].probability != scores[2].probability
    assert scores[1].left_space == 2
    assert scores[2].left_space == 3


def test_support_distribution_handles_empty_odd_and_even_state_counts() -> None:
    model = MkgsvModel()
    assert model.state_support_distribution() == {
        "uniqueTripleStates": 0,
        "tripleExposures": 0,
        "medianTripleSupport": 0.0,
        "singleExposureStates": 0,
        "doubleExposureStates": 0,
        "threeToFiveExposureStates": 0,
        "sixToTenExposureStates": 0,
        "overTenExposureStates": 0,
    }
    model.xyz_counts[(0, 1, 2)] = BinaryCounts(1, 3)
    assert model.state_support_distribution()["medianTripleSupport"] == 3.0
    model.xyz_counts[(1, 2, 3)] = BinaryCounts(1, 5)
    assert model.state_support_distribution() == {
        "uniqueTripleStates": 2,
        "tripleExposures": 8,
        "medianTripleSupport": 4.0,
        "singleExposureStates": 0,
        "doubleExposureStates": 0,
        "threeToFiveExposureStates": 2,
        "sixToTenExposureStates": 0,
        "overTenExposureStates": 0,
    }


def _draws(count: int) -> Draws:
    draws = Draws()
    for index in range(count):
        start = index % 44 + 1
        draws.add(Draw(*range(start, start + 6)))
    draws.prepare_predictions()
    return draws


def test_strategy_builds_complete_rankings_with_explanatory_details() -> None:
    draws = _draws(4)

    strategy = build_prediction_suites(
        draws.draws,
        history_start=3,
        enabled_strategy_ids=("mkgsv",),
    )[0].strategies[0]

    assert strategy.strategy_id == "mkgsv"
    assert strategy.name == "Markov Gap-Space Vector (Experimental)"
    assert len(strategy.numbers) == 49
    assert len(strategy.top_numbers) == 6
    assert strategy.numbers[0].details[0].startswith("State (")
    assert strategy.numbers[0].details[-1] == (
        "Prior strengths 64 single / 4 pair / 2 triple"
    )


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
