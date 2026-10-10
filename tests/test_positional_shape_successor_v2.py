"""Verify center/tail distance, confidence shrinkage, and independent V2 forecasts."""

import numpy as np
import pytest

from rand_ai import Draw, Draws
from rand_ai.gui_bridge import DEFAULT_STRATEGY_IDS
from rand_ai.positional_shape_successor import (
    PositionalShapeSuccessorModel,
    shape_features,
)
from rand_ai.positional_shape_successor_v2 import (
    PositionalShapeSuccessorV2Model,
    center_tail_profile,
    shape_distances,
)
from rand_ai.strategy_prediction import (
    _STRATEGY_DEPENDENCIES,
    _StrategyState,
    build_prediction_suites,
)

ID = "emd_positional_shape_v2_hybrid"
V1 = "positional_shape_successor"
A = (1, 8, 20, 30, 40, 49)
B = (2, 9, 21, 31, 41, 48)


def test_profile_boundaries_preserve_direction() -> None:
    assert center_tail_profile(
        np.array([-2, -0.51, -0.5, 0, 0.5, 0.51, 2])
    ).tolist() == [-1, -1, 0, 0, 0, 1, 1]


def test_joint_distance_uses_profiles_and_central_shares() -> None:
    features = np.stack([shape_features(draw) for draw in (A, B, A)])
    spread = np.std(features, axis=0)
    mask = spread > 0
    # A/B/A differs by 3/sqrt(2) SD in every active feature.
    continuous, counts = shape_distances(features, use_profile=False)
    assert continuous == pytest.approx([0, 4.5])
    # Compute categorical contributions independently from the public formula.
    from rand_ai.positional_shape_successor import CENTERS

    z = np.divide(features - CENTERS, spread, out=np.zeros_like(features), where=mask)
    labels = center_tail_profile(z)
    profile = 0.0
    for family in (slice(0, 6), slice(6, 21)):
        active = mask[family]
        before = labels[1, family][active]
        latest = labels[-1, family][active]
        profile += 0.25 * (
            np.mean(before != latest)
            + (np.mean(before == 0) - np.mean(latest == 0)) ** 2
        )
    combined, combined_counts = shape_distances(features)
    assert combined == pytest.approx([0, 0.5 * 4.5 + 0.5 * profile])
    assert counts == combined_counts
    assert counts == (
        np.count_nonzero(labels[-1, :6][mask[:6]] == 0),
        np.count_nonzero(labels[-1, 6:][mask[6:]] == 0),
    )


def test_confidence_shrinks_scores_without_changing_rank() -> None:
    confident = PositionalShapeSuccessorV2Model()
    raw = PositionalShapeSuccessorV2Model(use_confidence=False)
    for model in (confident, raw):
        for _ in range(9):
            model.observe(A)
    scores, details = confident.predict()
    raw_scores, _ = raw.predict()
    # Eight identical neighbors have similarity 1 and effective support 8.
    assert scores[1] == pytest.approx(6 / 49 + 0.5 * (raw_scores[1] - 6 / 49))
    assert scores[2] == pytest.approx(6 / 49 + 0.5 * (raw_scores[2] - 6 / 49))
    assert sum(scores.values()) == pytest.approx(6)
    assert "confidence 50.0%" in details[1][3]
    assert "0 positions, 0 groups" in details[1][2]
    assert sorted(scores, key=scores.__getitem__) == sorted(
        raw_scores, key=raw_scores.__getitem__
    )


def test_weak_and_scarce_matches_have_lower_confidence() -> None:
    near = PositionalShapeSuccessorV2Model()
    far = PositionalShapeSuccessorV2Model()
    for model in (near, far):
        for _ in range(33):
            model.observe(A)
    far.observe((40, 41, 42, 43, 44, 45))
    near_scores, near_details = near.predict()
    far_scores, far_details = far.predict()
    assert "32 of 32 neighbors" in near_details[1][0]
    assert "confidence 80.0%" in near_details[1][3]
    assert abs(far_scores[1] - 6 / 49) < abs(near_scores[1] - 6 / 49)
    assert "confidence 80.0%" not in far_details[1][3]
    assert all(np.isfinite(value) for value in far_scores.values())


def test_empty_short_history_and_ablation_matches_v1() -> None:
    model = PositionalShapeSuccessorV2Model()
    assert set(model.predict()[0].values()) == {6 / 49}
    model.observe(A)
    assert set(model.predict()[0].values()) == {6 / 49}
    assert "confidence 0.0%" in model.predict()[1][1][3]
    v1 = PositionalShapeSuccessorModel()
    ablation = PositionalShapeSuccessorV2Model(use_profile=False, use_confidence=False)
    for numbers in (A, B, A, (3, 9, 22, 32, 41, 48)):
        v1.observe(numbers)
        ablation.observe(numbers)
    assert ablation.predict()[0] == pytest.approx(v1.predict()[0])


def test_v2_is_causal_optional_and_preserves_v1() -> None:
    draws = Draws()
    for numbers in (A, B, A):
        draws.add(Draw(*numbers))
    draws.prepare_predictions()
    only_v1 = build_prediction_suites(draws.draws, enabled_strategy_ids=(V1,))
    together = build_prediction_suites(draws.draws, enabled_strategy_ids=(V1, ID))
    draws.add(Draw(3, 10, 22, 32, 42, 47))
    draws.prepare_predictions()
    extended = build_prediction_suites(draws.draws, enabled_strategy_ids=(V1, ID))
    for index, suite in enumerate(together):
        assert suite.strategies[0] == only_v1[index].strategies[0]
        before, after = suite.strategies[1], extended[index].strategies[1]
        assert before.numbers == after.numbers
        assert before.top_numbers == after.top_numbers
        assert (
            len(before.numbers) == len({item.number for item in before.numbers}) == 49
        )
        assert before.top_numbers == tuple(item.number for item in before.numbers[:6])
        assert [item.score for item in before.numbers] == sorted(
            (item.score for item in before.numbers), reverse=True
        )
    efficacy = extended[-1].strategies[1].efficacy
    assert efficacy is not None and efficacy.evaluated_draws == 3
    assert ID not in DEFAULT_STRATEGY_IDS
    assert all(
        ID not in dependencies for dependencies in _STRATEGY_DEPENDENCIES.values()
    )
    assert _StrategyState((V1,)).positional_shape_successor_v2 is None
    assert _StrategyState((ID,)).positional_shape_successor is None
