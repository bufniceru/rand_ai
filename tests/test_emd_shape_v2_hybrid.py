"""Verify the replacement EMD/V2 hybrid and hidden source dependencies."""

import pytest

from rand_ai import Draw, Draws
from rand_ai.gui_bridge import DEFAULT_STRATEGY_IDS
from rand_ai.strategy_prediction import (
    STRATEGY_IDS,
    _StrategyState,
    _emd_shape_v2_scores,
    _ranking_from_scores,
    build_prediction_suites,
)

ID = "emd_positional_shape_v2_hybrid"


def test_equal_rank_blend_and_existing_tie_breaking() -> None:
    forward = list(range(1, 50))
    scores, details = _emd_shape_v2_scores(
        {"emd": forward, "positional_shape_successor_v2": forward}
    )
    assert scores[1] == 1
    assert scores[25] == 0.5
    assert scores[49] == 0
    assert "EMD weight 50%; rank #1" in details[1]
    opposed, _ = _emd_shape_v2_scores(
        {"emd": forward, "positional_shape_successor_v2": forward[::-1]}
    )
    assert set(opposed.values()) == {0.5}
    gaps = dict.fromkeys(forward, 0)
    gaps[49] = 10
    assert _ranking_from_scores(opposed, gaps)[:6] == [49, 1, 2, 3, 4, 5]


def test_replaces_public_v2_and_keeps_sources_hidden() -> None:
    state = _StrategyState((ID,))
    assert state.requested_strategy_ids == {ID}
    assert state.enabled_strategy_ids == {ID, "emd", "positional_shape_successor_v2"}
    assert state.positional_shape_successor_v2 is not None
    assert ID in STRATEGY_IDS and ID not in DEFAULT_STRATEGY_IDS
    assert "positional_shape_successor_v2" not in STRATEGY_IDS
    draws = Draws()
    draws.add(Draw(1, 8, 20, 30, 40, 49))
    draws.prepare_predictions()
    suite = build_prediction_suites(draws.draws, enabled_strategy_ids=(ID,))[0]
    assert [strategy.strategy_id for strategy in suite.strategies] == [ID]
    assert len(suite.strategies[0].numbers) == 49
    assert len(set(suite.strategies[0].top_numbers)) == 6
    with pytest.raises(ValueError, match="Unknown prediction strategy"):
        build_prediction_suites(
            draws.draws, enabled_strategy_ids=("positional_shape_successor_v2",)
        )


def test_hybrid_preserves_emd_and_uses_both_source_rankings() -> None:
    draws = Draws()
    for numbers in (
        (1, 8, 20, 30, 40, 49),
        (2, 9, 21, 31, 41, 48),
        (3, 10, 22, 32, 42, 47),
    ):
        draws.add(Draw(*numbers))
    draws.prepare_predictions()
    standalone = build_prediction_suites(draws.draws, enabled_strategy_ids=("emd",))
    combined = build_prediction_suites(draws.draws, enabled_strategy_ids=("emd", ID))
    state = _StrategyState((ID,))
    for index, draw in enumerate(draws.draws):
        assert standalone[index].strategies[0] == combined[index].strategies[0]
        state.remember({ball.value for ball in draw.balls})
        gaps = state.current_gaps()
        emd_scores, _ = state._earth_mover_scores()
        assert state.positional_shape_successor_v2 is not None
        v2_scores, _ = state.positional_shape_successor_v2.predict()
        expected, _ = _emd_shape_v2_scores(
            {
                "emd": _ranking_from_scores(emd_scores, gaps),
                "positional_shape_successor_v2": _ranking_from_scores(v2_scores, gaps),
            }
        )
        hybrid = combined[index].strategies[1]
        assert {item.number: item.score for item in hybrid.numbers} == expected
        assert hybrid.top_numbers == tuple(_ranking_from_scores(expected, gaps)[:6])
