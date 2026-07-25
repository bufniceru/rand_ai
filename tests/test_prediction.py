"""Test exact-value combined freshness and proximity predictions."""

import pickle
from io import BytesIO

import pytest

from rand_ai import CombinedPrediction, Draw, Draws, NumberPrediction


def _prediction_draws() -> Draws:
    draws = Draws()
    draws.add(Draw(1, 2, 3, 4, 5, 6))
    draws.add(Draw(1, 7, 8, 9, 10, 11))
    draws.add(Draw(2, 12, 13, 14, 15, 16))
    return draws


def test_attaches_walk_forward_predictions_to_every_draw() -> None:
    """Predictions rank 49 candidates and retain next-draw outcomes for navigation."""
    draws = _prediction_draws()

    draws.prepare_predictions()

    first = draws[0].prediction
    second = draws[1].prediction
    latest = draws[2].prediction
    assert isinstance(first, CombinedPrediction)
    assert isinstance(second, CombinedPrediction)
    assert isinstance(latest, CombinedPrediction)
    assert first.reference_draw_number == 1
    assert first.target_draw_number == 2
    assert first.actual_numbers == (1, 7, 8, 9, 10, 11)
    assert latest.actual_numbers == ()
    assert len(first.numbers) == 49
    assert tuple(item.rank for item in first.numbers) == tuple(range(1, 50))
    assert first.top_numbers == tuple(item.number for item in first.numbers[:6])
    assert all(isinstance(item, NumberPrediction) for item in first.numbers)
    assert all(item.score == pytest.approx(6 / 49) for item in first.numbers)

    by_number = {item.number: item for item in second.numbers}
    assert by_number[1].gap == 0
    assert by_number[2].gap == 1
    assert by_number[49].gap == 2
    assert by_number[1].left_space == draws[1].num1.left_dist
    assert by_number[1].right_space == draws[1].num1.right_dist
    assert by_number[49].left_space is None
    assert by_number[49].right_space is None
    assert by_number[1].freshness_support == 6
    assert by_number[49].freshness_support == 0
    assert by_number[1].score == pytest.approx(
        (by_number[1].freshness_score + by_number[1].proximity_score) / 2
    )


def test_trusted_import_precomputes_predictions_and_empty_collection_is_safe() -> None:
    """Trusted loading performs calculation while an empty collection remains valid."""
    empty = Draws()
    empty.prepare_predictions()
    assert empty.draws == ()

    restored = Draws.load_trusted_pickle(BytesIO(pickle.dumps(_prediction_draws())))

    assert restored[0].prediction is not None
    with pytest.raises(AttributeError):
        restored[0].prediction = None
