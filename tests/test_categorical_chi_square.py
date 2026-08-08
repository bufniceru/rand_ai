"""Test exact-state categorical chi-square probability estimation."""

import math

import pytest

from rand_ai.categorical_chi_square import (
    Category,
    CategoricalChiSquareModel,
    ContingencyTable,
)


def _associated_table(current: Category, other: Category) -> ContingencyTable:
    table = ContingencyTable()
    for _ in range(8):
        table.observe(current, True)
    for _ in range(2):
        table.observe(current, False)
    for _ in range(2):
        table.observe(other, True)
    for _ in range(8):
        table.observe(other, False)
    return table


def test_contingency_table_calculates_incremental_statistics_and_evidence() -> None:
    table = _associated_table(1, 2)

    assert table.cell_counts(1) == (8, 10)
    assert table.cell_counts(3) == (0, 0)
    assert table.chi_square() == pytest.approx(7.2)
    assert table.residual(1) == pytest.approx(3 / math.sqrt(5))
    assert table.corrected_cramers_v() == pytest.approx(0.5696002497)

    evidence = table.evidence(1, 0.5, 12)
    assert evidence.probability == pytest.approx(14 / 22)
    assert evidence.support == 10
    assert evidence.hits == 8
    assert evidence.chi_square == pytest.approx(7.2)
    assert evidence.residual > 0
    assert evidence.cramers_v == pytest.approx(table.corrected_cramers_v())
    assert evidence.adjustment > 0


def test_contingency_table_handles_empty_and_degenerate_outcomes() -> None:
    empty = ContingencyTable()
    assert empty.chi_square() == 0
    assert empty.corrected_cramers_v() == 0
    assert empty.residual(3) == 0

    all_hits = ContingencyTable()
    all_hits.observe(1, True)
    assert all_hits.chi_square() == 0
    assert all_hits.corrected_cramers_v() == 0
    assert all_hits.residual(1) == 0

    two_singletons = ContingencyTable()
    two_singletons.observe(1, True)
    two_singletons.observe(2, False)
    assert two_singletons.corrected_cramers_v() == 0

    balanced = ContingencyTable()
    for category in (1, 2):
        balanced.observe(category, True)
        balanced.observe(category, False)
    evidence = balanced.evidence(1, 0.5, 12)
    assert evidence.residual == 0
    assert evidence.adjustment == 0


def test_model_records_pre_draw_states_and_circular_spaces_per_number() -> None:
    model = CategoricalChiSquareModel()
    drawn = {1, 2, 3, 4, 5, 49}

    model.learn(drawn)

    assert model.tables["triple"][1].cell_counts((0, None, None)) == (1, 1)
    assert model.tables["triple"][6].cell_counts((0, None, None)) == (0, 1)
    model.remember(drawn)
    assert model._state(1) == (0, 0, 0)
    assert model._state(49) == (0, 43, 0)
    assert model._state(6) == (1, None, None)

    scores, details = model.scores_and_details()
    assert len(scores) == len(details) == 49
    assert all(0 < score < 1 for score in scores.values())
    assert details[6][0] == "Exact state gap 1, left unseen, right unseen"


def test_hierarchical_effect_uses_triple_pair_single_and_baseline_backoff() -> None:
    def prepared_model() -> CategoricalChiSquareModel:
        model = CategoricalChiSquareModel()
        model.number_hits[1] = 10
        model.number_exposures[1] = 20
        return model

    baseline_model = prepared_model()
    baseline_score, baseline_details = baseline_model._score_number(1)
    assert baseline_details[-1].endswith("effective backoff baseline")

    single_model = prepared_model()
    current_categories = single_model._categories(single_model._state(1))
    for view, current in zip(
        single_model._VIEW_NAMES[:3],
        current_categories[:3],
        strict=True,
    ):
        single_model.tables[view][1] = _associated_table(current, (999,))
    single_score, single_details = single_model._score_number(1)
    assert single_score > baseline_score
    assert single_details[-1].endswith("effective backoff single")

    pair_model = prepared_model()
    pair_categories = pair_model._categories(pair_model._state(1))
    for view, current in zip(
        pair_model._VIEW_NAMES[3:6],
        pair_categories[3:6],
        strict=True,
    ):
        pair_model.tables[view][1] = _associated_table(current, (999, 999))
    pair_score, pair_details = pair_model._score_number(1)
    assert pair_score > baseline_score
    assert pair_details[-1].endswith("effective backoff pair")

    triple_model = prepared_model()
    triple_category = triple_model._categories(triple_model._state(1))[6]
    triple_model.tables["triple"][1] = _associated_table(
        triple_category,
        (999, 999, 999),
    )
    triple_score, triple_details = triple_model._score_number(1)
    assert triple_score > baseline_score
    assert triple_details[-1].endswith("effective backoff triple")


def test_negative_residual_reduces_the_probability() -> None:
    model = CategoricalChiSquareModel()
    model.number_hits[1] = 10
    model.number_exposures[1] = 20
    current = model._categories(model._state(1))[0]
    model.tables["gap"][1] = _associated_table((999,), current)

    probability, details = model._score_number(1)

    assert probability < model._baseline(1)
    assert "residual -" in details[2]
