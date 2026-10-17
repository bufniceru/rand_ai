"""Verify the standalone CDM posterior and desktop integration."""

import json
from zipfile import ZipFile

import pytest

from rand_ai import Draw, Draws
from rand_ai.gui_bridge import (
    DEFAULT_STRATEGY_IDS,
    _strategy_payload,
    parse_strategy_ids,
    write_export_archive,
)
from rand_ai.strategy_prediction import _StrategyState, build_prediction_suites


def _history(rows):
    draws = Draws()
    for row in rows:
        draws.add(Draw(*row))
    draws.prepare_predictions()
    return draws


def test_uniform_prior_and_first_draw_posterior():
    state = _StrategyState(("cdm",))
    assert state.enabled_strategy_ids == {"cdm"}
    shares = state._cdm_predictive_shares()
    assert list(shares.values()) == pytest.approx([1 / 49] * 49)
    assert sum(shares.values()) == pytest.approx(1)
    state.train({1, 2, 3, 4, 5, 6})
    state.remember({1, 2, 3, 4, 5, 6})
    shares = state._cdm_predictive_shares()
    assert shares[1] == pytest.approx(2 / 55)
    assert shares[7] == pytest.approx(1 / 55)
    assert sum(shares.values()) == pytest.approx(1)
    assert sum(6 * share for share in shares.values()) == pytest.approx(6)


def test_cdm_hand_calculation_ranking_ties_and_details():
    draws = _history([(1, 2, 3, 4, 5, 6), (1, 7, 8, 9, 10, 11)])
    suites = build_prediction_suites(draws.draws, enabled_strategy_ids=("cdm",))
    strategy = suites[-1].strategies[0]
    assert strategy.strategy_id == "cdm" and strategy.name == "CDM"
    assert len(strategy.numbers) == 49
    assert strategy.top_numbers == (1, 2, 3, 4, 5, 6)
    assert [item.number for item in strategy.numbers[:11]] == list(range(1, 12))
    by_number = {item.number: item for item in strategy.numbers}
    assert by_number[1].score == 1
    assert by_number[2].score == pytest.approx(0.5)
    assert by_number[12].score == 0
    assert by_number[1].details == (
        "Lifetime occurrences 2",
        "Completed draws 2",
        "Symmetric Dirichlet prior alpha 1 per number",
        f"Categorical predictive share {3 / 61:.8%}",
        f"Expected count (multinomial approximation) {18 / 61:.8f}",
        "Not a calibrated probability of inclusion in a distinct-number draw",
    )
    payload = _strategy_payload(strategy)
    assert payload["id"] == "cdm"
    assert payload["topNumbers"] == [1, 2, 3, 4, 5, 6]
    assert len(payload["numbers"]) == 49
    assert payload["numbers"][0]["details"] == list(by_number[1].details)


def test_cdm_all_tied_scores_and_deterministic_number_order():
    draws = _history([(1, 2, 3, 4, 5, 6)])
    state = _StrategyState(("cdm",))
    strategy = state.build_strategies(draws.draws[0].prediction, 0)[0]
    assert strategy.top_numbers == (1, 2, 3, 4, 5, 6)
    assert all(item.score == 0 for item in strategy.numbers)


def test_cdm_future_changes_and_append_do_not_change_prefix_forecasts():
    prefix = [(1, 2, 3, 4, 5, 6), (1, 7, 8, 9, 10, 11)]
    histories = [
        _history(prefix),
        _history([*prefix, (20, 21, 22, 23, 24, 25)]),
        _history([*prefix, (30, 31, 32, 33, 34, 35), (40, 41, 42, 43, 44, 45)]),
    ]
    forecasts = [
        build_prediction_suites(history.draws, enabled_strategy_ids=("cdm",))
        for history in histories
    ]
    for index in range(len(prefix)):
        expected = forecasts[0][index].strategies[0].numbers
        assert all(run[index].strategies[0].numbers == expected for run in forecasts)


def test_cdm_does_not_change_existing_bayesian_forecasts():
    draws = _history([(1, 2, 3, 4, 5, 6), (1, 7, 8, 9, 10, 11)])
    original = build_prediction_suites(draws.draws, enabled_strategy_ids=("bayesian",))
    combined = build_prediction_suites(draws.draws, enabled_strategy_ids=("bayesian", "cdm"))
    for before, after in zip(original, combined, strict=True):
        assert before.strategies[0] == after.strategies[0]


def test_cdm_default_registration_and_export(tmp_path):
    assert "cdm" in DEFAULT_STRATEGY_IDS
    assert parse_strategy_ids("cdm,cdm") == ("cdm",)
    draws = _history([(1, 2, 3, 4, 5, 6), (1, 7, 8, 9, 10, 11)])
    source = tmp_path / "draws.pkl"
    draws.save_pickle(source)
    output = tmp_path / "export.zip"
    write_export_archive(source, output, enabled_reports=(), enabled_strategies=("cdm",))
    with ZipFile(output) as archive:
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["enabledStrategies"] == ["cdm"]
