"""Verify study rules on controlled fixtures and causal forecasts."""

import json
from datetime import date, timedelta

import numpy as np
import pytest

from rand_ai import Draw, Draws
from rand_ai.strategy_prediction import _StrategyState, build_prediction_suites
import scripts.study_statistics as study_module
from scripts.study_statistics import (
    FAMILIES,
    ablate,
    collect,
    holm,
    merge_candidates,
    merge_ranks,
    paired,
    redundancy,
    representative_map,
    summary,
    validate_selection,
    write_report,
)


def test_bootstrap_reproducible_and_holm():
    values = np.tile([0, 1, -1, 2], 20)
    assert paired(values, 1000) == paired(values, 1000)
    assert paired(np.ones(25), 1000)["ci95"] == [1.0, 1.0]
    assert paired(np.zeros(25), 1000)["p"] == 1.0
    with pytest.raises(ValueError):
        paired([])
    claims = [{"p": 0.01}, {"p": 0.04}, {"p": 0.03}]
    holm(claims)
    assert [c["holmP"] for c in claims] == [0.03, 0.06, 0.06]


def test_duplicates_equivalence_and_hit_summary():
    left = [(1, 2, 3, 4, 5, 6)] * 50
    right = [(1, 2, 3, 4, 5, 7)] * 50
    zero = np.zeros(50)
    assert redundancy(left, left, zero, zero)["exactDuplicate"]
    assert redundancy(left, right, zero, zero)["consolidate"]
    assert not redundancy(left, right, np.ones(50), zero)["consolidate"]
    values = summary(np.arange(7), list(np.array_split(np.arange(7), 5)))
    assert values["totalHits"] == 21
    assert values["hitDistribution"] == [1] * 7
    assert values["threePlusRate"] == 4 / 7


def test_merge_selection_uses_discovery_only_and_ties():
    blocks = list(np.array_split(np.arange(50), 5))
    hits = {
        "a": np.array([2] * 10 + [0] * 10 + [1] * 30),
        "b": np.array([0] * 10 + [2] * 10 + [1] * 30),
        "c": np.zeros(50),
    }
    assert merge_candidates(hits, blocks) == [("a", "b")]
    hits["a"][30:] = 6
    assert merge_candidates(hits, blocks) == [("a", "b")]
    assert merge_ranks(list(range(1, 50)), list(range(49, 0, -1))) == tuple(
        range(1, 50)
    )
    for invalid in ([1] * 6, [1, 2, 3], [0, 1, 2, 3, 4, 5]):
        with pytest.raises(ValueError):
            validate_selection(invalid)


def test_equivalence_does_not_chain_unqualified_pairs():
    results = {"a": {"meanHits": 1.0}, "b": {"meanHits": 0.99}, "c": {"meanHits": 0.98}}
    pairs = [
        {"left": "a", "right": "b", "consolidate": True},
        {"left": "b", "right": "c", "consolidate": True},
    ]
    assert representative_map(results, pairs) == {"a": "a", "b": "a", "c": "c"}


@pytest.mark.parametrize("strategy", list(FAMILIES))
def test_ablation_masks_features_and_restores_method(strategy):
    name = {
        "svc": "_svc_features",
        "tbl": "_tbl_features",
        "sklearn_svm": "_sklearn_svm_features",
    }[strategy]
    original = getattr(_StrategyState, name)
    state = _StrategyState([strategy])
    args = (
        (
            10,
            {
                sid: list(range(1, 50))
                for sid in (
                    "mksp",
                    "doublet_triplet_markov",
                    "bayesian",
                    "tbl",
                    "mknp",
                    "emd",
                )
            },
        )
        if strategy == "sklearn_svm"
        else (10,)
    )
    before = original(state, *args)
    for family, indices in FAMILIES[strategy].items():
        with ablate(strategy, family):
            after = getattr(state, name)(*args)
            assert all(after[i] == 0 for i in indices)
            assert all(
                after[i] == before[i] for i in range(len(before)) if i not in indices
            )
        assert getattr(_StrategyState, name) is original


def fixture_draws(changed=False):
    draws = Draws()
    rng = np.random.default_rng(77)
    for index in range(126):
        values = rng.choice(np.arange(1, 50), 6, replace=False).tolist()
        if changed and index >= 124:
            values = [1, 2, 3, 4, 5, 6]
        draws.add(Draw(*values))
    draws.prepare_predictions()
    return draws


def test_future_changes_do_not_change_prior_rankings_or_pending_targets():
    original, _ = collect(fixture_draws(), ["svc", "tbl", "sklearn_svm"])
    changed, _ = collect(fixture_draws(True), ["svc", "tbl", "sklearn_svm"])
    assert [r["target"] for r in original] == list(range(121, 127))
    for a, b in zip(original, changed, strict=True):
        if a["target"] <= 125:
            for sid in a["strategies"]:
                assert (
                    a["strategies"][sid]["ranking"] == b["strategies"][sid]["ranking"]
                )
        for item in a["strategies"].values():
            assert item["hits"] == len(set(item["top"]) & set(a["actual"]))


def test_all_strategies_prefix_is_causal():
    original = Draws()
    changed = Draws()
    rng = np.random.default_rng(99)
    for index in range(8):
        values = rng.choice(np.arange(1, 50), 6, replace=False).tolist()
        original.add(Draw(*values))
        changed.add(Draw(*(values if index < 6 else [1, 2, 3, 4, 5, 6])))
    original.prepare_predictions()
    changed.prepare_predictions()
    left = build_prediction_suites(original.draws)
    right = build_prediction_suites(changed.draws)
    for a, b in zip(left[:6], right[:6], strict=True):
        assert [(s.strategy_id, s.top_numbers, s.numbers) for s in a.strategies] == [
            (s.strategy_id, s.top_numbers, s.numbers) for s in b.strategies
        ]


def test_report_writes_raw_predictions_and_compatible_inventory(tmp_path):
    result = {
        "datasetSha256": "abc",
        "drawCount": 121,
        "blockTargets": [],
        "strategies": {},
        "skipped": {"missing": "unevaluated"},
        "redundancy": [],
        "ablations": [],
        "merges": [],
        "statistics": [],
        "allStrategySeconds": 1,
    }
    records = [
        {
            "target": 121,
            "actual": [1, 2, 3, 4, 5, 6],
            "strategies": {"fixture": {"top": [1, 2, 3, 7, 8, 9], "hits": 3}},
        }
    ]
    write_report(result, records, tmp_path)
    assert json.loads((tmp_path / "statistics_study.json").read_text()) == result
    assert (
        json.loads((tmp_path / "statistics_study_predictions.jsonl").read_text())
        == records[0]
    )
    assert "fixture" in (tmp_path / "statistics_study_hits.csv").read_text()
    assert "Exploratory" in (tmp_path / "statistics_study.md").read_text()


def test_end_to_end_study_is_reproducible_and_marks_missing_artifact(
    monkeypatch, tmp_path
):
    draws = Draws()
    for index in range(125):
        draws.add(
            Draw(
                1,
                2,
                3,
                4,
                5,
                6,
                date=(date(2020, 1, 1) + timedelta(days=index)).isoformat(),
            )
        )
    monkeypatch.setattr(study_module, "load_lotto_results_yaml", lambda path: draws)
    monkeypatch.setattr(study_module, "load_sparse_neural_ticket", lambda: None)

    def fake_collect(draws, ids, cache=None):
        return [
            {
                "target": i + 1,
                "actual": [1, 2, 3, 4, 5, 6],
                "strategies": {
                    sid: {
                        "top": [1, 2, 3, 4, 5, 6],
                        "ranking": list(range(1, 50)),
                        "hits": 6,
                    }
                    for sid in ids
                    if sid != "sparse_neural_ticket" or i >= 122
                },
            }
            for i in range(120, len(draws))
        ], 1.0

    monkeypatch.setattr(study_module, "collect", fake_collect)

    class ImmediatePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, *args):
            from concurrent.futures import Future

            future = Future()
            future.set_result(function(*args))
            return future

    monkeypatch.setattr(study_module, "ProcessPoolExecutor", ImmediatePool)
    dataset = tmp_path / "fixture.yaml"
    dataset.write_text("fixture")
    result, records = study_module.study(dataset)
    repeated, repeated_records = study_module.study(dataset)
    assert result == repeated and records == repeated_records
    assert "sparse_neural_ticket" in result["skipped"]
    assert "sparse_neural_ticket" not in result["strategies"]
    assert len(result["ablations"]) == 13
    assert all(item["exactDuplicate"] for item in result["redundancy"])
    assert len(result["statistics"]) == 19
    write_report(result, records, tmp_path / "report")
    monkeypatch.setattr(study_module, "load_sparse_neural_ticket", lambda: object())
    partial, partial_records = study_module.study(dataset)
    ticket = partial["strategies"]["sparse_neural_ticket"]
    assert ticket["draws"] == 3
    assert ticket["missingForecasts"] == 2
    assert ticket["blocks"] == [None, None, 6.0, 6.0, 6.0]
    assert ticket["evaluatedTargetRange"] == [123, 125]
    assert not ticket["commonHorizon"]
    assert all(
        "sparse_neural_ticket" not in (p["left"], p["right"])
        for p in partial["redundancy"]
    )
    write_report(partial, partial_records, tmp_path / "partial")
    assert (
        "shorter evaluation period"
        in (tmp_path / "partial/statistics_study.md").read_text()
    )
    draws.add(Draw(1, 2, 3, 4, 5, 6, date="2019-01-01"))
    with pytest.raises(ValueError, match="chronological"):
        study_module.study(dataset)


def test_checkpoint_reuses_completed_forecasts(tmp_path, monkeypatch):
    draws = fixture_draws()
    cache = tmp_path / "forecasts.json"
    records, elapsed = collect(draws, ["svc"], cache)
    assert cache.exists()

    def should_not_recompute(*args, **kwargs):
        raise AssertionError("A completed checkpoint should be reused")

    monkeypatch.setattr(study_module, "build_prediction_suites", should_not_recompute)
    assert collect(draws, ["svc"], cache) == (records, elapsed)
