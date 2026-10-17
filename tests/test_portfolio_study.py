"""Verify research statistics, checkpoint provenance and production forecast causality."""

import json

import numpy as np
import pytest

from scripts import study_portfolio as study
from rand_ai import Draw, Draws


def test_streams_and_synthetic_forecasts_are_causal():
    assert study.stream_seed("history", 0) != study.stream_seed("portfolios", 0)
    first = study.synthetic_draws(14, study.stream_seed("history", 0))
    same = study.synthetic_draws(14, study.stream_seed("history", 0))
    assert [[b.value for b in d.balls] for d in first] == [
        [b.value for b in d.balls] for d in same
    ]
    changed = Draws()
    for draw in first.draws[:9]:
        changed.add(Draw(*(ball.value for ball in draw.balls)))
    for _ in range(5):
        changed.add(Draw(1, 2, 3, 4, 5, 6))
    # Distinct rankings are regenerated through the real engine, not copied fixtures.
    strategies = ("freshness", "co_occurrence", "bayesian", "lag_logistic")
    a = study.compact_history(first, strategies)
    b = study.compact_history(changed, strategies)
    assert a["records"][8]["actualNumbers"] != b["records"][8]["actualNumbers"]
    assert [r["strategies"] for r in a["records"][:9]] == [
        r["strategies"] for r in b["records"][:9]
    ]
    assert a["cacheKey"] != b["cacheKey"]
    assert all(len(s["ranking"]) == 49 for r in a["records"] for s in r["strategies"])


def test_bootstrap_holm_and_checkpoint_invalidation(tmp_path):
    assert study.paired_interval(np.zeros(25), 1, 100)["p"] == 1
    assert study.paired_interval(np.ones(50), 1, 100)["interval95"] == [1, 1]
    assert study.paired_interval(np.arange(25), 2, 100) == study.paired_interval(
        np.arange(25), 2, 100
    )
    with pytest.raises(ValueError):
        study.paired_interval([], 1)
    claims = [{"p": 0.01}, {"p": 0.04}, {"p": 0.03}]
    study.holm(claims)
    assert [r["holmP"] for r in claims] == [0.03, 0.06, 0.06]
    checkpoint = tmp_path / "job.json"
    assert not study.checkpoint_matches(checkpoint, "a")
    study.write_json(checkpoint, {"identity": "a"})
    assert study.checkpoint_matches(checkpoint, "a")
    assert not study.checkpoint_matches(checkpoint, "b")
    study.write_json(
        checkpoint,
        {"identity": "a", "data": {"x": 1}, "dataHash": study.digest({"x": 1})},
    )
    assert study.checkpoint_matches(checkpoint, "a")
    study.write_json(
        checkpoint,
        {"identity": "a", "data": {"x": 2}, "dataHash": study.digest({"x": 1})},
    )
    assert not study.checkpoint_matches(checkpoint, "a")
    study.write_json(checkpoint, {"identity": "a", "outputs": {"audit": "missing"}})
    assert not study.checkpoint_matches(checkpoint, "a")
    checkpoint.write_text("broken")
    assert not study.checkpoint_matches(checkpoint, "a")
    assert study.digest({"sources": "a"}) != study.digest({"sources": "b"})


def test_synthetic_history_keeps_frozen_artifact_inactive():
    data = study.compact_history(
        study.synthetic_draws(8, 17), ("freshness", "sparse_neural_ticket")
    )
    assert data["strategyIds"] == ["freshness", "sparse_neural_ticket"]
    assert all(
        [strategy["id"] for strategy in record["strategies"]] == ["freshness"]
        for record in data["records"]
    )


def test_summary_averages_replicates_within_targets_and_writes_report(tmp_path):
    rows = []
    for objective in study.PROTOCOL["objectives"]:
        for target in range(121, 126):
            for method, values in (
                (objective, [3]),
                ("random_global", [0, 2]),
                ("random_pool", [1, 3]),
            ):
                for hits in values:
                    rows.append(
                        {
                            "objective": objective,
                            "method": method,
                            "target": target,
                            "bestHits": hits,
                            "threePlus": int(hits >= 3),
                            "tripleCoverage": 0.5,
                            "coveredTriples": 10,
                            "generatedTickets": 2,
                            "generationMs": 5,
                        }
                    )
        rows.append({"objective": objective, "method": "runtime", "durationMs": 10})
    summary = study.summarize(rows, "fixture")
    random = next(r for r in summary["summaries"] if r["method"] == "random_global")
    assert random["bestHits"] == 1
    assert random["targets"] == 5
    assert random["bestHitDistribution"] == [0.5, 0, 0.5, 0, 0, 0, 0]
    assert len(summary["comparisons"]) == 10
    assert all(r["targets"] == 5 for r in summary["comparisons"])
    job = {
        "name": "fixture",
        "kind": "historical",
        "budget": 2,
        "summary": summary,
        "unavailable": {},
    }
    study.write_report(tmp_path, {"stage": "smoke"}, [job])
    assert "research-only" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert (
        json.loads((tmp_path / "aggregate.json").read_text())["jobs"][0]["budget"] == 2
    )
