"""Verify exact probabilities, causal forecasts and the activation policy."""

import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from rand_ai.draw import Draw
from rand_ai.draws import Draws
import rand_ai.first_number_direction as module
from rand_ai.first_number_direction import (
    CANDIDATES,
    analyze_direction,
    baseline_probabilities,
    cached_direction,
    direction,
    features,
    metrics,
    normalize,
    select_model,
)


def make_draws(minima, dates=None):
    draws = Draws()
    for index, minimum in enumerate(minima):
        draws.add(
            Draw(minimum, 45, 46, 47, 48, 49, date=dates[index] if dates else None)
        )
    return draws


@pytest.mark.parametrize("minimum", range(1, 45))
def test_exact_combinatorial_probabilities(minimum):
    counts = [math.comb(49 - m, 5) for m in range(1, 45)]
    denominator = sum(counts)
    expected = [
        sum(counts[: minimum - 1]) / denominator,
        counts[minimum - 1] / denominator,
        sum(counts[minimum:]) / denominator,
    ]
    assert baseline_probabilities(minimum) == pytest.approx(expected)
    assert sum(expected) == pytest.approx(1)


@pytest.mark.parametrize("value", [0, 45, True, 1.5])
def test_invalid_minimum(value):
    with pytest.raises(ValueError):
        baseline_probabilities(value)


def test_boundaries_normalization_and_ties():
    assert normalize([1, 1, 1], 1) == [0, 0.5, 0.5]
    assert normalize([1, 1, 1], 44) == [0.5, 0.5, 0]
    assert normalize([0, 0, 0], 2) == baseline_probabilities(2)
    assert normalize([float("nan"), 0, 1], 2) == baseline_probabilities(2)
    assert [direction(2, v) for v in (1, 2, 3)] == [0, 1, 2]
    assert int(np.argmax([0.5, 0.5, 0])) == 0


def test_empty_short_and_date_order():
    result = analyze_direction(Draws())
    assert result["forecast"] is None
    assert result["evaluation"]["gated"]["count"] == 0
    result = analyze_direction(make_draws([2]))
    assert result["forecast"]["activeModel"] == "baseline"
    assert result["forecast"]["probabilities"] == baseline_probabilities(2)
    with pytest.raises(ValueError, match="chronological"):
        analyze_direction(make_draws([1, 2], ["2026-02-01", "2026-01-01"]))


def test_features_and_training_are_past_only(monkeypatch):
    fitted = []
    original = module.make_pipeline

    def spy(*args):
        pipeline = original(*args)
        fit = pipeline.fit

        def track(x, y):
            fitted.append((np.array(x), np.array(y)))
            return fit(x, y)

        pipeline.fit = track
        return pipeline

    monkeypatch.setattr(module, "make_pipeline", spy)
    minima = [1 + (i * 7 % 19) for i in range(145)]
    prefix = analyze_direction(make_draws(minima[:130]))
    first_fit = fitted[0]
    fitted.clear()
    extended = analyze_direction(make_draws(minima))
    assert prefix["history"] == extended["history"][: len(prefix["history"])]
    assert prefix["forecast"] == {
        key: extended["history"][10][key] for key in prefix["forecast"]
    }
    np.testing.assert_array_equal(first_fit[0], fitted[0][0])
    np.testing.assert_array_equal(first_fit[1], fitted[0][1])
    assert first_fit[0][-1].tolist() == features(minima[:119])
    assert first_fit[1][-1] == direction(minima[118], minima[119])


def test_missing_classes_equality_and_determinism():
    result = analyze_direction(make_draws([2] * 142))
    assert all(row["actualDirection"] == "equal" for row in result["history"])
    assert result == analyze_direction(make_draws([2] * 142))
    # Two training classes must map their probabilities into the proper slots.
    result = analyze_direction(make_draws([2, 3] * 71))
    for row in result["history"]:
        assert sum(row["models"]["logistic"]) == pytest.approx(1)
        assert row["models"]["logistic"][1] == 0


def test_gate_support_fallback_and_minimum():
    losses = {"baseline": [1.0] * 200, **{name: [0.5] * 200 for name in CANDIDATES}}
    active, evidence = select_model(losses)
    assert active == "historical"
    assert all(item["supported"] for item in evidence["candidates"].values())
    assert select_model(losses) == (active, evidence)
    losses["baseline"] = [0.4] * 200
    assert select_model(losses)[0] == "baseline"
    assert (
        select_model({name: values[:199] for name, values in losses.items()})[0]
        == "baseline"
    )
    losses = {
        "baseline": [1.0] * 200,
        "historical": [1.0] * 200,
        "transition": [0.8, 1.4] * 100,
        "logistic": [1.1] * 200,
    }
    assert select_model(losses)[0] == "baseline"


def test_gate_history_is_causal(monkeypatch):
    calls = []

    def gate(losses):
        calls.append(len(losses["baseline"]))
        return ("historical" if len(losses["baseline"]) >= 200 else "baseline"), {
            "scoredCount": len(losses["baseline"])
        }

    monkeypatch.setattr(module, "select_model", gate)
    minima = [2, 3, 2, 4] * 86
    prefix = analyze_direction(make_draws(minima[:330]))
    extended = analyze_direction(make_draws(minima))
    assert prefix["history"] == extended["history"][:210]
    assert all(row["activeModel"] == "baseline" for row in prefix["history"][:200])
    assert prefix["history"][200]["activeModel"] == "historical"
    assert 200 in calls


def test_metrics():
    rows = [{"actualDirection": "equal", "models": {"test": [0, 1, 0]}}]
    result = metrics(rows, "test")
    assert result["logLoss"] == result["brier"] == 0
    assert result["accuracy"] == 1
    assert result["recall"] == [None, 1, None]


def test_cache_reuse_invalidation_and_recovery(tmp_path, monkeypatch):
    draws = make_draws([2])
    expected = cached_direction(draws, tmp_path)
    original = module.analyze_direction
    monkeypatch.setattr(
        module, "analyze_direction", lambda _: pytest.fail("Cache was not reused")
    )
    assert cached_direction(draws, tmp_path) == expected
    monkeypatch.setattr(module, "analyze_direction", original)
    changed = cached_direction(make_draws([3]), tmp_path)
    assert changed["datasetFingerprint"] != expected["datasetFingerprint"]
    assert len(list(tmp_path.glob("*.json"))) == 2
    for file in tmp_path.glob("*.json"):
        file.write_text("broken")
    assert cached_direction(draws, tmp_path) == expected
    assert cached_direction(draws, tmp_path, refresh=True) == expected
    monkeypatch.setitem(module.PROTOCOL, "version", 999)
    cached_direction(draws, tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 3
    blocked = tmp_path / "file"
    blocked.write_text("x")
    assert cached_direction(draws, blocked)["forecast"]["referenceMinimum"] == 2


def test_bridge_payload(tmp_path):
    from rand_ai.gui_bridge import build_analysis_payload, parse_report_ids

    path = tmp_path / "source.pkl"
    draws = make_draws([2, 3])
    draws.save_pickle(path)
    assert parse_report_ids("first-number-direction") == ("first-number-direction",)
    result = build_analysis_payload(
        draws,
        path,
        enabled_reports=("first-number-direction",),
        enabled_strategies=(),
        strategy_cache_dir=tmp_path / "cache",
    )
    assert (
        result["firstNumberDirection"]["positions"][0]["forecast"]["referenceValue"]
        == 3
    )
    assert result["predictionSuites"] == []
    json.dumps(result, allow_nan=False)
    result = build_analysis_payload(
        draws, path, enabled_reports=(), enabled_strategies=()
    )
    assert result["firstNumberDirection"] is None


def test_study_stages_preserve_only_compatible_results(tmp_path, monkeypatch):
    source = tmp_path / "draws.yaml"
    document = {
        "lotto_results": {
            "draws": [
                {"date": "2026-01-01", "numbers": [2, 10, 20, 30, 40, 49]},
            ]
        }
    }
    source.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "reports"
    monkeypatch.setitem(module.PROTOCOL, "syntheticHistories", 2)
    script = (
        Path(__file__).resolve().parents[1] / "scripts/study_first_number_direction.py"
    )

    def run(stage):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(script),
                "--dataset",
                str(source),
                "--output",
                str(output),
                "--stage",
                stage,
            ],
        )
        runpy.run_path(str(script), run_name="__main__")
        return json.loads((output / "aggregate.json").read_text(encoding="utf-8"))

    assert set(run("all")) == {"historical", "synthetic-00", "synthetic-01"}
    assert set(run("historical")) == {"historical", "synthetic-00", "synthetic-01"}
    assert "0/2" in (output / "report.md").read_text(encoding="utf-8")
    document["lotto_results"]["draws"][0]["numbers"][0] = 3
    source.write_text(json.dumps(document), encoding="utf-8")
    assert set(run("historical")) == {"historical"}


@pytest.mark.parametrize("position", range(1, 7))
def test_position_distribution_baseline_and_symmetry(position):
    prior = module.position_distribution(position)
    assert prior.sum() == pytest.approx(1)
    assert np.all(prior > 0)
    np.testing.assert_allclose(prior, module.position_distribution(7 - position)[::-1])
    for value in range(position, position + 44):
        p = module.position_baseline(value, position)
        # Independent CDF calculation: at least k of the six balls are <= value.
        cdf = sum(
            math.comb(value, count) * math.comb(49 - value, 6 - count)
            for count in range(position, 7)
            if count <= value and 6 - count <= 49 - value
        ) / math.comb(49, 6)
        assert sum(p) == pytest.approx(1)
        assert p[0] + p[1] == pytest.approx(cdf)
        assert min(p) >= 0
        assert p == pytest.approx(
            module.position_baseline(50 - value, 7 - position)[::-1]
        )
    assert module.position_baseline(position, position)[0] == 0
    assert module.position_baseline(position + 43, position)[2] == 0
    assert module.normalize([1, 1, 1], position, position) == [0, 0.5, 0.5]
    assert module.normalize([1, 1, 1], position + 43, position) == [0.5, 0.5, 0]
    for value in (position - 1, position + 44, True, 1.5):
        with pytest.raises(ValueError):
            module.position_baseline(value, position)


@pytest.mark.parametrize("position", [0, 7, True, 1.5])
def test_invalid_position(position):
    with pytest.raises(ValueError):
        module.position_distribution(position)


def random_draws(count):
    rng = np.random.default_rng(42)
    draws = Draws()
    for _ in range(count):
        draws.add(Draw(*map(int, rng.choice(np.arange(1, 50), 6, replace=False))))
    return draws


def test_six_positions_causality_and_legacy_equivalence():
    prefix = module.analyze_number_directions(random_draws(130))
    extended = module.analyze_number_directions(random_draws(145))
    assert prefix["schemaVersion"] == 2
    for position, first, second in zip(
        range(1, 7), prefix["positions"], extended["positions"]
    ):
        assert first["position"] == position
        assert first["history"] == second["history"][:10]
        assert first["forecast"] == {
            key: second["history"][10][key] for key in first["forecast"]
        }
    legacy = analyze_direction(random_draws(145))
    neutral = json.loads(json.dumps(extended["positions"][0]))
    for row in [neutral["forecast"], *neutral["history"]]:
        row["referenceMinimum"] = row.pop("referenceValue")
        if "actualValue" in row:
            row["actualMinimum"] = row.pop("actualValue")
    assert legacy == neutral


def test_position_models_are_independent():
    first, second = Draws(), Draws()
    for i in range(145):
        common = [1 + i % 3, 9 + i % 3, 17 + i % 3, 25 + i % 3, 33 + i % 3]
        first.add(Draw(*common, 44 + i % 3))
        second.add(Draw(*common, 49))
    for position in range(1, 6):
        a, b = (
            module.analyze_position(first, position),
            module.analyze_position(second, position),
        )
        assert a["forecast"] == b["forecast"]
        assert a["history"] == b["history"]
    assert (
        module.analyze_position(first, 6)["forecast"]
        != module.analyze_position(second, 6)["forecast"]
    )


def test_expanded_cache_and_empty_states(tmp_path, monkeypatch):
    empty = module.analyze_number_directions(Draws())
    assert len(empty["positions"]) == 6
    assert all(item["forecast"] is None for item in empty["positions"])
    draws = make_draws([2])
    legacy = cached_direction(draws, tmp_path)
    expanded = module.cached_number_directions(draws, tmp_path)
    assert (
        expanded["positions"][0]["forecast"]["referenceValue"]
        == legacy["forecast"]["referenceMinimum"]
    )
    assert len(list(tmp_path.glob("*.json"))) == 2
    original = module.analyze_number_directions
    monkeypatch.setattr(
        module, "analyze_number_directions", lambda _: pytest.fail("Cache miss")
    )
    assert module.cached_number_directions(draws, tmp_path) == expanded
    monkeypatch.setattr(module, "analyze_number_directions", original)
    for file in tmp_path.glob("number-directions-*.json"):
        file.write_text('{"schemaVersion": 1}', encoding="utf-8")
    assert module.cached_number_directions(draws, tmp_path) == expanded
    assert module.cached_number_directions(draws, tmp_path, True) == expanded
    for file in tmp_path.glob("number-directions-*.json"):
        file.write_text("broken", encoding="utf-8")
    assert module.cached_number_directions(draws, tmp_path) == expanded
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x", encoding="utf-8")
    assert module.cached_number_directions(draws, blocked) == expanded
    dated = make_draws([2], ["2026-01-01"])
    assert (
        module.cached_number_directions(dated, tmp_path)["datasetFingerprint"]
        != expanded["datasetFingerprint"]
    )
    monkeypatch.setitem(module.PROTOCOL, "version", 999)
    module.cached_number_directions(draws, tmp_path)
    assert len(list(tmp_path.glob("number-directions-*.json"))) == 3
