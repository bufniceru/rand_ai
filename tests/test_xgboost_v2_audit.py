"""Research exports, planted generators, and immutable prospective contracts."""

import copy
import json
import sys

import numpy as np
import pytest
import yaml

from scripts import study_xgboost_v2 as study


def dataset(tmp_path, count=243):
    history = study.synthetic_history("fair", 42, count)
    path = tmp_path / "draws.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "lotto_results": {
                    "draws": [
                        {
                            "date": draw.date,
                            "numbers": [ball.value for ball in draw.balls],
                        }
                        for draw in history.draws
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return path, history


def test_synthetic_generators_are_deterministic_valid_and_plant_signal():
    for scenario in ("fair", "bias", "persistence"):
        first = study.synthetic_history(scenario, 42, 300)
        repeated = study.synthetic_history(scenario, 42, 300)
        values = [[ball.value for ball in draw.balls] for draw in first.draws]
        assert values == [
            [ball.value for ball in draw.balls] for draw in repeated.draws
        ]
        assert all(
            len(set(row)) == 6 and min(row) >= 1 and max(row) <= 49 for row in values
        )
        if scenario == "bias":
            assert np.mean([len(set(row) & set(range(1, 7))) for row in values]) > 3
        if scenario == "persistence":
            assert (
                np.mean([len(set(a) & set(b)) for a, b in zip(values, values[1:])]) > 3
            )
    with pytest.raises(ValueError, match="Unknown synthetic"):
        study.synthetic_history("invalid", 42)


def test_feature_transitions_do_not_change_with_future_draws():
    first = study.synthetic_history("fair", 42, 15)
    second = study.synthetic_history("bias", 43, 5)
    rows = study.feature_transitions(first.draws)
    extended = study.feature_transitions([*first.draws, *second.draws])
    for (features, labels), (other_features, other_labels) in zip(rows, extended):
        np.testing.assert_array_equal(features, other_features)
        np.testing.assert_array_equal(labels, other_labels)
    assert rows[0][0].shape == (49, 14)
    assert rows[0][1].sum() == 6


def test_freeze_manifest_detects_prefix_policy_versions_and_no_future(tmp_path):
    source, history = dataset(tmp_path)
    manifest = study.freeze(source, tmp_path / "freeze.json", 241)
    assert study.verify_manifest(history.draws, manifest) == 241
    assert len(manifest["protocol"]["candidates"]) == 8
    altered = copy.deepcopy(manifest)
    altered["protocol"]["seed"] = 0
    with pytest.raises(ValueError, match="protocol"):
        study.verify_manifest(history.draws, altered)
    altered = copy.deepcopy(manifest)
    altered["prefixSha256"] = "changed"
    with pytest.raises(ValueError, match="prefix changed"):
        study.verify_manifest(history.draws, altered)
    with pytest.raises(ValueError, match="appended"):
        study.verify_manifest(history.draws[:241], manifest)
    with pytest.raises(ValueError, match="cutoff"):
        study.freeze(source, tmp_path / "bad.json", 240)
    with pytest.raises(ValueError, match="cutoff"):
        study.freeze(source, tmp_path / "bad.json", 244)


def test_audit_raw_and_deployed_share_targets_and_prospective_cutoff(tmp_path):
    source, _ = dataset(tmp_path)
    manifest = study.freeze(source, tmp_path / "freeze.json", 242)
    report = study.run(source, tmp_path / "audit", manifest, controls=False)
    assert report["evaluation"] == "appended draws after frozen cutoff"
    assert report["fallbackFrequency"] == 1
    assert report["learningControls"] == {"status": "not run"}
    assert all(value["draws"] == 1 for value in report["strategies"].values())
    assert len(report["pairedComparisons"]) == 10
    import csv

    with (tmp_path / "audit/xgboost_v2_predictions.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {row["targetDraw"] for row in rows} == {"243"}
    assert all(len(json.loads(row["probabilities"])) == 49 for row in rows)
    assert (tmp_path / "audit/xgboost_v2_audit.md").is_file()
    assert (
        report["strategies"]["xgboost_classifier_v2"]["logLoss"]
        == report["strategies"]["smoothed_frequency"]["logLoss"]
    )


def test_short_audit_and_cli_dispatch(tmp_path, monkeypatch, capsys):
    source, _ = dataset(tmp_path, 2)
    with pytest.raises(ValueError, match="242 draws"):
        study.run(source, tmp_path / "short", controls=False)
    mocked_report = {
        "strategies": {},
        "fallbackFrequency": 1,
        "learningControls": {"passed": True},
    }
    calls = []
    monkeypatch.setattr(study, "run", lambda *args: calls.append(args) or mocked_report)
    monkeypatch.setattr(sys, "argv", ["study", "audit", "--skip-controls"])
    study.main()
    assert calls[-1][2:] == (None, False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["study", "evaluate", "--manifest", str(manifest_path)]
    )
    study.main()
    assert calls[-1][2:] == ({}, True)
    mocked_report["learningControls"]["passed"] = False
    with pytest.raises(SystemExit, match="Planted-signal recovery failed"):
        study.main()
    monkeypatch.setattr(study, "freeze", lambda *args: {"cutoff": 241})
    monkeypatch.setattr(
        sys,
        "argv",
        ["study", "freeze", "--manifest", str(manifest_path), "--cutoff", "241"],
    )
    study.main()
    assert '"cutoff": 241' in capsys.readouterr().out
