"""Test the research-only Sparse Neural Ticket benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.benchmark_sparse_neural_ticket as sparse_ticket
from rand_ai import Draw, Draws
from rand_ai.strategy_prediction import (
    _LAG_LOGISTIC_FEATURE_NAMES,
    _SKLEARN_SVM_FEATURE_NAMES,
)
from scripts.benchmark_sparse_neural_ticket import (
    BenchmarkConfig,
    Evaluation,
    RoundResult,
    TrainedModel,
    _FEATURE_NAMES,
    _active_weight_count,
    _at_least_fifty,
    _clone_masks,
    _clone_state,
    _full_masks,
    _initial_state,
    _model_from_state,
    _non_negative,
    _total_weight_count,
    build_candidate_dataset,
    build_promotion,
    chronological_split,
    fit_standardizer,
    prune_global_magnitude,
    random_mask_like,
    run_pruning_seed,
    select_pruning_round,
)

torch = pytest.importorskip("torch")


def _draws(count: int, *, final_offset: int = 0) -> Draws:
    draws = Draws()
    for index in range(count):
        start = (index + (final_offset if index == count - 1 else 0)) % 44 + 1
        draws.add(Draw(*range(start, start + 6)))
    draws.prepare_predictions()
    return draws


def _targets(draw_count: int) -> np.ndarray:
    targets = np.zeros((draw_count, 49), dtype=np.float64)
    for index in range(draw_count):
        targets[index, index % 44 : index % 44 + 6] = 1
    return targets


def _evaluation(hits: list[int], brier: float) -> Evaluation:
    return Evaluation(
        hits=np.asarray(hits, dtype=np.int64),
        brier_score=brier,
        log_loss=0.5,
        probabilities=np.zeros((len(hits), 49), dtype=np.float64),
    )


def _round(
    round_index: int,
    active_weights: int,
    brier: float,
) -> RoundResult:
    initial = _initial_state(36, 7)
    masks = _full_masks(initial)
    trained = TrainedModel(
        state=initial,
        masks=masks,
        epochs=1,
        training_seconds=0.0,
    )
    return RoundResult(
        round_index=round_index,
        active_weights=active_weights,
        total_weights=100,
        trained=trained,
        validation=_evaluation([1, 1], brier),
    )


def test_builds_named_finite_features_and_next_draw_targets() -> None:
    draws = _draws(7)

    dataset = build_candidate_dataset(draws.draws)

    assert _FEATURE_NAMES == (
        *_SKLEARN_SVM_FEATURE_NAMES,
        *_LAG_LOGISTIC_FEATURE_NAMES[:4],
    )
    assert dataset.features.shape == (6, 49, 36)
    assert dataset.targets.shape == (6, 49)
    assert dataset.target_draw_numbers.tolist() == [2, 3, 4, 5, 6, 7]
    assert np.isfinite(dataset.features).all()
    assert dataset.targets[0].nonzero()[0].tolist() == [1, 2, 3, 4, 5, 6]
    assert set(dataset.baseline_top_numbers) == {
        "randomness",
        "sklearn_svm",
        "lag_logistic",
    }
    assert all(
        tickets.shape == (6, 6)
        for tickets in dataset.baseline_top_numbers.values()
    )


def test_feature_generation_is_independent_of_a_changed_future_draw() -> None:
    original = build_candidate_dataset(_draws(7).draws)
    changed = build_candidate_dataset(_draws(7, final_offset=13).draws)

    assert np.array_equal(original.features, changed.features)
    assert all(
        np.array_equal(
            original.baseline_top_numbers[strategy_id],
            changed.baseline_top_numbers[strategy_id],
        )
        for strategy_id in original.baseline_top_numbers
    )
    assert np.array_equal(original.targets[:-1], changed.targets[:-1])
    assert not np.array_equal(original.targets[-1], changed.targets[-1])


def test_chronological_split_and_scaler_use_training_rows_only() -> None:
    config = BenchmarkConfig(
        warmup_draws=1,
        validation_draws=2,
        holdout_draws=2,
        minimum_train_draws=2,
    )
    split = chronological_split(
        7,
        config,
        minimum_validation_draws=1,
        minimum_holdout_draws=1,
    )
    features = np.zeros((7, 49, 36), dtype=np.float64)
    features[split.train] = 3
    features[split.validation] = 100
    features[split.holdout] = 200

    standardizer = fit_standardizer(features[split.train])

    assert split.train == slice(1, 3)
    assert split.validation == slice(3, 5)
    assert split.holdout == slice(5, 7)
    assert np.array_equal(standardizer.mean, np.full(36, 3.0))
    assert np.array_equal(standardizer.scale, np.ones(36))
    assert np.array_equal(
        standardizer.transform(features[split.train]),
        np.zeros((2, 49, 36)),
    )


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (BenchmarkConfig(validation_draws=49), "Validation requires"),
        (BenchmarkConfig(holdout_draws=49), "Holdout requires"),
        (
            BenchmarkConfig(
                warmup_draws=100,
                validation_draws=150,
                holdout_draws=250,
                minimum_train_draws=200,
            ),
            "at least 700",
        ),
    ),
)
def test_rejects_invalid_or_insufficient_splits(
    config: BenchmarkConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        chronological_split(699, config)


def test_pruning_is_exact_cumulative_and_rewinding_preserves_biases() -> None:
    initial = _initial_state(36, 17)
    trained_state = _clone_state(initial)
    masks = _full_masks(initial)
    total = _total_weight_count(masks)
    ordinal = 1
    for name in masks:
        values = torch.arange(
            ordinal,
            ordinal + trained_state[name].numel(),
            dtype=trained_state[name].dtype,
        ).reshape(trained_state[name].shape)
        trained_state[name] = values
        ordinal += values.numel()

    active_counts = [total]
    for _round_index in range(7):
        previous_masks = _clone_masks(masks)
        masks = prune_global_magnitude(trained_state, masks, 0.20)
        active_counts.append(_active_weight_count(masks))
        assert all(
            torch.all(mask <= previous_masks[name])
            for name, mask in masks.items()
        )

    assert active_counts == [1680, 1344, 1075, 860, 688, 550, 440, 352]
    rewound = _model_from_state(36, initial, masks, 17).state_dict()
    for name, mask in masks.items():
        assert torch.equal(rewound[name][mask == 1], initial[name][mask == 1])
        assert torch.count_nonzero(rewound[name][mask == 0]) == 0
    for bias_name in ("0.bias", "2.bias", "4.bias"):
        assert torch.equal(rewound[bias_name], initial[bias_name])


def test_random_mask_has_exact_size_and_differs_from_selected_mask() -> None:
    initial = _initial_state(36, 19)
    selected = _full_masks(initial)
    selected = prune_global_magnitude(initial, selected, 0.20)

    random_mask = random_mask_like(
        selected,
        _active_weight_count(selected),
        seed=99,
    )

    assert _active_weight_count(random_mask) == _active_weight_count(selected)
    assert any(
        not torch.equal(random_mask[name], selected[name]) for name in selected
    )
    reinitialized = _initial_state(36, 1_000_019)
    assert any(
        not torch.equal(initial[name], reinitialized[name]) for name in selected
    )


def test_runs_and_repeats_every_requested_pruning_round() -> None:
    rng = np.random.default_rng(123)
    training_features = rng.normal(size=(4, 49, 36))
    validation_features = rng.normal(size=(2, 49, 36))
    training_targets = _targets(4)
    validation_targets = _targets(2)
    config = BenchmarkConfig(
        warmup_draws=0,
        validation_draws=2,
        holdout_draws=2,
        minimum_train_draws=1,
        seeds=(31,),
        pruning_rounds=2,
        max_epochs=2,
        patience=2,
        batch_size=64,
    )

    first = run_pruning_seed(
        training_features,
        training_targets,
        validation_features,
        validation_targets,
        seed=31,
        config=config,
    )
    repeated = run_pruning_seed(
        training_features,
        training_targets,
        validation_features,
        validation_targets,
        seed=31,
        config=config,
    )

    assert len(first) == len(repeated) == 3
    assert [result.active_weights for result in first] == [1680, 1344, 1075]
    assert all(
        np.array_equal(left.validation.hits, right.validation.hits)
        and left.validation.brier_score == pytest.approx(
            right.validation.brier_score
        )
        and all(
            torch.equal(left.trained.state[name], right.trained.state[name])
            for name in left.trained.state
        )
        for left, right in zip(first, repeated)
    )


def test_selects_by_hits_then_brier_then_smaller_model() -> None:
    seed_results = [
        (_round(0, 100, 0.2), _round(1, 80, 0.1), _round(2, 60, 0.1)),
        (_round(0, 100, 0.2), _round(1, 80, 0.1), _round(2, 60, 0.1)),
    ]

    assert select_pruning_round(seed_results) == 2


def test_promotion_reports_every_gate_and_is_json_serializable() -> None:
    payload: dict[str, Any] = {
        "meanAcrossSeeds": {"brierScore": 0.1}
    }
    ticket_hits = np.asarray([2.0] * 60)
    dense_hits = np.asarray([2.0] * 60)
    baselines = {
        "sklearn_svm": np.asarray([2.0] * 60),
        "lag_logistic": np.asarray([1.0] * 60),
        "randomness": np.asarray([0.0] * 60),
    }

    promotion = build_promotion(
        active_fraction=0.4,
        ticket_payload=payload,
        dense_payload=payload,
        ticket_hits=ticket_hits,
        dense_hits=dense_hits,
        baseline_hits=baselines,
        deterministic=True,
        finite=True,
    )

    assert promotion["passed"]
    assert len(promotion["checks"]) == 6
    assert promotion["bestCompactBaseline"] == "sklearn_svm"
    assert json.loads(json.dumps(promotion))["passed"] is True

    failed = build_promotion(
        active_fraction=0.8,
        ticket_payload=payload,
        dense_payload=payload,
        ticket_hits=baselines["randomness"],
        dense_hits=dense_hits,
        baseline_hits=baselines,
        deterministic=False,
        finite=False,
    )
    assert not failed["passed"]
    assert sum(check["passed"] for check in failed["checks"]) < 6


def test_small_end_to_end_report_obeys_the_json_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = _draws(103)
    monkeypatch.setattr(
        sparse_ticket,
        "load_lotto_results_yaml",
        lambda _path: draws,
    )
    config = BenchmarkConfig(
        warmup_draws=0,
        validation_draws=50,
        holdout_draws=50,
        minimum_train_draws=2,
        seeds=(41,),
        pruning_rounds=1,
        max_epochs=1,
        patience=1,
        batch_size=256,
    )

    report = sparse_ticket.benchmark(
        Path("synthetic.yaml"),
        config=config,
    )

    assert set(report) == {
        "config",
        "dataset",
        "splits",
        "featureSchema",
        "pruningRounds",
        "selectedSparsity",
        "validation",
        "holdout",
        "controls",
        "featureSurvival",
        "promotion",
    }
    assert report["featureSchema"]["count"] == 36
    assert len(report["pruningRounds"]) == 2
    assert report["splits"]["train"]["draws"] == 2
    assert len(report["promotion"]["checks"]) == 6
    assert json.loads(json.dumps(report, allow_nan=False))["config"]["seeds"] == [
        41
    ]


def test_cli_split_validators() -> None:
    assert _at_least_fifty("50") == 50
    assert _non_negative("0") == 0
    with pytest.raises(Exception, match="at least 50"):
        _at_least_fifty("49")
    with pytest.raises(Exception, match="cannot be negative"):
        _non_negative("-1")
