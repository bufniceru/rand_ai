"""Test the Python-calculated named prediction strategy suite."""

from typing import Any, cast

import pytest

from rand_ai import (
    Draw,
    Draws,
    PredictionSuite,
    StrategyEfficacy,
    StrategyEfficacyRecord,
    StrategyPrediction,
)
from rand_ai.strategy_prediction import (
    _BASE_PROBABILITY,
    _BAYESIAN_GAP_DECAY,
    _BAYESIAN_NUMBER_DECAY,
    _CIS_EXPERTS,
    _DECISION_TREE_EXPERT_IDS,
    _DECISION_TREE_FEATURE_NAMES,
    _DECISION_TREE_MAX_DEPTH,
    _DECISION_TREE_MINIMUM_LEAF_DRAWS,
    _DECISION_TREE_MINIMUM_TRAINING_DRAWS,
    _DECISION_TREE_TRAINING_WINDOW,
    _EXPECTED_RANDOM_HITS_PER_DRAW,
    _LAG_LOGISTIC_FEATURE_COUNT,
    _LAG_LOGISTIC_MIN_TRAINING_DRAWS,
    _MKFR_MAX_ORDER,
    _MKFR_MIN_CONTEXT_SUPPORT,
    _MKFR_PRIOR_STRENGTH,
    _MKSP_MAX_ORDER,
    _MKSP_MIN_CONTEXT_SUPPORT,
    _MKSP_PRIOR_STRENGTH,
    _MKSP_VALUE_COUNT,
    _NUMBERS_PER_DRAW,
    _SKLEARN_SVM_EXPERT_IDS,
    _SKLEARN_SVM_FEATURE_COUNT,
    _StrategyState,
    _median,
    _proximity_bucket,
    _rank_strength,
    _ranking_from_scores,
    _normalized_positions_for_numbers,
    _relative_dispersion_profile,
    _spaces_for_numbers,
    _variance,
    build_prediction_suites,
)


def test_builds_twenty_seven_named_rankings_and_reports_progress() -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.add(Draw(3, 6, 12, 22, 36, 47))
    draws.add(Draw(1, 9, 18, 27, 38, 45))
    draws.prepare_predictions()
    progress: list[tuple[int, int]] = []

    suites = build_prediction_suites(
        draws.draws,
        history_start=1,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert len(suites) == 2
    assert isinstance(suites[0], PredictionSuite)
    assert suites[0].actual_numbers == (1, 9, 18, 27, 38, 45)
    assert [strategy.name for strategy in suites[-1].strategies] == [
        "Prox",
        "Fresh",
        "EMD",
        "Rand",
        "FRnd",
        "Chi²",
        "Cat χ²",
        "Entr",
        "Mark",
        "Markov Gap-Space Vector (Experimental)",
        "MKFR",
        "MKSP",
        "MKNP",
        "MKRD",
        "Baye",
        "Grid",
        "CoOc",
        "Doublet & Triplet Markov",
        "Mix",
        "SVC",
        "TBL",
        "Scikit Online SVM",
        "Lagged Logistic",
        "CIS",
        "Decision Tree Selector",
        "RCOV",
        "Chained Strategy",
    ]
    assert all(
        isinstance(strategy, StrategyPrediction)
        and len(strategy.numbers) == 49
        and tuple(item.rank for item in strategy.numbers) == tuple(range(1, 50))
        and strategy.top_numbers == tuple(item.number for item in strategy.numbers[:6])
        for strategy in suites[-1].strategies
    )
    residual = next(
        strategy
        for strategy in suites[-1].strategies
        if strategy.strategy_id == "residual_coverage"
    )
    base_top_numbers = set().union(
        *(
            set(strategy.top_numbers)
            for strategy in suites[-1].strategies
            if strategy.strategy_id
            not in {
                "mknp",
                "mkrd",
                "mkgsv",
                "categorical_chi_square",
                "residual_coverage",
                "chained",
            }
        )
    )
    assert not set(residual.top_numbers).intersection(base_top_numbers)
    assert all(
        isinstance(strategy.efficacy, StrategyEfficacy)
        and strategy.efficacy.evaluated_draws == 2
        for strategy in suites[-1].strategies
    )
    random_efficacy = suites[-1].strategies[3].efficacy
    assert random_efficacy is not None
    assert random_efficacy.strategy_hits == random_efficacy.random_hits
    assert random_efficacy.hit_difference == 0
    assert random_efficacy.expected_random_hits == pytest.approx(
        2 * _EXPECTED_RANDOM_HITS_PER_DRAW
    )
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_decision_tree_selector_builds_with_hidden_experts_and_warm_up() -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.prepare_predictions()
    state = _StrategyState(("decision_tree_selector",))

    assert state.requested_strategy_ids == frozenset({"decision_tree_selector"})
    assert set(_DECISION_TREE_EXPERT_IDS).issubset(state.enabled_strategy_ids)
    assert state.decision_tree_selector is not None
    assert state.decision_tree_selector.max_depth == _DECISION_TREE_MAX_DEPTH
    assert (
        state.decision_tree_selector.min_samples_leaf
        == _DECISION_TREE_MINIMUM_LEAF_DRAWS
    )
    assert (
        state.decision_tree_selector_features.maxlen
        == _DECISION_TREE_TRAINING_WINDOW
    )

    first_draw = {1, 2, 8, 17, 31, 49}
    state.train(first_draw)
    state.remember(first_draw)
    combined = draws.draws[0].prediction
    assert combined is not None
    strategies = state.build_strategies(combined, 0)

    assert [strategy.strategy_id for strategy in strategies] == [
        "decision_tree_selector"
    ]
    selector = strategies[0]
    assert len(selector.numbers) == 49
    assert selector.numbers[0].details[0] == "Selected expert: Freshness"
    assert selector.numbers[0].details[1].startswith("Smoothed efficacy fallback")
    assert selector.numbers[0].details[4] == (
        "Warm-up fallback; no fitted tree path"
    )
    assert state.decision_tree_selector_pending_features is not None
    assert (
        len(state.decision_tree_selector_pending_features)
        == len(_DECISION_TREE_FEATURE_NAMES)
    )


def test_decision_tree_selector_trains_multi_output_tree_and_reports_path() -> None:
    state = _StrategyState(("decision_tree_selector",))
    rankings = {
        strategy_id: [
            ((number + expert_index * 3 - 1) % 49) + 1
            for number in range(1, 50)
        ]
        for expert_index, strategy_id in enumerate(_DECISION_TREE_EXPERT_IDS)
    }
    feature_count = len(_DECISION_TREE_FEATURE_NAMES)
    for draw_index in range(_DECISION_TREE_MINIMUM_TRAINING_DRAWS):
        state.decision_tree_selector_pending_features = (
            float(draw_index % 2),
            *(0.0 for _index in range(feature_count - 1)),
        )
        state.decision_tree_selector_pending_rankings = {
            strategy_id: list(ranking)
            for strategy_id, ranking in rankings.items()
        }
        target_expert = _DECISION_TREE_EXPERT_IDS[draw_index % 2]
        state._train_decision_tree_selector(
            set(rankings[target_expert][:_NUMBERS_PER_DRAW])
        )

    assert state.decision_tree_selector_fitted
    assert len(state.decision_tree_selector_features) == (
        _DECISION_TREE_MINIMUM_TRAINING_DRAWS
    )
    state.previous_draw = {1, 2, 8, 17, 31, 49}
    state.previous_previous_draw = {3, 6, 12, 22, 36, 47}
    gaps = {number: number % 17 for number in range(1, 50)}

    scores, details = state._decision_tree_selector_scores(rankings, gaps)
    selected_ranking = sorted(scores, key=scores.__getitem__, reverse=True)

    assert selected_ranking in rankings.values()
    assert details[selected_ranking[0]][1].startswith("Decision tree fitted on")
    assert details[selected_ranking[0]][4].startswith("Tree path: ")
    assert details[selected_ranking[0]][5] == "Selected-expert rank 1"


def test_decision_tree_selector_does_not_learn_from_its_future_draw() -> None:
    prefix = (
        Draw(1, 2, 8, 17, 31, 49),
        Draw(3, 6, 12, 22, 36, 47),
        Draw(1, 9, 18, 27, 38, 45),
        Draw(4, 11, 20, 29, 37, 48),
    )

    def prediction_before(future: Draw) -> StrategyPrediction:
        draws = Draws()
        for draw in (*prefix, future):
            draws.add(draw)
        draws.prepare_predictions()
        return build_prediction_suites(
            draws.draws,
            enabled_strategy_ids=("decision_tree_selector",),
        )[-2].strategies[0]

    first = prediction_before(Draw(5, 14, 23, 32, 40, 46))
    changed_future = prediction_before(Draw(7, 16, 24, 33, 41, 49))

    assert first.top_numbers == changed_future.top_numbers
    assert first.numbers == changed_future.numbers


def test_builds_only_selected_strategy_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.add(Draw(3, 6, 12, 22, 36, 47))
    draws.prepare_predictions()

    def unexpected_emd(
        _state: _StrategyState,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raise AssertionError("disabled EMD strategy was calculated")

    def unexpected_mkfr(
        _state: _StrategyState,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raise AssertionError("disabled MKFR strategy was calculated")

    monkeypatch.setattr(_StrategyState, "_earth_mover_scores", unexpected_emd)
    monkeypatch.setattr(_StrategyState, "_mkfr_scores", unexpected_mkfr)
    suites = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=("freshness", "entropy"),
    )

    assert [strategy.strategy_id for strategy in suites[-1].strategies] == [
        "freshness",
        "entropy",
    ]


def test_sklearn_svm_builds_bounded_features_and_hidden_dependencies() -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.prepare_predictions()
    state = _StrategyState(("sklearn_svm",))
    assert state.requested_strategy_ids == frozenset({"sklearn_svm"})
    assert set(_SKLEARN_SVM_EXPERT_IDS).issubset(state.enabled_strategy_ids)
    assert {"freshness", "proximity", "randomness"}.issubset(
        state.enabled_strategy_ids
    )
    assert state._sklearn_svm_recent_residual(1, 5) == 0
    assert state._sklearn_svm_relationship_residual(1) == 0
    state.previous_draw = {1}
    assert state._sklearn_svm_relationship_residual(1) == 0
    state.previous_draw = set()

    first_draw = {1, 2, 8, 17, 31, 49}
    state.train(first_draw)
    state.remember(first_draw)
    combined = draws.draws[0].prediction
    assert combined is not None
    strategies = state.build_strategies(combined, 0)

    assert [strategy.strategy_id for strategy in strategies] == ["sklearn_svm"]
    assert state.sklearn_svm_trained_draws == 0
    assert all(
        len(features) == _SKLEARN_SVM_FEATURE_COUNT
        and all(-1 <= feature <= 1 for feature in features)
        for features in state.sklearn_svm_pending_features.values()
    )
    assert all(
        item.details[0].startswith("Cold-start consensus")
        for item in strategies[0].numbers
    )


def test_sklearn_svm_trains_pending_batch_with_balanced_weights() -> None:
    class RecordingClassifier:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, Any, Any, Any]] = []

        def partial_fit(
            self,
            features: object,
            labels: object,
            *,
            classes: object = None,
            sample_weight: object = None,
        ) -> None:
            self.calls.append((features, labels, classes, sample_weight))

    state = _StrategyState(("sklearn_svm",))
    recorder = RecordingClassifier()
    cast(Any, state).sklearn_svm = recorder
    state.sklearn_svm_pending_features = {
        number: (number / 49,) * _SKLEARN_SVM_FEATURE_COUNT
        for number in range(1, 50)
    }
    state.sklearn_svm_pending_rankings = {
        strategy_id: list(range(1, 50))
        for strategy_id in _SKLEARN_SVM_EXPERT_IDS
    }
    drawn = {1, 2, 3, 4, 5, 6}

    state._train_sklearn_svm(drawn)

    assert state.sklearn_svm_fitted
    assert state.sklearn_svm_trained_draws == 1
    assert len(recorder.calls) == 1
    features, labels, classes, sample_weights = recorder.calls[0]
    assert getattr(features, "shape") == (49, _SKLEARN_SVM_FEATURE_COUNT)
    assert list(labels) == [1] * 6 + [0] * 43
    assert list(classes) == [0, 1]
    assert list(sample_weights)[:6] == pytest.approx([43 / 6] * 6)
    assert list(sample_weights)[6:] == pytest.approx([1.0] * 43)
    assert all(
        state.sklearn_svm_expert_evaluated_draws[strategy_id] == 1
        and state.sklearn_svm_expert_total_hits[strategy_id] == 6
        for strategy_id in _SKLEARN_SVM_EXPERT_IDS
    )


def test_sklearn_svm_is_deterministic_and_does_not_learn_from_future_draws() -> None:
    prefix = [
        Draw(1, 2, 8, 17, 31, 49),
        Draw(3, 6, 12, 22, 36, 47),
        Draw(1, 9, 18, 27, 38, 45),
        Draw(4, 11, 20, 29, 37, 46),
    ]

    def predictions(extra: Draw) -> tuple[tuple[int, ...], tuple[int, ...]]:
        draws = Draws()
        for draw in (*prefix, extra):
            draws.add(draw)
        draws.prepare_predictions()
        suites = build_prediction_suites(
            draws.draws,
            enabled_strategy_ids=("sklearn_svm",),
        )
        return (
            suites[3].strategies[0].top_numbers,
            suites[-1].strategies[0].top_numbers,
        )

    first = predictions(Draw(5, 14, 23, 32, 40, 48))
    repeated = predictions(Draw(5, 14, 23, 32, 40, 48))
    changed_future = predictions(Draw(7, 15, 24, 33, 41, 49))

    assert first == repeated
    assert first[0] == changed_future[0]


def test_lag_logistic_builds_compact_shared_candidate_features() -> None:
    state = _StrategyState(("lag_logistic",))
    draws = (
        {1, 2, 3, 4, 5, 6},
        {7, 8, 9, 10, 11, 12},
        {13, 14, 15, 16, 17, 18},
    )
    for drawn in draws:
        state.remember(drawn)

    features = state._lag_logistic_feature_matrix()

    assert features.shape == (49, _LAG_LOGISTIC_FEATURE_COUNT)
    assert list(features[0, :4]) == pytest.approx([0, 0, 1, 1 / 3])
    assert list(features[12, :4]) == pytest.approx([1, 0, 0, 1 / 3])
    assert all(-1 <= value <= 1 for value in features.flat)
    assert state._lag_logistic_pattern(1) == (0, 0, 1)
    assert state._lag_logistic_pattern(13) == (1, 0, 0)


def test_lag_logistic_trains_next_draw_targets_and_switches_to_probabilities() -> None:
    state = _StrategyState(("lag_logistic",))
    scores: dict[int, float] = {}
    details: dict[int, tuple[str, ...]] = {}
    for draw_index in range(_LAG_LOGISTIC_MIN_TRAINING_DRAWS + 1):
        start = draw_index % 44 + 1
        drawn = set(range(start, start + 6))
        state.train(drawn)
        state.remember(drawn)
        scores, details = state._lag_logistic_scores()

    assert state.lag_logistic_fitted
    assert state.lag_logistic_trained_draws == _LAG_LOGISTIC_MIN_TRAINING_DRAWS
    assert len(scores) == 49
    assert all(0 <= probability <= 1 for probability in scores.values())
    assert details[1][0].startswith("Estimated hit probability")
    assert details[1][2] == (
        f"Trained transitions {_LAG_LOGISTIC_MIN_TRAINING_DRAWS}"
    )


def test_lag_logistic_is_deterministic_and_does_not_learn_from_future_draws() -> None:
    prefix = [
        Draw(1, 2, 8, 17, 31, 49),
        Draw(3, 6, 12, 22, 36, 47),
        Draw(1, 9, 18, 27, 38, 45),
        Draw(4, 11, 20, 29, 37, 46),
    ]

    def prediction(extra: Draw) -> tuple[int, ...]:
        draws = Draws()
        for draw in (*prefix, extra):
            draws.add(draw)
        draws.prepare_predictions()
        suites = build_prediction_suites(
            draws.draws,
            enabled_strategy_ids=("lag_logistic",),
        )
        return suites[3].strategies[0].top_numbers

    first = prediction(Draw(5, 14, 23, 32, 40, 48))
    repeated = prediction(Draw(5, 14, 23, 32, 40, 48))
    changed_future = prediction(Draw(7, 15, 24, 33, 41, 49))

    assert first == repeated == changed_future


def test_builds_only_selected_composite_and_association_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.add(Draw(3, 6, 12, 22, 36, 47))
    draws.prepare_predictions()

    def unexpected_mknp(
        _state: _StrategyState,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raise AssertionError("standalone MKNP was calculated by an ensemble")

    def unexpected_mkrd(
        _state: _StrategyState,
    ) -> tuple[dict[int, float], dict[int, tuple[str, ...]]]:
        raise AssertionError("standalone MKRD was calculated by an ensemble")

    monkeypatch.setattr(_StrategyState, "_mknp_scores", unexpected_mknp)
    monkeypatch.setattr(_StrategyState, "_mkrd_scores", unexpected_mkrd)

    suites = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=(
            "fresh_random",
            "predictive_grid",
            "co_occurrence",
            "doublet_triplet_markov",
            "mixed",
            "cis",
            "residual_coverage",
            "chained",
        ),
    )

    assert [strategy.strategy_id for strategy in suites[-1].strategies] == [
        "fresh_random",
        "predictive_grid",
        "co_occurrence",
        "doublet_triplet_markov",
        "mixed",
        "cis",
        "residual_coverage",
        "chained",
    ]
    assert all(
        len(strategy.numbers) == 49 and len(strategy.top_numbers) == 6
        for strategy in suites[-1].strategies
    )


def test_new_strategy_math_and_cis_guarded_learner() -> None:
    ranking = list(range(1, 50))
    scores, details = _StrategyState._combine_rankings(((ranking, 1.0),))
    assert scores[1] == pytest.approx(1.12)
    assert details[1] == (
        "Top-quarter agreement 1/1",
        "Top-6 agreement 1/1",
    )

    state = _StrategyState(("cis",), total_draw_count=100)
    assert state._cis_previous_draw_features() == (0.0, 0.0, 0.0)
    state.cis_draw_count = 72
    rankings = {strategy_id: ranking for strategy_id, _label, _weight in _CIS_EXPERTS}
    cis_scores, cis_details = state._cis_scores(rankings)

    assert set(cis_scores) == set(range(1, 50))
    assert cis_details[1][0] == "Adaptive ensemble 100%"
    assert cis_details[1][1].startswith("Guarded ranking learner 0%")
    assert _rank_strength([1], 2) == 0
    assert _variance([]) == 0
    assert _median([]) == 0
    assert _median([3]) == 3
    assert _median([2, 4]) == 3


def test_residual_coverage_prioritizes_uncovered_overdue_numbers() -> None:
    rankings = {
        "first": list(range(1, 50)),
        "second": [*range(7, 50), *range(1, 7)],
    }
    gaps = {number: number for number in range(1, 50)}

    scores, details = _StrategyState._residual_coverage_scores(rankings, gaps)
    ranking = _ranking_from_scores(scores, gaps)

    assert ranking[:6] == [49, 48, 47, 46, 45, 44]
    assert not set(ranking[:6]).intersection(set(range(1, 13)))
    assert details[49] == (
        "Base Top-6 support 0/2",
        "Outside every base Top-6",
        "Current gap 49",
        "Average base rank 46.0",
    )
    assert details[1][1] == "Already covered by the base portfolio"


def test_chained_strategy_tracks_effectiveness_and_builds_six_stages() -> None:
    state = _StrategyState(("chained",))
    ascending = list(range(1, 50))
    descending = list(range(49, 0, -1))
    state.chain_pending_rankings = {
        "freshness": ascending,
        "proximity": descending,
    }
    state._train_chained_effectiveness(set(range(1, 7)))

    assert state.chain_evaluated_draws == 1
    assert state._chain_expert_weight("freshness") > 1
    assert state._chain_expert_weight("proximity") < 1

    gaps = {number: 1 for number in range(1, 50)}
    scores, details = state._chained_scores(
        {
            "freshness": ascending,
            "proximity": descending,
        },
        gaps,
    )
    ranking = _ranking_from_scores(scores, gaps)

    assert len(ranking) == 49
    assert len(set(ranking[:6])) == 6
    assert [details[number][0] for number in ranking[:6]] == [
        f"Chain pick {step}/6" for step in range(1, 7)
    ]
    assert details[ranking[0]][1].startswith(
        "Effectiveness-weighted consensus "
    )
    assert details[ranking[5]][4].startswith("Residual coverage ")
    assert details[ranking[6]][0] == "Reserve rank 7"


def test_chained_strategy_rewards_balanced_draw_shape() -> None:
    state = _StrategyState(("chained",))

    balanced = state._chain_shape_score((3, 12, 21, 30, 39), 48)
    clustered = state._chain_shape_score((3, 12, 21, 30, 39), 4)

    assert balanced > clustered


def test_cis_rewards_proven_experts_and_caps_the_ranking_learner() -> None:
    state = _StrategyState(("cis",))
    strategy_id, _label, base_weight = _CIS_EXPERTS[0]
    comparison_id, _label, comparison_weight = _CIS_EXPERTS[1]
    state.cis_evaluated_draws[strategy_id] = 80
    state.cis_evaluated_draws[comparison_id] = 80
    state.cis_total_hits[strategy_id] = 80
    state.cis_total_hits[comparison_id] = 40
    state.cis_recent_hits[strategy_id].extend([1] * 80)
    state.cis_recent_hits[comparison_id].extend([0] * 80)

    rewarded = state._cis_expert_weight(strategy_id, base_weight) / base_weight
    penalized = (
        state._cis_expert_weight(comparison_id, comparison_weight)
        / comparison_weight
    )

    assert rewarded > 1
    assert penalized < 1
    assert rewarded > penalized

    state.cis_draw_count = 100
    state.cis_recent_ensemble_hits.extend([0] * 80)
    state.cis_recent_learner_hits.extend([6] * 80)
    assert state._cis_learner_blend() == pytest.approx(0.15)


def test_cis_post_warm_up_prediction_is_independent_of_future_dataset_length() -> None:
    number_sets = (
        (1, 8, 15, 22, 29, 36),
        (2, 9, 16, 23, 30, 37),
        (3, 10, 17, 24, 31, 38),
        (4, 11, 18, 25, 32, 39),
    )
    prefix_draws = Draws()
    extended_draws = Draws()
    for index in range(80):
        numbers = number_sets[index % len(number_sets)]
        prefix_draws.add(Draw(*numbers))
        extended_draws.add(Draw(*numbers))
    extended_draws.add(Draw(5, 12, 19, 26, 33, 40))
    prefix_draws.prepare_predictions()
    extended_draws.prepare_predictions()

    prefix_cis = build_prediction_suites(
        prefix_draws.draws,
        history_start=79,
        enabled_strategy_ids=("cis",),
    )[0].strategies[0]
    extended_cis = build_prediction_suites(
        extended_draws.draws,
        history_start=79,
        enabled_strategy_ids=("cis",),
    )[0].strategies[0]

    assert prefix_cis.top_numbers == extended_cis.top_numbers
    assert prefix_cis.numbers == extended_cis.numbers


def test_predictive_grid_blends_earth_mover_similarity() -> None:
    state = _StrategyState(("predictive_grid",))
    scores, details = state._predictive_grid_scores(
        {number: 0 for number in range(1, 50)},
        {
            number: float(number == 1)
            for number in range(1, 50)
        },
    )

    assert scores[1] == pytest.approx(0.30)
    assert scores[2] == 0
    assert details[1][-1] == "Earth-mover similarity 100.0%"
    assert details[2][-1] == "Earth-mover similarity 0.0%"


def test_chi_square_ranks_signed_frequency_residuals() -> None:
    state = _StrategyState(("chi_square",))
    state.draw_count = 49
    for number, observed in enumerate((0, 2, 6, 10, 12), start=1):
        state.appearances[number] = observed

    scores, details = state._chi_square_scores()

    assert scores[1] == 0
    assert scores[3] == pytest.approx(0.5)
    assert scores[5] == 1
    assert [details[number][0] for number in range(1, 6)] == [
        "Strong under",
        "Mild under",
        "Near expected",
        "Mild over",
        "Strong over",
    ]
    assert details[5][1] == "Observed 12 vs expected 6.00"
    assert details[5][3] == "Chi-square contribution 6.000"


def test_composite_and_association_plugins_do_not_learn_from_future_draws() -> None:
    draws = Draws()
    for numbers in (
        (1, 2, 8, 17, 31, 49),
        (3, 6, 12, 22, 36, 47),
        (1, 9, 18, 27, 38, 45),
        (4, 10, 19, 28, 37, 46),
    ):
        draws.add(Draw(*numbers))
    draws.prepare_predictions()
    strategy_ids = (
        "fresh_random",
        "categorical_chi_square",
        "predictive_grid",
        "co_occurrence",
        "doublet_triplet_markov",
        "mixed",
        "cis",
        "residual_coverage",
        "chained",
    )

    prefix = build_prediction_suites(
        draws.draws[:3],
        enabled_strategy_ids=strategy_ids,
    )
    extended = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=strategy_ids,
    )

    assert [strategy.top_numbers for strategy in prefix[-1].strategies] == [
        strategy.top_numbers for strategy in extended[2].strategies
    ]


def test_co_occurrence_adjusts_pair_lift_for_candidate_frequency() -> None:
    state = _StrategyState(("co_occurrence",))
    state.draw_count = 100
    state.previous_draw = {10}
    state.appearances[1] = 40
    state.appearances[2] = 10
    state.appearances[10] = 20
    state.pair_counts[(1, 10)] = 6
    state.pair_counts[(2, 10)] = 6
    state.recent_draws.extend([{1, 2, 10}] * 10)

    scores, details = state._co_occurrence_scores()

    assert scores[2] > scores[1]
    assert details[1][0].startswith("Adjusted average lift ")
    assert details[2][1] == "Positive latest-draw partners 1/1"
    assert details[2][2] == "Recent window 10 draws"
    assert details[2][3].startswith("Strongest partner 10: ")
    assert details[49][1] == "Positive latest-draw partners 0/1"


def test_doublet_triplet_markov_learns_groups_and_next_draw_transitions() -> None:
    state = _StrategyState(("doublet_triplet_markov",))
    neutral_scores, neutral_details = state._doublet_triplet_markov_scores()
    first = {1, 2, 3, 10, 20, 30}
    second = {11, 12, 13, 25, 35, 49}

    state.train(first)
    state.remember(first)
    state.train(second)
    state.remember(second)

    assert neutral_scores == {number: 0 for number in range(1, 50)}
    assert len(neutral_details) == 49
    assert state.doublet_markov_counts[1] == 1
    assert state.doublet_markov_counts[2] == 1
    assert state.triplet_markov_counts[1] == 1
    assert state.doublet_markov_counts[11] == 1
    assert state.doublet_markov_transitions[1][11] == 1
    assert state.doublet_markov_transitions[30][11] == 1
    assert state.triplet_markov_transitions[1][11] == 1
    assert state.doublet_triplet_transition_totals[1] == 1
    assert state.doublet_triplet_shape_transitions[2][2] == 1
    assert state.doublet_triplet_shape_counts == [0, 0, 2]


def test_doublet_triplet_markov_promotes_a_supported_triplet() -> None:
    state = _StrategyState(("doublet_triplet_markov",))
    state.draw_count = 80
    state.previous_draw = {5, 15, 25, 35, 40, 49}
    state.doublet_triplet_shape_counts[:] = [36, 30, 14]
    state.doublet_triplet_shape_transitions[0][:] = [5, 8, 7]
    state.doublet_triplet_shape_transition_totals[0] = 20
    for start in (20, 21):
        state.doublet_markov_counts[start] = 24
    state.triplet_markov_counts[20] = 18
    for previous in state.previous_draw:
        state.doublet_triplet_transition_totals[previous] = 20
        state.doublet_markov_transitions[previous][20] = 8
        state.doublet_markov_transitions[previous][21] = 8
        state.triplet_markov_transitions[previous][20] = 7
    for _index in range(20):
        state.doublet_triplet_recent_groups.append(
            (frozenset({20, 21}), frozenset({20}))
        )

    scores, details = state._doublet_triplet_markov_scores()
    ranked = _ranking_from_scores(
        scores,
        {number: 0 for number in range(1, 50)},
    )

    assert {20, 21, 22}.issubset(ranked[:6])
    assert min(scores[number] for number in (20, 21, 22)) > scores[1]
    assert "Strongest triplet 20-21-22" in details[21][1]
    assert "conditioned on 6 prior numbers" in details[21][3]


def test_bayesian_v2_blends_shrunk_gap_and_recent_number_posteriors() -> None:
    state = _StrategyState(("bayesian",))
    state.bayesian_opportunities[5] = 100
    state.bayesian_hits[5] = 20
    state.bayesian_recent_opportunities[5] = 50
    state.bayesian_recent_hits[5] = 10
    for number in (1, 3, 4, 5, 6, 7):
        state.bayesian_recent_number_hits[number] = 10

    scores, details = state._gap_model_scores(
        {number: 5 for number in range(1, 50)},
        weighted=False,
    )

    assert scores[1] > scores[2]
    assert details[1][0].startswith("Model-averaged probability ")
    assert details[1][1] == "Gap bucket 5"
    assert details[1][2].startswith("Lifetime gap ")
    assert "1000-draw half-life" in details[1][3]
    assert "100-draw half-life" in details[1][4]
    assert details[1][5] == "Hierarchical prior strengths 1024 gap / 64 number"


def test_bayesian_v2_decays_recent_evidence_before_learning_each_draw() -> None:
    state = _StrategyState(("bayesian",))
    state.train({1, 2, 3, 4, 5, 6})
    state.remember({1, 2, 3, 4, 5, 6})
    state.train({7, 8, 9, 10, 11, 12})

    assert state.bayesian_recent_number_hits[1] == pytest.approx(
        _BAYESIAN_NUMBER_DECAY
    )
    assert state.bayesian_recent_number_hits[7] == 1
    assert sum(state.bayesian_recent_opportunities) == pytest.approx(49)
    assert sum(state.bayesian_recent_hits) == pytest.approx(6)

    state.remember({7, 8, 9, 10, 11, 12})
    state.train({13, 14, 15, 16, 17, 18})

    assert sum(state.bayesian_recent_opportunities) == pytest.approx(
        49 * (1 + _BAYESIAN_GAP_DECAY)
    )


def test_allows_every_strategy_plugin_to_be_disabled() -> None:
    suites = build_prediction_suites(
        (Draw(),),
        enabled_strategy_ids=(),
    )

    assert suites[0].strategies == ()


def test_reports_zero_efficacy_until_a_next_draw_is_available() -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.prepare_predictions()

    suite = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=("freshness",),
    )[0]
    efficacy = suite.strategies[0].efficacy

    assert efficacy == StrategyEfficacy(
        evaluated_draws=0,
        strategy_hits=0,
        random_hits=0,
        expected_random_hits=0.0,
        average_hits_per_draw=0.0,
        random_average_hits_per_draw=0.0,
        hit_difference=0,
    )


def test_efficacy_uses_only_results_available_at_each_historical_point() -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.add(Draw(3, 6, 12, 22, 36, 47))
    draws.add(Draw(1, 9, 18, 27, 38, 45))
    draws.prepare_predictions()

    suites = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=("freshness",),
    )

    assert [
        suite.strategies[0].efficacy.evaluated_draws
        for suite in suites
        if suite.strategies[0].efficacy is not None
    ] == [1, 2, 2]


def test_display_history_limit_does_not_limit_efficacy_evaluation() -> None:
    draws = Draws()
    draws.add(Draw(1, 2, 8, 17, 31, 49))
    draws.add(Draw(3, 6, 12, 22, 36, 47))
    draws.add(Draw(1, 9, 18, 27, 38, 45))
    draws.add(Draw(4, 11, 20, 29, 37, 48))
    draws.prepare_predictions()

    efficacy_records: list[StrategyEfficacyRecord] = []
    suites = build_prediction_suites(
        draws.draws,
        history_start=3,
        enabled_strategy_ids=("freshness",),
        efficacy_record=efficacy_records.append,
    )

    assert len(suites) == 1
    efficacy = suites[0].strategies[0].efficacy
    assert efficacy is not None
    assert efficacy.evaluated_draws == 3
    assert [record.target_draw_number for record in efficacy_records] == [2, 3, 4]
    assert all(
        record.strategy_hits[0][0] == "freshness"
        and record.actual_numbers
        == tuple(ball.value for ball in draws.draws[index + 1].balls)
        and 0 <= record.strategy_hits[0][1] <= 6
        and 0 <= record.random_hits <= 6
        for index, record in enumerate(efficacy_records)
    )


def test_rejects_unknown_strategy_plugin() -> None:
    with pytest.raises(ValueError, match="Unknown prediction strategy"):
        build_prediction_suites(
            (Draw(),),
            enabled_strategy_ids=("unknown",),
        )


def test_mkfr_learns_joint_binary_context_transitions_for_each_order() -> None:
    state = _StrategyState(("mkfr",))
    for outcome in (1, 0, 1):
        drawn = {1} if outcome else set()
        state.train(drawn)
        state.remember(drawn)

    assert state.mkfr_transitions[1][0][1] == [1, 0]
    assert state.mkfr_transitions[1][0][0] == [0, 1]
    assert state.mkfr_transitions[1][1][0b10] == [0, 1]
    assert list(state.mkfr_histories[1]) == [1, 0, 1]
    assert list(state.mkfr_histories[2]) == [0, 0, 0]


def test_mkfr_backs_off_from_an_unsupported_longer_context() -> None:
    state = _StrategyState(("mkfr",))
    state.draw_count = 100
    state.appearances[1] = 20
    state.mkfr_histories[1].extend((1, 0))
    state.mkfr_transitions[1][0][0] = [8, 2]
    state.mkfr_transitions[1][1][0b10] = [
        _MKFR_MIN_CONTEXT_SUPPORT - 1,
        0,
    ]
    baseline = (20 + _MKFR_PRIOR_STRENGTH * _BASE_PROBABILITY) / (
        100 + _MKFR_PRIOR_STRENGTH
    )
    expected_order_one = (2 + _MKFR_PRIOR_STRENGTH * baseline) / (
        10 + _MKFR_PRIOR_STRENGTH
    )

    probability, support, selected_order = state._mkfr_probability(1)

    assert probability == pytest.approx(expected_order_one)
    assert support == 10
    assert selected_order == 1

    state.mkfr_transitions[1][1][0b10] = [6, 2]
    expected_order_two = (2 + _MKFR_PRIOR_STRENGTH * expected_order_one) / (
        8 + _MKFR_PRIOR_STRENGTH
    )

    probability, support, selected_order = state._mkfr_probability(1)

    assert probability == pytest.approx(expected_order_two)
    assert support == 8
    assert selected_order == 2


def test_mkfr_supports_joint_contexts_through_order_twenty() -> None:
    state = _StrategyState(("mkfr",))
    state.mkfr_histories[1].extend([1] * _MKFR_MAX_ORDER)
    for order in range(1, _MKFR_MAX_ORDER + 1):
        context = (1 << order) - 1
        state.mkfr_transitions[1][order - 1][context] = [
            0,
            _MKFR_MIN_CONTEXT_SUPPORT,
        ]

    probability, support, selected_order = state._mkfr_probability(1)
    scores, details = state._mkfr_scores()

    assert probability > _BASE_PROBABILITY
    assert support == _MKFR_MIN_CONTEXT_SUPPORT
    assert selected_order == _MKFR_MAX_ORDER
    assert scores[1] > scores[2]
    assert details[1][3] == (
        f"Order {_MKFR_MAX_ORDER}/{_MKFR_MAX_ORDER}: {'1' * _MKFR_MAX_ORDER}"
    )
    assert details[1][4] == (f"Context support {_MKFR_MIN_CONTEXT_SUPPORT}")


def test_mkfr_ranks_transition_lift_above_each_numbers_baseline() -> None:
    state = _StrategyState(("mkfr",))
    state.draw_count = 100
    state.appearances[1] = 20
    state.appearances[2] = 10
    state.mkfr_histories[1].append(0)
    state.mkfr_histories[2].append(0)
    state.mkfr_transitions[1][0][0] = [79, 21]
    state.mkfr_transitions[2][0][0] = [85, 15]

    scores, details = state._mkfr_scores()

    assert state._mkfr_probability(1)[0] > state._mkfr_probability(2)[0]
    assert scores[2] > scores[1]
    assert details[1][2].startswith("Transition lift +")
    assert details[2][2].startswith("Transition lift +")


def test_mkfr_uses_prior_for_empty_history_and_truncates_to_twenty_draws() -> None:
    state = _StrategyState(("mkfr",))

    assert state._mkfr_probability(1) == (_BASE_PROBABILITY, 0, 0)

    outcomes = tuple(index % 2 for index in range(25))
    for outcome in outcomes:
        drawn = {1} if outcome else set()
        state.train(drawn)
        state.remember(drawn)

    assert list(state.mkfr_histories[1]) == list(outcomes[-_MKFR_MAX_ORDER:])


def test_mkfr_prediction_does_not_learn_from_a_future_draw() -> None:
    prefix = (
        Draw(1, 2, 8, 17, 31, 49),
        Draw(3, 6, 12, 22, 36, 47),
        Draw(1, 9, 18, 27, 38, 45),
    )
    without_future = Draws()
    with_future = Draws()
    for draw in prefix:
        without_future.add(draw)
        with_future.add(Draw(*(ball.value for ball in draw.balls)))
    with_future.add(Draw(4, 11, 20, 29, 37, 48))
    without_future.prepare_predictions()
    with_future.prepare_predictions()

    prefix_strategy = build_prediction_suites(
        without_future.draws,
        enabled_strategy_ids=("mkfr",),
    )[-1].strategies[0]
    future_strategy = build_prediction_suites(
        with_future.draws,
        enabled_strategy_ids=("mkfr",),
    )[-2].strategies[0]

    assert prefix_strategy.top_numbers == future_strategy.top_numbers
    assert prefix_strategy.numbers == future_strategy.numbers


def test_mksp_observes_all_six_space_values_and_context_orders() -> None:
    state = _StrategyState(("mksp",))
    draws = (
        {1, 10, 20, 30, 40, 49},
        {2, 11, 21, 31, 41, 48},
        {1, 2, 3, 4, 5, 6},
    )
    for drawn in draws:
        state.train(drawn)
        state.remember(drawn)

    assert _spaces_for_numbers(draws[0]) == (0, 8, 9, 9, 9, 8)
    assert [list(history) for history in state.mksp_histories] == [
        [0, 2, 43],
        [8, 8, 0],
        [9, 9, 0],
        [9, 9, 0],
        [9, 9, 0],
        [8, 6, 0],
    ]
    assert state.mksp_transitions[0][0][(0,)][2] == 1
    assert state.mksp_transitions[0][0][(2,)][43] == 1
    assert state.mksp_transitions[0][1][(0, 2)][43] == 1
    assert sum(sum(counts) for counts in state.mksp_value_counts) == 18
    with pytest.raises(ValueError, match="exactly six"):
        _spaces_for_numbers({1, 2, 3})


def test_mksp_backs_off_and_uses_categorical_bayesian_smoothing() -> None:
    state = _StrategyState(("mksp",))
    state.draw_count = 100
    state.mksp_value_counts[0][5] = 20
    state.mksp_histories[0].extend((3, 4))
    state.mksp_transitions[0][0][(4,)] = {5: 2, 6: 8}
    state.mksp_transitions[0][1][(3, 4)] = {
        5: 1,
        6: _MKSP_MIN_CONTEXT_SUPPORT - 2,
    }
    baseline = (
        20 + _MKSP_PRIOR_STRENGTH / _MKSP_VALUE_COUNT
    ) / (100 + _MKSP_PRIOR_STRENGTH)
    expected_order_one = (2 + _MKSP_PRIOR_STRENGTH * baseline) / (
        10 + _MKSP_PRIOR_STRENGTH
    )

    probability, support, selected_order = state._mksp_probability(0, 5)

    assert probability == pytest.approx(expected_order_one)
    assert support == 10
    assert selected_order == 1

    state.mksp_transitions[0][1][(3, 4)] = {5: 2, 6: 6}
    expected_order_two = (2 + _MKSP_PRIOR_STRENGTH * expected_order_one) / (
        8 + _MKSP_PRIOR_STRENGTH
    )

    probability, support, selected_order = state._mksp_probability(0, 5)

    assert probability == pytest.approx(expected_order_two)
    assert support == 8
    assert selected_order == 2


def test_mksp_supports_order_twenty_and_scores_valid_complete_draws() -> None:
    state = _StrategyState(("mksp",))
    state.draw_count = 100
    state.previous_draw = {1, 10, 20, 30, 40, 49}
    for position in range(6):
        state.mksp_histories[position].extend([1] * _MKSP_MAX_ORDER)
        state.mksp_value_counts[position][2] = 10
        for order in range(1, _MKSP_MAX_ORDER + 1):
            context = (1,) * order
            state.mksp_transitions[position][order - 1][context] = {
                2: _MKSP_MIN_CONTEXT_SUPPORT,
            }

    probability, support, selected_order = state._mksp_probability(0, 2)
    scores, details = state._mksp_scores()

    assert probability > state._mksp_baseline_probability(0, 2)
    assert support == _MKSP_MIN_CONTEXT_SUPPORT
    assert selected_order == _MKSP_MAX_ORDER
    assert set(scores) == set(range(1, 50))
    assert details[1][0].startswith("Marginal probability ")
    assert details[1][2].startswith("Best generated draw ")
    assert details[1][3].endswith("(sum 43)")
    assert details[1][5].startswith(f"Exact orders /{_MKSP_MAX_ORDER}: ")
    assert details[7][6].startswith("Valid-draw beam width ")


def test_mksp_uses_similar_historical_contexts_for_full_distributions() -> None:
    state = _StrategyState(("mksp",))
    assert state._mksp_normalize((0, 0)) == (0.5, 0.5)
    repeating_draws = (
        {1, 10, 20, 30, 40, 49},
        {2, 11, 21, 31, 41, 48},
        {3, 12, 22, 32, 42, 47},
    )
    for index in range(30):
        drawn = repeating_draws[index % len(repeating_draws)]
        state.train(drawn)
        state.remember(drawn)

    (
        distributions,
        anchor_distribution,
        effective_support,
        analogue_count,
        selected_orders,
        selected_supports,
    ) = state._mksp_distributions()
    beam = state._mksp_internal_beam(distributions)

    assert len(distributions) == 6
    assert all(len(distribution) == _MKSP_VALUE_COUNT for distribution in distributions)
    assert all(sum(distribution) == pytest.approx(1) for distribution in distributions)
    assert sum(anchor_distribution) == pytest.approx(1)
    assert effective_support > 0
    assert analogue_count == 29
    assert len(selected_orders) == len(selected_supports) == 6
    assert set(beam) == set(range(_MKSP_VALUE_COUNT))
    assert all(
        len(spaces) == 5 and sum(spaces) == total
        for total, paths in beam.items()
        for _log_probability, spaces in paths
    )


def test_mksp_keeps_twenty_states_and_does_not_learn_from_future_draws() -> None:
    alternating_draws = (
        {1, 10, 20, 30, 40, 49},
        {2, 11, 21, 31, 41, 48},
    )
    state = _StrategyState(("mksp",))
    for index in range(25):
        drawn = alternating_draws[index % 2]
        state.train(drawn)
        state.remember(drawn)
    assert all(len(history) == _MKSP_MAX_ORDER for history in state.mksp_histories)

    prefix = (
        Draw(1, 2, 8, 17, 31, 49),
        Draw(3, 6, 12, 22, 36, 47),
        Draw(1, 9, 18, 27, 38, 45),
    )
    without_future = Draws()
    with_future = Draws()
    for draw in prefix:
        without_future.add(draw)
        with_future.add(Draw(*(ball.value for ball in draw.balls)))
    with_future.add(Draw(4, 11, 20, 29, 37, 48))
    without_future.prepare_predictions()
    with_future.prepare_predictions()

    prefix_strategy = build_prediction_suites(
        without_future.draws,
        enabled_strategy_ids=("mksp",),
    )[-1].strategies[0]
    future_strategy = build_prediction_suites(
        with_future.draws,
        enabled_strategy_ids=("mksp",),
    )[-2].strategies[0]

    assert prefix_strategy.top_numbers == future_strategy.top_numbers
    assert prefix_strategy.numbers == future_strategy.numbers


@pytest.mark.parametrize(
    "numbers",
    (
        {2, 13, 16, 18, 19, 22},
        {1, 2, 3, 4, 5, 6},
        {1, 10, 20, 30, 40, 49},
    ),
)
def test_mknp_normalizes_positions_and_preserves_spread(
    numbers: set[int],
) -> None:
    positions = _normalized_positions_for_numbers(numbers)
    spaces = _spaces_for_numbers(numbers)

    assert positions[0] == 1
    assert all(left < right for left, right in zip(positions, positions[1:]))
    assert positions[-1] == 6 + sum(spaces[1:])
    assert positions[-1] == 49 - spaces[0]


def test_mknp_observes_normalized_positions_and_context_orders() -> None:
    state = _StrategyState(("mknp",))
    draws = (
        {1, 10, 20, 30, 40, 49},
        {2, 11, 21, 31, 41, 48},
        {1, 2, 3, 4, 5, 6},
    )
    for drawn in draws:
        state.train(drawn)
        state.remember(drawn)

    assert _normalized_positions_for_numbers(
        {2, 13, 16, 18, 19, 22}
    ) == (1, 12, 15, 17, 18, 21)
    assert [list(history) for history in state.mknp_histories] == [
        [10, 10, 2],
        [20, 20, 3],
        [30, 30, 4],
        [40, 40, 5],
        [49, 47, 6],
    ]
    assert state.mknp_transitions[0][0][(10,)][10] == 1
    assert state.mknp_transitions[0][0][(10,)][2] == 1
    assert state.mknp_transitions[0][1][(10, 10)][2] == 1
    assert sum(sum(counts) for counts in state.mknp_value_counts) == 15
    with pytest.raises(ValueError, match="exactly six"):
        _normalized_positions_for_numbers({1, 2, 3})


def test_mknp_backs_off_and_uses_categorical_bayesian_smoothing() -> None:
    state = _StrategyState(("mknp",))
    state.draw_count = 100
    state.mknp_value_counts[0][5] = 20
    state.mknp_histories[0].extend((3, 4))
    state.mknp_transitions[0][0][(4,)] = {5: 2, 6: 8}
    state.mknp_transitions[0][1][(3, 4)] = {
        5: 1,
        6: _MKSP_MIN_CONTEXT_SUPPORT - 2,
    }
    valid_count = len(state._mknp_valid_values(0))
    baseline = (
        20 + _MKSP_PRIOR_STRENGTH / valid_count
    ) / (100 + _MKSP_PRIOR_STRENGTH)
    expected_order_one = (2 + _MKSP_PRIOR_STRENGTH * baseline) / (
        10 + _MKSP_PRIOR_STRENGTH
    )

    probability, support, selected_order = state._mknp_probability(0, 5)

    assert probability == pytest.approx(expected_order_one)
    assert support == 10
    assert selected_order == 1


def test_mknp_builds_valid_shapes_and_scores_translated_draws() -> None:
    state = _StrategyState(("mknp",))
    repeating_draws = (
        {1, 10, 20, 30, 40, 49},
        {2, 11, 21, 31, 41, 48},
        {3, 12, 22, 32, 42, 47},
    )
    for index in range(30):
        drawn = repeating_draws[index % len(repeating_draws)]
        state.train(drawn)
        state.remember(drawn)

    (
        distributions,
        anchor_distribution,
        effective_support,
        analogue_count,
        selected_orders,
        selected_supports,
    ) = state._mknp_distributions()
    beam = state._mknp_shape_beam(distributions)
    scores, details = state._mknp_scores()

    assert len(distributions) == 5
    assert all(len(distribution) == 50 for distribution in distributions)
    assert all(sum(distribution) == pytest.approx(1) for distribution in distributions)
    assert sum(anchor_distribution) == pytest.approx(1)
    assert effective_support > 0
    assert analogue_count == 29
    assert len(selected_orders) == len(selected_supports) == 5
    assert set(beam) == set(range(6, 50))
    assert all(
        len(positions) == 6
        and positions[0] == 1
        and positions[-1] == spread
        and all(left < right for left, right in zip(positions, positions[1:]))
        for spread, paths in beam.items()
        for _log_probability, positions in paths
    )
    assert set(scores) == set(range(1, 50))
    assert details[1][0].startswith("Marginal probability ")
    assert details[1][3].startswith("Best normalized positions 1,")
    assert details[1][4].startswith("Spread ")
    assert details[49][5].startswith("First-number anchor ")
    assert details[7][8].startswith("Valid-shape beam width ")


def test_mknp_keeps_twenty_states_and_does_not_learn_from_future_draws() -> None:
    state = _StrategyState(("mknp",))
    for index in range(25):
        drawn = (
            {1, 10, 20, 30, 40, 49}
            if index % 2 == 0
            else {2, 11, 21, 31, 41, 48}
        )
        state.train(drawn)
        state.remember(drawn)
    assert all(len(history) == _MKSP_MAX_ORDER for history in state.mknp_histories)

    prefix = (
        Draw(1, 2, 8, 17, 31, 49),
        Draw(3, 6, 12, 22, 36, 47),
        Draw(1, 9, 18, 27, 38, 45),
    )
    without_future = Draws()
    with_future = Draws()
    for draw in prefix:
        without_future.add(draw)
        with_future.add(Draw(*(ball.value for ball in draw.balls)))
    with_future.add(Draw(4, 11, 20, 29, 37, 48))
    without_future.prepare_predictions()
    with_future.prepare_predictions()

    prefix_strategy = build_prediction_suites(
        without_future.draws,
        enabled_strategy_ids=("mknp",),
    )[-1].strategies[0]
    future_strategy = build_prediction_suites(
        with_future.draws,
        enabled_strategy_ids=("mknp",),
    )[-2].strategies[0]

    assert prefix_strategy.top_numbers == future_strategy.top_numbers
    assert prefix_strategy.numbers == future_strategy.numbers


def test_mkrd_profiles_translation_scale_and_dispersion() -> None:
    compact = _relative_dispersion_profile({2, 4, 6, 8, 10, 12})
    translated = _relative_dispersion_profile({12, 14, 16, 18, 20, 22})
    wide = _relative_dispersion_profile({2, 6, 10, 14, 18, 22})

    assert compact.relative_positions == pytest.approx(
        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    )
    assert compact.relative_positions == pytest.approx(
        translated.relative_positions
    )
    assert compact.relative_positions == pytest.approx(wide.relative_positions)
    assert compact.gap_shares == pytest.approx((0.2,) * 5)
    assert compact.uniformity == pytest.approx(0)
    assert compact.entropy == pytest.approx(1)
    assert compact.center_balance == pytest.approx(0.5)
    assert compact.span == translated.span == 11
    assert wide.span == 21
    assert compact.coverage == pytest.approx(11 / 49)

    assert _StrategyState._mkrd_profile_distance(
        compact,
        translated,
    ) == pytest.approx(0)
    assert _StrategyState._mkrd_profile_distance(
        compact,
        wide,
    ) == pytest.approx(0.20 * 10 / 49)

    uneven = _relative_dispersion_profile({1, 2, 3, 4, 5, 49})
    assert 0 < uneven.uniformity <= 1
    assert 0 <= uneven.entropy < 1
    assert 0 <= uneven.center_balance <= 1
    with pytest.raises(ValueError, match="exactly six"):
        _relative_dispersion_profile({1, 2, 3})
    with pytest.raises(ValueError, match="strictly increasing"):
        _relative_dispersion_profile([1, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="within 1–49"):
        _relative_dispersion_profile({0, 1, 2, 3, 4, 5})


def test_mkrd_observes_independent_position_histories_and_contexts() -> None:
    state = _StrategyState(("mkrd",))
    draws = (
        {1, 10, 20, 30, 40, 49},
        {2, 11, 21, 31, 41, 48},
        {1, 2, 3, 4, 5, 6},
    )
    for drawn in draws:
        state.train(drawn)
        state.remember(drawn)

    assert [list(history) for history in state.mkrd_histories] == [
        [10, 10, 2],
        [20, 20, 3],
        [30, 30, 4],
        [40, 40, 5],
        [49, 47, 6],
    ]
    assert state.mkrd_transitions[0][0][(10,)][10] == 1
    assert state.mkrd_transitions[0][0][(10,)][2] == 1
    assert state.mkrd_transitions[0][1][(10, 10)][2] == 1
    assert sum(sum(counts) for counts in state.mkrd_value_counts) == 15
    assert len(state.mkrd_observations) == 3
    assert not state.mknp_histories
    assert not state.mknp_observations


def test_mkrd_backs_off_with_bayesian_smoothing() -> None:
    state = _StrategyState(("mkrd",))
    state.draw_count = 100
    state.mkrd_value_counts[0][5] = 20
    state.mkrd_histories[0].extend((3, 4))
    state.mkrd_transitions[0][0][(4,)] = {5: 2, 6: 8}
    state.mkrd_transitions[0][1][(3, 4)] = {
        5: 1,
        6: _MKSP_MIN_CONTEXT_SUPPORT - 2,
    }
    valid_count = len(state._mknp_valid_values(0))
    baseline = (
        20 + _MKSP_PRIOR_STRENGTH / valid_count
    ) / (100 + _MKSP_PRIOR_STRENGTH)
    expected = (2 + _MKSP_PRIOR_STRENGTH * baseline) / (
        10 + _MKSP_PRIOR_STRENGTH
    )

    probability, support, selected_order = state._mkrd_probability(0, 5)

    assert probability == pytest.approx(expected)
    assert support == 10
    assert selected_order == 1


def test_mkrd_builds_valid_shapes_and_detailed_scores() -> None:
    state = _StrategyState(("mkrd",))
    repeating_draws = (
        {1, 10, 20, 30, 40, 49},
        {2, 11, 21, 31, 41, 48},
        {3, 12, 22, 32, 42, 47},
    )
    for index in range(30):
        drawn = repeating_draws[index % len(repeating_draws)]
        state.train(drawn)
        state.remember(drawn)

    (
        distributions,
        anchor_distribution,
        effective_support,
        analogue_count,
        selected_orders,
        selected_supports,
    ) = state._mkrd_distributions()
    beam = state._mknp_shape_beam(distributions)
    scores, details = state._mkrd_scores()

    assert len(distributions) == 5
    assert all(sum(distribution) == pytest.approx(1) for distribution in distributions)
    assert sum(anchor_distribution) == pytest.approx(1)
    assert effective_support > 0
    assert analogue_count == 29
    assert len(selected_orders) == len(selected_supports) == 5
    assert all(
        len(positions) == 6
        and positions[0] == 1
        and all(left < right for left, right in zip(positions, positions[1:]))
        for paths in beam.values()
        for _log_probability, positions in paths
    )
    assert set(scores) == set(range(1, 50))
    assert details[1][3].startswith("Relative positions 0.000,")
    assert details[1][4].startswith("Span ")
    assert details[1][5].startswith("Gap shares ")
    assert details[1][6].startswith("Uniformity deviation ")
    assert details[1][7].startswith("Internal-gap entropy ")
    assert details[1][8].startswith("Center balance ")
    assert details[7][12].startswith("Valid-shape beam width ")


def test_mkrd_keeps_twenty_states_and_does_not_learn_from_future_draws() -> None:
    state = _StrategyState(("mkrd",))
    for index in range(25):
        drawn = (
            {1, 10, 20, 30, 40, 49}
            if index % 2 == 0
            else {2, 11, 21, 31, 41, 48}
        )
        state.train(drawn)
        state.remember(drawn)
    assert all(len(history) == _MKSP_MAX_ORDER for history in state.mkrd_histories)

    prefix = (
        Draw(1, 2, 8, 17, 31, 49),
        Draw(3, 6, 12, 22, 36, 47),
        Draw(1, 9, 18, 27, 38, 45),
    )
    without_future = Draws()
    with_future = Draws()
    for draw in prefix:
        without_future.add(draw)
        with_future.add(Draw(*(ball.value for ball in draw.balls)))
    with_future.add(Draw(4, 11, 20, 29, 37, 48))
    without_future.prepare_predictions()
    with_future.prepare_predictions()

    prefix_strategy = build_prediction_suites(
        without_future.draws,
        enabled_strategy_ids=("mkrd",),
    )[-1].strategies[0]
    future_strategy = build_prediction_suites(
        with_future.draws,
        enabled_strategy_ids=("mkrd",),
    )[-2].strategies[0]

    assert prefix_strategy.top_numbers == future_strategy.top_numbers
    assert prefix_strategy.numbers == future_strategy.numbers


@pytest.mark.parametrize(
    ("distance", "bucket"),
    ((1, 0), (2, 1), (4, 2), (7, 3), (12, 4), (20, 5)),
)
def test_maps_every_proximity_distance_band(distance: int, bucket: int) -> None:
    assert _proximity_bucket(distance) == bucket


@pytest.mark.parametrize(
    ("distance", "label"),
    (
        (0, "Overlap"),
        (2, "Near"),
        (4, "Close"),
        (7, "Middle"),
        (10, "Far"),
        (14, "Distant"),
    ),
)
def test_maps_every_earth_mover_distance_band(
    distance: float,
    label: str,
) -> None:
    assert _StrategyState._earth_mover_bucket(distance) == label


def test_handles_empty_earth_mover_history_and_vectors() -> None:
    state = _StrategyState()

    scores, details = state._earth_mover_scores()

    assert scores == {number: 0 for number in range(1, 50)}
    assert details == {}
    assert state._earth_mover_distance((), ()) == 0


def test_defensive_missing_ranking_and_unprepared_draw_errors() -> None:
    state = _StrategyState()
    state.prior_rankings["freshness"] = []
    assert state._rank_score("freshness", 49) == 0
    neutral_scores, neutral_details = state._chained_scores(
        {},
        {number: 0 for number in range(1, 50)},
    )
    assert neutral_scores == {number: 0 for number in range(1, 50)}
    assert neutral_details == {}

    with pytest.raises(ValueError, match="must be prepared"):
        build_prediction_suites((Draw(),))
