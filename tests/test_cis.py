"""Test champion-preserving CIS v3 independently of its promotion result."""

from __future__ import annotations

from collections import deque

import pytest

from rand_ai.cis import (
    EXPECTED_RANDOM_HITS,
    SHADOW_MINIMUM_DRAWS,
    ChampionCis,
    CisConfig,
    _PendingPrediction,
    cis_configurations,
    ranking_correlations,
    spearman_rank_correlation,
)


def _config(**changes: int | float) -> CisConfig:
    values: dict[str, int | float] = {
        "recent_window": 40,
        "recent_weight": 0.8,
        "correlation_threshold": 0.8,
        "minimum_support": 0.2,
        "minimum_gain": 0.05,
        "maximum_replacements": 2,
    }
    values.update(changes)
    return CisConfig(**values)  # type: ignore[arg-type]


def _ranking(first: int = 1) -> tuple[int, ...]:
    numbers = list(range(1, 50))
    numbers.remove(first)
    return (first, *numbers)


def _pending(*, corrected: bool = True) -> _PendingPrediction:
    return _PendingPrediction(
        rankings={"freshness": tuple(range(1, 50))},
        champion_top=frozenset(range(1, 7)),
        shadow_top=(
            frozenset({1, 2, 3, 4, 5, 7})
            if corrected
            else frozenset(range(1, 7))
        ),
        has_shadow_correction=corrected,
    )


def test_configuration_grid_and_rank_correlations_are_deterministic() -> None:
    configurations = cis_configurations()
    ascending = tuple(range(1, 50))
    descending = tuple(reversed(ascending))

    assert len(configurations) == 96
    assert configurations[0] == _config(
        recent_window=20,
        recent_weight=0.6,
        maximum_replacements=1,
    )
    assert configurations[-1] == _config(
        recent_window=80,
        correlation_threshold=0.9,
        minimum_support=0.3,
        minimum_gain=0.10,
    )
    assert spearman_rank_correlation(ascending, ascending) == 1
    assert spearman_rank_correlation(ascending, descending) == -1
    assert ranking_correlations(
        {
            "freshness": ascending,
            "proximity": descending,
            "not_an_expert": ascending,
        }
    ) == {("freshness", "proximity"): -1}


def test_champion_uses_smoothed_past_hits_and_canonical_ties() -> None:
    engine = ChampionCis(_config())
    rankings = {
        "freshness": _ranking(1),
        "proximity": _ranking(49),
    }

    assert engine._champion(rankings) == "freshness"
    with pytest.raises(ValueError, match="at least one expert"):
        engine._champion({})

    engine.total_hits["proximity"] = 20
    engine.evaluated_draws["proximity"] = 10
    engine.recent_hits["proximity"].extend([2] * 10)

    assert engine.expert_quality("proximity") > EXPECTED_RANDOM_HITS
    assert engine._champion(rankings) == "proximity"


def test_peer_weight_penalizes_redundancy_and_ignores_unproven_peers() -> None:
    engine = ChampionCis(_config())
    rankings = {
        "freshness": _ranking(1),
        "emd": _ranking(2),
        "chi_square": _ranking(3),
    }

    scores, support = engine._peer_evidence(rankings, "freshness")
    assert set(scores.values()) == {0.0}
    assert set(support.values()) == {0.0}
    assert engine._mean_correlation("emd", "chi_square") == 0

    for strategy_id in ("emd", "chi_square"):
        engine.total_hits[strategy_id] = 20
        engine.evaluated_draws[strategy_id] = 10
        engine.recent_hits[strategy_id].extend([2] * 10)
    independent = engine._peer_weights(rankings, "freshness")
    engine.correlations[("emd", "chi_square")].append(1.0)
    redundant = engine._peer_weights(rankings, "freshness")

    assert redundant["emd"] < independent["emd"]
    assert redundant["chi_square"] < independent["chi_square"]
    assert engine._mean_correlation("chi_square", "emd") == 1
    scores, support = engine._peer_evidence(rankings, "freshness")
    assert scores[2] > scores[49]
    assert support[2] > support[49]


def test_shadow_ticket_applies_no_more_than_configured_supported_swaps() -> None:
    engine = ChampionCis(_config())
    ranking = tuple(range(1, 50))
    peer_scores = {number: 0.5 for number in ranking}
    peer_support = {number: 0.0 for number in ranking}
    peer_scores.update({5: 0.0, 6: 0.1, 7: 1.0, 8: 0.9})
    peer_support.update({7: 0.8, 8: 0.7})

    shadow, corrections = engine._shadow_ticket(
        ranking,
        peer_scores,
        peer_support,
    )

    assert shadow == (1, 2, 3, 4, 7, 8)
    assert [(item.removed, item.added) for item in corrections] == [(5, 7), (6, 8)]
    blocked_shadow, blocked = engine._shadow_ticket(
        ranking,
        peer_scores,
        {number: 0.0 for number in ranking},
    )
    assert blocked_shadow == ranking[:6]
    assert blocked == ()
    short_shadow, short_corrections = engine._shadow_ticket(
        ranking[:6],
        peer_scores,
        peer_support,
    )
    assert short_shadow == ranking[:6]
    assert short_corrections == ()


def test_prediction_falls_back_to_champion_until_shadow_gate_activates() -> None:
    engine = ChampionCis(_config(maximum_replacements=1))
    rankings = {
        "freshness": tuple(range(1, 50)),
        "proximity": (7, *range(1, 7), *range(8, 50)),
    }
    engine.total_hits["freshness"] = 30
    engine.evaluated_draws["freshness"] = 10
    engine.recent_hits["freshness"].extend([3] * 10)
    engine.total_hits["proximity"] = 20
    engine.evaluated_draws["proximity"] = 10
    engine.recent_hits["proximity"].extend([2] * 10)

    initial = engine.predict(rankings)
    assert initial.champion_id == "freshness"
    assert initial.top_numbers == initial.champion_top
    assert initial.applied_correction_count == 0

    engine.corrections_active = True
    corrected = engine.predict(rankings)
    assert corrected.top_numbers == corrected.shadow_top
    assert corrected.applied_correction_count == len(corrected.corrections)
    assert len(corrected.ranking) == 49
    assert len(set(corrected.ranking)) == 49


def test_observe_updates_only_pending_history_and_gates_shadow_corrections() -> None:
    engine = ChampionCis(_config())
    engine.observe({7})
    assert engine.evaluated_draws["freshness"] == 0

    engine.pending = _pending(corrected=False)
    engine.observe({1}, {})
    assert engine.total_hits["freshness"] == 1
    assert engine.evaluated_draws["freshness"] == 1
    assert engine.pending is None
    assert not engine.shadow_deltas

    engine.shadow_deltas = deque(
        [0] * (SHADOW_MINIMUM_DRAWS - 3) + [1, 1],
        maxlen=120,
    )
    engine.pending = _pending()
    engine.observe({7}, {("freshness", "proximity"): 0.5})
    assert engine.corrections_active
    assert sum(engine.shadow_deltas) == 3

    engine.shadow_deltas = deque([0] * 119, maxlen=120)
    engine.pending = _pending()
    engine.observe({1})
    assert not engine.corrections_active
