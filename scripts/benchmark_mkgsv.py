"""Benchmark guarded MKGSV v2 against its Markov 100 champion."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from rand_ai import load_lotto_results_yaml
from rand_ai.draw import Draw
from rand_ai.mkgsv import (
    BASE_HIT_RATE,
    NUMBER_COUNT,
    NUMBERS_PER_DRAW,
    MkgsvConfig,
    MkgsvModel,
    mkgsv_configurations,
)
from rand_ai.strategy_prediction import build_prediction_suites

WARM_UP_DRAWS = 320
VALIDATION_DRAWS = 200
HOLDOUT_DRAWS = 250
RELATED_STRATEGY_IDS = ("markov100", "freshness", "proximity")
MARKOV_PRIOR_STRENGTH = 8.0
MARKOV_DECAY = 0.5 ** (1 / 500)
MAX_GAP_BUCKET = 35


@dataclass(frozen=True, slots=True)
class ConfigRun:
    config: MkgsvConfig
    gated_hits: tuple[int, ...]
    shadow_hits: tuple[int, ...]
    base_hits: tuple[int, ...]
    gated_brier_scores: tuple[float, ...]
    shadow_brier_scores: tuple[float, ...]
    base_brier_scores: tuple[float, ...]
    champion_overlaps: tuple[int, ...]
    proposals: tuple[bool, ...]
    active_predictions: tuple[bool, ...]
    replacements: tuple[str | None, ...]
    model: MkgsvModel


class Markov100Anchor:
    """Small benchmark mirror of the production Markov 100 state."""

    def __init__(self) -> None:
        self.draw_count = 0
        self.last_seen: list[int | None] = [None] * (NUMBER_COUNT + 1)
        self.opportunities = [0.0] * (MAX_GAP_BUCKET + 1)
        self.hits = [0.0] * (MAX_GAP_BUCKET + 1)

    def gap(self, number: int) -> int:
        seen_at = self.last_seen[number]
        return self.draw_count if seen_at is None else self.draw_count - seen_at - 1

    def train(self, drawn: set[int]) -> None:
        if self.draw_count == 0:
            return
        for bucket in range(MAX_GAP_BUCKET + 1):
            self.opportunities[bucket] *= MARKOV_DECAY
            self.hits[bucket] *= MARKOV_DECAY
        for number in range(1, NUMBER_COUNT + 1):
            bucket = min(self.gap(number), MAX_GAP_BUCKET)
            self.opportunities[bucket] += 1
            if number in drawn:
                self.hits[bucket] += 1

    def remember(self, drawn: set[int]) -> None:
        for number in drawn:
            self.last_seen[number] = self.draw_count
        self.draw_count += 1

    def probabilities(self) -> dict[int, float]:
        bucket_probabilities = [
            (hits + MARKOV_PRIOR_STRENGTH * BASE_HIT_RATE)
            / (opportunities + MARKOV_PRIOR_STRENGTH)
            for hits, opportunities in zip(self.hits, self.opportunities)
        ]
        return {
            number: bucket_probabilities[min(self.gap(number), MAX_GAP_BUCKET)]
            for number in range(1, NUMBER_COUNT + 1)
        }

    def ranking(self, probabilities: dict[int, float]) -> tuple[int, ...]:
        return tuple(
            sorted(
                probabilities,
                key=lambda number: (
                    -probabilities[number],
                    -self.gap(number),
                    number,
                ),
            )
        )


def hit_summary(hits: Sequence[int]) -> dict[str, int | float]:
    distribution = Counter(hits)
    return {
        "evaluatedDraws": len(hits),
        "totalHits": sum(hits),
        "averageHitsPerDraw": mean(hits) if hits else 0.0,
        "zeroHits": distribution[0],
        "oneHit": distribution[1],
        "twoOrMoreHits": sum(
            count for value, count in distribution.items() if value >= 2
        ),
    }


def paired_summary(candidate: Sequence[int], baseline: Sequence[int]) -> dict[str, Any]:
    differences = [left - right for left, right in zip(candidate, baseline)]
    average = mean(differences) if differences else 0.0
    standard_error = (
        stdev(differences) / len(differences) ** 0.5
        if len(differences) > 1
        else 0.0
    )
    return {
        "candidateWins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "baselineWins": sum(value < 0 for value in differences),
        "meanHitDifference": average,
        "meanDifference95Interval": [
            average - 1.96 * standard_error,
            average + 1.96 * standard_error,
        ],
    }


def scope_slice[T](values: Sequence[T], scope: str) -> Sequence[T]:
    holdout_start = WARM_UP_DRAWS + VALIDATION_DRAWS
    if scope == "validation":
        return values[WARM_UP_DRAWS:holdout_start]
    if scope == "holdout":
        return values[holdout_start:]
    raise ValueError(f"Unknown benchmark scope: {scope}")


def _brier(probabilities: dict[int, float], actual: set[int]) -> float:
    return sum(
        (probabilities[number] - float(number in actual)) ** 2
        for number in range(1, NUMBER_COUNT + 1)
    ) / NUMBER_COUNT


def run_configuration(draws: Sequence[Any], config: MkgsvConfig) -> ConfigRun:
    model = MkgsvModel(config)
    anchor = Markov100Anchor()
    gated_hits: list[int] = []
    shadow_hits: list[int] = []
    base_hits: list[int] = []
    gated_brier_scores: list[float] = []
    shadow_brier_scores: list[float] = []
    base_brier_scores: list[float] = []
    champion_overlaps: list[int] = []
    proposals: list[bool] = []
    active_predictions: list[bool] = []
    replacements: list[str | None] = []
    for index, draw in enumerate(draws):
        drawn = {ball.value for ball in draw.balls}
        model.train(drawn)
        anchor.train(drawn)
        model.remember(drawn)
        anchor.remember(drawn)
        probabilities = anchor.probabilities()
        base_ranking = anchor.ranking(probabilities)
        decision = model.predict(probabilities, base_ranking)
        if index + 1 >= len(draws):
            continue
        actual = {ball.value for ball in draws[index + 1].balls}
        base_ticket = set(decision.base_ticket)
        shadow_ticket = set(decision.shadow_ticket)
        output_ticket = set(decision.output_ticket)
        base_hits.append(len(actual.intersection(base_ticket)))
        shadow_hits.append(len(actual.intersection(shadow_ticket)))
        gated_hits.append(len(actual.intersection(output_ticket)))
        champion_overlaps.append(len(base_ticket.intersection(output_ticket)))
        proposed = decision.proposed_outsider is not None
        proposals.append(proposed)
        active_predictions.append(decision.correction_active)
        replacements.append(
            None
            if not proposed
            else f"{decision.proposed_insider}->{decision.proposed_outsider}"
        )
        corrected_probabilities = {
            number: row.corrected_probability
            for number, row in decision.scores.items()
        }
        gated_probabilities = (
            corrected_probabilities
            if decision.correction_active
            else probabilities
        )
        base_brier_scores.append(_brier(probabilities, actual))
        shadow_brier_scores.append(_brier(corrected_probabilities, actual))
        gated_brier_scores.append(_brier(gated_probabilities, actual))
    return ConfigRun(
        config=config,
        gated_hits=tuple(gated_hits),
        shadow_hits=tuple(shadow_hits),
        base_hits=tuple(base_hits),
        gated_brier_scores=tuple(gated_brier_scores),
        shadow_brier_scores=tuple(shadow_brier_scores),
        base_brier_scores=tuple(base_brier_scores),
        champion_overlaps=tuple(champion_overlaps),
        proposals=tuple(proposals),
        active_predictions=tuple(active_predictions),
        replacements=tuple(replacements),
        model=model,
    )


def combined_top_numbers(draw: Draw) -> tuple[int, ...]:
    prediction = draw.prediction
    return () if prediction is None else prediction.top_numbers


def select_configuration(runs: Sequence[ConfigRun]) -> ConfigRun:
    """Select gated validation hits with conservative deterministic ties."""
    return max(
        enumerate(runs),
        key=lambda item: (
            sum(scope_slice(item[1].gated_hits, "validation")),
            -mean(scope_slice(item[1].gated_brier_scores, "validation")),
            -sum(scope_slice(item[1].proposals, "validation")),
            sum(scope_slice(item[1].champion_overlaps, "validation")),
            (
                item[1].config.single_strength
                + item[1].config.pair_strength
                + item[1].config.triple_strength
            ),
            -item[0],
        ),
    )[1]


def passes_promotion(
    validation_candidate: int,
    validation_markov: int,
    holdout_candidate: int,
    holdout_markov: int,
    validation_gain: int,
    holdout_gain: int,
) -> bool:
    """Apply the champion-improvement promotion gate."""
    return (
        validation_candidate > validation_markov
        and holdout_candidate >= holdout_markov
        and validation_gain > 0
        and holdout_gain >= 0
    )


def _scope_report(run: ConfigRun, scope: str) -> dict[str, Any]:
    gated_hits = scope_slice(run.gated_hits, scope)
    shadow_hits = scope_slice(run.shadow_hits, scope)
    base_hits = scope_slice(run.base_hits, scope)
    proposals = scope_slice(run.proposals, scope)
    active = scope_slice(run.active_predictions, scope)
    replacements = scope_slice(run.replacements, scope)
    return {
        "mkgsvGated": {
            **hit_summary(gated_hits),
            "brierScore": mean(scope_slice(run.gated_brier_scores, scope)),
        },
        "mkgsvRawShadow": {
            **hit_summary(shadow_hits),
            "brierScore": mean(scope_slice(run.shadow_brier_scores, scope)),
        },
        "markov100Champion": {
            **hit_summary(base_hits),
            "brierScore": mean(scope_slice(run.base_brier_scores, scope)),
        },
        "pairedGatedVsMarkov": paired_summary(gated_hits, base_hits),
        "pairedShadowVsMarkov": paired_summary(shadow_hits, base_hits),
        "gatedCorrectionNetGain": sum(gated_hits) - sum(base_hits),
        "rawShadowNetGain": sum(shadow_hits) - sum(base_hits),
        "proposalCount": sum(proposals),
        "activePredictionCount": sum(active),
        "meanChampionOverlap": mean(scope_slice(run.champion_overlaps, scope)),
        "replacementCounts": dict(
            sorted(Counter(value for value in replacements if value).items())
        ),
        "constantProbabilityBrier": BASE_HIT_RATE * (1 - BASE_HIT_RATE),
        "theoreticalRandomHits": len(gated_hits) * NUMBERS_PER_DRAW * BASE_HIT_RATE,
    }


def benchmark(dataset_path: Path) -> dict[str, Any]:
    draws = load_lotto_results_yaml(dataset_path)
    draws.prepare_predictions()
    evaluated_count = len(draws.draws) - 1
    expected = WARM_UP_DRAWS + VALIDATION_DRAWS + HOLDOUT_DRAWS
    if evaluated_count != expected:
        raise ValueError(
            f"MKGSV benchmark requires {expected} evaluated draws; "
            f"received {evaluated_count}"
        )
    runs = tuple(
        run_configuration(draws.draws, config)
        for config in mkgsv_configurations()
    )
    selected = select_configuration(runs)

    suites = build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=RELATED_STRATEGY_IDS,
    )[:-1]
    related_hits: dict[str, tuple[int, ...]] = {}
    for strategy_id in RELATED_STRATEGY_IDS:
        related_hits[strategy_id] = tuple(
            len(
                set(suite.actual_numbers).intersection(
                    next(
                        strategy.top_numbers
                        for strategy in suite.strategies
                        if strategy.strategy_id == strategy_id
                    )
                )
            )
            for suite in suites
        )
    related_hits["combined"] = tuple(
        len(
            set(suite.actual_numbers).intersection(
                combined_top_numbers(draws.draws[index])
            )
        )
        for index, suite in enumerate(suites)
    )
    if selected.base_hits != related_hits["markov100"]:
        raise AssertionError("Benchmark Markov anchor diverged from production Markov 100")

    scopes: dict[str, Any] = {}
    for scope in ("validation", "holdout"):
        scopes[scope] = {
            **_scope_report(selected, scope),
            "relatedBaselines": {
                strategy_id: hit_summary(scope_slice(hits, scope))
                for strategy_id, hits in related_hits.items()
            },
        }
    validation = scopes["validation"]
    holdout = scopes["holdout"]
    passed = passes_promotion(
        validation["mkgsvGated"]["totalHits"],
        validation["markov100Champion"]["totalHits"],
        holdout["mkgsvGated"]["totalHits"],
        holdout["markov100Champion"]["totalHits"],
        validation["gatedCorrectionNetGain"],
        holdout["gatedCorrectionNetGain"],
    )
    return {
        "schemaVersion": 2,
        "dataset": dataset_path.as_posix(),
        "evaluatedDraws": evaluated_count,
        "split": {
            "warmUp": WARM_UP_DRAWS,
            "validation": VALIDATION_DRAWS,
            "holdout": HOLDOUT_DRAWS,
        },
        "selectedConfig": asdict(selected.config),
        "stateSupport": selected.model.state_support_distribution(),
        "lifetimeShadow": {
            "settledResults": selected.model.shadow_results,
            "proposalCount": selected.model.proposal_count,
            "activationCount": selected.model.activation_count,
            "trailingGain": sum(selected.model.shadow_deltas),
        },
        "scopes": scopes,
        "promotion": {
            "passed": passed,
            "defaultEnabled": passed,
            "rule": (
                "Gated MKGSV must beat Markov 100 on validation, be no worse "
                "on holdout, and have positive/nonnegative correction gains."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the stable human-readable benchmark summary."""
    config = report["selectedConfig"]
    validation = report["scopes"]["validation"]
    holdout = report["scopes"]["holdout"]
    promotion = report["promotion"]
    lines = [
        "# MKGSV v2 promotion report",
        "",
        "Leakage-safe 320/200/250 warm-up, validation, and holdout benchmark.",
        "",
        "## Selected configuration",
        "",
        (
            f"Singles `{config['single_strength']:g}`, pairs "
            f"`{config['pair_strength']:g}`, triples "
            f"`{config['triple_strength']:g}`, "
            f"`{config['evidence_variant']}` evidence, replacement margin "
            f"`{config['replacement_margin']:.4f}`."
        ),
        "",
        "## Results",
        "",
        "| Scope | Gated MKGSV | Raw shadow | Markov 100 | Gated gain | Proposals | Active |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, scope in (("Validation", validation), ("Holdout", holdout)):
        lines.append(
            f"| {label} | {scope['mkgsvGated']['totalHits']} | "
            f"{scope['mkgsvRawShadow']['totalHits']} | "
            f"{scope['markov100Champion']['totalHits']} | "
            f"{scope['gatedCorrectionNetGain']:+d} | "
            f"{scope['proposalCount']} | {scope['activePredictionCount']} |"
        )
    lines.extend(
        [
            "",
            (
                "Validation gated/Markov Brier: "
                f"`{validation['mkgsvGated']['brierScore']:.6f}` / "
                f"`{validation['markov100Champion']['brierScore']:.6f}`."
            ),
            (
                "Holdout gated/Markov Brier: "
                f"`{holdout['mkgsvGated']['brierScore']:.6f}` / "
                f"`{holdout['markov100Champion']['brierScore']:.6f}`."
            ),
            "",
            "Complete distributions, paired differences, replacement identities, "
            "related baselines, and state support are in the JSON report.",
            "",
            "## Promotion decision",
            "",
            (
                "**Passed.** MKGSV v2 may be enabled by default."
                if promotion["passed"]
                else (
                    "**Failed.** MKGSV remains selectable, experimental, and "
                    "disabled by default."
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/lotto_results_2019.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    options = parser.parse_args()
    report = benchmark(options.dataset)
    rendered = json.dumps(report, indent=2)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered + "\n", encoding="utf-8")
    if options.markdown_output is not None:
        options.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        options.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
