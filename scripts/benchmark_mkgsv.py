"""Benchmark MKGSV v3 ticket motifs against the Markov 100 champion."""

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
MINIMUM_VALIDATION_PROPOSALS = 30
RELATED_STRATEGY_IDS = (
    "markov100",
    "doublet_triplet_markov",
    "mksp",
)
MARKOV_PRIOR_STRENGTH = 8.0
MARKOV_DECAY = 0.5 ** (1 / 500)
MAX_GAP_BUCKET = 35


@dataclass(frozen=True, slots=True)
class ConfigRun:
    config: MkgsvConfig
    gated_hits: tuple[int, ...]
    raw_hits: tuple[int, ...]
    base_hits: tuple[int, ...]
    gated_brier_scores: tuple[float, ...]
    raw_brier_scores: tuple[float, ...]
    base_brier_scores: tuple[float, ...]
    champion_overlaps: tuple[int, ...]
    proposals: tuple[bool, ...]
    active_predictions: tuple[bool, ...]
    replacements: tuple[str | None, ...]
    model: MkgsvModel


class Markov100Anchor:
    """Small benchmark mirror of production Markov 100 state."""

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
            for hits, opportunities in zip(self.hits, self.opportunities, strict=True)
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
    differences = [left - right for left, right in zip(candidate, baseline, strict=True)]
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


def _swapped_probabilities(
    probabilities: dict[int, float],
    insider: int | None,
    outsider: int | None,
) -> dict[int, float]:
    result = dict(probabilities)
    if insider is not None and outsider is not None:
        result[insider], result[outsider] = result[outsider], result[insider]
    return result


def run_configuration(draws: Sequence[Any], config: MkgsvConfig) -> ConfigRun:
    model = MkgsvModel(config, promotion_enabled=True)
    anchor = Markov100Anchor()
    gated_hits: list[int] = []
    raw_hits: list[int] = []
    base_hits: list[int] = []
    gated_brier_scores: list[float] = []
    raw_brier_scores: list[float] = []
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
        raw_ticket = set(decision.shadow_ticket)
        gated_ticket = set(decision.output_ticket)
        base_hits.append(len(actual.intersection(base_ticket)))
        raw_hits.append(len(actual.intersection(raw_ticket)))
        gated_hits.append(len(actual.intersection(gated_ticket)))
        champion_overlaps.append(len(base_ticket.intersection(raw_ticket)))
        proposed = decision.proposed_outsider is not None
        proposals.append(proposed)
        active_predictions.append(decision.correction_active)
        replacements.append(
            None
            if not proposed
            else f"{decision.proposed_insider}->{decision.proposed_outsider}"
        )
        raw_probabilities = _swapped_probabilities(
            probabilities,
            decision.proposed_insider,
            decision.proposed_outsider,
        )
        gated_probabilities = (
            raw_probabilities if decision.correction_active else probabilities
        )
        base_brier_scores.append(_brier(probabilities, actual))
        raw_brier_scores.append(_brier(raw_probabilities, actual))
        gated_brier_scores.append(_brier(gated_probabilities, actual))
    return ConfigRun(
        config=config,
        gated_hits=tuple(gated_hits),
        raw_hits=tuple(raw_hits),
        base_hits=tuple(base_hits),
        gated_brier_scores=tuple(gated_brier_scores),
        raw_brier_scores=tuple(raw_brier_scores),
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


def select_configuration(runs: Sequence[ConfigRun]) -> ConfigRun | None:
    """Reject nonpositive raw gain, then apply deterministic v3 ties."""
    eligible = [
        (index, run)
        for index, run in enumerate(runs)
        if sum(scope_slice(run.raw_hits, "validation"))
        > sum(scope_slice(run.base_hits, "validation"))
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            sum(scope_slice(item[1].raw_hits, "validation")),
            sum(scope_slice(item[1].gated_hits, "validation")),
            -mean(scope_slice(item[1].raw_brier_scores, "validation")),
            -sum(scope_slice(item[1].proposals, "validation")),
            sum(scope_slice(item[1].champion_overlaps, "validation")),
            item[1].config.prior_strength,
            -item[0],
        ),
    )[1]


def passes_promotion(
    validation_raw: int,
    validation_gated: int,
    validation_markov: int,
    holdout_raw: int,
    holdout_gated: int,
    holdout_markov: int,
    validation_gain: int,
    holdout_gain: int,
    validation_proposals: int,
) -> bool:
    """Apply every strict champion-preservation condition."""
    return (
        validation_raw > validation_markov
        and validation_gated > validation_markov
        and holdout_raw >= holdout_markov
        and holdout_gated >= holdout_markov
        and validation_gain > 0
        and holdout_gain >= 0
        and validation_proposals >= MINIMUM_VALIDATION_PROPOSALS
    )


def _scope_report(run: ConfigRun, scope: str) -> dict[str, Any]:
    gated_hits = scope_slice(run.gated_hits, scope)
    raw_hits = scope_slice(run.raw_hits, scope)
    base_hits = scope_slice(run.base_hits, scope)
    proposals = scope_slice(run.proposals, scope)
    active = scope_slice(run.active_predictions, scope)
    replacements = scope_slice(run.replacements, scope)
    return {
        "mkgsvGated": {
            **hit_summary(gated_hits),
            "brierScore": mean(scope_slice(run.gated_brier_scores, scope)),
        },
        "mkgsvRaw": {
            **hit_summary(raw_hits),
            "brierScore": mean(scope_slice(run.raw_brier_scores, scope)),
        },
        "markov100Champion": {
            **hit_summary(base_hits),
            "brierScore": mean(scope_slice(run.base_brier_scores, scope)),
        },
        "pairedGatedVsMarkov": paired_summary(gated_hits, base_hits),
        "pairedRawVsMarkov": paired_summary(raw_hits, base_hits),
        "gatedCorrectionNetGain": sum(gated_hits) - sum(base_hits),
        "rawCorrectionNetGain": sum(raw_hits) - sum(base_hits),
        "proposalCount": sum(proposals),
        "activePredictionCount": sum(active),
        "meanChampionOverlap": mean(scope_slice(run.champion_overlaps, scope)),
        "replacementCounts": dict(
            sorted(Counter(value for value in replacements if value).items())
        ),
        "constantProbabilityBrier": BASE_HIT_RATE * (1 - BASE_HIT_RATE),
        "theoreticalRandomHits": len(gated_hits) * NUMBERS_PER_DRAW * BASE_HIT_RATE,
    }


def _off_run(run: ConfigRun) -> ConfigRun:
    length = len(run.base_hits)
    return ConfigRun(
        config=run.config,
        gated_hits=run.base_hits,
        raw_hits=run.raw_hits,
        base_hits=run.base_hits,
        gated_brier_scores=run.base_brier_scores,
        raw_brier_scores=run.raw_brier_scores,
        base_brier_scores=run.base_brier_scores,
        champion_overlaps=run.champion_overlaps,
        proposals=run.proposals,
        active_predictions=tuple(False for _ in range(length)),
        replacements=run.replacements,
        model=run.model,
    )


def _ablation_report(runs: Sequence[ConfigRun]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    variants = dict.fromkeys(run.config.motif_variant for run in runs)
    for variant in variants:
        matching = [run for run in runs if run.config.motif_variant == variant]
        best = max(
            matching,
            key=lambda run: (
                sum(scope_slice(run.raw_hits, "validation")),
                -mean(scope_slice(run.raw_brier_scores, "validation")),
                run.config.prior_strength,
                -run.config.influence,
            ),
        )
        validation_raw = sum(scope_slice(best.raw_hits, "validation"))
        validation_base = sum(scope_slice(best.base_hits, "validation"))
        result[variant] = {
            "config": asdict(best.config),
            "validationRawHits": validation_raw,
            "validationRawGain": validation_raw - validation_base,
            "validationProposals": sum(scope_slice(best.proposals, "validation")),
        }
    return result


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
        run_configuration(draws.draws, config) for config in mkgsv_configurations()
    )
    research_best = max(
        runs,
        key=lambda run: (
            sum(scope_slice(run.raw_hits, "validation")),
            -mean(scope_slice(run.raw_brier_scores, "validation")),
        ),
    )
    selected = select_configuration(runs)
    evaluated = _off_run(research_best) if selected is None else selected

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
    if evaluated.base_hits != related_hits["markov100"]:
        raise AssertionError("Benchmark Markov anchor diverged from production Markov 100")

    scopes: dict[str, Any] = {}
    for scope in ("validation", "holdout"):
        scopes[scope] = {
            **_scope_report(evaluated, scope),
            "relatedBaselines": {
                strategy_id: hit_summary(scope_slice(hits, scope))
                for strategy_id, hits in related_hits.items()
            },
        }
    validation = scopes["validation"]
    holdout = scopes["holdout"]
    passed = selected is not None and passes_promotion(
        validation["mkgsvRaw"]["totalHits"],
        validation["mkgsvGated"]["totalHits"],
        validation["markov100Champion"]["totalHits"],
        holdout["mkgsvRaw"]["totalHits"],
        holdout["mkgsvGated"]["totalHits"],
        holdout["markov100Champion"]["totalHits"],
        validation["rawCorrectionNetGain"],
        holdout["rawCorrectionNetGain"],
        validation["proposalCount"],
    )
    return {
        "schemaVersion": 3,
        "dataset": dataset_path.as_posix(),
        "evaluatedDraws": evaluated_count,
        "split": {
            "warmUp": WARM_UP_DRAWS,
            "validation": VALIDATION_DRAWS,
            "holdout": HOLDOUT_DRAWS,
        },
        "selectedMode": "off" if selected is None else "ticket-motif",
        "selectedConfig": None if selected is None else asdict(selected.config),
        "bestRejectedConfig": (
            asdict(research_best.config) if selected is None else None
        ),
        "stateSupport": evaluated.model.state_support_distribution(),
        "componentAblations": _ablation_report(runs),
        "lifetimeShadow": {
            "settledProposals": evaluated.model.shadow_results,
            "proposalCount": evaluated.model.proposal_count,
            "activationCount": evaluated.model.activation_count,
            "lifetimeGain": evaluated.model.lifetime_shadow_gain,
            "trailing60Gain": sum(evaluated.model.shadow_deltas_60),
            "trailing120Gain": sum(evaluated.model.shadow_deltas_120),
        },
        "scopes": scopes,
        "promotion": {
            "passed": passed,
            "defaultEnabled": passed,
            "rule": (
                "Raw and gated MKGSV must beat Markov 100 on validation, be "
                "no worse on holdout, retain positive/nonnegative raw gains, "
                "and propose at least 30 validation replacements."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the stable human-readable v3 benchmark summary."""
    config = report["selectedConfig"]
    validation = report["scopes"]["validation"]
    holdout = report["scopes"]["holdout"]
    promotion = report["promotion"]
    config_text = (
        "Correction off; no configuration had positive raw validation gain."
        if config is None
        else (
            f"Prior `{config['prior_strength']:g}`, variant "
            f"`{config['motif_variant']}`, influence "
            f"`{config['influence']:.2f}`."
        )
    )
    lines = [
        "# MKGSV v3 promotion report",
        "",
        "Leakage-safe 320/200/250 warm-up, validation, and holdout benchmark.",
        "",
        "## Selected configuration",
        "",
        config_text,
        "",
        "## Results",
        "",
        "| Scope | Gated MKGSV | Raw motif | Markov 100 | Gated gain | Raw gain | Proposals | Active |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, scope in (("Validation", validation), ("Holdout", holdout)):
        lines.append(
            f"| {label} | {scope['mkgsvGated']['totalHits']} | "
            f"{scope['mkgsvRaw']['totalHits']} | "
            f"{scope['markov100Champion']['totalHits']} | "
            f"{scope['gatedCorrectionNetGain']:+d} | "
            f"{scope['rawCorrectionNetGain']:+d} | "
            f"{scope['proposalCount']} | {scope['activePredictionCount']} |"
        )
    lines.extend(
        [
            "",
            (
                "Validation gated/raw/Markov Brier: "
                f"`{validation['mkgsvGated']['brierScore']:.6f}` / "
                f"`{validation['mkgsvRaw']['brierScore']:.6f}` / "
                f"`{validation['markov100Champion']['brierScore']:.6f}`."
            ),
            (
                "Holdout gated/raw/Markov Brier: "
                f"`{holdout['mkgsvGated']['brierScore']:.6f}` / "
                f"`{holdout['mkgsvRaw']['brierScore']:.6f}` / "
                f"`{holdout['markov100Champion']['brierScore']:.6f}`."
            ),
            "",
            "Complete distributions, paired differences, replacements, component "
            "ablations, null support, and related baselines are in the JSON report.",
            "",
            "## Promotion decision",
            "",
            (
                "**Passed.** MKGSV v3 may be enabled by default with its runtime guard."
                if promotion["passed"]
                else (
                    "**Failed.** MKGSV remains experimental and disabled by default; "
                    "production output is exactly Markov 100."
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
