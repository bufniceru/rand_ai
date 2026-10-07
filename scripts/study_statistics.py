"""Reproducible exploratory Top-6 study; run with uv run python scripts/study_statistics.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rand_ai import DrawsStatistics, load_lotto_results_yaml
from rand_ai.gui_bridge import REPORT_IDS, STATISTICS_COMMAND_IDS
from rand_ai.sparse_neural_ticket import load_sparse_neural_ticket
from rand_ai.strategy_prediction import (
    STRATEGY_IDS,
    _STRATEGY_DEPENDENCIES,
    _StrategyState,
    build_prediction_suites,
)

SEED = 20261006
EXPECTED = 36 / 49
PROTOCOL = {
    "initialHistory": 120,
    "blocks": 5,
    "bootstrapBlockLength": 20,
    "bootstrapReplicates": 10000,
    "seed": SEED,
    "practicalMargin": 0.05,
    "mergeDiscoveryBlocks": 3,
    "maxMergePairs": 3,
    "mergeFormula": "equal mean of (49-rank)/48; ties number ascending",
    "interpretation": "Exploratory historical evidence; dataset previously used for selection.",
}
# Zero masks apply both when fitting and predicting, including direct TBL terms.
FAMILIES = {
    "svc": {
        "composition": (1, 2, 3),
        "freshness": (4, 5, 6, 7),
        "frequency": (8, 9, 10),
    },
    "tbl": {
        "composition": (1, 2, 3, 4),
        "freshness": (5, 6),
        "frequency": (7, 8, 9, 10),
        "relationships": (11,),
        "expert_ranks": (12, 13),
    },
    "sklearn_svm": {
        "composition": (0, 1, 2, 3),
        "freshness": (4, 5, 6, 7, 8),
        "frequency": (9, 10, 11, 12, 13),
        "relationships": (14,),
        "expert_ranks": tuple(range(15, 32)),
    },
}


@contextmanager
def ablate(strategy, family):
    """Research-only instrumentation; production APIs and defaults stay intact."""
    name = {
        "svc": "_svc_features",
        "tbl": "_tbl_features",
        "sklearn_svm": "_sklearn_svm_features",
    }[strategy]
    original = getattr(_StrategyState, name)

    def masked(self, *args, **kwargs):
        values = list(original(self, *args, **kwargs))
        for index in FAMILIES[strategy][family]:
            values[index] = 0.0
        return values

    with patch.object(_StrategyState, name, masked):
        yield


def summary(hits, blocks):
    values = np.asarray(hits)
    return {
        "draws": len(values),
        "totalHits": int(values.sum()),
        "meanHits": float(values.mean()),
        "liftOverRandom": float(values.mean() - EXPECTED),
        "hitDistribution": [int(np.sum(values == i)) for i in range(7)],
        "threePlusRate": float(np.mean(values >= 3)),
        "blocks": [float(values[b].mean()) if len(b) else None for b in blocks],
    }


def paired(difference, replicates=10000):
    """Non-circular moving blocks; centered bootstrap for a one-sided null test."""
    values = np.asarray(difference, dtype=float)
    if not len(values):
        raise ValueError("Paired comparison needs observations")
    length = min(20, len(values))
    rng = np.random.default_rng(SEED)
    starts = rng.integers(
        0,
        len(values) - length + 1,
        size=(replicates, (len(values) + length - 1) // length),
    )
    indices = (starts[:, :, None] + np.arange(length)).reshape(replicates, -1)[
        :, : len(values)
    ]
    means = values[indices].mean(axis=1)
    observed = float(values.mean())
    return {
        "difference": observed,
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "p": float((1 + np.sum(means - observed >= observed)) / (replicates + 1)),
    }


def holm(claims):
    """Adjust all improvement claims in this study as one family."""
    running = 0.0
    ordered = sorted(claims, key=lambda item: item["p"])
    for index, claim in enumerate(ordered):
        running = max(running, (len(ordered) - index) * claim["p"])
        claim["holmP"] = min(1.0, running)


def redundancy(left, right, left_hits, right_hits):
    overlap = np.array([len(set(a) & set(b)) for a, b in zip(left, right, strict=True)])
    result = paired(np.asarray(left_hits) - right_hits)
    identical = bool(np.all(overlap == 6))
    equivalent = (
        float(overlap.mean()) >= 5
        and result["ci95"][0] >= -0.05
        and result["ci95"][1] <= 0.05
    )
    return {
        "meanOverlap": float(overlap.mean()),
        "identicalSelections": int(np.sum(overlap == 6)),
        "exactDuplicate": identical,
        "consolidate": identical or equivalent,
        **result,
    }


def merge_candidates(hits, blocks):
    candidates = []
    for left, right in combinations(sorted(hits), 2):
        differences = [
            float((hits[left][b] - hits[right][b]).mean()) for b in blocks[:3]
        ]
        if max(differences) >= 0.05 and min(differences) <= -0.05:
            candidates.append((min(max(differences), -min(differences)), left, right))
    return [
        (left, right)
        for _, left, right in sorted(candidates, key=lambda x: (-x[0], x[1], x[2]))[:3]
    ]


def representative_map(results, pairs):
    """Equivalence is not transitive: every member must qualify against every other member."""
    eligible = {frozenset((p["left"], p["right"])) for p in pairs if p["consolidate"]}
    groups = {}
    representatives = {}
    for sid in sorted(
        results,
        key=lambda s: (
            -results[s]["meanHits"],
            len(results[s].get("dependencies", [])),
            s,
        ),
    ):
        representative = next(
            (
                rep
                for rep, members in groups.items()
                if all(frozenset((sid, member)) in eligible for member in members)
            ),
            sid,
        )
        groups.setdefault(representative, []).append(sid)
        representatives[sid] = representative
    return representatives


def merge_ranks(left, right):
    scores = {
        n: (49 - left.index(n) - 1 + 49 - right.index(n) - 1) / 96 for n in range(1, 50)
    }
    return tuple(sorted(scores, key=lambda n: (-scores[n], n)))


def validate_selection(numbers):
    if (
        len(numbers) != 6
        or len(set(numbers)) != 6
        or any(n < 1 or n > 49 for n in numbers)
    ):
        raise ValueError("Every selection must contain six unique numbers in 1–49")


def collect(draws, ids, cache=None):
    if cache is not None and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        print(f"Reusing {cache.name}", flush=True)
        return saved["records"], saved["seconds"]
    records = []
    started = time.perf_counter()

    def receive(suite):
        if suite.target_draw_number <= 120 or not suite.actual_numbers:
            return
        items = {}
        for strategy in suite.strategies:
            validate_selection(strategy.top_numbers)
            ranking = tuple(
                item.number for item in sorted(strategy.numbers, key=lambda x: x.rank)
            )
            if len(ranking) != 49 or set(ranking) != set(range(1, 50)):
                raise ValueError("Incomplete strategy ranking")
            items[strategy.strategy_id] = {
                "top": list(strategy.top_numbers),
                "ranking": list(ranking),
                "hits": len(set(strategy.top_numbers) & set(suite.actual_numbers)),
            }
        records.append(
            {
                "target": suite.target_draw_number,
                "actual": list(suite.actual_numbers),
                "strategies": items,
            }
        )

    build_prediction_suites(
        draws.draws,
        history_start=len(draws),
        enabled_strategy_ids=ids,
        evaluated_suite=receive,
        progress=lambda done, total: (
            print(f"  {done}/{total}", flush=True) if done % 100 == 0 else None
        ),
    )
    elapsed = time.perf_counter() - started
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"records": records, "seconds": elapsed}), encoding="utf-8"
        )
    return records, elapsed


def ablation_run(dataset, sid, family, cache):
    """Separate processes isolate research feature masks and reuse verified checkpoints."""
    draws = load_lotto_results_yaml(dataset)
    draws.prepare_predictions()
    with ablate(sid, family):
        reduced, seconds = collect(draws, [sid], cache)
    return sid, family, reduced, seconds


def cache_directory(dataset):
    digest = hashlib.sha256(dataset.read_bytes())
    digest.update(json.dumps(PROTOCOL, sort_keys=True).encode())
    digest.update(json.dumps(FAMILIES, sort_keys=True).encode())
    for path in sorted(Path("src/rand_ai").glob("*")):
        if path.suffix in {".py", ".npz"}:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return Path(".codex_tmp/statistics_study") / digest.hexdigest()


def study(dataset):
    draws = load_lotto_results_yaml(dataset)
    dates = [draw.date for draw in draws]
    if any(date is None for date in dates) or any(
        a > b for a, b in zip(dates, dates[1:])
    ):
        raise ValueError(
            "Study requires chronological dated draws; same-date source order is preserved"
        )
    if len(draws) < 125:
        raise ValueError("Study needs 120 initial draws and at least five targets")
    draws.prepare_predictions()
    skipped = {}
    ids = list(STRATEGY_IDS)
    try:
        artifact = load_sparse_neural_ticket()
    except FileNotFoundError, ValueError:
        artifact = None
    if artifact is None:
        ids.remove("sparse_neural_ticket")
        skipped["sparse_neural_ticket"] = (
            "Missing frozen artifact; unevaluated, retain experimental availability"
        )
    print("All-strategy walk-forward", flush=True)
    cache = cache_directory(dataset)
    records, elapsed = collect(draws, ids, cache / "baseline.json")
    blocks = list(np.array_split(np.arange(len(records)), 5))
    complete_ids = [sid for sid in ids if all(sid in r["strategies"] for r in records)]
    for sid in ids:
        if not any(sid in r["strategies"] for r in records):
            skipped[sid] = (
                "No runnable predictions on evaluated targets; retain experimental availability"
            )
    hits = {
        sid: np.array([r["strategies"][sid]["hits"] for r in records])
        for sid in complete_ids
    }
    tops = {sid: [r["strategies"][sid]["top"] for r in records] for sid in complete_ids}
    random_hits = []
    for record in records:
        top = (
            np.random.default_rng(SEED + record["target"])
            .choice(np.arange(1, 50), 6, replace=False)
            .tolist()
        )
        value = len(set(top) & set(record["actual"]))
        random_hits.append(value)
        record["strategies"]["seeded_random"] = {"top": top, "hits": value}
    claims = []
    results = {}
    for sid in ids:
        available = np.array(
            [i for i, r in enumerate(records) if sid in r["strategies"]], dtype=int
        )
        if not len(available):
            continue
        values = np.array([records[i]["strategies"][sid]["hits"] for i in available])
        local_blocks = [np.flatnonzero(np.isin(available, b)) for b in blocks]
        comparison = paired(values - np.array(random_hits)[available])
        theoretical = paired(values - EXPECTED)
        claims.extend((comparison, theoretical))
        results[sid] = {
            **summary(values, local_blocks),
            "againstSeededRandom": comparison,
            "againstTheoreticalRandom": theoretical,
            "evaluatedTargetRange": [
                records[available[0]]["target"],
                records[available[-1]]["target"],
            ],
            "commonHorizon": sid in complete_ids,
            "missingForecasts": len(records) - len(available),
            "dependencies": sorted(_STRATEGY_DEPENDENCIES.get(sid, set())),
            "cost": "shared all-strategy run; per-strategy isolated timing not measured",
        }
    pairs = []
    for left, right in combinations(sorted(complete_ids), 2):
        item = {
            "left": left,
            "right": right,
            **redundancy(tops[left], tops[right], hits[left], hits[right]),
        }
        pairs.append(item)
    ablations = []
    tasks = [(sid, family) for sid, families in FAMILIES.items() for family in families]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(
                ablation_run, dataset, sid, family, cache / f"{sid}__{family}.json"
            )
            for sid, family in tasks
        ]
        runs = [future.result() for future in futures]
    for sid, family, reduced, seconds in runs:
        if [r["target"] for r in reduced] != [r["target"] for r in records]:
            raise ValueError("Ablation targets must align")
        values = np.array([r["strategies"][sid]["hits"] for r in reduced])
        comparison = paired(values - hits[sid])
        claims.append(comparison)
        variant = f"{sid}__without_{family}"
        for record, reduced_record in zip(records, reduced, strict=True):
            record["strategies"][variant] = reduced_record["strategies"][sid]
        ablations.append(
            {
                "strategy": sid,
                "family": family,
                "maskedIndices": list(FAMILIES[sid][family]),
                "seconds": seconds,
                **summary(values, blocks),
                "againstParent": comparison,
            }
        )
    merges = []
    confirmation = np.concatenate(blocks[3:])
    for left, right in merge_candidates(hits, blocks):
        values = []
        for record in records:
            top = merge_ranks(
                record["strategies"][left]["ranking"],
                record["strategies"][right]["ranking"],
            )[:6]
            value = len(set(top) & set(record["actual"]))
            values.append(value)
            record["strategies"][f"merge__{left}__{right}"] = {
                "top": list(top),
                "hits": value,
            }
        values = np.array(values)
        comparisons = [
            paired((values - hits[parent])[confirmation]) for parent in (left, right)
        ]
        claims.extend(comparisons)
        merges.append(
            {
                "parents": [left, right],
                **summary(values, blocks),
                "confirmationMeanHits": float(values[confirmation].mean()),
                "againstParents": comparisons,
            }
        )
    holm(claims)
    for sid, result in results.items():
        result["disposition"] = (
            "retain experimental; no corrected improvement over both random baselines"
        )
        if all(
            result[key]["holmP"] < 0.05 and result[key]["difference"] >= 0.05
            for key in ("againstSeededRandom", "againstTheoreticalRandom")
        ):
            result["disposition"] = (
                "retain; historical hit evidence, future confirmation required"
            )
        result["uiChange"] = (
            "Keep strategy and saved configuration; label historical evidence as exploratory"
        )
    representatives = representative_map(
        {sid: results[sid] for sid in complete_ids}, pairs
    )
    for sid in complete_ids:
        if representatives[sid] != sid:
            results[sid]["disposition"] = (
                f"consolidate default presentation into {representatives[sid]}; retain backend"
            )
            results[sid]["uiChange"] = (
                "Offer original strategy in an expandable equivalent-method group"
            )
    for item in ablations:
        comparison = item["againstParent"]
        item["disposition"] = (
            "candidate input removal; future confirmation required"
            if comparison["difference"] >= 0.05 and comparison["holmP"] < 0.05
            else "retain input; removal has no corrected practical improvement"
        )
        item["uiChange"] = "No production feature removal from exploratory evidence"
    for item in merges:
        item["promote"] = all(
            c["difference"] >= 0.05 and c["holmP"] < 0.05
            for c in item["againstParents"]
        )
        item["disposition"] = (
            "add experimental merged option"
            if item["promote"]
            else "do not promote; retain parents"
        )
    statistics = []
    for table, frame in DrawsStatistics(draws).export_tables().items():
        statistics.append(
            {
                "id": table,
                "fields": list(frame.columns),
                "rows": len(frame),
                "role": "diagnostic"
                if "correlation" in table or "randomness" in table
                else "descriptive",
                "disposition": "retain accessible; no standalone predictive claim",
                "uiChange": "Shared expandable circular-space panel"
                if table == "space_frequencies"
                else "Expandable descriptive table"
                if table in {"number_descriptive", "space_descriptive", "summary"}
                else "keep",
                "evidence": "Describes history; predictive usefulness tested separately in consuming models",
                "dependencies": [],
                "uncertainty": "No direct six-number forecast from this table",
            }
        )
    return {
        "protocol": PROTOCOL,
        "dataset": str(dataset),
        "datasetSha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "drawCount": len(draws),
        "blockTargets": [
            [records[b[0]]["target"], records[b[-1]]["target"]] for b in blocks
        ],
        "allStrategySeconds": elapsed,
        "strategies": results,
        "skipped": skipped,
        "seededRandom": summary(random_hits, blocks),
        "redundancy": pairs,
        "ablations": ablations,
        "merges": merges,
        "statistics": statistics,
    }, records


def write_report(result, records, output):
    output.mkdir(parents=True, exist_ok=True)
    representatives = representative_map(
        {
            sid: item
            for sid, item in result["strategies"].items()
            if item.get("commonHorizon", True)
        },
        result["redundancy"],
    )
    groups = [
        {
            "representative": representative,
            "alternatives": sorted(
                sid
                for sid, parent in representatives.items()
                if parent == representative and sid != parent
            ),
        }
        for representative in sorted(set(representatives.values()))
        if any(
            sid != representative and parent == representative
            for sid, parent in representatives.items()
        )
    ]
    presentation = {
        "datasetSha256": result["datasetSha256"],
        "drawCount": result["drawCount"],
        "scope": "Historical equivalence in the lotto_results_2019.yaml study; future results may differ",
        "groups": groups,
    }
    (output / "statistics_study_presentation.json").write_text(
        json.dumps(presentation, indent=2) + "\n", encoding="utf-8"
    )
    (output / "statistics_study.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output / "statistics_study_predictions.jsonl").write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    with (output / "statistics_study_hits.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["target", "strategy", "actual", "prediction", "hits"])
        for record in records:
            for sid, item in record["strategies"].items():
                writer.writerow(
                    [
                        record["target"],
                        sid,
                        " ".join(map(str, record["actual"])),
                        " ".join(map(str, item["top"])),
                        item["hits"],
                    ]
                )
    lines = [
        "# Statistics and prediction simplification study",
        "",
        "Exploratory historical evidence only: this dataset previously influenced strategy selection. "
        "The last two blocks are a frozen within-study comparison, not independent confirmation.",
        "",
        "Reproduce: `uv run python scripts/study_statistics.py`",
        "",
        f"Dataset SHA-256: `{result['datasetSha256']}`. Draws: {result['drawCount']}.",
        f"Targets by block: {result['blockTargets']}. Random expectation: {EXPECTED:.6f} hits/draw.",
        "Protocol and raw rankings, selections, hits, dependencies, export fields, and uncertainty are in the adjacent JSON/JSONL/CSV files.",
        "Moving-block bootstrap: length 20, 10,000 replicates, seed 20261006. "
        "One-sided centered bootstrap tests use Holm correction across all baseline, ablation, and merge improvement claims. "
        "Equivalence uses paired 95% intervals; it is a simplification screen, not proof of future equivalence.",
        "",
        "## Strategies",
        "",
        "| Strategy | Draws | Hits | Mean | 3+ rate | Five block means | Decision |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for sid, item in sorted(
        result["strategies"].items(), key=lambda x: (-x[1]["meanHits"], x[0])
    ):
        lines.append(
            f"| {sid} | {item['draws']} | {item['totalHits']} | {item['meanHits']:.4f} | {item['threePlusRate']:.3%} | "
            + ", ".join(
                f"{v:.3f}" if v is not None else "unavailable" for v in item["blocks"]
            )
            + f" | {item['disposition']} |"
        )
    if "seededRandom" in result:
        item = result["seededRandom"]
        lines.append(
            f"| seeded_random | {item['draws']} | {item['totalHits']} | {item['meanHits']:.4f} | "
            f"{item['threePlusRate']:.3%} | "
            + ", ".join(f"{v:.3f}" for v in item["blocks"])
            + " | fixed independent random comparison |"
        )
    for sid, item in result["strategies"].items():
        if not item.get("commonHorizon", True):
            lines.append(
                f"\n{sid} has a shorter evaluation period: targets {item['evaluatedTargetRange']}, "
                f"{item['missingForecasts']} unavailable forecasts. Its mean is not comparable to full-horizon means.\n"
            )
    for sid, reason in result["skipped"].items():
        lines.append(f"\n{sid}: {reason}.")
    lines.extend(["", "## Redundancy", ""])
    for item in result["redundancy"]:
        if item["meanOverlap"] >= 5 or item["consolidate"]:
            lines.append(
                f"- {item['left']} / {item['right']}: overlap {item['meanOverlap']:.3f}; "
                f"paired interval {item['ci95']}; consolidate: {item['consolidate']}."
            )
    if not any(p["consolidate"] for p in result["redundancy"]):
        lines.append(
            "No pair meets the duplicate/equivalence consolidation rule. Retain all strategy controls."
        )
    if groups:
        lines.extend(["", "Default comparison charts use the following representatives. Original methods remain "
                      "in the full ranking and in expandable chart controls; enablement and saved settings do not change. "
                      "Every pair within a group must qualify: equivalence is not assumed to be transitive. "
                      "Representatives use highest observed mean, then fewer declared dependencies, then identifier.", "",
                      "| Representative | Expandable alternatives |", "|---|---|"])
        lines.extend(f"| {group['representative']} | {', '.join(group['alternatives'])} |" for group in groups)
    lines.extend(
        [
            "",
            "## Input removal experiments",
            "",
            "Families are removed at both training and prediction time; each model is retrained from scratch. "
            "SVC, TBL and online SVM are covered. Other models' inputs are untested and retained.",
            "",
            "| Consumer / removed family | Mean | Difference | Paired 95% interval | Holm p | Decision |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for item in result["ablations"]:
        c = item["againstParent"]
        lines.append(
            f"| {item['strategy']} / {item['family']} | {item['meanHits']:.4f} | {c['difference']:+.4f} | "
            f"{c['ci95']} | {c['holmP']:.4f} | {item['disposition']} |"
        )
    lines.extend(["", "## Merge experiments", ""])
    for item in result["merges"]:
        lines.append(
            f"- {' + '.join(item['parents'])}: confirmation mean {item['confirmationMeanHits']:.4f}; "
            f"parent comparisons {item['againstParents']}; {item['disposition']}."
        )
    lines.extend(
        [
            "",
            "## Statistics and presentation",
            "",
            "Circular-space position charts and their combined heatmap share one expandable detail panel under the aggregate distribution. "
            "These are views of the same counts, rather than independent predictive signals. "
            "Summary and number/space descriptive tables become expandable details. Charts, exports, settings and diagnostic access remain available.",
            "Drop the summed hit total and summed hit-distribution row across strategies: they count the same draw multiple times "
            "and are not the performance of a six-number prediction. Replace the aggregate card with theoretical random expectation. "
            "Keep all seven per-strategy match counts (including zero), display each strategy's actual evaluated-draw count, "
            "and exclude unavailable forecasts from averages and timeline curves.",
            "",
            "| Statistic | Role | Fields | Decision | UI |",
            "|---|---|---|---|---|",
        ]
    )
    for item in result["statistics"]:
        lines.append(
            f"| {item['id']} | {item['role']} | {', '.join(item['fields'])} | {item['disposition']} | {item['uiChange']} |"
        )
    lines.extend(["", "## Auxiliary panels and on-demand commands", ""])
    for item in result.get("panels", []):
        lines.append(
            f"- {item['id']}: {item['role']}; {item['disposition']}. {item['evidence']}"
        )
    lines.extend(
        [
            "",
            "Frequency count, appearance rate, observation share, deviation and standardized residual "
            "are transformations of the same observed count given the dataset size. "
            "Keep the existing single on-demand frequency chart and hover information; preserve exported fields. "
            "Space-position bars and the heatmap encode identical counts; the aggregate sums those position counts. "
            "Same-date draws retain their YAML source order.",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Cost and limitations",
            "",
            f"Shared strategy evaluation took {result['allStrategySeconds']:.2f} seconds on this machine. "
            "Ablation runtimes are recorded individually; shared computation prevents attributing the full-run time to individual strategies. "
            "Feature masks test specific consumers, not every use of a statistic. "
            "No uncertain statistical input is deleted, and no historical win is presented as proven future prediction.",
            "",
        ]
    )
    (output / "statistics_study.md").write_text("\n".join(lines), encoding="utf-8")


def enrich_inventory(result, dataset):
    """Include auxiliary report and command access beyond the compact export tables."""
    stats = DrawsStatistics(load_lotto_results_yaml(dataset))
    existing = {item["id"] for item in result["statistics"]}
    for name, frame in {
        "sampled_spaces": stats.sampled_spaces(),
        "group_count_frequencies": stats.group_count_frequencies(3),
        "group_signature_frequencies": stats.group_signature_frequencies(3),
    }.items():
        if name not in existing:
            result["statistics"].append(
                {
                    "id": name,
                    "fields": list(frame.columns),
                    "rows": len(frame),
                    "role": "descriptive",
                    "disposition": "retain accessible; no standalone predictive claim",
                    "uiChange": "keep",
                    "evidence": "Historical circular-space/group distribution, border space 3",
                    "dependencies": [],
                    "uncertainty": "Group-count prediction is distinct from number hits",
                }
            )
    for item in result["statistics"]:
        item["cost"] = (
            "deterministic heavy-analysis sample"
            if "spearman" in item["id"] or item["id"] == "sampled_spaces"
            else "one pass over history; bounded number/space categories"
        )
    predictive = {
        "predictions",
        "prediction-audit",
        "draw-comparison",
        "strategy-effectiveness",
        "strategy-hit-statistics",
        "draw-portfolio",
        "possible-draw",
    }
    result["panels"] = [
        {
            "id": name,
            "role": "prediction/evaluation"
            if name in predictive
            else "diagnostic"
            if name
            in {"relationships", "randomness", "nonlinear-dynamics", "autocorrelation"}
            else "descriptive",
            "disposition": "retain access and settings",
            "uiChange": "shared expandable details"
            if name in {"spaces", "numbers", "overview"}
            else "keep",
            "evidence": "Six-number effectiveness is evaluated in strategy comparisons; other historical displays do not establish hit lift",
            "uncertainty": "historical selection bias",
            "dependencies": [],
            "cost": "derived from existing history or calculated on demand",
        }
        for name in (*REPORT_IDS, *STATISTICS_COMMAND_IDS)
    ]
    result["protocol"]["sameDateOrder"] = (
        "preserve YAML source order; decreasing dates rejected"
    )
    try:
        artifact = load_sparse_neural_ticket()
    except FileNotFoundError, ValueError:
        artifact = None
    if artifact is not None and "sparse_neural_ticket" in result["strategies"]:
        result["strategies"]["sparse_neural_ticket"]["activationReferenceDraw"] = (
            artifact.activation_reference_draw
        )
        result["strategies"]["sparse_neural_ticket"]["uncertainty"] = (
            "Evaluated only after frozen-artifact activation; absent earlier predictions are not scored; artifact selected on historical data"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/lotto_results_2019.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("reports"))
    options = parser.parse_args()
    result, records = study(options.dataset)
    enrich_inventory(result, options.dataset)
    write_report(result, records, options.output)
    print(f"Study saved to {options.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
