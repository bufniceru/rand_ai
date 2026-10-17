"""Research-first portfolio study: uv run python scripts/study_portfolio.py --stage smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path

import numpy as np

from rand_ai import Draw, Draws, load_lotto_results_yaml
from rand_ai.gui_bridge import DEFAULT_STRATEGY_IDS
from rand_ai.strategy_prediction import build_prediction_suites

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261017
PROTOCOL = {
    "version": 1,
    "seed": SEED,
    "initialHistory": 120,
    "randomReplicates": 100,
    "bootstrapBlockLength": 20,
    "bootstrapReplicates": 10000,
    "blocks": 5,
    "objectives": ["blended", "coverage_first"],
    "baselines": ["random_global", "random_pool"],
    "interpretation": "Exploratory historical evidence; dataset previously used for strategy selection. Synthetic runs diagnose leakage and false signals, not predictive improvement.",
    "coverage": "Exact unweighted pool-triple coverage; weighted coverage is a separate model-based metric.",
    "randomness": "Independent SHA-256 derived streams for synthetic histories, portfolios and bootstrap; random tickets are distinct within each portfolio.",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stream_seed(*parts):
    return int(digest([SEED, *parts])[:8], 16)


def source_hashes():
    paths = [
        *sorted((ROOT / "src/rand_ai").glob("*.py")),
        *sorted((ROOT / "src/rand_ai").glob("*.npz")),
        ROOT / "scripts/study_portfolio.py",
        ROOT / "web/package.json",
        ROOT / "web/package-lock.json",
        ROOT / "uv.lock",
        ROOT / "web/src/types.ts",
        ROOT / "web/scripts/portfolio-study-runner.ts",
        ROOT / "web/scripts/build-portfolio-study.mjs",
    ]
    paths.extend(
        ROOT / "web/src/lib" / name
        for name in (
            "drawPortfolio.ts",
            "drawPortfolioBacktest.ts",
            "portfolioStudy.ts",
        )
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.exists()
    }


def synthetic_draws(count, seed):
    rng = np.random.default_rng(seed)
    draws = Draws()
    for _ in range(count):
        draws.add(
            Draw(*(int(n) for n in rng.choice(np.arange(1, 50), 6, replace=False)))
        )
    return draws


def compact_history(draws, strategies):
    """Use the production causal engine, including its artifact activation guards."""
    dates = [draw.date for draw in draws if draw.date is not None]
    if dates != sorted(dates):
        raise ValueError(
            "Dataset dates must be nondecreasing; preserve source order for ties"
        )
    draws.prepare_predictions()
    records = []

    def capture(suite):
        records.append(
            {
                "referenceDrawNumber": suite.reference_draw_number,
                "targetDrawNumber": suite.target_draw_number,
                "date": draws[suite.target_draw_number - 1].date,
                "actualNumbers": list(suite.actual_numbers),
                "strategies": [
                    {"id": s.strategy_id, "ranking": [n.number for n in s.numbers]}
                    for s in suite.strategies
                ],
            }
        )

    build_prediction_suites(
        draws.draws,
        enabled_strategy_ids=strategies,
        history_start=len(draws),
        evaluated_suite=capture,
    )
    numbers = [
        {"date": draw.date, "numbers": [ball.value for ball in draw.balls]}
        for draw in draws
    ]
    return {
        "cacheKey": digest(numbers),
        "strategyIds": list(strategies),
        "draws": numbers,
        "records": records,
    }


def paired_interval(differences, seed, replicates=10000, block_length=20):
    values = np.asarray(differences, dtype=float)
    if not len(values):
        raise ValueError("Cannot compare an empty evaluation")
    rng = np.random.default_rng(seed)
    length = min(block_length, len(values))
    blocks = (len(values) + length - 1) // length
    means = np.empty(replicates)
    offsets = np.arange(length)
    for start in range(0, replicates, 250):
        batch = min(250, replicates - start)
        starts = rng.integers(0, len(values) - length + 1, size=(batch, blocks))
        indices = (starts[:, :, None] + offsets).reshape(batch, -1)[:, : len(values)]
        means[start : start + batch] = values[indices].mean(axis=1)
    observed = float(values.mean())
    return {
        "difference": observed,
        "interval95": np.quantile(means, [0.025, 0.975]).tolist(),
        "p": float(
            (1 + np.count_nonzero(means - observed >= observed)) / (replicates + 1)
        ),
        "targets": len(values),
        "blockLength": length,
        "preliminary": len(values) < 2 * block_length,
    }


def holm(comparisons):
    ordered = sorted(comparisons, key=lambda row: row["p"])
    previous = 0.0
    for index, row in enumerate(ordered):
        previous = max(previous, min(1.0, row["p"] * (len(ordered) - index)))
        row["holmP"] = previous


def summarize(rows, identity):
    groups = defaultdict(list)
    for row in rows:
        if row["method"] != "runtime":
            groups[(row["objective"], row["method"], row["target"])].append(row)
    target_means = {
        key: {
            metric: float(np.mean([r[metric] for r in values]))
            for metric in (
                "bestHits",
                "threePlus",
                "tripleCoverage",
                "coveredTriples",
                "generatedTickets",
            )
        }
        for key, values in groups.items()
    }
    summaries = []
    for objective, method in sorted({key[:2] for key in groups}):
        keys = sorted(key for key in groups if key[:2] == (objective, method))
        samples = [r for key in keys for r in groups[key]]
        summaries.append(
            {
                "objective": objective,
                "method": method,
                "targets": len(keys),
                "generationMs": float(np.mean([r["generationMs"] for r in samples]))
                if method in PROTOCOL["objectives"]
                else None,
                "modelWeightedTripleCoverage": float(
                    np.mean([r["weightedTripleCoverage"] for r in samples])
                )
                if method in PROTOCOL["objectives"]
                and all("weightedTripleCoverage" in r for r in samples)
                else None,
                "guaranteedTargets": sum(bool(r.get("guaranteed")) for r in samples),
                "meanUncoveredTriples": float(
                    np.mean([r["uncoveredTriples"] for r in samples])
                )
                if method in PROTOCOL["objectives"]
                and all("uncoveredTriples" in r for r in samples)
                else None,
                **{
                    metric: float(np.mean([target_means[key][metric] for key in keys]))
                    for metric in (
                        "bestHits",
                        "threePlus",
                        "tripleCoverage",
                        "coveredTriples",
                        "generatedTickets",
                    )
                },
                "bestHitDistribution": [
                    sum(r["bestHits"] == n for r in samples) / len(samples)
                    for n in range(7)
                ],
                "blocks": [
                    {
                        "targets": len(block),
                        "bestHits": float(
                            np.mean(
                                [target_means[keys[int(i)]]["bestHits"] for i in block]
                            )
                        )
                        if len(block)
                        else None,
                        "threePlus": float(
                            np.mean(
                                [target_means[keys[int(i)]]["threePlus"] for i in block]
                            )
                        )
                        if len(block)
                        else None,
                    }
                    for block in np.array_split(np.arange(len(keys)), 5)
                ],
            }
        )
    comparisons = []
    for metric in ("bestHits", "threePlus"):
        pairs = [
            (objective, objective, objective, baseline)
            for objective in PROTOCOL["objectives"]
            for baseline in PROTOCOL["baselines"]
        ]
        pairs.append(("coverage_first", "coverage_first", "blended", "blended"))
        for lo, lm, ro, rm in pairs:
            targets = sorted(
                {key[2] for key in groups if key[:2] == (lo, lm)}
                & {key[2] for key in groups if key[:2] == (ro, rm)}
            )
            difference = [
                target_means[(lo, lm, target)][metric]
                - target_means[(ro, rm, target)][metric]
                for target in targets
            ]
            label = f"{lo}/{lm} vs {ro}/{rm}"
            comparisons.append(
                {
                    "comparison": label,
                    "metric": metric,
                    **paired_interval(
                        difference, stream_seed("bootstrap", identity, label, metric)
                    ),
                }
            )
    return {
        "summaries": summaries,
        "comparisons": comparisons,
        "runtimes": [row for row in rows if row["method"] == "runtime"],
    }


def checkpoint_matches(path, identity):
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("identity") != identity:
            return False
        if "data" in value and value.get("dataHash") != digest(value["data"]):
            return False
        for suffix, expected in value.get("outputs", {}).items():
            artifact = path.with_name(f"{path.stem}-{suffix}.jsonl")
            if (
                not artifact.exists()
                or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected
            ):
                return False
        return True
    except ValueError, OSError:
        return False


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_report(output, manifest, jobs):
    # A single historical family covers every budget, metric and primary comparison.
    for kind in ("historical", "synthetic"):
        holm(
            [
                comparison
                for job in jobs
                if job["kind"] == kind
                for comparison in job["summary"]["comparisons"]
            ]
        )
    write_json(output / "aggregate.json", {"manifest": manifest, "jobs": jobs})
    lines = [
        "# Portfolio coverage study",
        "",
        PROTOCOL["interpretation"],
        "",
        f"Stage: {manifest['stage']}. Seed: {SEED}. No UI defaults changed.",
        "",
        "Coverage-first is a greedy heuristic, not a minimum-wheel solver. Guarantees apply only to complete verified pool-triple coverage.",
        "",
        "Random global coverage uses the 49-number universe; random-pool and model coverage use the selected pool. These denominators must not be conflated.",
        "",
        "| Job | Budget | Objective | Method | Targets | Mean best hits | At least one 3+ | Exact triple coverage |",
        "|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for job in jobs:
        for row in job["summary"]["summaries"]:
            lines.append(
                f"| {job['name']} | {job['budget']} | {row['objective']} | {row['method']} | {row['targets']} | {row['bestHits']:.4f} | {row['threePlus']:.2%} | {row['tripleCoverage']:.2%} |"
            )
    lines += [
        "",
        "## Paired comparisons",
        "",
        "Positive differences favor the first method. Intervals are paired moving-block bootstrap intervals; Holm adjustment covers all primary comparisons within historical/synthetic families. Short smoke histories are preliminary.",
        "",
        "| Job/budget | Comparison | Metric | Difference | 95% interval | Holm p |",
        "|---|---|---|---:|---|---:|",
    ]
    for job in jobs:
        for row in job["summary"]["comparisons"]:
            lines.append(
                f"| {job['name']}/{job['budget']} | {row['comparison']} | {row['metric']} | {row['difference']:.4f} | {row['interval95']} | {row['holmP']:.4f} |"
            )
    lines += [
        "",
        "## Runtime and availability",
        "",
        "| Job/budget | Objective | Generation ms/target | Model-weighted triple coverage | Mean uncovered triples | Guaranteed targets |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for job in jobs:
        for row in job["summary"]["summaries"]:
            if row["method"] in PROTOCOL["objectives"]:
                lines.append(
                    f"| {job['name']}/{job['budget']} | {row['objective']} | {row['generationMs']:.2f} | {row['modelWeightedTripleCoverage']} | {row['meanUncoveredTriples']} | {row['guaranteedTargets']} |"
                )
    for job in jobs:
        unavailable = {key: count for key, count in job["unavailable"].items() if count}
        lines.append(
            f"- {job['name']}/{job['budget']}: total evaluation runtime milliseconds {job['summary']['runtimes']}; unavailable forecast counts {unavailable or 'none'}."
        )
    lines += ["", "## Recommendation", ""]
    if manifest["stage"] == "smoke":
        lines.append(
            "Keep coverage-first research-only. This smoke run validates execution and reports preliminary diagnostics; complete historical and synthetic stages before deciding on UI exposure."
        )
    else:
        historical = [job for job in jobs if job["kind"] == "historical"]
        acceptable = bool(historical) and all(
            next(
                r
                for r in job["summary"]["summaries"]
                if r["method"] == "coverage_first"
            )["tripleCoverage"]
            > next(r for r in job["summary"]["summaries"] if r["method"] == "blended")[
                "tripleCoverage"
            ]
            and all(
                r["interval95"][0] >= (-0.05 if r["metric"] == "bestHits" else -0.01)
                for r in job["summary"]["comparisons"]
                if r["comparison"] == "coverage_first/coverage_first vs blended/blended"
            )
            and next(
                r
                for r in job["summary"]["summaries"]
                if r["method"] == "coverage_first"
            )["generationMs"]
            <= 2000
            for job in historical
        )
        lines.append(
            "Coverage and historical tolerance/runtime screening passed; consider an optional UI control only after reviewing the synthetic diagnostics. This establishes no predictive superiority."
            if acceptable
            else "Keep coverage-first research-only: historical coverage, uncertainty or runtime screening is incomplete or did not pass. Synthetic-only results cannot justify UI promotion."
        )
    lines += [
        "",
        "Screening defaults: positive exact coverage gain at every budget; paired lower bounds at least −0.05 best hits and −0.01 three-plus rate; coverage-first generation at most 2 seconds per target. No predictive-superiority claim without independent future evidence.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("smoke", "historical", "synthetic", "all"), default="smoke"
    )
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/lotto_results_2019.yaml"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/portfolio_study")
    args = parser.parse_args()
    output = args.output.resolve() / ("smoke" if args.stage == "smoke" else "full")
    output.mkdir(parents=True, exist_ok=True)
    work = ROOT / "web/dist/portfolio-study"
    work.mkdir(parents=True, exist_ok=True)
    roster = tuple(DEFAULT_STRATEGY_IDS)
    manifest = {
        "stage": args.stage,
        "protocol": PROTOCOL,
        "strategies": roster,
        "sources": source_hashes(),
        "historicalSource": str(args.dataset.resolve()),
        "historicalSourceSha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest()
        if args.dataset.exists()
        else None,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "scikit-learn": version("scikit-learn"),
            "scipy": version("scipy"),
            "xgboost": version("xgboost"),
            "node": subprocess.check_output(["node", "--version"], text=True).strip(),
        },
        "stages": {
            "smoke": {
                "budgets": [10],
                "maxTargets": 30,
                "syntheticHistories": 3,
                "syntheticDraws": 150,
            },
            "historical": {"budgets": [5, 10, 20]},
            "synthetic": {"budgets": [5, 10, 20], "histories": 20, "draws": 360},
        },
    }
    run_identity = digest(
        {key: value for key, value in manifest.items() if key != "stage"}
    )
    manifest["runIdentity"] = run_identity
    write_json(output / "protocol.json", manifest)
    subprocess.run(
        ["node", "scripts/build-portfolio-study.mjs"], cwd=ROOT / "web", check=True
    )
    datasets = []
    if args.stage in ("smoke", "historical", "all"):
        historical = load_lotto_results_yaml(args.dataset)
        if args.stage == "smoke":
            limited = Draws()
            for draw in historical.draws[:150]:
                limited.add(draw)
            historical = limited
        datasets.append(("historical", "historical", historical))
    if args.stage in ("smoke", "synthetic", "all"):
        for i in range(3 if args.stage == "smoke" else 20):
            datasets.append(
                (
                    f"synthetic-{i:02}",
                    "synthetic",
                    synthetic_draws(
                        150 if args.stage == "smoke" else 360, stream_seed("history", i)
                    ),
                )
            )
    jobs = []
    # Historical and synthetic stages share compatible completed jobs; smoke stays isolated.
    for path in sorted(output.glob("*-b*.json")):
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("runIdentity") == run_identity and checkpoint_matches(
                path, saved.get("identity")
            ):
                jobs.append(saved)
        except ValueError, OSError:
            continue
    for name, kind, draws in datasets:
        dataset_hash = digest(
            [[draw.date, [ball.value for ball in draw.balls]] for draw in draws]
        )
        base_identity = digest([run_identity, name, dataset_hash])
        history_path = output / f"{name}-history.json"
        if checkpoint_matches(history_path, base_identity):
            data = json.loads(history_path.read_text())["data"]
        else:
            print(
                f"Generating causal rankings: {name} ({len(draws)} draws)", flush=True
            )
            data = compact_history(draws, roster)
            write_json(
                history_path,
                {"identity": base_identity, "dataHash": digest(data), "data": data},
            )
        if len(data["records"]) < 120:
            raise ValueError("Study needs at least one target after 120 warm-up draws")
        for budget in [10] if args.stage == "smoke" else [5, 10, 20]:
            identity = digest([base_identity, budget])
            checkpoint = output / f"{name}-b{budget}.json"
            if checkpoint_matches(checkpoint, identity):
                result = json.loads(checkpoint.read_text())
                print(f"Resumed {name}, budget {budget}", flush=True)
            else:
                print(f"Comparing portfolios: {name}, budget {budget}", flush=True)
                input_path = work / "input.json"
                result_path = work / "result.json"
                write_json(
                    input_path,
                    {
                        "data": data,
                        "options": {
                            "budget": budget,
                            "seed": stream_seed("portfolios", name, budget),
                            "initialHistory": 120,
                            "maxTargets": 30 if args.stage == "smoke" else None,
                            "randomReplicates": 100,
                        },
                    },
                )
                subprocess.run(
                    [
                        "node",
                        str(work / "runner.mjs"),
                        str(input_path),
                        str(result_path),
                    ],
                    check=True,
                )
                raw = json.loads(result_path.read_text())
                evaluated = [
                    r for r in data["records"] if r["referenceDrawNumber"] >= 120
                ]
                if args.stage == "smoke":
                    evaluated = evaluated[:30]
                unavailable = {
                    strategy: sum(
                        not any(s["id"] == strategy for s in r["strategies"])
                        for r in evaluated
                    )
                    for strategy in roster
                }
                result = {
                    "identity": identity,
                    "runIdentity": run_identity,
                    "name": name,
                    "kind": kind,
                    "budget": budget,
                    "datasetHash": dataset_hash,
                    "portfolioSeed": stream_seed("portfolios", name, budget),
                    "unavailable": unavailable,
                    "summary": summarize(raw["rows"], identity),
                }
                for suffix, values in (
                    ("audit", raw["rows"]),
                    ("uncovered", raw["uncovered"]),
                ):
                    (output / f"{name}-b{budget}-{suffix}.jsonl").write_text(
                        "".join(
                            json.dumps(row, separators=(",", ":")) + "\n"
                            for row in values
                        ),
                        encoding="utf-8",
                    )
                result["outputs"] = {
                    suffix: hashlib.sha256(
                        (output / f"{name}-b{budget}-{suffix}.jsonl").read_bytes()
                    ).hexdigest()
                    for suffix in ("audit", "uncovered")
                }
                write_json(checkpoint, result)
            jobs = [
                job for job in jobs if (job["name"], job["budget"]) != (name, budget)
            ]
            jobs.append(result)
            write_report(output, manifest, jobs)
    print(f"Study saved to {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
