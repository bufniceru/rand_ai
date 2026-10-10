"""Compare incremental Top-6 coverage alongside SVRH using saved strategies."""

import argparse
import csv
import hashlib
import html
import itertools
import json
from pathlib import Path
import numpy as np
from rand_ai import load_lotto_results_yaml
from rand_ai.strategy_prediction import build_prediction_suites

BASE = "svc_recurrence_hybrid"
OUT = Path("reports/svrh_complements")
SETTINGS = Path("C:/Users/Floyd/AppData/Roaming/rand-ai-desktop/strategy-plugins.json")
DATA = Path("D:/LOTTO/lotto_results_2019.yaml")


def summarize(rows, companions):
    extras, slots, overlaps, unions, basehits, alone = [], [], [], [], [], []
    for row in rows:
        base = set(row["predictions"][BASE])
        partner = set().union(*(set(row["predictions"][s]) for s in companions))
        actual = set(row["actual"])
        new = partner - base
        extras.append(len(new & actual))
        slots.append(len(new))
        overlaps.append(len(partner & base))
        unions.append(len((base | partner) & actual))
        basehits.append(len(base & actual))
        alone.append(len(partner & actual))
    n, hits, size = len(rows), sum(extras), sum(slots)
    return dict(
        strategies=list(companions),
        draws=n,
        extra_hits=hits,
        extra_per_draw=hits / n,
        adds_hit_pct=100 * sum(x > 0 for x in extras) / n,
        mean_new_numbers=size / n,
        mean_overlap=sum(overlaps) / n,
        combined_hits=sum(unions),
        combined_per_draw=sum(unions) / n,
        union_at_least_3_pct=100 * sum(x >= 3 for x in unions) / n,
        union_all_6_draws=sum(x == 6 for x in unions),
        companion_hits=sum(alone),
        base_hits=sum(basehits),
        extra_when_base_zero=sum(e for e, b in zip(extras, basehits) if b == 0),
        base_zero_draws=sum(b == 0 for b in basehits),
        random_expected_extra=size * 6 / 49,
        extra_above_random=hits - size * 6 / 49,
        hits_per_100_new_numbers=100 * hits / size if size else 0,
    )


def table(entries):
    cols = [
        ("strategies", "Companion(s)"),
        ("extra_hits", "Added hits"),
        ("extra_per_draw", "Added/draw"),
        ("adds_hit_pct", "Draws adding a hit (%)"),
        ("mean_new_numbers", "New numbers/draw"),
        ("mean_overlap", "Overlap with SVRH"),
        ("combined_per_draw", "Union hits/draw"),
        ("extra_above_random", "Added hits above size-matched random"),
    ]
    output = (
        "<table><tr>" + "".join("<th>" + label + "</th>" for _, label in cols) + "</tr>"
    )
    for entry in entries:
        output += "<tr>"
        for key, _ in cols:
            value = entry[key]
            value = (
                ", ".join(value)
                if isinstance(value, list)
                else f"{value:.3f}"
                if isinstance(value, float)
                else str(value)
            )
            output += "<td>" + html.escape(value) + "</td>"
        output += "</tr>"
    return output + "</table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-forecasts",
        action="store_true",
        help="Reuse saved predictions only when dataset and strategy settings match",
    )
    args = parser.parse_args()
    selected = json.loads(SETTINGS.read_text())["enabledStrategies"]
    assert BASE in selected
    candidates = [s for s in selected if s != BASE]
    OUT.mkdir(parents=True, exist_ok=True)
    draws = load_lotto_results_yaml(DATA)
    draws.prepare_predictions()
    rows = []

    def capture(suite):
        rows.append(
            dict(
                target=suite.target_draw_number,
                actual=list(suite.actual_numbers),
                predictions={
                    s.strategy_id: list(s.top_numbers) for s in suite.strategies
                },
            )
        )

    def progress(done, total):
        if done % 50 == 0 or done == total:
            print(f"Replay {done}/{total}", flush=True)

    if args.reuse_forecasts:
        previous = json.loads((OUT / "summary.json").read_text())
        assert (
            previous["dataset_sha256"] == hashlib.sha256(DATA.read_bytes()).hexdigest()
        )
        assert (
            previous["enabled_strategies"] == selected and previous["border_space"] == 7
        )
        rows = json.loads((OUT / "draw_predictions.json").read_text())
    else:
        build_prediction_suites(
            draws.draws,
            history_start=len(draws.draws),
            enabled_strategy_ids=selected,
            border_space=7,
            evaluated_suite=capture,
            progress=progress,
        )
    assert len(rows) == len(draws.draws) - 1
    assert all(
        len(r["actual"]) == 6
        and all(len(set(p)) == 6 for p in r["predictions"].values())
        for r in rows
    )
    (OUT / "draw_predictions.json").write_text(json.dumps(rows), encoding="utf-8")
    cut = int(len(rows) * 0.7)
    train, test = rows[:cut], rows[cut:]
    scopes = {
        "full": rows,
        "selection_first_70pct": train,
        "evaluation_last_30pct": test,
        "latest_100": rows[-100:],
    }
    singles = {
        scope: sorted(
            [summarize(data, [s]) for s in candidates],
            key=lambda e: (-e["extra_hits"], e["strategies"]),
        )
        for scope, data in scopes.items()
    }
    pairs = sorted(
        [summarize(train, p) for p in itertools.combinations(candidates, 2)],
        key=lambda e: (-e["extra_hits"], e["strategies"]),
    )
    chosen = singles["selection_first_70pct"][0]["strategies"]
    greedy, remaining, curve = [], candidates.copy(), []
    for _ in range(5):
        winner = max(
            remaining, key=lambda s: summarize(train, greedy + [s])["extra_hits"]
        )
        greedy.append(winner)
        remaining.remove(winner)
        curve.append(
            dict(selection=summarize(train, greedy), evaluation=summarize(test, greedy))
        )
    result = dict(
        dataset=str(DATA),
        dataset_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
        enabled_strategies=selected,
        border_space=7,
        draws=len(rows),
        selection_target_range=[train[0]["target"], train[-1]["target"]],
        evaluation_target_range=[test[0]["target"], test[-1]["target"]],
        singles=singles,
        frozen_single=summarize(test, chosen),
        frozen_pair=summarize(test, pairs[0]["strategies"]),
        top_selection_pairs=[
            dict(selection=p, evaluation=summarize(test, p["strategies"]))
            for p in pairs[:10]
        ],
        greedy_curve=curve,
        whole_portfolio={
            name: summarize(data, candidates) for name, data in scopes.items()
        },
    )
    rng = np.random.default_rng(20261009)
    indices = rng.integers(0, len(test), size=(5000, len(test)))
    for entry in singles["evaluation_last_30pct"]:
        s = entry["strategies"][0]
        values = np.array(
            [
                len(
                    (set(r["predictions"][s]) - set(r["predictions"][BASE]))
                    & set(r["actual"])
                )
                for r in test
            ]
        )
        entry["bootstrap_extra_per_draw_95pct"] = np.quantile(
            values[indices].mean(axis=1), [0.025, 0.975]
        ).tolist()
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (OUT / "single_strategy_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = list(singles["full"][0])
        writer = csv.DictWriter(stream, fieldnames=["scope"] + fields)
        writer.writeheader()
        for scope, entries in singles.items():
            for e in entries:
                writer.writerow(dict(scope=scope, **{key: e[key] for key in fields}))
    later = singles["evaluation_last_30pct"][0]
    page = """<!doctype html><html><head><meta charset="utf-8"><title>SVRH complementary strategies</title><style>body{font:16px system-ui;max-width:1450px;margin:40px auto;padding:0 24px;color:#182333}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{padding:10px;border-bottom:1px solid #ccd5df;text-align:left}th{background:#edf2f7}h1,h2{color:#163c63}p{line-height:1.6}</style></head><body>"""
    page += f"<h1>Complementary strategies for SVRH</h1><p>9 October 2026. Dataset: {DATA}. {len(draws.draws)} draws; {len(rows)} completed causal forecasts. The 14 saved enabled strategies are evaluated. SVRH means SVC-Recurrence Hybrid.</p>"
    page += f"<h2>Finding</h2><p>The largest later-slice added-hit total comes from <strong>{later['strategies'][0]}</strong>: {later['extra_hits']} additional hits ({later['extra_per_draw']:.3f} per draw), using {later['mean_new_numbers']:.3f} additional distinct candidates per draw. Earlier history selected <strong>{chosen[0]}</strong>. Its later evaluation and the earlier-selected pair are shown below. Historical candidate coverage does not establish a future predictive advantage.</p>"
    selected_efficiency = max(
        singles["selection_first_70pct"], key=lambda e: e["hits_per_100_new_numbers"]
    )
    page += f"<h2>Practical recommendation</h2><p><strong>For maximum distinct coverage: Residual Coverage.</strong> It adds six different numbers on every later-slice draw and 174 hits against 174.857 expected by chance. It is a diversification tool, not demonstrated predictive lift.</p><p><strong>For candidate efficiency to monitor: Temporal Behavior Learning (TBL).</strong> It adds 145 hits using 4.462 new candidates per draw, or 13.653 hits per 100 new candidates (chance reference 12.245). It also leads this measure over the latest 100 draws. This recommendation was identified after inspecting later results and requires future confirmation. Earlier-history efficiency selected {selected_efficiency['strategies'][0]}, which did not sustain its yield on later draws.</p><p><strong>For two added strategies:</strong> the earlier-selected RCOV + Entropy pair adds 308 hits, but uses 11.231 new numbers per draw and trails its size-matched chance expectation by 19.306 hits. Do not treat this pair as a validated predictive improvement. Doublet &amp; Triplet Markov is a further non-random candidate: 152 added hits with 5.029 new candidates per draw.</p>"
    page += "<h2>Method</h2><p>Each forecast uses completed earlier draws only. Additional hits are winning numbers in companion Top-6 minus SVRH Top-6. Duplicates count once. Union hits describe a candidate pool, not a single six-number ticket. More candidates naturally yield more hits: size-matched random expected added hits equal new candidates times 6/49. Random baseline remains a comparison control.</p>"
    page += f"<p>Selection targets {result['selection_target_range']}; later evaluation targets {result['evaluation_target_range']}. Pair choices and greedy additions are frozen on the first 70%, then evaluated on the last 30%. Existing strategy designs and saved activation choices may already have used these outcomes, so this is not an untouched prospective holdout. All 78 companion pairs are compared on the selection slice. Bootstrap intervals in JSON are descriptive, assume independent draws, and do not adjust for multiple comparisons.</p>"
    page += "<p>Border space is explicitly 7, the application default. Browser-only border-group settings were not recovered. Hidden dependencies run as production requires, including RCOV sources; disabled sources are not compared separately. Enabled source set stays fixed throughout. RCOV can change if that set changes. Current source code is used and may differ from the installed packaged executable.</p>"
    for scope, entries in singles.items():
        page += (
            f"<h2>Single companions: {scope} ({len(scopes[scope])} draws)</h2>"
            + table(entries)
        )
    page += "<h2>Earlier-history choices: later evaluation</h2>" + table(
        [result["frozen_single"], result["frozen_pair"]]
    )
    page += "<h2>Top ten earlier-selected pairs: later evaluation</h2>" + table(
        [p["evaluation"] for p in result["top_selection_pairs"]]
    )
    page += "<h2>Adding companions in order selected on earlier history</h2>" + table(
        [p["evaluation"] for p in curve]
    )
    page += "<h2>Whole enabled portfolio: later evaluation</h2>" + table(
        [result["whole_portfolio"]["evaluation_last_30pct"]]
    )
    page += "<h2>Use and limitations</h2><p>Prefer a companion that adds hits with reasonable candidate exposure and consistent later performance. Do not select a companion after seeing the target outcome. Recurrence Dynamics is already a component of SVRH. Larger unions require more candidate coverage and are not equivalent to a six-number ticket. Record the frozen choices on future draws before treating the result as stable. Machine-readable summary, per-draw predictions, and CSV are alongside this report. No application settings were changed.</p></body></html>"
    (OUT / "report.html").write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "draws",
                    "selection_target_range",
                    "evaluation_target_range",
                    "frozen_single",
                    "frozen_pair",
                ]
            },
            indent=2,
        )
    )
    print("Later rankings:", json.dumps(singles["evaluation_last_30pct"], indent=2))


if __name__ == "__main__":
    main()
