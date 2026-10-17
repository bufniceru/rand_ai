"""Reproducible, leakage-safe experimental LSTM research (no UI registration)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rand_ai import Draw, load_lotto_results_yaml
from rand_ai.lstm import (
    HIDDEN,
    SEEDS,
    SHAPES,
    WINDOW,
    encode_draws,
    model_probabilities,
    prefix_fingerprint,
    top_six,
)
from rand_ai.strategy_prediction import _StrategyState

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("uniform", "random", "frequency", "cdm", "bayesian")


@dataclass(frozen=True)
class Config:
    validation: int = 150
    holdout: int = 250
    minimum_train: int = 200
    max_epochs: int = 100
    patience: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    bootstrap: int = 10000
    block: int = 20

    def __post_init__(self):
        if (
            any(
                value < 1
                for value in (
                    self.validation,
                    self.holdout,
                    self.minimum_train,
                    self.max_epochs,
                    self.patience,
                    self.batch_size,
                    self.bootstrap,
                    self.block,
                )
            )
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("LSTM configuration values must be positive and finite")


def build_dataset(draws: list[Draw], config: Config):
    encoded = encode_draws(draws)
    count = len(draws) - WINDOW
    train_end = count - config.validation - config.holdout
    if train_end < config.minimum_train:
        raise ValueError(
            "Insufficient draws for chronological training/validation/test split"
        )
    windows = np.stack([encoded[i : i + WINDOW] for i in range(count)])
    targets = encoded[WINDOW:]
    return windows, targets, train_end, train_end + config.validation


def torch_runtime():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install research dependencies with uv sync --group research"
        ) from exc
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Use the same explicit CPU recurrence in training and portable inference.
    torch.backends.mkldnn.enabled = False  # ty: ignore[invalid-assignment]
    return torch


def create_model(seed: int):
    torch = torch_runtime()
    torch.manual_seed(seed)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(49, HIDDEN, batch_first=True)
            self.head = torch.nn.Linear(HIDDEN, 49)
            torch.nn.init.constant_(self.head.bias, math.log(6 / 43))

        def forward(self, windows):
            output, _state = self.lstm(windows)
            return self.head(output[:, -1])

    return Model()


def train_model(
    windows, targets, train_end: int, validation_end: int, seed: int, config: Config
):
    """Never inspect holdout tensors during optimization/checkpoint selection."""
    torch = torch_runtime()
    model = create_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    x = torch.tensor(windows[:validation_end], dtype=torch.float32)
    y = torch.tensor(targets[:validation_end], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best = None
    best_epoch = 0
    losses = []
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(train_end, generator=generator)
        for indices in order.split(config.batch_size):
            optimizer.zero_grad()
            loss = criterion(model(x[indices]), y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(x[train_end:]), y[train_end:]))
        if not math.isfinite(validation_loss):
            raise ValueError("Non-finite LSTM validation loss")
        losses.append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best = {
                name: value.detach().numpy().astype(np.float64).copy()
                for name, value in model.state_dict().items()
            }
        if epoch - best_epoch >= config.patience:
            break
    assert best is not None
    model.load_state_dict(
        {name: torch.tensor(value, dtype=torch.float32) for name, value in best.items()}
    )
    with torch.no_grad():
        reference = torch.sigmoid(model(x[train_end:])).numpy()
    portable = model_probabilities(best, windows[train_end:validation_end])
    error = float(np.max(np.abs(reference - portable)))
    if error > 1e-6:
        raise AssertionError(f"PyTorch/NumPy inference parity failed: {error}")
    return best, {
        "seed": seed,
        "bestEpoch": best_epoch,
        "epochs": len(losses),
        "validationLosses": losses,
        "parityMaxAbsError": error,
        "seconds": time.perf_counter() - started,
    }


def baseline_predictions(draws: list[Draw]):
    """Use exactly the production state updates and raw Bayesian estimates."""
    state = _StrategyState(("bayesian", "cdm"))
    probabilities = {name: [] for name in ("uniform", "frequency", "bayesian")}
    tickets = {name: [] for name in BASELINES}
    rng = np.random.default_rng(20261017)
    for index, draw in enumerate(draws[:-1]):
        drawn = {ball.value for ball in draw.balls}
        state.train(drawn)
        state.remember(drawn, draw.date)
        if index + 1 < WINDOW:
            continue
        gaps = state.current_gaps()
        raw, _details = state._gap_model_scores(
            gaps, weighted=False, raw_probabilities=True
        )
        values = {
            "uniform": np.full(49, 6 / 49),
            "frequency": np.asarray(
                [state.appearances[n] / state.draw_count for n in range(1, 50)]
            ),
            "bayesian": np.asarray([raw[n] for n in range(1, 50)]),
        }
        for name, value in values.items():
            probabilities[name].append(value)
        tickets["uniform"].append(top_six(values["uniform"][None])[0])
        tickets["random"].append(rng.choice(np.arange(1, 50), 6, replace=False))
        # Both lifetime baselines share the production CDM gap/number tie-break.
        ranked = sorted(
            range(1, 50), key=lambda n: (-state.appearances[n], -gaps[n], n)
        )[:6]
        tickets["frequency"].append(ranked)
        tickets["cdm"].append(ranked)
        tickets["bayesian"].append(
            sorted(range(1, 50), key=lambda n: (-raw[n], -gaps[n], n))[:6]
        )
    return (
        {name: np.asarray(rows) for name, rows in probabilities.items()},
        {name: np.asarray(rows) for name, rows in tickets.items()},
    )


def hit_counts(labels, tickets):
    return np.take_along_axis(labels, tickets - 1, axis=1).sum(axis=1).astype(int)


def metrics(labels, tickets, probabilities=None):
    hits = hit_counts(labels, tickets)
    result = {
        "draws": len(labels),
        "meanHits": float(hits.mean()),
        "matchCounts": np.bincount(hits, minlength=7).tolist(),
        "atLeastThreeRate": float(np.mean(hits >= 3)),
    }
    if probabilities is not None:
        top_six(probabilities)  # validates unscaled probabilities
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        result["brier"] = float(np.mean((labels - probabilities) ** 2))
        result["logLoss"] = float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
        )
        bins = np.minimum((probabilities * 10).astype(int), 9)
        result["calibration"] = [
            {
                "lower": i / 10,
                "upper": (i + 1) / 10,
                "count": int(np.sum(bins == i)),
                "meanPrediction": float(probabilities[bins == i].mean())
                if np.any(bins == i)
                else None,
                "observedRate": float(labels[bins == i].mean())
                if np.any(bins == i)
                else None,
            }
            for i in range(10)
        ]
    return result


def paired_comparison(difference, config: Config):
    """Non-circular moving-block bootstrap; centered one-sided null test."""
    difference = np.asarray(difference, dtype=float)
    length = min(config.block, len(difference))
    rng = np.random.default_rng(20261017)
    means = []
    for _ in range(config.bootstrap):
        starts = rng.integers(
            0, len(difference) - length + 1, size=math.ceil(len(difference) / length)
        )
        sample = np.concatenate(
            [difference[start : start + length] for start in starts]
        )[: len(difference)]
        means.append(sample.mean())
    average = float(difference.mean())
    centered = np.asarray(means) - average
    return {
        "difference": average,
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "p": float((1 + np.sum(centered >= average)) / (config.bootstrap + 1)),
    }


def holm(comparisons):
    adjusted = 0.0
    ordered = sorted(comparisons, key=lambda name: comparisons[name]["p"])
    for rank, name in enumerate(ordered):
        adjusted = max(
            adjusted, min(1.0, (len(ordered) - rank) * comparisons[name]["p"])
        )
        comparisons[name]["holmP"] = adjusted


def write_artifact(path, models, draws, activation, passed):
    metadata = {
        "schemaVersion": 1,
        "model": "lstm-7x49-h32",
        "seeds": list(SEEDS),
        "activationReferenceDraw": activation,
        "prefixFingerprint": prefix_fingerprint(draws[:activation]),
        "historicalGatePassed": bool(passed),
        "status": "exploratory research only",
    }
    arrays = {
        f"seed{i}/{name}": weights[name]
        for i, weights in enumerate(models)
        for name in SHAPES
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=json.dumps(metadata), **arrays)


def json_write(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_job(draws: list[Draw], config: Config, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    x, y, train_end, val_end = build_dataset(draws, config)
    identity = {
        "history": prefix_fingerprint(draws),
        "config": asdict(config),
        "source": source_fingerprint(),
        "torch": str(torch_runtime().__version__),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    identity_path = output / "identity.json"
    if not identity_path.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing unrecognized run directory: {output}")
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(f"Refusing incompatible run directory: {output}")
    json_write(identity_path, identity)
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text()), None
    baselines, tickets = baseline_predictions(draws)
    models, training, predictions = [], [], []
    for seed in SEEDS:
        weight_path = output / f"seed-{seed}.npz"
        log_path = output / f"seed-{seed}.json"
        if weight_path.exists() and log_path.exists():
            with np.load(weight_path, allow_pickle=False) as archive:
                weights = {name: archive[name] for name in SHAPES}
            log = json.loads(log_path.read_text())
        else:
            weights, log = train_model(x, y, train_end, val_end, seed, config)
            np.savez_compressed(weight_path, **weights)
            json_write(log_path, log)
        models.append(weights)
        training.append(log)
        predictions.append(model_probabilities(weights, x[train_end:]))
    ensemble = np.mean(predictions, axis=0)
    ensemble_tickets = top_six(ensemble)
    scopes: dict[str, Any] = {}
    for name, selection, offset in (
        ("validation", slice(train_end, val_end), 0),
        ("holdout", slice(val_end, len(y)), config.validation),
    ):
        labels = y[selection]
        p = ensemble[offset : offset + len(labels)]
        selected = ensemble_tickets[offset : offset + len(labels)]
        hits = hit_counts(labels, selected)
        comparisons = {
            baseline: paired_comparison(
                hits - hit_counts(labels, rows[selection]), config
            )
            for baseline, rows in tickets.items()
            if baseline != "uniform"
        }
        holm(comparisons)
        scopes[name] = {
            "ensemble": metrics(labels, selected, p),
            "individualSeeds": [
                metrics(
                    labels,
                    top_six(values[offset : offset + len(labels)]),
                    values[offset : offset + len(labels)],
                )
                for values in predictions
            ],
            "baselines": {
                baseline: metrics(
                    labels,
                    rows[selection],
                    baselines[baseline][selection] if baseline in baselines else None,
                )
                for baseline, rows in tickets.items()
            },
            "pairedMeanHits": comparisons,
        }
    holdout = scopes["holdout"]
    passed = all(
        c["ci95"][0] > 0 and c["holmP"] < 0.05
        for c in holdout["pairedMeanHits"].values()
    )
    passed = passed and all(
        holdout["ensemble"][metric] < holdout["baselines"]["uniform"][metric]
        for metric in ("brier", "logLoss")
    )
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in range(train_end, len(y)):
            handle.write(
                json.dumps(
                    {
                        "targetDraw": row + WINDOW + 1,
                        "scope": "validation" if row < val_end else "holdout",
                        "actual": (np.flatnonzero(y[row]) + 1).tolist(),
                        "ensembleProbabilities": ensemble[row - train_end].tolist(),
                        "ensembleTicket": ensemble_tickets[row - train_end].tolist(),
                        "seedProbabilities": [
                            values[row - train_end].tolist() for values in predictions
                        ],
                        "baselineTickets": {
                            name: rows[row].tolist() for name, rows in tickets.items()
                        },
                        "baselineProbabilities": {
                            name: rows[row].tolist() for name, rows in baselines.items()
                        },
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    result = {
        "splits": {
            "trainTargets": [WINDOW + 1, WINDOW + train_end],
            "validationTargets": [WINDOW + train_end + 1, WINDOW + val_end],
            "holdoutTargets": [WINDOW + val_end + 1, len(draws)],
        },
        "training": training,
        **scopes,
        "historicalGatePassed": bool(passed),
    }
    json_write(output / "result.json", result)
    return result, (models, WINDOW + val_end)


def source_fingerprint():
    names = [
        Path(__file__),
        ROOT / "src/rand_ai/lstm.py",
        ROOT / "src/rand_ai/strategy_prediction.py",
    ]
    return hashlib.sha256(b"".join(path.read_bytes() for path in names)).hexdigest()


def write_report(output, protocol):
    jobs = {
        path.parent.name: json.loads(path.read_text())
        for path in sorted(output.glob("*/result.json"))
    }
    lines = [
        "# Experimental LSTM study",
        "",
        "Exploratory historical evidence only; this history previously influenced development.",
        "No desktop registration, automatic installation, or claim of confirmed advantage.",
        "",
        "Fixed 7 x 49 inputs; one 32-unit LSTM; five-seed probability ensemble.",
        "",
        "| Dataset | Mean hits | 3+ rate | Brier | Log loss | Historical gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, result in jobs.items():
        metric = result["holdout"]["ensemble"]
        lines.append(
            f"| {name} | {metric['meanHits']:.6f} | {metric['atLeastThreeRate']:.3%} | "
            f"{metric['brier']:.6f} | {metric['logLoss']:.6f} | {result['historicalGatePassed']} |"
        )
    synthetic = [
        result for name, result in jobs.items() if name.startswith("synthetic-")
    ]
    if synthetic:
        aggregate = {
            "histories": len(synthetic),
            "independentUnit": "history (not training seed)",
            "meanMetrics": {
                key: float(np.mean([r["holdout"]["ensemble"][key] for r in synthetic]))
                for key in ("meanHits", "atLeastThreeRate", "brier", "logLoss")
            },
            "gatePassCount": sum(r["historicalGatePassed"] for r in synthetic),
        }
        json_write(output / "synthetic-aggregate.json", aggregate)
        lines.extend(
            [
                "",
                f"Synthetic controls completed: {len(synthetic)}/20. "
                "Seeds are averaged within each history, not counted as independent evidence.",
            ]
        )
    if "historical" in jobs:
        result = jobs["historical"]["holdout"]
        lines += [
            "",
            "## Historical comparisons",
            "",
            "| Baseline | Mean hits | LSTM difference | Paired 95% interval | Holm p |",
            "|---|---:|---:|---|---:|",
        ]
        for name, comparison in result["pairedMeanHits"].items():
            lines.append(
                f"| {name} | {result['baselines'][name]['meanHits']:.6f} | "
                f"{comparison['difference']:.6f} | {comparison['ci95']} | {comparison['holmP']:.6f} |"
            )
    lines += [
        "",
        "Full metrics, fixed-bin reliability data, individual seed results, training logs,",
        "split boundaries, raw forecasts and fingerprints are in adjacent JSON/JSONL/NPZ files.",
        "CDM has ranking metrics only; its categorical shares are not inclusion probabilities.",
        "Historical gate: positive paired lower bounds and Holm p < .05 against all four",
        "ticket baselines, plus lower Brier and log loss than uniform probabilities.",
        "No tuning or refitting follows inspection of holdout results.",
        "",
        "Reproduce with the command recorded in protocol.json.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/lotto_results_2019.yaml"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/lstm_study/full")
    parser.add_argument(
        "--stage", choices=("smoke", "historical", "synthetic", "all"), default="all"
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        help="Explicit research artifact export; never installed automatically",
    )
    args = parser.parse_args()
    if args.artifact and args.stage not in ("historical", "all"):
        parser.error("--artifact requires historical or all stage")
    if args.artifact and args.artifact.suffix.lower() != ".npz":
        parser.error("--artifact must have .npz extension")
    draws = list(load_lotto_results_yaml(args.dataset).draws)
    smoke = args.stage == "smoke"
    if smoke:
        draws = draws[:80]
    config = (
        Config(
            validation=15,
            holdout=20,
            minimum_train=20,
            max_epochs=3,
            patience=2,
            bootstrap=100,
        )
        if smoke
        else Config()
    )
    build_dataset(draws, config)
    protocol = {
        "schemaVersion": 1,
        "history": prefix_fingerprint(draws),
        "datasetSha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "source": source_fingerprint(),
        "config": asdict(config),
        "seeds": list(SEEDS),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch_runtime().__version__),
        "syntheticHistories": 20,
        "architecture": "LSTM(49,32), Dense(32,49); unweighted BCE; sigmoid inference",
        "training": "CPU deterministic; clip norm 1; independent windows; no scaling; no refit",
        "command": "uv run --group research python scripts/benchmark_lstm.py --stage "
        + ("smoke" if smoke else "all")
        + " --dataset "
        + str(args.dataset)
        + " --output "
        + str(args.output),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if not path.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing unrecognized output directory")
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Refusing incompatible run directory; choose a new --output")
    json_write(path, protocol)
    if args.stage in ("historical", "all", "smoke"):
        result, artifact = run_job(
            draws, config, args.output / ("smoke" if smoke else "historical")
        )
        if args.artifact:
            if args.artifact.exists():
                raise ValueError("Refusing to overwrite an existing artifact")
            if artifact is None:
                folder = args.output / "historical"
                models = []
                for seed in SEEDS:
                    with np.load(
                        folder / f"seed-{seed}.npz", allow_pickle=False
                    ) as archive:
                        models.append({name: archive[name] for name in SHAPES})
                artifact = models, result["splits"]["holdoutTargets"][0] - 1
            write_artifact(
                args.artifact,
                artifact[0],
                draws,
                artifact[1],
                result["historicalGatePassed"],
            )
        print(
            f"Historical/smoke complete: gate={result['historicalGatePassed']}",
            flush=True,
        )
        write_report(args.output, protocol)
    if args.stage in ("synthetic", "all"):
        for index in range(20):
            rng = np.random.default_rng(20261017 + index)
            synthetic = [
                Draw(*map(int, rng.choice(np.arange(1, 50), 6, replace=False)))
                for _ in draws
            ]
            run_job(synthetic, config, args.output / f"synthetic-{index:02d}")
            write_report(args.output, protocol)
            print(f"Synthetic history {index + 1}/20 complete", flush=True)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
