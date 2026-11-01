"""Causal minimum-direction forecasts and an evidence-gated research protocol."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rand_ai.draws import Draws

DIRECTIONS = ("lower", "equal", "higher")
CANDIDATES = ("historical", "transition", "logistic")
MODELS = ("baseline", *CANDIDATES, "gated")
PROTOCOL = {
    "version": 2,
    "warmup": 120,
    "refitInterval": 20,
    "minimumScores": 200,
    "priorStrength": 50,
    "logisticC": 0.1,
    "bootstrapBlockLength": 20,
    "bootstrapReplicates": 2000,
    "seed": 20261101,
    "syntheticHistories": 20,
    "interpretation": "Exploratory historical evidence; history previously used for research.",
}
MINIMUM_PRIOR = np.array(
    [math.comb(49 - m, 5) / math.comb(49, 6) for m in range(1, 45)]
)


def baseline_probabilities(minimum: int) -> list[float]:
    """Return exact lower/equal/higher probabilities for a fair 6/49 draw."""
    if type(minimum) is not int or not 1 <= minimum <= 44:
        raise ValueError("Minimum must be an integer from 1 through 44")
    denominator = math.comb(49, 6)
    return [
        1 - math.comb(50 - minimum, 6) / denominator,
        math.comb(49 - minimum, 5) / denominator,
        math.comb(49 - minimum, 6) / denominator,
    ]


def position_distribution(position: int) -> np.ndarray:
    """Exact probability mass over the 44 valid values of a sorted position."""
    if type(position) is not int or not 1 <= position <= 6:
        raise ValueError("Position must be an integer from 1 through 6")
    return np.array(
        [
            math.comb(x - 1, position - 1)
            * math.comb(49 - x, 6 - position)
            / math.comb(49, 6)
            for x in range(position, position + 44)
        ]
    )


def position_baseline(value: int, position: int) -> list[float]:
    prior = position_distribution(position)
    if type(value) is not int or not position <= value <= position + 43:
        raise ValueError("Value is outside the sorted position's valid range")
    if position == 1:
        return baseline_probabilities(value)
    offset = value - position
    return [
        float(prior[:offset].sum()),
        float(prior[offset]),
        float(prior[offset + 1 :].sum()),
    ]


def normalize(probabilities, minimum: int, position: int = 1) -> list[float]:
    """Remove impossible classes and normalize with a safe exact fallback."""
    p = np.maximum(np.asarray(probabilities, dtype=float), 0)
    p[np.asarray(position_baseline(minimum, position)) == 0] = 0
    if not np.isfinite(p).all() or p.sum() <= 0:
        return position_baseline(minimum, position)
    return (p / p.sum()).tolist()


def direction(previous: int, current: int) -> int:
    return 0 if current < previous else 1 if current == previous else 2


def features(history: list[int]) -> list[float]:
    """Features available immediately after the final observed draw."""
    return [
        *map(float, history[-5:]),
        *[float(direction(a, b)) for a, b in zip(history[-4:-1], history[-3:])],
        *[
            float(value)
            for window in (10, 30)
            for value in (np.mean(history[-window:]), np.std(history[-window:]))
        ],
    ]


def select_model(losses: dict[str, list[float]]) -> tuple[str, dict]:
    """Paired moving-block bootstrap and Holm step-down evidence gate."""
    n = len(losses["baseline"])
    evidence = {
        "scoredCount": n,
        "eligible": n >= PROTOCOL["minimumScores"],
        "candidates": {},
    }
    if not evidence["eligible"]:
        return "baseline", evidence
    rng = np.random.default_rng(PROTOCOL["seed"])
    length = PROTOCOL["bootstrapBlockLength"]
    starts = rng.integers(
        0, n - length + 1, size=(PROTOCOL["bootstrapReplicates"], math.ceil(n / length))
    )
    indices = (starts[..., None] + np.arange(length)).reshape(len(starts), -1)[:, :n]
    p_values = {}
    for name in CANDIDATES:
        differences = np.asarray(losses["baseline"]) - np.asarray(losses[name])
        boot = differences[indices].mean(axis=1)
        # Centered bootstrap null distribution for a one-sided improvement test.
        p_value = float(
            (1 + np.count_nonzero(boot - differences.mean() >= differences.mean()))
            / (len(boot) + 1)
        )
        p_values[name] = p_value
        evidence["candidates"][name] = {
            "improvement": float(differences.mean()),
            "lower95": float(np.quantile(boot, 0.025)),
            "pValue": p_value,
            "holmPValue": 1.0,
            "supported": False,
        }
    adjusted = 0.0
    for rank, name in enumerate(sorted(CANDIDATES, key=p_values.get)):
        adjusted = max(adjusted, min(1.0, (len(CANDIDATES) - rank) * p_values[name]))
        item = evidence["candidates"][name]
        item["holmPValue"] = adjusted
        item["supported"] = item["lower95"] > 0 and adjusted <= 0.05
    best = min(CANDIDATES, key=lambda name: np.mean(losses[name]))
    return (best if evidence["candidates"][best]["supported"] else "baseline"), evidence


def metrics(records: list[dict], model: str) -> dict:
    confusion = np.zeros((3, 3), dtype=int)
    losses, briers = [], []
    for row in records:
        p = row["models"][model]
        actual = DIRECTIONS.index(row["actualDirection"])
        confusion[actual, int(np.argmax(p))] += 1
        losses.append(-math.log(max(p[actual], 1e-15)))
        briers.append(sum((value - int(i == actual)) ** 2 for i, value in enumerate(p)))
    return {
        "count": len(records),
        "logLoss": float(np.mean(losses)) if losses else None,
        "brier": float(np.mean(briers)) if briers else None,
        "accuracy": float(np.trace(confusion) / len(records)) if records else None,
        "confusionMatrix": confusion.tolist(),
        "recall": [
            float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None
            for i in range(3)
        ],
    }


def analyze_position(draws: Draws, position: int) -> dict:
    """Forecast before observing each target; never fit on future observations."""
    prior = position_distribution(position)
    entries = draws.draws
    dates = [draw.date for draw in entries if draw.date is not None]
    if any(a > b for a, b in zip(dates, dates[1:])):
        raise ValueError("Number Directions requires chronological draw dates")
    minima = [draw.balls[position - 1].value for draw in entries]
    identity = [[draw.date, *[b.value for b in draw.balls]] for draw in entries]
    fingerprint = hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode()
    ).hexdigest()
    records = []
    losses = {name: [] for name in ("baseline", *CANDIDATES)}
    active, evidence = select_model(losses)
    counts = np.zeros(44)
    transitions = np.zeros((44, 44))
    training_x, training_y = [], []
    model = None
    forecast = None
    for target in range(1, len(minima) + 1):
        previous = minima[target - 1]
        counts[previous - position] += 1
        if target >= 2:
            transitions[minima[target - 2] - position, previous - position] += 1
        if target >= 31:
            training_x.append(features(minima[: target - 1]))
            training_y.append(direction(minima[target - 2], previous))
        if target < PROTOCOL["warmup"]:
            continue
        if (target - PROTOCOL["warmup"]) % PROTOCOL["refitInterval"] == 0:
            if len(set(training_y)) >= 2:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=PROTOCOL["logisticC"],
                        max_iter=1000,
                        random_state=PROTOCOL["seed"],
                    ),
                )
                model.fit(training_x, training_y)
            active, evidence = select_model(losses)
        strength = PROTOCOL["priorStrength"]
        historical = (counts + strength * prior) / (counts.sum() + strength)
        transition = (transitions[previous - position] + strength * historical) / (
            transitions[previous - position].sum() + strength
        )

        def aggregate(distribution):
            return normalize(
                [
                    distribution[: previous - position].sum(),
                    distribution[previous - position],
                    distribution[previous - position + 1 :].sum(),
                ],
                previous,
                position,
            )

        probabilities = {
            "baseline": position_baseline(previous, position),
            "historical": aggregate(historical),
            "transition": aggregate(transition),
        }
        logistic = np.array(probabilities["baseline"])
        if model is not None:
            logistic = np.zeros(3)
            logistic[model.classes_.astype(int)] = model.predict_proba(
                [features(minima[:target])]
            )[0]
        probabilities["logistic"] = normalize(logistic, previous, position)
        probabilities["gated"] = probabilities[active].copy()
        forecast = {
            "referenceDrawNumber": target,
            "referenceDate": entries[target - 1].date,
            "referenceValue": previous,
            "targetDrawNumber": target + 1,
            "probabilities": probabilities["gated"],
            "baselineProbabilities": probabilities["baseline"],
            "predictedDirection": DIRECTIONS[int(np.argmax(probabilities["gated"]))],
            "activeModel": active,
            "activationEvidence": evidence,
            "models": probabilities,
        }
        if target < len(minima):
            actual = direction(previous, minima[target])
            row = {
                **forecast,
                "actualValue": minima[target],
                "actualDate": entries[target].date,
                "actualDirection": DIRECTIONS[actual],
            }
            records.append(row)
            for name in losses:
                losses[name].append(-math.log(max(probabilities[name][actual], 1e-15)))
    if minima and forecast is None:
        p = position_baseline(minima[-1], position)
        forecast = {
            "referenceDrawNumber": len(minima),
            "referenceDate": entries[-1].date,
            "referenceValue": minima[-1],
            "targetDrawNumber": len(minima) + 1,
            "probabilities": p,
            "baselineProbabilities": p,
            "predictedDirection": DIRECTIONS[int(np.argmax(p))],
            "activeModel": "baseline",
            "activationEvidence": evidence,
            "models": {},
        }
    blocks = [list(block) for block in np.array_split(np.arange(len(records)), 5)]
    return {
        "position": position,
        "protocol": dict(PROTOCOL),
        "datasetFingerprint": fingerprint,
        "forecast": forecast,
        "evaluation": {name: metrics(records, name) for name in MODELS},
        "blocks": [
            {name: metrics([records[i] for i in block], name) for name in MODELS}
            for block in blocks
        ],
        "history": records,
    }


def analyze_direction(draws: Draws) -> dict:
    """Compatibility wrapper retaining the original position-one field names."""
    result = analyze_position(draws, 1)
    for row in [result["forecast"], *result["history"]]:
        if row is not None:
            row["referenceMinimum"] = row.pop("referenceValue")
            if "actualValue" in row:
                row["actualMinimum"] = row.pop("actualValue")
    return result


def analyze_number_directions(draws: Draws) -> dict:
    reports = [analyze_position(draws, position) for position in range(1, 7)]
    return {
        "schemaVersion": 2,
        "protocol": dict(PROTOCOL),
        "datasetFingerprint": reports[0]["datasetFingerprint"],
        "positions": reports,
    }


def cached_number_directions(
    draws: Draws, cache_dir: Path | None = None, refresh: bool = False
) -> dict:
    """Cache the expanded report independently of legacy single-position payloads."""
    identity = {
        "protocol": PROTOCOL,
        "schemaVersion": 2,
        "draws": [[d.date, *[b.value for b in d.balls]] for d in draws.draws],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"number-directions-{key}.json" if cache_dir else None
    if path and path.is_file() and not refresh:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if (
                result["schemaVersion"] == 2
                and result["protocol"] == PROTOCOL
                and [p["position"] for p in result["positions"]] == list(range(1, 7))
            ):
                return result
        except ValueError, KeyError, TypeError, OSError:
            pass
    result = analyze_number_directions(draws)
    if path:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        except OSError:
            pass
    return result


def cached_direction(
    draws: Draws, cache_dir: Path | None = None, refresh: bool = False
) -> dict:
    """Content-addressed cache, invalidated by any draw or protocol change."""
    identity = {
        "protocol": PROTOCOL,
        "draws": [[d.date, *[b.value for b in d.balls]] for d in draws.draws],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"first-number-direction-{key}.json" if cache_dir else None
    if path and path.is_file() and not refresh:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value["protocol"] == PROTOCOL
                and "forecast" in value
                and "history" in value
            ):
                return value
        except ValueError, KeyError, TypeError, OSError:
            pass
    result = analyze_direction(draws)
    if path:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        except OSError:
            pass  # Cache availability must not prevent analysis.
    return result
