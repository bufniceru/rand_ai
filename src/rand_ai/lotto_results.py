"""Import historical lottery results from YAML into Draws datasets."""

from datetime import date as calendar_date
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import yaml

from rand_ai.draw import Draw
from rand_ai.draws import Draws


class _YamlDraw(TypedDict):
    """Describe the fields used from one YAML draw entry."""

    date: str
    numbers: list[int]


class _YamlResults(TypedDict):
    """Describe the fields used from the YAML results mapping."""

    draws: list[_YamlDraw]
    total_draws: NotRequired[int]
    first_draw: NotRequired[str]
    last_draw: NotRequired[str]


class _YamlDocument(TypedDict):
    """Describe the top-level YAML document."""

    lotto_results: _YamlResults


def load_lotto_results_yaml(file_path: str | Path) -> Draws:
    """Load every YAML number set into a new Draws collection."""
    with Path(file_path).open(encoding="utf-8") as yaml_file:
        document = cast(_YamlDocument, yaml.safe_load(yaml_file))

    draws = Draws()
    for yaml_draw in document["lotto_results"]["draws"]:
        draws.add(Draw(*yaml_draw["numbers"], date=yaml_draw["date"]))
    return draws


def create_lotto_results_pickle(
    yaml_path: str | Path, pickle_path: str | Path
) -> Draws:
    """Create and persist a Draws collection from a lottery-results YAML file."""
    draws = load_lotto_results_yaml(yaml_path)
    draws.save_pickle(pickle_path)
    return draws


def resolve_lotto_results_yaml(pickle_path: str | Path) -> Path:
    """Return the YAML source paired with a managed lottery-results pickle."""
    source = Path(pickle_path).resolve()
    yaml_paths = (source.with_suffix(".yaml"), source.with_suffix(".yml"))
    for yaml_path in yaml_paths:
        if yaml_path.is_file():
            return yaml_path
    raise FileNotFoundError(
        f"No paired YAML file was found at {yaml_paths[0]} or {yaml_paths[1]}. "
        "Draw editing is available for YAML-managed pickle datasets."
    )


def lotto_results_editor_payload(pickle_path: str | Path) -> dict[str, Any]:
    """Return all YAML draws for the 7x7 Electron draw editor."""
    pickle_source = Path(pickle_path).resolve()
    yaml_path = resolve_lotto_results_yaml(pickle_source)
    with yaml_path.open(encoding="utf-8") as yaml_file:
        document = cast(_YamlDocument, yaml.safe_load(yaml_file))
    entries = document["lotto_results"]["draws"]
    return {
        "picklePath": str(pickle_source),
        "yamlPath": str(yaml_path),
        "draws": [
            {
                "index": index,
                "date": entry["date"],
                "numbers": sorted(entry["numbers"]),
            }
            for index, entry in enumerate(entries)
        ],
    }


def upsert_lotto_result(
    yaml_path: str | Path,
    pickle_path: str | Path,
    *,
    draw_date: str,
    numbers: list[int],
    original_date: str | None = None,
) -> Draws:
    """Update YAML first, then regenerate its equivalent pickle."""
    normalized_date = calendar_date.fromisoformat(draw_date).isoformat()
    validated_draw = Draw(*numbers, date=normalized_date)
    normalized_numbers = [ball.value for ball in validated_draw.balls]
    source = Path(yaml_path)
    with source.open(encoding="utf-8") as yaml_file:
        raw_document = yaml.safe_load(yaml_file)
    if not isinstance(raw_document, dict):
        raise ValueError("YAML root must be a mapping")
    results = raw_document.get("lotto_results")
    if not isinstance(results, dict) or not isinstance(results.get("draws"), list):
        raise ValueError("YAML must contain lotto_results.draws")

    entries = cast(list[dict[str, Any]], results["draws"])
    match_date = original_date or normalized_date
    match = next(
        (entry for entry in entries if entry.get("date") == match_date),
        None,
    )
    if match is None and original_date is not None:
        raise ValueError(f"Draw dated {original_date} no longer exists")
    duplicate = next(
        (
            entry
            for entry in entries
            if entry is not match and entry.get("date") == normalized_date
        ),
        None,
    )
    if duplicate is not None:
        raise ValueError(f"A draw already exists on {normalized_date}")
    if match is None:
        entries.append({"date": normalized_date, "numbers": normalized_numbers})
    else:
        match["date"] = normalized_date
        match["numbers"] = normalized_numbers
    entries.sort(key=lambda entry: str(entry["date"]))
    results["total_draws"] = len(entries)
    results["first_draw"] = entries[0]["date"] if entries else None
    results["last_draw"] = entries[-1]["date"] if entries else None

    source.write_text(
        yaml.safe_dump(raw_document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return create_lotto_results_pickle(source, pickle_path)
