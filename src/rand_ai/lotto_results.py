"""Import historical lottery results from YAML into Draws datasets."""

from pathlib import Path
from typing import TypedDict, cast

import yaml

from rand_ai.draw import Draw
from rand_ai.draws import Draws


class _YamlDraw(TypedDict):
    """Describe the fields used from one YAML draw entry."""

    numbers: list[int]


class _YamlResults(TypedDict):
    """Describe the fields used from the YAML results mapping."""

    draws: list[_YamlDraw]


class _YamlDocument(TypedDict):
    """Describe the top-level YAML document."""

    lotto_results: _YamlResults


def load_lotto_results_yaml(file_path: str | Path) -> Draws:
    """Load every YAML number set into a new Draws collection."""
    with Path(file_path).open(encoding="utf-8") as yaml_file:
        document = cast(_YamlDocument, yaml.safe_load(yaml_file))

    draws = Draws()
    for yaml_draw in document["lotto_results"]["draws"]:
        draws.add(Draw(*yaml_draw["numbers"]))
    return draws


def create_lotto_results_pickle(
    yaml_path: str | Path, pickle_path: str | Path
) -> Draws:
    """Create and persist a Draws collection from a lottery-results YAML file."""
    draws = load_lotto_results_yaml(yaml_path)
    draws.save_pickle(pickle_path)
    return draws
