"""Sphinx configuration for the curated Rand AI application guide."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

project = "Rand AI"
author = "Rand AI"
copyright = "2026, Rand AI"

try:
    release = package_version("rand-ai")
except PackageNotFoundError:
    release = "development"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
]

source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns: list[str] = []
nitpicky = True

autosectionlabel_prefix_document = True
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_title = f"Rand AI {release} application guide"
html_show_sourcelink = False
html_theme_options = {
    "collapse_navigation": False,
    "includehidden": True,
    "navigation_depth": 4,
    "sticky_navigation": True,
    "titles_only": False,
}
