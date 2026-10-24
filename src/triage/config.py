"""Per-repo component vocabulary, read from config/repos.toml."""

from __future__ import annotations

import tomllib
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPOS_TOML = REPO_ROOT / "config" / "repos.toml"


@cache
def component_vocabulary() -> dict[str, tuple[str, ...]]:
    """Map repo key -> allowed component names (may be empty, e.g. textual)."""
    with REPOS_TOML.open("rb") as f:
        data = tomllib.load(f)
    vocab: dict[str, tuple[str, ...]] = {}
    for repo in data.get("repos", []):
        components = repo.get("component_labels", {})
        vocab[str(repo["key"])] = tuple(str(k) for k in components)
    return vocab


def components_for(repo: str) -> tuple[str, ...]:
    return component_vocabulary().get(repo, ())
