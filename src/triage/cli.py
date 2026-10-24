"""Command line: take an issue, print a triage record as JSON.

    uv run python -m triage.cli --split dev --number 3891 --repo hono
    uv run python -m triage.cli --split dev --index 0
    uv run python -m triage.cli --file issue.json
    uv run python -m triage.cli --split dev --limit 20 > out.jsonl

The sealed test split is refused on purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from triage.config import REPO_ROOT
from triage.extract import configure_lm, triage
from triage.schema import Issue

SNAPSHOT = REPO_ROOT / "data" / "snapshot"
ALLOWED_SPLITS = ("train", "dev")  # test.json is sealed


def load_split(split: str) -> list[dict[str, Any]]:
    if split not in ALLOWED_SPLITS:
        raise SystemExit(f"split must be one of {ALLOWED_SPLITS} (test is sealed)")
    with (SNAPSHOT / f"{split}.json").open(encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def select_issues(args: argparse.Namespace) -> list[Issue]:
    if args.file:
        raw: dict[str, Any] = json.loads(Path(args.file).read_text(encoding="utf-8"))
        return [Issue.from_raw(raw)]

    issues = load_split(args.split)
    if args.number is not None:
        matches: list[dict[str, Any]] = [
            i
            for i in issues
            if str(i.get("number")) == args.number
            and (args.repo is None or i.get("repo") == args.repo)
        ]
        if not matches:
            raise SystemExit(f"issue #{args.number} not found in {args.split}")
        if len(matches) > 1:
            raise SystemExit(f"#{args.number} is ambiguous; pass --repo")
        return [Issue.from_raw(matches[0])]
    if args.index is not None:
        return [Issue.from_raw(issues[int(args.index)])]
    if args.repo is not None:
        issues = [i for i in issues if i.get("repo") == args.repo]
    return [Issue.from_raw(i) for i in issues[: int(args.limit)]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Triage GitHub issues with an LLM.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", help="JSON file holding one issue object.")
    source.add_argument("--number", help="Issue number within --split.")
    source.add_argument("--index", type=int, help="Position within --split.")
    source.add_argument("--limit", type=int, default=1, help="First N of --split.")
    parser.add_argument("--split", default="dev", help="train or dev (default dev).")
    parser.add_argument("--repo", help="litestar/hono/textual: filters --limit, disambiguates --number.")
    parser.add_argument("--model", help="Override LLM_MODEL from .env.")
    args = parser.parse_args(argv)

    if sys.stdout.encoding.lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    configure_lm(args.model)
    for issue in select_issues(args):
        record = triage(issue)
        print(record.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
