"""Offline tests: the model is replaced by a fake predictor, so no API calls."""

from __future__ import annotations

import json
import os
from typing import Any

import dspy
import pytest

from triage import cli
from triage.cli import load_split
from triage.extract import configure_lm, triage
from triage.schema import Issue, TriageRecord


class FakePredictor:
    """Stands in for dspy.Predict: records its inputs, returns fixed outputs."""

    def __init__(self, outputs: dict[str, Any] | None = None, raise_error: bool = False) -> None:
        self.outputs = outputs or {}
        self.raise_error = raise_error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dspy.Prediction:
        self.calls.append(kwargs)
        if self.raise_error:
            raise RuntimeError("boom")
        return dspy.Prediction(**self.outputs)


GOOD: dict[str, Any] = {
    "issue_type": "bug",
    "component": "middleware",
    "severity": "high",
    "has_reproduction_steps": True,
    "summary": "CORS middleware drops headers on preflight.",
    "confidence": 0.9,
}


def make_issue(repo: str = "hono", body: str = "steps...") -> Issue:
    return Issue(repo=repo, number="1", title="CORS broken", body=body)


def test_valid_output_becomes_record() -> None:
    record = triage(make_issue(), FakePredictor(GOOD))
    assert record.issue_type == "bug"
    assert record.component == "middleware"
    assert record.severity == "high"
    assert record.has_reproduction_steps is True
    assert record.needs_human_review is False
    assert record.review_reasons == []


def test_component_outside_repo_vocabulary_is_rejected() -> None:
    # "middleware" is a hono component, not a litestar one.
    record = triage(make_issue(repo="litestar"), FakePredictor(GOOD))
    assert record.component is None
    assert record.needs_human_review is True


def test_textual_has_no_components() -> None:
    record = triage(make_issue(repo="textual"), FakePredictor({**GOOD, "component": "none"}))
    assert record.component is None
    assert record.needs_human_review is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"issue_type": "Feature"},
        {"severity": "urgent!!"},
        {"confidence": "not a number"},
        {"confidence": 0.2},
        {"issue_type": "other"},
    ],
)
def test_bad_or_uncertain_output_is_flagged(overrides: dict[str, Any]) -> None:
    record = triage(make_issue(), FakePredictor({**GOOD, **overrides}))
    assert record.needs_human_review is True
    assert record.review_reasons


def test_normalises_case_and_clamps_confidence() -> None:
    record = triage(
        make_issue(), FakePredictor({**GOOD, "issue_type": " BUG ", "confidence": 7})
    )
    assert record.issue_type == "bug"
    assert record.confidence == 1.0


def test_model_failure_degrades_to_review() -> None:
    record = triage(make_issue(), FakePredictor(raise_error=True))
    assert record.issue_type == "other"
    assert record.confidence == 0.0
    assert record.needs_human_review is True


def test_long_body_is_truncated() -> None:
    fake = FakePredictor(GOOD)
    triage(make_issue(body="x" * 100_000), fake)
    assert len(fake.calls[0]["body"]) <= 6000


def test_gold_labels_never_reach_the_model() -> None:
    raw = load_split("dev")[0]
    fake = FakePredictor(GOOD)
    triage(Issue.from_raw(raw), fake)
    assert set(fake.calls[0]) == {"repo", "allowed_components", "title", "body"}


def test_sealed_test_split_is_refused() -> None:
    with pytest.raises(SystemExit):
        load_split("test")


def test_cli_prints_one_valid_record_for_dev_issue(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI output for hono #3994 is one JSON line in the TriageRecord format."""
    fake = FakePredictor({**GOOD, "issue_type": "bug", "component": "none"})

    def no_lm(model: str | None = None) -> None:
        return None

    def fake_triage(issue: Issue) -> TriageRecord:
        return triage(issue, fake)

    monkeypatch.setattr(cli, "configure_lm", no_lm)
    monkeypatch.setattr(cli, "triage", fake_triage)

    assert cli.main(["--split", "dev", "--number", "3994", "--repo", "hono"]) == 0

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    out = json.loads(lines[0])

    assert list(out) == [
        "repo", "number", "url", "issue_type", "component", "severity",
        "has_reproduction_steps", "summary", "confidence",
        "needs_human_review", "review_reasons",
    ]
    assert out["repo"] == "hono"
    assert out["number"] == "3994"
    assert out["url"] == "https://github.com/honojs/hono/issues/3994"
    assert out["issue_type"] in {"bug", "enhancement", "documentation", "question", "other"}
    assert out["severity"] in {"critical", "high", "medium", "low"}
    assert out["component"] is None or out["component"] in {"middleware", "testing"}
    assert isinstance(out["has_reproduction_steps"], bool)
    assert isinstance(out["summary"], str) and out["summary"]
    assert 0.0 <= out["confidence"] <= 1.0
    assert isinstance(out["needs_human_review"], bool)
    assert isinstance(out["review_reasons"], list)
    # The CLI passed this issue's own body to the model.
    assert "TypeError" in fake.calls[0]["body"]
    assert TriageRecord.model_validate(out).number == "3994"


@pytest.mark.skipif(
    os.environ.get("TRIAGE_LIVE_TESTS") != "1",
    reason="live model test; set TRIAGE_LIVE_TESTS=1 to run (uses API quota)",
)
def test_live_model_returns_valid_record_for_dev_issue() -> None:
    """Calls the real LLM_MODEL from .env on hono #3994 and checks the record."""
    configure_lm()
    raw = next(
        i for i in load_split("dev") if i["repo"] == "hono" and str(i["number"]) == "3994"
    )

    record = triage(Issue.from_raw(raw))

    failure = next((r for r in record.review_reasons if r.startswith("model call failed")), None)
    if failure is not None and "RateLimit" in failure:
        pytest.skip(f"provider quota exhausted: {failure}")
    assert failure is None, failure
    # The model's own output must have been valid, not coerced to a default.
    assert not any(r.startswith("invalid") for r in record.review_reasons)
    assert "component outside repo vocabulary" not in record.review_reasons

    assert record.number == "3994"
    assert record.issue_type in {"bug", "enhancement", "documentation", "question", "other"}
    assert record.severity in {"critical", "high", "medium", "low"}
    assert record.component in {None, "middleware", "testing"}
    assert record.summary.strip()
    assert 0.0 <= record.confidence <= 1.0
