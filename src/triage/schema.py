"""Data types for the triage feature: what goes in, and what comes out."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

IssueType = Literal["bug", "enhancement", "documentation", "question", "other"]
Severity = Literal["critical", "high", "medium", "low"]

# Below this model-reported confidence, a record is flagged for a human.
REVIEW_THRESHOLD = 0.6

# Issue bodies can be huge (logs, stack traces). Only the head is sent to the model.
MAX_BODY_CHARS = 6000


class Issue(BaseModel):
    """A GitHub issue as stored in data/snapshot/*.json.

    Only the fields triage needs are modelled; gold labels and GitHub labels
    are deliberately ignored so the model never sees the answer.
    """

    repo: str
    number: str
    title: str
    body: str = ""
    url: str = ""

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> Issue:
        return cls(
            repo=str(raw.get("repo", "")),
            number=str(raw.get("number", "")),
            title=str(raw.get("title") or ""),
            body=str(raw.get("body") or ""),
            url=str(raw.get("url") or ""),
        )


class TriageRecord(BaseModel):
    """One structured triage record: the input to a later routing step."""

    repo: str
    number: str
    url: str
    issue_type: IssueType
    component: str | None = Field(
        description="One of the repo's component labels, or None if none applies."
    )
    severity: Severity
    has_reproduction_steps: bool
    summary: str = Field(description="One-sentence neutral summary of the issue.")
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
    review_reasons: list[str] = Field(default_factory=list[str])
