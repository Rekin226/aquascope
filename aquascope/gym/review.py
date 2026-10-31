"""Validate and look up explicit decisions about benchmark decline reasons.

Records are JSON objects with ``case_id``, ``reason_sha256``, ``verdict``,
``explanation`` and ``rubric_version``. The digest binds a decision to the
exact UTF-8 bytes of the candidate reason; it is not proof of who reviewed it.
The caller is responsible for selecting a trusted review file and reviewer.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_FIELDS = frozenset({"case_id", "reason_sha256", "verdict", "explanation", "rubric_version"})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def reason_sha256(reason: str) -> str:
    """Digest the original reason exactly; do not normalize text or whitespace."""
    return hashlib.sha256(reason.encode("utf-8")).hexdigest()


def lookup_decline_review(case_id: str, reason: str | None, rubric_version: str,
                          records: Any) -> dict[str, str]:
    """Find an unambiguous decision, or return ``pending``.

    ``records`` is the decoded JSON array. Every entry must have the exact
    schema and valid values; malformed evidence invalidates the entire input.
    Identical duplicate entries are harmless, but differing entries for the
    same case/reason/rubric are ambiguous, even if their verdicts agree.
    """
    def pending(explanation: str) -> dict[str, str]:
        return {"verdict": "pending", "explanation": explanation}

    if not isinstance(case_id, str) or not case_id.strip():
        return pending("missing case id")
    if not isinstance(reason, str) or not reason.strip():
        return {"verdict": "invalid", "explanation": "decline has no reason"}
    if not isinstance(rubric_version, str) or not rubric_version.strip():
        return pending("missing rubric version")
    if not isinstance(records, list):
        return pending("reviews must be a JSON array")

    for item in records:
        if not isinstance(item, dict) or set(item) != _FIELDS:
            return pending("invalid review record schema")
        if any(not isinstance(item[key], str) or not item[key].strip() for key in _FIELDS):
            return pending("review fields must be nonempty strings")
        if not _DIGEST.fullmatch(item["reason_sha256"]) or item["verdict"] not in ("valid", "invalid"):
            return pending("invalid digest or verdict")

    try:
        digest = reason_sha256(reason)
    except UnicodeEncodeError:
        return pending("decline reason is not valid UTF-8 text")
    matches = [item for item in records if item["case_id"] == case_id and item["reason_sha256"] == digest
               and item["rubric_version"] == rubric_version]
    if not matches:
        return pending("no matching review")
    if any(item != matches[0] for item in matches[1:]):
        return pending("conflicting reviews")
    return {"verdict": matches[0]["verdict"], "explanation": matches[0]["explanation"]}
