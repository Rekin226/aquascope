#!/usr/bin/env python3
"""CI check requiring a CHANGELOG.md update on PRs unless labeled 'no-changelog'."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

OPT_OUT_LABEL = "no-changelog"
CHANGELOG_FILENAME = "CHANGELOG.md"


def get_pr_labels(event_path: str | Path | None = None) -> list[str]:
    """Extract PR labels from GitHub event JSON payload."""
    if event_path is None:
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")

    if not event_path or not Path(event_path).exists():
        return []

    try:
        with open(event_path, encoding="utf-8") as f:
            payload = json.load(f)
        pr_labels = payload.get("pull_request", {}).get("labels", [])
        return [lbl.get("name", "") for lbl in pr_labels if isinstance(lbl, dict)]
    except Exception:
        return []


def get_changed_files(base_ref: str | None = None) -> list[str]:
    """Get list of changed files in git diff relative to base_ref."""
    if not base_ref:
        base_ref = os.environ.get("BASE_REF") or os.environ.get("GITHUB_BASE_REF") or "main"

    commands = [
        ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"],
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        ["git", "diff", "--name-only", "HEAD~1"],
    ]

    for cmd in commands:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if files:
                return files
        except Exception:
            continue

    return []


def check_changelog(
    event_name: str | None = None,
    event_path: str | Path | None = None,
    changed_files: list[str] | set[str] | None = None,
    base_ref: str | None = None,
) -> tuple[bool, str]:
    """Verify that a pull request updates CHANGELOG.md or carries 'no-changelog'.

    Returns (success: bool, message: str).
    """
    if event_name is None:
        event_name = os.environ.get("GITHUB_EVENT_NAME", "")

    if event_name and event_name != "pull_request":
        return True, f"Skipping CHANGELOG check for non-pull_request event ({event_name!r})."

    labels = get_pr_labels(event_path)
    if OPT_OUT_LABEL in labels:
        return True, f"Skipping CHANGELOG check because '{OPT_OUT_LABEL}' label is set on PR."

    if changed_files is None:
        changed_files = get_changed_files(base_ref)

    # Tighten to exact root CHANGELOG.md match
    normalized_paths = {Path(f).as_posix().lstrip("./") for f in changed_files}
    if CHANGELOG_FILENAME in normalized_paths:
        return True, "CHANGELOG.md was updated in this pull request."

    err_msg = (
        "::error::CHANGELOG.md entry required!\n"
        "This pull request does not modify CHANGELOG.md and does not have the 'no-changelog' label.\n\n"
        "Please update CHANGELOG.md to describe your changes.\n"
        "If this PR is a CI-only, refactor, or dependency update that does not require release notes, "
        "add the 'no-changelog' label to the PR to opt out."
    )
    return False, err_msg


def main() -> None:
    success, message = check_changelog()
    if success:
        print(f"✅ {message}")
        sys.exit(0)
    else:
        print(f"❌ {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
