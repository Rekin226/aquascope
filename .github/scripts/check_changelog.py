#!/usr/bin/env python3
"""CI check requiring a CHANGELOG.md entry under ## [Unreleased] on PRs unless labeled 'no-changelog'.

On a pull_request event the workflow checks out GitHub's merge ref, the PR merged
with the current base, and the diff is taken from the base to that merge. So the
check sees each added CHANGELOG line where it will land on main. A branch that
waited across a release keeps its line where it was, which after the release roll
is inside the released section, and the check fails until the line is moved.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

OPT_OUT_LABEL = "no-changelog"
CHANGELOG_FILENAME = "CHANGELOG.md"
UNRELEASED_HEADING = "## [Unreleased]"

_UNRELEASED_RE = re.compile(r"^##\s+\[?unreleased\]?\s*$", re.IGNORECASE)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
# Enough context for git to print the whole CHANGELOG as one hunk, so every
# added line can be placed under its section heading.
_FULL_CONTEXT = "--unified=1000000"
_MAX_LISTED = 10


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


def _resolve_base_ref(base_ref: str | None) -> str:
    return base_ref or os.environ.get("BASE_REF") or os.environ.get("GITHUB_BASE_REF") or "main"


def _git_diff(options: list[str], base_ref: str | None, paths: list[str] | None = None) -> str:
    """Return the output of the first `git diff` against the base that runs and prints something."""
    base_ref = _resolve_base_ref(base_ref)
    for rev in (f"origin/{base_ref}...HEAD", f"{base_ref}...HEAD", "HEAD~1"):
        cmd = ["git", "diff", *options, rev]
        if paths:
            cmd += ["--", *paths]
        try:
            res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", check=True)
        except Exception:
            continue
        if res.stdout.strip():
            return res.stdout
    return ""


def get_changed_files(base_ref: str | None = None) -> list[str]:
    """Get list of changed files in git diff relative to base_ref."""
    out = _git_diff(["--name-only"], base_ref)
    return [line.strip() for line in out.splitlines() if line.strip()]


def get_changelog_diff(base_ref: str | None = None) -> str:
    """Get the unified diff of CHANGELOG.md relative to base_ref, with the whole file as context."""
    return _git_diff(["--no-color", "--no-ext-diff", _FULL_CONTEXT], base_ref, [CHANGELOG_FILENAME])


def find_misplaced_lines(changelog_diff: str) -> list[tuple[int, str | None, str]]:
    """Return (line number, section heading, text) for each added line outside ## [Unreleased].

    ``changelog_diff`` is a unified diff of CHANGELOG.md with the whole file as
    context, so its context and added lines are the new file in order. Blank lines
    and ``## `` headings are not entries and are not checked. A section whose
    heading the diff itself adds (a release roll) belongs to the PR and may hold
    its lines. The heading is None for a line above the first ``## `` heading.
    """
    misplaced: list[tuple[int, str | None, str]] = []
    section: str | None = None
    section_is_new = False
    lineno = 0
    in_hunk = False
    for raw in changelog_diff.splitlines():
        if raw.startswith("diff --git"):
            in_hunk = False
            continue
        match = _HUNK_RE.match(raw)
        if match:
            lineno = int(match.group(1)) - 1
            in_hunk = True
            continue
        if not in_hunk or raw.startswith("\\"):
            continue
        tag, text = (raw[:1], raw[1:]) if raw else (" ", "")
        if tag not in (" ", "+"):
            continue
        lineno += 1
        added = tag == "+"
        if text.startswith("## "):
            section = text.strip()
            section_is_new = added
            continue
        if not added or not text.strip() or section_is_new:
            continue
        if section is not None and _UNRELEASED_RE.match(section):
            continue
        misplaced.append((lineno, section, text))
    return misplaced


def _section_name(section: str | None) -> str:
    return section if section is not None else f"the top of the file, above {UNRELEASED_HEADING}"


def _misplaced_message(misplaced: list[tuple[int, str | None, str]], base_ref: str | None) -> str:
    base = _resolve_base_ref(base_ref)
    sections = ", ".join(dict.fromkeys(_section_name(s) for _, s, _ in misplaced))
    count = len(misplaced)
    noun = "line" if count == 1 else "lines"
    lines = [
        f"::error::CHANGELOG.md {noun} under {sections}, not {UNRELEASED_HEADING}. "
        f"Merge {base} and move the {noun} under {UNRELEASED_HEADING}.",
        f"Merged with {base}, this pull request adds {count} CHANGELOG.md {noun} outside the "
        f"{UNRELEASED_HEADING} section:",
    ]
    for lineno, section, text in misplaced[:_MAX_LISTED]:
        shown = text.strip()
        if len(shown) > 100:
            shown = shown[:97] + "..."
        lines.append(f"  line {lineno}, under {_section_name(section)}: {shown}")
    if count > _MAX_LISTED:
        lines.append(f"  and {count - _MAX_LISTED} more")
    lines += [
        "",
        f"Usually this means {base} has had a release since your branch started: the release turned "
        f"{UNRELEASED_HEADING} into a version heading, and git merges your line into that released "
        "section, so the change would be documented as part of a release it did not ship in.",
        "",
        f"To fix it, merge {base} into your branch (git fetch origin && git merge origin/{base}), "
        f"move the {noun} under {UNRELEASED_HEADING} at the top of CHANGELOG.md, and push.",
        "If this pull request edits a released section on purpose (a correction to an old entry), "
        f"add the '{OPT_OUT_LABEL}' label to skip this check.",
    ]
    return "\n".join(lines)


def check_changelog(
    event_name: str | None = None,
    event_path: str | Path | None = None,
    changed_files: list[str] | set[str] | None = None,
    base_ref: str | None = None,
    changelog_diff: str | None = None,
) -> tuple[bool, str]:
    """Verify that a pull request adds CHANGELOG.md lines under ## [Unreleased] or carries 'no-changelog'.

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
    if CHANGELOG_FILENAME not in normalized_paths:
        err_msg = (
            "::error::CHANGELOG.md entry required!\n"
            "This pull request does not modify CHANGELOG.md and does not have the 'no-changelog' label.\n\n"
            "Please update CHANGELOG.md to describe your changes.\n"
            "If this PR is a CI-only, refactor, or dependency update that does not require release notes, "
            "add the 'no-changelog' label to the PR to opt out."
        )
        return False, err_msg

    if changelog_diff is None:
        changelog_diff = get_changelog_diff(base_ref)
    misplaced = find_misplaced_lines(changelog_diff)
    if misplaced:
        return False, _misplaced_message(misplaced, base_ref)

    return True, f"CHANGELOG.md was updated in this pull request, under {UNRELEASED_HEADING}."


def main() -> None:
    success, message = check_changelog()
    if success:
        print(f"✅ {message}")
        sys.exit(0)
    else:
        # The message opens with a ::error:: workflow command, which GitHub only
        # reads at the start of a line, so the marker goes on its own line.
        print("❌ CHANGELOG check failed.", file=sys.stderr)
        print(message, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
