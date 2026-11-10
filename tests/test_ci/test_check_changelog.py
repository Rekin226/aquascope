"""Unit tests for .github/scripts/check_changelog.py (Issue #144)."""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess

# Import the check_changelog script
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / ".github" / "scripts"))

from check_changelog import check_changelog, find_misplaced_lines, get_pr_labels  # noqa: E402

# main as it stands after a release: the release rolled ## [Unreleased] into
# ## [0.27.0] and opened a fresh, empty ## [Unreleased] above it.
CHANGELOG_ON_MAIN = """# Changelog

## [Unreleased]

### Fixed
- A fix already on main.

## [0.27.0] - 2026-10-12

### Added
- Shipped in 0.27.0.
- Also shipped in 0.27.0.

## [0.26.0] - 2026-10-09

### Added
- Shipped in 0.26.0.
"""

NEW_LINE = "- Quality flags from the EA API (#485).\n"


def _event(tmp_path, *labels):
    event_path = tmp_path / "event.json"
    payload = {"pull_request": {"labels": [{"name": name} for name in labels]}}
    event_path.write_text(json.dumps(payload), encoding="utf-8")
    return event_path


def _diff(before: str, after: str) -> str:
    """A unified diff with the whole file as context, as the guard asks git for."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            "a/CHANGELOG.md",
            "b/CHANGELOG.md",
            n=1_000_000,
        )
    )


def _insert_after(text: str, anchor: str, new: str) -> str:
    assert anchor in text
    return text.replace(anchor, anchor + new, 1)


class TestCheckChangelog:
    """Test suite for CHANGELOG.md CI enforcement script."""

    def test_non_pull_request_event_skips(self):
        success, msg = check_changelog(event_name="push")
        assert success is True
        assert "Skipping CHANGELOG check for non-pull_request event" in msg

    def test_pr_with_no_changelog_label_passes(self, tmp_path):
        payload = {
            "pull_request": {
                "labels": [{"name": "bug"}, {"name": "no-changelog"}]
            }
        }
        event_path = tmp_path / "event.json"
        event_path.write_text(json.dumps(payload), encoding="utf-8")

        success, msg = check_changelog(
            event_name="pull_request",
            event_path=event_path,
            changed_files=["aquascope/cli.py"],
        )
        assert success is True
        assert "no-changelog" in msg

    def test_pr_with_changelog_modified_passes(self, tmp_path):
        payload = {"pull_request": {"labels": [{"name": "enhancement"}]}}
        event_path = tmp_path / "event.json"
        event_path.write_text(json.dumps(payload), encoding="utf-8")
        after = _insert_after(CHANGELOG_ON_MAIN, "- A fix already on main.\n", NEW_LINE)

        success, msg = check_changelog(
            event_name="pull_request",
            event_path=event_path,
            changed_files=["aquascope/cli.py", "CHANGELOG.md"],
            changelog_diff=_diff(CHANGELOG_ON_MAIN, after),
        )
        assert success is True
        assert "CHANGELOG.md was updated" in msg

    def test_pr_missing_changelog_and_label_fails(self, tmp_path):
        payload = {"pull_request": {"labels": [{"name": "bug"}]}}
        event_path = tmp_path / "event.json"
        event_path.write_text(json.dumps(payload), encoding="utf-8")

        success, msg = check_changelog(
            event_name="pull_request",
            event_path=event_path,
            changed_files=["aquascope/cli.py", "tests/test_cli.py"],
        )
        assert success is False
        assert "::error::CHANGELOG.md entry required!" in msg
        assert "no-changelog" in msg

    def test_subpath_changelog_does_not_pass(self, tmp_path):
        payload = {"pull_request": {"labels": [{"name": "enhancement"}]}}
        event_path = tmp_path / "event.json"
        event_path.write_text(json.dumps(payload), encoding="utf-8")

        success, _ = check_changelog(
            event_name="pull_request",
            event_path=event_path,
            changed_files=["docs/API_CHANGELOG.md", "OLD_CHANGELOG.md"],
        )
        assert success is False

    def test_get_pr_labels_missing_file_returns_empty(self):
        labels = get_pr_labels(Path("/nonexistent/event.json"))
        assert labels == []


class TestChangelogPlacement:
    """Lines the PR adds must sit under ## [Unreleased] once merged with main."""

    def test_line_under_unreleased_passes(self, tmp_path):
        after = _insert_after(CHANGELOG_ON_MAIN, "- A fix already on main.\n", NEW_LINE)
        success, msg = check_changelog(
            event_name="pull_request",
            event_path=_event(tmp_path, "enhancement"),
            changed_files=["aquascope/collectors/uk_ea.py", "CHANGELOG.md"],
            changelog_diff=_diff(CHANGELOG_ON_MAIN, after),
        )
        assert success is True
        assert "## [Unreleased]" in msg

    def test_line_under_released_version_fails_naming_the_section(self, tmp_path):
        # What git makes of a branch cut before 0.27.0: its line stays where it
        # was, which is now inside the released section.
        after = _insert_after(CHANGELOG_ON_MAIN, "- Shipped in 0.27.0.\n", NEW_LINE)
        success, msg = check_changelog(
            event_name="pull_request",
            event_path=_event(tmp_path, "enhancement"),
            changed_files=["aquascope/collectors/uk_ea.py", "CHANGELOG.md"],
            base_ref="main",
            changelog_diff=_diff(CHANGELOG_ON_MAIN, after),
        )
        assert success is False
        assert msg.startswith("::error::")
        assert "under ## [0.27.0] - 2026-10-12" in msg
        assert "line 12, under ## [0.27.0] - 2026-10-12: - Quality flags from the EA API (#485)." in msg
        assert "merge main into your branch" in msg
        assert "move the line under ## [Unreleased]" in msg
        assert "no-changelog" in msg

    def test_no_changelog_label_skips_the_placement_check(self, tmp_path):
        after = _insert_after(CHANGELOG_ON_MAIN, "- Shipped in 0.27.0.\n", NEW_LINE)
        success, msg = check_changelog(
            event_name="pull_request",
            event_path=_event(tmp_path, "no-changelog"),
            changed_files=["CHANGELOG.md"],
            changelog_diff=_diff(CHANGELOG_ON_MAIN, after),
        )
        assert success is True
        assert "no-changelog" in msg

    def test_release_roll_may_edit_the_section_it_creates(self, tmp_path):
        before = CHANGELOG_ON_MAIN
        after = before.replace(
            "## [Unreleased]\n\n### Fixed\n- A fix already on main.\n",
            "## [Unreleased]\n\n## [0.28.0] - 2026-10-20\n\n### Fixed\n- A fix already on main, tidied.\n",
        )
        assert find_misplaced_lines(_diff(before, after)) == []

    def test_edit_to_an_older_release_is_flagged(self):
        after = CHANGELOG_ON_MAIN.replace("- Shipped in 0.26.0.", "- Shipped in 0.26.0, reworded.")
        misplaced = find_misplaced_lines(_diff(CHANGELOG_ON_MAIN, after))
        assert misplaced == [(17, "## [0.26.0] - 2026-10-09", "- Shipped in 0.26.0, reworded.")]

    def test_removed_and_blank_lines_are_not_entries(self):
        after = CHANGELOG_ON_MAIN.replace("- Also shipped in 0.27.0.\n", "\n")
        assert find_misplaced_lines(_diff(CHANGELOG_ON_MAIN, after)) == []


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
         "-c", "core.autocrlf=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_branch_cut_before_a_release_fails_on_the_merge_with_main(tmp_path, monkeypatch):
    """The #485 case end to end: git itself puts the line in the released section."""
    repo = tmp_path / "repo"
    repo.mkdir()
    before_release = CHANGELOG_ON_MAIN.replace(
        "## [Unreleased]\n\n### Fixed\n- A fix already on main.\n\n## [0.27.0] - 2026-10-12\n\n",
        "## [Unreleased]\n\n",
    )
    changelog = repo / "CHANGELOG.md"
    _git(repo, "init", "-q", "-b", "main")
    changelog.write_text(before_release, encoding="utf-8")
    _git(repo, "add", "CHANGELOG.md")
    _git(repo, "commit", "-q", "-m", "before 0.27.0")

    # The contributor's branch adds its line under ## [Unreleased], before the release.
    _git(repo, "checkout", "-q", "-b", "feature")
    changelog.write_text(_insert_after(before_release, "- Shipped in 0.27.0.\n", NEW_LINE), encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "feature")

    # main releases 0.27.0 and takes a fix.
    _git(repo, "checkout", "-q", "main")
    changelog.write_text(CHANGELOG_ON_MAIN, encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "release 0.27.0")

    monkeypatch.chdir(repo)
    event_path = _event(tmp_path, "enhancement")

    # The branch on its own looks fine: its line is under ## [Unreleased] there.
    _git(repo, "checkout", "-q", "feature")
    success, _ = check_changelog(event_name="pull_request", event_path=event_path, base_ref="main")
    assert success is True

    # GitHub's merge ref, the branch merged with main, is what the workflow checks out.
    _git(repo, "checkout", "-q", "--detach", "main")
    _git(repo, "merge", "-q", "--no-edit", "feature")
    assert NEW_LINE in changelog.read_text(encoding="utf-8").split("## [0.27.0]")[1].split("## [0.26.0]")[0]
    success, msg = check_changelog(event_name="pull_request", event_path=event_path, base_ref="main")
    assert success is False
    assert "under ## [0.27.0] - 2026-10-12" in msg
