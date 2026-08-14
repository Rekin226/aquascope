"""Unit tests for .github/scripts/check_changelog.py (Issue #144)."""

from __future__ import annotations

import json

# Import the check_changelog script
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / ".github" / "scripts"))

from check_changelog import check_changelog, get_pr_labels  # noqa: E402


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

        success, msg = check_changelog(
            event_name="pull_request",
            event_path=event_path,
            changed_files=["aquascope/cli.py", "CHANGELOG.md"],
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
