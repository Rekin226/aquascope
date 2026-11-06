"""Tests for the documentation count maintenance fixer."""

from __future__ import annotations

from pathlib import Path

import pytest

from aquascope.maintenance import docs_counts


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    source_count: int = 3,
    cli_count: int = 4,
) -> None:
    monkeypatch.setattr(docs_counts, "ROOT", root)
    monkeypatch.setattr(
        docs_counts,
        "SOURCE_COUNT_PATTERNS",
        {"README.md": [r"supports (\d+) sources"]},
    )
    monkeypatch.setattr(
        docs_counts,
        "CLI_PATTERNS",
        {"README.md": r"ships a (\d+)-command CLI"},
    )
    monkeypatch.setattr(
        docs_counts,
        "canonical_counts",
        lambda: {
            "sources": source_count,
            "cli_commands": cli_count,
        },
    )


def test_apply_dry_run_reports_drift_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    readme = tmp_path / "README.md"
    readme.write_text(
        "AquaScope supports 2 sources and ships a 3-command CLI.\n",
        encoding="utf-8",
    )
    original = readme.read_text(encoding="utf-8")

    _configure(monkeypatch, tmp_path)

    changes = docs_counts.apply(write=False)

    assert len(changes) == 2
    assert readme.read_text(encoding="utf-8") == original


def test_apply_fix_rewrites_stale_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    readme = tmp_path / "README.md"
    readme.write_text(
        "AquaScope supports 2 sources and ships a 3-command CLI.\n",
        encoding="utf-8",
    )

    _configure(monkeypatch, tmp_path)

    changes = docs_counts.apply(write=True)

    assert len(changes) == 2
    assert readme.read_text(encoding="utf-8") == (
        "AquaScope supports 3 sources and ships a 4-command CLI.\n"
    )


def test_apply_leaves_correct_file_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    readme = tmp_path / "README.md"
    expected = "AquaScope supports 3 sources and ships a 4-command CLI.\n"
    readme.write_text(expected, encoding="utf-8")

    _configure(monkeypatch, tmp_path)

    changes = docs_counts.apply(write=True)

    assert changes == []
    assert readme.read_text(encoding="utf-8") == expected


def test_apply_fails_if_expected_phrase_is_reworded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    readme = tmp_path / "README.md"
    readme.write_text(
        "AquaScope provides 2 sources and ships a 3-command CLI.\n",
        encoding="utf-8",
    )

    _configure(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="exactly one phrase"):
        docs_counts.apply(write=True)

    assert readme.read_text(encoding="utf-8") == (
        "AquaScope provides 2 sources and ships a 3-command CLI.\n"
    )


def test_apply_fails_if_phrase_is_duplicated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    readme = tmp_path / "README.md"
    original = (
        "AquaScope supports 2 sources and ships a 3-command CLI.\n"
        "Another section supports 2 sources.\n"
    )
    readme.write_text(original, encoding="utf-8")

    _configure(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="exactly one phrase"):
        docs_counts.apply(write=True)

    assert readme.read_text(encoding="utf-8") == original
