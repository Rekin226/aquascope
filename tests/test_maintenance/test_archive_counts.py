"""Tests for regenerating archive documentation statistics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquascope.maintenance.archive_counts import (
    archive_counts,
    update_archive_docs,
)


def _fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small repository and a representative harvest."""
    root = tmp_path / "repo"
    archive = root / "archive"

    (archive / "obs").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "explorer").mkdir()

    health = {
        "run_at": "2026-11-09T00:00:00+00:00",
        "n_stations": 3,
    }

    manifest = {
        "version": 2,
        "sources": {
            "usgs/discharge": {
                "source": "usgs",
                "variable": "discharge",
                "stations": {
                    "A": {"n": 10},
                    "empty": {"n": 0},
                },
            },
            "usgs/water_level": {
                "source": "usgs",
                "variable": "water_level",
                "stations": {
                    "A": {"n": 4},
                    "B": {"n": 2},
                },
            },
        },
    }

    (archive / "health.json").write_text(
        json.dumps(health), encoding="utf-8"
    )
    (archive / "obs" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    summary = (
        "The [Archive](docs/archive.md) publishes that output as one open "
        "dataset: a catalog of **75,892 stations**, with mirrored daily "
        "observations for **6,231 of them** (2026-11-02 harvest).\n"
    )

    (root / "README.md").write_text(summary, encoding="utf-8")
    (root / "docs" / "index.md").write_text(
        summary.replace("docs/archive.md", "archive.md"),
        encoding="utf-8",
    )

    (root / "ROADMAP.md").write_text(
        "- [x] The Archive, Phase 0 + 1 + 2: a 75,892-station GeoParquet "
        "catalog with per-station daily observations mirrored for 6,231 "
        "of them (discharge, water level, rainfall, groundwater level) "
        "and one Parquet bundle per variable and source on Hugging Face, "
        "weekly harvest with collector-health issues (#188)\n",
        encoding="utf-8",
    )

    (root / "docs" / "archive.md").write_text(
        "The catalog and the mirror are two different numbers, and this "
        "page keeps them apart: the **catalog** is the worldwide index "
        "of what the agencies publish, and the **mirror** is the subset "
        "the harvest has actually reached with a licence that allows "
        "redistribution. On the 2026-11-02 harvest the catalog held "
        "**75,892 stations** and the mirror held daily observations for "
        "**6,231 of them**.\n\n"
        "How far the mirror has reached, per variable, on the same "
        "2026-11-02 harvest. A station that measures two things appears "
        "once in each row, so the rows add up to more than the 6,231 "
        "stations above:\n\n"
        "| variable | mirrored stations | where the files are |\n"
        "| --- | --- | --- |\n"
        "| `discharge` | 4,330 | `obs/discharge/<source>/` |\n"
        "| `water_level` | 1,158 | `obs/water_level/<source>/` |\n",
        encoding="utf-8",
    )

    (root / "explorer" / "SPACE_README.md").write_text(
        "- Station catalog: [`Rekin226/aquascope-gauges`]"
        "(https://huggingface.co/datasets/Rekin226/aquascope-gauges)\n"
        "  (GeoParquet, harvested weekly), read in place with DuckDB-WASM. "
        "The catalog\n"
        "  holds **75,892 stations** and the observation mirror holds "
        "daily records for\n"
        "  **6,231 of them** (2026-11-02 harvest).\n",
        encoding="utf-8",
    )

    return root, archive


def test_archive_counts_deduplicates_stations_by_source_and_variable(
    tmp_path: Path,
):
    _, archive = _fixture_repo(tmp_path)

    counts = archive_counts(archive)

    assert counts == {
        "catalogued": 3,
        "mirrored": 2,
        "by_variable": {
            "discharge": 1,
            "water_level": 2,
        },
        "harvest_date": "2026-11-09",
    }


def test_update_archive_docs_updates_all_five_files(tmp_path: Path):
    root, archive = _fixture_repo(tmp_path)

    changed = update_archive_docs(root, archive)

    assert set(changed) == {
        "README.md",
        "docs/index.md",
        "ROADMAP.md",
        "docs/archive.md",
        "explorer/SPACE_README.md",
    }

    for relative_path in changed:
        content = (root / relative_path).read_text(encoding="utf-8")
        assert "2026-11-09" in content

    assert "3 stations" in (root / "README.md").read_text(encoding="utf-8")
    assert "2 of them" in (root / "README.md").read_text(encoding="utf-8")
    assert "3-station" in (root / "ROADMAP.md").read_text(encoding="utf-8")
    assert "2 of them" in (root / "ROADMAP.md").read_text(encoding="utf-8")

    archive_page = (root / "docs" / "archive.md").read_text(
        encoding="utf-8"
    )
    assert "| `discharge` | 1 |" in archive_page
    assert "| `water_level` | 2 |" in archive_page


def test_update_archive_docs_is_idempotent(tmp_path: Path):
    root, archive = _fixture_repo(tmp_path)

    update_archive_docs(root, archive)
    changed = update_archive_docs(root, archive)

    assert changed == []


def test_update_archive_docs_does_not_partially_write_on_invalid_section(
    tmp_path: Path,
):
    root, archive = _fixture_repo(tmp_path)

    readme = root / "README.md"
    original_readme = readme.read_text(encoding="utf-8")

    explorer = root / "explorer" / "SPACE_README.md"
    explorer.write_text("Unexpected content\n", encoding="utf-8")

    with pytest.raises(ValueError, match="explorer/SPACE_README.md"):
        update_archive_docs(root, archive)

    assert readme.read_text(encoding="utf-8") == original_readme
