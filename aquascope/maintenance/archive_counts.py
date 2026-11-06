
"""Regenerate repository archive statistics from a published harvest."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from aquascope.archive.observations import _migrate_manifest


def archive_counts(archive_dir: str | Path) -> dict[str, Any]:
    """Calculate catalog and mirrored station counts from a harvest directory."""
    archive = Path(archive_dir)

    health_path = archive / "health.json"
    manifest_path = archive / "obs" / "manifest.json"

    if not health_path.is_file():
        raise FileNotFoundError(f"Missing harvest health file: {health_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing observation manifest: {manifest_path}")

    health = json.loads(health_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _migrate_manifest(manifest)

    mirrored: set[tuple[str, str]] = set()
    by_variable: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for key, entry in manifest.get("sources", {}).items():
        source = entry.get("source") or key.split("/", 1)[0]
        variable = entry.get("variable")

        if not variable:
            continue

        for station_id, station in entry.get("stations", {}).items():
            if int(station.get("n") or 0) <= 0:
                continue

            station_key = (source, station_id)
            mirrored.add(station_key)
            by_variable[variable].add(station_key)

    return {
        "catalogued": int(health["n_stations"]),
        "mirrored": len(mirrored),
        "by_variable": {
            variable: len(stations)
            for variable, stations in sorted(by_variable.items())
        },
        "harvest_date": health["run_at"][:10],
    }


def _format_count(value: int) -> str:
    """Format a count with thousands separators."""
    return f"{value:,}"


def _variable_table(counts: dict[str, Any]) -> str:
    """Build the per-variable mirrored-station table."""
    rows = [
        "| variable | mirrored stations | where the files are |",
        "| --- | ---: | --- |",
    ]

    for variable, count in counts["by_variable"].items():
        rows.append(
            f"| `{variable}` | {_format_count(count)} "
            f"| `obs/{variable}/<source>/` |"
        )

    return "\n".join(rows)


def _sub_once(
    pattern: str,
    replacement: str,
    text: str,
    filename: str,
    flags: int = 0,
) -> str:
    """Replace exactly one matching documentation section."""
    updated, count = re.subn(pattern, replacement, text, flags=flags)

    if count != 1:
        raise ValueError(
            f"Expected exactly one matching section in {filename}; found {count}"
        )

    return updated


def _update_summary_text(
    text: str,
    counts: dict[str, Any],
    filename: str,
) -> str:
    """Update the archive summary sentence without changing surrounding text."""
    pattern = (
        r"^(The \[Archive\]\([^)]+\) publishes .*?catalog of \*\*)"
        r"[\d,]+"
        r"( stations\*\*, with mirrored daily observations for \*\*)"
        r"[\d,]+"
        r"( of them\*\* \()"
        r"\d{4}-\d{2}-\d{2}"
        r"( harvest\)\.)$"
    )

    replacement = (
        rf"\g<1>{_format_count(counts['catalogued'])}"
        rf"\g<2>{_format_count(counts['mirrored'])}"
        rf"\g<3>{counts['harvest_date']}"
        rf"\g<4>"
    )

    return _sub_once(
        pattern,
        replacement,
        text,
        filename,
        flags=re.MULTILINE,
    )


def _update_summary_file(
    repo_root: Path,
    relative_path: str,
    counts: dict[str, Any],
) -> None:
    """Update the archive summary in one documentation file."""
    path = repo_root / relative_path
    text = path.read_text(encoding="utf-8")
    updated = _update_summary_text(text, counts, relative_path)

    if updated != text:
        path.write_text(updated, encoding="utf-8")


def _update_roadmap_text(
    text: str,
    counts: dict[str, Any],
) -> str:
    """Update archive counts and harvest date in the roadmap entry."""
    pattern = (
        r"(^- \[x\] The Archive, Phase 0 \+ 1 \+ 2: a )"
        r"[\d,]+"
        r"(-station GeoParquet catalog with per-station daily observations "
        r"mirrored for )"
        r"[\d,]+"
        r"( of them)"
        r"(?: \(\d{4}-\d{2}-\d{2} harvest\))?"
    )

    replacement = (
        rf"\g<1>{_format_count(counts['catalogued'])}"
        rf"\g<2>{_format_count(counts['mirrored'])}"
        rf"\g<3> ({counts['harvest_date']} harvest)"
    )

    return _sub_once(
        pattern,
        replacement,
        text,
        "ROADMAP.md",
        flags=re.MULTILINE,
    )


def _update_archive_page_text(
    text: str,
    counts: dict[str, Any],
) -> str:
    """Update archive-page summary and per-variable statistics."""
    summary_pattern = (
        r"(that allows redistribution\. On the )"
        r"\d{4}-\d{2}-\d{2}"
        r"( harvest the catalog held\s+\*\*)"
        r"[\d,]+"
        r"( stations\*\* and the mirror held daily observations for \*\*)"
        r"[\d,]+"
        r"( of\s+them\*\*\.)"
    )

    summary_replacement = (
        rf"\g<1>{counts['harvest_date']}"
        rf"\g<2>{_format_count(counts['catalogued'])}"
        rf"\g<3>{_format_count(counts['mirrored'])}"
        rf"\g<4>"
    )

    updated = _sub_once(
        summary_pattern,
        summary_replacement,
        text,
        "docs/archive.md summary",
        flags=re.MULTILINE,
    )

    table_pattern = (
        r"(^How far the mirror has reached, per variable, on the same )"
        r"\d{4}-\d{2}-\d{2}"
        r" harvest\.[\s\S]*?\n\n"
        r"\| variable \| mirrored stations \| where the files are \|\n"
        r"\|[^\n]*\|\n"
        r"(?:\|[^\n]*\|\n)*"
    )

    table_replacement = (
        rf"\g<1>{counts['harvest_date']} harvest. A\n"
        "station that measures two things appears once in each row, so the rows add up\n"
        f"to more than the {_format_count(counts['mirrored'])} stations above:\n\n"
        f"{_variable_table(counts)}\n"
    )

    return _sub_once(
        table_pattern,
        table_replacement,
        updated,
        "docs/archive.md variable table",
        flags=re.MULTILINE,
    )

def _update_explorer_text(
    text: str,
    counts: dict[str, Any],
) -> str:
    """Update the station catalog statistics in the Explorer Space README."""
    pattern = (
        r"(^- Station catalog: .*\n"
        r"  \(GeoParquet, harvested weekly\), read in place with DuckDB-WASM\. The catalog\n"
        r"  holds \*\*)"
        r"[\d,]+"
        r"( stations\*\* and the observation mirror holds daily records for\n"
        r"  \*\*)"
        r"[\d,]+"
        r"( of them\*\* \()"
        r"\d{4}-\d{2}-\d{2}"
        r"( harvest\.\))"
    )

    replacement = (
        rf"\g<1>{_format_count(counts['catalogued'])}"
        rf"\g<2>{_format_count(counts['mirrored'])}"
        rf"\g<3>{counts['harvest_date']}"
        rf"\g<4>"
    )

    return _sub_once(
        pattern,
        replacement,
        text,
        "explorer/SPACE_README.md",
        flags=re.MULTILINE,
    )
