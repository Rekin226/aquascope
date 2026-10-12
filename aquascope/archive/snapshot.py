"""Prepare an immutable archive deposit locally; never upload or assign a DOI.

Usage: python -m aquascope.archive.snapshot ARCHIVE --revision HUB_COMMIT --date YYYY-MM-DD --out DIRECTORY
The archive directory must be downloaded from that exact Hugging Face commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

from aquascope.archive.catalog import load_stations
from aquascope.registry import SOURCES


def prepare(archive: Path, out: Path, *, revision: str, snapshot_date: str) -> dict:
    """Package catalog, health, manifest and declared bundles, retaining mixed source rights."""
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Use an immutable 40-character Hugging Face commit, not main or a date label.")
    date.fromisoformat(snapshot_date)
    archive = archive.resolve()
    manifest = json.loads((archive / "obs/manifest.json").read_text(encoding="utf-8"))
    paths = {"stations.parquet", "health.json", "obs/manifest.json"}
    mirrored = set()
    for entry in manifest.get("bundles", {}).values():
        name = str(entry.get("file") or "")
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Unsafe or missing bundle path in archive manifest.")
        source = str(entry.get("source") or "")
        if source not in SOURCES or not SOURCES[source].redistributable:
            raise ValueError(f"Bundle source {source!r} is not approved for redistribution.")
        mirrored.add(source)
        paths.add(name)
    if not mirrored:
        raise ValueError("No mirrored observation bundles declared; build and verify bundles first.")
    # Resolve symlinks as well as textual '..' to keep the deposit within its supplied archive.
    for name in paths:
        path = (archive / name).resolve()
        if not path.is_relative_to(archive) or not path.is_file():
            raise ValueError(f"Missing or unsafe snapshot file: {name}")
    catalog_sources = {str(r["source"]) for r in load_stations(path=archive / "stations.parquet")}
    credits = []
    for key in sorted(catalog_sources | mirrored):
        meta = SOURCES.get(key)
        credits.append({"source": key, "agency": meta.agency if meta else "unverified",
                        "license": meta.license if meta else "unverified",
                        "homepage": meta.homepage if meta else None,
                        "scope": "catalog and mirrored observations" if key in mirrored else "catalog only"})
    inventory = [{"path": name, "bytes": (archive / name).stat().st_size,
                  "sha256": hashlib.sha256((archive / name).read_bytes()).hexdigest()} for name in sorted(paths)]
    metadata = {
        "title": f"AquaScope Hydrology gauge archive: {snapshot_date} snapshot",
        "resource_type": "dataset", "version": snapshot_date,
        "archive_revision": revision,
        "source": f"https://huggingface.co/datasets/Rekin226/aquascope-gauges/tree/{revision}",
        "description": "Station catalog plus available mirrored daily observations; some pins have no observations. "
                       "Coverage and last successful observation refresh are recorded in the manifest. "
                       "Each agency retains its own licence; no blanket relicensing is asserted.",
        "agency_credits_and_rights": credits, "files": inventory,
        "doi": None, "concept_doi": None, "publication_status": "prepared locally; not deposited or published",
        "cadence": "Quarterly reviewed snapshots; weekly operational harvests remain on Hugging Face.",
    }
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / "snapshot-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with zipfile.ZipFile(out / f"aquascope-gauges-{snapshot_date}.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(paths):
            bundle.write(archive / name, name)
        bundle.write(metadata_path, "snapshot-metadata.json")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.archive, args.out, revision=args.revision, snapshot_date=args.date)
    print(f"Prepared {len(result['files'])} files. No DOI assigned and nothing uploaded.")


if __name__ == "__main__":
    main()
