"""Sample model results in the Archive (#591): files the Explorer's "Try a sample model" opens.

A browser page can only fetch a file from a host that allows it (CORS). FloPy's Freyberg example is fetched
straight from GitHub (``raw.githubusercontent.com`` answers any origin), so it is not copied. The HEC-RAS 2D
sample is not: HydroShare hands the file to a signed Google Cloud Storage address that sends no
``Access-Control-Allow-Origin``, so the page cannot read it there. This module copies that one file, unchanged,
into the Archive dataset under ``samples/``, with a manifest that says where it came from, under what licence,
who made it, and its SHA-256.

Layout under ``<out>/samples/`` (the folder the Archive dataset publishes):

    hec-ras/Flood_model.p01_BC12_S6_R6.hdf      the plan result, byte for byte as HydroShare serves it
    hec-ras/manifest.json                       id, title, about, files (path, name, bytes, sha256),
                                                source (url, resource, title, creator, licence, licence_url),
                                                attribution, mirrored_at, made_by

A one-off, run by ``.github/workflows/mirror-samples.yml`` (manual); every step is also a CLI subcommand:

    python -m aquascope.archive.samples build --out build
    python -m aquascope.archive.samples publish --out build

The build downloads the file, refuses it unless its SHA-256 is the one recorded here (the file read for #592,
``docs/assessments/hec-ras.md``), and writes the manifest. Publishing needs the ``archive`` extra and a token with
write access (HF_TOKEN), and uploads only ``samples/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The Archive's folder for sample model results.
FOLDER = "samples"

#: The HEC-RAS 2D sample: one plan result from HydroShare resource 32aac3af553c4e1fbb997ec58d9de9b6 (CC BY 4.0),
#: the run #592 checked depth and volumes on. ``sha256`` and ``bytes`` were measured on the file HydroShare served
#: on 2026-10-12; a build refuses a file that differs.
HEC_RAS: dict[str, Any] = {
    "id": "hec-ras-portsmouth",
    "folder": "hec-ras",
    "name": "Flood_model.p01_BC12_S6_R6.hdf",
    "bytes": 29_906_342,
    "sha256": "11c603af678d7f608ab75ef04c2cf8db97483702c8683c868e76b1405cd46fa5",
    "title": "HEC-RAS 2D: compound flooding in Portsmouth, Virginia",
    "about": ("A HEC-RAS 6.5 plan result: one 2D area (6,272 cells), rain on the grid and three stage boundaries, "
              "109 steps every 15 minutes from 30 June 2021 00:00 to 1 July 03:00, NAD83 / UTM zone 18N."),
    "source": {
        "url": ("https://www.hydroshare.org/resource/32aac3af553c4e1fbb997ec58d9de9b6/data/contents/GNN4CF_dataset/"
                "raw_hecras/test_multi_boundary/Flood_model.p01_BC12_S6_R6.hdf"),
        "resource": "https://www.hydroshare.org/resource/32aac3af553c4e1fbb997ec58d9de9b6/",
        "title": "GNN4CF: Dataset for Boundary-Aware Graph Neural Networks for Compound Flood Forecasting",
        "creator": "Zandsalimi, Zanko",
        "publisher": "HydroShare",
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
    },
}

SAMPLES: dict[str, dict[str, Any]] = {HEC_RAS["id"]: HEC_RAS}


def attribution(sample: dict[str, Any]) -> str:
    """The credit line the Explorer shows: creator, title, publisher, licence, resource."""
    s = sample["source"]
    return f"{s['creator']}, \"{s['title']}\", {s['publisher']}, {s['licence']}. {s['resource']}"


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str | Path) -> Path:
    """Stream ``url`` (following HydroShare's redirects) to ``dest``."""
    import httpx

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)
    return dest


def manifest(sample: dict[str, Any], *, made_by: str | None = None, now: datetime | None = None) -> dict[str, Any]:
    """The manifest published beside the sample (what the Explorer reads before it offers the file)."""
    if made_by is None:
        from aquascope import __version__

        made_by = f"aquascope {__version__} (python -m aquascope.archive.samples)"
    when = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = f"{FOLDER}/{sample['folder']}/{sample['name']}"
    return {
        "id": sample["id"], "title": sample["title"], "about": sample["about"],
        "files": [{"path": path, "name": sample["name"], "bytes": sample["bytes"], "sha256": sample["sha256"]}],
        "source": dict(sample["source"]), "attribution": attribution(sample),
        "mirrored_at": when, "made_by": made_by,
    }


def build(out: str | Path, *, sample_id: str = HEC_RAS["id"], source: str | Path | None = None) -> dict[str, Any]:
    """Download a sample into ``<out>/samples/<folder>/`` (or copy ``source``, a local copy), check its size and
    SHA-256 against the recorded ones, and write its manifest. Raises ValueError when the file is not the one
    recorded: a changed file at the source is a new sample, to be checked before it is offered."""
    sample = SAMPLES[sample_id]
    folder = Path(out) / FOLDER / sample["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / sample["name"]
    if source:
        shutil.copyfile(source, dest)
    else:
        download(sample["source"]["url"], dest)
    size, digest = dest.stat().st_size, sha256_of(dest)
    if size != sample["bytes"] or digest != sample["sha256"]:
        dest.unlink()
        raise ValueError(f"{sample['name']} is {size:,} bytes with SHA-256 {digest}, not the recorded "
                         f"{sample['bytes']:,} bytes and {sample['sha256']}: the file changed at the source")
    info = manifest(sample)
    (folder / "manifest.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    logger.info("%s: %s bytes, SHA-256 %s, %s", sample["id"], f"{size:,}", digest, sample["source"]["licence"])
    return info


#: What a publish uploads: the samples' files and manifests, nothing else.
PUBLISH_PATTERNS = [f"{FOLDER}/*/*.hdf", f"{FOLDER}/*/manifest.json"]


def publish(out: str | Path, *, repo_id: str = "Rekin226/aquascope-gauges", token: str | None = None) -> str:
    """Upload ``<out>/samples`` to the Archive dataset (needs HF_TOKEN with write access). Only ``samples/`` is
    staged, so nothing outside it can land in the dataset."""
    from aquascope.archive.publish import publish_folder

    src = Path(out) / FOLDER
    if not any(src.glob("*/manifest.json")):
        raise FileNotFoundError(f"no sample manifest under {src}: run the build first")
    names = ", ".join(sorted(p.parent.name for p in src.glob("*/manifest.json")))
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(src, Path(tmp) / FOLDER)
        return publish_folder(Path(tmp), repo_id, token=token, commit_message=f"samples: {names}",
                              allow_patterns=PUBLISH_PATTERNS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.archive.samples", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="download and check the samples, write their manifests")
    p.add_argument("--out", required=True)
    p.add_argument("--source", default=None, help="a local copy of the HEC-RAS sample instead of downloading")
    p = sub.add_parser("publish", help="upload <out>/samples to the Archive dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default=os.environ.get("HF_DATASET", "Rekin226/aquascope-gauges"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.cmd == "build":
        info: Any = build(a.out, source=a.source)
    else:
        info = {"commit": publish(a.out, repo_id=a.repo)}
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
