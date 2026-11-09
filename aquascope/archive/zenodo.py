"""Deposit a snapshot of the Archive on Zenodo, so the data has its own DOI (#444).

Internal: a maintainer tool, run by ``.github/workflows/archive-zenodo.yml``. Not part of the public API.

``build`` makes a deposit bundle from the published Archive (downloaded from Hugging Face at one commit, or a
local harvest folder) and never talks to Zenodo. ``deposit`` uploads a built bundle: a new record, or a new
version of an existing concept record, left as a draft unless ``--publish`` is given.

    python -m aquascope.archive.zenodo build --out deposit [--archive DIR | --revision SHA] [--dry-run]
    python -m aquascope.archive.zenodo deposit --bundle deposit [--sandbox] [--concept-id N] [--publish]

The bundle holds the catalogue (``stations.parquet``, ``stations.geojson``), ``signatures.parquet``,
``health.json``, the dataset card ``README.md``, the ``obs/`` JSON files, one ``.tar`` per (variable, source)
with that source's Parquet bundle and per-station CSVs at their Archive paths, ``SOURCES.md`` (every source's
licence and attribution) and ``MANIFEST.json`` (every file's size and SHA-256, and the snapshot date). The
derived folders (basins, forecasts, ...) are not deposited; the description links to them on Hugging Face.
Only verified licences leave Hugging Face: the rows of a source whose registry licence is ``"unknown"`` (or that
is not in the registry) are left out of the catalogue files, and SOURCES.md and the description name them.

The token comes from ``ZENODO_TOKEN`` and is only ever sent to Zenodo in the Authorization header.
Zenodo's REST API: https://developers.zenodo.org/
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tarfile
import time
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from aquascope.archive.publish import DEFAULT_REPO_ID
from aquascope.registry import SOURCES, VARIABLES

ZENODO_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"
SOFTWARE_CONCEPT_DOI = "10.5281/zenodo.21903143"
HF_URL = f"https://huggingface.co/datasets/{DEFAULT_REPO_ID}"
LAYOUT_URL = "https://rekin226.github.io/aquascope/archive-layout/"

#: As in CITATION.cff (a test keeps the two the same).
CREATORS = [{
    "name": "Ouédraogo, Abdoul Rachid",
    "affiliation": "POUK YAM, Ouagadougou, Burkina Faso",
    "orcid": "0000-0002-4616-4153",
}]

#: The Archive mixes per-source licences (public domain, OGL, Etalab, CC BY, CC BY-SA, national open data
#: licences), so no single open licence is true of every file. Zenodo's "Other (Attribution)" says what they share:
#: reuse is open and attribution is required. Each source's own licence is in the description and SOURCES.md.
RECORD_LICENSE = "other-at"

TITLE = "AquaScope Archive: public river gauge catalogue and mirrored daily records"

#: Root files deposited as they are.
ROOT_FILES = ("stations.parquet", "stations.geojson", "signatures.parquet", "health.json", "README.md")
REQUIRED = ("stations.parquet", "health.json", "obs/manifest.json")

#: Derived folders left on Hugging Face (linked from the description).
NOT_DEPOSITED = ("basins", "bulletins", "context", "feeds", "flowstatus", "forecasts", "health", "samples",
                 "scout", "skill", "waterbalance")

#: Zenodo's limits for the files API: 100 files and 50 GB per record.
MAX_FILES = 100
MAX_BYTES = 50 * 10**9

KEYWORDS = ["hydrology", "river discharge", "streamflow", "water level", "groundwater level", "precipitation",
            "gauging stations", "open data", "GeoParquet", "AquaScope"]

_CHUNK = 1 << 20


class ZenodoError(RuntimeError):
    """Zenodo refused a request, or the token is missing."""


# ── build ──────────────────────────────────────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - Zenodo reports an MD5 checksum; this only compares against it
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(out: Path, *, revision: str | None = None, repo_id: str = DEFAULT_REPO_ID) -> tuple[Path, str]:
    """Download what a deposit needs from the Hugging Face dataset at one commit; return (folder, commit)."""
    from aquascope.utils.imports import require

    hub = require("huggingface_hub", feature="the Zenodo deposit", group="archive")
    sha = hub.HfApi().dataset_info(repo_id, revision=revision or "main").sha
    hub.snapshot_download(repo_id, repo_type="dataset", revision=sha, local_dir=str(out),
                          allow_patterns=[*ROOT_FILES, "obs/**"])
    return out, sha


def _station_counts(archive: Path) -> tuple[int, dict[str, int]]:
    import pyarrow.parquet as pq

    table = pq.read_table(archive / "stations.parquet", columns=["source"])
    per_source: dict[str, int] = {}
    for key in table.column("source").to_pylist():
        per_source[str(key)] = per_source.get(str(key), 0) + 1
    return table.num_rows, per_source


def _bundle_stations(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    return {str(s) for s in pq.read_table(path, columns=["station_id"]).column("station_id").unique().to_pylist()}


def unverified(source: str) -> bool:
    """True when the registry has no verified licence for a source (absent, or ``license="unknown"``)."""
    meta = SOURCES.get(source)
    return meta is None or meta.license == "unknown"


def _filter_parquet(src: Path, dst: Path) -> dict[str, int]:
    """Copy a Parquet file without the rows of unverified sources; return the rows left out per source.

    The schema metadata (GeoParquet ``geo``) is kept, with its bbox recomputed for the rows that remain."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(src)
    keep = [not unverified(str(key)) for key in table.column("source").to_pylist()]
    dropped: dict[str, int] = {}
    for key, kept in zip(table.column("source").to_pylist(), keep):
        if not kept:
            dropped[str(key)] = dropped.get(str(key), 0) + 1
    if dropped:
        table = table.filter(pa.array(keep, type=pa.bool_()))
        meta = dict(table.schema.metadata or {})
        if b"geo" in meta and {"latitude", "longitude"} <= set(table.column_names) and table.num_rows:
            geo = json.loads(meta[b"geo"])
            column = geo.get("primary_column", "geometry")
            if "bbox" in geo.get("columns", {}).get(column, {}):
                lon, lat = pc.min_max(table.column("longitude")), pc.min_max(table.column("latitude"))
                geo["columns"][column]["bbox"] = [lon["min"].as_py(), lat["min"].as_py(),
                                                  lon["max"].as_py(), lat["max"].as_py()]
                meta[b"geo"] = json.dumps(geo).encode()
        table = table.replace_schema_metadata(meta)
    source_meta = pq.ParquetFile(src).metadata
    codec = source_meta.row_group(0).column(0).compression.lower() if source_meta.num_row_groups else "snappy"
    pq.write_table(table, dst, compression=codec if codec in ("snappy", "gzip", "brotli", "zstd", "lz4") else "zstd")
    return dropped


def _filter_geojson(src: Path, dst: Path) -> dict[str, int]:
    """Copy the GeoJSON catalogue without the features of unverified sources."""
    doc = json.loads(src.read_text(encoding="utf-8"))
    dropped: dict[str, int] = {}
    kept = []
    for feature in doc.get("features", []):
        key = str((feature.get("properties") or {}).get("source", ""))
        if unverified(key):
            dropped[key] = dropped.get(key, 0) + 1
        else:
            kept.append(feature)
    doc["features"] = kept
    dst.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return dropped


def _scan_obs(archive: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[Path]]:
    """Group obs/ into (variable, source) pairs, refusing anything that is not a redistributable source."""
    obs = archive / "obs"
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    loose: list[Path] = []
    for path in sorted(obs.rglob("*")):
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(archive.resolve()):
            raise ValueError(f"Unsafe file in obs/: {path.relative_to(archive)}")
        parts = path.relative_to(obs).parts
        if len(parts) == 1:
            if path.suffix != ".json":
                raise ValueError(f"Unexpected file in obs/: {parts[0]}")
            loose.append(path)
            continue
        variable = parts[0]
        source = parts[1].removesuffix(".parquet") if len(parts) == 2 else parts[1]
        bundle = len(parts) == 2 and parts[1].endswith(".parquet")
        ok_name = bundle or (len(parts) == 3 and parts[2].endswith(".csv.gz"))
        if variable not in VARIABLES or not ok_name:
            raise ValueError(f"Unexpected path in obs/: {path.relative_to(archive).as_posix()}")
        meta = SOURCES.get(source)
        if meta is None or not meta.redistributable:
            raise ValueError(f"obs/{variable}/{source} is from a source that is not approved for redistribution; "
                             "it must not be deposited.")
        entry = pairs.setdefault((variable, source), {"parquet": None, "csv": []})
        if len(parts) == 2:
            entry["parquet"] = path
        else:
            entry["csv"].append(path)
    return pairs, loose


def _safe_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def _sources_md(credits: list[dict[str, Any]], excluded: list[dict[str, Any]], snapshot_date: str) -> str:
    lines = [
        f"# Sources in the AquaScope Archive snapshot of {snapshot_date}",
        "",
        "The Archive has no single licence. Each source keeps its own licence and attribution, below, and the",
        "files from a source are reused on that source's terms. Catalogue rows are station metadata (id, name,",
        "coordinates, variables, a link to the agency); observations are mirrored only from sources whose terms",
        "allow redistribution.",
        "",
        "| Source | Agency | Licence | In this deposit | Stations catalogued | Stations mirrored | Attribution |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for c in credits:
        cells = [c["source"], c["agency"], c["license"], c["scope"], str(c["catalogued"]), str(c["mirrored"]),
                 c["attribution"]]
        lines.append("| " + " | ".join(x.replace("|", "/") for x in cells) + " |")
    if excluded:
        lines += ["", "## Left out until their reuse terms are verified", "",
                  "These sources are in the Archive's catalogue on Hugging Face, but their licence is not verified, "
                  "so their rows are not in this deposit (not in `stations.parquet`, `stations.geojson` or "
                  "`signatures.parquet`).", "",
                  "| Source | Agency | Licence | Rows left out |", "| --- | --- | --- | --- |"]
        for x in excluded:
            rows = ", ".join(f"{n:,} in {f}" for f, n in sorted(x["rows_left_out"].items()))
            lines.append(f"| {x['source']} | {x['agency']} | {x['license']} | {rows} |")
    lines += ["", "Licence details and homepages: `aquascope.registry.SOURCES` and "
              "https://rekin226.github.io/aquascope/data_sources/", ""]
    return "\n".join(lines)


def _description(snapshot: dict[str, Any]) -> str:
    e = html.escape
    cat, mir = snapshot["catalogued"], snapshot["mirrored"]
    rows = "".join(f"<li>{e(v)}: {n:,} stations</li>" for v, n in sorted(mir["per_variable"].items()))
    sources = "".join(
        f"<li><b>{e(c['source'])}</b> ({e(c['agency'])}): {e(c['license'])}; {e(c['scope'])}. "
        f"{e(c['attribution'])}</li>" for c in snapshot["sources"])
    derived = ", ".join(f"<code>{d}/</code>" for d in NOT_DEPOSITED)
    left = "".join(f"<li><b>{e(x['source'])}</b> ({e(x['agency'])}): {x['stations_left_out']:,} stations</li>"
                   for x in snapshot.get("left_out", []))
    left_out = (f"<p><b>Left out until their reuse terms are verified.</b> These sources are catalogued on "
                f"Hugging Face, but their licence is not verified, so their rows are not in this deposit:</p>"
                f"<ul>{left}</ul>") if left else ""
    return (
        f"<p>A snapshot of the AquaScope Archive, the Hugging Face dataset "
        f"<a href=\"{HF_URL}\">{DEFAULT_REPO_ID}</a> at commit <code>{e(snapshot['revision'])}</code>, "
        f"taken on {e(snapshot['date'])}.</p>"
        f"<p><b>Catalogued and mirrored are different numbers.</b> The catalogue lists {cat['stations']:,} gauging "
        f"stations from {cat['sources']} public sources: station metadata (id, name, coordinates, variables, a link "
        f"to the agency), not observations. Daily observations are mirrored for {mir['stations']:,} of them, from "
        f"{mir['sources']} sources whose terms allow redistribution:</p><ul>{rows}</ul>"
        f"<p>The other catalogued stations have no observations in this deposit; read them from the agency. "
        f"Mirrored records are daily values as each agency publishes them, after unit conversion (discharge m3/s, "
        f"levels m, precipitation mm per day), without quality flags.</p>"
        f"{left_out}"
        f"<p><b>Files.</b> <code>stations.parquet</code> (GeoParquet) and <code>stations.geojson</code>: the "
        f"catalogue. <code>signatures.parquet</code>: flow signatures of the mirrored discharge stations. "
        f"<code>health.json</code>: the harvest's per-source report. <code>README.md</code>: the dataset card. "
        f"<code>obs_*.json</code>: the observation manifest and run status. <code>obs_&lt;variable&gt;_&lt;source&gt;"
        f".tar</code>: that source's Parquet bundle and per-station CSVs at their Archive paths (extract every "
        f"tar in one folder to rebuild <code>obs/</code>). <code>SOURCES.md</code>: every source's licence and "
        f"attribution. <code>MANIFEST.json</code>: every file's size and SHA-256. Columns and units: "
        f"<a href=\"{LAYOUT_URL}\">Archive layout</a>.</p>"
        f"<p><b>Not deposited.</b> The derived folders ({derived}) stay on "
        f"<a href=\"{HF_URL}\">Hugging Face</a>, where they are rebuilt on their own schedules.</p>"
        f"<p><b>Licence.</b> There is no single licence: each source keeps its own, and its files are reused on "
        f"its terms. The record licence is set to Other (Attribution) because reuse is open and every source asks "
        f"to be credited. Per source (also in <code>SOURCES.md</code>):</p><ul>{sources}</ul>"
        f"<p><b>Citing.</b> Cite this dataset for the data, together with the agencies above. Cite the AquaScope "
        f"software (<a href=\"https://doi.org/{SOFTWARE_CONCEPT_DOI}\">{SOFTWARE_CONCEPT_DOI}</a>) for the code.</p>"
    )


def build_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Zenodo deposition metadata for a built snapshot (https://developers.zenodo.org/#representation)."""
    agencies = sorted({c["agency"] for c in snapshot["sources"] if c["agency"] and c["agency"] != "unverified"})
    return {
        "upload_type": "dataset",
        "title": TITLE,
        "creators": CREATORS,
        "description": _description(snapshot),
        "access_right": "open",
        "license": RECORD_LICENSE,
        "version": snapshot["date"],
        "keywords": KEYWORDS,
        "language": "eng",
        "contributors": [{"name": a, "type": "DataCollector"} for a in agencies],
        "related_identifiers": [
            {"relation": "isSupplementTo", "identifier": SOFTWARE_CONCEPT_DOI, "resource_type": "software"},
            {"relation": "isDerivedFrom", "identifier": f"{HF_URL}/tree/{snapshot['revision']}",
             "resource_type": "dataset"},
            {"relation": "isDocumentedBy", "identifier": LAYOUT_URL},
        ],
        "notes": "Snapshot cadence: quarterly, reviewed by the maintainer. The weekly harvest keeps updating "
                 "the Hugging Face dataset between snapshots; every snapshot is a new version of this record.",
    }


def build(archive: Path, out: Path, *, revision: str, snapshot_date: str | None = None) -> dict[str, Any]:
    """Build the deposit bundle in ``out`` from an Archive folder. No network. Returns the MANIFEST dict,
    with the Zenodo metadata under ``"zenodo_metadata"``."""
    archive = archive.resolve()
    snapshot_date = snapshot_date or date.today().isoformat()
    date.fromisoformat(snapshot_date)
    if not re.fullmatch(r"[0-9a-f]{40}", revision) and revision != "local":
        raise ValueError("revision must be the 40-character Hugging Face commit the folder came from, or 'local'.")
    for name in REQUIRED:
        if not (archive / name).is_file():
            raise ValueError(f"Not an Archive folder: {name} is missing from {archive}")
    pairs, loose = _scan_obs(archive)
    if not pairs:
        raise ValueError("No mirrored observations under obs/; nothing to deposit beside the catalogue.")

    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; build into a new folder.")
    out.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []

    def record(name: str, archive_path: str, **extra: Any) -> None:
        path = out / name
        files.append({"name": name, "archive_path": archive_path, "bytes": path.stat().st_size,
                      "sha256": _sha256(path), **extra})

    # Only verified licences leave Hugging Face: rows of a source whose licence is unknown are left out.
    left_out: dict[str, dict[str, int]] = {}
    for name in ROOT_FILES:
        if not (archive / name).is_file():
            continue
        if name.endswith(".parquet"):
            dropped = _filter_parquet(archive / name, out / name)
        elif name == "stations.geojson":
            dropped = _filter_geojson(archive / name, out / name)
        else:
            shutil.copyfile(archive / name, out / name)
            dropped = {}
        for key, n in dropped.items():
            left_out.setdefault(key, {})[name] = n
        record(name, name, **({"rows_left_out": dropped} if dropped else {}))
    for path in loose:
        name = f"obs_{path.name}"
        shutil.copyfile(path, out / name)
        record(name, f"obs/{path.name}")

    mirrored: dict[str, set[str]] = {}
    per_variable: dict[str, set[tuple[str, str]]] = {}
    for (variable, source), entry in sorted(pairs.items()):
        name = f"obs_{variable}_{source}.tar"
        members = ([entry["parquet"]] if entry["parquet"] else []) + sorted(entry["csv"])
        with tarfile.open(out / name, "w", format=tarfile.PAX_FORMAT) as tar:
            for member in members:
                tar.add(member, arcname=member.relative_to(archive).as_posix(), recursive=False,
                        filter=_safe_tarinfo)
        if entry["parquet"]:
            stations = _bundle_stations(entry["parquet"])
        else:
            stations = {p.name.removesuffix(".csv.gz") for p in entry["csv"]}
        mirrored.setdefault(source, set()).update(stations)
        per_variable.setdefault(variable, set()).update((source, s) for s in stations)
        record(name, f"obs/{variable}/{source}", variable=variable, source=source, members=len(members),
               stations=len(stations))

    n_catalogued, catalogued = _station_counts(out)
    excluded = [{"source": key, "agency": ((SOURCES[key].agency if key in SOURCES else "") or "unverified"),
                 "license": SOURCES[key].license if key in SOURCES else "not in the registry",
                 "rows_left_out": rows, "stations_left_out": rows.get("stations.parquet", 0)}
                for key, rows in sorted(left_out.items())]
    credits = []
    for key in sorted(set(catalogued) | set(mirrored)):
        meta = SOURCES.get(key)
        credits.append({
            "source": key,
            "agency": (meta.agency if meta else "") or "unverified",
            "license": meta.license if meta else "unverified",
            "attribution": (meta.attribution if meta else "") or "",
            "homepage": meta.homepage if meta else None,
            "scope": "catalogue and mirrored observations" if key in mirrored else "catalogue only",
            "catalogued": catalogued.get(key, 0),
            "mirrored": len(mirrored.get(key, ())),
        })
    snapshot = {
        "date": snapshot_date,
        "revision": revision,
        "catalogued": {"stations": n_catalogued, "sources": len(catalogued)},
        "mirrored": {"stations": sum(len(s) for s in mirrored.values()), "sources": len(mirrored),
                     "per_variable": {v: len(s) for v, s in per_variable.items()}},
        "sources": credits,
        "left_out": excluded,
    }
    (out / "SOURCES.md").write_text(_sources_md(credits, excluded, snapshot_date), encoding="utf-8")
    record("SOURCES.md", "")

    metadata = build_metadata(snapshot)
    manifest = {
        "title": TITLE,
        "snapshot_date": snapshot_date,
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": f"{HF_URL}/tree/{revision}" if revision != "local" else "a local Archive folder",
        **snapshot,
        "not_deposited": [f"{d}/" for d in NOT_DEPOSITED],
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "zenodo-metadata.json").write_text(json.dumps({"metadata": metadata}, indent=2, ensure_ascii=False)
                                               + "\n", encoding="utf-8")
    n_files = len(files) + 1  # with MANIFEST.json
    total = manifest["total_bytes"] + (out / "MANIFEST.json").stat().st_size
    if n_files > MAX_FILES or total > MAX_BYTES:
        raise ValueError(f"The bundle has {n_files} files and {total:,} bytes; Zenodo takes at most {MAX_FILES} "
                         f"files and {MAX_BYTES:,} bytes per record.")
    return {**manifest, "zenodo_metadata": metadata}


def bundle_files(bundle: Path) -> list[Path]:
    """The files of a built bundle to upload, in MANIFEST order, MANIFEST.json last."""
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    paths = []
    for f in manifest["files"]:
        path = bundle / PurePosixPath(f["name"]).name
        if not path.is_file() or path.stat().st_size != f["bytes"]:
            raise ValueError(f"{f['name']} is missing or changed since the build; rebuild the bundle.")
        paths.append(path)
    return [*paths, bundle / "MANIFEST.json"]


# ── deposit ────────────────────────────────────────────────────────────────────────────────────────────────


def _check(resp: httpx.Response, what: str, ok: tuple[int, ...]) -> httpx.Response:
    if resp.status_code in ok:
        return resp
    try:
        body = resp.json()
        detail = body.get("message", "")
        errors = "; ".join(f"{e.get('field', '')}: {e.get('message', '')}" for e in body.get("errors") or [])
        detail = f"{detail} ({errors})" if errors else detail
    except ValueError:
        detail = resp.text[:300]
    hint = {401: " The token was refused: check it is valid for this Zenodo (sandbox and production differ).",
            403: " The token lacks a scope: it needs deposit:write, and deposit:actions for a new version or "
                 "publishing."}.get(resp.status_code, "")
    raise ZenodoError(f"Zenodo refused {what}: HTTP {resp.status_code} {detail}.{hint}")


def _concept_number(concept: str) -> str:
    match = re.fullmatch(r"(?:https://doi\.org/)?(?:10\.(?:5281|5072)/zenodo\.)?(\d+)", concept.strip())
    if not match:
        raise ValueError(f"concept id {concept!r} is not a Zenodo record number or zenodo DOI")
    return match.group(1)


def deposit(
    bundle: Path,
    *,
    token: str | None = None,
    sandbox: bool = True,
    concept_id: str | None = None,
    publish: bool = False,
    client: httpx.Client | None = None,
    pause: float = 5.0,
) -> dict[str, Any]:
    """Upload a built bundle to Zenodo and return what happened.

    With ``concept_id`` (the concept record of an earlier deposit), a new version of it; otherwise a new
    record. Files go to the deposition's bucket; the metadata is set from ``zenodo-metadata.json``. The
    deposition stays a draft unless ``publish``.
    """
    token = token or os.environ.get("ZENODO_TOKEN", "")
    if not token:
        raise ZenodoError("No Zenodo token: set ZENODO_TOKEN (the workflow maps ZENODO_SANDBOX_TOKEN or "
                          "ZENODO_TOKEN to it).")
    files = bundle_files(bundle)
    metadata = json.loads((bundle / "zenodo-metadata.json").read_text(encoding="utf-8"))
    # Sent explicitly, so a new version does not keep the date inherited from the previous one.
    metadata["metadata"].setdefault("publication_date", date.today().isoformat())
    api = SANDBOX_API if sandbox else ZENODO_API
    own = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(120.0, write=1800.0), follow_redirects=True)
    http.headers["Authorization"] = f"Bearer {token}"
    try:
        if concept_id:
            concept = _concept_number(concept_id)
            # GET /api/records/<concept> resolves to the latest published version.
            latest = _check(http.get(f"{api}/records/{concept}"), f"reading record {concept}", (200,)).json()
            if str(latest.get("conceptrecid")) != concept:
                raise ZenodoError(f"Record {concept} is not a concept record (its concept is "
                                  f"{latest.get('conceptrecid')}); pass the concept record id.")
            # The new version must be made from the latest version's id, not the concept id.
            made = _check(http.post(f"{api}/deposit/depositions/{latest['id']}/actions/newversion"),
                          "a new version", (201,)).json()
            draft = _check(http.get(made["links"]["latest_draft"]), "reading the new version", (200,)).json()
            for old in draft.get("files") or []:
                _check(http.delete(f"{api}/deposit/depositions/{draft['id']}/files/{old['id']}"),
                       f"removing the earlier version's {old.get('filename')}", (204,))
        else:
            draft = _check(http.post(f"{api}/deposit/depositions", json={}), "a new deposition", (201,)).json()
        dep_id, bucket = draft["id"], draft["links"]["bucket"]

        for path in files:
            md5 = _md5(path)
            for attempt in range(3):
                with path.open("rb") as fh:
                    resp = http.put(f"{bucket}/{path.name}", content=fh,
                                    headers={"Content-Type": "application/octet-stream"})
                if resp.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                    break
                time.sleep(pause * (attempt + 1))
            got = _check(resp, f"the upload of {path.name}", (200, 201)).json()
            if str(got.get("checksum", "")).removeprefix("md5:") not in ("", md5):
                raise ZenodoError(f"{path.name} arrived with checksum {got['checksum']}, expected md5:{md5}.")

        updated = _check(http.put(f"{api}/deposit/depositions/{dep_id}", json=metadata), "the metadata",
                         (200,)).json()
        result = {"id": dep_id, "sandbox": sandbox, "files": len(files),
                  "bytes": sum(p.stat().st_size for p in files),
                  "html": updated.get("links", {}).get("html") or draft.get("links", {}).get("html"),
                  "conceptrecid": updated.get("conceptrecid") or draft.get("conceptrecid"),
                  "published": False, "doi": None, "conceptdoi": None}
        if publish:
            done = _check(http.post(f"{api}/deposit/depositions/{dep_id}/actions/publish"), "publishing",
                          (202,)).json()
            result.update(published=True, doi=done.get("doi"), conceptdoi=done.get("conceptdoi"),
                          html=done.get("doi_url") or done.get("record_url") or result["html"])
        return result
    finally:
        if own:
            http.close()


# ── command line ───────────────────────────────────────────────────────────────────────────────────────────


def _human(n: int) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1000
    return str(n)


def _summary(manifest: dict[str, Any], bundle: Path) -> str:
    files = [*manifest["files"], {"name": "MANIFEST.json", "bytes": (bundle / "MANIFEST.json").stat().st_size}]
    lines = [f"Snapshot {manifest['snapshot_date']} from {manifest['source']}",
             f"Catalogued: {manifest['catalogued']['stations']:,} stations from "
             f"{manifest['catalogued']['sources']} sources. Mirrored: {manifest['mirrored']['stations']:,} "
             f"stations from {manifest['mirrored']['sources']} sources.",
             f"{len(files)} files, {_human(sum(f['bytes'] for f in files))}:"]
    lines += [f"  {f['name']:<44} {_human(f['bytes']):>10}" for f in files]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aquascope.archive.zenodo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the deposit bundle (no Zenodo)")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--archive", type=Path, help="a local Archive folder (default: download from Hugging Face)")
    b.add_argument("--revision", help="the Hugging Face commit (default: main's current commit; with --archive, "
                                      "the commit the folder came from, or 'local')")
    b.add_argument("--date", help="snapshot date, YYYY-MM-DD (default: today)")
    b.add_argument("--download-dir", type=Path, help="where to download the Archive (default: <out>.download)")
    b.add_argument("--dry-run", action="store_true", help="also print the Zenodo metadata JSON")
    d = sub.add_parser("deposit", help="upload a built bundle to Zenodo (token in ZENODO_TOKEN)")
    d.add_argument("--bundle", type=Path, required=True)
    d.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    d.add_argument("--concept-id", default="", help="the concept record of an earlier deposit, for a new version")
    d.add_argument("--publish", action="store_true", help="publish (default: leave a draft to check on Zenodo)")
    args = parser.parse_args(argv)

    if args.cmd == "build":
        if args.archive:
            archive, revision = args.archive, args.revision or "local"
        else:
            archive, revision = download(args.download_dir or args.out.with_name(args.out.name + ".download"),
                                         revision=args.revision)
        manifest = build(archive, args.out, revision=revision, snapshot_date=args.date)
        print(_summary(manifest, args.out))
        if args.dry_run:
            print(json.dumps({"metadata": manifest["zenodo_metadata"]}, indent=2, ensure_ascii=False))
        print("Built only: nothing was sent to Zenodo.")
        return 0

    try:
        result = deposit(args.bundle, sandbox=args.sandbox, concept_id=args.concept_id or None,
                         publish=args.publish)
    except ZenodoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    where = "sandbox.zenodo.org" if result["sandbox"] else "zenodo.org"
    state = f"published, DOI {result['doi']} (concept {result['conceptdoi']})" if result["published"] else \
        "left as a draft: check it on Zenodo and publish it there"
    print(f"Deposition {result['id']} on {where}: {result['files']} files, {_human(result['bytes'])}, {state}.")
    print(f"Open it: {result['html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
