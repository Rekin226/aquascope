"""Read the published station catalog (the Archive) without any agency call.

``load_stations()`` downloads ``stations.parquet`` from the Hugging Face
dataset once a day into a local cache and returns plain dicts, so the MCP
server, the CLI and notebooks can answer "which gauges are near X" in
milliseconds. Falls back to ``stations.geojson`` when ``pyarrow`` is missing.
"""

from __future__ import annotations

import json
import logging
import os
import time
import unicodedata
from pathlib import Path
from typing import Any

import httpx

from aquascope.schemas.station import in_bbox

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "Rekin226/aquascope-gauges"
CACHE_TTL_SECONDS = 24 * 3600

_OVERRIDE: list[dict[str, Any]] | None = None
_OVERRIDE_VERSION = 0  # bumped by set_catalog, so a derived index knows its rows changed


def set_catalog(rows: list[dict[str, Any]] | None) -> None:
    """Make :func:`load_stations` return ``rows`` instead of downloading the catalog.

    Used by the Explorer's browser worker, which already holds the catalog in
    DuckDB-WASM and cannot use httpx or pyarrow; also handy in tests. Pass
    ``None`` to go back to the Hub.
    """
    global _OVERRIDE, _OVERRIDE_VERSION
    _OVERRIDE = list(rows) if rows is not None else None
    _OVERRIDE_VERSION += 1


def catalog_url(repo_id: str = DEFAULT_REPO_ID, filename: str = "stations.parquet") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def cache_dir() -> Path:
    root = os.environ.get("AQUASCOPE_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "aquascope")
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path, refresh: bool) -> Path:
    if dest.exists() and not refresh and time.time() - dest.stat().st_mtime < CACHE_TTL_SECONDS:
        return dest
    logger.info("Downloading %s", url)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        resp = client.get(url)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def load_stations(
    *, repo_id: str = DEFAULT_REPO_ID, refresh: bool = False, path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Return every station in the published catalog as a list of dicts.

    Keys: ``source, station_id, name, latitude, longitude, variables (list),
    period_start, period_end, url, river, country, agency, license,
    redistributable, extra (dict)``. ``path`` reads a local ``stations.parquet``
    (a fresh harvest) instead of the Hub.
    """
    if _OVERRIDE is not None and path is None:
        return _OVERRIDE
    try:
        import pyarrow.parquet as pq  # noqa: F401
    except ImportError:
        pq = None
    if path is not None and pq is None:
        raise ImportError("reading a local stations.parquet needs pyarrow (pip install 'aquascope[archive]')")
    if pq is not None:
        if path is not None:
            dest = Path(path)
        else:
            local = cache_dir() / f"{repo_id.replace('/', '__')}.parquet"
            dest = _download(catalog_url(repo_id, "stations.parquet"), local, refresh)
        return _rows_from_parquet(dest)
    local = cache_dir() / f"{repo_id.replace('/', '__')}.geojson"
    dest = _download(catalog_url(repo_id, "stations.geojson"), local, refresh)
    return _rows_from_geojson(dest)


def _rows_from_parquet(dest: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(dest, columns=[c for c in [
        "source", "station_id", "name", "latitude", "longitude", "variables", "period_start", "period_end",
        "url", "river", "country", "agency", "license", "redistributable", "extra",
    ]])
    rows = table.to_pylist()
    for r in rows:
        for k in ("period_start", "period_end"):
            if r.get(k) is not None:
                r[k] = r[k].isoformat()
        r["variables"] = list(r.get("variables") or [])
        r["extra"] = json.loads(r["extra"]) if r.get("extra") else {}
    return rows


def _rows_from_geojson(dest: Path) -> list[dict[str, Any]]:
    gj = json.loads(dest.read_text(encoding="utf-8"))
    rows = []
    for f in gj.get("features", []):
        p = dict(f.get("properties") or {})
        lon, lat = f["geometry"]["coordinates"]
        p.update({"latitude": lat, "longitude": lon, "variables": list(p.get("variables") or [])})
        rows.append(p)
    return rows


# Connectives that carry no signal in a multi-word query ("Thames at Kingston", "Rhône à Anthon").
_STOP_TOKENS = frozenset({
    "a", "an", "am", "at", "in", "on", "of", "the", "de", "du", "des", "la", "le", "les", "der", "die", "das",
    "river", "riviere", "fluss", "rio",
})


def fold_text(text: Any) -> str:
    """Lower-case, accent-stripped text for matching: ``"Rhône à Anthon"`` becomes ``"rhone a anthon"``."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def query_tokens(query: str | None) -> list[str]:
    """The words of a catalog query, folded; connectives are dropped when other words remain."""
    tokens = fold_text(query).split()
    if len(tokens) > 1:
        kept = [t for t in tokens if t not in _STOP_TOKENS]
        tokens = kept or tokens
    return tokens


def _query_score(row: dict[str, Any], tokens: list[str]) -> tuple[int, int]:
    """``(tokens matched in name, id or river, tokens matched in name or id)``: higher is better."""
    name = fold_text(row.get("name"))
    sid = fold_text(row.get("station_id"))
    river = fold_text(row.get("river"))
    in_name = sum(1 for t in tokens if t in name or t in sid)
    matched = sum(1 for t in tokens if t in name or t in sid or t in river)
    return matched, in_name


# (what the index was built from, {(source, station_id): (period_start, period_end)})
_PERIOD_INDEX: tuple[Any, dict[tuple[str, str], tuple[str | None, str | None]]] | None = None


def _rows_at_hand(repo_id: str = DEFAULT_REPO_ID) -> tuple[Any, list[dict[str, Any]] | None]:
    """The catalog rows already here, with a key that says which copy, or ``None``.

    The rows handed over with :func:`set_catalog` come first, then a copy
    :func:`load_stations` has cached on disk, whatever its age. Nothing is
    downloaded: a fetch should not wait on the Hub for a date it can do without.
    """
    if _OVERRIDE is not None:
        return ("override", _OVERRIDE_VERSION), _OVERRIDE
    parquet = cache_dir() / f"{repo_id.replace('/', '__')}.parquet"
    geojson = cache_dir() / f"{repo_id.replace('/', '__')}.geojson"
    for dest, reader in ((parquet, _rows_from_parquet), (geojson, _rows_from_geojson)):
        if not dest.exists():
            continue
        try:
            return (str(dest), dest.stat().st_mtime), reader(dest)
        except Exception as exc:  # noqa: BLE001 - a broken cache is a missing cache
            logger.info("could not read the cached catalog %s: %s", dest, exc)
    return None, None


def catalog_period(source: str, station_id: str) -> tuple[str | None, str | None]:
    """The catalog's ``(period_start, period_end)`` for one station, without a download.

    Reads the rows handed over with :func:`set_catalog` (the Explorer's worker)
    or the copy :func:`load_stations` already cached on disk (an MCP server or
    a CLI that has searched the catalog). Returns ``(None, None)`` when neither
    is at hand or the station is not listed, so the caller falls back to a
    generous window rather than waiting on the Hub. ISO date strings.
    """
    global _PERIOD_INDEX
    key, rows = _rows_at_hand()
    if rows is None:
        return None, None
    if _PERIOD_INDEX is None or _PERIOD_INDEX[0] != key:
        index: dict[tuple[str, str], tuple[str | None, str | None]] = {}
        for r in rows:
            start, end = r.get("period_start"), r.get("period_end")
            index[(str(r.get("source")), str(r.get("station_id")))] = (
                str(start)[:10] if start else None, str(end)[:10] if end else None,
            )
        _PERIOD_INDEX = (key, index)
    return _PERIOD_INDEX[1].get((source, station_id), (None, None))


def search_stations(
    rows: list[dict[str, Any]],
    *,
    bbox: tuple[float, float, float, float] | None = None,
    variable: str | None = None,
    sources: list[str] | None = None,
    query: str | None = None,
    near: tuple[float, float] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Filter the catalog: bbox, variable, sources, a name query, and optional nearest-first ordering.

    ``query`` is matched accent-folded and case-insensitively. One word is a
    substring of the station name or id, as before. Several words ("Kingston
    Thames", "Thames at Kingston") match when every word appears somewhere in
    the name, the id or the river; when nothing matches every word, the rows
    matching the most words come back first, name matches ahead of river-only
    ones. Connectives ("at", "river", "de", ...) are ignored in a multi-word query.
    """
    tokens = query_tokens(query)
    src = set(sources or [])
    scored: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for r in rows:
        if src and r["source"] not in src:
            continue
        if variable and variable not in (r.get("variables") or []):
            continue
        if bbox and not in_bbox(r["latitude"], r["longitude"], bbox):
            continue
        score = (0, 0)
        if len(tokens) == 1:
            t = tokens[0]
            if t not in fold_text(r.get("name")) and t not in fold_text(r.get("station_id")):
                continue
        elif tokens:
            score = _query_score(r, tokens)
            if not score[0]:
                continue
        scored.append((score, r))

    def distance(r: dict[str, Any]) -> float:
        lat0, lon0 = near  # type: ignore[misc]
        return (r["latitude"] - lat0) ** 2 + ((r["longitude"] - lon0) * 0.7) ** 2

    if len(tokens) > 1:
        # Every word matched first, then the most words; among equals, the nearest when a
        # position is given, else the shortest name (the most exact match).
        scored.sort(key=lambda sr: (
            -sr[0][0], -sr[0][1], distance(sr[1]) if near else len(fold_text(sr[1].get("name"))),
        ))
    elif near:
        scored.sort(key=lambda sr: distance(sr[1]))
    return [r for _, r in scored][: max(0, limit)]
