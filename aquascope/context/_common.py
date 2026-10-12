"""Shared plumbing for the place-context layers: HTTP, the Archive mirror, cells and the result shape."""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import math
import sys
import threading
from collections.abc import Callable, Iterable
from typing import Any

from aquascope.registry import CONTEXT_LAYERS

logger = logging.getLogger(__name__)

IS_EMSCRIPTEN = sys.platform == "emscripten"
DEFAULT_REPO_ID = "Rekin226/aquascope-gauges"
#: The folder of the Archive dataset that holds the context mirrors (aquascope.archive.context_mirror).
MIRROR_FOLDER = "context"
#: Mirror cells are CELL_DEG x CELL_DEG degrees, so a point or a small box touches one to four files.
CELL_DEG = 2

MIRROR_MISSING = ("The {what} mirror is not published to the Archive yet (it is built by the mirror-context "
                  "workflow), so this part of the context is empty for now.")

_memo: dict[str, Any] = {}
_memo_lock = threading.Lock()


def memo(key: str, build: Callable[[], Any]) -> Any:
    """Build a value once per process (station lists, tile indexes, the mirror manifest)."""
    with _memo_lock:
        if key in _memo:
            return _memo[key]
    value = build()
    with _memo_lock:
        _memo[key] = value
    return value


def clear_memo() -> None:
    with _memo_lock:
        _memo.clear()


def mirror_url(path: str, repo_id: str = DEFAULT_REPO_ID) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{MIRROR_FOLDER}/{path}"


def get_bytes(url: str, *, timeout: float = 60.0) -> bytes | None:
    """The body of ``url``, or None on a 404/403. Works in CPython (httpx) and in the browser (urllib)."""
    if IS_EMSCRIPTEN:  # pragma: no cover - only under Pyodide
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - https data hosts
                status, data = int(getattr(resp, "status", 200) or 0), resp.read()
        except urllib.error.HTTPError as exc:
            status, data = exc.code, b""
        # pyodide-http answers every status as a response rather than raising
        if status in (403, 404):
            return None
        if status == 0 or status >= 400:
            raise RuntimeError(f"{url}: HTTP {status or 'blocked (CORS or network)'}")
        return data
    from aquascope.utils.cog import _httpx_client

    resp = _httpx_client().get(url, timeout=timeout)
    if resp.status_code in (403, 404):
        return None
    if resp.status_code >= 400:
        raise RuntimeError(f"{url}: HTTP {resp.status_code}")
    return resp.content


def http_client() -> Any:
    """A CachedHTTPClient for the JSON and text the layers read (listings, station lists), cached a week."""
    def build() -> Any:
        from aquascope.utils.http_client import CachedHTTPClient

        return CachedHTTPClient(timeout=60.0, retries=2, cache_ttl_seconds=7 * 86400)

    return memo("http_client", build)


def manifest(repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any] | None:
    """The mirror's manifest.json (what is published, row counts, cells), or None when nothing is published."""
    def build() -> dict[str, Any] | None:
        try:
            raw = get_bytes(mirror_url("manifest.json", repo_id))
        except Exception as exc:  # noqa: BLE001 - a missing mirror must never break the other layers
            logger.info("context mirror manifest unreachable: %s", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return None

    return memo(f"manifest:{repo_id}", build)


def dataset_info(name: str, repo_id: str = DEFAULT_REPO_ID) -> dict[str, Any] | None:
    m = manifest(repo_id)
    if not m:
        return None
    info = (m.get("datasets") or {}).get(name)
    return info if isinstance(info, dict) else None


# ── geometry ─────────────────────────────────────────────────────────────────


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


def check_point(lat: float, lon: float) -> tuple[float, float]:
    lat, lon = float(lat), float(lon)
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise ValueError(f"({lat}, {lon}) is not a latitude/longitude")
    return lat, lon


def check_bbox(west: float, south: float, east: float, north: float) -> tuple[float, float, float, float]:
    west, south, east, north = (float(v) for v in (west, south, east, north))
    if not (-90 <= south < north <= 90) or not (-180 <= west <= 180 and -180 <= east <= 180) or west >= east:
        raise ValueError(f"({west}, {south}, {east}, {north}) is not a west, south, east, north box")
    return west, south, east, north


def radius_bbox(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    dlat = radius_km / 111.0
    dlon = radius_km / max(1e-6, 111.0 * math.cos(math.radians(min(89.0, abs(lat)))))
    return (max(-180.0, lon - dlon), max(-90.0, lat - dlat), min(180.0, lon + dlon), min(90.0, lat + dlat))


def cell_key(lat: float, lon: float, deg: int = CELL_DEG) -> str:
    """The mirror cell holding a point: ``n50_w002`` is the cell from 50 N, 2 W, ``deg`` degrees on a side."""
    lat0 = int(math.floor(lat / deg) * deg)
    lon0 = int(math.floor(lon / deg) * deg)
    lat0 = min(lat0, 90 - deg)
    lon0 = min(lon0, 180 - deg)
    return f"{'n' if lat0 >= 0 else 's'}{abs(lat0):02d}_{'e' if lon0 >= 0 else 'w'}{abs(lon0):03d}"


def cells_for_bbox(west: float, south: float, east: float, north: float, deg: int = CELL_DEG) -> list[str]:
    keys: list[str] = []
    lat = math.floor(south / deg) * deg
    while lat < north:
        lon = math.floor(west / deg) * deg
        while lon < east:
            keys.append(cell_key(lat + deg / 2, lon + deg / 2, deg))
            lon += deg
        lat += deg
    return sorted(set(keys))


def in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    west, south, east, north = bbox
    return south <= lat <= north and west <= lon <= east


def grid_points(bbox: tuple[float, float, float, float], n: int = 3) -> list[tuple[float, float]]:
    """``n`` x ``n`` sample points (lat, lon) at the centres of an even grid over the box."""
    west, south, east, north = bbox
    pts = []
    for i in range(n):
        for j in range(n):
            pts.append((south + (i + 0.5) * (north - south) / n, west + (j + 0.5) * (east - west) / n))
    return pts


# ── mirror cells ─────────────────────────────────────────────────────────────


def read_csv_gz(data: bytes) -> list[dict[str, str]]:
    text = gzip.decompress(data).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def read_cells(dataset: str, keys: Iterable[str], *, repo_id: str = DEFAULT_REPO_ID) -> list[dict[str, str]] | None:
    """Rows of a mirrored dataset in the given cells; None when the dataset is not published.

    The manifest lists the cells that exist, so empty cells cost no request.
    """
    info = dataset_info(dataset, repo_id)
    if info is None:
        return None
    present = set(info.get("cells") or [])
    rows: list[dict[str, str]] = []
    folder = info.get("folder") or dataset
    for key in keys:
        if present and key not in present:
            continue
        cache = f"cell:{repo_id}:{dataset}:{key}"
        cell_rows = memo(cache, lambda k=key: _fetch_cell(folder, k, repo_id))
        rows.extend(cell_rows)
    return rows


def _fetch_cell(folder: str, key: str, repo_id: str) -> list[dict[str, str]]:
    raw = get_bytes(mirror_url(f"{folder}/cells/{key}.csv.gz", repo_id))
    return read_csv_gz(raw) if raw else []


def num(value: Any) -> float | None:
    """A float from a CSV cell, None for blanks and the usual missing-value codes."""
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or v in (-99.0, -999.0, -9999.0):
        return None
    return v


# ── the result shape ─────────────────────────────────────────────────────────


def layer_result(layer: str, sources: Iterable[str], **fields: Any) -> dict[str, Any]:
    """Every layer answers with its name, a one-line ``summary`` and the licence of what it read."""
    metas = [CONTEXT_LAYERS[s] for s in sources]
    out: dict[str, Any] = {"layer": layer}
    out.update(fields)
    out["sources"] = [
        {"key": m.key, "label": m.label, "short": m.short or m.label, "licence": m.license,
         "attribution": m.attribution,
         "homepage": m.homepage, **({"citation": m.citation} if m.citation else {})}
        for m in metas
    ]
    out["attribution"] = "; ".join(m.attribution for m in metas)
    return out


def failed(layer: str, sources: Iterable[str], exc: BaseException) -> dict[str, Any]:
    return layer_result(layer, sources, ok=False, summary=f"Could not read this layer: {exc}",
                        error=f"{type(exc).__name__}: {exc}")


def fmt_int(x: float | None) -> str:
    return "unknown" if x is None else f"{x:,.0f}"
