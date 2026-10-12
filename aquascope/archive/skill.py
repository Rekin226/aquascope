"""The monthly skill run behind the evidence ladder (#518): every model scored at every Archive gauge.

``model-skill.yml`` runs :func:`build` once a month on the gauges whose daily discharge the Archive mirrors and
publishes ``skill/model_skill.parquet`` and ``skill/manifest.json`` to the dataset (nothing outside ``skill/``).
One row per gauge and model, the columns of :func:`aquascope.evidence.score` plus the gauge's ``source``,
``station_id``, ``lat``, ``lon``, ``computed_at`` and ``is_best``.

What is read here, and why here:

* **GEOGLOWS v2** from its daily Zarr on AWS (``geoglows-v2/retrospective/daily.zarr``, CC BY 4.0): two chunks
  per reach, so a thousand gauges cost a few GB, not a thousand slow API calls. The reach is chosen exactly as
  the page chooses it (:func:`aquascope.evidence.geoglows_site`).
* **NWM v3.0 retrospective** (``noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr``, public domain): the
  store is chunked 672 hours by 30,000 reaches, so one reach's 44 years touch about 570 chunks of 4 to 7 MB.
  Gauges that share a reach chunk are read together, the window is the last ``nwm_years`` of the record (it ends
  in February 2023), and ``nwm_max_columns`` caps how many reach chunks a run reads; what is left out is logged
  in the manifest and taken next month. The reach is the one NWM's own gauge table links to the USGS site
  (``gage_id``), else the NLDI's COMID for the site.
* **Google GRRR** (``gs://flood-forecasting/hydrologic_predictions/model_id_8583a5c2_v0/reanalysis``, CC BY 4.0,
  daily m3/s from 1980 to 2023 at the outlets of HydroBASINS level-12 basins): the bucket sends no CORS headers,
  so only CI reads it. The gauge's own level-12 sub-basin (``basins/station_catchments.parquet``) is the site;
  its upstream area against the gauge's catchment is recorded as the area ratio.
* **GloFAS** is not scored here: Open-Meteo's free tier counts a request longer than two weeks as several calls,
  so decades at thousands of gauges would exhaust it. The page scores GloFAS live for the gauge it opens.

The heavy readers need ``zstandard`` (NWM) and ``numcodecs`` (GEOGLOWS, GRRR); the workflow installs them.
``python -m aquascope.archive.skill`` and ``aquascope evidence build`` are the same entry point.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NWM_ZARR = "https://noaa-nwm-retrospective-3-0-pds.s3.amazonaws.com/CONUS/zarr/chrtout.zarr"
GRRR_ZARR = ("https://storage.googleapis.com/flood-forecasting/hydrologic_predictions/model_id_8583a5c2_v0/"
             "reanalysis/streamflow.zarr")
GEOGLOWS_ZARR = "https://geoglows-v2.s3.us-west-2.amazonaws.com/retrospective/daily.zarr"
NLDI_SITE = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{site}?f=json"

NWM_EPOCH = "1979-02-01T01:00:00"        # time units of chrtout.zarr: hours since this
GRRR_EPOCH = "1980-01-01"                # days since
GEOGLOWS_EPOCH = "1940-01-01"            # seconds since

MANIFEST_VERSION = 1
DEFAULT_MAX_GAUGES = 4000
DEFAULT_NWM_YEARS = 10
DEFAULT_NWM_MAX_COLUMNS = 24
DEFAULT_GRRR_MAX_CHUNKS = 2000
SMOKE_GAUGES = 6
CI_MODELS = ("geoglows", "nwm", "grrr")


# ── HTTP and chunk decoding (tests replace _get) ────────────────────────────

_HTTP: Any = None
_HTTP_LOCK = threading.Lock()


def _get(url: str) -> bytes:
    """The bytes at ``url`` (three tries); a missing object raises ``FileNotFoundError``."""
    global _HTTP
    import httpx

    with _HTTP_LOCK:  # the readers fetch chunks from several threads; one shared client
        if _HTTP is None:
            _HTTP = httpx.Client(timeout=180.0, follow_redirects=True)
    last: Exception | None = None
    for attempt in range(3):
        try:
            r = _HTTP.get(url)
            if r.status_code in (403, 404):
                raise FileNotFoundError(url)
            r.raise_for_status()
            return r.content
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def _decode(raw: bytes, compressor: dict[str, Any] | None) -> bytes:
    cid = (compressor or {}).get("id")
    if cid is None:
        return raw
    if cid == "zstd":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - the workflow installs it
            raise ImportError("reading the NWM store needs zstandard (pip install zstandard)") from exc
        return zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 31)
    if cid == "blosc":
        try:
            import numcodecs
        except ImportError as exc:  # pragma: no cover - the workflow installs it
            raise ImportError("reading the GEOGLOWS and GRRR stores needs numcodecs (pip install numcodecs)") from exc
        return bytes(numcodecs.Blosc().decode(raw))
    raise ValueError(f"unsupported Zarr compressor {cid!r}")


class _Zarr:
    """A Zarr v2 store over HTTP, read a chunk at a time (consolidated metadata)."""

    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.meta = json.loads(_get(f"{self.base}/.zmetadata"))["metadata"]

    def array(self, name: str) -> dict[str, Any]:
        return self.meta[f"{name}/.zarray"]

    def attrs(self, name: str) -> dict[str, Any]:
        return self.meta.get(f"{name}/.zattrs") or {}

    def chunk(self, name: str, *idx: int) -> Any:
        import numpy as np

        a = self.array(name)
        raw = _get(f"{self.base}/{name}/{'.'.join(str(i) for i in idx)}")
        arr = np.frombuffer(_decode(raw, a.get("compressor")), dtype=np.dtype(a["dtype"]))
        return arr.reshape(a["chunks"])

    def whole(self, name: str) -> Any:
        """A 1-D array, every chunk read and trimmed to its shape."""
        import numpy as np

        a = self.array(name)
        n, c = int(a["shape"][0]), int(a["chunks"][0])
        parts = [self.chunk(name, i) for i in range((n + c - 1) // c)]
        return np.concatenate(parts)[:n]


def _daily_from(values: Any, times: Any) -> Any:
    import pandas as pd

    s = pd.Series(values, index=pd.DatetimeIndex(times))
    s = s[s.notna() & (s >= 0)]
    return s.resample("D").mean().dropna() if len(s) else s


# ── GEOGLOWS v2 (the daily Zarr) ────────────────────────────────────────────


def geoglows_daily(river_ids: list[int], *, store: _Zarr | None = None, workers: int = 8) -> dict[int, Any]:
    """Daily discharge per reach from the GEOGLOWS v2 retrospective Zarr (reaches sharing a chunk read once,
    ``workers`` reach chunks at a time: a full run reads thousands of them)."""
    import numpy as np
    import pandas as pd

    if not river_ids:
        return {}
    z = store or _Zarr(GEOGLOWS_ZARR)
    ids = z.whole("river_id").astype(np.int64)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    seconds = z.whole("time").astype(float)
    times = pd.Timestamp(GEOGLOWS_EPOCH) + pd.to_timedelta(seconds, unit="s")
    qa = z.array("Q")
    tch, cch = int(qa["chunks"][0]), int(qa["chunks"][1])
    n_t = int(qa["shape"][0])
    by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rid in sorted(set(int(r) for r in river_ids)):
        k = int(np.searchsorted(sorted_ids, rid))
        if k < len(sorted_ids) and sorted_ids[k] == rid:
            col = int(order[k])
            by_col[col // cch].append((rid, col % cch))
    def read(cc: int) -> dict[int, Any]:
        blocks = [z.chunk("Q", ti, cc) for ti in range((n_t + tch - 1) // tch)]
        full = np.concatenate(blocks, axis=0)[:n_t]
        return {rid: _daily_from(full[:, j].astype(float), times) for rid, j in by_col[cc]}

    out: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for part in pool.map(read, list(by_col)):
            out.update(part)
    return out


# ── NWM v3.0 retrospective ──────────────────────────────────────────────────


def usgs_site_number(station_id: str) -> str | None:
    """The USGS site number in a catalog id ("USGS-01013500", "CA574-09527500" -> "01013500" / "09527500")."""
    m = re.search(r"(\d{8,15})$", str(station_id))
    return m.group(1) if m else None


def nwm_feature_ids(sites: list[str], *, store: _Zarr | None = None, nldi: bool = True,
                    max_nldi: int = 400) -> dict[str, tuple[int, str]]:
    """USGS site number -> (NWM feature_id, how it was found): NWM's own gauge link (``gage_id``) first, then
    the NLDI's COMID for the site (NHDPlus v2 COMIDs are NWM's feature ids)."""
    import numpy as np

    z = store or _Zarr(NWM_ZARR)
    gage = np.char.strip(z.whole("gage_id").astype("S15")).astype(str)
    fid = z.whole("feature_id").astype(np.int64)
    linked = {g: int(f) for g, f in zip(gage, fid) if g}
    out: dict[str, tuple[int, str]] = {}
    missing = []
    for s in sites:
        if s in linked:
            out[s] = (linked[s], "nwm_gage")
        else:
            missing.append(s)
    if nldi:
        for s in missing[:max_nldi]:
            try:
                feats = json.loads(_get(NLDI_SITE.format(site=s))).get("features") or []
                comid = (feats[0].get("properties") or {}).get("comid") if feats else None
                if comid:
                    out[s] = (int(comid), "nldi_comid")
            except Exception as exc:  # noqa: BLE001 - a site NLDI does not know is left out
                logger.info("NLDI has no COMID for USGS-%s: %s", s, exc)
    return out


def nwm_daily(feature_ids: list[int], *, years: int = DEFAULT_NWM_YEARS, max_columns: int = DEFAULT_NWM_MAX_COLUMNS,
              workers: int = 8, store: _Zarr | None = None, dropped: dict[str, Any] | None = None,
              last_scored: dict[int, str] | None = None) -> dict[int, Any]:
    """Daily mean discharge per NWM reach over the last ``years`` of the retrospective.

    Reaches are grouped by the 30,000-reach chunk they sit in; at most ``max_columns`` such chunks are read and
    the rest are reported in ``dropped``. ``last_scored`` (feature id -> when its row was last computed) puts the
    chunks whose reaches were never scored, or scored longest ago, first, so the chunks left out this month come
    first next month; among equals, the chunks holding the most gauges go first.
    """
    import numpy as np
    import pandas as pd

    if not feature_ids:
        return {}
    z = store or _Zarr(NWM_ZARR)
    fid = z.whole("feature_id").astype(np.int64)
    pos = {int(f): i for i, f in enumerate(fid)}
    sa = z.array("streamflow")
    tch, cch = int(sa["chunks"][0]), int(sa["chunks"][1])
    n_t = int(sa["shape"][0])
    attrs = z.attrs("streamflow")
    scale = float(attrs.get("scale_factor", 1.0))
    missing = attrs.get("missing_value")
    by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for f in sorted(set(int(x) for x in feature_ids)):
        if f in pos:
            by_col[pos[f] // cch].append((f, pos[f] % cch))
    seen = last_scored or {}
    cols = sorted(by_col, key=lambda c: (min(seen.get(f, "") for f, _j in by_col[c]), -len(by_col[c]), c))
    keep, skip = cols[:max(0, int(max_columns))], cols[max(0, int(max_columns)):]
    if dropped is not None and skip:
        dropped["nwm_columns_over_cap"] = len(skip)
        dropped["nwm_gauges_over_cap"] = sum(len(by_col[c]) for c in skip)
    n_chunks = (n_t + tch - 1) // tch
    first = max(0, n_chunks - int(math.ceil(years * 8766 / tch)))
    t_chunks = list(range(first, n_chunks))
    epoch = pd.Timestamp(NWM_EPOCH)
    time_blocks = {ti: z.chunk("time", ti).astype(np.int64) for ti in t_chunks}

    out: dict[int, Any] = {}
    for c in keep:
        members = by_col[c]
        cols_j = np.array([j for _f, j in members])

        def read(ti: int, c: int = c, cols_j: Any = cols_j) -> tuple[int, Any]:
            block = z.chunk("streamflow", ti, c)[:, cols_j].astype(float)
            if missing is not None:
                block[block == float(missing)] = np.nan
            return ti, block * scale

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            parts = dict(pool.map(read, t_chunks))
        vals = np.concatenate([parts[ti] for ti in t_chunks], axis=0)
        hours = np.concatenate([time_blocks[ti] for ti in t_chunks])
        # the last time chunk is padded past the end of the record (hour 0, missing flow): keep only real hours
        n = min(len(hours), len(vals), n_t - first * tch)
        valid = hours[:n] >= 0
        times = epoch + pd.to_timedelta(hours[:n][valid], unit="h")
        for k, (f, _j) in enumerate(members):
            out[f] = _daily_from(vals[:n][valid, k], times)
        logger.info("NWM reach chunk %d: %d gauges, %d time chunks", c, len(members), len(t_chunks))
    return out


# ── Google GRRR ─────────────────────────────────────────────────────────────


def grrr_daily(hybas_ids: list[int], *, max_chunks: int = DEFAULT_GRRR_MAX_CHUNKS, store: _Zarr | None = None,
               dropped: dict[str, Any] | None = None, workers: int = 8) -> dict[int, Any]:
    """Daily discharge per HydroBASINS level-12 outlet from the GRRR reanalysis (128 basins a chunk, ``workers``
    chunks at a time)."""
    import numpy as np
    import pandas as pd

    if not hybas_ids:
        return {}
    z = store or _Zarr(GRRR_ZARR)
    gid = z.whole("gauge_id").astype(str)
    sa = z.array("streamflow")
    rch, tch = int(sa["chunks"][0]), int(sa["chunks"][1])
    n_t = int(sa["shape"][1])
    days = z.whole("time").astype(np.int64)
    times = pd.Timestamp(GRRR_EPOCH) + pd.to_timedelta(days, unit="D")
    by_row: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for h in sorted(set(int(x) for x in hybas_ids)):
        key = f"hybas_{h}"
        k = int(np.searchsorted(gid, key))
        if k < len(gid) and gid[k] == key:
            by_row[k // rch].append((h, k % rch))
    rows = sorted(by_row)
    keep, skip = rows[:max(0, int(max_chunks))], rows[max(0, int(max_chunks)):]
    if dropped is not None and skip:
        dropped["grrr_chunks_over_cap"] = len(skip)
    def read(r: int) -> dict[int, Any]:
        block = np.concatenate([z.chunk("streamflow", r, ti) for ti in range((n_t + tch - 1) // tch)], axis=1)[:, :n_t]
        return {h: _daily_from(block[i].astype(float), times) for h, i in by_row[r]}

    out: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for part in pool.map(read, keep):
            out.update(part)
    return out


# ── inputs ──────────────────────────────────────────────────────────────────


def _load_inputs(archive: Path) -> tuple[list[dict[str, Any]], Any, Any]:
    """The catalog rows, every discharge bundle as one frame, and the station catchments (or None)."""
    import pandas as pd

    from aquascope.archive.catalog import load_stations

    catalog = load_stations(path=archive / "stations.parquet")
    parts = []
    for p in sorted((archive / "obs" / "discharge").glob("*.parquet")):
        df = pd.read_parquet(p, columns=["station_id", "date", "value"])
        df["source"] = p.stem
        parts.append(df)
    obs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["station_id", "date", "value",
                                                                                  "source"])
    cpath = archive / "basins" / "station_catchments.parquet"
    catch = pd.read_parquet(cpath) if cpath.exists() else None
    return catalog, obs, catch


def _previous(out_dir: Path, archive: Path) -> Any:
    import pandas as pd

    for p in (out_dir / "model_skill.parquet", archive / "skill" / "model_skill.parquet"):
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception as exc:  # noqa: BLE001 - a broken previous table only costs the carry-over
                logger.warning("previous skill table unreadable (%s): %s", p, exc)
    return None


def build(archive: str | Path, out: str | Path, *, max_gauges: int = DEFAULT_MAX_GAUGES,
          models: tuple[str, ...] = CI_MODELS, years: int = 30, nwm_years: int = DEFAULT_NWM_YEARS,
          nwm_max_columns: int = DEFAULT_NWM_MAX_COLUMNS, grrr_max_chunks: int = DEFAULT_GRRR_MAX_CHUNKS,
          workers: int = 8, smoke: bool = False) -> dict[str, Any]:
    """Score the CI models at the Archive gauges with a mirrored daily discharge record and write
    ``<out>/model_skill.parquet`` and ``<out>/manifest.json``.

    ``archive`` is a local copy of the dataset holding ``stations.parquet``, ``obs/discharge/*.parquet`` and,
    when published, ``basins/station_catchments.parquet`` and the previous ``skill/model_skill.parquet``. Gauges
    never scored come first, then the longest unscored ones; ``max_gauges`` caps a run and the rest keep last
    month's rows. ``smoke`` takes :data:`SMOKE_GAUGES` gauges and marks the manifest so it is never published.
    Returns the manifest.
    """
    import pandas as pd

    from aquascope import evidence

    t0 = time.perf_counter()
    archive, out_dir = Path(archive), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog, obs, catch = _load_inputs(archive)
    by_key = {(r["source"], str(r["station_id"])): r for r in catalog}
    cat_rows = {}
    if catch is not None:
        for r in catch.to_dict("records"):
            cat_rows[(r["source"], str(r["station_id"]))] = r
    dropped: dict[str, Any] = {"short_record": 0, "not_in_catalog": 0, "over_cap": 0}
    min_days = evidence.GRADING["min_overlap_days"]
    candidates = []
    for (src, sid), g in obs.groupby(["source", "station_id"]):
        if len(g) < min_days:
            dropped["short_record"] += 1
            continue
        if (src, str(sid)) not in by_key:
            dropped["not_in_catalog"] += 1
            continue
        candidates.append((src, str(sid)))
    prev = _previous(out_dir, archive)
    last: dict[tuple[str, str], str] = {}
    nwm_last: dict[int, str] = {}
    if prev is not None and len(prev):
        for r in prev[["source", "station_id", "model", "site_id", "computed_at"]].to_dict("records"):
            k, stamp = (r["source"], str(r["station_id"])), str(r["computed_at"])
            last[k] = max(last.get(k, ""), stamp)
            if r["model"] == "nwm" and str(r["site_id"]).isdigit():
                nwm_last[int(r["site_id"])] = max(nwm_last.get(int(r["site_id"]), ""), stamp)
    candidates.sort(key=lambda k: (last.get(k, ""), k))
    if smoke:
        # a few gauges from each source in turn, and a light NWM read, so a smoke run touches every reader cheaply
        by_src: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for k in candidates:
            by_src[k[0]].append(k)
        mixed = [k for group in zip(*[iter(v) for v in by_src.values()]) for k in group] if by_src else []
        mixed += [k for k in candidates if k not in mixed]
        candidates = mixed
        nwm_years, nwm_max_columns = min(nwm_years, 1), min(nwm_max_columns, 2)
    cap = SMOKE_GAUGES if smoke else int(max_gauges)
    run, rest = candidates[:cap], candidates[cap:]
    dropped["over_cap"] = len(rest)
    logger.info("skill run: %d gauges this run, %d over the cap, %d short records", len(run), len(rest),
                dropped["short_record"])

    series: dict[tuple[str, str], Any] = {}
    obs_idx = obs.set_index(["source", "station_id"]).sort_index()
    for k in run:
        g = obs_idx.loc[k]
        s = pd.Series(g["value"].to_numpy(dtype=float), index=pd.to_datetime(g["date"]))
        s = evidence._daily(s)
        s = s[s.index >= s.index.max() - pd.Timedelta(days=int(years * 365.25))]
        series[k] = s

    def area_of(k: tuple[str, str]) -> float | None:
        c = cat_rows.get(k) or {}
        for v in (c.get("area_km2"), (by_key[k].get("extra") or {}).get("catchment_area_km2")):
            try:
                if v is not None and float(v) > 0:
                    return float(v)
            except (TypeError, ValueError):
                continue
        return None

    sims: dict[str, dict[tuple[str, str], tuple[Any, dict[str, Any]]]] = {m: {} for m in models}
    errors: dict[str, int] = defaultdict(int)

    if "geoglows" in models:
        def site(k: tuple[str, str]) -> tuple[tuple[str, str], dict[str, Any]]:
            r = by_key[k]
            try:
                return k, evidence.geoglows_site(float(r["latitude"]), float(r["longitude"]), area_of(k))
            except Exception as exc:  # noqa: BLE001
                return k, {"error": str(exc)}

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            sites = dict(pool.map(site, run))
        ok = {k: s for k, s in sites.items() if not s.get("error")}
        errors["geoglows_no_reach"] += len(sites) - len(ok)
        try:
            got = geoglows_daily([int(s["site_id"]) for s in ok.values()], workers=workers)
        except Exception as exc:  # noqa: BLE001 - one model failing leaves the others
            logger.warning("GEOGLOWS Zarr unreadable: %s", exc)
            got, errors["geoglows_read_failed"] = {}, len(ok)
        for k, s in ok.items():
            if int(s["site_id"]) in got:
                sims["geoglows"][k] = (got[int(s["site_id"])], s)

    if "grrr" in models:
        want = {}
        for k in run:
            c = cat_rows.get(k)
            if c and c.get("hybas_id"):
                want[k] = int(c["hybas_id"])
        try:
            got = grrr_daily(list(want.values()), max_chunks=grrr_max_chunks, dropped=dropped, workers=workers)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GRRR store unreadable: %s", exc)
            got, errors["grrr_read_failed"] = {}, len(want)
        for k, h in want.items():
            if h in got:
                up = (cat_rows.get(k) or {}).get("up_area")
                a = area_of(k)
                ratio = round(float(up) / a, 3) if up and a else None
                sims["grrr"][k] = (got[h], {"site_id": f"hybas_{h}", "model_area_km2": _f(up), "area_ratio": ratio,
                                            "match": "sub_basin"})
        errors["grrr_no_basin"] += len(run) - len(want)

    if "nwm" in models:
        sites_us = {k: usgs_site_number(k[1]) for k in run if k[0] == "usgs"}
        sites_us = {k: s for k, s in sites_us.items() if s}
        if sites_us:
            try:
                fids = nwm_feature_ids(sorted(set(sites_us.values())))
                got = nwm_daily([f for f, _how in fids.values()], years=nwm_years, max_columns=nwm_max_columns,
                                workers=workers, dropped=dropped, last_scored=nwm_last)
            except Exception as exc:  # noqa: BLE001
                logger.warning("NWM store unreadable: %s", exc)
                fids, got, errors["nwm_read_failed"] = {}, {}, len(sites_us)
            errors["nwm_no_reach"] += sum(1 for s in sites_us.values() if s not in fids)
            for k, s in sites_us.items():
                hit = fids.get(s)
                if hit and hit[0] in got:
                    sims["nwm"][k] = (got[hit[0]], {"site_id": str(hit[0]), "model_area_km2": None,
                                                    "area_ratio": None, "match": hit[1]})

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, Any]] = []
    for k in run:
        r = by_key[k]
        mine = []
        for m in models:
            if k not in sims[m]:
                continue
            sim, site = sims[m][k]
            row = evidence.score(series[k], sim, model=m, site=site)
            row.update(source=k[0], station_id=k[1], lat=_f(r["latitude"]), lon=_f(r["longitude"]),
                       area_km2=area_of(k), computed_at=now)
            mine.append(row)
        rows.extend(mine)
    table = pd.DataFrame(rows)
    if prev is not None and len(prev) and not smoke:
        # Last month's row for every (gauge, model) this run did not score: the gauges over the cap, and the
        # models a gauge's run left out (an NWM chunk over its cap, a store that could not be read this time).
        # Gauges the Archive no longer mirrors are let go.
        fresh = {(s, str(i), m) for s, i, m in zip(table["source"], table["station_id"], table["model"])} \
            if len(table) else set()
        mirrored = set(candidates)
        carried = prev[[(s, str(i), m) not in fresh and (s, str(i)) in mirrored
                        for s, i, m in zip(prev["source"], prev["station_id"], prev["model"])]]
        table = pd.concat([table, carried], ignore_index=True) if len(table) else carried
    table = _mark_best(_tidy(table))
    path = out_dir / "model_skill.parquet"
    table.to_parquet(path, index=False)
    counts = {m: int((table["model"] == m).sum()) if len(table) else 0 for m in models}
    graded = int(table["grade"].notna().sum()) if len(table) else 0
    manifest = {
        "version": MANIFEST_VERSION, "built_at": now, "smoke": bool(smoke),
        "seconds": round(time.perf_counter() - t0, 1),
        "file": "skill/model_skill.parquet", "n_rows": int(len(table)), "n_graded": graded,
        "n_gauges": int(table[["source", "station_id"]].drop_duplicates().shape[0]) if len(table) else 0,
        "scored_this_run": len(run), "rows_by_model": counts, "dropped": dropped, "failures": dict(errors),
        "window_years": years, "nwm_years": nwm_years,
        "models": {m: {k: evidence.MODELS[m][k] for k in ("label", "licence", "attribution", "site")} for m in models},
        "not_scored_here": {"glofas": "scored live in the Explorer: Open-Meteo's free tier counts a long request "
                                      "as several calls, too many for every gauge each month"},
        "grading": evidence.GRADING,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")
    logger.info("skill table: %d rows (%d graded) for %d gauges -> %s", manifest["n_rows"], graded,
                manifest["n_gauges"], path)
    return manifest


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


_COLUMNS = ["source", "station_id", "lat", "lon", "area_km2", "model", "label", "grade", "why", "is_best", "kge", "r",
            "alpha", "beta", "nse", "pbias", "mean_gauge", "mean_model", "n_days", "start", "end", "flood_years",
            "q2_gauge", "q2_model", "q2_error_pct", "q10_gauge", "q10_model", "q10_error_pct", "q100_gauge",
            "q100_model", "q100_error_pct", "site_id", "match", "distance_m", "model_area_km2", "area_ratio",
            "sentence", "licence", "computed_at"]


def _tidy(df: Any) -> Any:
    """The published columns, in order, with stable types (a column a run happened not to fill still exists)."""
    import pandas as pd

    df = df.copy() if df is not None and len(df) else pd.DataFrame(columns=_COLUMNS)
    for c in _COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[_COLUMNS]
    for c in ("source", "station_id", "model", "label", "grade", "why", "site_id", "match", "start", "end",
              "sentence", "licence", "computed_at"):
        df[c] = df[c].astype("string")
    for c in ("lat", "lon", "area_km2", "kge", "r", "alpha", "beta", "nse", "pbias", "mean_gauge", "mean_model",
              "q2_gauge", "q2_model", "q2_error_pct", "q10_gauge", "q10_model", "q10_error_pct", "q100_gauge",
              "q100_model", "q100_error_pct", "distance_m", "model_area_km2", "area_ratio"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in ("n_days", "flood_years"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["is_best"] = df["is_best"].fillna(False).astype(bool)
    return df.sort_values(["source", "station_id", "model"]).reset_index(drop=True)


def _mark_best(df: Any) -> Any:
    """``is_best`` on the graded row with the highest KGE at each gauge (fresh and carried rows alike)."""
    if not len(df):
        return df
    df["is_best"] = False
    graded = df[df["grade"].notna() & df["kge"].notna()]
    if len(graded):
        df.loc[graded.groupby(["source", "station_id"])["kge"].idxmax().to_numpy(), "is_best"] = True
    return df


def publish(out: str | Path, *, repo_id: str = "Rekin226/aquascope-gauges", token: str | None = None) -> str:
    """Upload ``<out>`` as ``skill/`` in the Archive dataset; refuses a smoke build."""
    import shutil

    from aquascope.archive.publish import publish_folder

    src = Path(out)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("smoke"):
        raise RuntimeError("a smoke build is never published: it would replace the full table")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "skill"
        stage.mkdir()
        for name in ("model_skill.parquet", "manifest.json"):
            shutil.copy2(src / name, stage / name)
        return publish_folder(Path(tmp), repo_id, token=token or os.environ.get("HF_TOKEN"),
                              commit_message=f"model skill: {manifest['n_rows']} rows, {manifest['n_gauges']} gauges")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aquascope.archive.skill", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--archive", required=True, help="a local copy of the dataset (stations.parquet, obs/discharge)")
    b.add_argument("--out", required=True)
    b.add_argument("--max-gauges", type=int, default=DEFAULT_MAX_GAUGES)
    b.add_argument("--nwm-years", type=int, default=DEFAULT_NWM_YEARS)
    b.add_argument("--nwm-max-columns", type=int, default=DEFAULT_NWM_MAX_COLUMNS)
    b.add_argument("--grrr-max-chunks", type=int, default=DEFAULT_GRRR_MAX_CHUNKS)
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--smoke", action="store_true")
    p = sub.add_parser("publish")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default="Rekin226/aquascope-gauges")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.cmd == "build":
        m = build(args.archive, args.out, max_gauges=args.max_gauges, nwm_years=args.nwm_years,
                  nwm_max_columns=args.nwm_max_columns, grrr_max_chunks=args.grrr_max_chunks, workers=args.workers,
                  smoke=args.smoke)
        print(json.dumps({k: m[k] for k in ("n_rows", "n_graded", "n_gauges", "scored_this_run", "rows_by_model",
                                            "dropped", "failures")}, indent=1))
        return 0
    print(publish(args.out, repo_id=args.repo))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
