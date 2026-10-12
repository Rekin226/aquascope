"""Poland: IMGW-PIB river gauges, the live network state plus the daily archive back to 1951.

The Institute of Meteorology and Water Management (IMGW-PIB) publishes the
national hydrological network on ``danepubliczne.imgw.pl``, open and keyless,
through two very different doors:

- ``/api/data/hydro/``: one GET returns the *current* state of every station
  (913 on 2026-09-14), each with WGS84 coordinates, water level, discharge,
  water temperature, ice and overgrowth codes and, unusually, the alarm and
  warning stages. Each value carries its own timestamp and they are often far
  apart: a level two hours old next to a discharge seven months old.
- ``/data/dane_pomiarowo_obserwacyjne/dane_hydrologiczne/dobowe/<year>/``: the
  daily archive, one directory per *hydrological* year from 1951. Through
  2022 a year is twelve zips, one per hydrological month
  (``codz_<year>_<mm>.zip``); from 2023 it is one zip (``codz_<year>.zip``).
  Ten columns, no header::

      station id, name, river, hydro_year, hydro_month, day,
      level_cm, discharge_cms, temp_c, calendar_month

  A hydrological year runs November to October, so it is published only once
  it has ended: the archive stops at the last October, and today's reading is
  in the live API.

Four traps, every one of them producing wrong data rather than an error if
skipped, and every one handled here:

1. **The date columns are the hydrological year.** Hydro month 01 is
   November of the previous calendar year. The tenth column is the calendar
   month; calendar months 11 and 12 belong to ``hydro_year - 1``.
2. **The quoting differs by file.** 2024 wraps every line in one more layer
   of quotes, with the inner quotes doubled (``""2024""``), so a plain CSV
   reader sees one field per line. 2025 and the monthly files quote fields
   normally, some with a leading space in the id.
3. **The delimiter and encoding differ by file.** 2023 is semicolon-separated
   UTF-8 with a byte-order mark; everything else seen is comma-separated
   cp1250. Both are sniffed per file.
4. **Water temperature uses 99.9 as missing** in the monthly and 2023 files
   and an empty field in the later ones.

Terms (``danepubliczne.imgw.pl/regulations``): re-use is open to everyone under
Poland's open data act and EU regulation 2023/138 on high-value datasets, on
condition that products state „Źródłem pochodzenia danych jest Instytut
Meteorologii i Gospodarki Wodnej – Państwowy Instytut Badawczy" and, where the
data were processed, „Dane IMGW-PIB zostały przetworzone". Both sit in the
registry attribution. Commercial use of basic-network data beyond the
high-value datasets needs an agreement with IMGW-PIB (§ 3 of the terms).

A full-record walk reads up to ~870 archive files (about 130 MB) and keeps
them parsed in memory for the rest of the run, roughly 0.5 GB, so that the
harvest can serve hundreds of stations from one download. Zips are cached on
disk under ``data/cache/poland_imgw``; a finished hydrological year never
changes, so only the latest two are re-fetched after a week.
"""

from __future__ import annotations

import csv
import io
import logging
import time
import zipfile
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import BaseModel

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.station import Station, in_bbox
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)
from aquascope.utils.http_client import DEFAULT_CACHE_DIR, CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

SITE_BASE = "https://danepubliczne.imgw.pl"
LIVE_URL = f"{SITE_BASE}/api/data/hydro/"
ARCHIVE_BASE = f"{SITE_BASE}/data/dane_pomiarowo_obserwacyjne/dane_hydrologiczne/dobowe"

#: First hydrological year in the archive (November 1950 to October 1951).
FIRST_HYDRO_YEAR = 1951
#: One zip per hydrological year from here on; one per hydrological month before.
YEARLY_FROM = 2023
#: Water temperature sentinel for "not measured" in the older files.
TEMP_MISSING = 99.9
#: Variables this collector serves, in the order the Explorer tries them.
VARIABLES_SERVED: tuple[str, ...] = ("discharge", "water_level", "water_quality")
#: The ten archive columns, in file order.
COLUMNS = (
    "station_id", "name", "river", "hydro_year", "hydro_month", "day",
    "level_cm", "discharge_cms", "temp_c", "calendar_month",
)
#: How long a cached zip for one of the two latest hydrological years is trusted.
RECENT_TTL_SECONDS = 7 * 86400


# ── hydrological calendar ────────────────────────────────────────────────


def hydro_year(d: date) -> int:
    """The hydrological year a calendar date belongs to (November starts the next one)."""
    return d.year + 1 if d.month >= 11 else d.year


def hydro_month(d: date) -> int:
    """1 for November through 12 for October."""
    return (d.month - 11) % 12 + 1


def calendar_year(hydro_yr: int, cal_month: int) -> int:
    """Undo the hydrological year: November and December belong to the year before."""
    return hydro_yr - 1 if cal_month >= 11 else hydro_yr


def archive_files(start: date, end: date) -> list[tuple[int, str]]:
    """``(hydro_year, filename)`` for every archive file that can hold a day in ``[start, end]``.

    Years before the archive begins are dropped; years not yet published are
    kept and fail softly at download time (a 404 is "not out yet").
    """
    out: list[tuple[int, str]] = []
    if end < start:
        return out
    for hy in range(max(FIRST_HYDRO_YEAR, hydro_year(start)), hydro_year(end) + 1):
        if hy >= YEARLY_FROM:
            out.append((hy, f"codz_{hy}.zip"))
            continue
        for hm in range(1, 13):
            cm = (hm + 9) % 12 + 1  # hydro month 1 -> November ... 12 -> October
            cy = calendar_year(hy, cm)
            first = date(cy, cm, 1)
            last = date(cy + (cm == 12), cm % 12 + 1, 1) - timedelta(days=1)
            if last < start or first > end:
                continue
            out.append((hy, f"codz_{hy}_{hm:02d}.zip"))
    return out


# ── archive parsing ──────────────────────────────────────────────────────


def decode_archive(raw: bytes) -> str:
    """UTF-8 when the bytes are valid UTF-8 (the 2023 file, with its BOM), else cp1250."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1250")
    return text.lstrip("\ufeff")


def _split(line: str, delimiter: str) -> list[str]:
    row = next(csv.reader([line], delimiter=delimiter))
    if len(row) == 1:
        # The whole line wrapped in one more layer of quotes (2024): the
        # single field is itself a CSV record, with doubled inner quotes undone.
        row = next(csv.reader([row[0]], delimiter=delimiter))
    return [c.strip() for c in row]


def parse_archive(text: str) -> pd.DataFrame:
    """One archive file to a frame of ``station_id, name, river, date, level_cm, discharge_cms, temp_c``.

    Dates are calendar dates rebuilt from the hydrological year and the
    calendar-month column. Missing values are NaN, including the 99.9
    temperature sentinel. Malformed lines are counted and skipped.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    empty = pd.DataFrame(columns=["station_id", "name", "river", "date", "level_cm", "discharge_cms", "temp_c"])
    if not lines:
        return empty
    delimiter = ";" if lines[0].count(";") > lines[0].count(",") else ","
    rows: list[list[str]] = []
    bad = 0
    for ln in lines:
        row = _split(ln, delimiter)
        if len(row) != len(COLUMNS):
            bad += 1
            continue
        rows.append(row)
    if bad:
        logger.warning("IMGW archive: skipped %d malformed line(s) of %d", bad, len(lines))
    if not rows:
        return empty
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    hy = pd.to_numeric(df["hydro_year"], errors="coerce")
    cm = pd.to_numeric(df["calendar_month"], errors="coerce")
    day = pd.to_numeric(df["day"], errors="coerce")
    cy = hy.where(cm < 11, hy - 1)
    df["date"] = pd.to_datetime(pd.DataFrame({"year": cy, "month": cm, "day": day}), errors="coerce")
    for col in ("level_cm", "discharge_cms", "temp_c"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    df.loc[df["temp_c"] >= TEMP_MISSING - 1e-3, "temp_c"] = float("nan")
    df = df.dropna(subset=["date"])
    out = df[["station_id", "name", "river", "date", "level_cm", "discharge_cms", "temp_c"]].reset_index(drop=True)
    out["station_id"] = out["station_id"].astype("category")
    return out


def _as_date(value: str | date | datetime | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _as_datetime(value: str | None) -> datetime | None:
    """``"2026-09-14 02:50:00"`` (Polish local time, kept naive) to datetime."""
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── collector ────────────────────────────────────────────────────────────


class PolandIMGWCollector(BaseCollector):
    """IMGW-PIB river gauges: live network state and the daily archive from 1951.

    Parameters
    ----------
    client : CachedHTTPClient, optional
        Used for the live API. Injected for testing.
    cache_dir : Path, optional
        Where archive zips are kept between runs (default ``data/cache/poland_imgw``).
    timeout : float
        Per-download timeout for archive zips, seconds.
    """

    name = "poland_imgw"

    #: Parsed archive frames for the whole process, keyed by (cache dir, file). The harvest builds one
    #: collector per station, and with a per-instance memo every station re-read and re-parsed all ~870
    #: zips from disk (about 3 minutes each; 3 stations in a 16-minute run on 2026-09-23). Recent
    #: hydrological years expire like their cached zips, so a long-lived process never serves a stale year.
    _shared_frames: dict[tuple[str, str], tuple[float, pd.DataFrame | None]] = {}

    def __init__(
        self,
        client: CachedHTTPClient | None = None,
        cache_dir: Path | None = None,
        timeout: float = 120.0,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                rate_limiter=RateLimiter(max_calls=30, period_seconds=60),
                cache_ttl_seconds=600,  # the live API refreshes every 10 minutes
            )
        )
        self._cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR / "poland_imgw"
        self._timeout = timeout
        self._live: list[dict[str, Any]] | None = None

    # ── live API ─────────────────────────────────────────────────────────

    def fetch_live(self) -> list[dict[str, Any]]:
        """The current state of every station, one request, cached for the run."""
        if self._live is None:
            payload = self.client.get_json(LIVE_URL)
            self._live = list(payload) if isinstance(payload, list) else []
        return self._live

    @staticmethod
    def _variables_of(raw: dict[str, Any]) -> tuple[str, ...]:
        out = []
        if _as_float(raw.get("przeplyw")) is not None:
            out.append("discharge")
        if _as_float(raw.get("stan_wody")) is not None:
            out.append("water_level")
        if _as_float(raw.get("temperatura_wody")) is not None:
            out.append("water_quality")
        return tuple(out)

    def stations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        variable: str | None = None,
        max_items: int | None = None,
    ) -> list[Station]:
        """The network from the live API: every station carries coordinates.

        ``variables`` reflects what the station is reporting *now*; the
        archive may hold more (a gauge whose discharge rating lapsed still
        has decades of discharge behind it).
        """
        if variable is not None and variable not in VARIABLES_SERVED:
            return []
        out: list[Station] = []
        for raw in self.fetch_live():
            lat, lon = _as_float(raw.get("lat")), _as_float(raw.get("lon"))
            if lat is None or lon is None or not in_bbox(lat, lon, bbox):
                continue
            variables = self._variables_of(raw)
            if not variables or (variable is not None and variable not in variables):
                continue
            stamps = [
                _as_datetime(raw.get(k))
                for k in ("stan_wody_data_pomiaru", "przeplyw_data", "temperatura_wody_data_pomiaru")
            ]
            latest = max((s for s in stamps if s is not None), default=None)
            founded = _as_float(raw.get("rok_zalozenia_stacji"))
            out.append(
                Station(
                    source=self.name,
                    station_id=str(raw["id_stacji"]).strip(),
                    name=(raw.get("stacja") or "").strip() or None,
                    latitude=lat,
                    longitude=lon,
                    variables=variables,
                    period_start=date(int(founded), 1, 1) if founded and 1800 < founded <= date.today().year else None,
                    period_end=latest.date() if latest else None,
                    river=(raw.get("rzeka") or "").strip() or None,
                    country="POL",
                    extra={
                        "voivodeship": raw.get("wojewodztwo"),
                        "gauge_datum_m_asl": _as_float(raw.get("rzedna_zerawodowskazu")),
                        "river_km": _as_float(raw.get("kilometr_biegu_rzeki")),
                        "alarm_stage_cm": _as_float(raw.get("stan_alarmowy")),
                        "warning_stage_cm": _as_float(raw.get("stan_ostrzegawczy")),
                        "ice_code": raw.get("zjawisko_lodowe"),
                        "overgrowth_code": raw.get("zjawisko_zarastania"),
                    },
                )
            )
            if max_items is not None and len(out) >= max_items:
                break
        return out

    # ── archive files ────────────────────────────────────────────────────

    def _download(self, url: str) -> bytes | None:
        """GET one zip. ``None`` on 404: a hydrological year not published yet."""
        try:
            resp = httpx.get(url, follow_redirects=True, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"IMGW archive download failed for {url}: {exc}") from exc
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content

    def _cached_is_fresh(self, path: Path, hy: int) -> bool:
        if hy < hydro_year(date.today()) - 1:
            return True  # a finished hydrological year does not change
        return (time.time() - path.stat().st_mtime) < RECENT_TTL_SECONDS

    def _archive_bytes(self, hy: int, filename: str) -> bytes | None:
        path = self._cache_dir / filename
        if path.exists() and self._cached_is_fresh(path, hy):
            return path.read_bytes()
        data = self._download(f"{ARCHIVE_BASE}/{hy}/{filename}")
        if data is None:
            logger.info("IMGW archive: %s not published (yet)", filename)
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    def _load_archive(self, hy: int, filename: str) -> pd.DataFrame | None:
        """The parsed frame for one archive file, memoised for the process (misses too)."""
        key = (str(self._cache_dir), filename)
        hit = PolandIMGWCollector._shared_frames.get(key)
        if hit is not None:
            loaded_at, cached = hit
            if hy < hydro_year(date.today()) - 1 or time.time() - loaded_at < RECENT_TTL_SECONDS:
                return cached
        raw = self._archive_bytes(hy, filename)
        frame: pd.DataFrame | None = None
        if raw is not None:
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                inner = next((n for n in z.namelist() if not n.endswith("/")), None)
                if inner is None:
                    raise ValueError(f"IMGW archive {filename} is an empty zip")
                frame = parse_archive(decode_archive(z.read(inner)))
            logger.debug("IMGW archive: %s parsed, %d rows", filename, len(frame))
        PolandIMGWCollector._shared_frames[key] = (time.time(), frame)
        return frame

    # ── observations ─────────────────────────────────────────────────────

    def _wanted_ids(
        self,
        station_ids: Iterable[str | int] | None,
        bbox: tuple[float, float, float, float] | None,
        max_stations: int | None,
    ) -> list[str]:
        if station_ids is not None:
            ids = [str(s).strip() for s in station_ids]
        else:
            if bbox is None:
                logger.warning("IMGW: no station_ids and no bbox, this walks the whole network")
            ids = [st.station_id for st in self.stations(bbox=bbox)]
        return ids[:max_stations] if max_stations is not None else ids

    def _latest_rows(self, ids: list[str], variable: str) -> list[dict[str, Any]]:
        """One row per station from the live API, each value with its own timestamp."""
        by_id = {str(r.get("id_stacji")).strip(): r for r in self.fetch_live()}
        value_key, stamp_key = {
            "discharge": ("przeplyw", "przeplyw_data"),
            "water_level": ("stan_wody", "stan_wody_data_pomiaru"),
            "water_quality": ("temperatura_wody", "temperatura_wody_data_pomiaru"),
        }[variable]
        rows: list[dict[str, Any]] = []
        for sid in ids:
            raw = by_id.get(sid)
            if raw is None:
                continue
            value, stamp = _as_float(raw.get(value_key)), _as_datetime(raw.get(stamp_key))
            if value is None or stamp is None:
                continue
            rows.append({
                "station_id": sid,
                "station_name": (raw.get("stacja") or "").strip() or None,
                "river": (raw.get("rzeka") or "").strip() or None,
                "latitude": _as_float(raw.get("lat")),
                "longitude": _as_float(raw.get("lon")),
                "datetime": stamp,
                "variable": variable,
                "value": value / 100.0 if variable == "water_level" else value,
            })
        return rows

    def fetch_raw(
        self,
        variable: str = "discharge",
        station_ids: Iterable[str | int] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        max_stations: int | None = None,
        latest_only: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Daily rows for one variable from the archive, or the live reading with ``latest_only``.

        ``start``/``end`` select archive files by hydrological year and month,
        then rows by calendar date; the default window is the last 365 days.
        Files for years not yet published are skipped, so a window reaching
        into the current hydrological year simply stops at the last October.
        """
        if variable not in VARIABLES_SERVED:
            raise ValueError(f"Unknown variable {variable!r}; expected one of {VARIABLES_SERVED}")
        ids = self._wanted_ids(station_ids, bbox, max_stations)
        if not ids:
            return []
        if latest_only:
            return self._latest_rows(ids, variable)

        end_d = _as_date(end) or date.today()
        start_d = _as_date(start) or end_d - timedelta(days=365)
        column = {"discharge": "discharge_cms", "water_level": "level_cm", "water_quality": "temp_c"}[variable]
        wanted = set(ids)
        rows: list[dict[str, Any]] = []
        for hy, filename in archive_files(start_d, end_d):
            frame = self._load_archive(hy, filename)
            if frame is None or frame.empty:
                continue
            sub = frame[frame["station_id"].isin(wanted)]
            sub = sub[(sub["date"] >= pd.Timestamp(start_d)) & (sub["date"] <= pd.Timestamp(end_d))]
            sub = sub.dropna(subset=[column])
            for rec in sub.itertuples(index=False):
                # float32 in the frame keeps a full-record walk in memory; the
                # archive prints three decimals, so that is what the record carries.
                value = round(float(getattr(rec, column)), 3)
                rows.append({
                    "station_id": str(rec.station_id),
                    "station_name": rec.name or None,
                    "river": rec.river or None,
                    "datetime": rec.date.to_pydatetime(),
                    "variable": variable,
                    "value": value / 100.0 if variable == "water_level" else value,
                })
        return rows

    def normalise(self, raw: list[dict[str, Any]]) -> Sequence[BaseModel]:
        """Discharge to :class:`StreamflowReading` (m3/s), stage to :class:`WaterLevelReading`
        (metres above the gauge datum), water temperature to :class:`WaterQualitySample` (°C)."""
        records: list[BaseModel] = []
        for row in raw:
            lat, lon = row.get("latitude"), row.get("longitude")
            location = GeoLocation(latitude=lat, longitude=lon) if lat is not None and lon is not None else None
            common = {
                "source": DataSource.POLAND_IMGW,
                "station_id": row["station_id"],
                "station_name": row.get("station_name"),
                "location": location,
            }
            variable, value, stamp = row["variable"], float(row["value"]), row["datetime"]
            if variable == "discharge":
                records.append(StreamflowReading(**common, reading_datetime=stamp, discharge_cms=value, source_type="in_situ"))
            elif variable == "water_level":
                records.append(WaterLevelReading(**common, reading_datetime=stamp, water_level=value, unit="m",
                                                 remark="above the gauge datum (stan wody)"))
            elif variable == "water_quality":
                records.append(WaterQualitySample(**common, sample_datetime=stamp, parameter="water_temperature",
                                                  value=value, unit="degC", river=row.get("river")))
        return records
