"""
Collector for Greece — Hydroscope, the National Databank for Hydrological
and Meteorological Information (Εθνικό Υδρολογικό και Μετεωρολογικό Δίκτυο).

Hydroscope federates the monitoring networks of four Greek agencies. Each one
runs its own `Enhydris <https://github.com/openmeteo/enhydris>`_ instance with
an open, unauthenticated REST API, and ``main.hydroscope.gr`` is the national
catalog that unions them under composite station ids::

    composite_id = database_prefix * 100_000_000 + native_id

Three of the four source instances serve observations and are used here:

===  ============================  =====================================
db   host                          agency
===  ============================  =====================================
1    ``kyy.hydroscope.gr``         Ministry of Environment and Energy
2    ``emy.hydroscope.gr``         Hellenic National Meteorological Service
3    ``ypaat.hydroscope.gr``       Ministry of Rural Development and Food
===  ============================  =====================================

The fourth (``db=4``, ΔΕΗ / Public Power Corporation, 639 stations) is listed
in the national catalog but its own instance does not resolve, and the catalog
host returns empty bodies for its series. Those stations are therefore *not*
emitted: a catalog entry that cannot be dereferenced is worse than an absent
one. Revisit if the PPC instance comes back online.

Two upstream endpoints per instance:

- ``/api/Station/`` — the station list (id, name, WKT point, basin, dates).
- ``/api/Timeseries/`` — one record per series, carrying ``gentity`` (the
  station), ``variable``, ``unit_of_measurement`` and the real
  ``start_date_utc`` / ``end_date_utc`` span. The national catalog leaves those
  span fields null, which is why the per-instance catalogs are used instead.

Series data comes from ``/timeseries/d/<id>/download/`` in openmeteo's ``.hts``
text format: ``key=value`` headers, a blank line, then ``timestamp,value,flags``
rows. A date window can be pushed to the server as a path suffix
(``/download/<start>/<end>/``), which matters because a single 60-year daily
stage file is ~0.6 MB.

Two traps this collector works around, both confirmed against live responses:

* **The catalog's unit is not the file's unit.** Series carrying variable 101
  (ΥΔΡΟΜΕΤΡΗΣΗ) advertise ``m³/s`` in ``/api/UnitOfMeasurement/`` while the
  downloaded file header says ``Unit=l/s``. The file header wins, and discharge
  is converted to m³/s from whichever unit it declares.
* **A catalogued series is often empty.** Of 67 discharge series on the
  Ministry of Environment instance, only 2 hold data (``Count=0`` for the
  rest). ``start_date_utc`` being null is the cheap catalog-side signal, and
  such series are skipped without a request.

Coverage is historical rather than real-time: daily stage runs 1949-2013,
discharge is monthly and largely stops in the 2000s, and rainfall is the
deepest layer. No API key, no rate-limit documented; the client is throttled
conservatively anyway.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.station import Station, in_bbox
from aquascope.schemas.water_data import (
    ClimateReading,
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

MAIN_BASE = "http://main.hydroscope.gr"

#: Source instances that actually serve observations, keyed by the database
#: prefix used in the national catalog's composite ids.
INSTANCES: dict[int, str] = {
    1: "http://kyy.hydroscope.gr",
    2: "http://emy.hydroscope.gr",
    3: "http://ypaat.hydroscope.gr",
}

AGENCIES: dict[int, str] = {
    1: "Ministry of Environment and Energy (ΥΠΕΝ)",
    2: "Hellenic National Meteorological Service (ΕΜΥ)",
    3: "Ministry of Rural Development and Food (ΥΠΑΑΤ)",
    4: "Public Power Corporation (ΔΕΗ)",
}

ID_SCALE = 100_000_000

#: Enhydris variable ids, shared across the three instances. Greek labels are
#: kept in comments because that is how they appear in ``/api/Variable/``.
VARIABLE_IDS: dict[str, frozenset[int]] = {
    "discharge": frozenset({85, 101}),  # ΠΑΡΟΧΗ, ΥΔΡΟΜΕΤΡΗΣΗ
    # 89 (ΣΗΜΕΙΑΚΗ ΣΤΑΘΜΗ) is deliberately absent: no instance carries a series
    # for it, and "point stage" is ambiguous enough that mapping it to surface
    # water level unverified would be a guess.
    "water_level": frozenset({88, 103}),  # ΣΤΑΘΜΗ, ΣΤΑΘΜΗ (ΠΛΗΜΜΥΡΑ)
    "precipitation": frozenset({8}),  # ΒΡΟΧΟΠΤΩΣΗ
}

#: Preference order inside one variable, most-preferred first. A station can
#: hold several series for the same quantity: station 200082 has two ΣΤΑΘΜΗ
#: series covering 1950-1983 and 1950-1982 plus a ΣΤΑΘΜΗ (ΠΛΗΜΜΥΡΑ) one. Those
#: are neither interchangeable nor safe to concatenate, so exactly one wins.
#: Ordinary stage beats flood stage, and ``ΠΑΡΟΧΗ`` beats ``ΥΔΡΟΜΕΤΡΗΣΗ``
#: (individual gaugings) for discharge.
VARIABLE_PREFERENCE: dict[str, tuple[int, ...]] = {
    "discharge": (85, 101),
    "water_level": (88, 103),
    "precipitation": (8,),
}

#: Multiplier onto m3/s, keyed by the unit the ``.hts`` header declares.
DISCHARGE_UNIT_FACTORS: dict[str, float] = {"m3/s": 1.0, "m³/s": 1.0, "l/s": 0.001, "lt/s": 0.001}

LEVEL_UNIT_FACTORS: dict[str, float] = {"m": 1.0, "cm": 0.01, "mm": 0.001}

PRECIPITATION_UNITS = frozenset({"mm"})

_POINT_RE = re.compile(r"POINT\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)")
_HTS_ROW_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})\s*,([^,]*),")


class GreeceHydroscopeCollector(BaseCollector):
    """
    Collect river stage, discharge and rainfall from Greece's Hydroscope network.

    Parameters
    ----------
    client : CachedHTTPClient, optional
        Injected for testing; a default client is created otherwise.
    """

    name = "greece_hydroscope"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                rate_limiter=RateLimiter(max_calls=20, period_seconds=60),
                # The archive is closed (nothing newer than 2013 on most
                # instances), so cache hard: a day is still conservative.
                cache_ttl_seconds=86_400,
            )
        )

    # ── upstream catalogs ────────────────────────────────────────────────

    def fetch_stations(self, db: int) -> list[dict[str, Any]]:
        """Fetch one instance's raw station list."""
        payload: list[dict[str, Any]] = self.client.get_json(f"{INSTANCES[db]}/api/Station/")
        return payload

    def fetch_timeseries(self, db: int) -> list[dict[str, Any]]:
        """Fetch one instance's raw series catalog."""
        payload: list[dict[str, Any]] = self.client.get_json(f"{INSTANCES[db]}/api/Timeseries/")
        return payload

    def _catalog(self, db: int) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
        """Return ``({station_id: station}, {station_id: [series, …]})`` for one instance.

        Only series with a non-null ``start_date_utc`` are kept: the rest are
        catalogued shells whose download returns ``Count=0``.
        """
        stations = {s["id"]: s for s in self.fetch_stations(db)}
        by_station: dict[int, list[dict[str, Any]]] = {}
        for series in self.fetch_timeseries(db):
            if not series.get("start_date_utc"):
                continue
            if self._variable_of(series) is None:
                continue
            station_id = series.get("gentity")
            if station_id is None:
                continue
            by_station.setdefault(int(station_id), []).append(series)
        return stations, by_station

    @staticmethod
    def _variable_of(series: dict[str, Any]) -> str | None:
        """Map an Enhydris series onto an AquaScope variable name, or ``None``."""
        raw = series.get("variable")
        if raw is None:
            return None
        native = int(raw) % ID_SCALE
        for variable, ids in VARIABLE_IDS.items():
            if native in ids:
                return variable
        return None

    # ── station catalog ──────────────────────────────────────────────────

    def stations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        variable: str | None = None,
        max_items: int | None = None,
    ) -> list[Station]:
        """Hydroscope station catalog, restricted to stations that hold data."""
        if variable is not None and variable not in VARIABLE_IDS:
            return []

        out: list[Station] = []
        for db in INSTANCES:
            try:
                raw_stations, by_station = self._catalog(db)
            except Exception as exc:  # noqa: BLE001 - one dead instance shouldn't empty the catalog
                logger.warning("Hydroscope: instance db=%d catalog unavailable: %s", db, exc)
                continue

            for station_id, series_list in by_station.items():
                raw = raw_stations.get(station_id)
                if raw is None:
                    continue
                coords = _parse_point(raw.get("point"))
                if coords is None:
                    continue
                lat, lon = coords
                if not in_bbox(lat, lon, bbox):
                    continue

                variables = tuple(sorted({v for v in (self._variable_of(s) for s in series_list) if v}))
                if variable is not None and variable not in variables:
                    continue

                starts = [s["start_date_utc"][:10] for s in series_list if s.get("start_date_utc")]
                ends = [s["end_date_utc"][:10] for s in series_list if s.get("end_date_utc")]
                composite = _composite_id(db, station_id)
                out.append(
                    Station(
                        source=self.name,
                        station_id=composite,
                        name=(raw.get("name") or "").strip() or None,
                        latitude=lat,
                        longitude=lon,
                        variables=variables,
                        period_start=_as_date(min(starts)) if starts else None,
                        period_end=_as_date(max(ends)) if ends else None,
                        url=f"{MAIN_BASE}/stations/d/{composite}/",
                        country="GRC",
                        extra={
                            "agency": AGENCIES[db],
                            "instance": INSTANCES[db],
                            "native_id": station_id,
                            "n_series": len(series_list),
                        },
                    )
                )
                if max_items is not None and len(out) >= max_items:
                    return out
        return out

    # ── observations ─────────────────────────────────────────────────────

    def fetch_raw(
        self,
        variable: str = "discharge",
        station_ids: Iterable[str] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        max_stations: int | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Fetch observation rows for one variable.

        Parameters
        ----------
        variable : str
            One of ``"discharge"``, ``"water_level"`` or ``"precipitation"``.
        station_ids : iterable of str, optional
            Composite station ids (as emitted by :meth:`stations`). All
            stations holding the variable are used when omitted.
        bbox : tuple, optional
            ``(west, south, east, north)`` in WGS84 degrees.
        start, end : str or date, optional
            Inclusive window, pushed to the server as a download path suffix.
            Both must be given for the server-side filter to apply; a single
            bound is applied client-side instead.
        max_stations : int, optional
            Cap on the number of stations fetched. Left uncapped by default,
            but worth setting: the full daily stage archive is ~50 MB.
        """
        if variable not in VARIABLE_IDS:
            raise ValueError(f"Unknown variable {variable!r}; expected one of {sorted(VARIABLE_IDS)}")

        wanted = {str(s) for s in station_ids} if station_ids is not None else None
        window = _date_suffix(start, end)
        lo, hi = _as_date(start), _as_date(end)

        rows: list[dict[str, Any]] = []
        n_stations = 0
        for db in INSTANCES:
            if max_stations is not None and n_stations >= max_stations:
                break
            try:
                raw_stations, by_station = self._catalog(db)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Hydroscope: instance db=%d catalog unavailable: %s", db, exc)
                continue

            for station_id, series_list in by_station.items():
                if max_stations is not None and n_stations >= max_stations:
                    break
                composite = _composite_id(db, station_id)
                if wanted is not None and composite not in wanted:
                    continue

                raw = raw_stations.get(station_id) or {}
                coords = _parse_point(raw.get("point"))
                if bbox is not None and (coords is None or not in_bbox(coords[0], coords[1], bbox)):
                    continue

                matching = _pick_series([s for s in series_list if self._variable_of(s) == variable], variable)
                if not matching:
                    continue

                fetched_any = False
                for series in matching:
                    text = self._download(db, series["id"], window)
                    if text is None:
                        continue
                    header, values = _parse_hts(text)
                    unit = header.get("Unit", "")
                    for stamp, value in values:
                        if lo is not None and stamp.date() < lo:
                            continue
                        if hi is not None and stamp.date() > hi:
                            continue
                        rows.append(
                            {
                                "station_id": composite,
                                "station_name": (raw.get("name") or "").strip() or None,
                                "latitude": coords[0] if coords else None,
                                "longitude": coords[1] if coords else None,
                                "datetime": stamp,
                                "value": value,
                                "unit": unit,
                                "variable": variable,
                                "series_id": series["id"],
                                "agency": AGENCIES[db],
                            }
                        )
                    fetched_any = True
                if fetched_any:
                    n_stations += 1

        return rows

    def _download(self, db: int, series_id: int, window: str) -> str | None:
        """GET one series' ``.hts`` file, returning ``None`` if it is unavailable."""
        url = f"{INSTANCES[db]}/timeseries/d/{series_id}/download/{window}"
        try:
            text: str = self.client.get_text(url)
            return text
        except Exception as exc:  # noqa: BLE001 - one bad series shouldn't abort the run
            logger.debug("Hydroscope: failed to fetch %s: %s", url, exc)
            return None

    # ── normalisation ────────────────────────────────────────────────────

    def normalise(self, raw: list[dict[str, Any]]) -> Sequence[BaseModel]:
        """Convert rows into the model matching each row's variable.

        Discharge becomes :class:`StreamflowReading` in m3/s, water level
        becomes :class:`WaterLevelReading` in m, and rainfall becomes
        :class:`ClimateReading`. Rows whose declared unit is not convertible
        are dropped rather than assumed.
        """
        records: list[BaseModel] = []
        skipped = 0
        for row in raw:
            try:
                record = self._to_record(row)
            except (ValueError, KeyError, TypeError) as exc:
                record = None
                logger.debug("Skipping Hydroscope row: %s", exc)
            if record is None:
                skipped += 1
                continue
            records.append(record)

        if skipped:
            logger.warning(
                "Hydroscope normalise(): skipped %d/%d row(s) (unconvertible unit or invalid fields)",
                skipped,
                len(raw),
            )
        return records

    def _to_record(self, row: dict[str, Any]) -> BaseModel | None:
        lat, lon = row.get("latitude"), row.get("longitude")
        location = GeoLocation(latitude=lat, longitude=lon) if lat is not None and lon is not None else None
        unit = (row.get("unit") or "").strip()
        value = float(row["value"])
        common = {
            "source": DataSource.GREECE_HYDROSCOPE,
            "station_id": row["station_id"],
            "station_name": row.get("station_name"),
            "location": location,
        }

        variable = row["variable"]
        if variable == "discharge":
            factor = DISCHARGE_UNIT_FACTORS.get(unit)
            if factor is None:
                return None
            return StreamflowReading(
                **common,
                reading_datetime=row["datetime"],
                discharge_cms=value * factor,
                source_type="in_situ",
                remark=row.get("agency"),
            )
        if variable == "water_level":
            factor = LEVEL_UNIT_FACTORS.get(unit)
            if factor is None:
                return None
            return WaterLevelReading(
                **common,
                reading_datetime=row["datetime"],
                water_level=value * factor,
                remark=row.get("agency"),
            )
        if variable == "precipitation":
            if unit not in PRECIPITATION_UNITS:
                return None
            return ClimateReading(
                **common,
                sample_datetime=row["datetime"],
                parameter="rainfall_mm",
                value=value,
                unit=unit,
                remark=row.get("agency"),
            )
        return None


# ── module helpers ───────────────────────────────────────────────────────


def _pick_series(candidates: list[dict[str, Any]], variable: str) -> list[dict[str, Any]]:
    """Choose the one series to represent ``variable`` at a station.

    Concatenating every matching series is wrong twice over: it interleaves
    quantities that only look alike (ordinary stage against flood stage) and it
    emits the same timestamp more than once where two series overlap, which at
    station 200082 meant 515,650 rows carrying 11,219 duplicate timestamps.

    The winner is the most-preferred variable id present, and within that the
    longest record. The rest are logged rather than silently discarded.
    """
    if len(candidates) <= 1:
        return candidates

    preference = VARIABLE_PREFERENCE.get(variable, ())

    def rank(series: dict[str, Any]) -> tuple[int, float]:
        native = int(series["variable"]) % ID_SCALE
        position = preference.index(native) if native in preference else len(preference)
        start, end = _as_date(series.get("start_date_utc")), _as_date(series.get("end_date_utc"))
        span = (end - start).days if start and end else 0
        return (position, -span)  # most-preferred id first, then longest record

    ordered = sorted(candidates, key=rank)
    winner = ordered[0]
    logger.debug(
        "Hydroscope: station has %d %s series; keeping %s and skipping %s",
        len(candidates),
        variable,
        winner["id"],
        [s["id"] for s in ordered[1:]],
    )
    return [winner]


def _composite_id(db: int, native_id: int) -> str:
    """Build the national-catalog id so local ids match main.hydroscope.gr."""
    return str(db * ID_SCALE + int(native_id))


def _parse_point(wkt: str | None) -> tuple[float, float] | None:
    """Extract ``(lat, lon)`` from an EWKT point like ``SRID=4326;POINT (22.4 39.1)``.

    Only SRID 4326 is accepted; Enhydris also stores Greek Grid (SRID 2100)
    geometries, and silently reading those as degrees would put stations in
    the Atlantic.
    """
    if not wkt:
        return None
    if "SRID=" in wkt and "SRID=4326" not in wkt:
        return None
    match = _POINT_RE.search(wkt)
    if match is None:
        return None
    lon, lat = float(match.group(1)), float(match.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _parse_hts(text: str) -> tuple[dict[str, str], list[tuple[datetime, float]]]:
    """Split an openmeteo ``.hts`` payload into headers and ``(timestamp, value)`` pairs.

    ``Comment`` appears more than once per file; the last one wins, which is
    fine because nothing here reads it. Rows with an empty value column are
    gaps in the record and are dropped.
    """
    headers: dict[str, str] = {}
    values: list[tuple[datetime, float]] = []
    in_body = False

    for line in text.splitlines():
        if not in_body:
            stripped = line.strip()
            if not stripped:
                in_body = True
                continue
            key, _, value = stripped.partition("=")
            if value or "=" in stripped:
                headers[key.strip()] = value.strip()
            continue

        match = _HTS_ROW_RE.match(line.strip())
        if match is None:
            continue
        raw_value = match.group(2).strip()
        if not raw_value:
            continue
        try:
            stamp = datetime.strptime(match.group(1).replace("T", " "), "%Y-%m-%d %H:%M")
            values.append((stamp, float(raw_value)))
        except ValueError:
            continue

    return headers, values


def _as_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _date_suffix(start: str | date | None, end: str | date | None) -> str:
    """Build the server-side window suffix, or ``""`` when the range is open.

    Enhydris only honours the two-sided path form
    ``/download/<start>/<end>/``; a half-open range is filtered client-side.
    """
    lo, hi = _as_date(start), _as_date(end)
    if lo is None or hi is None:
        return ""
    return f"{lo.isoformat()}/{hi.isoformat()}/"
