"""
Collector for Greece — OpenHi.net, the Open Hydrosystem Information Network
run by ITIA at the National Technical University of Athens.

This is the live half of Greece's open hydrometry. The archival half, the
national Hydroscope databank, is `greece_hydroscope`; the two share the
Enhydris lineage and nothing else. OpenHi runs a modern Enhydris 3 REST API,
open and unregistered, with 15-minute telemetry that reaches the current day.

**The API host is ``system.openhi.net``, not ``openhi.net``.** The public site
is a separate WordPress host that answers 404 to every API path, which is an
easy way to conclude the API does not exist.

Reaching a value is a four-step walk, every step wrapped in the same
paginated envelope (``count`` / ``next`` / ``previous`` / ``results``)::

    /api/stations/
    /api/stations/<sid>/timeseriesgroups/
    /api/stations/<sid>/timeseriesgroups/<gid>/timeseries/
    /api/stations/<sid>/timeseriesgroups/<gid>/timeseries/<tid>/data/

The last one returns headerless CSV (``timestamp,value,flags``) and accepts
``start_date`` / ``end_date``, which matters: one Mandra discharge series is
12.9 MB whole and 27 KB for a five-day window. ``/bottom/`` returns just the
latest row, so liveness is one cheap request rather than a full download.

Variable and unit ids resolve at ``/api/variables/<id>/`` (English name under
``translations.en.descr``) and ``/api/units/<id>/``. The ids in
:data:`VARIABLE_MAP` were read off the live vocabulary rather than assumed:
there is no groundwater variable in this network, stage is 14 and water level
is 5710, and no variable 5689 or 28 exists.

Four response shapes have to stay distinguishable, which is the whole point
of issue #165: a **populated** series (200, CSV rows with values), an
**empty** one (200 with a zero-byte body, how Enhydris says "catalogued but
never populated"), one carrying **no values** (200, timestamps present but the
value column blank, which is what station 1534's water-quality sensors return
today), and a **restricted** one (401, e.g. station 28082's discharge).
Collapsing any of those into "failed" is how a collector goes quiet without
anyone noticing, so :meth:`GreeceOpenhiCollector.fetch_raw` counts them
separately and logs the tally.

Coverage as observed: 65 stations, 37 with a ``last_update`` in the current
year. Discharge is genuinely live at the three Mandra stations (15-minute,
through today) and hourly at Anthili; the Karveliotis discharge archive stops
in 2014. Stage is the widest variable at 58 groups, and the network also
carries rainfall, air and soil measurements and a water-quality set.

Licensed CC BY-SA 4.0 (https://openhi.net/licence-el/), so observations are
redistributable with attribution and share-alike.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any, NamedTuple

from pydantic import BaseModel

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.station import Station, in_bbox
from aquascope.schemas.water_data import (
    ClimateReading,
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

API_BASE = "https://system.openhi.net/api"
SITE_BASE = "https://openhi.net"


class _Var(NamedTuple):
    """How one Enhydris variable maps onto AquaScope's vocabulary."""

    variable: str  # a member of aquascope.schemas.station.VARIABLES
    parameter: str  # the label used by ClimateReading / WaterQualitySample


#: Enhydris variable id -> mapping, read off ``/api/variables/`` on the live
#: instance. Ids absent here (battery voltage, signal quality, raw distance,
#: "Other") are instrument telemetry, not observations, and are skipped.
VARIABLE_MAP: dict[int, _Var] = {
    2: _Var("discharge", "discharge_cms"),
    14: _Var("water_level", "stage_m"),
    5710: _Var("water_level", "water_level_m"),
    1: _Var("precipitation", "rainfall_mm"),
    7: _Var("evapotranspiration", "evaporation_mm"),
    5683: _Var("climate", "temperature_air_c"),
    4: _Var("climate", "humidity_relative_pct"),
    6: _Var("climate", "wind_speed_ms"),
    20: _Var("climate", "wind_direction_deg"),
    22: _Var("climate", "pressure_barometric_hpa"),
    23: _Var("climate", "solar_radiation"),
    5694: _Var("climate", "soil_moisture"),
    5701: _Var("climate", "temperature_soil_c"),
    5687: _Var("water_quality", "electrical_conductivity"),
    5688: _Var("water_quality", "dissolved_oxygen"),
    5690: _Var("water_quality", "orp"),
    5691: _Var("water_quality", "pH"),
    5693: _Var("water_quality", "salinity"),
    5700: _Var("water_quality", "temperature_water_c"),
    5695: _Var("water_quality", "water_velocity"),
    5698: _Var("water_quality", "water_speed"),
    5696: _Var("water_quality", "pressure_hydrostatic"),
}

#: Multiplier onto m3/s and onto metres, by the unit symbol the API reports.
DISCHARGE_UNIT_FACTORS: dict[str, float] = {"m³/s": 1.0, "m3/s": 1.0, "l/s": 0.001}
LEVEL_UNIT_FACTORS: dict[str, float] = {"m": 1.0, "cm": 0.01, "mm": 0.001}

#: Enhydris series ``type``, most-preferred first. "Initial" is the raw record;
#: an Aggregated series is a derived roll-up and is only used when the raw one
#: turns out to be empty (station 1458 has exactly that shape).
SERIES_TYPE_PREFERENCE = ("Initial", "Aggregated")

_POINT_RE = re.compile(r"POINT\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)")


class _Fetch(NamedTuple):
    """One series download, with *why* it produced no rows kept separate.

    ``outcome`` is one of ``ok``, ``empty`` (200 with a zero-byte body:
    catalogued but never populated), ``no_values`` (200 with timestamps but a
    blank value column), ``restricted`` (401/403) or ``failed`` (transport or
    unexpected status). See the module docstring.
    """

    rows: list[tuple[datetime, float]]
    outcome: str


class GreeceOpenhiCollector(BaseCollector):
    """
    Collect live discharge, stage, rainfall, climate and water quality from
    Greece's OpenHi.net network.

    Parameters
    ----------
    client : CachedHTTPClient, optional
        Injected for testing; a default client is created otherwise.
    """

    name = "greece_openhi"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                rate_limiter=RateLimiter(max_calls=30, period_seconds=60),
                cache_ttl_seconds=900,  # telemetry lands every 15 minutes
            )
        )
        self._units: dict[int, str] | None = None

    # ── paginated walk ───────────────────────────────────────────────────

    def _paged(self, url: str) -> list[dict[str, Any]]:
        """Follow ``next`` to the end and return every result row.

        Every list endpoint on this API is paginated, including the nested
        ones, so reading only the first page silently truncates: the station
        list alone is 65 rows over two pages of 20.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        while url and url not in seen:
            seen.add(url)
            payload = self.client.get_json(url)
            if isinstance(payload, list):  # an unpaginated response, defensively
                return out + payload
            out.extend(payload.get("results") or [])
            url = payload.get("next") or ""
        return out

    def fetch_stations(self) -> list[dict[str, Any]]:
        """Fetch every station record."""
        return self._paged(f"{API_BASE}/stations/?format=json")

    def fetch_groups(self, station_id: int | str) -> list[dict[str, Any]]:
        """Fetch one station's time-series groups (each carries a ``variable``)."""
        return self._paged(f"{API_BASE}/stations/{station_id}/timeseriesgroups/?format=json")

    def fetch_series(self, station_id: int | str, group_id: int | str) -> list[dict[str, Any]]:
        """Fetch the series inside one group."""
        return self._paged(f"{API_BASE}/stations/{station_id}/timeseriesgroups/{group_id}/timeseries/?format=json")

    def unit_symbol(self, unit_id: int | None) -> str:
        """Resolve a unit id to its symbol, with the lookup table cached."""
        if unit_id is None:
            return ""
        if self._units is None:
            self._units = {}
        if unit_id not in self._units:
            try:
                payload = self.client.get_json(f"{API_BASE}/units/{unit_id}/?format=json")
                self._units[unit_id] = str(payload.get("symbol") or "")
            except Exception as exc:  # noqa: BLE001 - an unresolvable unit is not fatal
                logger.debug("OpenHi: unit %s did not resolve: %s", unit_id, exc)
                self._units[unit_id] = ""
        return self._units[unit_id]

    # ── station catalog ──────────────────────────────────────────────────

    def stations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        variable: str | None = None,
        max_items: int | None = None,
    ) -> list[Station]:
        """OpenHi station catalog, with each station's variables resolved.

        Resolving ``variables`` costs one request per station, since what a
        station measures is only visible in its groups. The client's cache
        absorbs that on repeat calls.
        """
        known = {v.variable for v in VARIABLE_MAP.values()}
        if variable is not None and variable not in known:
            return []

        out: list[Station] = []
        for raw in self.fetch_stations():
            coords = _parse_point(raw.get("geom"))
            if coords is None:
                continue
            lat, lon = coords
            if not in_bbox(lat, lon, bbox):
                continue

            try:
                groups = self.fetch_groups(raw["id"])
            except Exception as exc:  # noqa: BLE001 - one bad station shouldn't empty the catalog
                logger.debug("OpenHi: groups for station %s unavailable: %s", raw.get("id"), exc)
                continue

            variables = tuple(
                sorted({VARIABLE_MAP[g["variable"]].variable for g in groups if g.get("variable") in VARIABLE_MAP})
            )
            if not variables:
                continue
            if variable is not None and variable not in variables:
                continue

            out.append(
                Station(
                    source=self.name,
                    station_id=str(raw["id"]),
                    name=(raw.get("name") or "").strip() or None,
                    latitude=lat,
                    longitude=lon,
                    variables=variables,
                    period_start=_as_date(raw.get("start_date")),
                    period_end=_as_date(raw.get("last_update") or raw.get("end_date")),
                    url=f"{SITE_BASE}/stations/{raw['id']}/",
                    country="GRC",
                    extra={
                        "altitude_m": raw.get("altitude"),
                        "code": (raw.get("code") or "").strip() or None,
                        "display_timezone": raw.get("display_timezone"),
                        "last_update": raw.get("last_update"),
                        "n_groups": len(groups),
                    },
                )
            )
            if max_items is not None and len(out) >= max_items:
                break
        return out

    # ── observations ─────────────────────────────────────────────────────

    def fetch_raw(
        self,
        variable: str = "discharge",
        station_ids: Iterable[str | int] | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
        max_stations: int | None = None,
        latest_only: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Fetch observation rows for one variable.

        Parameters
        ----------
        variable : str
            An AquaScope variable name present in :data:`VARIABLE_MAP`, e.g.
            ``"discharge"``, ``"water_level"``, ``"precipitation"``,
            ``"climate"``, ``"water_quality"``.
        station_ids : iterable, optional
            OpenHi station ids. All stations carrying the variable otherwise.
        bbox : tuple, optional
            ``(west, south, east, north)`` in WGS84 degrees.
        start, end : str or date, optional
            Pushed to the API as ``start_date`` / ``end_date``. Strongly
            recommended: the largest series here is 12.9 MB unwindowed.
        max_stations : int, optional
            Cap on stations fetched.
        latest_only : bool, default False
            Use ``/bottom/`` to take just each series' most recent row. Cheap
            enough to sweep the whole network for a liveness check.
        """
        known = {v.variable for v in VARIABLE_MAP.values()}
        if variable not in known:
            raise ValueError(f"Unknown variable {variable!r}; expected one of {sorted(known)}")

        wanted = {str(s) for s in station_ids} if station_ids is not None else None
        params = _window_params(start, end)
        lo, hi = _as_date(start), _as_date(end)

        rows: list[dict[str, Any]] = []
        tally = {"ok": 0, "empty": 0, "no_values": 0, "restricted": 0, "failed": 0}
        n_stations = 0

        for raw in self.fetch_stations():
            if max_stations is not None and n_stations >= max_stations:
                break
            station_id = str(raw["id"])
            if wanted is not None and station_id not in wanted:
                continue
            coords = _parse_point(raw.get("geom"))
            if bbox is not None and (coords is None or not in_bbox(coords[0], coords[1], bbox)):
                continue

            try:
                groups = self.fetch_groups(station_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("OpenHi: groups for station %s unavailable: %s", station_id, exc)
                tally["failed"] += 1
                continue

            matching = [
                g
                for g in groups
                if g.get("variable") in VARIABLE_MAP and VARIABLE_MAP[g["variable"]].variable == variable
            ]
            if not matching:
                continue

            fetched_any = False
            for group in matching:
                mapping = VARIABLE_MAP[group["variable"]]
                unit = self.unit_symbol(group.get("unit_of_measurement"))
                for stamp, value in self._group_rows(station_id, group, params, latest_only, tally):
                    if lo is not None and stamp.date() < lo:
                        continue
                    if hi is not None and stamp.date() > hi:
                        continue
                    rows.append(
                        {
                            "station_id": station_id,
                            "station_name": (raw.get("name") or "").strip() or None,
                            "latitude": coords[0] if coords else None,
                            "longitude": coords[1] if coords else None,
                            "altitude_m": raw.get("altitude"),
                            "datetime": stamp,
                            "value": value,
                            "unit": unit,
                            "variable": mapping.variable,
                            "parameter": mapping.parameter,
                            "group_id": group["id"],
                        }
                    )
                    fetched_any = True
            if fetched_any:
                n_stations += 1

        if any(tally[k] for k in ("empty", "no_values", "restricted", "failed")):
            logger.info(
                "OpenHi %s: %d series with data, %d catalogued but empty, %d returning only null values, "
                "%d restricted (401/403), %d failed",
                variable,
                tally["ok"],
                tally["empty"],
                tally["no_values"],
                tally["restricted"],
                tally["failed"],
            )
        return rows

    def _group_rows(
        self,
        station_id: str,
        group: dict[str, Any],
        params: dict[str, str],
        latest_only: bool,
        tally: dict[str, int],
    ) -> list[tuple[datetime, float]]:
        """Return the best available rows for one group.

        Series are tried in :data:`SERIES_TYPE_PREFERENCE` order and the first
        populated one wins. An empty raw series therefore falls through to the
        aggregated roll-up rather than reporting the group as dataless.
        """
        try:
            series = self.fetch_series(station_id, group["id"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenHi: series for group %s unavailable: %s", group.get("id"), exc)
            tally["failed"] += 1
            return []

        def rank(s: dict[str, Any]) -> int:
            kind = s.get("type") or ""
            return SERIES_TYPE_PREFERENCE.index(kind) if kind in SERIES_TYPE_PREFERENCE else len(SERIES_TYPE_PREFERENCE)

        for s in sorted(series, key=rank):
            result = self._fetch_series_data(station_id, group["id"], s["id"], params, latest_only)
            tally[result.outcome] += 1
            if result.rows:
                return result.rows
        return []

    def _fetch_series_data(
        self,
        station_id: str,
        group_id: int | str,
        series_id: int | str,
        params: dict[str, str],
        latest_only: bool,
    ) -> _Fetch:
        """GET one series, classifying the three no-data cases separately."""
        leaf = "bottom" if latest_only else "data"
        url = f"{API_BASE}/stations/{station_id}/timeseriesgroups/{group_id}/timeseries/{series_id}/{leaf}/"
        try:
            text = self.client.get_text(url, params=None if latest_only else (params or None))
        except Exception as exc:  # noqa: BLE001 - classified below, not swallowed
            outcome = _classify_failure(exc)
            logger.debug("OpenHi: %s for %s (%s)", outcome, url, exc)
            return _Fetch([], outcome)

        if not text.strip():
            # 200 with a zero-byte body: catalogued, never populated. This is
            # data-absent, not an error, and must not be reported as one (#165).
            logger.debug("OpenHi: series %s is catalogued but empty", series_id)
            return _Fetch([], "empty")
        rows = _parse_csv(text)
        if not rows:
            # Timestamps arrived but every value column was blank: the sensor
            # is reporting and the readings are gaps. Also data-absent, and
            # worth telling apart from "never populated" when chasing a source.
            logger.debug("OpenHi: series %s returned only null values", series_id)
            return _Fetch([], "no_values")
        return _Fetch(rows, "ok")

    # ── normalisation ────────────────────────────────────────────────────

    def normalise(self, raw: list[dict[str, Any]]) -> Sequence[BaseModel]:
        """Convert rows into the model matching each row's variable.

        Discharge becomes :class:`StreamflowReading` in m3/s and stage
        :class:`WaterLevelReading` in metres, both converted from whatever
        unit the API declares; a unit that is not convertible drops the row
        rather than being assumed. Rainfall, evaporation and the atmospheric
        and soil variables become :class:`ClimateReading`, and the in-water
        chemistry becomes :class:`WaterQualitySample`.
        """
        records: list[BaseModel] = []
        skipped = 0
        for row in raw:
            try:
                record = self._to_record(row)
            except (ValueError, KeyError, TypeError) as exc:
                record = None
                logger.debug("Skipping OpenHi row: %s", exc)
            if record is None:
                skipped += 1
                continue
            records.append(record)

        if skipped:
            logger.warning(
                "OpenHi normalise(): skipped %d/%d row(s) (unconvertible unit or invalid fields)",
                skipped,
                len(raw),
            )
        return records

    def _to_record(self, row: dict[str, Any]) -> BaseModel | None:
        lat, lon = row.get("latitude"), row.get("longitude")
        location = GeoLocation(latitude=lat, longitude=lon) if lat is not None and lon is not None else None
        unit = (row.get("unit") or "").strip()
        value = float(row["value"])
        stamp = row["datetime"]
        common = {
            "source": DataSource.GREECE_OPENHI,
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
                reading_datetime=stamp,
                discharge_cms=value * factor,
                source_type="in_situ",
            )
        if variable == "water_level":
            factor = LEVEL_UNIT_FACTORS.get(unit)
            if factor is None:
                return None
            return WaterLevelReading(**common, reading_datetime=stamp, water_level=value * factor)
        if variable == "water_quality":
            return WaterQualitySample(
                **common,
                sample_datetime=stamp,
                parameter=row["parameter"],
                value=value,
                unit=unit,
            )
        # precipitation, evapotranspiration and climate all land here.
        return ClimateReading(
            **common,
            altitude_m=row.get("altitude_m"),
            sample_datetime=stamp,
            parameter=row["parameter"],
            value=value,
            unit=unit,
        )


# ── module helpers ───────────────────────────────────────────────────────


def _parse_point(ewkt: str | None) -> tuple[float, float] | None:
    """Extract ``(lat, lon)`` from ``SRID=4326;POINT (lon lat)``.

    The API has no latitude/longitude fields, only this EWKT string, and the
    coordinate order inside it is **lon lat**. Anything not declared as SRID
    4326 is rejected rather than read as degrees.
    """
    if not ewkt:
        return None
    if "SRID=" in ewkt and "SRID=4326" not in ewkt:
        return None
    match = _POINT_RE.search(ewkt)
    if match is None:
        return None
    lon, lat = float(match.group(1)), float(match.group(2))
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _parse_csv(text: str) -> list[tuple[datetime, float]]:
    """Parse headerless ``timestamp,value,flags`` rows.

    Rows with an empty value column are gaps in the telemetry (the first row
    of station 1458's discharge is exactly that) and are dropped.
    """
    out: list[tuple[datetime, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2 or not parts[1].strip():
            continue
        try:
            out.append((_parse_timestamp(parts[0].strip()), float(parts[1].strip())))
        except ValueError:
            continue
    return out


def _parse_timestamp(value: str) -> datetime:
    """Parse ``YYYY-MM-DD HH:MM`` (with optional seconds or a ``T`` separator).

    Kept tz-naive to match the other collectors. Python 3.10 cannot parse a
    trailing ``Z`` in ``fromisoformat``, so it is normalised first.
    """
    value = value.replace("T", " ").replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(value).replace(tzinfo=None)


def _classify_failure(exc: Exception) -> str:
    """Tell an authorisation refusal apart from a transport failure.

    ``CachedHTTPClient`` retries then raises ``RuntimeError`` with the last
    httpx error as ``__cause__``, so the status code is read from there rather
    than matched out of the message text.
    """
    cause = getattr(exc, "__cause__", None)
    status = getattr(getattr(cause, "response", None), "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return "restricted" if status in (401, 403) else "failed"


def _as_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    return date.fromisoformat(text[:10]) if len(text) >= 10 else None


def _window_params(start: str | date | None, end: str | date | None) -> dict[str, str]:
    """Build the server-side ``start_date`` / ``end_date`` filter."""
    params: dict[str, str] = {}
    lo, hi = _as_date(start), _as_date(end)
    if lo is not None:
        params["start_date"] = lo.isoformat()
    if hi is not None:
        params["end_date"] = hi.isoformat()
    return params
