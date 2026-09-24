"""Taiwan Central Weather Administration (CWA) climate-station collector.

Daily climate observations (rainfall, temperature, humidity, radiation, wind,
pan evaporation) from Taiwan's official weather-station network, via the CODIS
backend API (codis.cwa.gov.tw). No API key required. Archive depth reaches back
decades (verified to 1960 for station 466920, Taipei), which makes this the
observed-forcing layer for Caravan/CAMELS-style Taiwan datasets (#100, #177).

Endpoints
---------
- ``POST /api/station_list`` — station metadata (WGS84 coordinates, altitude,
  names, operating period), grouped by station attribute (``cwb`` manned,
  ``auto`` automatic).
- ``POST /api/station`` with ``type=report_month`` and a ``start``/``end``
  window — daily records, one structured block per variable.

The host's certificate chain lacks the Subject Key Identifier extension, which
Python 3.13+ rejects under its default strict profile; the client is created
with ``relax_strict_tls=True`` (full chain and hostname verification stay on).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.station import Station, in_bbox
from aquascope.schemas.water_data import ClimateReading, DataSource, GeoLocation
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

CODIS_BASE = "https://codis.cwa.gov.tw/api"


def codis_station_type(station_id: str) -> str:
    """The CODIS ``stn_type`` a station answers to, from its id.

    Checked against CODIS on 2026-09-24, one open station per id prefix: numeric ids (466920 Taipei) are the
    staffed network, ``cwb``; ``C0`` and ``C1`` are automatic stations, ``auto_C0`` and ``auto_C1``; every
    other prefix in the catalogue (A2, B2, C2, CA, E2, G2, K2, U2, V2) is the agricultural network, ``agr``.
    Asked as ``cwb`` (the old fixed default), all of those answered empty: the 2026-09-23 harvest got 0 of 15.
    """
    sid = str(station_id).strip().upper()
    if sid[:1].isdigit():
        return "cwb"
    if sid.startswith("C0"):
        return "auto_C0"
    if sid.startswith("C1"):
        return "auto_C1"
    return "agr"

# parameter key → (CODIS variable block, sub-field, unit)
PARAMETER_MAP: dict[str, tuple[str, str, str]] = {
    "rainfall_mm": ("Precipitation", "Accumulation", "mm"),
    "temperature_mean_c": ("AirTemperature", "Mean", "degC"),
    "temperature_max_c": ("AirTemperature", "Maximum", "degC"),
    "temperature_min_c": ("AirTemperature", "Minimum", "degC"),
    "relative_humidity_pct": ("RelativeHumidity", "Mean", "%"),
    "solar_radiation_mj_m2": ("GlobalSolarRadiation", "Accumulation", "MJ/m2"),
    "wind_speed_ms": ("WindSpeed", "Mean", "m/s"),
    "pan_evaporation_mm": ("EvaporationClassAPan", "Accumulation", "mm"),
}

# Accumulation-type parameters cannot be negative; CODIS occasionally carries
# negative sentinel/QC artifacts (e.g. pan evaporation -0.9) that must not
# leak into analyses.
_NON_NEGATIVE = {"rainfall_mm", "solar_radiation_mj_m2", "pan_evaporation_mm"}


def _parse_cwa_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


class TaiwanCWACollector(BaseCollector):
    """Collect daily climate observations from Taiwan CWA stations via CODIS."""

    name = "taiwan_cwa"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=CODIS_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                relax_strict_tls=True,
            )
        )
        self._station_meta: dict[str, dict] | None = None

    # ── station metadata ─────────────────────────────────────────────
    def _load_station_meta(self) -> dict[str, dict]:
        """Fetch and cache the station registry (id → metadata dict)."""
        if self._station_meta is not None:
            return self._station_meta
        meta: dict[str, dict] = {}
        try:
            payload = self.client.post_json("station_list", form={})
            for group in payload.get("data", []):
                for item in group.get("item", []):
                    sid = item.get("stationID")
                    if sid:
                        meta[sid] = item
        except RuntimeError as exc:
            logger.warning("CODIS station_list unavailable (%s); records will lack coordinates.", exc)
        self._station_meta = meta
        return meta

    def stations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        variable: str | None = None,
        max_items: int | None = None,
    ) -> list[Station]:
        """CWA station catalog from CODIS ``station_list`` (manned, agricultural, automatic).

        Every station reports precipitation; the manned (``cwb``) network also
        carries the full climate set. ``extra["attribute"]`` keeps the CODIS
        group (``cwb`` / ``agr`` / ``auto``) so callers can pick long-archive
        stations.
        """
        if variable and variable not in ("precipitation", "climate"):
            return []
        payload = self.client.post_json("station_list", form={})
        stations: list[Station] = []
        for group in payload.get("data", []):
            attribute = group.get("stationAttribute")
            for item in group.get("item", []):
                sid = item.get("stationID")
                lat, lon = item.get("latitude"), item.get("longitude")
                if not sid or lat is None or lon is None:
                    continue
                lat, lon = float(lat), float(lon)
                if not in_bbox(lat, lon, bbox):
                    continue
                stations.append(
                    Station(
                        source="taiwan_cwa",
                        station_id=str(sid),
                        name=item.get("stationName"),
                        latitude=lat,
                        longitude=lon,
                        variables=("climate", "precipitation"),
                        period_start=_parse_cwa_date(item.get("stationStartDate")),
                        period_end=_parse_cwa_date(item.get("stationEndDate")),
                        url=f"https://codis.cwa.gov.tw/StationData?station={sid}",
                        country="TWN",
                        extra={
                            k: v
                            for k, v in (
                                ("attribute", attribute),
                                ("altitude_m", item.get("altitude")),
                                ("county", item.get("countryName")),
                            )
                            if v not in (None, "")
                        },
                    )
                )
                if max_items is not None and len(stations) >= max_items:
                    return stations
        return stations

    # ── fetch ────────────────────────────────────────────────────────
    def fetch_raw(
        self,
        station_ids: list[str] | str | None = None,
        start: str | None = None,
        end: str | None = None,
        stn_type: str | None = None,
        **kwargs,
    ) -> list[dict]:
        """Fetch daily climate records for one or more stations.

        Parameters
        ----------
        station_ids : list[str] | str, optional
            CWA station identifiers (e.g. ``["466920"]`` for Taipei).
            Defaults to Taipei. A single string is accepted.
        start, end : str, optional
            Inclusive ISO dates (``"YYYY-MM-DD"``). Defaults to the last
            30 days. Requests are paged in calendar-year windows to stay
            polite with the unauthenticated API.
        stn_type : str, optional
            CODIS station attribute. By default it follows from the station id
            (:func:`codis_station_type`): ``"cwb"`` for the staffed synoptic
            network (numeric ids, long archives), ``"auto_C0"`` / ``"auto_C1"``
            for automatic stations, ``"agr"`` for the agricultural network.
            Asked with the wrong one, CODIS returns an empty answer, not an
            error.
        """
        if station_ids is None:
            station_ids = ["466920"]
        elif isinstance(station_ids, str):
            station_ids = [station_ids]
        if kwargs.get("station_id"):
            station_ids = [str(kwargs["station_id"])]

        end_d = date.fromisoformat(end) if end else date.today()
        start_d = date.fromisoformat(start) if start else end_d - timedelta(days=30)
        if start_d > end_d:
            msg = f"start ({start_d}) is after end ({end_d})"
            raise ValueError(msg)

        raw: list[dict] = []
        for sid in station_ids:
            win_start = start_d
            while win_start <= end_d:
                # Calendar-year window (the API serves at most ~366 days).
                win_end = min(date(win_start.year, 12, 31), end_d)
                form = {
                    "type": "report_month",
                    "date": f"{win_start.isoformat()}T00:00:00",
                    "start": f"{win_start.isoformat()}T00:00:00",
                    "end": f"{win_end.isoformat()}T00:00:00",
                    "stn_ID": sid,
                    "stn_type": stn_type or codis_station_type(sid),
                }
                payload = self.client.post_json("station", form=form)
                if payload and payload.get("data"):
                    for block in payload["data"]:
                        for day in block.get("dts", []):
                            raw.append({"station_id": sid, "day": day})
                else:
                    logger.warning(
                        "CODIS returned no data for station %s window %s..%s "
                        "(gap or unknown station).",
                        sid, win_start, win_end,
                    )
                win_start = date(win_start.year + 1, 1, 1)
        return raw

    # ── normalise ────────────────────────────────────────────────────
    def normalise(self, raw: list[dict]) -> Sequence[ClimateReading]:
        meta = self._load_station_meta()
        records: list[ClimateReading] = []
        for entry in raw:
            sid = entry["station_id"]
            day = entry["day"]
            try:
                dt = datetime.fromisoformat(day["DataDate"])
            except (KeyError, ValueError, TypeError):
                continue

            m = meta.get(sid, {})
            loc = None
            if m.get("latitude") is not None and m.get("longitude") is not None:
                loc = GeoLocation(latitude=m["latitude"], longitude=m["longitude"])

            for param, (block_name, sub, unit) in PARAMETER_MAP.items():
                block = day.get(block_name)
                if not isinstance(block, dict):
                    continue
                val = block.get(sub)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if param in _NON_NEGATIVE and fval < 0:
                    continue
                records.append(
                    ClimateReading(
                        source=DataSource.TAIWAN_CWA,
                        station_id=sid,
                        station_name=m.get("stationName"),
                        location=loc,
                        altitude_m=m.get("altitude"),
                        sample_datetime=dt,
                        parameter=param,
                        value=fval,
                        unit=unit,
                    )
                )
        return records
