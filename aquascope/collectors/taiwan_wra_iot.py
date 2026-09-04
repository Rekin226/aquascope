"""
Collector for WRA 水利署水文開放資料 IoT API (v2).

API docs : https://iot.wra.gov.tw  (Swagger: https://iot.wra.gov.tw/swagger/v1/swagger.json)

Provides real-time access to:
  - Groundwater level data (地下水位) — unauthenticated (免驗證).

Rainfall is intentionally NOT supported by this collector. In the v2 API the
only rainfall endpoints (``/precipitation/basins``, ``/precipitation/CwaFormat``)
are gated behind "高階會員" (higher-tier membership) and return 401/403
without an API key — there is no free/anonymous rainfall endpoint any more.
Passing ``data_type="rainfall"`` raises ``NotImplementedError`` rather than
silently 404-looping or requiring undocumented credentials; see #169.

Response shape (verified 2026-08-31 against ``/groundwaterlevel/stations``):
a flat JSON list of station objects, each with station metadata and a nested
``Measurements`` list (one entry per physical quantity — currently always a
single 地下水位 reading per station). Note the API's own field name typo,
``Longtiude`` (not ``Longitude``), which we read as-is since it is what the
server actually sends.

``iot.wra.gov.tw`` chains to the Taiwan Government Root CA, whose
certificates lack the Subject Key Identifier extension. Python 3.13+ rejects
that under its default strict profile (``ssl.VERIFY_X509_STRICT``), which
surfaces as ``SSL: CERTIFICATE_VERIFY_FAILED, Missing Subject Key
Identifier`` even though the chain itself is otherwise valid (system curl,
which trusts the OS keychain, succeeds against the same host). The client is
created with ``relax_strict_tls=True``, which drops only that strict-profile
check — full chain and hostname verification stay on (#169, see also #177 /
``taiwan_cwa.py`` for the same fix against ``codis.cwa.gov.tw``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

IOT_BASE = "https://iot.wra.gov.tw"

# Path per supported data type. "rainfall" is deliberately absent — see the
# module docstring; requesting it raises NotImplementedError in __init__.
_DATA_TYPE_PATHS: dict[str, str] = {
    "groundwater": "groundwaterlevel/stations",
}

# Hint the server to return JSON rather than an HTML error page.
_JSON_HEADERS = {"Accept": "application/json"}

_PARAM_NAMES: dict[str, str] = {
    "groundwater": "GroundwaterLevel",
}

_RAINFALL_ERROR = (
    "data_type='rainfall' is not supported: the v2 IoT API only exposes "
    "rainfall via /precipitation/basins and /precipitation/CwaFormat, both "
    "restricted to higher-tier (高階會員) accounts and requiring an API key "
    "this collector does not have. See https://iot.wra.gov.tw/swagger for "
    "the current endpoint list. (#169)"
)


class TaiwanWRAIoTCollector(BaseCollector):
    """
    Collect real-time hydrological data from the WRA IoT open-data API.

    Parameters
    ----------
    data_type : str
        Currently only ``"groundwater"`` (地下水位) is supported without an
        API key. Defaults to ``"groundwater"``.
    """

    name = "taiwan_wra_iot"

    def __init__(
        self,
        data_type: str = "groundwater",
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=IOT_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=600,
                relax_strict_tls=True,
            )
        )
        if data_type == "rainfall":
            raise NotImplementedError(_RAINFALL_ERROR)
        if data_type not in _DATA_TYPE_PATHS:
            raise ValueError(
                f"data_type must be one of {sorted(_DATA_TYPE_PATHS)}, got {data_type!r}"
            )
        self.data_type = data_type

    def fetch_raw(self, **kwargs) -> list[dict]:
        """Fetch the station list (each with a nested ``Measurements`` array).

        ``CachedHTTPClient.get_json`` strips any BOM / leading whitespace and
        checks Content-Type before parsing, so a non-JSON body (e.g. an HTML
        error page from a stale path) surfaces as ``ValueError`` with a
        preview of the response rather than an opaque ``JSONDecodeError``.
        """
        path = _DATA_TYPE_PATHS[self.data_type]
        try:
            data = self.client.get_json(path, headers=_JSON_HEADERS)
        except ValueError as exc:
            raise RuntimeError(
                f"[{self.name}] {path!r} returned a non-JSON or malformed body: {exc}. "
                f"The endpoint may have moved again — check "
                f"https://iot.wra.gov.tw/swagger and update _DATA_TYPE_PATHS."
            ) from exc

        if isinstance(data, list):
            return data
        # Defensive fallback in case the API ever wraps the list in an
        # envelope object (it does not today, as of 2026-08-31).
        return data.get("Data", data.get("data", data.get("records", [])))

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        """Flatten each station's nested ``Measurements`` into samples.

        One ``WaterQualitySample`` is emitted per (station, measurement)
        pair — today that's one reading per station, but the API models it
        as a list so a station could in principle report more than one
        physical quantity in the future.
        """
        param_name = _PARAM_NAMES[self.data_type]
        samples: list[WaterQualitySample] = []

        for station in raw:
            try:
                loc = None
                lat = station.get("Latitude")
                # The API's own field name is misspelled "Longtiude" — that
                # is what the server actually sends, not a typo on our end.
                lon = station.get("Longtiude") or station.get("Longitude")
                if lat is not None and lon is not None:
                    loc = GeoLocation(latitude=float(lat), longitude=float(lon))

                station_id = str(station.get("StationId") or "unknown")
                station_name = station.get("Name")
                county = station.get("CountyName")

                for meas in station.get("Measurements", []):
                    try:
                        value = meas.get("Value")
                        if value is None or str(value).strip() in ("", "-", "--", "ND"):
                            continue

                        time_str = meas.get("TimeStamp") or ""
                        sample_dt = (
                            datetime.fromisoformat(time_str) if time_str else datetime.utcnow()
                        )

                        samples.append(
                            WaterQualitySample(
                                source=DataSource.TAIWAN_WRA_IOT,
                                station_id=station_id,
                                station_name=station_name,
                                location=loc,
                                sample_datetime=sample_dt,
                                parameter=param_name,
                                value=float(value),
                                unit=meas.get("SIUnit") or "m",
                                county=county,
                            )
                        )
                    except (ValueError, KeyError, TypeError) as exc:
                        logger.debug(
                            "Skipping WRA IoT measurement for station %s: %s",
                            station_id,
                            exc,
                        )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping WRA IoT station record: %s", exc)

        return samples
