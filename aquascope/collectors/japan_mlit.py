"""
Collector for Japan MLIT — Ministry of Land, Infrastructure, Transport and Tourism.

Data is provided by the Water Information System (水文水質データベース):
    http://www1.river.go.jp/

The collector fetches water-level, discharge, rainfall, and water-quality
observations from MLIT's CGI-based query interface and normalises them into
AquaScope's unified schema.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prefecture codes (ISO 3166-2:JP numeric part)
# ---------------------------------------------------------------------------
PREFECTURE_CODES: dict[str, str] = {
    "Hokkaido": "01",
    "Aomori": "02",
    "Iwate": "03",
    "Miyagi": "04",
    "Akita": "05",
    "Yamagata": "06",
    "Fukushima": "07",
    "Ibaraki": "08",
    "Tochigi": "09",
    "Gunma": "10",
    "Saitama": "11",
    "Chiba": "12",
    "Tokyo": "13",
    "Kanagawa": "14",
    "Niigata": "15",
    "Nagano": "20",
    "Shizuoka": "22",
    "Aichi": "23",
    "Kyoto": "26",
    "Osaka": "27",
    "Hyogo": "28",
    "Hiroshima": "34",
    "Fukuoka": "40",
    "Nagasaki": "42",
    "Kumamoto": "43",
    "Okinawa": "47",
}

# ---------------------------------------------------------------------------
# Major river systems (Japanese name → English transliteration)
# ---------------------------------------------------------------------------
JAPAN_RIVER_SYSTEMS: dict[str, str] = {
    "利根川": "Tone",
    "信濃川": "Shinano",
    "石狩川": "Ishikari",
    "天竜川": "Tenryu",
    "木曽川": "Kiso",
    "淀川": "Yodo",
    "筑後川": "Chikugo",
    "北上川": "Kitakami",
    "最上川": "Mogami",
    "阿賀野川": "Agano",
    "荒川": "Ara",
    "多摩川": "Tama",
    "四万十川": "Shimanto",
    "吉野川": "Yoshino",
    "那珂川": "Naka",
}

# ---------------------------------------------------------------------------
# Japanese parameter name → English equivalent
# ---------------------------------------------------------------------------
PARAMETER_MAP_JA: dict[str, str] = {
    "水位": "water_level",
    "流量": "discharge",
    "雨量": "rainfall",
    "水温": "water_temperature",
    "pH": "pH",
    "DO": "DO",
    "BOD": "BOD",
    "COD": "COD",
    "SS": "SS",
    "大腸菌群数": "coliform",
    "全窒素": "Total Nitrogen",
    "全燐": "Total Phosphorus",
    "透視度": "transparency",
}

# ---------------------------------------------------------------------------
# MLIT parameter keyword → query path segment
# ---------------------------------------------------------------------------
_PARAM_ENDPOINT: dict[str, str] = {
    "water_level": "SrchWaterLevel",
    "discharge": "SrchDischarge",
    "water_quality": "SrchWaterQuality",
    "rainfall": "SrchRainfall",
}

# ---------------------------------------------------------------------------
# Japanese Environmental Quality Standards (Basic Environment Law)
#
# Each classification maps to threshold values:
#   BOD (mg/L), SS (mg/L), DO (mg/L)
# See: https://www.env.go.jp/en/water/wq/wp.pdf
# ---------------------------------------------------------------------------
QUALITY_STANDARDS_JAPAN: dict[str, dict[str, float]] = {
    "AA": {"BOD": 1.0, "SS": 25.0, "DO": 7.5},
    "A": {"BOD": 2.0, "SS": 25.0, "DO": 7.5},
    "B": {"BOD": 3.0, "SS": 25.0, "DO": 5.0},
    "C": {"BOD": 5.0, "SS": 50.0, "DO": 5.0},
    "D": {"BOD": 8.0, "SS": 100.0, "DO": 2.0},
    "E": {"BOD": 10.0, "SS": -1.0, "DO": 2.0},
}


class JapanMLITCollector(BaseCollector):
    """
    Collect water data from Japan MLIT Water Information System.

    Supports water-level, discharge, water-quality, and rainfall
    observations.  Results are normalised to ``WaterQualitySample``
    records with ``source = DataSource.JAPAN_MLIT``.
    """

    name: str = "japan_mlit"
    BASE_URL: str = "http://www1.river.go.jp/cgi-bin/"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=3600,
            )
        )

    # ------------------------------------------------------------------ #
    # fetch_raw
    # ------------------------------------------------------------------ #
    def fetch_raw(
        self,
        station_id: str | None = None,
        prefecture: str | None = None,
        parameter: str = "water_level",
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        """
        Fetch raw observation data from the MLIT Water Information System.

        Parameters
        ----------
        station_id : str | None
            MLIT station code (e.g. ``"305041281005030"``).
        prefecture : str | None
            Prefecture name in English (e.g. ``"Tokyo"``).  Mapped to a
            numeric code via ``PREFECTURE_CODES``.
        parameter : str
            One of ``water_level``, ``discharge``, ``water_quality``,
            ``rainfall``.
        start_date : str | None
            Start date in ISO format (``YYYY-MM-DD``).
        end_date : str | None
            End date in ISO format (``YYYY-MM-DD``).
        **kwargs
            Additional keyword arguments forwarded to the HTTP request.

        Returns
        -------
        list[dict]
            Raw observation records.  Returns an empty list when the
            upstream API is unreachable or no data matches.

        Raises
        ------
        ValueError
            If *parameter* is not one of the supported types.
        """
        if parameter not in _PARAM_ENDPOINT:
            raise ValueError(
                f"Unsupported parameter '{parameter}'. "
                f"Choose from: {list(_PARAM_ENDPOINT.keys())}"
            )

        endpoint = _PARAM_ENDPOINT[parameter]
        url = f"{self.BASE_URL}{endpoint}"

        params: dict[str, str] = {}
        if station_id:
            params["StationID"] = station_id
        if prefecture:
            pref_code = PREFECTURE_CODES.get(prefecture)
            if pref_code:
                params["PrefCode"] = pref_code
            else:
                logger.warning("Unknown prefecture '%s'; ignoring filter.", prefecture)
        if start_date:
            params["StartDate"] = start_date.replace("-", "")
        if end_date:
            params["EndDate"] = end_date.replace("-", "")
        params["format"] = "json"

        try:
            response = self.client.get(url, params=params)
            data = response.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("data", data.get("results", [data]))
            return []
        except Exception:
            logger.warning("MLIT API request failed for %s", url, exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    # normalise
    # ------------------------------------------------------------------ #
    def normalise(self, raw: list[dict]) -> list[WaterQualitySample]:
        """
        Normalise raw MLIT data into ``WaterQualitySample`` records.

        Parameters
        ----------
        raw : list[dict]
            Raw records as returned by ``fetch_raw``.

        Returns
        -------
        list[WaterQualitySample]
            Unified water-quality sample records.
        """
        if not raw:
            return []

        samples: list[WaterQualitySample] = []
        for row in raw:
            try:
                raw_param = str(row.get("parameter", ""))
                param_en = PARAMETER_MAP_JA.get(raw_param, raw_param)

                value = row.get("value")
                if value is None or str(value).strip() in ("", "-", "ND", "--"):
                    continue
                value = float(value)

                dt_str = row.get("datetime", row.get("date", ""))
                if not dt_str:
                    logger.debug("Skipping MLIT row without datetime")
                    continue
                sample_dt = datetime.fromisoformat(str(dt_str))

                location = None
                lat = row.get("latitude")
                lon = row.get("longitude")
                if lat is not None and lon is not None:
                    location = GeoLocation(latitude=float(lat), longitude=float(lon))

                river_jp = row.get("river_system", "")
                river_en = JAPAN_RIVER_SYSTEMS.get(river_jp, river_jp)

                samples.append(
                    WaterQualitySample(
                        source=DataSource.JAPAN_MLIT,
                        station_id=str(row.get("station_code", row.get("station_id", "unknown"))),
                        station_name=row.get("station_name"),
                        location=location,
                        sample_datetime=sample_dt,
                        parameter=param_en,
                        value=value,
                        unit=str(row.get("unit", "")),
                        basin=river_en or None,
                        river=river_en or None,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping MLIT row: %s", exc)

        return samples
