"""
Collector for Korea WAMIS — Water Management Information System.

WAMIS (국가 수자원관리 종합정보시스템) is operated by Korea's Ministry of
Environment.  The data portal is at http://www.wamis.go.kr/ and the
Open API at https://www.water.or.kr/.

The collector fetches water-level, discharge, water-quality, and dam-storage
data and normalises them into AquaScope's unified schema.
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
# Major river basins (English → Korean)
# ---------------------------------------------------------------------------
KOREA_MAJOR_BASINS: dict[str, str] = {
    "Han": "한강",
    "Nakdong": "낙동강",
    "Geum": "금강",
    "Yeongsan": "영산강",
    "Seomjin": "섬진강",
    "Anseong": "안성천",
    "Hyeongsan": "형산강",
    "Tamjin": "탐진강",
    "Mangyeong": "만경강",
    "Dongjin": "동진강",
}

# ---------------------------------------------------------------------------
# Korean parameter name → English equivalent
# ---------------------------------------------------------------------------
PARAMETER_MAP_KO: dict[str, str] = {
    "수위": "water_level",
    "유량": "discharge",
    "강수량": "rainfall",
    "수온": "water_temperature",
    "pH": "pH",
    "DO": "DO",
    "BOD": "BOD",
    "COD": "COD",
    "SS": "SS",
    "T-N": "Total Nitrogen",
    "T-P": "Total Phosphorus",
    "전기전도도": "conductivity",
    "탁도": "turbidity",
    "총대장균군": "coliform",
    "클로로필-a": "chlorophyll_a",
    "저수량": "dam_storage",
}

# ---------------------------------------------------------------------------
# WAMIS parameter keyword → API endpoint segment
# ---------------------------------------------------------------------------
_PARAM_ENDPOINT: dict[str, str] = {
    "water_level": "wl_dubdata",
    "discharge": "rf_dubrfdata",
    "water_quality": "wq_wqdata",
    "dam_storage": "dam_damdata",
}

# ---------------------------------------------------------------------------
# Korean water quality grades (Water Environment Conservation Act)
#
# Each grade maps to threshold values:
#   BOD (mg/L), COD (mg/L), DO (mg/L)
# ---------------------------------------------------------------------------
KOREA_QUALITY_GRADES: dict[str, dict[str, float]] = {
    "Ia": {"BOD": 1.0, "COD": 2.0, "DO": 7.5},
    "Ib": {"BOD": 2.0, "COD": 4.0, "DO": 5.0},
    "II": {"BOD": 3.0, "COD": 5.0, "DO": 5.0},
    "III": {"BOD": 5.0, "COD": 7.0, "DO": 2.0},
    "IV": {"BOD": 8.0, "COD": 9.0, "DO": 2.0},
    "V": {"BOD": 10.0, "COD": 11.0, "DO": 2.0},
    "VI": {"BOD": 10.0, "COD": 11.0, "DO": 0.0},
}

# ---------------------------------------------------------------------------
# Major dams (name → metadata dict)
# ---------------------------------------------------------------------------
KOREA_DAM_LIST: dict[str, dict[str, Any]] = {
    "Chungju": {
        "korean_name": "충주댐",
        "basin": "Han",
        "capacity_mcm": 2750.0,
        "completion_year": 1985,
    },
    "Soyang": {
        "korean_name": "소양강댐",
        "basin": "Han",
        "capacity_mcm": 2900.0,
        "completion_year": 1973,
    },
    "Andong": {
        "korean_name": "안동댐",
        "basin": "Nakdong",
        "capacity_mcm": 1248.0,
        "completion_year": 1976,
    },
    "Hapcheon": {
        "korean_name": "합천댐",
        "basin": "Nakdong",
        "capacity_mcm": 790.0,
        "completion_year": 1989,
    },
    "Imha": {
        "korean_name": "임하댐",
        "basin": "Nakdong",
        "capacity_mcm": 595.0,
        "completion_year": 1993,
    },
    "Daecheong": {
        "korean_name": "대청댐",
        "basin": "Geum",
        "capacity_mcm": 1490.0,
        "completion_year": 1980,
    },
    "Juam": {
        "korean_name": "주암댐",
        "basin": "Seomjin",
        "capacity_mcm": 457.0,
        "completion_year": 1991,
    },
}


class KoreaWAMISCollector(BaseCollector):
    """
    Collect water data from Korea WAMIS Open API.

    Supports water-level, discharge, water-quality, and dam-storage
    observations.  Results are normalised to ``WaterQualitySample``
    records with ``source = DataSource.KOREA_WAMIS``.
    """

    name: str = "korea_wamis"
    BASE_URL: str = "http://www.wamis.go.kr/openapi/"

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
        basin: str | None = None,
        parameter: str = "water_level",
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        """
        Fetch raw observation data from the WAMIS Open API.

        Parameters
        ----------
        station_id : str | None
            WAMIS station code.
        basin : str | None
            Basin name in English (e.g. ``"Han"``).  Mapped to the
            Korean name via ``KOREA_MAJOR_BASINS``.
        parameter : str
            One of ``water_level``, ``discharge``, ``water_quality``,
            ``dam_storage``.
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
        url = f"{self.BASE_URL}{endpoint}.do"

        params: dict[str, str] = {"output": "json"}
        if station_id:
            params["obscd"] = station_id
        if basin:
            basin_ko = KOREA_MAJOR_BASINS.get(basin)
            if basin_ko:
                params["basin"] = basin_ko
            else:
                logger.warning("Unknown basin '%s'; ignoring filter.", basin)
        if start_date:
            params["startdt"] = start_date.replace("-", "")
        if end_date:
            params["enddt"] = end_date.replace("-", "")

        try:
            response = self.client.get(url, params=params)
            data = response.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("list", data.get("data", data.get("results", [data])))
            return []
        except Exception:
            logger.warning("WAMIS API request failed for %s", url, exc_info=True)
            return []

    # ------------------------------------------------------------------ #
    # normalise
    # ------------------------------------------------------------------ #
    def normalise(self, raw: list[dict]) -> list[WaterQualitySample]:
        """
        Normalise raw WAMIS data into ``WaterQualitySample`` records.

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
                param_en = PARAMETER_MAP_KO.get(raw_param, raw_param)

                value = row.get("value")
                if value is None or str(value).strip() in ("", "-", "ND", "--"):
                    continue
                value = float(value)

                dt_str = row.get("datetime", row.get("date", row.get("ymdhm", "")))
                if not dt_str:
                    logger.debug("Skipping WAMIS row without datetime")
                    continue
                sample_dt = datetime.fromisoformat(str(dt_str))

                location = None
                lat = row.get("latitude", row.get("lat"))
                lon = row.get("longitude", row.get("lon"))
                if lat is not None and lon is not None:
                    location = GeoLocation(latitude=float(lat), longitude=float(lon))

                basin_en = row.get("basin", "")
                # Reverse-lookup: if Korean, find the English name
                basin_reverse = {v: k for k, v in KOREA_MAJOR_BASINS.items()}
                basin_name = basin_reverse.get(basin_en, basin_en) or None

                samples.append(
                    WaterQualitySample(
                        source=DataSource.KOREA_WAMIS,
                        station_id=str(row.get("station_code", row.get("obscd", "unknown"))),
                        station_name=row.get("station_name", row.get("obsnm")),
                        location=location,
                        sample_datetime=sample_dt,
                        parameter=param_en,
                        value=value,
                        unit=str(row.get("unit", "")),
                        basin=basin_name,
                        river=row.get("river") or None,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping WAMIS row: %s", exc)

        return samples
