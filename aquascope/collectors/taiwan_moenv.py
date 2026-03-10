"""
Collector for Taiwan Ministry of Environment (MOENV) water quality data.

Data portal : https://data.moenv.gov.tw
Datasets    :
  - River water quality monitoring (dataset id: AQX_P_07)
  - Monthly tap water quality sampling
  - Beach / recreational water quality

All endpoints follow:
    GET https://data.moenv.gov.tw/api/v2/{dataset_id}
        ?api_key={api_key}
        &limit={limit}
        &offset={offset}
        &format=json
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
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

# ── Dataset IDs ──────────────────────────────────────────────────────
RIVER_WQ_DATASET = "AQX_P_07"
TAP_WATER_DATASET = "AQX_P_432"

MOENV_BASE = "https://data.moenv.gov.tw/api/v2"


class TaiwanMOENVCollector(BaseCollector):
    """
    Collect river water-quality monitoring data from Taiwan MOENV.

    Parameters
    ----------
    api_key : str
        Free key obtained at https://data.moenv.gov.tw/en/apikey
    dataset_id : str
        Dataset identifier (default: river water quality ``AQX_P_07``).
    """

    name = "taiwan_moenv"

    PARAMETER_MAP: dict[str, str] = {
        "溶氧量": "DO",
        "生化需氧量": "BOD5",
        "化學需氧量": "COD",
        "懸浮固體": "SS",
        "氨氮": "NH3-N",
        "總磷": "TP",
        "導電度": "Conductivity",
        "氫離子濃度指數": "pH",
        "水溫": "Temperature",
        "大腸桿菌群": "E.coli",
    }

    def __init__(
        self,
        api_key: str = "",
        dataset_id: str = RIVER_WQ_DATASET,
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=MOENV_BASE,
                rate_limiter=RateLimiter(max_calls=20, period_seconds=60),
            )
        )
        self.api_key = api_key
        self.dataset_id = dataset_id

    # ── fetch ────────────────────────────────────────────────────────
    def fetch_raw(self, limit: int = 1000, offset: int = 0, **kwargs) -> list[dict]:
        """
        Page through MOENV open-data endpoint and return raw records.
        """
        all_records: list[dict] = []
        while True:
            params: dict[str, Any] = {
                "limit": limit,
                "offset": offset,
                "format": "json",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            data = self.client.get_json(self.dataset_id, params=params)
            records = data.get("records", [])
            if not records:
                break
            all_records.extend(records)
            # stop if we got fewer than the page size (last page)
            if len(records) < limit:
                break
            offset += limit
            logger.debug("Fetched %d cumulative records …", len(all_records))

        return all_records

    # ── normalise ────────────────────────────────────────────────────
    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        samples: list[WaterQualitySample] = []
        for rec in raw:
            try:
                item_name = rec.get("itemname", "")
                param = self.PARAMETER_MAP.get(item_name, rec.get("itemengabbreviation", item_name))
                value_str = rec.get("itemvalue", "")
                if not value_str or value_str in ("-", "ND", "--"):
                    continue
                value = float(value_str)

                loc = None
                if rec.get("twd97lat") and rec.get("twd97lon"):
                    loc = GeoLocation(
                        latitude=float(rec["twd97lat"]),
                        longitude=float(rec["twd97lon"]),
                    )

                sample_dt = datetime.strptime(
                    f"{rec.get('sampledate', '')} {rec.get('sampletime', '00:00')}",
                    "%Y-%m-%d %H:%M",
                )

                samples.append(
                    WaterQualitySample(
                        source=DataSource.TAIWAN_MOENV,
                        station_id=rec.get("sitename", "unknown"),
                        station_name=rec.get("siteengname"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=param,
                        value=value,
                        unit=rec.get("itemunit", "mg/L"),
                        basin=rec.get("basin"),
                        river=rec.get("river"),
                        county=rec.get("county"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping record: %s — %s", exc, rec)
        return samples
