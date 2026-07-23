"""
Collector for UN SDG 6 (Clean Water and Sanitation) indicator data.

Uses the official UN Statistics SDG API:
    https://unstats.un.org/SDGAPI/swagger/

Indicators:
  6.1.1 SH_H2O_SAFE     Safely managed drinking water services (%)
  6.2.1 SH_SAN_SAFE     Safely managed sanitation services (%)
  6.3.1 EN_H2O_WASTAVR  Safely treated wastewater (%)
  6.3.2 EN_H2O_AMBQ     Water bodies with good ambient water quality (%)
  6.4.1 ER_H2O_USEEFF   Water-use efficiency (USD/m^3)
  6.4.2 ER_H2O_STRESS   Water stress: freshwater withdrawal vs. available (%)
  6.5.1 EN_WBE_IWRM     IWRM implementation (0-100)
  6.5.2 EN_WBE_TCA      Transboundary cooperation (%)
  6.6.1 ER_H2O_WBADQ    Water-related ecosystems change (%)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import SDG6Indicator
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

SDG6_BASE = "https://unstats.un.org/SDGAPI/v1/sdg"

# SDG 6 indicator → official UN series code
INDICATOR_TO_SERIES: dict[str, str] = {
    "6.1.1": "SH_H2O_SAFE",
    "6.2.1": "SH_SAN_SAFE",
    "6.3.1": "EN_H2O_WASTAVR",
    "6.3.2": "EN_H2O_AMBQ",
    "6.4.1": "ER_H2O_USEEFF",
    "6.4.2": "ER_H2O_STRESS",
    "6.5.1": "EN_WBE_IWRM",
    "6.5.2": "EN_WBE_TCA",
    "6.6.1": "ER_H2O_WBADQ",
}

ALL_SDG6_INDICATORS = list(INDICATOR_TO_SERIES.keys())

# ISO3 → UN M49 numeric area code for the most common countries the dashboard
# might query. The UN API only accepts M49 codes; we translate transparently.
ISO3_TO_M49: dict[str, int] = {
    "USA": 840, "CAN": 124, "MEX": 484, "BRA": 76, "ARG": 32, "CHL": 152,
    "GBR": 826, "FRA": 250, "DEU": 276, "ITA": 380, "ESP": 724, "PRT": 620,
    "NLD": 528, "BEL": 56, "CHE": 756, "AUT": 40, "SWE": 752, "NOR": 578,
    "FIN": 246, "DNK": 208, "POL": 616, "RUS": 643, "UKR": 804, "TUR": 792,
    "CHN": 156, "JPN": 392, "KOR": 410, "TWN": 158,  # Taiwan often absent in UN data
    "IDN": 360, "PHL": 608, "VNM": 704, "THA": 764, "MYS": 458, "SGP": 702,
    "IND": 356, "PAK": 586, "BGD": 50, "LKA": 144, "NPL": 524,
    "AUS": 36, "NZL": 554,
    "ZAF": 710, "EGY": 818, "NGA": 566, "KEN": 404, "ETH": 231, "MAR": 504,
    "SAU": 682, "ARE": 784, "ISR": 376, "IRN": 364, "IRQ": 368,
    "BFA": 854, "MLI": 466, "SEN": 686, "GHA": 288, "CIV": 384,
}


def _to_m49(code: str) -> int | None:
    """Convert ISO3 or numeric string to M49 integer code."""
    code = code.strip().upper()
    if code.isdigit():
        return int(code)
    return ISO3_TO_M49.get(code)


class SDG6Collector(BaseCollector):
    """
    Collect SDG 6 indicator data per country/year from the UN-Stats API.
    """

    name = "sdg6"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=SDG6_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=86400,
            )
        )

    def fetch_raw(
        self,
        indicator_codes: list[str] | None = None,
        country_codes: str | None = None,
        page_size: int = 200,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch SDG 6 indicator records.

        Parameters
        ----------
        indicator_codes : list[str] | None
            e.g. ["6.4.2", "6.4.1"]. Defaults to all 9 SDG 6 indicators.
        country_codes : str | None
            Comma-separated ISO3 or M49 numeric codes, e.g. "USA,DEU,IND".
            Omit to return data for all countries.
        page_size : int
            Records per API page (max 5000 per UN docs).
        """
        codes = indicator_codes or ALL_SDG6_INDICATORS
        area_codes: list[int] = []
        if country_codes:
            for raw in country_codes.split(","):
                m49 = _to_m49(raw)
                if m49 is not None:
                    area_codes.append(m49)
                else:
                    logger.debug("Unknown country code %r — skipping", raw)

        all_records: list[dict] = []
        for code in codes:
            series = INDICATOR_TO_SERIES.get(code)
            if not series:
                logger.warning("Unknown SDG6 indicator %r — skipping", code)
                continue

            params: dict[str, Any] = {"seriesCode": series, "pageSize": page_size}
            if area_codes:
                params["areaCode"] = area_codes

            page = 1
            while True:
                params["page"] = page
                data = self.client.get_json("Series/Data", params=params)
                items = data.get("data", [])
                # Tag each record with the SDG indicator code (the API returns
                # the series code; we keep the human-friendly SDG dotted code).
                for it in items:
                    it["_sdg_indicator"] = code
                all_records.extend(items)
                if page >= data.get("totalPages", 1):
                    break
                page += 1

        return all_records

    def normalise(self, raw: list[dict]) -> Sequence[SDG6Indicator]:
        records: list[SDG6Indicator] = []
        for rec in raw:
            try:
                val = rec.get("value")
                year_raw = rec.get("timePeriodStart")
                year = int(float(year_raw)) if year_raw not in (None, "") else 0
                value: float | None
                if val in (None, "", "NA"):
                    value = None
                else:
                    try:
                        value = float(val)
                    except (TypeError, ValueError):
                        value = None
                records.append(
                    SDG6Indicator(
                        indicator_code=rec.get("_sdg_indicator", ""),
                        indicator_name=rec.get("seriesDescription"),
                        country_code=str(rec.get("geoAreaCode", "")),
                        country_name=rec.get("geoAreaName"),
                        year=year,
                        value=value,
                        unit=(rec.get("attributes") or {}).get("Units"),
                        series_code=rec.get("series"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping SDG6 record: %s", exc)
        return records
