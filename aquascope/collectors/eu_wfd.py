"""
Collector for the EU Water Framework Directive (WFD) water quality data.

The European Environment Agency (EEA) publishes water quality monitoring
data through the WISE (Water Information System for Europe) SoE dataset.

API endpoint: https://discodata.eea.europa.eu/sql
Dataset:      WISE_SOE_Waterbase
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlencode

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

EEA_DISCO_BASE = "https://discodata.eea.europa.eu"

WFD_STATUS_CLASSES: dict[str, int] = {
    "High": 1,
    "Good": 2,
    "Moderate": 3,
    "Poor": 4,
    "Bad": 5,
}
"""Ecological status classes used in WFD assessment, ranked 1 (best) to 5 (worst)."""

# Environmental Quality Standards (EQS) thresholds per parameter.
# Each entry maps a canonical parameter name to a dict of status-class boundaries.
# For "lower_is_better" parameters the value must be <= the threshold;
# for the rest (e.g. Dissolved Oxygen) it must be >= the threshold.
_EQS_THRESHOLDS: dict[str, dict] = {
    "Dissolved oxygen": {
        "lower_is_better": False,
        "Good": 6.0,
        "Moderate": 4.0,
        "unit": "mg/L",
    },
    "BOD5": {
        "lower_is_better": True,
        "Good": 4.0,
        "Moderate": 6.0,
        "unit": "mg/L",
    },
    "Phosphate": {
        "lower_is_better": True,
        "Good": 0.1,
        "Moderate": 0.25,
        "unit": "mg/L",
    },
    "Nitrate": {
        "lower_is_better": True,
        "Good": 25.0,
        "Moderate": 40.0,
        "unit": "mg/L",
    },
    "Ammonium": {
        "lower_is_better": True,
        "Good": 0.3,
        "Moderate": 0.6,
        "unit": "mg/L",
    },
    "pH": {
        "lower_is_better": None,  # range-based
        "Good_min": 6.0,
        "Good_max": 9.0,
        "unit": "",
    },
}


@dataclass
class WFDComplianceResult:
    """Result of a WFD Environmental Quality Standards compliance check.

    Attributes:
        parameter: The assessed parameter name.
        n_samples: Total number of samples evaluated.
        n_compliant: Number of samples meeting the *Good* threshold.
        compliance_pct: Percentage of compliant samples (0–100).
        status_class: Overall WFD status class derived from the mean compliance.
        eqs_threshold: The EQS threshold value(s) used for the *Good* class.
    """

    parameter: str
    n_samples: int
    n_compliant: int
    compliance_pct: float
    status_class: str
    eqs_threshold: float | str


def _classify_value(value: float, spec: dict) -> str:
    """Classify a single measurement against EQS thresholds.

    Parameters:
        value: The measured value.
        spec: EQS specification dict for the parameter.

    Returns:
        WFD status class string.
    """
    lower_is_better = spec.get("lower_is_better")

    # pH uses a range check
    if lower_is_better is None:
        good_min = spec["Good_min"]
        good_max = spec["Good_max"]
        if good_min <= value <= good_max:
            return "Good"
        return "Poor"

    if lower_is_better:
        if value <= spec["Good"]:
            return "Good"
        if value <= spec["Moderate"]:
            return "Moderate"
        return "Poor"

    # Higher is better (e.g. Dissolved Oxygen)
    if value >= spec["Good"]:
        return "Good"
    if value >= spec["Moderate"]:
        return "Moderate"
    return "Poor"


def _overall_status(n_samples: int, n_compliant: int) -> str:
    """Derive the overall status class from compliance percentage.

    Parameters:
        n_samples: Total number of samples.
        n_compliant: Number of compliant samples.

    Returns:
        WFD status class string.
    """
    if n_samples == 0:
        return "Unknown"
    pct = n_compliant / n_samples * 100
    if pct >= 95:
        return "High"
    if pct >= 75:
        return "Good"
    if pct >= 50:
        return "Moderate"
    if pct >= 25:
        return "Poor"
    return "Bad"


def check_wfd_compliance(samples: list[WaterQualitySample], parameter: str) -> WFDComplianceResult:
    """Check whether *samples* meet WFD Environmental Quality Standards.

    Parameters:
        samples: Water quality samples to evaluate.
        parameter: Canonical parameter name (e.g. ``"Dissolved oxygen"``).

    Returns:
        A ``WFDComplianceResult`` summarising compliance.

    Raises:
        ValueError: If *parameter* is not in the EQS threshold table.
    """
    if parameter not in _EQS_THRESHOLDS:
        raise ValueError(
            f"Unknown EQS parameter '{parameter}'. "
            f"Supported: {list(_EQS_THRESHOLDS.keys())}"
        )

    spec = _EQS_THRESHOLDS[parameter]

    # Determine threshold representation for the result
    if spec.get("lower_is_better") is None:
        eqs_threshold: float | str = f"{spec['Good_min']}-{spec['Good_max']}"
    else:
        eqs_threshold = spec["Good"]

    relevant = [s for s in samples if s.parameter == parameter]
    n_samples = len(relevant)
    n_compliant = sum(1 for s in relevant if _classify_value(s.value, spec) == "Good")
    compliance_pct = (n_compliant / n_samples * 100) if n_samples else 0.0
    status_class = _overall_status(n_samples, n_compliant)

    return WFDComplianceResult(
        parameter=parameter,
        n_samples=n_samples,
        n_compliant=n_compliant,
        compliance_pct=compliance_pct,
        status_class=status_class,
        eqs_threshold=eqs_threshold,
    )


class EUWFDCollector(BaseCollector):
    """Collect water quality data from the EEA WISE SoE Waterbase.

    Uses the DiscoData SQL API published by the European Environment Agency.
    """

    name: str = "eu_wfd"

    def __init__(self, client: CachedHTTPClient | None = None):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=EEA_DISCO_BASE,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=3600,
            )
        )

    def _build_query(
        self,
        country: str | None = None,
        water_body_type: str = "river",
        year: int | None = None,
    ) -> str:
        """Build a SQL query string for the DiscoData endpoint.

        Parameters:
            country: ISO-2 country code filter (e.g. ``"DE"``).
            water_body_type: One of ``river``, ``lake``, ``groundwater``.
            year: Calendar year filter.

        Returns:
            SQL query string.
        """
        clauses: list[str] = []
        body_map = {
            "river": "RW",
            "lake": "LW",
            "groundwater": "GW",
        }
        body_code = body_map.get(water_body_type.lower(), "RW")
        clauses.append(f"waterBodyCategory = '{body_code}'")

        if country:
            clauses.append(f"countryCode = '{country.upper()}'")
        if year:
            clauses.append(f"YEAR(phenomenonTimeSamplingDate) = {year}")

        where = " AND ".join(clauses)
        return (
            "SELECT monitoringSiteIdentifier, waterBodyName, countryCode, "
            "parameterWaterBodyCategory, resultMeanValue, resultUom, "
            "phenomenonTimeSamplingDate, lat, lon "
            f"FROM [WISE_SOE_Waterbase].[v1].[Disaggregated_data] WHERE {where}"
        )

    def fetch_raw(
        self,
        country: str | None = None,
        water_body_type: str = "river",
        year: int | None = None,
        **kwargs,
    ) -> list[dict]:
        """Fetch raw water quality records from the EEA DiscoData API.

        Parameters:
            country: ISO-2 country code (e.g. ``"DE"``, ``"FR"``).
            water_body_type: ``"river"``, ``"lake"``, or ``"groundwater"``.
            year: Calendar year to filter on.

        Returns:
            List of raw record dicts from the API response.
        """
        query = self._build_query(country=country, water_body_type=water_body_type, year=year)
        url = f"{EEA_DISCO_BASE}/sql?{urlencode({'query': query})}"

        try:
            data = self.client.get_json(url, use_cache=True)
        except Exception as exc:
            logger.warning("EEA DiscoData fetch failed: %s", exc)
            return []

        if isinstance(data, dict) and "results" in data:
            return data["results"]
        if isinstance(data, list):
            return data
        return []

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        """Convert raw EEA records into unified ``WaterQualitySample`` objects.

        Parameters:
            raw: List of dicts from ``fetch_raw``.

        Returns:
            Sequence of ``WaterQualitySample`` instances.
        """
        samples: list[WaterQualitySample] = []
        for row in raw:
            try:
                val = row.get("resultMeanValue")
                if val is None:
                    continue
                value = float(val)

                date_str = row.get("phenomenonTimeSamplingDate", "")
                if not date_str:
                    continue
                # Try ISO format first, then date-only
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try:
                        sample_dt = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    logger.debug("Unparseable date '%s', skipping row.", date_str)
                    continue

                loc = None
                lat = row.get("lat")
                lon = row.get("lon")
                if lat is not None and lon is not None:
                    try:
                        loc = GeoLocation(latitude=float(lat), longitude=float(lon))
                    except (ValueError, TypeError):
                        pass

                samples.append(
                    WaterQualitySample(
                        source=DataSource.EU_WFD,
                        station_id=row.get("monitoringSiteIdentifier", "unknown"),
                        station_name=row.get("waterBodyName"),
                        location=loc,
                        sample_datetime=sample_dt,
                        parameter=row.get("parameterWaterBodyCategory", "unknown"),
                        value=value,
                        unit=row.get("resultUom", ""),
                        county=row.get("countryCode"),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping EEA row: %s", exc)

        return samples
