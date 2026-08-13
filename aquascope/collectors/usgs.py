"""
Collector for USGS (United States Geological Survey) water data.

Uses the new OGC-compliant API:
    https://api.waterdata.usgs.gov/ogcapi/v0/

Collections
-----------
- ``daily``       — daily-value statistics (mean, min, max)
- ``sta``         — continuous (instantaneous) sensor readings
- ``discrete``    — discrete field measurements
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

USGS_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"

# Common USGS parameter codes relevant to water quality
PARAM_LABELS: dict[str, str] = {
    "00010": "Temperature",
    "00060": "Discharge",
    "00065": "Gage height",
    "00095": "Conductivity",
    "00300": "DO",
    "00400": "pH",
    "00410": "Alkalinity",
    "00600": "TN",
    "00665": "TP",
    "00680": "TOC",
    "00940": "Chloride",
    "00945": "Sulfate",
    "71846": "NH3-N",
    "80154": "SS",
}

MILES2_TO_KM2 = 2.589988110336
FT3S_TO_M3S = 0.028316846592

class USGSCollector(BaseCollector):
    """
    Collect daily-value water data from USGS via OGC API.

    Parameters
    ----------
    api_key : str | None
        USGS API key for higher rate limits (get one at
        https://api.waterdata.usgs.gov/docs/ogcapi/#api-keys). If omitted,
        the collector reads the ``USGS_API_KEY`` environment variable, and
        falls back to the shared ``DEMO_KEY`` (heavily rate-limited) with a
        warning if neither is set.
    """

    name = "usgs"

    def __init__(
        self,
        api_key: str | None = None,
        client: CachedHTTPClient | None = None,
    ):
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=USGS_BASE,
                rate_limiter=RateLimiter(max_calls=25, period_seconds=60),
            )
        )
        resolved = api_key or os.environ.get("USGS_API_KEY")
        if not resolved:
            logger.warning(
                "No USGS API key provided (pass api_key=... or set USGS_API_KEY). "
                "Falling back to the shared DEMO_KEY, which is heavily "
                "rate-limited and may fail under load."
            )
            resolved = "DEMO_KEY"
        self.api_key = resolved

    def fetch_raw(
        self,
        collection: str = "daily",
        datetime_range: str | None = None,
        days: int | None = None,
        limit: int = 10_000,
        bbox: str | None = None,
        max_items: int | None = 2_000,
        **kwargs,
    ) -> list[dict]:
        """
        Fetch features from a USGS OGC collection.

        Parameters
        ----------
        collection : str
            ``"daily"`` | ``"sta"`` | ``"discrete"``
        datetime_range : str, optional
            Explicit ISO 8601 interval ``"<start>/<end>"`` (USGS does NOT accept
            ISO durations like ``P7D``). If omitted, an interval is built from
            ``days``.
        days : int, optional
            Last N days from now (UTC). Defaults to 30 when ``datetime_range``
            is not supplied.
        limit : int
            Max features per page. Larger values mean fewer round-trips.
        bbox : str, optional
            Bounding box filter ``"minLon,minLat,maxLon,maxLat"`` (WGS84).
            Without this the API returns data for every US monitoring location,
            which can require hundreds of paginated requests.
        max_items : int, optional
            Hard cap on total records fetched (across all pages). Keeps response
            times predictable. ``None`` means no cap.
        """
        if datetime_range is None:
            window_days = days if days is not None else 30
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=window_days)
            datetime_range = (
                f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')}/"
                f"{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            )

        if self.api_key == "DEMO_KEY" or not self.api_key:
            if collection not in ("daily", "sta"):
                raise ValueError(
                    f"Collection '{collection}' is not supported on the keyless legacy USGS API path. "
                    "Only 'daily' and 'sta' collections are supported without an API key. "
                    "Please provide a valid USGS_API_KEY to use the OGC API path. "
                    "For a reliable keyless demo source, use OpenMeteoCollector."
                )

            sites = kwargs.get("station_id") or kwargs.get("sites") or kwargs.get("monitoring_location_id")
            parameter_cd = kwargs.get("parameter") or kwargs.get("parameterCd") or kwargs.get("parameter_code")
            bbox_val = bbox or kwargs.get("bBox")
            state_cd = kwargs.get("stateCd")
            county_cd = kwargs.get("countyCd")
            huc_val = kwargs.get("huc")

            if not any([sites, bbox_val, state_cd, county_cd, huc_val]):
                raise ValueError(
                    "USGS keyless path requires a filter parameter (station_id, bbox, stateCd, countyCd, or huc). "
                    "To request unfiltered data, you must provide a valid USGS_API_KEY via api_key or the "
                    "USGS_API_KEY environment variable. For a reliable keyless demo source, use OpenMeteoCollector."
                )

            parts = datetime_range.split("/")
            start_date = parts[0].split("T")[0] if len(parts) == 2 else None
            end_date = parts[1].split("T")[0] if len(parts) == 2 else None

            endpoint = "dv" if collection == "daily" else "iv"
            url = f"https://waterservices.usgs.gov/nwis/{endpoint}/"

            params = {
                "format": "json",
            }
            if sites:
                params["sites"] = sites
            if parameter_cd:
                params["parameterCd"] = parameter_cd
            if bbox_val:
                params["bBox"] = bbox_val
            if state_cd:
                params["stateCd"] = state_cd
            if county_cd:
                params["countyCd"] = county_cd
            if huc_val:
                params["huc"] = huc_val

            if start_date:
                params["startDT"] = start_date
            if end_date:
                params["endDT"] = end_date

            response = self.client.get_json(url, params=params)
            time_series_list = response.get("value", {}).get("timeSeries", [])

            all_features = []
            for ts in time_series_list:
                source_info = ts.get("sourceInfo", {})
                site_codes = source_info.get("siteCode", [])
                site_id = site_codes[0].get("value", "unknown") if site_codes else "unknown"

                geo_loc = source_info.get("geoLocation", {}).get("geogLocation", {})
                latitude = geo_loc.get("latitude")
                longitude = geo_loc.get("longitude")

                var_info = ts.get("variable", {})
                var_codes = var_info.get("variableCode", [])
                param_code = var_codes[0].get("value", "") if var_codes else ""
                unit = var_info.get("unit", {}).get("unitCode", "")
                no_data_val = var_info.get("noDataValue")

                for values_block in ts.get("values", []):
                    for value in values_block.get("value", []):
                        val = value.get("value")
                        dt = value.get("dateTime")
                        if val is None or dt is None:
                            continue

                        try:
                            float_val = float(val)
                            if no_data_val is not None and abs(float_val - no_data_val) < 1e-3:
                                continue
                        except (ValueError, TypeError):
                            continue

                        drainage_area = source_info.get("drainageArea", None)
                        if drainage_area is not None:
                            try:
                                drainage_area = float(drainage_area)
                            except (TypeError, ValueError):
                                drainage_area = None

                        catchment_area_km2 = (
                            drainage_area * MILES2_TO_KM2
                            if drainage_area is not None
                            else None
                        )

                        all_features.append({
                            "geometry": {
                                "coordinates": [longitude, latitude]
                            },
                            "properties": {
                                "monitoring_location_id": site_id,
                                "parameter_code": param_code,
                                "value": val,
                                "time": dt,
                                "unit_of_measure": unit,
                                "catchment_area_km2": catchment_area_km2
                            }
                        })

            if max_items is not None and len(all_features) >= max_items:
                all_features = all_features[:max_items]

            return all_features

        all_features: list[dict] = []
        params: dict[str, Any] = {
            "f": "json",
            "limit": limit,
            "datetime": datetime_range,
            "api_key": self.api_key,
        }
        if bbox:
            params["bbox"] = bbox

        url = f"collections/{collection}/items"
        while True:
            data = self.client.get_json(url, params=params)
            features = data.get("features", [])
            all_features.extend(features)

            if max_items is not None and len(all_features) >= max_items:
                all_features = all_features[:max_items]
                logger.debug("USGS max_items=%d reached — stopping pagination.", max_items)
                break

            # follow pagination
            next_link = next(
                (lnk["href"] for lnk in data.get("links", []) if lnk.get("rel") == "next"),
                None,
            )
            if not next_link or len(features) == 0:
                break
            # next_link is absolute; switch to direct fetch
            url = next_link
            params = {}

        return all_features

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample | StreamflowReading]:
        samples: Sequence[WaterQualitySample | StreamflowReading] = []
        for feat in raw:
            try:
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [None, None]) if geom else [None, None]

                param_code = props.get("parameter_code", "")
                param_label = PARAM_LABELS.get(param_code, param_code)

                val = props.get("value")
                if val is None:
                    continue

                loc = None
                if coords[0] is not None:
                    loc = GeoLocation(latitude=coords[1], longitude=coords[0])

                if param_code == "00060":  # Discharge
                    discharge_sig_figs = self._count_sig_figs(val)
                    if not discharge_sig_figs:
                        discharge_sig_figs = 3  # default to 3 significant figures if unable to determine
                    discharge_cms = float(val) * FT3S_TO_M3S
                    rounded_discharge_cms = USGSCollector._round_to_sig_figs(discharge_cms, discharge_sig_figs)

                    catchment_area_km2 = props.get("catchment_area_km2", None)
                    if catchment_area_km2 is None:
                        catchment_area_km2 = self._get_monitoring_location_catchment_area(props.get("monitoring_location_id", ""))

                    samples.append(
                        StreamflowReading(
                            source=DataSource.USGS,
                            station_id=props.get("monitoring_location_id"),
                            station_name=props.get("station_name"),
                            location=loc,
                            reading_datetime=datetime.fromisoformat(props["time"]),
                            discharge_cms=rounded_discharge_cms,
                            source_type="in_situ",
                            uncertainty_cms=None,
                            catchment_area_km2=catchment_area_km2,
                            unit="m3/s",
                        )
                    )

                else:
                    samples.append(
                        WaterQualitySample(
                            source=DataSource.USGS,
                            station_id=props.get("monitoring_location_id", "unknown"),
                            location=loc,
                            sample_datetime=datetime.fromisoformat(props["time"]),
                            parameter=param_label,
                            value=float(val),
                            unit=props.get("unit_of_measure", ""),
                        )
                    )

            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping USGS feature: %s", exc)

        return samples

    def _get_monitoring_location_catchment_area(self, location_id: str) -> float | None:
        if not location_id:
            return None

        if not location_id.startswith("USGS-"):
            location_id = f"USGS-{location_id}"

        try:
            feature = self.client.get_json(
                f"collections/monitoring-locations/items/{location_id}",
                params={"f": "json"},
            )
        except RuntimeError:
            logger.warning(
                f"Cannot obtain metadata for station {location_id} - catchment area data is unavailable."
            )
            return None

        area = feature.get("properties", {}).get("drainage_area", None)
        if area is None:
            logger.warning(
                f"Metadata for station {location_id} does not contain catchment area data."
            )
            return None

        sig_figs = USGSCollector._count_sig_figs(area)
        if not sig_figs:
            sig_figs = 3  # default to 3 significant figures if unable to determine
        area_km2 = float(area) * MILES2_TO_KM2
        rounded_catchment_area = USGSCollector._round_to_sig_figs(area_km2, sig_figs)

        return rounded_catchment_area

    @staticmethod
    def _count_sig_figs(value: str | float) -> int:
        text = str(value).strip()

        if not text or text.lower() in {"nan", "+inf", "inf", "-inf"}:
            return 0

        text = text.lstrip("+-")
        if "." in text:
            if text[-1] == ".":
                return len(text.rstrip("."))
            text = text.replace(".", "")
        else:
            text = text.rstrip("0")

        text = text.lstrip("0")
        if not text:
            return 1

        return len(text)

    @staticmethod
    def _round_to_sig_figs(value: float, sigfigs: int) -> float:
        if value == 0 or sigfigs <= 0:
            return value
        digits = sigfigs - int(math.floor(math.log10(abs(value)))) - 1
        return round(value, digits)
