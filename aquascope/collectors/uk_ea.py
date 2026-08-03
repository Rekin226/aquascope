"""
Collector for the UK Environment Agency hydrology data service.

Uses the publical Hydrology API:
    https://environment.data.gov.uk/hydrology

For further information on the data collected, view the UKEA Hydrology API's Documentation:
    https://environment.data.gov.uk/hydrology/doc/reference

No API key required (open data, unauthenticated).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterLevelReading, WaterQualitySample
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

UKEA_BASE = "https://environment.data.gov.uk/hydrology"
MAPPED_OBSERVED_PROPERTY_UNITS = {
    "waterFlow": "m3/s",
    "waterLevel": "m",
    "rainfall": "mm",
    "groundwaterLevel": "mAOD (metres Above Ordnance Datum)"
}
MAPPED_OBSERVED_PROPERTY_MEASURE_CODE = {
    "flow": "waterFlow",
    "level": "waterLevel",
    "rainfall": "rainfall",
    "gw": "groundwaterLevel"
}
COLLECTION_PERIOD_VALUES = {
    "15min": 900,
    "daily": 86400
}

class UKEACollector(BaseCollector):
    """Collect readings from the UK Environment Agency hydrology API.

    Parameters
    ----------
    client : CachedHTTPClient | None
        Optional preconfigured HTTP client used for API requests. If not passed, a default client is used.
    """

    name = "uk_ea"

    def __init__(
        self,
        client: CachedHTTPClient | None = None,
    ):
        """Initialise the collector."""
        super().__init__(
            client
            or CachedHTTPClient(
                base_url=UKEA_BASE,
                rate_limiter=RateLimiter(max_calls=5, period_seconds=60),
                cache_ttl_seconds=3600,
            )
        )

    def fetch_raw(
        self,
        collection: str | None = "15min",
        observed_property: str | None = None,
        measure: str | None = None,
        station: str | None = None,
        station_wiski_id: str | None = None,
        bbox: str | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        days: int | None = None,
        limit: int = 10_000,
        max_items: int | None = 2_000,
        **kwargs,
    ) -> list[dict]:
        """Fetch readings from the UK Environment Agency Hydrology API.

        Parameters
        ----------
        collection : str | None
            Frequency of data collection. Supported values are ``"15min"`` and
            ``"daily"``.
        observed_property : str | None
            Normalised observed property name. Supported values:
            ``"waterFlow"``, ``"waterLevel"``,
            ``"rainfall"``, ``"groundwaterLevel"``.
        measure : str | None
            Exact measure identifier. When supplied, the collector uses the
            measure directly and ignores station and collection filters.
        station : str | None
            Station SUID, used to collect data from a specific station.
            A 36-character GUID-style identifier.
        station_wiski_id : str | None
            WISKI station identifier, primarily used to differentiate
            between co-located stations that share the same station SUID.
        bbox : str | None
            Bounding box for location-based queries.
            Expressed as ``"min-long,min-lat,max-long,max-lat"``.
        min_date : str | None
            Inclusive lower bound for the requested date range.
            If not accompanied by a max_date,
            data from the next ``days`` days are collected (until today).
        max_date : str | None
            Inclusive upper bound for the requested date range.
            If the max_date is in the future, the value defaults to today's date.
            If not accompanied by a min_date,
            data from the previous ``days`` days are collected.
        days : int | None
            Relative lookback window in days used when explicit date range
            is not used absent. If there is no date range and no days value passed,
            data from the last 30 days are passed.
        limit : int
            Number of records requested per page from the API.
        max_items : int | None
            Maximum number of records to return after the metadata row is prepended.
            ``None`` means no hard cap.

        Returns
        -------
        list[dict]
            Raw API items with a prepended metadata dictionary used by
            :meth:`normalise`.
        """
        valid_observed_property = observed_property in MAPPED_OBSERVED_PROPERTY_UNITS.keys()
        if not valid_observed_property and not measure:
            raise ValueError(
                "One of the following observedProperty values must be passed: "
                f"{', '.join(MAPPED_OBSERVED_PROPERTY_UNITS.keys())}. "
                f"Alternatively, you can pass an exact measure."
            )

        if measure:
            if observed_property:
                logger.warning(
                    "Both measure and observedProperty provided. "
                    "Ignoring observedProperty and using the exact measure provided."
                )
            observed_property_code = UKEACollector._extract_observed_property_from_measure_id(measure)
            observed_property = MAPPED_OBSERVED_PROPERTY_MEASURE_CODE.get(observed_property_code, None)
            if observed_property is None:
                    raise ValueError(
                        f"Invalid measure: {measure}. "
                        "Could not determine a valid observedProperty."
                    )

        if measure and (station or station_wiski_id):
            logger.warning(
                "Both measure and station id provided. "
                "Ignoring station id and using the exact measure provided.")
            station = None
            station_wiski_id = None

        if measure and collection:
            logger.warning(
                "Both measure and collection provided. "
                "Ignoring collection and using the exact measure provided"
            )
            collection = None

        if collection:
            period = COLLECTION_PERIOD_VALUES.get(collection, None)
            if period is None:
                raise ValueError(
                    "Invalid collection period."
                    "One of the following collection values must be passed: "
                    f"{', '.join(COLLECTION_PERIOD_VALUES.keys())}. "
                )

        if bbox:
            bounding_box_limits = UKEACollector._parse_bbox(bbox)
            if not bounding_box_limits:
                raise ValueError(
                    'Invalid bbox string. Must be a string of 4 comma-separated floats '
                    'in the form "min-long,min-lat,max-long,max-lat)". '
                    'For example, "2.0,51.1,3.3,52.7"',
                )
        else:
            bounding_box_limits = None

        min_date, max_date = UKEACollector._compute_date_range(min_date, max_date, days)

        params: dict[str, Any] = {
            "_limit": limit,
        }
        if collection:
            params["period"] = period
        if observed_property:
            params["observedProperty"] = observed_property
        if measure:
            params["measure"] = measure
        if station:
            params["station"] = station
        if station_wiski_id:
            params["station.wiskiID"] = station_wiski_id
        if bounding_box_limits:
            min_long, min_lat, max_long, max_lat = bounding_box_limits
            params["mineq-long"] = min_long
            params["mineq-lat"] = min_lat
            params["maxeq-long"] = max_long
            params["maxeq-lat"] = max_lat
        if min_date:
            params["mineq-date"] = min_date
        if max_date:
            params["maxeq-date"] = max_date
        params.update(kwargs)

        station_meta = None
        if station or station_wiski_id or measure:
            station_id = station or UKEACollector._extract_station_suid_from_measure_id(measure)
            if station_id:
                station_meta = self._fetch_station_metadata(
                    station=station_id,
                    station_wiski_id=station_wiski_id,
                )

        all_items: list[dict] = []
        all_items.append(params) # Metadata for use in normalise()
        if max_items:
            max_items += 1 # Increment max_items to account for prepended metadata
        offset = 0

        while True:
            params["_offset"] = offset
            try:
                data = self.client.get_json("data/readings.json", params=params)
            except Exception as exc:
                logger.error("UKEA fetch failed: %s", exc)
                return []

            page_items = data.get("items", [])
            if not page_items:
                break

            for item in page_items:
                if station_meta:
                    item["_station"] = station_meta

            all_items.extend(page_items)

            if max_items is not None and len(all_items) >= max_items:
                all_items = all_items[:max_items]
                logger.debug("UKEA max_items=%d reached — stopping pagination.", max_items)
                break

            offset += limit

        return all_items

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample] | Sequence[WaterLevelReading]:
        """Convert raw UK EA readings into AquaScope schema objects.

        Parameters
        ----------
        raw : list[dict]
            Raw payload returned by :meth:`fetch_raw`, including a prepended
            request-metadata dictionary.

        Returns
        -------
        Sequence[WaterQualitySample] | Sequence[WaterLevelReading]
            Normalised Water Quality or Water Level records.
            If the observed property requested does not map to a schema object,
            or there is no raw data available, an empty list is returned.
        """
        if not raw:
            return []

        raw_request_metadata = raw[0]
        observed_property = raw_request_metadata.get("observedProperty")
        if observed_property in {"waterFlow", "rainfall"}:
            return self._normalise_water_quality_samples(raw[1:], observed_property)
        elif observed_property in {"waterLevel", "groundwaterLevel"}:
            return self._normalise_water_level_readings(raw[1:], observed_property)
        else:
            return []

    def _normalise_water_quality_samples(self, raw: list[dict], observed_property: str) -> Sequence[WaterQualitySample]:
        """Normalise raw items into Water Quality samples.

        Parameters
        ----------
        raw : list[dict]
            Raw payload returned by :meth:`fetch_raw`, including a prepended
            request-metadata dictionary.
        observed_property: str
            The observed property collected within the raw data.

        Returns
        -------
        Sequence[WaterQualitySample]
            Normalised Water Quality samples.
        """
        samples: list[WaterQualitySample] = []
        for item in raw:
            try:
                station_suid, value, sample_datetime, remark = UKEACollector._extract_reading_data(item)
                unit = MAPPED_OBSERVED_PROPERTY_UNITS.get(observed_property, None)
                if not unit:
                    raise ValueError("Incomplete raw data reading")

                station_meta = item.get("_station", None)
                if station_meta:
                    station_name, river, location = UKEACollector._extract_water_quality_sample_metadata(station_meta)
                else:
                    station_name, river, location = None, None, None

                if observed_property == "rainfall":
                    river = "N/A"

                samples.append(
                    WaterQualitySample(
                        source=DataSource.UK_EA,
                        station_id=station_suid,
                        station_name=station_name,
                        location=location,
                        sample_datetime=sample_datetime,
                        parameter=observed_property,
                        value=float(value),
                        unit=unit,
                        river=river,
                        remark=remark,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.debug("Skipping UKEA item: %s", exc)

        return samples


    def _normalise_water_level_readings(self, raw: list[dict], observed_property: str) -> Sequence[WaterLevelReading]:
        """Normalise raw items into Water Level samples.

        Parameters
        ----------
        raw : list[dict]
            Raw payload returned by :meth:`fetch_raw`, including a prepended
            request-metadata dictionary.
        observed_property: str
            The observed property collected within the raw data.

        Returns
        -------
        Sequence[WaterLevelReading]
            Normalised Water Level readings.
        """
        samples: list[WaterLevelReading] = []
        for item in raw:
            try:
                station_suid, value, reading_datetime, remark = UKEACollector._extract_reading_data(item)
                unit = MAPPED_OBSERVED_PROPERTY_UNITS.get(observed_property, None)
                if not unit:
                    raise ValueError("Incomplete raw data reading")
                remark += f" Parameter: {observed_property}."

                station_meta = item.get("_station", None)
                if station_meta:
                    station_name, location = UKEACollector._extract_water_level_reading_metadata(station_meta)
                else:
                    station_name, location = None, None

                samples.append(
                    WaterLevelReading(
                        source=DataSource.UK_EA,
                        station_id=station_suid,
                        station_name=station_name,
                        location=location,
                        reading_datetime=reading_datetime,
                        water_level=float(value),
                        unit=unit,
                        remark=remark,
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                logger.info("Skipping UKEA item: %s", exc)

        return samples

    def _fetch_station_metadata(
        self,
        station: str | None = None,
        station_wiski_id: str | None = None,
    ) -> dict | None:
        """Fetch station metadata for a given set of station identifiers.

        Parameters
        ----------
        station : str | None
            Station GUID to look up.
        station_wiski_id : str | None
            WISKI station identifier to look up.

        Returns
        -------
        dict | None
            The first matching station metadata item, if available.
        """
        if not station and not station_wiski_id:
            return None

        params: dict[str, Any] = {}
        if station:
            params["stationGuid"] = station
        if station_wiski_id:
            params["wiskiID"] = station_wiski_id

        try:
            data = self.client.get_json("id/stations.json", params=params)
        except Exception as exc:
            logger.warning("Failed to fetch UKEA station metadata: %s", exc)
            return None

        items = data.get("items", []) or []
        if len(items) > 1:
            logger.warning(
                "Multiple UKEA stations found for station=%s, station_wiski_id=%s. First result used.",
                station,
                station_wiski_id,
            )
        return items[0] if items else None

    @staticmethod
    def _extract_station_suid_from_measure_id(
        measure: str | None
    ) -> str | None:
        """Extract the station SUID from a measure identifier.

        Parameters
        ----------
        measure : str | None
            Measure identifier from the UKEA API.

        Returns
        -------
        str | None
            The station SUID, which is the first 36 characters of any valid measure ID, if present.
        """
        if not measure:
            return None

        # Since the SUID is based on GUID style identifiers, the SUID is always the first 36 characters of the measure ID.
        return measure[:36]

    @staticmethod
    def _extract_observed_property_from_measure_id(
        measure: str | None
    ) -> str | None:
        """Extract the observed property from a measure identifier.

        Parameters
        ----------
        measure : str | None
            Measure identifier from the UK EA API.

        Returns
        -------
        str | None
            The observed-property token found after the station SUID, if present.
        """
        if not measure:
            return None

        # The observedProperty can be extracted from the measure's id. It appears after the station SUID.
        measure_without_suid = measure[37:]
        observed_property = measure_without_suid.split("-", 1)[0]
        return observed_property or None

    @staticmethod
    def _parse_bbox(value: str) -> tuple[float, float, float, float] | None:
        """Convert a bbox string into a 4-float tuple.

        Parameters
        ----------
        value : str
            Bounding box expressed as a comma-separated string in the form
            ``"min-long,min-lat,max-long,max-lat"``.

        Returns
        -------
        tuple[float, float, float, float] | None
            Parsed bounds as a 4-tuple when valid, otherwise ``None``.
        """
        if not isinstance(value, str):
            return None

        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 4:
            return None

        try:
            min_long, min_lat, max_long, max_lat = (float(part) for part in parts)
        except (TypeError, ValueError):
            return None

        return min_long, min_lat, max_long, max_lat

    @staticmethod
    def _compute_date_range(min_date: str | None, max_date: str | None, days: int | None) -> tuple[str, str]:
        """Compute a default or user-specified date range for the request.

        Parameters
        ----------
        min_date : str | None
            Inclusive lower bound for the requested date range.
        max_date : str | None
            Inclusive upper bound for the requested date range.
        days : int | None
            Relative lookback window in days when explicit dates are not supplied.

        Returns
        -------
        tuple[str, str]
            The computed start and end dates as ISO date strings.
        """
        if days is not None and days < 0:
            raise ValueError("Provided days must be a positive number")

        # If min_date and max_date not provided, set date range to the last `days` days (default 30)
        if min_date is None and max_date is None:
            end = date.today()
            start = end - timedelta(days=days if days is not None else 30)
            min_date = start.isoformat()
            max_date = end.isoformat()

        # If only min_date is provided, set max_date to `days` days (default 30) after min_date (or, if that exceeds today, set max_date to today)
        elif min_date is not None and max_date is None:
            end = date.fromisoformat(min_date) + timedelta(days=days if days is not None else 30)
            if end > date.today():
                end = date.today()
            max_date = end.isoformat()

        # If only max_date is provided, set min_date to `days` days (default 30) before max_date
        elif min_date is None and max_date is not None:
            start = date.fromisoformat(max_date) - timedelta(days=days if days is not None else 30)
            min_date = start.isoformat()

        elif min_date is not None and max_date is not None and days is not None:
            logger.warning(
                "Both min_date/max_date and days were provided. Ignoring days and using min_date/max_date range."
            )

        return min_date, max_date

    @staticmethod
    def _extract_reading_data(item: dict) -> tuple:
        """Extract common reading fields from a raw UK EA item.

        Parameters
        ----------
        item : dict
            Raw reading item returned by the UK EA API.

        Returns
        -------
        tuple
            A tuple containing the station SUID, value, sample datetime, and remark.
            Raises ValueError if a station SUID, value or datetime is not provided.
        """
        measure = item.get("measure", {})
        measure_id = measure.get("@id", None)
        measure_name = measure_id.rsplit("/", 1)[-1] if measure_id else None
        if measure_name:
            station_suid = UKEACollector._extract_station_suid_from_measure_id(measure_name)
        else:
            station_suid = None

        value = item.get("value", None)

        datetime_str = item.get("dateTime") or item.get("date")
        if datetime_str:
            sample_datetime = datetime.fromisoformat(datetime_str)
        else:
            sample_datetime = None

        remark = (f"Data Completeness: {item.get('completeness', 'N/A')}; "
                    f"Data Quality: {item.get('quality', 'N/A')}.")

        if not all(x is not None for x in (station_suid, value, sample_datetime)):
            raise ValueError("Incomplete raw data reading")

        return station_suid, value, sample_datetime, remark

    @staticmethod
    def _extract_water_quality_sample_metadata(station_meta: dict) -> tuple:
        """Extract station metadata suitable for a water-quality sample.

        Parameters
        ----------
        station_meta : dict
            Station metadata from the UKEA API.

        Returns
        -------
        tuple
            A tuple containing the station name, river name, and geolocation.
        """
        station_name = station_meta.get("label", None)
        river = station_meta.get("riverName", None)
        lat = station_meta.get("lat", None)
        long = station_meta.get("long", None)
        if lat is not None and long is not None:
            location = UKEACollector._build_location_from_lat_long(
                float(lat), float(long)
            )
        else:
            location = None

        return station_name, river, location

    @staticmethod
    def _extract_water_level_reading_metadata(station_meta: dict) -> tuple:
        """Extract station metadata suitable for a water-level reading.

        Parameters
        ----------
        station_meta : dict
            Station metadata from the UKEA API.

        Returns
        -------
        tuple
            A tuple containing the station name and geolocation.
        """
        station_name = station_meta.get("label", None)
        lat = station_meta.get("lat", None)
        long = station_meta.get("long", None)
        if lat is not None and long is not None:
            location = UKEACollector._build_location_from_lat_long(float(lat), float(long))
        else:
            location = None

        return station_name, location

    @staticmethod
    def _build_location_from_lat_long(lat: float, long: float) -> tuple | None:
        """Build a GeoLocation from latitude and longitude values.

        Parameters
        ----------
        lat : float
            Latitude value.
        long : float
            Longitude value.

        Returns
        -------
        tuple | None
            A ``GeoLocation`` object when the coordinates are valid, otherwise ``None``.
        """
        if lat is not None and long is not None:
            try:
                return GeoLocation(latitude=float(lat), longitude=float(long))
            except (ValueError, TypeError):
                return None

