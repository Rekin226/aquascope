"""
Collector for Scotland's SEPA hydrometric time-series API.

SEPA publishes river flow and river-level observations through a Kisters
KiWIS service.

The collector works in two steps:

1. ``getTimeseriesList`` resolves the relevant time-series ID for a station.
2. ``getTimeseriesValues`` fetches observations for that time series.

SEPA parameter codes:
- ``Q`` = river flow / discharge
- ``SG`` = river level

API documentation:
https://timeseriesdoc.sepa.org.uk/
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    StreamflowReading,
    WaterLevelReading,
)

logger = logging.getLogger(__name__)


SEPA_BASE_URL = "https://timeseries.sepa.org.uk/KiWIS/KiWIS"

# SEPA uses these station parameter codes in KiWIS.
FLOW_PARAMETER = "Q"
LEVEL_PARAMETER = "SG"

VALID_PARAMETERS = {FLOW_PARAMETER, LEVEL_PARAMETER}

# Prefer units reported by the API, but keep these as safe fallbacks.
PARAMETER_UNITS: dict[str, str] = {
    FLOW_PARAMETER: "m3/s",
    LEVEL_PARAMETER: "m",
}


class ScotlandSepaCollector(BaseCollector):
    """Collect river flow and river-level observations from SEPA."""

    name: str = "scotland_sepa"

    # ------------------------------------------------------------------ #
    # fetch_raw
    # ------------------------------------------------------------------ #
    def fetch_raw(
        self,
        station_id: str | None = None,
        parameter: str = FLOW_PARAMETER,
        start_date: str | None = None,
        end_date: str | None = None,
        days: int | None = None,
        **kwargs: Any,
    ) -> list[dict]:
        """Fetch raw hydrometric observations for a SEPA station.

        Parameters
        ----------
        station_id : str
            SEPA station number.
        parameter : str
            SEPA station parameter. ``Q`` represents river flow and ``SG``
            represents river level.
        start_date : str | None
            Start date in ``YYYY-MM-DD`` format.
        end_date : str | None
            End date in ``YYYY-MM-DD`` format.
        days : int | None
            Convenience alternative to ``start_date`` for requesting the
            last N days. Defaults to 30 days when no date range is supplied.

        Returns
        -------
        list[dict]
            Raw observation rows merged with station and time-series metadata.
        """
        if not station_id:
            raise ValueError(
                "ScotlandSepaCollector.fetch_raw requires a station_id."
            )

        if parameter not in VALID_PARAMETERS:
            raise ValueError(
                "SEPA parameter must be 'Q' (flow) or 'SG' (river level)."
            )

        # Supply a useful default date window when one is not provided.
        if start_date is None:
            window_days = days if days is not None else 30
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=window_days)

            start_date = start.strftime("%Y-%m-%d")

            if end_date is None:
                end_date = end.strftime("%Y-%m-%d")

        ts_meta = self._resolve_timeseries(station_id, parameter)

        if ts_meta is None:
            logger.warning(
                "No SEPA timeseries found for station_no=%s, parameter=%s",
                station_id,
                parameter,
            )
            return []

        metadata, values = self._fetch_timeseries_values(
            ts_meta["ts_id"],
            start_date,
            end_date,
        )

        rows: list[dict] = []

        for timestamp, value, quality_code in values:
            rows.append(
                {
                    "station_no": ts_meta.get("station_no", station_id),
                    "station_name": ts_meta.get("station_name"),
                    "parameter": parameter,
                    "unit": metadata.get("unit") or ts_meta.get("unit"),
                    "latitude": (
                        metadata.get("latitude")
                        if metadata.get("latitude") is not None
                        else ts_meta.get("latitude")
                    ),
                    "longitude": (
                        metadata.get("longitude")
                        if metadata.get("longitude") is not None
                        else ts_meta.get("longitude")
                    ),
                    "timestamp": timestamp,
                    "value": value,
                    "quality_code": quality_code,
                }
            )

        return rows

    # ------------------------------------------------------------------ #
    # KiWIS helpers
    # ------------------------------------------------------------------ #
    def _resolve_timeseries(
        self,
        station_id: str,
        parameter: str,
    ) -> dict | None:
        """Resolve a SEPA station and parameter to its 15-minute time series.

        SEPA stations can publish several series for the same parameter,
        including annual maxima, peaks over threshold, gaugings, daily means,
        and regular 15-minute observations.

        AquaScope uses the 15-minute observation series so normal collection
        does not accidentally select an aggregated or event-based product.
        """
        params = {
            "service": "kisters",
            "type": "queryServices",
            "datasource": 0,
            "request": "getTimeseriesList",
            "station_no": station_id,
            "stationparameter_no": parameter,
            "ts_name": "15minute",
            "returnfields": (
                "station_no,station_name,station_latitude,"
                "station_longitude,stationparameter_name,"
                "ts_name,ts_id,ts_path"
            ),
            "format": "json",
        }

        try:
            data = self.client.get_json(SEPA_BASE_URL, params=params)
        except Exception:
            logger.warning(
                "SEPA getTimeseriesList request failed for station %s",
                station_id,
                exc_info=True,
            )
            return None

        row = self._first_data_row(data)

        if row is None:
            return None

        ts_id = row.get("ts_id")

        if not ts_id:
            return None

        return {
            "ts_id": ts_id,
            "station_no": row.get("station_no", station_id),
            "station_name": row.get("station_name"),
            "latitude": self._float_or_none(
                row.get("station_latitude")
            ),
            "longitude": self._float_or_none(
                row.get("station_longitude")
            ),
            "unit": PARAMETER_UNITS.get(parameter),
        }

    def _fetch_timeseries_values(
        self,
        ts_id: str,
        start_date: str,
        end_date: str | None,
    ) -> tuple[dict, list[tuple[str, str, str | None]]]:
        """Fetch observations and metadata for a SEPA time-series ID."""
        params: dict[str, Any] = {
            "service": "kisters",
            "type": "queryServices",
            "datasource": 0,
            "request": "getTimeseriesValues",
            "ts_id": ts_id,
            "from": start_date,
            "returnfields": "Timestamp,Value,Quality Code",
            "metadata": "true",
            "format": "json",
        }

        if end_date:
            params["to"] = end_date

        try:
            data = self.client.get_json(SEPA_BASE_URL, params=params)
        except Exception:
            logger.warning(
                "SEPA getTimeseriesValues request failed for ts_id %s",
                ts_id,
                exc_info=True,
            )
            return {}, []

        series = self._first_series(data)

        if series is None:
            return {}, []

        metadata = {
            "latitude": self._float_or_none(
                series.get("station_latitude")
            ),
            "longitude": self._float_or_none(
                series.get("station_longitude")
            ),
            "unit": series.get("ts_unitsymbol"),
        }

        columns = [
            column.strip()
            for column in series.get("columns", "").split(",")
        ]

        observations: list[tuple[str, str, str | None]] = []

        for record in series.get("data", []):
            row = dict(zip(columns, record))

            timestamp = row.get("Timestamp")
            value = row.get("Value")

            if timestamp is None or value is None:
                continue

            observations.append(
                (
                    timestamp,
                    value,
                    row.get("Quality Code"),
                )
            )

        return metadata, observations

    @staticmethod
    def _first_data_row(data: Any) -> dict | None:
        """Return the first row from a KiWIS list-style response."""
        if not isinstance(data, list) or not data:
            return None

        if data[0] == "No matches.":
            return None

        header, *rows = data

        if not rows or not isinstance(header, list):
            return None

        return dict(zip(header, rows[0]))

    @staticmethod
    def _first_series(data: Any) -> dict | None:
        """Return the first series from a KiWIS values response."""
        if isinstance(data, dict):
            return data

        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]

        return None

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        """Convert an API metadata value to a float when possible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------ #
    # normalise
    # ------------------------------------------------------------------ #
    def normalise(
        self,
        raw: list[dict],
    ) -> list[StreamflowReading | WaterLevelReading]:
        """Convert raw SEPA observations into AquaScope records."""
        if not raw:
            return []

        records: list[StreamflowReading | WaterLevelReading] = []
        skipped = 0

        for row in raw:
            try:
                value = row.get("value")

                if value is None or str(value).strip() in {
                    "",
                    "NaN",
                    "--",
                }:
                    continue

                value = float(value)

                timestamp = row.get("timestamp")

                if not timestamp:
                    continue

                reading_datetime = datetime.fromisoformat(
                    str(timestamp)
                ).replace(tzinfo=None)

                location = None
                latitude = row.get("latitude")
                longitude = row.get("longitude")

                if latitude is not None and longitude is not None:
                    location = GeoLocation(
                        latitude=float(latitude),
                        longitude=float(longitude),
                    )

                parameter = row.get("parameter", "")

                unit = (
                    row.get("unit")
                    or PARAMETER_UNITS.get(parameter, "")
                )

                quality_code = row.get("quality_code")

                remark = (
                    f"Quality Code: {quality_code}"
                    if quality_code is not None
                    else None
                )

                if parameter == FLOW_PARAMETER:
                    records.append(
                        StreamflowReading(
                            source=DataSource.SCOTLAND_SEPA,
                            station_id=str(
                                row.get("station_no", "unknown")
                            ),
                            station_name=row.get("station_name"),
                            location=location,
                            reading_datetime=reading_datetime,
                            discharge_cms=value,
                            source_type="in_situ",
                            unit=unit or "m3/s",
                            remark=remark,
                        )
                    )

                elif parameter == LEVEL_PARAMETER:
                    records.append(
                        WaterLevelReading(
                            source=DataSource.SCOTLAND_SEPA,
                            station_id=str(
                                row.get("station_no", "unknown")
                            ),
                            station_name=row.get("station_name"),
                            location=location,
                            reading_datetime=reading_datetime,
                            water_level=value,
                            unit=unit or "m",
                            remark=remark,
                        )
                    )

            except (TypeError, ValueError, KeyError) as exc:
                skipped += 1
                logger.debug(
                    "Skipping SEPA row: %s",
                    exc,
                )

        if skipped:
            logger.warning(
                "SEPA normalise: skipped %d of %d row(s) that failed to parse.",
                skipped,
                len(raw),
            )

        return records
