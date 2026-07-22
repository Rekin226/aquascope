from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import DataSource, GeoLocation, WaterLevelReading


class IrelandOPWCollector(BaseCollector):
    """Collector for Ireland's Office of Public Works (waterlevel.ie) hydrometric network."""

    BASE_URL = "https://waterlevel.ie"
    STATIONS_URL = f"{BASE_URL}/geojson/latest/"

    def __init__(self):
        super().__init__()

    def _get_station_data_url(self, station_ref: str) -> str:
        """Construct the URL for a specific station's time series JSON."""
        return f"{self.BASE_URL}/data/month/{station_ref}_0001.json"

    async def fetch_stations(self) -> list[dict[str, Any]]:
        """Fetch the GeoJSON list of all active OPW stations."""
        content = await self.http_client.get(self.STATIONS_URL)
        import json
        data = json.loads(content)
        return data.get("features", [])

    async def fetch_raw(
        self,
        station_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Fetch raw JSON time series data for a specific station."""
        url = self._get_station_data_url(station_id)

        try:
            content = await self.http_client.get(url)
            import json
            data = json.loads(content)
        except Exception as e:
            self.logger.warning(f"Failed to fetch data for station {station_id}: {e}")
            return

        for record in data:
            yield record

    def normalise(
        self, raw_record: dict[str, Any], station_id: str, metadata: dict[str, Any] | None = None
    ) -> WaterLevelReading:
        """Normalize a raw waterlevel.ie record into the unified schema."""
        dt = datetime.strptime(raw_record["datetime"], "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)

        lat = None
        lon = None
        name = None
        if metadata:
            coords = metadata.get("geometry", {}).get("coordinates")
            if coords and len(coords) >= 2:
                lon, lat = coords[0], coords[1]
            name = metadata.get("properties", {}).get("station_name")

        location = GeoLocation(latitude=lat, longitude=lon) if lat is not None and lon is not None else None

        return WaterLevelReading(
            source=DataSource.IRELAND_OPW,
            station_id=station_id,
            reading_datetime=dt,
            water_level=float(raw_record["value"]),
            location=location,
            station_name=name,
        )
