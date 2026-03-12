# Adding a New Data Source

This guide walks you through adding a new water data API as an AquaScope collector. Contributions from all countries and regions are welcome.

## Step 1: Add the DataSource Enum

Edit `aquascope/schemas/water_data.py` and add your source to the `DataSource` enum:

```python
class DataSource(str, Enum):
    # ... existing sources ...
    YOUR_SOURCE = "your_source"
```

## Step 2: Create the Collector Module

Create `aquascope/collectors/your_source.py`:

```python
"""
Collector for [Your Source Name].

API docs: https://...
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from aquascope.collectors.base import BaseCollector
from aquascope.schemas.water_data import (
    DataSource,
    GeoLocation,
    WaterQualitySample,
)
from aquascope.utils.http_client import CachedHTTPClient, RateLimiter

logger = logging.getLogger(__name__)

BASE_URL = "https://api.example.org/v1"


class YourSourceCollector(BaseCollector):
    """Collect water data from [Your Source]."""

    name = "your_source"

    def __init__(self, api_key: str = "", client: CachedHTTPClient | None = None):
        super().__init__(
            client or CachedHTTPClient(
                base_url=BASE_URL,
                rate_limiter=RateLimiter(max_calls=10, period_seconds=60),
                cache_ttl_seconds=3600,
            )
        )
        self.api_key = api_key

    def fetch_raw(self, **kwargs) -> list[dict]:
        """Fetch raw data from the API."""
        params = {"key": self.api_key}
        # Add your API-specific parameters
        data = self.client.get_json("/endpoint", params=params)
        return data.get("records", [])

    def normalise(self, raw: list[dict]) -> Sequence[WaterQualitySample]:
        """Transform raw API records into WaterQualitySample objects."""
        samples = []
        for row in raw:
            try:
                samples.append(
                    WaterQualitySample(
                        source=DataSource.YOUR_SOURCE,
                        station_id=row["station_id"],
                        station_name=row.get("name"),
                        sample_datetime=row["datetime"],
                        parameter=row["parameter"],
                        value=float(row["value"]),
                        unit=row.get("unit", ""),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.debug("Skipping row: %s", exc)
        return samples
```

## Step 3: Register the Collector

Edit `aquascope/collectors/__init__.py`:

```python
from aquascope.collectors.your_source import YourSourceCollector

__all__ = [
    # ... existing exports ...
    "YourSourceCollector",
]
```

## Step 4: Add CLI Support

Edit `aquascope/cli.py`, adding your source to:
1. The `collector_map` dictionary in `cmd_collect()`
2. The `--source` choices list
3. The `source_info` dictionary in `cmd_list_sources()`

## Step 5: Write Tests

Create `tests/test_collectors/test_your_source.py` with:
- Sample raw API response data (mocked)
- Test `normalise()` produces correct records
- Test edge cases (missing fields, invalid values)

## Step 6: Update Documentation

- Add your source to the table in `README.md`
- Update `docs/guides/architecture.md` if needed

## Guidelines

- **Use `CachedHTTPClient`** — It provides caching and rate limiting out of the box
- **Handle errors gracefully** — Skip invalid records with `logger.debug()`, don't crash
- **Include geographic data** — Set `GeoLocation` when lat/lon are available
- **Respect rate limits** — Configure `RateLimiter` based on the API's actual limits
- **Document the API** — Include the API docs URL and any key requirements
