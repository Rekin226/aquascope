# Adding a New Data Source

This guide walks you through adding a new water data API as an AquaScope collector. Contributions from all countries and regions are welcome.

## Step 0: Verify the source terms

Different data sources offer different levels of permission for data collection, storage, and redistribution, with limitations or even restrictions on automated collection, data caching, and redistributing data from third-party sources like our own. The first step to adding a new collector is to read the publisher's terms and conditions and clarify what functions of AquaScope can be performed on the data.

Here's a checklist for steps to take before adding a data source:

- [ ] Find the first-party terms or licence page covering the dataset or API you are calling.
- [ ] Record the licence name or identifier exactly as the publisher writes it.
- [ ] If the provider requires an exact attribution statement, note it down verbatim. If the provider doesn't provide an exact statement but requires attribution, write a statement that is accurate and concise. If the provider doesn't require attribution, write "none stated".
- [ ] Record the terms URL and the date you checked them in the body of a new issue or pull request.

This information is used in Step 3 to populate the `license`, `attribution`, and `redistributable` fields in the registry entry for the new collector.

If both collection and redistribution are permitted according to the terms, set `redistributable=True`.

If collection is permitted but redistribution is not permitted, or the terms for redistribution are unclear, document the restriction and leave `redistributable=False`; the source can still be added and a collector can be built, but it is not mirrored into the Archive.

If collection is not permitted, or if the terms are unclear, do not add the source. Instead, open an issue to discuss with the maintainers.

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

## Step 3: Register it once, in `aquascope/registry.py`

The registry is the single source of truth. One entry there is what makes the
source show up in `aquascope collect --source`, `aquascope list-sources`,
`aquascope stations`, the dashboard Collect page, `aquascope.collect()`, the
harvest job and the MCP server. Nothing else needs a per-source edit.

Two things to add:

1. Export the class from `aquascope/collectors/__init__.py` (add it to the
   import list and `__all__`).
2. Add a `SOURCES["your_source"]` entry and a `build_collector()` factory
   line in `aquascope/registry.py`:

```python
"your_source": _s(
    key="your_source", label="Your Agency", region="Your Country",
    description="What it serves, in one line",
    agency="Your Agency full name", country="XXX",          # ISO 3166-1 alpha-3
    homepage="https://...",
    variables=("discharge", "water_level"),                # from schemas/station.py VARIABLES
    supports_bbox=False, supports_station_lookup=False,   # flip when the collector supports them
    output_model="StreamflowReading",
    license="unknown", redistributable=False,              # see Step 0 for how to decide
    attribution="Your Agency (licence name)",
),
```

`redistributable=True` requires a real licence id in `license` (the drift-guard
test enforces it), and it is what lets the archive mirror the observations.
When you are not sure, leave it `False`; the source still works everywhere.

### Optional: a station catalog

If the API can list its stations, override `stations()` on the collector and
set `supports_station_lookup=True` in the registry entry:

```python
from aquascope.schemas.station import Station, in_bbox

def stations(self, *, bbox=None, variable=None, max_items=None) -> list[Station]:
    rows = self.client.get_json("stations")
    out = []
    for r in rows:
        if not in_bbox(r["lat"], r["lon"], bbox):
            continue
        out.append(Station(source="your_source", station_id=r["id"], name=r.get("name"),
                           latitude=r["lat"], longitude=r["lon"], variables=("discharge",),
                           url=f"https://.../{r['id']}"))
        if max_items is not None and len(out) >= max_items:
            break
    return out
```

`aquascope/collectors/uk_ea.py`, `usgs.py`, `france_hubeau.py`, `pegelonline.py`,
`ireland_opw.py` and `taiwan_cwa.py` are the reference implementations. A test
in `tests/test_registry.py` asserts that the registry flag matches whether the
method is really overridden.

## Step 4: Dashboard form (only if the source needs parameters)

The Collect page lists every registered source automatically. If yours needs
user inputs (dates, a station id, a mode), add a branch to `_source_form()` in
`aquascope/dashboard/views/collect.py` that fills `ctor` (constructor kwargs)
or `fetch` (`collect()` kwargs). A source with no parameters needs nothing.

## Step 5: Write Tests

Create `tests/test_collectors/test_your_source.py` with:
- Sample raw API response data (mocked)
- Test `normalise()` produces correct records
- Test edge cases (missing fields, invalid values)

## Step 6: Update Documentation

- Add a row to the table in `docs/data_sources.md`, with your registry key in
  the `--source` column. `tests/test_docs_counts.py` asserts the table lists
  exactly the ids in `aquascope.registry.SOURCES`, so a collector without a row
  fails CI
- Bump the source count wherever it is written out: README (six places),
  `docs/index.md`, `docs/features.md`, `docs/i18n/README.fr.md` and the
  `CITATION.cff` abstract. The same test checks each of them, and the failure
  message names the file
- Add your source to the regional bullets in the README ("Data sources at a
  glance") and to `docs/features.md`
- Update `docs/guides/architecture.md` if needed

## Guidelines

- **Use `CachedHTTPClient`** — It provides caching and rate limiting out of the box. Exception: one-time bulk-download sources (a single static archive rather than a paginated API, e.g. `grdc.py`, `camels_cl.py`) skip it and stream the file into `data/cache/` instead — see those collectors for the pattern
- **Handle errors gracefully** — Skip invalid records with `logger.debug()`, don't crash
- **Include geographic data** — Set `GeoLocation` when lat/lon are available
- **Respect rate limits** — Configure `RateLimiter` based on the API's actual limits (not applicable to bulk downloads)
- **Document the API** — Include the API docs URL and any key requirements
