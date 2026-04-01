#!/usr/bin/env python3
"""Example 01 — Collect data from a single source.

Demonstrates how to use AquaScope collectors to fetch and normalise
water data from a single API.  Three examples are shown:

1. Taiwan EPA water quality monitoring
2. USGS real-time river gauges
3. Open-Meteo historical weather

Each collector follows the same pattern: ``fetch_raw()`` → ``normalise()``.
"""

from aquascope.collectors import OpenMeteoCollector, TaiwanMoenvCollector, USGSCollector
from aquascope.utils.storage import save_records

# ── 1. Taiwan EPA water quality ──────────────────────────────────────────

print("▸ Fetching Taiwan MOENV water quality data …")
tw = TaiwanMoenvCollector()
raw_tw = tw.fetch_raw()
records_tw = tw.normalise(raw_tw)
print(f"  Got {len(records_tw)} records")
if records_tw:
    r = records_tw[0]
    print(f"  Sample: {r.station_name} | {r.parameter}={r.value} {r.unit}")

# ── 2. USGS real-time discharge ──────────────────────────────────────────

print("\n▸ Fetching USGS discharge data (last 7 days) …")
usgs = USGSCollector()
raw_usgs = usgs.fetch_raw(days=7)
records_usgs = usgs.normalise(raw_usgs)
print(f"  Got {len(records_usgs)} records")

# ── 3. Open-Meteo historical weather ────────────────────────────────────

print("\n▸ Fetching Open-Meteo weather for Taipei (last 30 days) …")
meteo = OpenMeteoCollector()
raw_meteo = meteo.fetch_raw(
    lat=25.03, lon=121.57,
    mode="weather",
    start_date="2024-01-01",
    end_date="2024-01-31",
)
records_meteo = meteo.normalise(raw_meteo)
print(f"  Got {len(records_meteo)} records")

# ── Save all to files ────────────────────────────────────────────────────

for label, records in [("taiwan", records_tw), ("usgs", records_usgs), ("openmeteo", records_meteo)]:
    if records:
        path = save_records(records, prefix=f"example_{label}", fmt="json")
        print(f"\n  ✓ Saved {label} → {path}")

print("\nDone!  Run `aquascope eda --file <path>` on any output file.")
