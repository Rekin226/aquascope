#!/usr/bin/env python3
"""
AquaScope Quick Start Example
==============================
Demonstrates how to:
1. Collect water quality data from Taiwan MOENV
2. Collect SDG 6 indicators
3. Run the AI methodology recommender
"""

from aquascope.ai_engine.recommender import DatasetProfile, recommend
from aquascope.collectors import SDG6Collector, TaiwanMOENVCollector
from aquascope.utils.storage import save_records

# ── Step 1: Collect Taiwan river water quality data ──────────────────
print("=" * 60)
print("Step 1: Collecting Taiwan MOENV river water quality data …")
print("=" * 60)

moenv = TaiwanMOENVCollector(api_key="")  # works without key (limited rate)
try:
    samples = moenv.collect(limit=100)  # small batch for demo
    print(f"  → Collected {len(samples)} water quality samples")
    if samples:
        path = save_records(samples, prefix="taiwan_moenv_demo")
        print(f"  → Saved to {path}")
        # Show first record
        print(f"  → Example: {samples[0].station_name} | {samples[0].parameter} = {samples[0].value} {samples[0].unit}")
except Exception as e:
    print(f"  → Collection skipped (API may be unavailable): {e}")
    samples = []


# ── Step 2: Collect SDG 6 data for Taiwan and Burkina Faso ──────────
print("\n" + "=" * 60)
print("Step 2: Collecting UN SDG 6 indicators …")
print("=" * 60)

sdg6 = SDG6Collector()
try:
    indicators = sdg6.collect(
        indicator_codes=["6.3.1", "6.4.2"],
        country_codes="TWN,BFA",
        year_range="2015:2023",
    )
    print(f"  → Collected {len(indicators)} SDG 6 records")
    if indicators:
        path = save_records(indicators, prefix="sdg6_demo")
        print(f"  → Saved to {path}")
except Exception as e:
    print(f"  → Collection skipped: {e}")


# ── Step 3: Get methodology recommendations ─────────────────────────
print("\n" + "=" * 60)
print("Step 3: AI Methodology Recommendations")
print("=" * 60)

profile = DatasetProfile(
    parameters=["DO", "BOD5", "COD", "NH3-N", "SS", "pH"],
    n_records=500,
    n_stations=15,
    time_span_years=8.0,
    geographic_scope="Taiwan — Tamsui River Basin",
    research_goal="Assess long-term water quality trends and identify pollution sources",
    keywords=["trend", "monitoring", "source apportionment"],
)

recs = recommend(profile, top_k=5)

for i, rec in enumerate(recs, 1):
    m = rec.methodology
    print(f"\n  {i}. {m.name}  (score: {rec.score})")
    print(f"     Category   : {m.category}")
    print(f"     Complexity : {m.complexity}")
    print(f"     Rationale  : {rec.rationale}")

print("\n✓ Quick start complete!")
