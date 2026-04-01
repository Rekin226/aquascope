#!/usr/bin/env python3
"""Example 03 — AI methodology recommendations.

Demonstrates how AquaScope's AI engine analyses your dataset and
recommends the most suitable research methodologies.

Two modes:
1. **From file** — Loads collected data, builds a profile, and recommends.
2. **Manual profile** — Specify parameters directly (no data file needed).
"""

# ── Mode 1: Recommendations from a data file ────────────────────────────

print("▸ Mode 1: Recommendations from dataset profile\n")

from aquascope.ai_engine.recommender import MethodologyRecommender
from aquascope.analysis.eda import DatasetProfile

profile = DatasetProfile(
    n_records=5000,
    n_stations=12,
    n_parameters=8,
    time_span_years=5.0,
    parameters=["pH", "DO", "BOD5", "NH3-N", "COD", "turbidity", "conductivity", "water_temperature"],
    geographic_scope="Taiwan",
    has_temporal_data=True,
    has_spatial_data=True,
    goal="Assess long-term water quality trends and identify pollution hotspots",
    keywords=["trend analysis", "spatial patterns", "pollution"],
)

recommender = MethodologyRecommender()
results = recommender.recommend(profile, top_k=5)

for i, rec in enumerate(results, 1):
    print(f"  {i}. {rec.methodology.name} (score: {rec.score:.2f})")
    print(f"     Category:  {rec.methodology.category}")
    print(f"     Why:       {rec.reasoning}")
    print()

# ── Mode 2: Quick recommendation from the top-level API ─────────────────

print("▸ Mode 2: Quick recommendation via top-level API\n")

from aquascope import recommend

recs = recommend(goal="flood risk assessment using historical discharge data", top_k=3)
for i, rec in enumerate(recs, 1):
    print(f"  {i}. {rec.methodology.name} — {rec.reasoning}")

print("\nDone!  Use `aquascope recommend --from-file <path>` from the CLI.")
