#!/usr/bin/env python3
"""Example 02 — Run water quality analysis.

Shows the full quality assessment workflow:

1. Load previously collected data
2. Run EDA (exploratory data analysis)
3. Run quality assessment
4. Check WHO guideline exceedances
5. Visualise results
"""

import json
from pathlib import Path

import pandas as pd

# ── Step 1: Load data ────────────────────────────────────────────────────

DATA_DIR = Path("data/raw")
files = sorted(DATA_DIR.glob("*.json"))

if not files:
    print("No data files found. Run example 01 first, or:")
    print("  aquascope collect --source taiwan_moenv --format json")
    raise SystemExit(1)

data_file = files[0]
print(f"▸ Loading {data_file} …")
with open(data_file) as f:
    raw = json.load(f)
print(f"  Loaded {len(raw)} records")

# ── Step 2: EDA ──────────────────────────────────────────────────────────

print("\n▸ Running Exploratory Data Analysis …")
from aquascope.analysis.eda import run_eda

report = run_eda(str(data_file))
print(f"  Records:    {report.n_records}")
print(f"  Stations:   {report.n_stations}")
print(f"  Parameters: {report.n_parameters}")
print(f"  Date range: {report.date_range}")
print(f"  Completeness: {report.completeness_pct:.1f}%")

for p in report.parameters[:5]:
    print(f"    {p.name:>12s}: mean={p.mean:.2f}, std={p.std:.2f}, outliers={p.outlier_count}")

# ── Step 3: Quality assessment ───────────────────────────────────────────

print("\n▸ Running Quality Assessment …")
from aquascope.analysis.quality import assess_quality

qr = assess_quality(str(data_file))
print(f"  Duplicates:    {qr.n_duplicates}")
print(f"  Completeness:  {qr.completeness_pct:.1f}%")
print(f"  Null counts:   {qr.null_counts}")
print(f"  Recommendations: {len(qr.recommended_steps)}")
for step in qr.recommended_steps:
    print(f"    • {step}")

# ── Step 4: WHO guidelines (if quality data) ─────────────────────────────

print("\n▸ Checking WHO guideline exceedances …")
try:
    from aquascope.challenges.quality import WaterQualityChallenge

    # Build a long-format DataFrame from the records
    rows = [{"parameter": r["parameter"], "value": r["value"]} for r in raw if "parameter" in r and "value" in r]
    if rows:
        df = pd.DataFrame(rows)
        wq = WaterQualityChallenge(df)
        who_df = wq.check_who_guidelines()
        if not who_df.empty:
            print(who_df.to_string(index=False))
        else:
            print("  No WHO-relevant parameters found")
    else:
        print("  No parameter/value pairs found in data")
except Exception as e:
    print(f"  Skipped: {e}")

# ── Step 5: Visualise ────────────────────────────────────────────────────

print("\n▸ Generating EDA summary plot …")
try:
    from aquascope.viz import plot_eda_summary

    plot_eda_summary(report, save_path="data/eda_summary.png")
    print("  ✓ Saved to data/eda_summary.png")
except ImportError:
    print("  Install viz deps: pip install aquascope[viz]")

print("\nDone!")
