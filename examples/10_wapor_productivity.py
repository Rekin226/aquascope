#!/usr/bin/env python3
"""AquaScope WaPOR productivity example."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquascope.agri import estimate_wapor_productivity


def _frame(cube_code: str, values: list[float], unit: str) -> pd.DataFrame:
    dates = pd.date_range("2026-04-01", periods=len(values), freq="MS")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "cube_code": cube_code,
            "value": values,
            "unit": unit,
            "bbox": ["30.5,29.8,31.1,30.2"] * len(values),
        }
    )


def _aquastat_context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Area": "Egypt",
                "Area Code": "EGY",
                "Year": 2023,
                "Element": "Agricultural water withdrawal",
                "Value": 55.0,
                "Unit": "10^9 m3/year",
            },
            {
                "Area": "Egypt",
                "Area Code": "EGY",
                "Year": 2023,
                "Element": "Total water withdrawal",
                "Value": 68.0,
                "Unit": "10^9 m3/year",
            },
            {
                "Area": "Egypt",
                "Area Code": "EGY",
                "Year": 2023,
                "Element": "Total area equipped for irrigation",
                "Value": 4.4,
                "Unit": "1000 ha",
            },
            {
                "Area": "Morocco",
                "Area Code": "MAR",
                "Year": 2023,
                "Element": "Agricultural water withdrawal",
                "Value": 9.5,
                "Unit": "10^9 m3/year",
            },
            {
                "Area": "Morocco",
                "Area Code": "MAR",
                "Year": 2023,
                "Element": "Total water withdrawal",
                "Value": 12.0,
                "Unit": "10^9 m3/year",
            },
            {
                "Area": "Morocco",
                "Area Code": "MAR",
                "Year": 2023,
                "Element": "Total area equipped for irrigation",
                "Value": 1.6,
                "Unit": "1000 ha",
            },
        ]
    )


def main() -> None:
    aeti = _frame("AETI", [95.0, 118.0, 132.0], "mm/month")
    npp = _frame("NPP", [155.0, 225.0, 244.0], "g/m2/month")
    ret = _frame("RET", [120.0, 140.0, 150.0], "mm/month")

    result = estimate_wapor_productivity(
        metric_id="biomass_water_productivity",
        aeti_df=aeti,
        npp_df=npp,
        ret_df=ret,
        aquastat_df=_aquastat_context(),
        aquastat_countries=["EGY", "MAR"],
    )

    print("=" * 70)
    print("AquaScope WaPOR Productivity Example")
    print("=" * 70)
    print(result.summary)
    print()
    print(result.table.to_string(index=False))

    if result.aquastat_context:
        print()
        print("AQUASTAT context")
        for context in result.aquastat_context:
            print()
            print(context.summary)
            print(context.table.to_string(index=False))


if __name__ == "__main__":
    main()