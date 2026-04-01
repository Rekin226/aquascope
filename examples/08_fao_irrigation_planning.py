#!/usr/bin/env python3
"""AquaScope FAO irrigation planning example.

Demonstrates the high-level planning workflow using synthetic ET0 and
precipitation series. This keeps the example reproducible while exercising the
same planning API used by the CLI.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquascope.agri import plan_irrigation
from aquascope.agri.crop_water import DEFAULT_STAGE_LENGTHS
from aquascope.agri.water_balance import SoilProperties


def main() -> None:
    season_days = sum(DEFAULT_STAGE_LENGTHS["maize"].values()) + 10
    dates = pd.date_range("2026-04-01", periods=season_days, freq="D")

    eto = pd.Series(5.2, index=dates, name="eto_mm")
    precipitation = pd.Series(
        [2.0 if i % 7 in (4, 5) else 0.4 for i in range(season_days)],
        index=dates,
        name="precipitation_sum",
    )

    soil = SoilProperties(field_capacity=0.30, wilting_point=0.15, root_depth=1.0)
    plan = plan_irrigation(
        crop="maize",
        planting_date=date(2026, 4, 1),
        eto_series=eto,
        precip_series=precipitation,
        soil=soil,
        efficiency=0.7,
    )

    print("=" * 70)
    print("AquaScope FAO Irrigation Planning Example")
    print("=" * 70)
    print(f"Crop: {plan.crop}")
    print(f"Planting date: {plan.planting_date}")
    print(f"Season end: {plan.season_end_date}")
    print(f"Total ET0: {plan.total_eto_mm:.2f} mm")
    print(f"Total effective rain: {plan.total_effective_rain_mm:.2f} mm")
    print(f"Total ETc: {plan.total_etc_mm:.2f} mm")
    print(f"Net irrigation demand: {plan.total_net_irrigation_mm:.2f} mm")
    print(f"Applied irrigation: {plan.total_applied_irrigation_mm:.2f} mm")
    print(f"Irrigation trigger days: {plan.irrigation_trigger_days}")
    print("\nDaily schedule preview:")
    print(plan.schedule.head(10).to_string(index=False))


if __name__ == "__main__":
    main()