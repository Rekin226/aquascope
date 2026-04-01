#!/usr/bin/env python3
"""AquaScope AQUASTAT benchmarking example."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquascope.agri import benchmark_aquastat


def main() -> None:
    data = pd.DataFrame(
        [
            {
                "country": "Egypt",
                "country_code": "EGY",
                "year": 2023,
                "variable": "Agricultural water withdrawal",
                "value": 61.0,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Egypt",
                "country_code": "EGY",
                "year": 2023,
                "variable": "Total water withdrawal",
                "value": 76.0,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Egypt",
                "country_code": "EGY",
                "year": 2023,
                "variable": "Total area equipped for irrigation",
                "value": 3650.0,
                "unit": "1000 ha",
            },
            {
                "country": "Morocco",
                "country_code": "MAR",
                "year": 2023,
                "variable": "Agricultural water withdrawal",
                "value": 11.0,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Morocco",
                "country_code": "MAR",
                "year": 2023,
                "variable": "Total water withdrawal",
                "value": 13.5,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Morocco",
                "country_code": "MAR",
                "year": 2023,
                "variable": "Total area equipped for irrigation",
                "value": 1680.0,
                "unit": "1000 ha",
            },
            {
                "country": "Jordan",
                "country_code": "JOR",
                "year": 2023,
                "variable": "Agricultural water withdrawal",
                "value": 1.1,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Jordan",
                "country_code": "JOR",
                "year": 2023,
                "variable": "Total water withdrawal",
                "value": 1.4,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Jordan",
                "country_code": "JOR",
                "year": 2023,
                "variable": "Total area equipped for irrigation",
                "value": 88.0,
                "unit": "1000 ha",
            },
        ]
    )

    result = benchmark_aquastat(data, "agricultural_withdrawal_share_pct")

    print("=" * 70)
    print("AquaScope AQUASTAT Benchmark Example")
    print("=" * 70)
    print(result.summary)
    print()
    print(result.table.to_string(index=False))


if __name__ == "__main__":
    main()