"""Tests for AQUASTAT benchmarking workflows."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from aquascope.agri import benchmark_aquastat
from aquascope.cli import main


def _sample_aquastat_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": "Egypt",
                "country_code": "EGY",
                "year": 2021,
                "variable": "Agricultural water withdrawal",
                "value": 60.0,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Egypt",
                "country_code": "EGY",
                "year": 2021,
                "variable": "Total water withdrawal",
                "value": 75.0,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Egypt",
                "country_code": "EGY",
                "year": 2021,
                "variable": "Total renewable water resources",
                "value": 58.0,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Egypt",
                "country_code": "EGY",
                "year": 2021,
                "variable": "Total area equipped for irrigation",
                "value": 3600.0,
                "unit": "1000 ha",
            },
            {
                "country": "Jordan",
                "country_code": "JOR",
                "year": 2020,
                "variable": "Agricultural water withdrawal",
                "value": 1.0,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Jordan",
                "country_code": "JOR",
                "year": 2020,
                "variable": "Total water withdrawal",
                "value": 1.3,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Jordan",
                "country_code": "JOR",
                "year": 2020,
                "variable": "Total renewable water resources",
                "value": 0.9,
                "unit": "10^9 m3/year",
            },
            {
                "country": "Jordan",
                "country_code": "JOR",
                "year": 2020,
                "variable": "Total area equipped for irrigation",
                "value": 85.0,
                "unit": "1000 ha",
            },
        ]
    )


class TestBenchmarkAquastat:
    def test_agricultural_withdrawal_share_pct(self) -> None:
        result = benchmark_aquastat(_sample_aquastat_df(), "agricultural_withdrawal_share_pct", year=2021)

        assert result.output_unit == "%"
        assert len(result.table) == 1
        assert result.table.iloc[0]["country"] == "Egypt"
        assert abs(result.table.iloc[0]["metric_value"] - 80.0) < 0.001

    def test_agricultural_withdrawal_per_irrigated_area_converts_units(self) -> None:
        result = benchmark_aquastat(
            _sample_aquastat_df(),
            "agricultural_withdrawal_per_irrigated_area",
            year=2021,
        )

        assert result.output_unit == "m3/ha"
        assert abs(result.table.iloc[0]["metric_value"] - 16666.6667) < 0.1

    def test_latest_only_defaults_to_latest_per_country(self) -> None:
        data = pd.concat(
            [
                _sample_aquastat_df(),
                pd.DataFrame(
                    [
                        {
                            "country": "Jordan",
                            "country_code": "JOR",
                            "year": 2021,
                            "variable": "Agricultural water withdrawal",
                            "value": 1.1,
                            "unit": "10^9 m3/year",
                        },
                        {
                            "country": "Jordan",
                            "country_code": "JOR",
                            "year": 2021,
                            "variable": "Total water withdrawal",
                            "value": 1.4,
                            "unit": "10^9 m3/year",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        result = benchmark_aquastat(data, "agricultural_withdrawal_share_pct")

        assert len(result.table) == 2
        jordan_row = result.table[result.table["country_code"] == "JOR"].iloc[0]
        assert jordan_row["year"] == 2021


class TestAgriBenchmarkCli:
    def test_cli_benchmark_writes_json(self, tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        aquastat_path = tmp_path / "aquastat.json"
        output_path = tmp_path / "benchmark.json"
        _sample_aquastat_df().to_json(aquastat_path, orient="records", indent=2)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "aquascope",
                "agri",
                "benchmark",
                "--aquastat-file",
                str(aquastat_path),
                "--metric",
                "agricultural_withdrawal_share_pct",
                "--year",
                "2021",
                "--output",
                str(output_path),
            ],
        )

        main()

        output = capsys.readouterr().out
        assert "AquaScope — Agriculture Benchmark" in output
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["metric_id"] == "agricultural_withdrawal_share_pct"
        assert data["table"]
