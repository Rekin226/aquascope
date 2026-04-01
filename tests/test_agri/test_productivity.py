"""Tests for WaPOR productivity workflows."""

from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from aquascope.agri import estimate_wapor_productivity
from aquascope.cli import main
from aquascope.collectors.wapor import WaPORCollector
from aquascope.schemas.agriculture import WaPORObservation


def _wapor_frame(cube_code: str, values: list[float], unit: str, *, start: str = "2024-04-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "cube_code": cube_code,
            "value": values,
            "unit": unit,
            "bbox": ["30.5,29.8,31.1,30.2"] * len(values),
        }
    )


def _aquastat_frame() -> pd.DataFrame:
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


class TestWaPORCollector:
    def test_normalise_valid_record(self) -> None:
        collector = WaPORCollector()
        raw = [
            {
                "date": "2024-04-01",
                "value": 4.5,
                "unit": "mm/day",
                "_cube_code": "RET",
                "_cube_label": "Reference EvapoTranspiration",
                "_bbox": (30.5, 29.8, 31.1, 30.2),
            }
        ]

        records = collector.normalise(raw)
        assert len(records) == 1
        assert isinstance(records[0], WaPORObservation)
        assert records[0].cube_code == "RET"
        assert records[0].value == 4.5
        assert records[0].unit == "mm/day"


class TestEstimateWaPORProductivity:
    def test_biomass_water_productivity(self) -> None:
        result = estimate_wapor_productivity(
            metric_id="biomass_water_productivity",
            aeti_df=_wapor_frame("AETI", [100.0, 120.0], "mm/day"),
            npp_df=_wapor_frame("NPP", [150.0, 210.0], "g/m2/day"),
        )

        assert result.output_unit == "kg/m3"
        assert abs(result.aggregate_value - 1.6364) < 0.001
        assert abs(result.table.iloc[0]["metric_value"] - 1.5) < 0.001

    def test_relative_evapotranspiration_pct(self) -> None:
        result = estimate_wapor_productivity(
            metric_id="relative_evapotranspiration_pct",
            aeti_df=_wapor_frame("AETI", [80.0, 100.0], "mm/day"),
            ret_df=_wapor_frame("RET", [100.0, 125.0], "mm/day"),
        )

        assert result.output_unit == "%"
        assert abs(result.aggregate_value - 80.0) < 0.001
        assert abs(result.table.iloc[1]["metric_value"] - 80.0) < 0.001

    def test_attaches_optional_aquastat_context(self) -> None:
        result = estimate_wapor_productivity(
            metric_id="biomass_water_productivity",
            aeti_df=_wapor_frame("AETI", [100.0, 120.0], "mm/day"),
            npp_df=_wapor_frame("NPP", [150.0, 210.0], "g/m2/day"),
            aquastat_df=_aquastat_frame(),
            aquastat_countries=["EGY", "MAR"],
        )

        assert result.aquastat_context is not None
        assert [context.metric_id for context in result.aquastat_context] == [
            "agricultural_withdrawal_share_pct",
            "agricultural_withdrawal_per_irrigated_area",
        ]
        assert "AQUASTAT benchmark table" in result.summary
        serialized = result.to_dict()
        assert len(serialized["aquastat_context"]) == 2
        assert serialized["aquastat_context"][0]["table"][0]["country_code"] == "EGY"

    def test_requires_matching_inputs(self) -> None:
        with pytest.raises(ValueError, match="requires WaPOR files"):
            estimate_wapor_productivity(
                metric_id="biomass_water_productivity",
                aeti_df=_wapor_frame("AETI", [100.0], "mm/day"),
            )


class TestAgriProductivityCli:
    def test_cli_productivity_writes_json(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        aeti_path = tmp_path / "aeti.json"
        npp_path = tmp_path / "npp.json"
        aquastat_path = tmp_path / "aquastat.json"
        output_path = tmp_path / "productivity.json"

        _wapor_frame("AETI", [100.0, 120.0], "mm/day").to_json(aeti_path, orient="records", indent=2)
        _wapor_frame("NPP", [150.0, 210.0], "g/m2/day").to_json(npp_path, orient="records", indent=2)
        _aquastat_frame().to_json(aquastat_path, orient="records", indent=2)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "aquascope",
                "agri",
                "productivity",
                "--metric",
                "biomass_water_productivity",
                "--aeti-file",
                str(aeti_path),
                "--npp-file",
                str(npp_path),
                "--aquastat-file",
                str(aquastat_path),
                "--aquastat-countries",
                "EGY,MAR",
                "--output",
                str(output_path),
            ],
        )

        main()

        output = capsys.readouterr().out
        assert "AquaScope — WaPOR Productivity" in output
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["metric_id"] == "biomass_water_productivity"
        assert data["aggregate_value"] > 0
        assert len(data["aquastat_context"]) == 2
