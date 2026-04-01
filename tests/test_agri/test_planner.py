"""Tests for the high-level agriculture planning workflow."""

from __future__ import annotations

import json
import sys
from datetime import date

import pandas as pd
import pytest

from aquascope.agri.crop_water import DEFAULT_STAGE_LENGTHS
from aquascope.agri.planner import (
    fetch_openmeteo_plan_inputs,
    plan_irrigation,
    series_from_dataframe,
)
from aquascope.agri.water_balance import SoilProperties
from aquascope.cli import main
from aquascope.collectors.openmeteo import OpenMeteoCollector


def _make_daily_series(n_days: int, value: float, start: str = "2024-04-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=n_days, freq="D")
    return pd.Series([value] * n_days, index=idx)


class TestSeriesFromDataFrame:
    def test_extracts_long_form_parameter_series(self) -> None:
        df = pd.DataFrame(
            {
                "sample_datetime": ["2024-04-01", "2024-04-01", "2024-04-02"],
                "parameter": ["et0_fao_evapotranspiration", "precipitation_sum", "et0_fao_evapotranspiration"],
                "value": [5.0, 1.5, 6.0],
            }
        )

        eto = series_from_dataframe(
            df,
            value_columns=("eto_mm", "value", "et0_fao_evapotranspiration"),
            parameter="et0_fao_evapotranspiration",
        )

        assert len(eto) == 2
        assert eto.iloc[0] == 5.0
        assert eto.iloc[1] == 6.0


class TestFetchOpenMeteoPlanInputs:
    def test_fetches_et0_and_precipitation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_fetch_raw(self, **kwargs):  # noqa: ANN001, ANN202
            return {
                "daily": {
                    "time": ["2024-04-01", "2024-04-02"],
                    "et0_fao_evapotranspiration": [4.2, 4.8],
                    "precipitation_sum": [0.0, 3.5],
                }
            }

        monkeypatch.setattr(OpenMeteoCollector, "fetch_raw", fake_fetch_raw)

        eto, precip = fetch_openmeteo_plan_inputs(25.0, 121.0, "2024-04-01", "2024-04-02")

        assert list(eto.round(1)) == [4.2, 4.8]
        assert list(precip.round(1)) == [0.0, 3.5]


class TestPlanIrrigation:
    def test_returns_schedule_and_balance(self) -> None:
        n_days = sum(DEFAULT_STAGE_LENGTHS["maize"].values()) + 10
        eto = _make_daily_series(n_days, 5.5)
        precip = _make_daily_series(n_days, 1.0)
        soil = SoilProperties(field_capacity=0.30, wilting_point=0.15, root_depth=1.0)

        plan = plan_irrigation(
            crop="maize",
            planting_date=date(2024, 4, 1),
            eto_series=eto,
            precip_series=precip,
            soil=soil,
            efficiency=0.7,
        )

        assert len(plan.schedule) == sum(DEFAULT_STAGE_LENGTHS["maize"].values())
        assert len(plan.balance) == len(plan.schedule)
        assert plan.total_eto_mm > 0
        assert plan.total_etc_mm > 0
        assert plan.total_gross_irrigation_mm >= plan.total_net_irrigation_mm
        assert plan.total_applied_irrigation_mm > 0


class TestAgriCli:
    def test_collect_help_lists_fao_sources(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(sys, "argv", ["aquascope", "collect", "--help"])

        with pytest.raises(SystemExit) as excinfo:
            main()

        assert excinfo.value.code == 0
        help_text = capsys.readouterr().out
        assert "aquastat" in help_text
        assert "wapor" in help_text

    def test_agri_plan_cli_from_files(self, tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        n_days = sum(DEFAULT_STAGE_LENGTHS["maize"].values()) + 10
        dates = pd.date_range("2024-04-01", periods=n_days, freq="D")

        eto_path = tmp_path / "eto.csv"
        precip_path = tmp_path / "precip.csv"
        output_path = tmp_path / "plan.json"

        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "eto_mm": [5.0] * n_days}).to_csv(eto_path, index=False)
        pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "precipitation_sum": [1.0] * n_days}).to_csv(
            precip_path,
            index=False,
        )

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "aquascope",
                "agri",
                "plan",
                "--crop",
                "maize",
                "--planting-date",
                "2024-04-01",
                "--eto-file",
                str(eto_path),
                "--precip-file",
                str(precip_path),
                "--output",
                str(output_path),
            ],
        )

        main()

        output = capsys.readouterr().out
        assert "AquaScope — Irrigation Plan" in output
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert data["crop"] == "maize"
        assert data["schedule"]
        assert data["balance"]
