"""Tests for the groundwater CLI command, focusing on p-value formatting."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from aquascope import cli
from aquascope.groundwater.wells import WellTrendResult


def test_groundwater_trend_cli_formats_tiny_p_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """Verify that a tiny p-value (< 0.001) in well trend CLI outputs '< 0.001' rather than '0.0000'."""
    csv_file = tmp_path / "well_levels.csv"
    csv_file.write_text("datetime,water_level\n2020-01-01,10.5\n2020-01-02,10.4\n2020-01-03,10.3\n")

    fake_result = WellTrendResult(
        trend="decreasing",
        slope=-0.1,
        intercept=10.5,
        z_statistic=-3.5,
        p_value=1e-12,
        method="mann_kendall",
    )

    with mock.patch("aquascope.groundwater.wells.trend_detection", return_value=fake_result):
        monkeypatch.setattr(sys, "argv", ["aquascope", "groundwater", "--file", str(csv_file), "--analysis", "trend"])
        cli.main()

    captured = capsys.readouterr()
    assert "Well Trend Analysis (Mann-Kendall)" in captured.out
    assert "Trend: decreasing" in captured.out
    assert "p-value: < 0.001" in captured.out
    assert "0.0000" not in captured.out


def test_groundwater_trend_cli_formats_standard_p_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    """Verify that a moderate p-value (>= 0.001) in well trend CLI outputs rounded 3 decimals."""
    csv_file = tmp_path / "well_levels.csv"
    csv_file.write_text("datetime,water_level\n2020-01-01,10.5\n2020-01-02,10.4\n2020-01-03,10.3\n")

    fake_result = WellTrendResult(
        trend="decreasing",
        slope=-0.05,
        intercept=10.5,
        z_statistic=-1.8,
        p_value=0.0423,
        method="mann_kendall",
    )

    with mock.patch("aquascope.groundwater.wells.trend_detection", return_value=fake_result):
        monkeypatch.setattr(sys, "argv", ["aquascope", "groundwater", "--file", str(csv_file), "--analysis", "trend"])
        cli.main()

    captured = capsys.readouterr()
    assert "Well Trend Analysis (Mann-Kendall)" in captured.out
    assert "p-value: 0.042" in captured.out
