"""Tests for aquascope.viz module.

Uses matplotlib's non-interactive Agg backend so tests run headless in CI.
Tests verify that plot functions return Figure objects and optionally save
files without errors — pixel-level image comparison is NOT tested.
"""

from __future__ import annotations

import os
import tempfile

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402


class TestTimeseriesPlots:
    """Tests for aquascope.viz.timeseries functions."""

    def setup_method(self):
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        self.df = pd.DataFrame(
            {"value": np.sin(np.linspace(0, 4 * np.pi, 365)) * 10 + 50},
            index=dates,
        )
        self.forecast_df = pd.DataFrame(
            {
                "yhat": np.random.default_rng(42).normal(50, 5, 30),
                "yhat_lower": np.random.default_rng(42).normal(45, 5, 30),
                "yhat_upper": np.random.default_rng(42).normal(55, 5, 30),
            },
            index=pd.date_range("2021-01-01", periods=30, freq="D"),
        )

    def test_plot_timeseries_returns_figure(self):
        from aquascope.viz import plot_timeseries

        fig = plot_timeseries(self.df, title="Test TS")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_timeseries_save(self):
        from aquascope.viz import plot_timeseries

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plot_timeseries(self.df, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000
        finally:
            os.unlink(path)

    def test_plot_multi_param(self):
        from aquascope.viz import plot_multi_param

        df = self.df.copy()
        df["value2"] = df["value"] * 1.5
        fig = plot_multi_param(df, columns=["value", "value2"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_forecast(self):
        from aquascope.viz import plot_forecast

        fig = plot_forecast(observed=self.df, forecast=self.forecast_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_forecast_no_observed(self):
        from aquascope.viz import plot_forecast

        fig = plot_forecast(forecast=self.forecast_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_observed_vs_predicted(self):
        from aquascope.viz.timeseries import plot_observed_vs_predicted

        obs = pd.Series(np.random.default_rng(1).normal(50, 5, 100))
        pred = obs + np.random.default_rng(2).normal(0, 2, 100)
        metrics = {"NSE": 0.85, "KGE": 0.90, "RMSE": 2.1}
        fig = plot_observed_vs_predicted(obs, pred, metrics=metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_residuals(self):
        from aquascope.viz.timeseries import plot_residuals

        obs = pd.Series(np.random.default_rng(1).normal(50, 5, 100))
        pred = obs + np.random.default_rng(2).normal(0, 2, 100)
        fig = plot_residuals(obs, pred)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestQualityPlots:
    """Tests for aquascope.viz.quality functions."""

    def setup_method(self):
        rng = np.random.default_rng(42)
        n = 200
        self.df = pd.DataFrame({
            "value": rng.normal(7, 1, n),
            "station_name": rng.choice(["Station A", "Station B", "Station C"], n),
            "parameter": rng.choice(["pH", "DO", "BOD5"], n),
        })

    def test_plot_boxplot(self):
        from aquascope.viz import plot_boxplot

        fig = plot_boxplot(self.df, value_col="value", group_col="station_name")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_heatmap(self):
        from aquascope.viz import plot_heatmap

        num_df = pd.DataFrame(np.random.default_rng(42).normal(0, 1, (50, 4)), columns=["pH", "DO", "BOD5", "COD"])
        fig = plot_heatmap(num_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_who_exceedances(self):
        from aquascope.viz import plot_who_exceedances

        who_df = pd.DataFrame({
            "variable": ["pH", "DO", "turbidity", "nitrate"],
            "pct_exceedances": [5.0, 12.0, 3.0, 25.0],
            "status": ["PASS", "FAIL", "PASS", "FAIL"],
        })
        fig = plot_who_exceedances(who_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_eda_summary(self):
        from dataclasses import dataclass, field

        from aquascope.viz import plot_eda_summary

        @dataclass
        class MockParam:
            name: str
            count: int
            missing: int
            mean: float
            std: float
            min: float
            q25: float
            median: float
            q75: float
            max: float
            outlier_count: int = 0

        @dataclass
        class MockReport:
            parameters: list = field(default_factory=list)

        report = MockReport(parameters=[
            MockParam("pH", 100, 5, 7.2, 0.5, 5.0, 6.8, 7.2, 7.5, 9.0, 3),
            MockParam("DO", 90, 10, 8.5, 1.2, 3.0, 7.5, 8.5, 9.5, 12.0, 5),
        ])
        fig = plot_eda_summary(report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_param_comparison(self):
        from aquascope.viz import plot_param_comparison

        fig = plot_param_comparison(self.df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSpatialPlots:
    """Tests for aquascope.viz.spatial functions."""

    def setup_method(self):
        self.stations = pd.DataFrame({
            "station_name": ["Taipei", "Taichung", "Kaohsiung"],
            "latitude": [25.03, 24.15, 22.63],
            "longitude": [121.57, 120.68, 120.30],
            "value": [7.2, 6.8, 7.5],
        })

    def test_plot_station_scatter(self):
        from aquascope.viz import plot_station_scatter

        fig = plot_station_scatter(self.stations, value_col="value")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_station_scatter_no_value(self):
        from aquascope.viz import plot_station_scatter

        fig = plot_station_scatter(self.stations)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_station_map_folium(self):
        """Test folium interactive map (skipped if folium not installed)."""
        try:
            import folium  # noqa: F401
        except ImportError:
            pytest.skip("folium not installed")

        from aquascope.viz import plot_station_map

        m = plot_station_map(self.stations)
        assert hasattr(m, "_repr_html_")

    def test_plot_station_map_save(self):
        """Test folium map saving to HTML."""
        try:
            import folium  # noqa: F401
        except ImportError:
            pytest.skip("folium not installed")

        from aquascope.viz import plot_station_map

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            plot_station_map(self.stations, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 500
        finally:
            os.unlink(path)


class TestHydroPlots:
    """Tests for aquascope.viz.hydro functions."""

    def setup_method(self):
        dates = pd.date_range("2015-01-01", periods=3650, freq="D")
        rng = np.random.default_rng(42)
        self.discharge = pd.Series(
            rng.exponential(20, 3650) + 5, index=dates, name="discharge",
        )

    def test_plot_fdc(self):
        from aquascope.viz import plot_fdc

        fig = plot_fdc(self.discharge)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_hydrograph(self):
        from aquascope.viz.hydro import plot_hydrograph

        df = pd.DataFrame({
            "discharge": self.discharge.values[:365],
            "baseflow": self.discharge.values[:365] * 0.6,
        }, index=self.discharge.index[:365])
        fig = plot_hydrograph(df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_spi_timeline(self):
        from aquascope.viz import plot_spi_timeline

        dates = pd.date_range("2015-01-01", periods=120, freq="MS")
        rng = np.random.default_rng(42)
        spi_df = pd.DataFrame({"spi_3": rng.normal(0, 1.2, 120)}, index=dates)
        fig = plot_spi_timeline(spi_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_return_periods(self):
        from aquascope.viz import plot_return_periods

        rp = {2: 50.0, 5: 80.0, 10: 100.0, 25: 130.0, 50: 155.0, 100: 180.0}
        fig = plot_return_periods(rp, observed_max=120.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
