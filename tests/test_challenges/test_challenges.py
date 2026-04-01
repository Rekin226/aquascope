"""Tests for aquascope.challenges — flood, drought, water quality."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_discharge(n: int = 730) -> pd.DataFrame:
    """Synthetic daily discharge data (2 years)."""
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    t = np.arange(n, dtype=float)
    values = 50 + 30 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 5, n)
    values = np.clip(values, 1, None)
    return pd.DataFrame({"value": values}, index=idx)


def _make_precip(n: int = 1095) -> pd.DataFrame:
    """Synthetic daily precipitation data (3 years)."""
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    rng = np.random.default_rng(123)
    values = rng.gamma(2, 3, n)
    return pd.DataFrame({"value": values}, index=idx)


def _make_wq_data() -> dict[str, pd.DataFrame]:
    """Synthetic water quality data for pH and nitrate."""
    idx = pd.date_range("2022-01-01", periods=365, freq="D")
    rng = np.random.default_rng(99)
    ph = pd.DataFrame({"value": rng.normal(7.5, 0.5, 365)}, index=idx)
    nitrate = pd.DataFrame({"value": rng.exponential(10, 365)}, index=idx)
    return {"ph": ph, "nitrate": nitrate}


class TestFloodChallenge:
    def test_load_and_fit(self):
        from aquascope.challenges.flood import FloodChallenge

        fc = FloodChallenge(lat=13.5, lon=2.1, name="Test River")
        fc.load_dataframe(_make_discharge())
        fc.fit(model="arima")
        assert fc._model is not None

    def test_forecast(self):
        from aquascope.challenges.flood import FloodChallenge

        fc = FloodChallenge(name="Test")
        fc.load_dataframe(_make_discharge())
        fc.fit(model="arima")
        forecast = fc.forecast(days=7)
        assert len(forecast) == 7
        assert "yhat" in forecast.columns

    def test_risk_assessment(self):
        from aquascope.challenges.flood import FloodChallenge

        fc = FloodChallenge(name="Test")
        fc.load_dataframe(_make_discharge())
        fc.fit(model="arima")
        risk = fc.assess_risk()
        assert "risk_level" in risk
        assert risk["risk_level"] in ("NORMAL", "LOW", "MODERATE", "HIGH", "EXTREME")

    def test_return_periods(self):
        from aquascope.challenges.flood import FloodChallenge

        fc = FloodChallenge(name="Test")
        fc.load_dataframe(_make_discharge(365 * 10))  # 10 years
        fc.fit(model="arima")
        assert fc._return_periods is not None
        assert "10yr" in fc._return_periods


class TestDroughtChallenge:
    def test_compute_spi(self):
        from aquascope.challenges.drought import DroughtChallenge

        dc = DroughtChallenge(lat=15.0, lon=0.0, name="Sahel")
        dc.load_dataframe(_make_precip())
        spi = dc.compute_spi(timescales=[1, 3])
        assert "SPI_1" in spi.columns
        assert "SPI_3" in spi.columns

    def test_current_status(self):
        from aquascope.challenges.drought import DroughtChallenge

        dc = DroughtChallenge(lat=15.0, lon=0.0)
        # Need enough monthly data for SPI gamma fitting (5+ years)
        dc.load_dataframe(_make_precip(365 * 6))
        status = dc.current_status()
        assert "overall" in status

    def test_water_balance(self):
        from aquascope.challenges.drought import DroughtChallenge

        precip = _make_precip(365)
        et = pd.DataFrame(
            {"value": np.random.default_rng(77).gamma(1.5, 2, 365)},
            index=precip.index,
        )
        dc = DroughtChallenge(lat=0, lon=0)
        dc.load_dataframe(precip, et_df=et)
        wb = dc.water_balance()
        assert "water_balance_mm" in wb.columns
        assert "surplus_deficit" in wb.columns


class TestWaterQualityChallenge:
    def test_who_guidelines(self):
        from aquascope.challenges.quality import WaterQualityChallenge

        wq = WaterQualityChallenge(site_id="TEST-001")
        wq.load_dataframes(_make_wq_data())
        result = wq.check_who_guidelines()
        assert len(result) == 2  # ph and nitrate
        assert "status" in result.columns

    def test_detect_anomalies(self):
        from aquascope.challenges.quality import WaterQualityChallenge

        data = _make_wq_data()
        # Inject anomalies
        data["ph"].iloc[50, 0] = 2.0
        data["ph"].iloc[100, 0] = 14.0

        wq = WaterQualityChallenge(site_id="TEST-001")
        wq.load_dataframes(data)
        anomalies = wq.detect_anomalies()
        assert len(anomalies) > 0
        assert "variable" in anomalies.columns

    def test_trend_analysis(self):
        from aquascope.challenges.quality import WaterQualityChallenge

        data = _make_wq_data()
        wq = WaterQualityChallenge(site_id="TEST-001")
        wq.load_dataframes(data)
        result = wq.trend_analysis("ph")
        assert "trend" in result
        assert "mann_kendall_tau" in result

    def test_summary(self):
        from aquascope.challenges.quality import WaterQualityChallenge

        wq = WaterQualityChallenge(site_id="TEST-001")
        wq.load_dataframes(_make_wq_data())
        summary = wq.summary()
        assert len(summary) == 2
        assert "mean" in summary.columns
