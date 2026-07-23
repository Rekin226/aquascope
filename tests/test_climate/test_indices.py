"""Tests for aquascope.climate.indices — climate index calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquascope.climate.indices import (
    aridity_index,
    consecutive_dry_days,
    consecutive_wet_days,
    heat_wave_index,
    palmer_drought_severity_index,
    precipitation_concentration_index,
)


class TestPalmerDroughtSeverityIndex:
    def setup_method(self):
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=120, freq="MS")
        # Moderate rainfall and PET — neutral conditions
        self.precip = pd.Series(np.random.uniform(50, 100, 120), index=idx)
        self.pet = pd.Series(np.random.uniform(40, 80, 120), index=idx)

    def test_returns_series(self):
        result = palmer_drought_severity_index(self.precip, self.pet)
        assert isinstance(result, pd.Series)
        assert len(result) == 120

    def test_index_has_same_length(self):
        result = palmer_drought_severity_index(self.precip, self.pet)
        assert (result.index == self.precip.index).all()

    def test_drought_signal(self):
        # Very low precip should produce negative PDSI
        idx = pd.date_range("2000-01-01", periods=60, freq="MS")
        dry_precip = pd.Series(np.full(60, 10.0), index=idx)
        high_pet = pd.Series(np.full(60, 100.0), index=idx)
        result = palmer_drought_severity_index(dry_precip, high_pet)
        # Last values should be negative (drought)
        assert result.iloc[-1] < 0

    def test_wet_signal(self):
        idx = pd.date_range("2000-01-01", periods=60, freq="MS")
        wet_precip = pd.Series(np.full(60, 200.0), index=idx)
        low_pet = pd.Series(np.full(60, 30.0), index=idx)
        result = palmer_drought_severity_index(wet_precip, low_pet)
        assert result.iloc[-1] > 0

    def test_series_name_is_pdsi(self):
        result = palmer_drought_severity_index(self.precip, self.pet)
        assert result.name == "PDSI"

    def test_no_nan_in_output(self):
        result = palmer_drought_severity_index(self.precip, self.pet)
        assert not result.isna().any()

    def test_first_value_is_zero(self):
        result = palmer_drought_severity_index(self.precip, self.pet)
        assert result.iloc[0] == 0.0

    def test_drier_gives_lower_pdsi(self):
        idx = pd.date_range("2000-01-01", periods=36, freq="MS")
        pet = pd.Series(np.full(36, 60.0), index=idx)
        wet = palmer_drought_severity_index(pd.Series(np.full(36, 120.0), index=idx), pet)
        dry = palmer_drought_severity_index(pd.Series(np.full(36, 5.0), index=idx), pet)
        assert dry.mean() < wet.mean()


class TestAridityIndex:
    def test_humid(self):
        result = aridity_index(1200, 800)
        assert result.classification == "humid"
        assert abs(result.index - 1.5) < 0.01

    def test_arid(self):
        result = aridity_index(100, 1000)
        assert result.classification == "arid"

    def test_semi_arid(self):
        result = aridity_index(300, 1000)
        assert result.classification == "semi-arid"

    def test_hyper_arid(self):
        result = aridity_index(10, 1000)
        assert result.classification == "hyper-arid"

    def test_zero_pet_raises(self):
        try:
            aridity_index(100, 0)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_dry_sub_humid(self):
        result = aridity_index(600, 1000)
        assert result.classification == "dry sub-humid"

    def test_negative_pet_raises(self):
        try:
            aridity_index(500, -100)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_higher_precip_higher_index(self):
        r1 = aridity_index(200, 1000)
        r2 = aridity_index(600, 1000)
        assert r2.index > r1.index


class TestHeatWaveIndex:
    def setup_method(self):
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        # Base temperature around 30°C
        vals = np.random.normal(30, 3, 365)
        # Insert a heat wave: days 100–106 very hot
        vals[100:107] = 42.0
        # Insert another: days 200–204
        vals[200:205] = 41.0
        self.tmax = pd.Series(vals, index=idx)

    def test_detects_heat_waves(self):
        result = heat_wave_index(self.tmax, threshold_percentile=90, min_duration=3)
        assert result.n_events >= 2

    def test_max_duration(self):
        result = heat_wave_index(self.tmax, threshold_percentile=90, min_duration=3)
        assert result.max_duration >= 5

    def test_events_have_positive_intensity(self):
        result = heat_wave_index(self.tmax, threshold_percentile=90, min_duration=3)
        for event in result.events:
            assert event.peak_intensity > 0

    def test_no_events_when_min_duration_high(self):
        result = heat_wave_index(self.tmax, threshold_percentile=90, min_duration=30)
        assert result.n_events == 0

    def test_zero_events_fields(self):
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        tmax = pd.Series(np.full(365, 25.0), index=idx)
        result = heat_wave_index(tmax, min_duration=30)
        assert result.n_events == 0
        assert result.max_duration == 0
        assert result.mean_duration == 0.0
        assert result.mean_intensity == 0.0

    def test_events_list_length_matches_n_events(self):
        result = heat_wave_index(self.tmax, threshold_percentile=90, min_duration=3)
        assert len(result.events) == result.n_events


class TestConsecutiveDryDays:
    def setup_method(self):
        idx = pd.date_range("2000-01-01", periods=730, freq="D")  # 2 years
        vals = np.random.uniform(0, 5, 730)
        # Force a 20-day dry spell in year 1
        vals[50:70] = 0.0
        # Force a 10-day dry spell in year 2
        vals[400:410] = 0.0
        self.precip = pd.Series(vals, index=idx)

    def test_max_cdd(self):
        result = consecutive_dry_days(self.precip, threshold_mm=1.0)
        assert result.max_cdd >= 20

    def test_by_year_has_entries(self):
        result = consecutive_dry_days(self.precip, threshold_mm=1.0)
        assert 2000 in result.by_year
        assert 2001 in result.by_year

    def test_year1_longer_than_year2(self):
        result = consecutive_dry_days(self.precip, threshold_mm=1.0)
        assert result.by_year[2000] >= result.by_year[2001]

    def test_all_dry_max_cdd_equals_year_length(self):
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        precip = pd.Series(np.zeros(365), index=idx)
        result = consecutive_dry_days(precip, threshold_mm=1.0)
        assert result.max_cdd == 365

    def test_all_wet_max_cdd_is_zero(self):
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        precip = pd.Series(np.full(365, 5.0), index=idx)
        result = consecutive_dry_days(precip, threshold_mm=1.0)
        assert result.max_cdd == 0

    def test_mean_cdd_matches_manual(self):
        result = consecutive_dry_days(self.precip, threshold_mm=1.0)
        expected = np.mean(list(result.by_year.values()))
        assert abs(result.mean_cdd - expected) < 1e-9


class TestConsecutiveWetDays:
    def setup_method(self):
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        vals = np.random.uniform(0, 0.5, 365)  # mostly dry
        # Force a 15-day wet spell
        vals[80:95] = 10.0
        self.precip = pd.Series(vals, index=idx)

    def test_max_cwd(self):
        result = consecutive_wet_days(self.precip, threshold_mm=1.0)
        assert result.max_cwd >= 15

    def test_mean_cwd_positive(self):
        result = consecutive_wet_days(self.precip, threshold_mm=1.0)
        assert result.mean_cwd > 0

    def test_all_wet_max_cwd_equals_year_length(self):
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        precip = pd.Series(np.full(365, 5.0), index=idx)
        result = consecutive_wet_days(precip, threshold_mm=1.0)
        assert result.max_cwd == 365

    def test_all_dry_max_cwd_is_zero(self):
        idx = pd.date_range("2000-01-01", periods=365, freq="D")
        precip = pd.Series(np.zeros(365), index=idx)
        result = consecutive_wet_days(precip, threshold_mm=1.0)
        assert result.max_cwd == 0

    def test_mean_cwd_matches_manual(self):
        result = consecutive_wet_days(self.precip, threshold_mm=1.0)
        expected = np.mean(list(result.by_year.values()))
        assert abs(result.mean_cwd - expected) < 1e-9


class TestPrecipitationConcentrationIndex:
    def test_uniform_distribution(self):
        # Equal monthly rainfall → PCI ≈ 8.3
        monthly = pd.Series(np.full(12, 100.0))
        pci = precipitation_concentration_index(monthly)
        assert abs(pci - 8.33) < 0.1

    def test_concentrated_rainfall(self):
        # All rain in one month → PCI = 100
        monthly = pd.Series([1200.0] + [0.0] * 11)
        pci = precipitation_concentration_index(monthly)
        assert abs(pci - 100.0) < 0.1

    def test_too_few_months_raises(self):
        try:
            precipitation_concentration_index(pd.Series([10.0] * 6))
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_multi_year_uses_first_12(self):
        monthly = pd.Series(np.full(24, 100.0))
        pci = precipitation_concentration_index(monthly)
        assert abs(pci - 8.33) < 0.1

    def test_all_zero_returns_zero(self):
        monthly = pd.Series(np.zeros(12))
        pci = precipitation_concentration_index(monthly)
        assert pci == 0.0

    def test_more_concentrated_gives_higher_pci(self):
        uniform = pd.Series(np.full(12, 100.0))
        seasonal = pd.Series([0.0] * 11 + [1200.0])
        assert precipitation_concentration_index(seasonal) > \
               precipitation_concentration_index(uniform)
