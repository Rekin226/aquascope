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
        """Very dry series should produce negative SPI."""
        from aquascope.climate.indices import spi
        rng = np.random.default_rng(7)
        idx = pd.date_range("2000-01-01", periods=120, freq="MS")
        dry = pd.Series(rng.gamma(shape=2.0, scale=0.5, size=120), index=idx)
        wet = pd.Series(rng.gamma(shape=2.0, scale=100.0, size=120), index=idx)
        dry_result = spi(dry, scale=3)
        wet_result = spi(wet, scale=3)
        assert dry_result.spi.dropna().mean() < wet_result.spi.dropna().mean()

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

class TestDroughtClass:
    def test_extremely_wet(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(2.5) == "extremely_wet"

    def test_severely_wet(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(1.7) == "severely_wet"

    def test_moderately_wet(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(1.2) == "moderately_wet"

    def test_near_normal(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(0.0) == "near_normal"

    def test_moderately_dry(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(-1.2) == "moderately_dry"

    def test_severely_dry(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(-1.7) == "severely_dry"

    def test_extremely_dry(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(-2.5) == "extremely_dry"

    def test_nan_returns_nan_string(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(float("nan")) == "nan"

    def test_boundary_2_is_extremely_wet(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(2.0) == "extremely_wet"

    def test_boundary_minus2_is_extremely_dry(self):
        from aquascope.climate.indices import drought_class
        assert drought_class(-2.0) == "extremely_dry"


class TestSPI:
    def setup_method(self):
        np.random.seed(42)
        idx = pd.date_range("2000-01-01", periods=120, freq="MS")
        self.precip = pd.Series(
            np.random.gamma(shape=2.0, scale=30.0, size=120),
            index=idx,
        )

    def test_returns_spi_result(self):
        from aquascope.climate.indices import SPIResult, spi
        result = spi(self.precip, scale=3)
        assert isinstance(result, SPIResult)

    def test_spi_series_length_matches_input(self):
        from aquascope.climate.indices import spi
        result = spi(self.precip, scale=3)
        assert len(result.spi) == len(self.precip)

    def test_spi_index_matches_input(self):
        from aquascope.climate.indices import spi
        result = spi(self.precip, scale=3)
        assert (result.spi.index == self.precip.index).all()

    def test_scale_stored_correctly(self):
        from aquascope.climate.indices import spi
        result = spi(self.precip, scale=6)
        assert result.scale == 6

    def test_first_scale_minus_one_values_are_nan(self):
        """Rolling window means first scale-1 values are NaN."""
        from aquascope.climate.indices import spi
        result = spi(self.precip, scale=3)
        assert result.spi.iloc[:2].isna().all()

    def test_valid_values_in_plausible_range(self):
        """SPI values should typically fall between -3 and 3."""
        from aquascope.climate.indices import spi
        result = spi(self.precip, scale=3)
        valid = result.spi.dropna()
        assert (valid >= -4.0).all()
        assert (valid <= 4.0).all()

    def test_drought_signal(self):
        """Very dry series should produce negative SPI."""
        from aquascope.climate.indices import spi
        idx = pd.date_range("2000-01-01", periods=120, freq="MS")
        dry = pd.Series(np.full(120, 1.0), index=idx)
        wet = pd.Series(np.full(120, 200.0), index=idx)
        dry_result = spi(dry, scale=3)
        wet_result = spi(wet, scale=3)
        assert dry_result.spi.dropna().mean() < wet_result.spi.dropna().mean()

    def test_drought_classes_length_matches_input(self):
        from aquascope.climate.indices import spi
        result = spi(self.precip, scale=3)
        assert len(result.drought_classes) == len(self.precip)

    def test_drought_classes_valid_strings(self):
        from aquascope.climate.indices import spi
        valid_classes = {
            "extremely_wet", "severely_wet", "moderately_wet",
            "near_normal", "moderately_dry", "severely_dry",
            "extremely_dry", "nan",
        }
        result = spi(self.precip, scale=3)
        assert set(result.drought_classes.unique()).issubset(valid_classes)

    def test_scale_1_produces_more_valid_values_than_scale_12(self):
        from aquascope.climate.indices import spi
        r1 = spi(self.precip, scale=1)
        r12 = spi(self.precip, scale=12)
        assert r1.spi.notna().sum() > r12.spi.notna().sum()

    def test_invalid_scale_raises(self):
        from aquascope.climate.indices import spi
        try:
            spi(self.precip, scale=0)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_too_short_series_raises(self):
        from aquascope.climate.indices import spi
        short = self.precip.iloc[:2]
        try:
            spi(short, scale=3)
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_zero_heavy_precip_handled(self):
        """Series with many zeros should not crash."""
        from aquascope.climate.indices import spi
        idx = pd.date_range("2000-01-01", periods=120, freq="MS")
        vals = np.zeros(120)
        vals[::3] = 50.0  # every 3rd month has rain
        precip = pd.Series(vals, index=idx)
        result = spi(precip, scale=3)
        assert isinstance(result.spi, pd.Series)
