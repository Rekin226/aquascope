"""Tests for the agriculture module (FAO-56 implementation).

Validates against worked examples from FAO Irrigation & Drainage Paper 56.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from aquascope.agri.crop_water import (
    DEFAULT_STAGE_LENGTHS,
    KC_TABLE,
    crop_et,
    crop_water_requirement,
    effective_rainfall,
    get_kc,
    irrigation_schedule,
)
from aquascope.agri.eto import (
    atmospheric_pressure,
    clear_sky_radiation,
    extraterrestrial_radiation,
    hargreaves,
    net_longwave_radiation,
    net_radiation,
    net_shortwave_radiation,
    penman_monteith_daily,
    psychrometric_constant,
    saturation_vapor_pressure,
    slope_vapor_pressure_curve,
)
from aquascope.agri.water_balance import SoilProperties, SoilWaterBalance
from aquascope.collectors.aquastat import AquastatCollector
from aquascope.collectors.wapor import WaPORCollector
from aquascope.schemas.agriculture import AquastatRecord, ETReference

# ── Helper to build ETo/precip series for crop water tests ──────────────


def _make_daily_series(n_days: int, value: float, start: str = "2024-01-01") -> pd.Series:
    """Create a constant daily series for testing."""
    idx = pd.date_range(start=start, periods=n_days, freq="D")
    return pd.Series([value] * n_days, index=idx)


# =========================================================================
# FAO-56 ET₀ helper functions
# =========================================================================


class TestSaturationVaporPressure:
    def test_at_20c(self):
        """FAO-56 Example 3: e°(20) = 2.338 kPa."""
        assert abs(saturation_vapor_pressure(20.0) - 2.338) < 0.001

    def test_at_35c(self):
        """e°(35) ≈ 5.624 kPa."""
        assert abs(saturation_vapor_pressure(35.0) - 5.624) < 0.01

    def test_at_0c(self):
        """e°(0) ≈ 0.6108 kPa (by definition of the equation)."""
        assert abs(saturation_vapor_pressure(0.0) - 0.6108) < 0.001


class TestSlopeVaporPressureCurve:
    def test_at_20c(self):
        """Slope at 20 °C should be approximately 0.1447 kPa/°C."""
        delta = slope_vapor_pressure_curve(20.0)
        assert abs(delta - 0.1447) < 0.001


class TestAtmosphericPressure:
    def test_at_sea_level(self):
        """P at z=0 should be 101.3 kPa."""
        assert abs(atmospheric_pressure(0.0) - 101.3) < 0.1

    def test_at_1800m(self):
        """FAO-56 Example 2: P at z=1800 → ~81.8 kPa."""
        p = atmospheric_pressure(1800.0)
        assert abs(p - 81.8) < 0.5


class TestPsychrometricConstant:
    def test_at_sea_level(self):
        """FAO-56 Example 2: γ at z=0 → P=101.3 → γ ≈ 0.0674 kPa/°C."""
        gamma = psychrometric_constant(0.0)
        assert abs(gamma - 0.0674) < 0.001

    def test_at_1800m(self):
        """FAO-56 Example 2: γ at z=1800m → ~0.054 kPa/°C."""
        gamma = psychrometric_constant(1800.0)
        assert abs(gamma - 0.054) < 0.002


class TestExtraterrestrialRadiation:
    def test_known_value(self):
        """FAO-56 Example 8: Ra at lat=-22.9° on Sep 3 (DOY 246) ≈ 32.2 MJ/m²/day."""
        ra = extraterrestrial_radiation(-22.9, 246)
        assert abs(ra - 32.2) < 1.5

    def test_positive_value(self):
        """Ra should always be positive for inhabited latitudes."""
        ra = extraterrestrial_radiation(45.0, 172)
        assert ra > 0


class TestClearSkyRadiation:
    def test_basic(self):
        """Rso at sea level should be 0.75 × Ra."""
        ra = 30.0
        rso = clear_sky_radiation(ra, 0.0)
        assert abs(rso - 0.75 * ra) < 0.01


class TestNetShortwaveRadiation:
    def test_default_albedo(self):
        """Rns = (1 - 0.23) × Rs."""
        rns = net_shortwave_radiation(20.0)
        assert abs(rns - 0.77 * 20.0) < 0.01


class TestNetLongwaveRadiation:
    def test_positive(self):
        """Net longwave radiation should be positive (outgoing)."""
        rnl = net_longwave_radiation(20.0, 30.0, 2.0, 18.0, 22.0)
        assert rnl > 0


class TestNetRadiation:
    def test_basic(self):
        """Rn = Rns - Rnl; should be positive for daytime conditions."""
        rn = net_radiation(20.0, 22.0, 15.0, 30.0, 2.0, 100.0)
        assert rn > 0


# =========================================================================
# Full Penman-Monteith ET₀
# =========================================================================


class TestPenmanMonteith:
    def test_fao56_example18(self):
        """FAO-56 Example 18: Bangkok, Thailand.

        T_min=25.6, T_max=34.8, RH_min=63, RH_max=84,
        u2=2.0, Rs=22.0, lat=13.73, elev=2, DOY=274.
        Expected ET₀ ≈ 5.0 mm/day (within ±0.5).
        """
        eto = penman_monteith_daily(
            t_min=25.6,
            t_max=34.8,
            rh_min=63.0,
            rh_max=84.0,
            u2=2.0,
            rs=22.0,
            latitude=13.73,
            elevation=2.0,
            doy=274,
        )
        assert abs(eto - 5.0) < 0.5

    def test_non_negative(self):
        """ET₀ should never be negative."""
        eto = penman_monteith_daily(
            t_min=5.0, t_max=10.0, rh_min=80.0, rh_max=95.0,
            u2=0.5, rs=5.0, latitude=50.0, elevation=100.0, doy=15,
        )
        assert eto >= 0.0


# =========================================================================
# Hargreaves
# =========================================================================


class TestHargreaves:
    def test_basic(self):
        """Hargreaves should give reasonable ET₀ (2–10 mm/day for tropics)."""
        ra = extraterrestrial_radiation(10.0, 180)
        eto = hargreaves(24.0, 34.0, ra)
        assert 2.0 < eto < 10.0

    def test_zero_temp_range(self):
        """Zero temperature range → zero ET₀."""
        eto = hargreaves(25.0, 25.0, 30.0)
        assert eto == 0.0


# =========================================================================
# Crop coefficients and water requirements
# =========================================================================


class TestKcTable:
    def test_known_crops_exist(self):
        """All 20 crops should be in KC_TABLE."""
        assert len(KC_TABLE) >= 20

    def test_get_kc_maize_mid(self):
        """Kc for maize mid-season = 1.20."""
        kc = get_kc("maize", "mid")
        assert kc == 1.20

    def test_get_kc_all_stages(self):
        """get_kc without stage returns a dict."""
        result = get_kc("wheat_winter")
        assert isinstance(result, dict)
        assert "initial" in result
        assert "mid" in result
        assert "late" in result

    def test_get_kc_unknown_crop(self):
        """Raises ValueError for unknown crop."""
        with pytest.raises(ValueError, match="Unknown crop"):
            get_kc("unicorn_grass")


class TestCropEt:
    def test_basic(self):
        """ETc = Kc × ET₀."""
        assert abs(crop_et(5.0, 1.2) - 6.0) < 0.001


class TestCropWaterRequirement:
    def test_maize_growing_season(self):
        """Full growing season CWR for maize returns a DataFrame."""
        n_days = sum(DEFAULT_STAGE_LENGTHS["maize"].values())
        eto = _make_daily_series(n_days + 30, 5.0)
        df = crop_water_requirement(eto, "maize", date(2024, 1, 1))
        assert len(df) == n_days
        assert "stage" in df.columns
        assert "kc" in df.columns
        assert "etc" in df.columns

    def test_stages_cover_full_season(self):
        """All days should be assigned a stage."""
        n_days = sum(DEFAULT_STAGE_LENGTHS["maize"].values())
        eto = _make_daily_series(n_days + 30, 5.0)
        df = crop_water_requirement(eto, "maize", date(2024, 1, 1))
        stages = set(df["stage"].unique())
        assert stages == {"initial", "development", "mid", "late"}

    def test_unknown_crop_raises(self):
        """Unknown crop without stage_lengths should raise."""
        eto = _make_daily_series(30, 5.0)
        with pytest.raises(ValueError):
            crop_water_requirement(eto, "unicorn_grass", date(2024, 1, 1))


class TestEffectiveRainfall:
    def test_usda_method(self):
        """USDA SCS method returns less than actual rainfall."""
        eff = effective_rainfall(50.0, method="usda")
        assert 0 < eff < 50.0

    def test_zero_rain(self):
        """Zero precip → zero effective rain."""
        assert effective_rainfall(0.0) == 0.0

    def test_fao_method(self):
        """FAO method should return a reasonable value."""
        eff = effective_rainfall(30.0, method="fao")
        assert 0 < eff < 30.0

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError):
            effective_rainfall(10.0, method="magic")


# =========================================================================
# Irrigation schedule
# =========================================================================


class TestIrrigationSchedule:
    def test_schedule_returns_dataframe(self):
        """Returns DataFrame with expected columns."""
        n_days = sum(DEFAULT_STAGE_LENGTHS["maize"].values()) + 30
        eto = _make_daily_series(n_days, 5.0)
        precip = _make_daily_series(n_days, 2.0)
        df = irrigation_schedule(eto, precip, "maize", date(2024, 1, 1))
        assert "net_irrigation" in df.columns
        assert "gross_irrigation" in df.columns
        assert len(df) > 0

    def test_gross_exceeds_net(self):
        """Gross irrigation = net / efficiency → gross >= net."""
        n_days = sum(DEFAULT_STAGE_LENGTHS["maize"].values()) + 30
        eto = _make_daily_series(n_days, 6.0)
        precip = _make_daily_series(n_days, 1.0)
        df = irrigation_schedule(eto, precip, "maize", date(2024, 1, 1), efficiency=0.7)
        mask = df["net_irrigation"] > 0
        if mask.any():
            assert (df.loc[mask, "gross_irrigation"] >= df.loc[mask, "net_irrigation"]).all()


# =========================================================================
# Soil water balance
# =========================================================================


class TestSoilWaterBalance:
    def setup_method(self):
        """Create a standard soil for tests."""
        self.soil = SoilProperties(field_capacity=0.30, wilting_point=0.15, root_depth=1.0)

    def test_total_available_water(self):
        """TAW = 1000 × (0.30 - 0.15) × 1.0 = 150 mm."""
        assert abs(self.soil.total_available_water - 150.0) < 0.1

    def test_readily_available_water(self):
        """RAW = 0.5 × 150 = 75 mm."""
        assert abs(self.soil.readily_available_water() - 75.0) < 0.1

    def test_field_capacity_start(self):
        """Starting at FC, no irrigation trigger initially."""
        swb = SoilWaterBalance(self.soil, initial_depletion=0.0)
        status = swb.step(etc=3.0, precipitation=3.0)
        assert not status.irrigation_trigger

    def test_depletion_increases_without_rain(self):
        """Depletion grows when ETc > 0 and no rain."""
        swb = SoilWaterBalance(self.soil, initial_depletion=0.0)
        status = swb.step(etc=5.0, precipitation=0.0)
        assert status.depletion_mm > 0

    def test_auto_irrigate_triggers(self):
        """Auto-irrigation fires when depletion > RAW."""
        swb = SoilWaterBalance(self.soil, initial_depletion=0.0)
        n_days = 30
        etc = _make_daily_series(n_days, 6.0)
        precip = _make_daily_series(n_days, 0.0)
        df = swb.auto_irrigate(etc, precip)
        assert df["irrigation_mm"].sum() > 0

    def test_deep_percolation_after_heavy_rain(self):
        """DP > 0 when P pushes moisture above FC."""
        swb = SoilWaterBalance(self.soil, initial_depletion=5.0)
        status = swb.step(etc=0.0, precipitation=100.0)
        assert status.deep_percolation_mm > 0

    def test_run_returns_dataframe(self):
        """Run method returns DataFrame with correct length."""
        swb = SoilWaterBalance(self.soil)
        n_days = 10
        etc = _make_daily_series(n_days, 4.0)
        precip = _make_daily_series(n_days, 1.0)
        df = swb.run(etc, precip)
        assert len(df) == n_days


# =========================================================================
# Schemas
# =========================================================================


class TestSchemas:
    def test_et_reference_model(self):
        """ETReference schema validates correctly."""
        record = ETReference(
            date=date(2024, 7, 1),
            eto_mm=5.2,
            method="penman_monteith",
            t_min=22.0,
            t_max=35.0,
        )
        assert record.eto_mm == 5.2
        assert record.method == "penman_monteith"

    def test_aquastat_record(self):
        """AquastatRecord schema validates."""
        record = AquastatRecord(
            country="Egypt",
            country_code="EGY",
            year=2020,
            variable="Agricultural water withdrawal",
            value=61.35,
            unit="10^9 m3/year",
        )
        assert record.country == "Egypt"
        assert record.source == "AQUASTAT"


# =========================================================================
# Collectors (instantiation only – no live API calls)
# =========================================================================


class TestAquastatCollector:
    def test_collector_instantiates(self):
        """Can create AquastatCollector instance."""
        collector = AquastatCollector()
        assert collector.name == "aquastat"

    def test_normalise_empty(self):
        """Normalising an empty list returns an empty list."""
        collector = AquastatCollector()
        assert collector.normalise([]) == []

    def test_normalise_valid_record(self):
        """Normalise a valid raw record."""
        collector = AquastatCollector()
        raw = [
            {
                "Area": "Egypt", "Area Code": "EGY", "Year": "2020",
                "Element": "Total water withdrawal", "Value": "77.5",
                "Unit": "10^9 m3/year",
            }
        ]
        records = collector.normalise(raw)
        assert len(records) == 1
        assert records[0].country == "Egypt"


class TestWaPORCollector:
    def test_collector_instantiates(self):
        """Can create WaPORCollector instance."""
        collector = WaPORCollector()
        assert collector.name == "wapor"

    def test_normalise_empty(self):
        """Normalising an empty list returns an empty list."""
        collector = WaPORCollector()
        assert collector.normalise([]) == []
