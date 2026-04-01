"""Tests for groundwater recharge estimation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquascope.groundwater.recharge import (
    RechargeResult,
    chloride_mass_balance,
    soil_water_balance_recharge,
    water_table_fluctuation,
)


def _daily_index(n: int = 365, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


class TestWaterTableFluctuation:
    def setup_method(self):
        self.idx = _daily_index(365)

    def test_returns_recharge_result(self):
        levels = pd.Series(np.linspace(10, 12, 365), index=self.idx)
        result = water_table_fluctuation(levels, specific_yield=0.15)
        assert isinstance(result, RechargeResult)
        assert result.method == "water_table_fluctuation"

    def test_recharge_proportional_to_sy(self):
        levels = pd.Series(np.linspace(10, 12, 365), index=self.idx)
        r1 = water_table_fluctuation(levels, specific_yield=0.1)
        r2 = water_table_fluctuation(levels, specific_yield=0.2)
        assert r2.value_mm_per_year > r1.value_mm_per_year

    def test_rising_levels_produce_positive_recharge(self):
        levels = pd.Series(np.linspace(10, 15, 365), index=self.idx)
        result = water_table_fluctuation(levels, specific_yield=0.15)
        assert result.value_mm_per_year > 0

    def test_invalid_sy_raises(self):
        levels = pd.Series([10.0, 11.0], index=_daily_index(2))
        with pytest.raises(ValueError, match="Specific yield"):
            water_table_fluctuation(levels, specific_yield=1.5)


class TestChlorideMassBalance:
    def test_basic_calculation(self):
        # P = 1000 mm, Cl_p = 2 mg/L, Cl_gw = 20 mg/L
        # R = 1000 × (2/20) = 100 mm/year
        result = chloride_mass_balance(precip_cl=2.0, gw_cl=20.0, precip_mm=1000.0)
        assert isinstance(result, RechargeResult)
        assert result.method == "chloride_mass_balance"
        np.testing.assert_allclose(result.value_mm_per_year, 100.0)

    def test_higher_gw_cl_less_recharge(self):
        r1 = chloride_mass_balance(precip_cl=2.0, gw_cl=10.0, precip_mm=1000.0)
        r2 = chloride_mass_balance(precip_cl=2.0, gw_cl=50.0, precip_mm=1000.0)
        assert r1.value_mm_per_year > r2.value_mm_per_year

    def test_zero_cl_raises(self):
        with pytest.raises(ValueError, match="positive"):
            chloride_mass_balance(precip_cl=0.0, gw_cl=20.0, precip_mm=1000.0)

    def test_negative_precip_raises(self):
        with pytest.raises(ValueError, match="positive"):
            chloride_mass_balance(precip_cl=2.0, gw_cl=20.0, precip_mm=-100.0)


class TestSoilWaterBalance:
    def setup_method(self):
        self.idx = _daily_index(365)
        self.precip = pd.Series(np.ones(365) * 3.0, index=self.idx)  # 3 mm/day
        self.et = pd.Series(np.ones(365) * 1.5, index=self.idx)  # 1.5 mm/day
        self.runoff = pd.Series(np.ones(365) * 0.5, index=self.idx)  # 0.5 mm/day

    def test_basic_balance(self):
        # R = P - ET - Q = 3 - 1.5 - 0.5 = 1.0 mm/day → ~365 mm/year
        result = soil_water_balance_recharge(self.precip, self.et, self.runoff)
        assert isinstance(result, RechargeResult)
        assert result.method == "soil_water_balance"
        assert result.value_mm_per_year > 300  # ~365 mm/year

    def test_with_storage_change(self):
        delta_s = pd.Series(np.ones(365) * 0.2, index=self.idx)
        result = soil_water_balance_recharge(self.precip, self.et, self.runoff, delta_s=delta_s)
        # R = 3 - 1.5 - 0.5 - 0.2 = 0.8 mm/day
        result_no_ds = soil_water_balance_recharge(self.precip, self.et, self.runoff)
        assert result.value_mm_per_year < result_no_ds.value_mm_per_year

    def test_mismatched_lengths_raises(self):
        short = pd.Series([1.0, 2.0], index=_daily_index(2))
        with pytest.raises(ValueError, match="same length"):
            soil_water_balance_recharge(self.precip, self.et, short)
