from __future__ import annotations

import pandas as pd

from aquascope.agri.water_balance import SoilProperties, SoilWaterBalance


def make_balance(initial_depletion: float = 0.0) -> SoilWaterBalance:
    soil = SoilProperties(
        field_capacity=0.30,
        wilting_point=0.15,
        root_depth=1.0,
    )
    return SoilWaterBalance(soil=soil, initial_depletion=initial_depletion)


def test_auto_irrigate_triggers_during_long_dry_spell() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    etc_series = pd.Series([8.0] * 12, index=dates)
    precip_series = pd.Series([0.0] * 12, index=dates)

    balance = make_balance()
    result = balance.auto_irrigate(etc_series, precip_series)

    assert (result["irrigation_mm"] >= 0).all()
    assert (result["depletion_mm"] >= 0).all()
    assert result["irrigation_mm"].sum() > 0
    assert result["irrigation_trigger"].any()


def test_auto_irrigate_empty_precip_matches_zero_precip() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    etc_series = pd.Series([6.0] * 10, index=dates)
    empty_precip = pd.Series([], dtype=float)
    zero_precip = pd.Series([0.0] * 10, index=dates)

    empty_result = make_balance().auto_irrigate(etc_series, empty_precip)
    zero_result = make_balance().auto_irrigate(etc_series, zero_precip)

    pd.testing.assert_frame_equal(empty_result, zero_result)


def test_auto_irrigate_heavy_rain_keeps_depletion_non_negative() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    etc_series = pd.Series([5.0] * 10, index=dates)
    precip_series = pd.Series([200.0] * 10, index=dates)

    result = make_balance(initial_depletion=50.0).auto_irrigate(etc_series, precip_series)

    assert (result["depletion_mm"] >= 0).all()
    # Heavy rain refills the soil, so no irrigation should ever be needed.
    assert result["irrigation_mm"].sum() == 0


def test_auto_irrigate_low_efficiency_applies_more_water() -> None:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    etc_series = pd.Series([8.0] * 12, index=dates)
    precip_series = pd.Series([0.0] * 12, index=dates)

    high_eff = make_balance().auto_irrigate(etc_series, precip_series, efficiency=0.9)
    low_eff = make_balance().auto_irrigate(etc_series, precip_series, efficiency=0.5)

    assert low_eff["irrigation_mm"].sum() > high_eff["irrigation_mm"].sum()


def test_auto_irrigate_results_are_never_negative() -> None:
    dates = pd.date_range("2024-01-01", periods=15, freq="D")
    etc_series = pd.Series([7.0] * 15, index=dates)
    # Mix of dry days and occasional rain to exercise both branches.
    precip_series = pd.Series([0.0, 0.0, 0.0, 30.0] * 3 + [0.0, 0.0, 0.0], index=dates)

    result = make_balance().auto_irrigate(etc_series, precip_series)

    assert (result["irrigation_mm"] >= 0).all()
    assert (result["depletion_mm"] >= 0).all()
