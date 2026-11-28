"""Daily snow accounting, reference results and GR4J coupling."""

import numpy as np
import pandas as pd
import pytest

from aquascope.models.cemaneige import CemaNeige
from aquascope.models.rainfall_runoff import GR4J, calibrate


def test_daily_states_match_airgr_1_7_9():
    # Independent benchmark: stock airGR 1.7.9 frun_cemaneige, IsHyst=0,
    # zero initial states, mean annual solid precipitation 45, CTG=.5, Kf=4.
    # Input solid fractions from the -1/+3 C mean-temperature partition.
    dates = pd.date_range("2012-02-26", periods=8)
    precip = pd.Series([12, 3, 7, 0, 0, 0, 0, 0], index=dates)
    temp = pd.Series([-6, -3, 0, 1, 2, 4, 5, 8], index=dates)
    out = CemaNeige(precip, temp, mean_annual_snowfall=45).simulate(ctg=.5, kf=4)
    np.testing.assert_allclose(out.snowpack.iloc[:, 0], [12, 15, 20.25, 20.25, 15.849999899,
                                                         8.682277596, 6.138895465, 4.687538388], atol=1e-6, rtol=0)
    np.testing.assert_allclose(out.thermal_state.iloc[:, 0], [-3, -3, -1.5, -.25, 0, 0, 0, 0])
    np.testing.assert_allclose(out.liquid, [0, 0, 1.75, 0, 4.400000101, 7.167722303, 2.543382131, 1.451357077],
                               atol=1e-6, rtol=0)


def test_elevation_bands_and_daily_water_balance():
    dates = pd.date_range("2020-01-01", periods=5)
    precip = pd.Series([10., 3, 0, 0, 0], index=dates)
    temp = pd.Series([1., -6, 2, 8, 12], index=dates)
    snow = CemaNeige(precip, temp, band_elevations=[500, 1500], forcing_elevation=1000, lapse_rate=6)
    out = snow.simulate()
    # On day one the low band is at +4 C (rain), the high band at -2 C (snow).
    assert out.snowfall.iloc[0, 0] == 0
    assert out.snowfall.iloc[0, 1] == pytest.approx(20 * np.exp(.4) / (1 + np.exp(.4)))
    uniform = CemaNeige(precip, temp, band_elevations=[500, 1500], forcing_elevation=1000,
                        lapse_rate=6, precip_gradient=0).simulate()
    assert uniform.snowfall.iloc[0].tolist() == [0, 10]
    assert out.snowpack.index.equals(dates)
    assert out.snowpack.columns.tolist() == [500, 1500]
    assert (out.snowpack >= 0).all().all()
    change = out.snowpack - out.snowpack.shift(fill_value=0)
    np.testing.assert_allclose(change + out.melt, out.snowfall, atol=1e-12)
    np.testing.assert_allclose(change.mean(axis=1) + out.liquid, precip, atol=1e-12)


def test_all_rain_is_exactly_gr4j():
    dates = pd.date_range("2020-01-01", periods=40)
    precip = pd.Series(np.random.default_rng(17).uniform(0, 20, len(dates)), index=dates)
    temp = pd.Series(22., index=dates)
    pet = pd.Series(1., index=dates)
    model = CemaNeige(precip, temp, band_elevations=[500, 1500, 2500], forcing_elevation=1500)
    out = model.simulate()
    assert model.mean_annual_snowfall == 0
    np.testing.assert_array_equal(out.liquid, precip)
    np.testing.assert_array_equal(GR4J().simulate(out.liquid, pet).streamflow, GR4J().simulate(precip, pet).streamflow)


@pytest.mark.parametrize("objective", ["nse", "kge", "log_nse"])
def test_calibration_includes_snow_and_reproduces_returned_hydrograph(objective):
    dates = pd.date_range("2020-01-01", periods=65)
    precip = pd.Series(np.random.default_rng(14).uniform(0, 15, len(dates)), index=dates)
    temp = pd.Series(8 * np.sin(np.arange(len(dates)) * .3), index=dates)
    pet = pd.Series(1., index=dates)
    snow = CemaNeige(precip, temp)
    true_model = GR4J()
    obs = true_model.simulate(snow.simulate(.4, 4).liquid, pet).streamflow
    bounds = {k: (v, v) for k, v in true_model.params.items()}
    fit = calibrate(snow, pet, obs, objective=objective, param_bounds={**bounds, "CTG": (.4, .4), "Kf": (3., 5.)},
                    warmup_days=5, maxiter=1, seed=42)
    assert set(fit.params) == {*true_model.params, "CTG", "Kf"}
    assert fit.objective_value > .99
    replay = true_model.simulate(snow.simulate(fit.params["CTG"], fit.params["Kf"]).liquid, pet)
    np.testing.assert_array_equal(fit.simulated, replay.streamflow)


def test_reject_missing_days_misaligned_forcing_and_invalid_parameters():
    dates = pd.date_range("2020-01-01", periods=5)
    p = pd.Series(10., index=dates)
    with pytest.raises(ValueError, match="consecutive"):
        CemaNeige(p.drop(dates[2]), p.drop(dates[2]))
    with pytest.raises(ValueError, match="finite"):
        CemaNeige(p, p.mask(p.index == dates[2]))
    snow = CemaNeige(p, p)
    with pytest.raises(ValueError, match="CTG"):
        snow.simulate(1.1, 4)
    with pytest.raises(ValueError, match="same dates"):
        calibrate(snow, p.shift(freq="D"), p)
