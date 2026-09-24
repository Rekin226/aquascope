"""Tests for aquascope.hydrology.budyko — the Budyko framework."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aquascope.hydrology.budyko import budyko


class TestBudykoCurves:
    def test_analytic_values_at_unit_aridity(self):
        result = budyko(1.0, 1.0, curves=("schreiber", "oldekop", "turc_pike", "fu_zhang"))
        assert result.aridity_index == pytest.approx(1.0)
        assert result.predicted["schreiber"] == pytest.approx(1 - np.exp(-1.0))
        assert result.predicted["oldekop"] == pytest.approx(np.tanh(1.0))
        assert result.predicted["turc_pike"] == pytest.approx(1 / np.sqrt(2.0))
        assert result.predicted["fu_zhang"] == pytest.approx(2.0 - np.sqrt(2.0))

    def test_energy_limit_at_high_aridity(self):
        result = budyko(1.0, 1e4)
        assert result.aridity_index == pytest.approx(1e4)
        for values in result.predicted.values():
            assert values == pytest.approx(1.0, abs=1e-4)

    def test_water_limit_at_low_aridity(self):
        result = budyko(1.0, 1e-3)
        assert result.aridity_index == pytest.approx(1e-3)
        for values in result.predicted.values():
            assert values == pytest.approx(1e-3, abs=1e-5)

    def test_oldekop_near_zero_aridity(self):
        result = budyko(1000.0, 1e-3, curves=("oldekop",))
        assert result.aridity_index == pytest.approx(1e-6)
        assert result.predicted["oldekop"] == pytest.approx(1e-6, rel=1e-3)

    def test_monotonic_in_aridity(self):
        result = budyko(1.0, [0.5, 1.0, 2.0], curves=("fu_zhang",))
        values = result.predicted["fu_zhang"]
        assert np.all(np.diff(values) > 0)

    def test_result_carries_requested_curves_only(self):
        result = budyko(1000.0, 1200.0, curves=("fu_zhang",))
        assert set(result.predicted) == {"fu_zhang"}
        assert set(result.curves) == {"fu_zhang"}

    def test_curve_family_matches_prediction(self):
        result = budyko(1000.0, 1200.0, curves=("fu_zhang", "turc_pike"))
        phi = result.aridity_index
        for kind, family in result.curves.items():
            idx = int(np.argmin(np.abs(result.aridity_grid - phi)))
            assert family[idx] == pytest.approx(result.predicted[kind], abs=0.02)

    def test_aridity_grid_is_increasing(self):
        result = budyko(1000.0, 1200.0)
        assert np.all(np.diff(result.aridity_grid) > 0)

    def test_scalar_inputs_return_scalars(self):
        result = budyko(1000.0, 1200.0)
        assert isinstance(result.aridity_index, float)
        assert all(isinstance(v, float) for v in result.predicted.values())

    def test_array_inputs_preserve_shape(self):
        result = budyko(np.array([1000.0, 2000.0]), np.array([1500.0, 1000.0]))
        assert np.allclose(result.aridity_index, [1.5, 0.5])
        for values in result.predicted.values():
            assert isinstance(values, np.ndarray)
            assert values.shape == (2,)

    def test_broadcast_scalar_precipitation(self):
        result = budyko(1000.0, np.array([1000.0, 2000.0]))
        assert np.allclose(result.aridity_index, [1.0, 2.0])

    def test_energy_limit_melts_into_one(self):
        result = budyko(1.0, 1e6, curves=("fu_zhang", "turc_pike"))
        assert result.predicted["turc_pike"] == pytest.approx(1.0)
        assert result.predicted["fu_zhang"] == pytest.approx(1.0, abs=1e-5)

    def test_extreme_aridity_does_not_overflow(self):
        result = budyko(1.0, 1e200)
        assert result.aridity_index == pytest.approx(1e200)
        for values in result.predicted.values():
            assert np.isfinite(values)
            assert 0.0 <= values <= 1.0
        assert result.predicted["fu_zhang"] == pytest.approx(1.0)

    def test_fu_omega_limits(self):
        zero = budyko(1.0, 1.2, curves=("fu_zhang",), fu_omega=1.0)
        assert zero.predicted["fu_zhang"] == pytest.approx(0.0)
        perfect = budyko(1.0, 1.2, curves=("fu_zhang",), fu_omega=1e6)
        assert perfect.predicted["fu_zhang"] == pytest.approx(1.0, abs=1e-3)

    def test_fu_zhang_continuous_at_unit_aridity(self):
        for omega in (1.5, 3.0):
            at_one = 2.0 - 2.0 ** (1.0 / omega)
            below = budyko(1.0, 0.999, curves=("fu_zhang",), fu_omega=omega).predicted["fu_zhang"]
            above = budyko(1.0, 1.001, curves=("fu_zhang",), fu_omega=omega).predicted["fu_zhang"]
            assert below == pytest.approx(at_one, abs=1e-3)
            assert above == pytest.approx(at_one, abs=1e-3)

    def test_fu_zhang_matches_closed_form(self):
        grid = np.linspace(0.02, 4.0, 50)
        omega = 2.5
        result = budyko(1.0, grid, curves=("fu_zhang",), fu_omega=omega)
        closed = 1.0 + grid - (1.0 + grid**omega) ** (1.0 / omega)
        assert np.allclose(result.predicted["fu_zhang"], closed, rtol=1e-10, atol=1e-12)


class TestBudykoValidation:
    def test_non_positive_precipitation(self):
        with pytest.raises(ValueError, match="precipitation"):
            budyko(0.0, 1000.0)
        with pytest.raises(ValueError, match="positive"):
            budyko(-100.0, 1000.0)

    def test_curves_must_not_be_a_bare_string(self):
        with pytest.raises(ValueError, match="sequence"):
            budyko(1000.0, 1200.0, curves="schreiber")

    def test_non_positive_pet(self):
        with pytest.raises(ValueError, match="positive"):
            budyko(1000.0, 0.0)
        with pytest.raises(ValueError, match="positive"):
            budyko(1000.0, -50.0)

    def test_non_finite_inputs(self):
        with pytest.raises(ValueError, match="precipitation.*finite"):
            budyko(np.nan, 1000.0)
        with pytest.raises(ValueError, match="pet.*finite"):
            budyko(1000.0, np.inf)
        with pytest.raises(ValueError, match="observed_et.*finite"):
            budyko(1000.0, 1200.0, observed_et=np.nan)
        with pytest.raises(ValueError, match="observed_runoff.*finite"):
            budyko(1000.0, 1200.0, observed_runoff=np.inf)

    def test_unknown_curve(self):
        with pytest.raises(ValueError, match="Unknown Budyko curve"):
            budyko(1000.0, 1200.0, curves=("nope",))

    def test_fu_omega_below_one(self):
        with pytest.raises(ValueError, match="fu_omega"):
            budyko(1000.0, 1200.0, curves=("fu_zhang",), fu_omega=0.5)

    def test_incompatible_shapes(self):
        with pytest.raises(ValueError, match="broadcast"):
            budyko(np.array([1000.0, 2000.0]), np.array([1000.0, 2000.0, 3000.0]))

    def test_observed_et_out_of_range(self):
        with pytest.raises(ValueError, match="observed_et"):
            budyko(1000.0, 1200.0, observed_et=1500.0)
        with pytest.raises(ValueError, match="observed_et"):
            budyko(1000.0, 1200.0, observed_et=-1.0)

    def test_empty_curves(self):
        with pytest.raises(ValueError, match="at least one"):
            budyko(1000.0, 1200.0, curves=())

    def test_observed_not_broadcastable(self):
        with pytest.raises(ValueError, match="observed_et"):
            budyko(np.array([1000.0, 2000.0]), np.array([1500.0, 1000.0]),
                   observed_et=np.array([500.0, 600.0, 700.0]))


class TestBudykoZeroEtWarning:
    def test_zero_observed_et_warns(self):
        with pytest.warns(UserWarning, match="zero long-term"):
            budyko(1000.0, 1200.0, observed_et=0.0)

    def test_runoff_equals_precipitation_warns(self):
        with pytest.warns(UserWarning, match="zero long-term"):
            budyko(1000.0, 1200.0, observed_runoff=1000.0)

    def test_nonzero_et_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            budyko(1000.0, 1200.0, observed_et=600.0)
            budyko(1000.0, 1200.0, observed_runoff=400.0)


class TestBudykoObservedPosition:
    def test_observed_evaporative_ratio(self):
        result = budyko(1000.0, 1500.0, observed_et=600.0)
        assert result.observed_evaporative_ratio == pytest.approx(0.6)

    def test_deviation_signs_place_catchment(self):
        below = budyko(1000.0, 1500.0, curves=("fu_zhang",), observed_et=500.0)
        above = budyko(1000.0, 1500.0, curves=("fu_zhang",), observed_et=900.0)
        predicted = below.predicted["fu_zhang"]
        assert below.observed_deviation["fu_zhang"] == pytest.approx(0.5 - predicted)
        assert above.observed_deviation["fu_zhang"] == pytest.approx(0.9 - predicted)
        assert below.observed_deviation["fu_zhang"] < 0
        assert above.observed_deviation["fu_zhang"] > 0

    def test_no_observed_input_omits_position(self):
        result = budyko(1000.0, 1500.0)
        assert result.observed_evaporative_ratio is None
        assert result.observed_deviation is None


class TestBudykoRunoffPosition:
    def test_runoff_converted_via_water_balance(self):
        result = budyko(1000.0, 1500.0, observed_runoff=400.0)
        assert result.observed_evaporative_ratio == pytest.approx(0.6)
        assert result.observed_deviation is not None
        assert set(result.observed_deviation) == set(result.predicted)
        et = budyko(1000.0, 1500.0, observed_et=600.0)
        for kind in result.predicted:
            assert result.observed_deviation[kind] == pytest.approx(et.observed_deviation[kind])

    def test_runoff_bounds(self):
        with pytest.raises(ValueError, match="observed_runoff"):
            budyko(1000.0, 1200.0, observed_runoff=1100.0)
        with pytest.raises(ValueError, match="observed_runoff"):
            budyko(1000.0, 1200.0, observed_runoff=-1.0)

    def test_both_observed_inputs_rejected(self):
        with pytest.raises(ValueError, match="only one"):
            budyko(1000.0, 1200.0, observed_et=500.0, observed_runoff=400.0)

    def test_multi_point_runoff_supports_arrays(self):
        result = budyko(np.array([1000.0, 2000.0]), np.array([1500.0, 1000.0]),
                        observed_runoff=np.array([400.0, 700.0]))
        assert np.allclose(result.observed_evaporative_ratio, [0.6, 0.65])
        for kind, dev in result.observed_deviation.items():
            assert isinstance(dev, np.ndarray)
            assert dev.shape == (2,)

    def test_runoff_broadcast_with_scalar_precipitation(self):
        result = budyko(1000.0, 1200.0, observed_runoff=np.array([400.0, 500.0]))
        assert np.allclose(result.observed_evaporative_ratio, [0.6, 0.5])


class TestBudykoCurveFamilyGrid:
    def test_default_grid_when_observed_within_canonical_domain(self):
        result = budyko(1000.0, 1500.0, observed_et=600.0)
        assert result.aridity_grid.min() == pytest.approx(0.02)
        assert result.aridity_grid.max() == pytest.approx(4.0)
        assert result.aridity_grid.size == 300

    def test_grid_extends_to_hyper_arid_observed_point(self):
        result = budyko(1000.0, 8000.0, observed_et=600.0)
        assert result.aridity_grid.max() >= 8.0
        assert result.aridity_grid.min() == pytest.approx(0.02)

    def test_grid_extends_to_humid_observed_point(self):
        result = budyko(1000.0, 10.0, observed_et=900.0)
        assert result.aridity_grid.min() <= 0.01
        assert result.aridity_grid.max() == pytest.approx(4.0)

    def test_family_spans_extended_grid(self):
        result = budyko(1000.0, 8000.0, observed_et=600.0, curves=("schreiber",))
        assert result.curves["schreiber"].shape == result.aridity_grid.shape
        assert result.curves["schreiber"][-1] == pytest.approx(
            result.predicted["schreiber"], abs=1e-6)
