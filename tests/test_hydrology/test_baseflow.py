"""Edge-case tests for Lyne-Hollick and Eckhardt baseflow separation.

Locks in the core invariants (baseflow no greater than total, BFI between
zero and one, constant-flow steady state, parameter boundaries, index
preservation, empty input).
The ``ukih`` method is already covered in ``test_hydrology.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_discharge(years: int = 5, seed: int = 42) -> pd.Series:
    """Synthetic daily discharge with a seasonal pattern, noise, and storms."""
    rng = np.random.default_rng(seed)
    days = years * 365
    dates = pd.date_range("2010-01-01", periods=days, freq="D")

    t = np.arange(days) / 365.0
    seasonal = 20 + 15 * np.sin(2 * np.pi * t)
    noise = rng.exponential(5, days)
    storms = rng.choice([0, 0, 0, 0, 50], days) * rng.random(days)
    q = np.maximum(seasonal + noise + storms, 0.5)

    return pd.Series(q, index=dates, name="discharge")


class TestLyneHollick:
    """Edge cases for the Lyne-Hollick recursive digital filter."""

    def setup_method(self):
        self.q = _make_discharge()

    def test_baseflow_within_zero_and_total(self):
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q)
        assert (result.df["baseflow"] >= 0.0).all()
        assert (result.df["baseflow"] <= result.df["total"] + 1e-9).all()

    def test_bfi_in_unit_interval(self):
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q)
        assert 0.0 <= result.bfi <= 1.0

    def test_quickflow_is_residual(self):
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q)
        residual = result.df["total"] - result.df["baseflow"]
        pd.testing.assert_series_equal(
            result.df["quickflow"], residual, check_names=False
        )

    def test_constant_flow_baseflow_equals_flow(self):
        """Constant discharge makes quickflow decay to zero, so baseflow approximates flow."""
        from aquascope.hydrology import lyne_hollick

        idx = pd.date_range("2000-01-01", periods=40, freq="D")
        q = pd.Series(10.0, index=idx)
        result = lyne_hollick(q)
        np.testing.assert_allclose(
            result.df["baseflow"].values, q.values, rtol=1e-6
        )
        assert result.bfi == pytest.approx(1.0, abs=1e-6)

    def test_alpha_near_zero(self):
        """alpha near zero runs and keeps invariants (filter barely removes flow)."""
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q, alpha=0.001)
        assert (result.df["baseflow"] <= result.df["total"] + 1e-9).all()
        assert 0.0 <= result.bfi <= 1.0

    def test_alpha_near_one(self):
        """alpha near one runs and keeps invariants (filter removes most quickflow)."""
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q, alpha=0.999)
        assert (result.df["baseflow"] <= result.df["total"] + 1e-9).all()
        assert 0.0 <= result.bfi <= 1.0

    def test_higher_alpha_yields_less_baseflow(self):
        """Docstring invariant: higher alpha means less baseflow (lower BFI)."""
        from aquascope.hydrology import lyne_hollick

        low_alpha = lyne_hollick(self.q, alpha=0.90)
        high_alpha = lyne_hollick(self.q, alpha=0.99)
        assert high_alpha.bfi < low_alpha.bfi

    def test_output_length_and_index_preserved(self):
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(self.q)
        assert len(result.df) == len(self.q.dropna())
        assert result.df.index.equals(self.q.dropna().index)

    def test_nan_input_aligns_to_non_nan_index(self):
        """dropna() runs internally, so output aligns to the non-NaN index."""
        from aquascope.hydrology import lyne_hollick

        q = self.q.copy()
        q.iloc[5:10] = np.nan
        result = lyne_hollick(q)
        assert result.df.index.equals(q.dropna().index)
        assert len(result.df) == len(q.dropna())

    def test_empty_series(self):
        from aquascope.hydrology import lyne_hollick

        result = lyne_hollick(pd.Series([], dtype=float))
        assert len(result.df) == 0
        assert result.bfi == 0.0
        assert result.method == "lyne_hollick"


class TestEckhardt:
    """Edge cases for the Eckhardt two-parameter digital filter."""

    def setup_method(self):
        self.q = _make_discharge()

    def test_baseflow_within_zero_and_total(self):
        from aquascope.hydrology import eckhardt

        result = eckhardt(self.q)
        assert (result.df["baseflow"] >= 0.0).all()
        assert (result.df["baseflow"] <= result.df["total"] + 1e-9).all()

    def test_bfi_in_unit_interval(self):
        from aquascope.hydrology import eckhardt

        result = eckhardt(self.q)
        assert 0.0 <= result.bfi <= 1.0

    def test_quickflow_is_residual(self):
        from aquascope.hydrology import eckhardt

        result = eckhardt(self.q)
        residual = result.df["total"] - result.df["baseflow"]
        pd.testing.assert_series_equal(
            result.df["quickflow"], residual, check_names=False
        )

    def test_constant_flow_settles_at_bfi_max(self):
        """Eckhardt steady state is baseflow equal to bfi_max times q (default 0.80)."""
        from aquascope.hydrology import eckhardt

        idx = pd.date_range("2000-01-01", periods=40, freq="D")
        q = pd.Series(10.0, index=idx)
        result = eckhardt(q, bfi_max=0.80)
        np.testing.assert_allclose(
            result.df["baseflow"].values, 0.80 * q.values, rtol=1e-6
        )
        assert result.bfi == pytest.approx(0.80, abs=1e-6)

    def test_constant_flow_bfi_max_one_equals_flow(self):
        """With bfi_max = 1.0 the steady state is the full flow."""
        from aquascope.hydrology import eckhardt

        idx = pd.date_range("2000-01-01", periods=40, freq="D")
        q = pd.Series(10.0, index=idx)
        result = eckhardt(q, bfi_max=1.0)
        np.testing.assert_allclose(
            result.df["baseflow"].values, q.values, rtol=1e-6
        )
        assert result.bfi == pytest.approx(1.0, abs=1e-6)

    def test_bfi_max_near_zero(self):
        """bfi_max near zero drives BFI toward zero while keeping invariants."""
        from aquascope.hydrology import eckhardt

        result = eckhardt(self.q, bfi_max=0.01)
        assert (result.df["baseflow"] <= result.df["total"] + 1e-9).all()
        assert 0.0 <= result.bfi <= 0.05

    def test_bfi_max_near_one(self):
        """bfi_max near one drives BFI high while keeping invariants."""
        from aquascope.hydrology import eckhardt

        result = eckhardt(self.q, bfi_max=0.99)
        assert (result.df["baseflow"] <= result.df["total"] + 1e-9).all()
        assert 0.0 <= result.bfi <= 1.0

    def test_higher_bfi_max_yields_more_baseflow(self):
        """Larger bfi_max means larger BFI."""
        from aquascope.hydrology import eckhardt

        low = eckhardt(self.q, bfi_max=0.25)
        high = eckhardt(self.q, bfi_max=0.80)
        assert low.bfi < high.bfi

    def test_alpha_boundaries_keep_invariants(self):
        """Recession constant across its usual range keeps invariants."""
        from aquascope.hydrology import eckhardt

        for alpha in (0.90, 0.995):
            result = eckhardt(self.q, alpha=alpha)
            assert (result.df["baseflow"] <= result.df["total"] + 1e-9).all()
            assert 0.0 <= result.bfi <= 1.0

    def test_output_length_and_index_preserved(self):
        from aquascope.hydrology import eckhardt

        result = eckhardt(self.q)
        assert len(result.df) == len(self.q.dropna())
        assert result.df.index.equals(self.q.dropna().index)

    def test_nan_input_aligns_to_non_nan_index(self):
        """dropna() runs internally, so output aligns to the non-NaN index."""
        from aquascope.hydrology import eckhardt

        q = self.q.copy()
        q.iloc[5:10] = np.nan
        result = eckhardt(q)
        assert result.df.index.equals(q.dropna().index)
        assert len(result.df) == len(q.dropna())

    def test_empty_series(self):
        from aquascope.hydrology import eckhardt

        result = eckhardt(pd.Series([], dtype=float))
        assert len(result.df) == 0
        assert result.bfi == 0.0
        assert result.method == "eckhardt"
