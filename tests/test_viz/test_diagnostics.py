"""Tests for aquascope.viz.diagnostics and cross-validation utilities.

Uses matplotlib's non-interactive Agg backend so tests run headless in CI.
"""

from __future__ import annotations

import os

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scipy.stats import genextreme, pearson3  # noqa: E402
from string import alphabets # noqa: E402
from tempfile import TemporaryFile
from aquascope.hydrology.flood_frequency import FloodFreqResult, coverage_probability, leave_one_out_cv  # noqa: E402
from aquascope.viz.diagnostics import diagnostic_panel, pp_plot, qq_plot, return_level_plot, double_mass_plot  # noqa: E402


def _make_gev_data(size: int = 40, seed: int = 42) -> np.ndarray:
    """Generate realistic flood data from a GEV distribution."""
    return genextreme.rvs(c=-0.1, loc=100, scale=20, size=size, random_state=seed)


def _make_annual_max_series(size: int = 40, seed: int = 42) -> pd.Series:
    """Return a pd.Series with annual DatetimeIndex for cross-validation tests."""
    data = _make_gev_data(size=size, seed=seed)
    idx = pd.date_range("1980-01-01", periods=size, freq="YS")
    return pd.Series(data, index=idx)


class TestQQPlot:
    """Tests for qq_plot."""

    def setup_method(self):
        self.data = _make_gev_data()
        self.params = genextreme.fit(self.data)

    def test_qq_gev(self):
        fig = qq_plot(self.data, "gev", self.params)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qq_lp3(self):
        log_data = np.log10(self.data[self.data > 0])
        params = pearson3.fit(log_data)
        fig = qq_plot(self.data, "lp3", params)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_qq_save(self):
        save_path = "test_qq_output.png"
        try:
            fig = qq_plot(self.data, "gev", self.params, save_path=save_path)
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_qq_custom_ax(self):
        fig_outer, ax = plt.subplots()
        fig = qq_plot(self.data, "gev", self.params, ax=ax)
        assert fig is fig_outer
        plt.close(fig)


class TestPPPlot:
    """Tests for pp_plot."""

    def setup_method(self):
        self.data = _make_gev_data()
        self.params = genextreme.fit(self.data)

    def test_pp_gev(self):
        fig = pp_plot(self.data, "gev", self.params)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pp_values_between_0_1(self):
        fig = pp_plot(self.data, "gev", self.params)
        ax = fig.axes[0]
        # All scatter points should be within [0, 1]
        for coll in ax.collections:
            offsets = coll.get_offsets()
            assert np.all(offsets >= -0.05), "P-P values should be ≥ 0"
            assert np.all(offsets <= 1.05), "P-P values should be ≤ 1"
        plt.close(fig)

class TestDBMPlot:
    """ Execute unit tests across features in the double_mass_plot function """
    def __init__(self, n_rows:int, n_cols:int, val_min:int, val_max:int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.val_min = val_min
        self.val_max = val_max
        self.col_names = list(string.ascii_uppercase[:self.n_cols]) # label the columns of dataset
        self.obs = {col: np.random.uniform(low=val_min, high=val_max, size=(self.n_rows, self.n_cols))} # use random dataset
        self.pivot = random.sample(self.col_names, 2) # choose columns to plot
    
    # be sure that indexing through array isolates the columns 
    def _test_feat_selection():
        feat_A, feat_B = self.pivot
        # Extract the specific slices
        numeric_feats = [self.obs[feat_A], self.obs[feat_B]]
        # Validate that we successfully isolated exactly two arrays matching our pivots
        self.assertEqual(len(numeric_feats), 2)
        self.assertEqual(feat_A, self.pivot[0])
        self.assertEqual(feat_B, self.pivot[1])

    # is the cummulation score greater than the parts
    def test_cumm():
        feat_A, feat_B = self.pivot
        
        cumm_featA = np.cumsum(self.obs[feat_A])
        cumm_featB = np.cumsum(self.obs[feat_B])
        
        # Testing that cumulative values at the end are greater than early values (for positive data)
        self.assertGreater(cumm_featA[-1], cumm_featA[0])
        self.assertGreater(cumm_featB[-1], cumm_featB[0])

    def test_plot_generation(N=2):
        # assert N number of curves are displayed on the plot
        try:
            fp = tempfile.TemporaryFile()
            _ = double_mass_plot(observations=self.obs, 
                                pivots=self.pivot, 
                                save_path=fp
                                title = 'test double mass plot')
    
            # Assert 1 curve/line is displayed on the plot
            self.assertEqual(len(ax.lines), 1)
            
            # Be sure file is written to specified location
            tmp_file.seek(0, 2) # Move cursor to the end of the file
            self.assertGreater(tmp_file.tell(), 0) # File size should be > 0 bytes
            
            plt.close(fig) # Clean up plot memory
            
        finally:
            # Explicitly close the file to trigger automatic filesystem deletion
            tmp_file.close()

        fp = tempfile.TemporaryFile()
        fp.write(b'Hello world!')
        # read data from file
        fp.seek(0)
        fp.read()

        # OUTPUT: 'b'Hello world!''

        # close the file, it will be removed

        fp.close()
        # test the plot is save in a specified location
        # test the plot is displayed if the 

class TestReturnLevelPlot:
    """Tests for return_level_plot."""

    def _make_result(self) -> FloodFreqResult:
        data = _make_gev_data()
        params = genextreme.fit(data)
        rps = [2, 5, 10, 25, 50, 100]
        rp_map = {}
        ci_map = {}
        for rp in rps:
            prob = 1 - 1.0 / rp
            q = float(genextreme.ppf(prob, *params))
            rp_map[rp] = q
            ci_map[rp] = (q * 0.85, q * 1.15)
        return FloodFreqResult(
            return_periods=rp_map,
            distribution="GEV",
            params=params,
            annual_max=pd.Series(data),
            confidence_intervals=ci_map,
        )

    def test_return_level_basic(self):
        result = self._make_result()
        fig = return_level_plot(result, ci=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_return_level_with_ci(self):
        result = self._make_result()
        fig = return_level_plot(result, ci=True)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1
        plt.close(fig)


class TestDiagnosticPanel:
    """Tests for diagnostic_panel."""

    def setup_method(self):
        self.data = _make_gev_data()
        self.params = genextreme.fit(self.data)

    def test_panel_creates_4_subplots(self):
        fig = diagnostic_panel(self.data, "gev", self.params)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_panel_save(self):
        save_path = "test_panel_output.png"
        try:
            fig = diagnostic_panel(self.data, "gev", self.params, save_path=save_path)
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)


class TestCrossValidation:
    """Tests for leave_one_out_cv and coverage_probability."""

    def setup_method(self):
        self.series = _make_annual_max_series()

    def test_loo_cv_gev(self):
        result = leave_one_out_cv(self.series, distribution="gev")
        assert result["rmse"] > 0
        assert len(result["predictions"]) == len(result["observations"])
        assert len(result["predictions"]) == len(self.series)

    def test_loo_cv_lp3(self):
        result = leave_one_out_cv(self.series, distribution="lp3")
        assert result["rmse"] > 0
        assert "bias" in result
        assert "mae" in result

    def test_coverage_probability(self):
        # Use a small series, few splits, and few bootstraps to keep the test fast
        small_series = _make_annual_max_series(size=20, seed=42)
        cov = coverage_probability(
            small_series, distribution="gev", ci_level=0.90, n_splits=4, n_boot=50,
        )
        assert 0.0 <= cov <= 1.0
