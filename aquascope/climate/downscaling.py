"""
Statistical downscaling and bias-correction methods.

Implements delta-change, quantile mapping, and quantile-delta mapping
for adjusting GCM output to local observed scales.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ── Result dataclasses ──────────────────────────────────────────────────
@dataclass
class DownscalingMetrics:
    """Evaluation metrics for a downscaled / bias-corrected series.

    Attributes
    ----------
    rmse : float
        Root mean squared error.
    mae : float
        Mean absolute error.
    bias : float
        Mean bias (downscaled − observed).
    correlation : float
        Pearson correlation coefficient.
    ks_statistic : float
        Kolmogorov–Smirnov test statistic.
    ks_pvalue : float
        p-value from the KS test.
    percentile_errors : dict[int, float]
        Absolute error at selected percentiles (5, 25, 50, 75, 95).
    """

    rmse: float
    mae: float
    bias: float
    correlation: float
    ks_statistic: float
    ks_pvalue: float
    percentile_errors: dict[int, float]


# ── Public functions ────────────────────────────────────────────────────
def delta_method(
    obs: pd.Series,
    gcm_hist: pd.Series,
    gcm_future: pd.Series,
    method: str = "additive",
) -> pd.Series:
    """Apply the delta-change method to project future observed values.

    Parameters
    ----------
    obs : pd.Series
        Observed time series.
    gcm_hist : pd.Series
        GCM output for the historical period.
    gcm_future : pd.Series
        GCM output for the future period.
    method : str
        ``"additive"`` or ``"multiplicative"``.

    Returns
    -------
    pd.Series
        Projected series with the same index as *obs*.

    Raises
    ------
    ValueError
        If *method* is not ``"additive"`` or ``"multiplicative"``.
    """
    if method == "additive":
        delta = gcm_future.mean() - gcm_hist.mean()
        return obs + delta
    if method == "multiplicative":
        if gcm_hist.mean() == 0:
            raise ValueError("Cannot use multiplicative method: gcm_hist mean is zero")
        factor = gcm_future.mean() / gcm_hist.mean()
        return obs * factor
    raise ValueError(f"method must be 'additive' or 'multiplicative', got {method!r}")


def quantile_mapping(
    obs: pd.Series,
    gcm_hist: pd.Series,
    gcm_future: pd.Series,
    n_quantiles: int = 100,
) -> pd.Series:
    """Empirical quantile mapping using linear interpolation.

    Maps each GCM-future value to the observed distribution by matching
    its quantile in the GCM-historical distribution.

    Parameters
    ----------
    obs : pd.Series
        Observed time series.
    gcm_hist : pd.Series
        GCM output for the historical period.
    gcm_future : pd.Series
        GCM output for the future period.
    n_quantiles : int
        Number of quantile bins (default 100).

    Returns
    -------
    pd.Series
        Bias-corrected future series with the same index as *gcm_future*.
    """
    quantiles = np.linspace(0, 1, n_quantiles + 1)
    hist_q = np.quantile(gcm_hist.dropna().values, quantiles)
    obs_q = np.quantile(obs.dropna().values, quantiles)

    corrected = np.interp(gcm_future.values, hist_q, obs_q)
    return pd.Series(corrected, index=gcm_future.index, name="qm_corrected")


def quantile_delta_mapping(
    obs: pd.Series,
    gcm_hist: pd.Series,
    gcm_future: pd.Series,
    n_quantiles: int = 100,
) -> pd.Series:
    """Quantile delta mapping (Cannon et al., 2015).

    Preserves the relative change signal in each quantile while
    mapping to the observed distribution.

    Parameters
    ----------
    obs : pd.Series
        Observed time series.
    gcm_hist : pd.Series
        GCM output for the historical period.
    gcm_future : pd.Series
        GCM output for the future period.
    n_quantiles : int
        Number of quantile bins (default 100).

    Returns
    -------
    pd.Series
        Bias-corrected future series preserving relative changes.
    """
    quantiles = np.linspace(0, 1, n_quantiles + 1)
    hist_q = np.quantile(gcm_hist.dropna().values, quantiles)
    obs_q = np.quantile(obs.dropna().values, quantiles)

    future_vals = gcm_future.values.astype(float)
    corrected = np.empty_like(future_vals)

    for i, val in enumerate(future_vals):
        # Determine the quantile of this future value in the historical CDF
        tau = np.searchsorted(np.sort(gcm_hist.dropna().values), val) / len(gcm_hist.dropna())
        tau = np.clip(tau, 0, 1)

        # Corresponding historical quantile value
        hist_val = np.interp(tau, quantiles, hist_q)

        # Additive delta between future and historical at this quantile
        delta = val - hist_val

        # Map tau to the observed distribution and apply the delta
        obs_val = np.interp(tau, quantiles, obs_q)
        corrected[i] = obs_val + delta

    return pd.Series(corrected, index=gcm_future.index, name="qdm_corrected")


def bias_correction(
    gcm: pd.Series,
    obs: pd.Series,
    method: str = "quantile_mapping",
) -> pd.Series:
    """Convenience wrapper for bias-correcting a GCM series.

    Splits *gcm* in half — first half as "historical", second as "future" —
    and applies the requested correction.  For finer control, call the
    individual functions directly.

    Parameters
    ----------
    gcm : pd.Series
        Full GCM time series to correct.
    obs : pd.Series
        Observed time series for calibration.
    method : str
        ``"quantile_mapping"`` (default), ``"quantile_delta_mapping"``, or
        ``"delta"``.

    Returns
    -------
    pd.Series
        Bias-corrected series.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    mid = len(gcm) // 2
    gcm_hist = gcm.iloc[:mid]
    gcm_future = gcm.iloc[mid:]

    if method == "quantile_mapping":
        return quantile_mapping(obs, gcm_hist, gcm_future)
    if method == "quantile_delta_mapping":
        return quantile_delta_mapping(obs, gcm_hist, gcm_future)
    if method == "delta":
        return delta_method(obs, gcm_hist, gcm_future)
    raise ValueError(f"Unknown bias-correction method: {method!r}")


def evaluate_downscaling(obs: pd.Series, downscaled: pd.Series) -> DownscalingMetrics:
    """Evaluate a downscaled series against observations.

    Parameters
    ----------
    obs : pd.Series
        Observed reference values.
    downscaled : pd.Series
        Downscaled / bias-corrected values (same length as *obs*).

    Returns
    -------
    DownscalingMetrics
        Suite of error and distributional metrics.

    Raises
    ------
    ValueError
        If *obs* and *downscaled* have different lengths.
    """
    o = obs.values.astype(float)
    d = downscaled.values.astype(float)

    if len(o) != len(d):
        raise ValueError(f"Length mismatch: obs={len(o)}, downscaled={len(d)}")

    rmse = float(np.sqrt(np.mean((d - o) ** 2)))
    mae = float(np.mean(np.abs(d - o)))
    bias = float(np.mean(d - o))
    correlation = float(np.corrcoef(o, d)[0, 1])

    ks_stat, ks_p = stats.ks_2samp(o, d)

    pct_errors: dict[int, float] = {}
    for p in (5, 25, 50, 75, 95):
        pct_errors[p] = float(abs(np.percentile(d, p) - np.percentile(o, p)))

    return DownscalingMetrics(
        rmse=rmse,
        mae=mae,
        bias=bias,
        correlation=correlation,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_p),
        percentile_errors=pct_errors,
    )
