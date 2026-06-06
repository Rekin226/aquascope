"""Extreme-value and flood/drought frequency analysis.

Block-maxima frequency analysis for hydrological extremes. Fits the
Generalised Extreme Value (GEV), Log-Pearson Type III (LP3) and Gumbel
distributions to a series of annual maxima and estimates return levels
(design magnitudes) for a set of return periods, with parametric-bootstrap
confidence bounds.

The three public functions in this module are intentionally left without
type annotations — adding them (see issue #8) is a self-contained,
documentation-only improvement. Their structured return types already live
in :mod:`aquascope.models.analysis`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from aquascope.models.analysis import (
    DistributionFit,
    GEVParameters,
    ReturnPeriodResult,
)

logger = logging.getLogger(__name__)

DEFAULT_RETURN_PERIODS: tuple[float, ...] = (2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
_SUPPORTED = ("gev", "lp3", "gumbel")


def _annual_maxima(series: pd.Series) -> np.ndarray:
    """Reduce a series to a 1-D array of block (annual) maxima.

    If *series* has a :class:`~pandas.DatetimeIndex` it is resampled to
    yearly maxima; otherwise the values are treated as block maxima already.
    NaNs are dropped.
    """
    s = series.dropna()
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.resample("YE").max().dropna()
    data: np.ndarray = np.asarray(s, dtype=float)
    if data.size < 3:
        raise ValueError(
            f"need at least 3 block maxima for frequency analysis, got {data.size}"
        )
    return data


def _fit_params(data: np.ndarray, distribution: str) -> tuple[float, ...]:
    """Fit raw SciPy parameters for the requested distribution."""
    if distribution == "gev":
        raw = stats.genextreme.fit(data)
    elif distribution == "gumbel":
        raw = stats.gumbel_r.fit(data)
    elif distribution == "lp3":
        raw = stats.pearson3.fit(np.log10(data))
    else:
        raise ValueError(
            f"unknown distribution {distribution!r}; expected one of {_SUPPORTED}"
        )
    return tuple(float(p) for p in raw)


def _ppf(distribution: str, prob, params) -> np.ndarray:
    """Inverse CDF (quantile) for *prob* under the fitted *params*."""
    if distribution == "gev":
        q = stats.genextreme.ppf(prob, *params)
    elif distribution == "gumbel":
        q = stats.gumbel_r.ppf(prob, *params)
    else:
        # LP3 was fitted in log10 space.
        q = np.power(10.0, stats.pearson3.ppf(prob, *params))
    result: np.ndarray = np.asarray(q, dtype=float)
    return result


def _logpdf_sum(distribution: str, data: np.ndarray, params) -> float:
    """Total log-likelihood of *data* under the fitted distribution."""
    if distribution == "gev":
        return float(np.sum(stats.genextreme.logpdf(data, *params)))
    if distribution == "gumbel":
        return float(np.sum(stats.gumbel_r.logpdf(data, *params)))
    log_data = np.log10(data)
    # Jacobian of the log10 transform: d/dx log10(x) = 1/(x ln 10).
    jac = -np.log(data * np.log(10.0))
    return float(np.sum(stats.pearson3.logpdf(log_data, *params) + jac))


def compute_gev_parameters(data, method="mle"):
    """Fit a Generalised Extreme Value distribution to block maxima.

    Args:
        data: Block (annual) maxima as a :class:`~pandas.Series` or array.
        method: Estimation method label (only ``"mle"`` is implemented).

    Returns:
        The fitted GEV parameters in classic hydrology sign convention.
    """
    arr = _annual_maxima(data) if isinstance(data, pd.Series) else np.asarray(data, float)
    c, loc, scale = stats.genextreme.fit(arr)
    # SciPy's ``c`` is the negative of the classic hydrology shape parameter.
    return GEVParameters(shape=float(-c), location=float(loc), scale=float(scale), method=method)


def fit_distribution(series, distribution="gev"):
    """Fit one extreme-value distribution and score its goodness of fit.

    Args:
        series: Hydrological series (datetime-indexed) or block maxima.
        distribution: One of ``"gev"``, ``"lp3"`` or ``"gumbel"``.

    Returns:
        A goodness-of-fit summary including AIC and a KS p-value.
    """
    data = _annual_maxima(series)
    params = _fit_params(data, distribution)

    k = len(params)
    aic = 2 * k - 2 * _logpdf_sum(distribution, data, params)

    if distribution == "gev":
        ks = stats.kstest(data, "genextreme", args=params)
        named = {"shape": float(-params[0]), "location": float(params[1]), "scale": float(params[2])}
    elif distribution == "gumbel":
        ks = stats.kstest(data, "gumbel_r", args=params)
        named = {"location": float(params[0]), "scale": float(params[1])}
    else:
        ks = stats.kstest(np.log10(data), "pearson3", args=params)
        named = {"skew": float(params[0]), "location": float(params[1]), "scale": float(params[2])}

    return DistributionFit(
        distribution=distribution,
        parameters=named,
        aic=float(aic),
        ks_pvalue=float(ks.pvalue),
        n_samples=int(data.size),
    )


def estimate_return_periods(
    series,
    distribution="gev",
    return_periods=DEFAULT_RETURN_PERIODS,
    confidence_level=0.95,
    n_bootstrap=500,
    random_state=42,
):
    """Estimate return levels with parametric-bootstrap confidence bounds.

    For each return period ``T`` the non-exceedance probability is
    ``p = 1 - 1/T`` and the return level is the corresponding quantile of the
    fitted distribution. Confidence bounds come from refitting the
    distribution to *n_bootstrap* synthetic samples drawn from the fit.

    Args:
        series: Hydrological series (datetime-indexed) or block maxima.
        distribution: One of ``"gev"``, ``"lp3"`` or ``"gumbel"``.
        return_periods: Return periods ``T`` in years.
        confidence_level: Two-sided confidence level for the bounds.
        n_bootstrap: Number of parametric-bootstrap resamples.
        random_state: Seed for reproducible bootstrap sampling.

    Returns:
        Return levels and confidence bounds for every requested period.
    """
    data = _annual_maxima(series)
    fit = fit_distribution(series, distribution=distribution)
    params = _fit_params(data, distribution)

    periods = np.asarray(return_periods, dtype=float)
    probs = 1.0 - 1.0 / periods
    levels = _ppf(distribution, probs, params)

    rng = np.random.default_rng(random_state)
    n = data.size
    boot = np.empty((n_bootstrap, periods.size), dtype=float)
    for i in range(n_bootstrap):
        sample = _ppf(distribution, rng.uniform(size=n), params)
        sample = np.clip(sample, a_min=1e-9, a_max=None)
        try:
            bparams = _fit_params(sample, distribution)
            boot[i] = _ppf(distribution, probs, bparams)
        except Exception:  # noqa: BLE001 - a degenerate resample just gets reused
            boot[i] = levels

    alpha = (1.0 - confidence_level) / 2.0
    lower = np.nanpercentile(boot, 100 * alpha, axis=0)
    upper = np.nanpercentile(boot, 100 * (1.0 - alpha), axis=0)

    return ReturnPeriodResult(
        distribution=distribution,
        return_periods=[float(t) for t in periods],
        return_levels=[float(x) for x in levels],
        lower_bound=[float(x) for x in lower],
        upper_bound=[float(x) for x in upper],
        confidence_level=float(confidence_level),
        fit=fit,
    )
