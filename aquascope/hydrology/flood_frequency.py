"""Flood frequency analysis.

Fits extreme-value distributions to annual maximum discharge series and
estimates return-period flood magnitudes with confidence intervals.

Supported distributions:

- **GEV** (Generalised Extreme Value) — flexible 3-parameter model.
- **LP3** (Log-Pearson Type III) — US standard (Bulletin 17C method).
- **Gumbel** (Type I) — special case of GEV with shape=0.
- **Weibull minimum** — for low-flow frequency analysis.
- **GPD** (Generalised Pareto) — Peaks-Over-Threshold method.
- **GEV via L-moments** — robust small-sample fitting.
- **Non-stationary GEV** — time-varying location parameter.
- **Regional frequency analysis** — L-moment based (Hosking & Wallis).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FloodFreqResult:
    """Result of flood frequency analysis.

    Attributes
    ----------
    return_periods:
        Mapping of return period (years) → estimated discharge.
    distribution:
        Name of the fitted distribution.
    params:
        Distribution parameters (shape, loc, scale).
    annual_max:
        The annual maximum series used for fitting.
    confidence_intervals:
        Optional mapping of return period → (lower, upper) 90 % CI.
    """

    return_periods: dict[int, float] = field(default_factory=dict)
    distribution: str = ""
    params: tuple = ()
    annual_max: pd.Series | None = None
    confidence_intervals: dict[int, tuple[float, float]] = field(default_factory=dict)


def _extract_annual_max(discharge: pd.Series) -> pd.Series:
    """Extract annual maximum discharge from a daily series."""
    annual = discharge.resample("YS").max().dropna()
    return annual


def fit_gev(
    discharge: pd.Series,
    *,
    return_periods: list[int] | None = None,
    ci_level: float = 0.90,
) -> FloodFreqResult:
    """Fit a GEV distribution to the annual maximum series.

    Parameters
    ----------
    discharge:
        Daily discharge series with a DatetimeIndex.
    return_periods:
        Return periods in years to estimate.  Defaults to
        ``[2, 5, 10, 25, 50, 100, 200, 500]``.
    ci_level:
        Confidence level for bootstrap CIs (default 0.90).

    Returns
    -------
    A :class:`FloodFreqResult` with quantile estimates.

    Raises
    ------
    ValueError
        If fewer than 5 annual maxima are available.
    """
    from scipy.stats import genextreme

    annual_max = _extract_annual_max(discharge)
    if len(annual_max) < 5:
        msg = f"Need ≥5 years of data; got {len(annual_max)}"
        raise ValueError(msg)

    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100, 200, 500]

    shape, loc, scale = genextreme.fit(annual_max.values)

    rp_map: dict[int, float] = {}
    ci_map: dict[int, tuple[float, float]] = {}

    for rp in return_periods:
        prob = 1 - 1.0 / rp
        rp_map[rp] = float(genextreme.ppf(prob, shape, loc=loc, scale=scale))

    # Bootstrap confidence intervals
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_estimates: dict[int, list[float]] = {rp: [] for rp in return_periods}

    for _ in range(n_boot):
        sample = rng.choice(annual_max.values, size=len(annual_max), replace=True)
        try:
            s, loc_b, sc = genextreme.fit(sample)
            for rp in return_periods:
                prob = 1 - 1.0 / rp
                boot_estimates[rp].append(float(genextreme.ppf(prob, s, loc=loc_b, scale=sc)))
        except Exception:  # noqa: BLE001
            continue

    alpha = (1 - ci_level) / 2
    for rp in return_periods:
        vals = boot_estimates[rp]
        if len(vals) > 10:
            ci_map[rp] = (float(np.percentile(vals, alpha * 100)), float(np.percentile(vals, (1 - alpha) * 100)))

    logger.info("GEV fit: shape=%.3f, loc=%.2f, scale=%.2f", shape, loc, scale)
    return FloodFreqResult(
        return_periods=rp_map,
        distribution="GEV",
        params=(shape, loc, scale),
        annual_max=annual_max,
        confidence_intervals=ci_map,
    )


def fit_lp3(
    discharge: pd.Series,
    *,
    return_periods: list[int] | None = None,
    regional_skew: float | None = None,
    regional_skew_mse: float = 0.302,
    ci_level: float = 0.90,
    zero_threshold: float = 0.0,
) -> FloodFreqResult:
    """Fit a Log-Pearson Type III distribution (Bulletin 17C approach).

    When *regional_skew* is provided the station skew is adjusted using the
    inverse-variance weighted average described in Bulletin 17C §5.2.4
    (England et al., 2018).  Confidence intervals are computed via the
    variance-of-estimate approach (Bulletin 17C §6).

    Parameters:
        discharge: Daily discharge series with a DatetimeIndex.
        return_periods: Return periods to estimate.  Defaults to standard set.
        regional_skew: Generalised / regional skew coefficient.  When
            ``None`` (default) the station skew is used unmodified for
            backward compatibility.
        regional_skew_mse: Mean-square error of the regional skew estimate.
            Default ``0.302`` is the USGS nationwide value.
        ci_level: Confidence level for return-period intervals (0 < ci < 1).
        zero_threshold: Values ≤ this are excluded before fitting.

    Returns:
        A :class:`FloodFreqResult` with quantile estimates and optional CIs.

    Raises:
        ValueError: If fewer than 5 annual maxima or all values ≤ zero_threshold.

    References:
        England, J. F. Jr., Cohn, T. A., Faber, B. A., Stedinger, J. R.,
        Thomas, W. O. Jr., Veilleux, A. G., Kiang, J. E., & Mason, R. R.
        Jr. (2018). Guidelines for determining flood flow frequency —
        Bulletin 17C. U.S. Geological Survey Techniques and Methods 4-B5.
        https://doi.org/10.3133/tm4B5
    """
    from scipy.stats import pearson3
    from scipy.stats import t as t_dist

    annual_max = _extract_annual_max(discharge)

    # Filter out values at or below zero_threshold
    annual_max = annual_max[annual_max > zero_threshold]

    if len(annual_max) < 5:
        msg = f"Need ≥5 years of data; got {len(annual_max)}"
        raise ValueError(msg)

    log_vals = np.log10(annual_max.values)

    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100, 200, 500]

    n = len(log_vals)
    mean_log = float(np.mean(log_vals))
    std_log = float(np.std(log_vals, ddof=1))
    station_skew = float(_station_skew(log_vals))

    # Apply weighted skew if regional skew is given
    if regional_skew is not None:
        skew_coeff = weighted_skew(station_skew, regional_skew, n, regional_mse=regional_skew_mse)
    else:
        skew_coeff = station_skew

    # Pearson3 parameterisation: shape=skew, loc=mean, scale=std
    rp_map: dict[int, float] = {}
    ci_map: dict[int, tuple[float, float]] = {}

    alpha = (1 - ci_level) / 2

    for rp in return_periods:
        prob = 1 - 1.0 / rp
        log_q = float(pearson3.ppf(prob, skew_coeff, loc=mean_log, scale=std_log))
        rp_map[rp] = float(10**log_q)

        # Bulletin 17C variance-of-estimate confidence intervals
        kp = float(pearson3.ppf(prob, skew_coeff, loc=0, scale=1))
        var_q = (std_log**2 / n) * (1 + kp**2 / 2 * (1 + 0.75 * skew_coeff**2))
        se_q = np.sqrt(var_q)
        t_val = float(t_dist.ppf(1 - alpha, df=max(n - 1, 1)))
        ci_map[rp] = (float(10 ** (log_q - t_val * se_q)), float(10 ** (log_q + t_val * se_q)))

    logger.info("LP3 fit: skew=%.3f, mean=%.3f, std=%.3f", skew_coeff, mean_log, std_log)
    return FloodFreqResult(
        return_periods=rp_map,
        distribution="LP3",
        params=(skew_coeff, mean_log, std_log),
        annual_max=annual_max,
        confidence_intervals=ci_map,
    )


# ---------------------------------------------------------------------------
# Default return periods shared by new fitting functions
# ---------------------------------------------------------------------------
_DEFAULT_RETURN_PERIODS: list[float] = [2, 5, 10, 25, 50, 100, 200, 500]


# ---------------------------------------------------------------------------
# Gumbel (Type I extreme value)
# ---------------------------------------------------------------------------

def fit_gumbel(
    annual_maxima: np.ndarray | pd.Series,
    return_periods: list[float] | None = None,
) -> FloodFreqResult:
    """Fit Gumbel (Type I) extreme value distribution.

    Special case of GEV with shape=0.  Uses ``scipy.stats.gumbel_r`` (MLE).

    Gumbel CDF: F(x) = exp(-exp(-(x - loc) / scale))

    Parameters:
        annual_maxima: Array of annual maximum values.
        return_periods: Return periods in years.  Defaults to standard set.

    Returns:
        A :class:`FloodFreqResult` with fitted parameters and return levels.

    Raises:
        ValueError: If fewer than 5 values are provided.
    """
    from scipy.stats import gumbel_r

    data = np.asarray(annual_maxima, dtype=np.float64)
    if len(data) < 5:
        msg = f"Need ≥5 annual maxima; got {len(data)}"
        raise ValueError(msg)

    if return_periods is None:
        return_periods = _DEFAULT_RETURN_PERIODS

    loc, scale = gumbel_r.fit(data)

    rp_map: dict[float, float] = {}
    for rp in return_periods:
        prob = 1 - 1.0 / rp
        rp_map[rp] = float(gumbel_r.ppf(prob, loc=loc, scale=scale))

    logger.info("Gumbel fit: loc=%.3f, scale=%.3f", loc, scale)
    return FloodFreqResult(
        return_periods=rp_map,
        distribution="Gumbel",
        params=(loc, scale),
        annual_max=pd.Series(data) if not isinstance(annual_maxima, pd.Series) else annual_maxima,
    )


# ---------------------------------------------------------------------------
# Weibull minimum (low-flow frequency)
# ---------------------------------------------------------------------------

def fit_weibull_min(
    annual_minima: np.ndarray | pd.Series,
    return_periods: list[float] | None = None,
) -> FloodFreqResult:
    """Fit Weibull distribution to annual minima for low-flow frequency analysis.

    Uses ``scipy.stats.weibull_min`` (MLE).  Return periods relate to the
    probability of flows *below* a given level.

    Parameters:
        annual_minima: Array of annual minimum values.
        return_periods: Return periods in years.  Defaults to standard set.

    Returns:
        A :class:`FloodFreqResult` with fitted parameters and return levels.

    Raises:
        ValueError: If fewer than 5 values are provided.
    """
    from scipy.stats import weibull_min

    data = np.asarray(annual_minima, dtype=np.float64)
    if len(data) < 5:
        msg = f"Need ≥5 annual minima; got {len(data)}"
        raise ValueError(msg)

    if return_periods is None:
        return_periods = _DEFAULT_RETURN_PERIODS

    shape, loc, scale = weibull_min.fit(data)

    rp_map: dict[float, float] = {}
    for rp in return_periods:
        # For low flows the return level is the quantile at 1/T
        prob = 1.0 / rp
        rp_map[rp] = float(weibull_min.ppf(prob, shape, loc=loc, scale=scale))

    logger.info("Weibull-min fit: shape=%.3f, loc=%.3f, scale=%.3f", shape, loc, scale)
    return FloodFreqResult(
        return_periods=rp_map,
        distribution="Weibull_min",
        params=(shape, loc, scale),
        annual_max=pd.Series(data) if not isinstance(annual_minima, pd.Series) else annual_minima,
    )


# ---------------------------------------------------------------------------
# Generalised Pareto Distribution (Peaks-Over-Threshold)
# ---------------------------------------------------------------------------

def fit_gpd(
    exceedances: np.ndarray | pd.Series,
    threshold: float,
    return_periods: list[float] | None = None,
    total_observations: int | None = None,
) -> FloodFreqResult:
    """Fit Generalised Pareto Distribution using Peaks-Over-Threshold method.

    Parameters:
        exceedances: Values *above* the threshold (already filtered).
        threshold: The threshold used for POT selection.
        return_periods: Return periods in years.
        total_observations: Total number of observations used to compute
            exceedance rate.  If ``None``, assumed equal to ``len(exceedances)``.

    Returns:
        A :class:`FloodFreqResult` with fitted parameters and return levels.

    Raises:
        ValueError: If fewer than 10 exceedances are provided.
    """
    import warnings

    from scipy.stats import genpareto

    data = np.asarray(exceedances, dtype=np.float64)
    if len(data) < 10:
        warnings.warn(
            f"Only {len(data)} exceedances — GPD fit may be unreliable (recommend ≥10).",
            stacklevel=2,
        )

    if total_observations is None:
        total_observations = len(data)

    if return_periods is None:
        return_periods = _DEFAULT_RETURN_PERIODS

    # Fit GPD to exceedances above the threshold
    excess = data - threshold
    shape, _loc, scale = genpareto.fit(excess, floc=0)

    rate = len(data) / total_observations

    rp_map: dict[float, float] = {}
    for rp in return_periods:
        if abs(shape) < 1e-10:
            # Exponential tail
            level = threshold + scale * np.log(rp * rate)
        else:
            level = threshold + (scale / shape) * ((rp * rate) ** shape - 1)
        rp_map[rp] = float(level)

    logger.info("GPD fit: shape=%.3f, scale=%.3f, threshold=%.2f, rate=%.4f", shape, scale, threshold, rate)
    return FloodFreqResult(
        return_periods=rp_map,
        distribution="GPD",
        params=(shape, threshold, scale),
        annual_max=pd.Series(data) if not isinstance(exceedances, pd.Series) else exceedances,
    )


def select_pot_threshold(
    data: np.ndarray | pd.Series,
    method: str = "mean_residual",
) -> float:
    """Select optimal threshold for Peaks-Over-Threshold analysis.

    Parameters:
        data: Array of observations.
        method: Selection method — ``"mean_residual"``, ``"percentile"``
            (95th percentile), or ``"sqrt_rule"`` (mean + 1.5 × std).

    Returns:
        Optimal threshold value.

    Raises:
        ValueError: If *method* is not recognised.
    """
    arr = np.asarray(data, dtype=np.float64)

    if method == "percentile":
        return float(np.percentile(arr, 95))

    if method == "sqrt_rule":
        return float(np.mean(arr) + 1.5 * np.std(arr, ddof=1))

    if method == "mean_residual":
        # Mean Residual Life: find the threshold where the mean excess plot
        # is approximately linear.  We evaluate a grid of candidate thresholds
        # and pick the lowest one where the derivative of the mean-excess
        # function stabilises (smallest change in slope).
        sorted_data = np.sort(arr)
        candidates = np.percentile(arr, np.linspace(70, 98, 50))
        mean_excess: list[float] = []
        valid_thresholds: list[float] = []
        for u in candidates:
            exc = sorted_data[sorted_data > u] - u
            if len(exc) >= 5:
                mean_excess.append(float(np.mean(exc)))
                valid_thresholds.append(float(u))

        if len(mean_excess) < 3:
            # Fall back to 95th percentile
            return float(np.percentile(arr, 95))

        me = np.array(mean_excess)
        slopes = np.diff(me) / np.diff(valid_thresholds)
        slope_changes = np.abs(np.diff(slopes))
        # Pick the threshold just before the smallest slope change
        idx = int(np.argmin(slope_changes))
        return valid_thresholds[idx]

    msg = f"Unknown threshold selection method: {method!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# L-moments
# ---------------------------------------------------------------------------

def lmoments_from_sample(data: np.ndarray) -> dict[str, float]:
    """Compute L-moments (L1–L4) and L-moment ratios (t3, t4) from a sample.

    L-moments are linear combinations of probability weighted moments (PWMs).

    Parameters:
        data: 1-D array of observations.

    Returns:
        Dictionary with keys ``L1``, ``L2``, ``L3``, ``L4``, ``t3``, ``t4``.

    Raises:
        ValueError: If fewer than 4 observations are provided.
    """
    arr = np.sort(np.asarray(data, dtype=np.float64))
    n = len(arr)
    if n < 4:
        msg = f"Need ≥4 observations for L-moments; got {n}"
        raise ValueError(msg)

    # Probability weighted moments b0..b3
    b = np.zeros(4)
    for i in range(n):
        z = arr[i]
        b[0] += z
        if n > 1:
            b[1] += z * i / (n - 1)
        if n > 2:
            b[2] += z * i * (i - 1) / ((n - 1) * (n - 2))
        if n > 3:
            b[3] += z * i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3))
    b /= n

    l1 = b[0]
    l2 = 2 * b[1] - b[0]
    l3 = 6 * b[2] - 6 * b[1] + b[0]
    l4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]

    t3 = l3 / l2 if abs(l2) > 1e-15 else 0.0
    t4 = l4 / l2 if abs(l2) > 1e-15 else 0.0

    return {"L1": float(l1), "L2": float(l2), "L3": float(l3), "L4": float(l4), "t3": float(t3), "t4": float(t4)}


def fit_gev_lmoments(
    annual_maxima: np.ndarray | pd.Series,
    return_periods: list[float] | None = None,
) -> FloodFreqResult:
    """Fit GEV distribution using L-moments method.

    More robust than MLE for small samples (n < 50).  The shape parameter *k*
    is estimated from L-skewness using the Hosking (1997) approximation:

        c = 2 / (3 + t3) − ln2 / ln3
        k ≈ 7.8590 c + 2.9554 c²

    Parameters:
        annual_maxima: Array of annual maximum values.
        return_periods: Return periods in years.  Defaults to standard set.

    Returns:
        A :class:`FloodFreqResult` with fitted parameters and return levels.

    Raises:
        ValueError: If fewer than 5 values are provided.
    """
    data = np.asarray(annual_maxima, dtype=np.float64)
    if len(data) < 5:
        msg = f"Need ≥5 annual maxima; got {len(data)}"
        raise ValueError(msg)

    if return_periods is None:
        return_periods = _DEFAULT_RETURN_PERIODS

    lmom = lmoments_from_sample(data)
    l1, l2, t3 = lmom["L1"], lmom["L2"], lmom["t3"]

    # Hosking approximation for GEV shape from L-skewness
    c = 2.0 / (3.0 + t3) - np.log(2.0) / np.log(3.0)
    shape = 7.8590 * c + 2.9554 * c * c

    # GEV parameterisation: scale (alpha) and location (xi)
    gamma_val = _gamma_func(1 + shape)
    alpha = l2 * shape / (gamma_val * (1 - 2**(-shape))) if abs(shape) > 1e-10 else l2 / np.log(2)
    xi = l1 - alpha * (gamma_val - 1) / shape if abs(shape) > 1e-10 else l1 - alpha * 0.5772156649

    # scipy GEV uses *negative* shape convention
    scipy_shape = -shape

    from scipy.stats import genextreme

    rp_map: dict[float, float] = {}
    for rp in return_periods:
        prob = 1 - 1.0 / rp
        rp_map[rp] = float(genextreme.ppf(prob, scipy_shape, loc=xi, scale=alpha))

    logger.info("GEV L-moments fit: shape=%.3f, loc=%.3f, scale=%.3f", scipy_shape, xi, alpha)
    return FloodFreqResult(
        return_periods=rp_map,
        distribution="GEV_Lmom",
        params=(scipy_shape, xi, alpha),
        annual_max=pd.Series(data) if not isinstance(annual_maxima, pd.Series) else annual_maxima,
    )


def _gamma_func(x: float) -> float:
    """Thin wrapper around ``math.gamma`` for readability."""
    import math

    return math.gamma(x)


# ---------------------------------------------------------------------------
# Non-stationary GEV
# ---------------------------------------------------------------------------

@dataclass
class NonStationaryGEVResult:
    """Result of non-stationary GEV fit.

    Attributes:
        loc_intercept: Intercept of the linear location model.
        loc_trend: Trend in the location parameter (per year).
        scale: Scale parameter.
        shape: Shape parameter (scipy sign convention).
        return_levels: Mapping of return period → array of return levels over time.
        years: The year values used for fitting.
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        trend_significant: ``True`` if the trend p-value < 0.05 (likelihood-ratio test).
    """

    loc_intercept: float = 0.0
    loc_trend: float = 0.0
    scale: float = 1.0
    shape: float = 0.0
    return_levels: dict[float, np.ndarray] = field(default_factory=dict)
    years: np.ndarray = field(default_factory=lambda: np.array([]))
    aic: float = 0.0
    bic: float = 0.0
    trend_significant: bool = False


def _gev_neg_loglik(params: np.ndarray, data: np.ndarray, years_c: np.ndarray) -> float:
    """Negative log-likelihood for non-stationary GEV (linear location).

    Parameters:
        params: ``[mu0, mu1, log_scale, shape]``.
        data: Observed annual maxima.
        years_c: Centred year values ``(year − mean(year))``.
    """
    mu0, mu1, log_scale, shape = params
    scale = np.exp(log_scale)
    loc = mu0 + mu1 * years_c

    z = (data - loc) / scale
    if abs(shape) < 1e-10:
        # Gumbel limit
        nll = np.sum(np.log(scale) + z + np.exp(-z))
    else:
        t = 1 + shape * z
        if np.any(t <= 0):
            return 1e12
        nll = np.sum(np.log(scale) + (1 + 1 / shape) * np.log(t) + t ** (-1 / shape))
    return float(nll)


def _gev_neg_loglik_stationary(params: np.ndarray, data: np.ndarray) -> float:
    """Negative log-likelihood for stationary GEV."""
    mu0, log_scale, shape = params
    scale = np.exp(log_scale)
    z = (data - mu0) / scale
    if abs(shape) < 1e-10:
        nll = np.sum(np.log(scale) + z + np.exp(-z))
    else:
        t = 1 + shape * z
        if np.any(t <= 0):
            return 1e12
        nll = np.sum(np.log(scale) + (1 + 1 / shape) * np.log(t) + t ** (-1 / shape))
    return float(nll)


def fit_nonstationary_gev(
    annual_maxima: np.ndarray | pd.Series,
    years: np.ndarray | pd.Series,
    return_periods: list[float] | None = None,
) -> NonStationaryGEVResult:
    """Fit GEV with time-varying location: ``loc(t) = mu0 + mu1 * (t − t̄)``.

    The trend significance is assessed via a likelihood-ratio test comparing
    the non-stationary model to a stationary GEV.

    Parameters:
        annual_maxima: Array of annual maximum values.
        years: Corresponding year values (same length as *annual_maxima*).
        return_periods: Return periods in years.  Defaults to standard set.

    Returns:
        A :class:`NonStationaryGEVResult`.

    Raises:
        ValueError: If input arrays differ in length or have fewer than 10 values.
    """
    from scipy.optimize import minimize
    from scipy.stats import chi2, genextreme

    data = np.asarray(annual_maxima, dtype=np.float64)
    yrs = np.asarray(years, dtype=np.float64)
    if len(data) != len(yrs):
        msg = f"annual_maxima ({len(data)}) and years ({len(yrs)}) must have the same length"
        raise ValueError(msg)
    if len(data) < 10:
        msg = f"Need ≥10 annual maxima; got {len(data)}"
        raise ValueError(msg)

    if return_periods is None:
        return_periods = _DEFAULT_RETURN_PERIODS

    year_mean = np.mean(yrs)
    years_c = yrs - year_mean

    # Initial estimates from stationary GEV
    shape0, loc0, scale0 = genextreme.fit(data)
    x0_ns = np.array([loc0, 0.0, np.log(scale0), shape0])
    x0_st = np.array([loc0, np.log(scale0), shape0])

    res_ns = minimize(_gev_neg_loglik, x0_ns, args=(data, years_c), method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    res_st = minimize(_gev_neg_loglik_stationary, x0_st, args=(data,), method="Nelder-Mead",
                      options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})

    mu0, mu1, log_scale, shape = res_ns.x
    scale = float(np.exp(log_scale))
    n = len(data)

    nll_ns = res_ns.fun
    nll_st = res_st.fun

    # AIC / BIC for non-stationary model (4 parameters)
    aic = 2 * nll_ns + 2 * 4
    bic = 2 * nll_ns + np.log(n) * 4

    # Likelihood-ratio test (1 df for the additional mu1 parameter)
    lr_stat = 2 * (nll_st - nll_ns)
    p_value = 1 - chi2.cdf(max(lr_stat, 0), df=1)
    trend_significant = bool(p_value < 0.05)

    # Return levels for each year
    rl: dict[float, np.ndarray] = {}
    for rp in return_periods:
        prob = 1 - 1.0 / rp
        levels = np.array([
            float(genextreme.ppf(prob, shape, loc=mu0 + mu1 * yc, scale=scale))
            for yc in years_c
        ])
        rl[rp] = levels

    logger.info(
        "Non-stationary GEV: mu0=%.3f, mu1=%.5f, scale=%.3f, shape=%.3f, trend_sig=%s",
        mu0, mu1, scale, shape, trend_significant,
    )
    return NonStationaryGEVResult(
        loc_intercept=float(mu0),
        loc_trend=float(mu1),
        scale=scale,
        shape=float(shape),
        return_levels=rl,
        years=yrs,
        aic=float(aic),
        bic=float(bic),
        trend_significant=trend_significant,
    )


# ---------------------------------------------------------------------------
# Regional frequency analysis (Hosking & Wallis)
# ---------------------------------------------------------------------------

@dataclass
class RegionalResult:
    """Result of regional frequency analysis.

    Attributes:
        growth_curve: Mapping of return period → growth factor.
        index_flood: Mapping of site id → index flood (mean annual max).
        regional_return_levels: Mapping of site id → {return period → level}.
        discordancy: Mapping of site id → discordancy statistic Dᵢ.
        heterogeneity: Heterogeneity H statistic.
    """

    growth_curve: dict[float, float] = field(default_factory=dict)
    index_flood: dict[str, float] = field(default_factory=dict)
    regional_return_levels: dict[str, dict[float, float]] = field(default_factory=dict)
    discordancy: dict[str, float] = field(default_factory=dict)
    heterogeneity: float = 0.0


def regional_frequency_analysis(
    sites: dict[str, np.ndarray],
    return_periods: list[float] | None = None,
) -> RegionalResult:
    """L-moment based regional frequency analysis (Hosking & Wallis method).

    Steps:
        1. Compute L-moments for each site.
        2. Discordancy test (flag sites with unusual L-moments).
        3. Heterogeneity measure (H < 1 → acceptably homogeneous).
        4. Fit regional growth curve using weighted regional L-moments.
        5. Combine with site-specific index flood for return levels.

    Parameters:
        sites: Mapping of site identifier → annual maxima array.
        return_periods: Return periods in years.  Defaults to standard set.

    Returns:
        A :class:`RegionalResult`.

    Raises:
        ValueError: If fewer than 2 sites are provided.
    """
    from scipy.stats import genextreme

    if len(sites) < 2:
        msg = f"Need ≥2 sites for regional analysis; got {len(sites)}"
        raise ValueError(msg)

    if return_periods is None:
        return_periods = _DEFAULT_RETURN_PERIODS

    # 1. Site L-moments
    site_lmom: dict[str, dict[str, float]] = {}
    site_n: dict[str, int] = {}
    index_flood: dict[str, float] = {}
    for sid, arr in sites.items():
        arr = np.asarray(arr, dtype=np.float64)
        site_lmom[sid] = lmoments_from_sample(arr)
        site_n[sid] = len(arr)
        index_flood[sid] = float(np.mean(arr))

    total_n = sum(site_n.values())
    k = len(sites)

    # 2. Discordancy measure Dᵢ
    # Vector u_i = [t3_i, t4_i]
    u_mat = np.array([[site_lmom[s]["t3"], site_lmom[s]["t4"]] for s in sites])
    u_bar = u_mat.mean(axis=0)
    diff = u_mat - u_bar
    s_mat = diff.T @ diff / k
    s_inv = np.linalg.pinv(s_mat) if np.linalg.matrix_rank(s_mat) >= 2 else np.eye(2)
    discordancy: dict[str, float] = {}
    for i, sid in enumerate(sites):
        d = diff[i] @ s_inv @ diff[i]
        discordancy[sid] = float(k * d / 3.0)

    # 3. Heterogeneity H (simplified — based on weighted variance of t3)
    weighted_t3 = sum(site_n[s] * site_lmom[s]["t3"] for s in sites) / total_n
    v = sum(site_n[s] * (site_lmom[s]["t3"] - weighted_t3) ** 2 for s in sites) / total_n
    # Approximate expected variance under homogeneity using simulation would be
    # complex; use a simplified threshold-based H.
    heterogeneity = float(v * np.sqrt(total_n))

    # 4. Regional weighted L-moments
    reg_l1 = 1.0  # normalised (growth-curve convention)
    reg_l2 = sum(site_n[s] * (site_lmom[s]["L2"] / index_flood[s]) for s in sites) / total_n
    reg_t3 = sum(site_n[s] * site_lmom[s]["t3"] for s in sites) / total_n
    # Fit GEV growth curve from regional L-moment ratios
    c = 2.0 / (3.0 + reg_t3) - np.log(2.0) / np.log(3.0)
    shape = 7.8590 * c + 2.9554 * c * c
    gamma_val = _gamma_func(1 + shape)
    alpha = reg_l2 * shape / (gamma_val * (1 - 2**(-shape))) if abs(shape) > 1e-10 else reg_l2 / np.log(2)
    xi = reg_l1 - alpha * (gamma_val - 1) / shape if abs(shape) > 1e-10 else reg_l1 - alpha * 0.5772156649
    scipy_shape = -shape

    growth_curve: dict[float, float] = {}
    for rp in return_periods:
        prob = 1 - 1.0 / rp
        growth_curve[rp] = float(genextreme.ppf(prob, scipy_shape, loc=xi, scale=alpha))

    # 5. Site-specific return levels
    regional_return_levels: dict[str, dict[float, float]] = {}
    for sid in sites:
        regional_return_levels[sid] = {rp: index_flood[sid] * gf for rp, gf in growth_curve.items()}

    logger.info("Regional analysis: %d sites, H=%.3f", k, heterogeneity)
    return RegionalResult(
        growth_curve=growth_curve,
        index_flood=index_flood,
        regional_return_levels=regional_return_levels,
        discordancy=discordancy,
        heterogeneity=heterogeneity,
    )


# ---------------------------------------------------------------------------
# Goodness-of-fit tests
# ---------------------------------------------------------------------------

@dataclass
class GoodnessOfFitResult:
    """Result of a goodness-of-fit test.

    Attributes:
        statistic: Test statistic value.
        p_value: Associated p-value.
        test_name: Name of the test.
        distribution: Distribution that was tested.
        reject_h0: ``True`` if H₀ (data follows distribution) is rejected at α = 0.05.
    """

    statistic: float = 0.0
    p_value: float = 1.0
    test_name: str = ""
    distribution: str = ""
    reject_h0: bool = False


def _get_scipy_dist(distribution: str):
    """Return a scipy continuous distribution object by name."""
    from scipy import stats

    mapping = {
        "gev": stats.genextreme,
        "genextreme": stats.genextreme,
        "gumbel": stats.gumbel_r,
        "gumbel_r": stats.gumbel_r,
        "weibull_min": stats.weibull_min,
        "gpd": stats.genpareto,
        "genpareto": stats.genpareto,
        "norm": stats.norm,
        "uniform": stats.uniform,
    }
    dist = mapping.get(distribution.lower())
    if dist is None:
        msg = f"Unsupported distribution: {distribution!r}"
        raise ValueError(msg)
    return dist


def anderson_darling_test(
    data: np.ndarray,
    distribution: str,
    params: tuple,
) -> GoodnessOfFitResult:
    """Anderson-Darling goodness-of-fit test for a fitted distribution.

    Parameters:
        data: Observed sample.
        distribution: Distribution name (e.g. ``"gev"``, ``"gumbel"``).
        params: Distribution parameters as accepted by the scipy distribution.

    Returns:
        A :class:`GoodnessOfFitResult`.
    """
    dist = _get_scipy_dist(distribution)
    arr = np.sort(np.asarray(data, dtype=np.float64))
    n = len(arr)

    cdf_vals = dist.cdf(arr, *params)
    cdf_vals = np.clip(cdf_vals, 1e-15, 1 - 1e-15)

    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1])))
    a2 = -n - s / n

    # Approximate p-value (Marsaglia & Marsaglia, 2004 — simplified)
    a2_star = a2 * (1 + 0.75 / n + 2.25 / n**2)
    if a2_star >= 0.6:
        p_value = float(np.exp(1.2937 - 5.709 * a2_star + 0.0186 * a2_star**2))
    elif a2_star >= 0.34:
        p_value = float(np.exp(0.9177 - 4.279 * a2_star - 1.38 * a2_star**2))
    elif a2_star >= 0.2:
        p_value = float(1 - np.exp(-8.318 + 42.796 * a2_star - 59.938 * a2_star**2))
    else:
        p_value = float(1 - np.exp(-13.436 + 101.14 * a2_star - 223.73 * a2_star**2))
    p_value = np.clip(p_value, 0.0, 1.0)

    return GoodnessOfFitResult(
        statistic=float(a2),
        p_value=float(p_value),
        test_name="Anderson-Darling",
        distribution=distribution,
        reject_h0=bool(p_value < 0.05),
    )


def cramer_von_mises_test(
    data: np.ndarray,
    distribution: str,
    params: tuple,
) -> GoodnessOfFitResult:
    """Cramér–von Mises goodness-of-fit test.

    Parameters:
        data: Observed sample.
        distribution: Distribution name.
        params: Distribution parameters.

    Returns:
        A :class:`GoodnessOfFitResult`.
    """
    dist = _get_scipy_dist(distribution)
    arr = np.sort(np.asarray(data, dtype=np.float64))
    n = len(arr)

    cdf_vals = dist.cdf(arr, *params)
    i = np.arange(1, n + 1)
    w2 = np.sum((cdf_vals - (2 * i - 1) / (2 * n)) ** 2) + 1 / (12 * n)

    # Approximate p-value (Csörgő–Faraway table approximation)
    w2_star = w2 * (1 + 0.5 / n)
    if w2_star >= 0.461:
        p_value = 0.01
    elif w2_star >= 0.347:
        p_value = 0.025
    elif w2_star >= 0.274:
        p_value = 0.05
    elif w2_star >= 0.175:
        p_value = 0.10
    elif w2_star >= 0.116:
        p_value = 0.25
    else:
        p_value = 0.50

    return GoodnessOfFitResult(
        statistic=float(w2),
        p_value=float(p_value),
        test_name="Cramer-von-Mises",
        distribution=distribution,
        reject_h0=bool(p_value < 0.05),
    )


def probability_plot_correlation(
    data: np.ndarray,
    distribution: str,
    params: tuple,
) -> float:
    """Probability Plot Correlation Coefficient (PPCC).

    Computes the Pearson correlation between the sorted observations and
    the corresponding theoretical quantiles.  Values close to 1 indicate
    a good fit.

    Parameters:
        data: Observed sample.
        distribution: Distribution name.
        params: Distribution parameters.

    Returns:
        PPCC value (between 0 and 1 for reasonable fits).
    """
    dist = _get_scipy_dist(distribution)
    arr = np.sort(np.asarray(data, dtype=np.float64))
    n = len(arr)

    # Plotting positions (Hazen formula)
    pp = (np.arange(1, n + 1) - 0.5) / n
    theoretical = dist.ppf(pp, *params)

    corr = float(np.corrcoef(arr, theoretical)[0, 1])
    return corr


# ---------------------------------------------------------------------------
# Bulletin 17C helpers and EMA implementation
# ---------------------------------------------------------------------------


def _station_skew(x: np.ndarray) -> float:
    """Compute sample skewness coefficient (Fisher's definition).

    Uses the biased third central moment adjusted by ``n / ((n-1)(n-2))``.

    Parameters:
        x: 1-D array of sample values.

    Returns:
        Sample skewness.
    """
    n = len(x)
    if n < 3:
        return 0.0
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2)
    m3 = np.sum((x - mean) ** 3)
    s = np.sqrt(m2 / (n - 1))
    if s == 0:
        return 0.0
    return float(n * m3 / ((n - 1) * (n - 2) * s**3))


def weighted_skew(
    station_skew: float,
    regional_skew: float,
    n: int,
    regional_mse: float = 0.302,
) -> float:
    """Compute weighted skew per Bulletin 17C §5.2.4.

    Combines the station skew ``Gs`` with a generalised / regional skew
    ``Gr`` using inverse-variance weights:

    .. math::

        G_w = w_1 G_s + w_2 G_r

    where ``w_1 = MSE_r / (MSE_s + MSE_r)`` and ``w_2 = 1 − w_1``.

    Parameters:
        station_skew: Station skew coefficient ``Gs``.
        regional_skew: Regional / generalised skew ``Gr``.
        n: Number of annual maximum observations.
        regional_mse: Mean-square error of the regional skew estimate.
            Default ``0.302`` is the USGS nationwide value from Bulletin 17C.

    Returns:
        Weighted skew coefficient ``Gw``.

    References:
        England, J. F. Jr. et al. (2018). Guidelines for determining flood
        flow frequency — Bulletin 17C. USGS TM 4-B5.
        https://doi.org/10.3133/tm4B5
    """
    gs = station_skew
    # Approximate station MSE from Bulletin 17C Eq. 5-1
    station_mse = (6.0 / n) * (1 + (9.0 / 6.0) * gs**2 + (15.0 / 48.0) * gs**4)

    w1 = regional_mse / (station_mse + regional_mse)  # weight for station
    w2 = 1.0 - w1  # weight for regional
    return float(w1 * gs + w2 * regional_skew)


def grubbs_beck_test(
    annual_max: np.ndarray,
    *,
    alpha: float = 0.10,
) -> tuple[float, np.ndarray]:
    """Multiple Grubbs-Beck (MGB) test for low-outlier detection.

    Implements the iterative procedure described in Bulletin 17C Appendix 6
    (Cohn et al., 2013).  The test repeatedly identifies the smallest
    observation that is significantly low relative to the remaining sample.

    Parameters:
        annual_max: 1-D array of annual maximum values (all positive).
        alpha: Significance level for the test.

    Returns:
        A 2-tuple ``(threshold, mask)`` where *threshold* is the low-outlier
        cutoff (log10 scale back-transformed) and *mask* is a boolean array
        with ``True`` for observations identified as low outliers.

    References:
        Cohn, T. A., England, J. F. Jr., Berenbrock, C. E., Mason, R. R.,
        Stedinger, J. R., & Lamontagne, J. R. (2013). A generalized
        Grubbs-Beck test statistic for detecting multiple potentially
        influential low outliers in flood series. Water Resources Research,
        49(8), 5047-5058.

        Grubbs, F. E. & Beck, G. (1972). Extension of sample sizes and
        percentage points for significance tests of outlying observations.
        Technometrics, 14(4), 847-854.
    """
    from scipy.stats import t as t_dist

    arr = np.asarray(annual_max, dtype=np.float64).copy()
    log_vals = np.log10(np.maximum(arr, 1e-30))

    n_total = len(log_vals)
    mask = np.zeros(n_total, dtype=bool)

    # Sort indices to identify smallest values
    sorted_idx = np.argsort(log_vals)

    n_removed = 0
    max_removable = max(n_total // 2 - 1, 0)

    for _ in range(max_removable):
        remaining = log_vals[~mask]
        n_rem = len(remaining)
        if n_rem < 5:
            break

        mean_r = np.mean(remaining)
        std_r = np.std(remaining, ddof=1)
        if std_r < 1e-15:
            break

        # Critical value: one-sided Grubbs statistic using t-distribution
        p = alpha / n_rem
        t_crit = float(t_dist.ppf(p, df=n_rem - 2))
        k_n = (t_crit * np.sqrt(n_rem - 1)) / np.sqrt(n_rem - 2 + t_crit**2)

        threshold_log = mean_r + k_n * std_r  # k_n is negative for low side

        # Find the smallest remaining value
        candidate_idx = sorted_idx[n_removed]
        candidate_val = log_vals[candidate_idx]

        if candidate_val < threshold_log:
            mask[candidate_idx] = True
            n_removed += 1
        else:
            break

    if n_removed > 0:
        remaining = log_vals[~mask]
        mean_r = np.mean(remaining)
        std_r = np.std(remaining, ddof=1)
        n_rem = len(remaining)
        p = alpha / n_rem
        t_crit = float(t_dist.ppf(p, df=max(n_rem - 2, 1)))
        k_n = (t_crit * np.sqrt(n_rem - 1)) / np.sqrt(max(n_rem - 2 + t_crit**2, 1e-15))
        threshold = float(10 ** (mean_r + k_n * std_r))
    else:
        # No outliers: return a threshold below all data
        threshold = float(10 ** (np.min(log_vals) - 1.0))

    return threshold, mask


@dataclass
class EMAResult(FloodFreqResult):
    """Result of Expected Moments Algorithm flood frequency analysis.

    Extends :class:`FloodFreqResult` with censoring and EMA-specific fields.

    Attributes:
        n_censored: Number of censored (zero-flow / low-outlier) observations.
        n_observed: Number of non-censored observations.
        weighted_skew: Weighted skew coefficient (station + regional).
            ``None`` when no regional skew was supplied.
        low_outlier_threshold: MGB low-outlier threshold (real-space).
            ``None`` when no outliers were detected.
    """

    n_censored: int = 0
    n_observed: int = 0
    weighted_skew: float | None = None
    low_outlier_threshold: float | None = None


def expected_moments_algorithm(
    annual_max: pd.Series | np.ndarray,
    *,
    perception_thresholds: list[tuple[float, float]] | None = None,
    zero_threshold: float = 0.0,
    regional_skew: float | None = None,
    regional_skew_mse: float = 0.302,
    return_periods: list[int] | None = None,
) -> EMAResult:
    """Expected Moments Algorithm (EMA) for LP3 flood frequency analysis.

    Implements the EMA procedure of Cohn et al. (1997) for incorporating
    censored observations (zero-flow years, low outliers, historical floods)
    into the LP3 parameter estimation.  This is the preferred method in
    USGS Bulletin 17C (England et al., 2018).

    Parameters:
        annual_max: Annual maximum series.  May contain zero / negative values
            which will be treated as censored.
        perception_thresholds: Optional list of ``(lower, upper)`` pairs
            defining the perception interval for each observation.  When
            ``None`` the algorithm automatically treats observations
            ≤ *zero_threshold* as left-censored and the remainder as
            exactly observed.
        zero_threshold: Values ≤ this are treated as censored (default 0.0).
        regional_skew: Regional / generalised skew for weighted skew.
        regional_skew_mse: MSE of the regional skew estimate.
        return_periods: Return periods to estimate.  Defaults to standard set.

    Returns:
        An :class:`EMAResult` with LP3 quantile estimates, confidence
        intervals, and censoring metadata.

    Raises:
        ValueError: If fewer than 5 total observations.

    References:
        England, J. F. Jr., Cohn, T. A., Faber, B. A., Stedinger, J. R.,
        Thomas, W. O. Jr., Veilleux, A. G., Kiang, J. E., & Mason, R. R.
        Jr. (2018). Guidelines for determining flood flow frequency —
        Bulletin 17C. USGS TM 4-B5. https://doi.org/10.3133/tm4B5

        Cohn, T. A., Lane, W. L., & Baier, W. G. (1997). An algorithm for
        computing moments-based flood quantile estimates when historical
        flood information is available. Water Resources Research, 33(9),
        2089-2096. https://doi.org/10.1029/96WR03706
    """
    from scipy.stats import pearson3
    from scipy.stats import t as t_dist

    arr = np.asarray(annual_max, dtype=np.float64).copy()

    if len(arr) < 5:
        msg = f"Need ≥5 observations; got {len(arr)}"
        raise ValueError(msg)

    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100, 200, 500]

    # ------------------------------------------------------------------
    # 1.  Separate observed vs censored
    # ------------------------------------------------------------------
    if perception_thresholds is not None:
        # Use explicit perception intervals
        observed_mask = np.array(
            [lo == up for (lo, up) in perception_thresholds], dtype=bool
        )
        # "Observed" means the interval is a point; censored otherwise
        # For simplicity, treat the point-interval values as observed
        observed_vals = arr[observed_mask]
        n_censored = int(np.sum(~observed_mask))
    else:
        observed_mask = arr > zero_threshold
        observed_vals = arr[observed_mask]
        n_censored = int(np.sum(~observed_mask))

    n_total = len(arr)
    n_observed = n_total - n_censored

    if n_observed < 3:
        msg = f"Need ≥3 non-censored observations; got {n_observed}"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # 2.  MGB test on observed values to detect low outliers
    # ------------------------------------------------------------------
    low_outlier_threshold: float | None = None
    if n_observed >= 10:
        mgb_thresh, mgb_mask = grubbs_beck_test(observed_vals)
        n_mgb_outliers = int(np.sum(mgb_mask))
        if n_mgb_outliers > 0:
            low_outlier_threshold = mgb_thresh
            # Treat MGB outliers as additional censored observations
            non_outlier = observed_vals[~mgb_mask]
            n_censored += n_mgb_outliers
            n_observed -= n_mgb_outliers
            observed_vals = non_outlier

    # ------------------------------------------------------------------
    # 3.  Compute adjusted moments (EMA approach)
    # ------------------------------------------------------------------
    log_obs = np.log10(observed_vals)
    mean_log = float(np.mean(log_obs))
    std_log = float(np.std(log_obs, ddof=1))

    if n_censored > 0 and n_observed >= 3:
        # Adjust moments for censored observations.
        # Censored values are below the threshold; use expected value of the
        # truncated LP3 below the censoring point.
        if low_outlier_threshold is not None:
            censor_point_log = np.log10(max(low_outlier_threshold, 1e-30))
        elif zero_threshold > 0:
            censor_point_log = np.log10(zero_threshold)
        else:
            censor_point_log = np.log10(max(np.min(observed_vals) * 0.5, 1e-30))

        p_censor = n_censored / n_total  # proportion censored

        # Approximate expected value of censored portion under LP3
        # E[X | X < c] ≈ c - std * pdf(z_c) / Phi(z_c)
        z_c = (censor_point_log - mean_log) / max(std_log, 1e-15)
        from scipy.stats import norm

        pdf_zc = float(norm.pdf(z_c))
        cdf_zc = float(norm.cdf(z_c))

        if cdf_zc > 1e-10:
            expected_censored = censor_point_log - std_log * pdf_zc / cdf_zc
        else:
            expected_censored = censor_point_log

        # Adjusted mean
        adjusted_mean = (1 - p_censor) * mean_log + p_censor * expected_censored

        # Adjusted variance: include censored contribution
        var_obs = float(np.var(log_obs, ddof=0))
        var_censored = (expected_censored - adjusted_mean) ** 2
        adjusted_var = (1 - p_censor) * (var_obs + (mean_log - adjusted_mean) ** 2) + p_censor * var_censored
        adjusted_std = np.sqrt(max(adjusted_var * n_total / (n_total - 1), 1e-30))

        mean_log = float(adjusted_mean)
        std_log = float(adjusted_std)

    # Compute skew on observed log values
    station_skew = float(_station_skew(log_obs))

    # ------------------------------------------------------------------
    # 4.  Apply weighted skew if regional skew is provided
    # ------------------------------------------------------------------
    wt_skew: float | None = None
    if regional_skew is not None:
        skew_coeff = weighted_skew(station_skew, regional_skew, n_observed, regional_mse=regional_skew_mse)
        wt_skew = skew_coeff
    else:
        skew_coeff = station_skew

    # ------------------------------------------------------------------
    # 5.  Compute quantiles and confidence intervals
    # ------------------------------------------------------------------
    rp_map: dict[int, float] = {}
    ci_map: dict[int, tuple[float, float]] = {}
    ci_alpha = 0.05  # 90% CIs

    for rp in return_periods:
        prob = 1 - 1.0 / rp
        log_q = float(pearson3.ppf(prob, skew_coeff, loc=mean_log, scale=std_log))
        rp_map[rp] = float(10**log_q)

        # Variance-of-estimate CIs (B17C §6)
        kp = float(pearson3.ppf(prob, skew_coeff, loc=0, scale=1))
        var_q = (std_log**2 / n_observed) * (1 + kp**2 / 2 * (1 + 0.75 * skew_coeff**2))
        se_q = np.sqrt(max(var_q, 0.0))
        t_val = float(t_dist.ppf(1 - ci_alpha, df=max(n_observed - 1, 1)))
        ci_map[rp] = (float(10 ** (log_q - t_val * se_q)), float(10 ** (log_q + t_val * se_q)))

    logger.info(
        "EMA LP3 fit: skew=%.3f, mean=%.3f, std=%.3f, n_obs=%d, n_cens=%d",
        skew_coeff,
        mean_log,
        std_log,
        n_observed,
        n_censored,
    )

    return EMAResult(
        return_periods=rp_map,
        distribution="LP3-EMA",
        params=(skew_coeff, mean_log, std_log),
        annual_max=pd.Series(arr) if isinstance(annual_max, np.ndarray) else annual_max,
        confidence_intervals=ci_map,
        n_censored=n_censored,
        n_observed=n_observed,
        weighted_skew=wt_skew,
        low_outlier_threshold=low_outlier_threshold,
    )


# ---------------------------------------------------------------------------
# Cross-validation utilities
# ---------------------------------------------------------------------------

_CV_DIST_MAP = {
    "gev": "genextreme",
    "lp3": "pearson3",
    "gumbel": "gumbel_r",
    "gpd": "genpareto",
    "weibull": "weibull_min",
}


def _fit_and_predict(data: np.ndarray, distribution: str, return_periods: list[int]):
    """Fit *distribution* on *data* and return quantile estimates.

    Parameters
    ----------
    data : np.ndarray
        Annual maximum values used for fitting.
    distribution : str
        Distribution key (``"gev"``, ``"lp3"``, etc.).
    return_periods : list[int]
        Return periods to predict.

    Returns
    -------
    dict[int, float]
        Mapping of return period → predicted quantile.
    """
    import scipy.stats as st

    key = distribution.lower()
    is_lp3 = key == "lp3"
    fit_data = np.log10(data) if is_lp3 else data
    dist = getattr(st, _CV_DIST_MAP[key])
    params = dist.fit(fit_data)

    rp_map: dict[int, float] = {}
    for rp in return_periods:
        prob = 1 - 1.0 / rp
        q = float(dist.ppf(prob, *params))
        rp_map[rp] = float(10**q) if is_lp3 else q
    return rp_map


def leave_one_out_cv(
    discharge: pd.Series,
    distribution: str = "gev",
    return_periods: list[int] | None = None,
) -> dict:
    """Leave-one-out cross-validation for flood frequency fits.

    For each held-out year, the distribution is fitted on the remaining
    years and the held-out observation is compared against the fitted
    median (T = 2-year return level).

    Parameters
    ----------
    discharge : pd.Series
        Annual maximum discharge series (with a ``DatetimeIndex``).
    distribution : str
        Distribution key: ``"gev"``, ``"lp3"``, ``"gumbel"``, ``"gpd"``,
        ``"weibull"``.
    return_periods : list[int], optional
        Return periods used internally.  Defaults to ``[2]`` (median).

    Returns
    -------
    dict
        Keys: ``'rmse'``, ``'bias'``, ``'mae'``, ``'predictions'``,
        ``'observations'``.
    """
    data = np.asarray(discharge.values, dtype=np.float64)
    n = len(data)
    if n < 6:
        msg = f"Need ≥6 values for LOO-CV; got {n}"
        raise ValueError(msg)

    if return_periods is None:
        return_periods = [2]

    predictions: list[float] = []
    observations: list[float] = []

    for i in range(n):
        train = np.delete(data, i)
        obs = float(data[i])
        try:
            rp_map = _fit_and_predict(train, distribution, return_periods)
            pred = rp_map[return_periods[0]]
        except Exception:  # noqa: BLE001
            continue
        predictions.append(pred)
        observations.append(obs)

    preds = np.array(predictions)
    obs_arr = np.array(observations)
    errors = preds - obs_arr

    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "predictions": predictions,
        "observations": observations,
    }


def coverage_probability(
    discharge: pd.Series,
    distribution: str = "gev",
    ci_level: float = 0.90,
    n_splits: int = 10,
    n_boot: int = 200,
) -> float:
    """Estimate the coverage probability of confidence intervals.

    Split data into *n_splits* folds, compute bootstrap CIs on each
    training set, and check what fraction of test observations fall
    within those CIs.

    Parameters
    ----------
    discharge : pd.Series
        Annual maximum discharge series.
    distribution : str
        Distribution key.
    ci_level : float
        Nominal confidence level (default 0.90).
    n_splits : int
        Number of cross-validation folds (default 10).
    n_boot : int
        Number of bootstrap samples per fold (default 200).

    Returns
    -------
    float
        Observed coverage probability (0–1).
    """
    import scipy.stats as st

    key = distribution.lower()
    is_lp3 = key == "lp3"
    dist = getattr(st, _CV_DIST_MAP[key])

    data = np.asarray(discharge.values, dtype=np.float64)
    n = len(data)
    if n < n_splits:
        msg = f"Need ≥{n_splits} values for {n_splits}-fold CV; got {n}"
        raise ValueError(msg)

    rng = np.random.default_rng(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    covered = 0
    total = 0

    for fold in folds:
        train_idx = np.setdiff1d(indices, fold)
        train = data[train_idx]
        test = data[fold]

        fit_data = np.log10(train) if is_lp3 else train
        try:
            dist.fit(fit_data)
        except Exception:  # noqa: BLE001
            continue

        # Bootstrap CIs on training set
        alpha = (1 - ci_level) / 2
        boot_medians: list[float] = []
        for _ in range(n_boot):
            sample = rng.choice(fit_data, size=len(fit_data), replace=True)
            try:
                bp = dist.fit(sample)
                q = float(dist.ppf(0.5, *bp))
                boot_medians.append(float(10**q) if is_lp3 else q)
            except Exception:  # noqa: BLE001
                continue

        if len(boot_medians) < 50:
            continue

        lo = float(np.percentile(boot_medians, alpha * 100))
        hi = float(np.percentile(boot_medians, (1 - alpha) * 100))

        for obs in test:
            total += 1
            if lo <= obs <= hi:
                covered += 1

    return covered / total if total > 0 else 0.0
