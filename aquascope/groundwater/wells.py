"""Well data analysis for groundwater monitoring.

Provides hydrograph processing, trend detection (Mann-Kendall / Sen's slope),
seasonal decomposition using moving averages, and recession analysis for
estimating aquifer time constants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class HydrographResult:
    """Result of well hydrograph analysis.

    Attributes
    ----------
    series:
        Cleaned water-level time series.
    stats:
        Descriptive statistics dict (mean, median, std, min, max, range, count).
    correlation:
        Pearson correlation with precipitation, or None.
    lag_days:
        Lag (days) of maximum cross-correlation with precipitation, or None.
    """

    series: pd.Series
    stats: dict[str, float]
    correlation: float | None = None
    lag_days: int | None = None


@dataclass
class WellTrendResult:
    """Result of trend detection in a well time series.

    Attributes
    ----------
    trend:
        ``"increasing"``, ``"decreasing"``, or ``"no trend"``.
    p_value:
        P-value of the trend test.
    slope:
        Sen's slope estimate (units per time step).
    intercept:
        Sen's intercept.
    z_statistic:
        Mann-Kendall Z statistic.
    method:
        The method used (``"mann_kendall"``).
    """

    trend: str
    p_value: float
    slope: float
    intercept: float
    z_statistic: float
    method: str


@dataclass
class SeasonalResult:
    """Result of seasonal decomposition.

    Attributes
    ----------
    trend:
        Trend component.
    seasonal:
        Seasonal component.
    residual:
        Residual component.
    period:
        Period used for decomposition.
    strength:
        Seasonal strength (0–1), where 1 = purely seasonal.
    """

    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    period: int
    strength: float


@dataclass
class RecessionResult:
    """Result of recession analysis.

    Attributes
    ----------
    events:
        List of (start, end) index pairs for each recession event.
    recession_constants:
        Fitted exponential decay constant for each event (days).
    mean_constant:
        Mean recession constant across all events.
    master_constant:
        Recession constant from the composite master recession curve.
    fitted_curves:
        List of fitted recession curves (as pd.Series), one per event.
    """

    events: list[tuple[int, int]]
    recession_constants: list[float]
    mean_constant: float
    master_constant: float
    fitted_curves: list[pd.Series] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def well_hydrograph(
    levels: pd.Series,
    precip: pd.Series | None = None,
) -> HydrographResult:
    """Process a well water-level time series.

    Parameters
    ----------
    levels:
        Water-level observations (depth to water, m) with DatetimeIndex.
    precip:
        Optional precipitation series aligned with *levels* for
        cross-correlation analysis.

    Returns
    -------
    HydrographResult
        Descriptive statistics and optional precipitation correlation.

    Raises
    ------
    ValueError
        If *levels* is empty.
    """
    if len(levels) == 0:
        raise ValueError("Water-level series is empty.")

    clean = levels.dropna()
    desc: dict[str, float] = {
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "range": float(clean.max() - clean.min()),
        "count": float(len(clean)),
    }

    corr: float | None = None
    lag: int | None = None

    if precip is not None and len(precip) > 0:
        # Align on common dates
        common = clean.index.intersection(precip.dropna().index)
        if len(common) > 2:
            l_aligned = clean.loc[common].values.astype(float)
            p_aligned = precip.loc[common].values.astype(float)
            corr = float(np.corrcoef(l_aligned, p_aligned)[0, 1])

            # Cross-correlation to find lag of max correlation
            max_lag = min(len(common) // 3, 365)
            if max_lag > 0:
                best_corr = -np.inf
                best_lag = 0
                for shift in range(max_lag + 1):
                    if shift >= len(l_aligned):
                        break
                    c = np.corrcoef(l_aligned[shift:], p_aligned[: len(l_aligned) - shift])[0, 1]
                    if c > best_corr:
                        best_corr = c
                        best_lag = shift
                lag = best_lag

    logger.info("Hydrograph: n=%d, mean=%.2f, std=%.2f", len(clean), desc["mean"], desc["std"])
    return HydrographResult(series=clean, stats=desc, correlation=corr, lag_days=lag)


def trend_detection(
    levels: pd.Series,
    method: str = "mann_kendall",
) -> WellTrendResult:
    """Detect monotonic trend using Mann-Kendall test with Sen's slope.

    Uses a pure scipy implementation — no external ``pymannkendall``
    dependency required.

    Parameters
    ----------
    levels:
        Water-level series with DatetimeIndex.
    method:
        Trend detection method. Currently only ``"mann_kendall"`` is
        supported.

    Returns
    -------
    WellTrendResult
        Trend direction, p-value, and Sen's slope.

    Raises
    ------
    ValueError
        If series has fewer than 3 observations or method is unknown.
    """
    if method != "mann_kendall":
        raise ValueError(f"Unknown method: {method!r}. Supported: 'mann_kendall'.")
    if len(levels) < 3:
        raise ValueError("Need at least 3 data points for trend detection.")

    y = levels.dropna().values.astype(float)
    n = len(y)

    # Mann-Kendall S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = y[j] - y[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S (accounting for ties)
    unique, counts = np.unique(y, return_counts=True)
    tie_sum = sum(t * (t - 1) * (2 * t + 5) for t in counts if t > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18.0

    # Z statistic
    if var_s == 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2.0 * stats.norm.sf(abs(z))

    # Sen's slope
    slopes: list[float] = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if j != i:
                slopes.append((y[j] - y[i]) / (j - i))
    sen_slope = float(np.median(slopes)) if slopes else 0.0

    # Sen's intercept
    sen_intercept = float(np.median(y - sen_slope * np.arange(n)))

    if p_value < 0.05:
        trend = "increasing" if sen_slope > 0 else "decreasing"
    else:
        trend = "no trend"

    logger.info("Trend: %s (slope=%.4f, p=%.4e)", trend, sen_slope, p_value)
    return WellTrendResult(
        trend=trend,
        p_value=float(p_value),
        slope=sen_slope,
        intercept=sen_intercept,
        z_statistic=float(z),
        method=method,
    )


def seasonal_decomposition(
    levels: pd.Series,
    period: int = 12,
) -> SeasonalResult:
    """Decompose a water-level series into trend, seasonal, and residual.

    Uses a centred moving-average approach (additive model) implemented
    with numpy — does **not** require statsmodels.

    Parameters
    ----------
    levels:
        Water-level series.
    period:
        Seasonal period (e.g. 12 for monthly, 365 for daily).

    Returns
    -------
    SeasonalResult
        Decomposed components and seasonal strength.

    Raises
    ------
    ValueError
        If series length is shorter than two full periods.
    """
    if len(levels) < 2 * period:
        raise ValueError(f"Series length ({len(levels)}) must be >= 2 × period ({2 * period}).")

    y = levels.values.astype(float)
    n = len(y)

    # Centred moving average for trend
    trend = np.full(n, np.nan)
    half = period // 2
    if period % 2 == 0:
        # Even period: two-step centred average
        ma = np.convolve(y, np.ones(period) / period, mode="valid")
        ma2 = (ma[:-1] + ma[1:]) / 2.0
        start = half
        end = start + len(ma2)
        trend[start:end] = ma2
    else:
        ma = np.convolve(y, np.ones(period) / period, mode="valid")
        start = half
        end = start + len(ma)
        trend[start:end] = ma

    # Detrended series
    detrended = y - trend

    # Average seasonal component for each phase
    seasonal_avg = np.zeros(period)
    for p in range(period):
        vals = detrended[p::period]
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            seasonal_avg[p] = float(np.mean(valid))

    # Centre the seasonal component (subtract its mean)
    seasonal_avg -= seasonal_avg.mean()

    # Tile seasonal component to full length
    seasonal = np.tile(seasonal_avg, n // period + 1)[:n]

    # Residual
    residual = y - trend - seasonal

    # Seasonal strength: 1 - Var(residual) / Var(detrended)
    valid_mask = ~np.isnan(trend)
    if np.sum(valid_mask) > 0:
        var_resid = float(np.nanvar(residual[valid_mask]))
        var_detrended = float(np.nanvar(detrended[valid_mask]))
        strength = max(0.0, 1.0 - var_resid / var_detrended) if var_detrended > 0 else 0.0
    else:
        strength = 0.0

    trend_series = pd.Series(trend, index=levels.index, name="trend")
    seasonal_series = pd.Series(seasonal, index=levels.index, name="seasonal")
    residual_series = pd.Series(residual, index=levels.index, name="residual")

    logger.info("Seasonal decomposition: period=%d, strength=%.3f", period, strength)
    return SeasonalResult(
        trend=trend_series,
        seasonal=seasonal_series,
        residual=residual_series,
        period=period,
        strength=strength,
    )


def recession_analysis(
    levels: pd.Series,
    min_recession_days: int = 5,
) -> RecessionResult:
    """Identify recession events and estimate aquifer time constants.

    Finds continuous declining segments in the water-level series,
    fits an exponential decay ``h(t) = h0 × exp(-t/τ)`` to each,
    and estimates the aquifer time constant τ.

    Parameters
    ----------
    levels:
        Water-level series (depth to water, with DatetimeIndex).
    min_recession_days:
        Minimum number of consecutive declining steps to qualify
        as a recession event.

    Returns
    -------
    RecessionResult
        Identified events, fitted constants, and master recession constant.

    Raises
    ------
    ValueError
        If no recession events are found.
    """
    y = levels.dropna().values.astype(float)
    n = len(y)

    # Identify recession segments (water level increasing = water table dropping
    # when measured as depth-to-water, but here we look for monotonic decline
    # in the raw values, which may represent hydraulic head)
    events: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(1, n):
        if y[i] < y[i - 1]:
            if start is None:
                start = i - 1
        else:
            if start is not None and (i - 1 - start) >= min_recession_days:
                events.append((start, i - 1))
            start = None
    if start is not None and (n - 1 - start) >= min_recession_days:
        events.append((start, n - 1))

    if not events:
        raise ValueError("No recession events found with the given criteria.")

    # Fit exponential decay to each event
    constants: list[float] = []
    fitted_curves: list[pd.Series] = []
    for s, e in events:
        segment = y[s : e + 1]
        t = np.arange(len(segment), dtype=float)
        h0 = segment[0]

        if h0 <= 0:
            continue

        # Linear regression on log(h/h0) = -t/τ
        ratios = segment / h0
        valid = ratios > 0
        if np.sum(valid) < 2:
            continue

        log_ratios = np.log(ratios[valid])
        t_valid = t[valid]

        slope, intercept, *_ = stats.linregress(t_valid, log_ratios)
        if slope >= 0:
            # Not a valid recession (should be negative slope)
            continue

        tau = -1.0 / slope
        constants.append(float(tau))

        fitted = h0 * np.exp(-t / tau)
        idx = levels.dropna().index[s : e + 1]
        fitted_curves.append(pd.Series(fitted, index=idx, name=f"recession_{len(constants)}"))

    if not constants:
        raise ValueError("Could not fit any recession events.")

    mean_const = float(np.mean(constants))

    # Master recession curve from all events combined
    all_log_ratios: list[float] = []
    all_t: list[float] = []
    for s, e in events:
        segment = y[s : e + 1]
        h0 = segment[0]
        if h0 <= 0:
            continue
        t = np.arange(len(segment), dtype=float)
        ratios = segment / h0
        valid = ratios > 0
        all_log_ratios.extend(np.log(ratios[valid]).tolist())
        all_t.extend(t[valid].tolist())

    if len(all_t) >= 2:
        slope, *_ = stats.linregress(all_t, all_log_ratios)
        master_const = -1.0 / slope if slope < 0 else mean_const
    else:
        master_const = mean_const

    logger.info("Recession: %d events, mean τ=%.1f days", len(constants), mean_const)
    return RecessionResult(
        events=events,
        recession_constants=constants,
        mean_constant=mean_const,
        master_constant=float(master_const),
        fitted_curves=fitted_curves,
    )


def storage_coefficient(
    levels: pd.Series,
    recharge_events: list[tuple],
    area_km2: float,
) -> float:
    """Estimate specific yield from recharge events.

    Computes Sy = ΔV / (Δh × A) for each recharge event and returns
    the median estimate.

    Parameters
    ----------
    levels:
        Water-level series with DatetimeIndex.
    recharge_events:
        List of ``(start_idx, end_idx, recharge_volume_m3)`` tuples.
    area_km2:
        Contributing area in km².

    Returns
    -------
    float
        Estimated specific yield (dimensionless).

    Raises
    ------
    ValueError
        If no valid recharge events are provided.
    """
    if not recharge_events:
        raise ValueError("No recharge events provided.")

    y = levels.values.astype(float)
    area_m2 = area_km2 * 1e6
    sy_estimates: list[float] = []

    for start, end, volume in recharge_events:
        dh = abs(y[end] - y[start])
        if dh > 0:
            sy = volume / (dh * area_m2)
            if 0 < sy < 1:
                sy_estimates.append(sy)

    if not sy_estimates:
        raise ValueError("Could not estimate Sy from any recharge event.")

    result = float(np.median(sy_estimates))
    logger.info("Specific yield: Sy=%.4f (from %d events)", result, len(sy_estimates))
    return result
