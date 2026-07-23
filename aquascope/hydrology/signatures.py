"""Hydrological signature metrics for catchment characterisation.

Hydrological signatures are standardised metrics that summarise the key
features of a catchment's flow regime.  They are widely used for:

- **Model evaluation** — comparing simulated vs observed signatures.
- **Regionalisation** — transferring information to ungauged basins.
- **Catchment classification** — clustering catchments by behaviour.
- **Change detection** — identifying shifts in flow regime over time.

The central entry-point is :func:`compute_signatures`, which returns a
:class:`SignatureReport` dataclass containing ~20 metrics covering flow
magnitude, variability, high/low flow behaviour, timing, rate of change,
and recession characteristics.

Individual signature functions (e.g. :func:`flashiness_index`,
:func:`seasonality_index`) are also exposed for targeted use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from aquascope.hydrology.baseflow import lyne_hollick

logger = logging.getLogger(__name__)


@dataclass
class SignatureReport:
    """Complete set of hydrological signatures for a streamflow record.

    Attributes are grouped by the aspect of the flow regime they describe.
    Fields set to ``None`` require optional inputs (e.g. precipitation).
    """

    # Flow magnitude
    mean_flow: float
    median_flow: float
    q5: float  # 5th percentile (high flow)
    q95: float  # 95th percentile (low flow)
    q5_q95_ratio: float  # flashiness indicator

    # Flow variability
    cv: float  # coefficient of variation
    iqr: float  # interquartile range

    # High flow signatures
    high_flow_frequency: float  # days/year > 3*median
    high_flow_duration: float  # mean consecutive days > median
    q_peak_mean: float  # mean annual peak / mean flow

    # Low flow signatures
    low_flow_frequency: float  # days/year < 0.2*median
    low_flow_duration: float  # mean consecutive days < 0.2*median
    baseflow_index: float  # baseflow / total flow
    zero_flow_fraction: float  # fraction of zero-flow days

    # Timing & seasonality
    peak_month: int  # month with highest mean flow (1-12)
    seasonality_index: float  # 0=uniform, 1=all flow in one month (Markham)

    # Rate of change
    rising_limb_density: float  # positive dQ/dt days / total days
    flashiness_index: float  # Richards-Baker flashiness index

    # Recession
    mean_recession_constant: float  # average -dQ/dt during recession

    # Overall (require precipitation)
    runoff_ratio: float | None  # total_Q / total_P if precip provided
    elasticity: float | None  # Sankarasubramanian elasticity if precip provided


# ---------------------------------------------------------------------------
# Individual signature functions
# ---------------------------------------------------------------------------


def flashiness_index(discharge: pd.Series) -> float:
    """Richards-Baker Flashiness Index.

    .. math:: FI = \\frac{\\sum |Q_i - Q_{i-1}|}{\\sum Q_i}

    Higher values indicate a more flashy/responsive catchment.

    Parameters:
        discharge: Daily discharge time series with DatetimeIndex.

    Returns:
        Flashiness index (dimensionless, ≥ 0).
    """
    q = discharge.dropna().values.astype(float)
    total = q.sum()
    if total == 0:
        return 0.0
    return float(np.abs(np.diff(q)).sum() / total)


def seasonality_index(discharge: pd.Series) -> tuple[float, int]:
    """Markham's seasonality index and concentration month.

    Monthly mean flows are treated as vectors with direction equal to the
    month's angular position on a unit circle.  The resultant's magnitude
    (normalised) gives the seasonality index and its direction gives the
    peak month.

    Parameters:
        discharge: Daily discharge time series with DatetimeIndex.

    Returns:
        Tuple of ``(index, peak_month)`` where *index* ranges from 0
        (uniform flow throughout the year) to 1 (all flow concentrated
        in a single month), and *peak_month* is 1–12.
    """
    monthly = discharge.groupby(discharge.index.month).mean()
    if monthly.sum() == 0:
        return 0.0, 1

    angles = np.array([2 * np.pi * (m - 1) / 12 for m in monthly.index])
    weights = monthly.values.astype(float)
    total_weight = weights.sum()

    x = (weights * np.cos(angles)).sum() / total_weight
    y = (weights * np.sin(angles)).sum() / total_weight

    magnitude = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += 2 * np.pi

    peak_month = int(np.round(angle * 12 / (2 * np.pi))) + 1
    if peak_month > 12:
        peak_month = 1

    return float(magnitude), peak_month


def flow_elasticity(discharge: pd.Series, precipitation: pd.Series) -> float:
    """Sankarasubramanian precipitation-streamflow elasticity.

    Computed year-by-year as:

    .. math:: E = \\text{median}\\left(\\frac{dQ / \\bar{Q}}{dP / \\bar{P}}\\right)

    where *dQ* and *dP* are annual departures from the long-term mean.
    An elasticity > 1 means streamflow is proportionally more variable
    than precipitation.

    Parameters:
        discharge: Daily discharge time series with DatetimeIndex.
        precipitation: Daily precipitation with the same DatetimeIndex.

    Returns:
        Elasticity coefficient (dimensionless).

    Raises:
        ValueError: If fewer than 2 complete years are available.
    """
    annual_q = discharge.groupby(discharge.index.year).sum()
    annual_p = precipitation.groupby(precipitation.index.year).sum()

    common_years = annual_q.index.intersection(annual_p.index)
    if len(common_years) < 2:
        raise ValueError("Need at least 2 complete years for elasticity calculation.")

    annual_q = annual_q.loc[common_years]
    annual_p = annual_p.loc[common_years]

    q_mean = annual_q.mean()
    p_mean = annual_p.mean()

    if q_mean == 0 or p_mean == 0:
        return 0.0

    dq = annual_q - q_mean
    dp = annual_p - p_mean

    # Avoid division by zero for years where dP ≈ 0
    valid = dp.abs() > 1e-12
    if valid.sum() < 2:
        return 0.0

    ratios = (dq[valid] / q_mean) / (dp[valid] / p_mean)
    return float(ratios.median())


def baseflow_index_simple(discharge: pd.Series) -> float:
    """Quick BFI using the Lyne–Hollick 1-pass digital filter.

    Uses ``alpha=0.925`` and a single forward pass for speed.  For a more
    robust estimate use :func:`aquascope.hydrology.baseflow.lyne_hollick`
    directly with multiple passes.

    Parameters:
        discharge: Daily discharge time series with DatetimeIndex.

    Returns:
        Baseflow index (0–1).
    """
    result = lyne_hollick(discharge, alpha=0.925, n_passes=1)
    return result.bfi


def recession_constant(discharge: pd.Series, min_length: int = 5) -> float:
    """Mean recession constant from all recession segments.

    A recession segment is a run of consecutive days where discharge
    decreases.  For each segment of at least *min_length* days the
    exponential decay rate *k* is estimated by linear regression of
    ``ln(Q)`` against time.  The median *k* across all segments is
    returned.

    Parameters:
        discharge: Daily discharge time series with DatetimeIndex.
        min_length: Minimum number of consecutive falling days to qualify
            as a recession segment.

    Returns:
        Median recession constant *k* (day⁻¹, positive).
    """
    q = discharge.dropna().values.astype(float)
    n = len(q)

    segments: list[np.ndarray] = []
    start = 0
    for i in range(1, n):
        if q[i] >= q[i - 1]:
            if i - start >= min_length:
                seg = q[start:i]
                if seg.min() > 0:
                    segments.append(seg)
            start = i

    # Handle trailing segment
    if n - start >= min_length:
        seg = q[start:n]
        if seg.min() > 0:
            segments.append(seg)

    if not segments:
        return 0.0

    k_values: list[float] = []
    for seg in segments:
        ln_q = np.log(seg)
        t = np.arange(len(seg), dtype=float)
        # Linear regression: ln(Q) = -k*t + c  →  slope = -k
        slope, _ = np.polyfit(t, ln_q, 1)
        k_values.append(-slope)

    return float(np.median(k_values))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _mean_event_duration(discharge: pd.Series, threshold: float) -> float:
    """Mean duration (days) of consecutive periods where discharge > threshold."""
    above = (discharge > threshold).astype(int).values
    durations: list[int] = []
    count = 0
    for val in above:
        if val:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0


def _mean_event_duration_below(discharge: pd.Series, threshold: float) -> float:
    """Mean duration (days) of consecutive periods where discharge < threshold."""
    below = (discharge < threshold).astype(int).values
    durations: list[int] = []
    count = 0
    for val in below:
        if val:
            count += 1
        elif count > 0:
            durations.append(count)
            count = 0
    if count > 0:
        durations.append(count)
    return float(np.mean(durations)) if durations else 0.0


def _days_per_year(discharge: pd.Series, condition: np.ndarray) -> float:
    """Average number of days per year satisfying *condition*."""
    n_years = max(1, len(discharge) / 365.25)
    return float(condition.sum() / n_years)


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------


def compute_signatures(
    discharge: pd.Series,
    precipitation: pd.Series | None = None,
    area_km2: float | None = None,
) -> SignatureReport:
    """Compute comprehensive hydrological signatures from daily streamflow.

    Parameters:
        discharge: Daily discharge time series (pd.Series with DatetimeIndex).
            Must contain at least 365 non-NaN values.
        precipitation: Optional daily precipitation series aligned with
            *discharge*.  When provided, runoff ratio and elasticity are
            calculated.
        area_km2: Optional catchment area in km².  Currently reserved for
            future unit-conversion but not required for any signature.

    Returns:
        :class:`SignatureReport` with all computed signatures.

    Raises:
        ValueError: If *discharge* has fewer than 365 non-NaN values.
    """
    q = discharge.dropna()
    if len(q) < 365:
        raise ValueError(f"Need at least 365 discharge values, got {len(q)}.")

    q_vals = q.values.astype(float)

    # -- magnitude --
    mean_q = float(q_vals.mean())
    median_q = float(np.median(q_vals))
    q5 = float(np.percentile(q_vals, 5))
    q95 = float(np.percentile(q_vals, 95))
    q5_q95 = q5 / q95 if q95 > 0 else float("inf")

    # -- variability --
    cv = float(q_vals.std() / mean_q) if mean_q > 0 else 0.0
    iqr = float(np.percentile(q_vals, 75) - np.percentile(q_vals, 25))

    # -- high flow --
    high_thresh = 3 * median_q
    high_mask = q_vals > high_thresh
    high_freq = _days_per_year(q, high_mask)
    high_dur = _mean_event_duration(q, median_q)

    annual_peaks = q.groupby(q.index.year).max()
    q_peak_mean = float(annual_peaks.mean() / mean_q) if mean_q > 0 else 0.0

    # -- low flow --
    low_thresh = 0.2 * median_q
    low_mask = q_vals < low_thresh
    low_freq = _days_per_year(q, low_mask)
    low_dur = _mean_event_duration_below(q, low_thresh)

    bfi = baseflow_index_simple(q)
    zero_frac = float((q_vals == 0).sum() / len(q_vals))

    # -- timing --
    si, peak_m = seasonality_index(q)

    # -- rate of change --
    diffs = np.diff(q_vals)
    rising_density = float((diffs > 0).sum() / len(q_vals))
    fi = flashiness_index(q)

    # -- recession --
    rec_k = recession_constant(q)

    # -- precipitation-dependent --
    runoff_ratio: float | None = None
    elast: float | None = None

    if precipitation is not None:
        p = precipitation.dropna()
        total_p = p.sum()
        total_q = q.sum()
        if total_p > 0:
            runoff_ratio = float(total_q / total_p)
        try:
            elast = flow_elasticity(q, p)
        except ValueError:
            logger.warning("Could not compute elasticity — insufficient overlapping years.")
            elast = None

    logger.info(
        "Computed signatures: mean=%.2f, BFI=%.3f, FI=%.4f, SI=%.3f",
        mean_q, bfi, fi, si,
    )

    return SignatureReport(
        mean_flow=mean_q,
        median_flow=median_q,
        q5=q5,
        q95=q95,
        q5_q95_ratio=q5_q95,
        cv=cv,
        iqr=iqr,
        high_flow_frequency=high_freq,
        high_flow_duration=high_dur,
        q_peak_mean=q_peak_mean,
        low_flow_frequency=low_freq,
        low_flow_duration=low_dur,
        baseflow_index=bfi,
        zero_flow_fraction=zero_frac,
        peak_month=peak_m,
        seasonality_index=si,
        rising_limb_density=rising_density,
        flashiness_index=fi,
        mean_recession_constant=rec_k,
        runoff_ratio=runoff_ratio,
        elasticity=elast,
    )


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def compare_signatures(sig1: SignatureReport, sig2: SignatureReport) -> dict[str, float]:
    """Compare two signature reports field-by-field.

    Parameters:
        sig1: First :class:`SignatureReport`.
        sig2: Second :class:`SignatureReport`.

    Returns:
        Dictionary of ``{field_name: absolute_percent_difference}`` for every
        numeric field.  Fields that are ``None`` in either report are skipped.
    """
    result: dict[str, float] = {}
    for f in fields(sig1):
        v1 = getattr(sig1, f.name)
        v2 = getattr(sig2, f.name)
        if v1 is None or v2 is None:
            continue
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            continue
        denom = abs(v1) if abs(v1) > 0 else 1.0
        result[f.name] = abs(v1 - v2) / denom * 100.0
    return result


def similarity_score(
    sig1: SignatureReport,
    sig2: SignatureReport,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute overall similarity between two catchments.

    The score is a weighted Euclidean distance in normalised signature
    space.  A score of 0 means the reports are identical; higher values
    indicate greater dissimilarity.

    Parameters:
        sig1: First :class:`SignatureReport`.
        sig2: Second :class:`SignatureReport`.
        weights: Optional mapping of field name → weight.  Fields not in
            the dict receive a weight of 1.0.  Defaults emphasise BFI,
            flashiness, seasonality, and runoff ratio.

    Returns:
        Weighted Euclidean distance (≥ 0).
    """
    if weights is None:
        weights = {
            "baseflow_index": 2.0,
            "flashiness_index": 2.0,
            "seasonality_index": 2.0,
            "runoff_ratio": 2.0,
        }

    dist_sq = 0.0
    for f in fields(sig1):
        v1 = getattr(sig1, f.name)
        v2 = getattr(sig2, f.name)
        if v1 is None or v2 is None:
            continue
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            continue
        # Normalise by the mean of the two values (or 1 if both zero)
        norm = (abs(v1) + abs(v2)) / 2.0
        if norm == 0:
            continue
        w = weights.get(f.name, 1.0)
        dist_sq += w * ((v1 - v2) / norm) ** 2

    return float(np.sqrt(dist_sq))
