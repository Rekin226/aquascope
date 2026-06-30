"""
Climate indices for hydro-meteorological analysis.

Provides implementations of commonly used climate and drought indices
including the Palmer Drought Severity Index, aridity index, heat-wave
detection, and precipitation concentration metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ── Result dataclasses ──────────────────────────────────────────────────
@dataclass
class AridityResult:
    """Aridity index value and UNEP classification.

    Attributes
    ----------
    index : float
        Aridity index (P / PET).
    classification : str
        UNEP aridity classification.
    """

    index: float
    classification: str


@dataclass
class HeatWaveEvent:
    """A single heat-wave event.

    Attributes
    ----------
    start : object
        Start date / index label.
    end : object
        End date / index label.
    duration : int
        Number of consecutive days.
    peak_intensity : float
        Maximum exceedance above the threshold.
    """

    start: object
    end: object
    duration: int
    peak_intensity: float


@dataclass
class HeatWaveResult:
    """Summary of heat-wave detection.

    Attributes
    ----------
    n_events : int
        Total number of heat waves detected.
    max_duration : int
        Duration of the longest heat wave.
    mean_duration : float
        Mean duration across all heat waves.
    mean_intensity : float
        Mean peak intensity across events.
    events : list[HeatWaveEvent]
        Individual heat-wave events.
    """

    n_events: int
    max_duration: int
    mean_duration: float
    mean_intensity: float
    events: list[HeatWaveEvent] = field(default_factory=list)


@dataclass
class CDDResult:
    """Consecutive Dry Days result.

    Attributes
    ----------
    max_cdd : int
        Maximum CDD across all years.
    mean_cdd : float
        Mean annual maximum CDD.
    by_year : dict[int, int]
        Maximum CDD for each year.
    """

    max_cdd: int
    mean_cdd: float
    by_year: dict[int, int]


@dataclass
class CWDResult:
    """Consecutive Wet Days result.

    Attributes
    ----------
    max_cwd : int
        Maximum CWD across all years.
    mean_cwd : float
        Mean annual maximum CWD.
    by_year : dict[int, int]
        Maximum CWD for each year.
    """

    max_cwd: int
    mean_cwd: float
    by_year: dict[int, int]


# ── Helpers ─────────────────────────────────────────────────────────────
def _max_consecutive(mask: np.ndarray) -> int:
    """Return the length of the longest consecutive-True run in *mask*."""
    max_run = 0
    current = 0
    for v in mask:
        if v:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


# ── Public functions ────────────────────────────────────────────────────
def palmer_drought_severity_index(
    precip: pd.Series,
    pet: pd.Series,
    awc: float = 100.0,
) -> pd.Series:
    """Compute a simplified Palmer Drought Severity Index (PDSI).

    Uses a two-layer bucket water-balance model, derives the moisture
    anomaly z-index, and applies the PDSI recursion.

    Parameters
    ----------
    precip : pd.Series
        Monthly precipitation (mm), with a ``DatetimeIndex``.
    pet : pd.Series
        Monthly potential evapotranspiration (mm), same index as *precip*.
    awc : float
        Available water capacity of the soil (mm, default 100).

    Returns
    -------
    pd.Series
        PDSI values on the same index as *precip*.
    """
    n = len(precip)
    p = precip.values.astype(float)
    pe = pet.values.astype(float)

    # Two-layer bucket model
    ss = awc / 3.0  # surface layer capacity
    su = awc - ss  # underlying layer capacity
    s_s = ss  # current surface storage (start full)
    s_u = su  # current underlying storage

    et = np.zeros(n)
    r = np.zeros(n)    # recharge
    ro = np.zeros(n)   # runoff
    loss = np.zeros(n)  # loss

    for i in range(n):
        # Evapotranspiration — limited by available soil water
        if pe[i] <= s_s:
            et[i] = pe[i]
            s_s -= pe[i]
        else:
            et[i] = s_s
            remaining_pe = pe[i] - s_s
            s_s = 0.0
            if remaining_pe <= s_u:
                et[i] += remaining_pe
                s_u -= remaining_pe
            else:
                et[i] += s_u
                s_u = 0.0

        # Precipitation allocation
        available = p[i]
        # Recharge surface layer first
        recharge_s = min(available, ss - s_s)
        s_s += recharge_s
        available -= recharge_s
        # Then underlying layer
        recharge_u = min(available, su - s_u)
        s_u += recharge_u
        available -= recharge_u
        r[i] = recharge_s + recharge_u

        # Runoff is any leftover
        ro[i] = available
        loss[i] = pe[i] - et[i]

    # CAFEC coefficient (simplified)
    alpha = np.where(pe > 0, et / pe, 1.0)

    # Simplified: use long-term means for CAFEC
    alpha_mean = np.nanmean(alpha)
    pe_hat = alpha_mean * pe
    d = p - pe_hat  # moisture departure

    # Normalise to z-index using a simple scaling
    k = 1.0 / (np.std(d) + 1e-10)
    z = d * k

    # PDSI recursion: X_i = 0.897 * X_{i-1} + z_i / 3
    pdsi = np.zeros(n)
    for i in range(1, n):
        pdsi[i] = 0.897 * pdsi[i - 1] + z[i] / 3.0

    return pd.Series(pdsi, index=precip.index, name="PDSI")


def aridity_index(precip_annual: float, pet_annual: float) -> AridityResult:
    """Compute the UNEP aridity index.

    Parameters
    ----------
    precip_annual : float
        Total annual precipitation (mm).
    pet_annual : float
        Total annual potential evapotranspiration (mm).

    Returns
    -------
    AridityResult
        Index value and UNEP classification.

    Raises
    ------
    ValueError
        If *pet_annual* is zero or negative.
    """
    if pet_annual <= 0:
        raise ValueError("pet_annual must be positive")

    ai = precip_annual / pet_annual

    if ai < 0.03:
        classification = "hyper-arid"
    elif ai < 0.20:
        classification = "arid"
    elif ai < 0.50:
        classification = "semi-arid"
    elif ai < 0.65:
        classification = "dry sub-humid"
    else:
        classification = "humid"

    return AridityResult(index=ai, classification=classification)


def heat_wave_index(
    tmax: pd.Series,
    threshold_percentile: float = 90.0,
    min_duration: int = 3,
) -> HeatWaveResult:
    """Detect heat-wave events in a daily maximum-temperature series.

    A heat wave is defined as *min_duration* or more consecutive days
    where daily maximum temperature exceeds the *threshold_percentile*
    of the full record.

    Parameters
    ----------
    tmax : pd.Series
        Daily maximum temperature series with a ``DatetimeIndex``.
    threshold_percentile : float
        Percentile used as the exceedance threshold (default 90).
    min_duration : int
        Minimum consecutive days to qualify as a heat wave (default 3).

    Returns
    -------
    HeatWaveResult
        Count, durations, intensities, and individual events.
    """
    threshold = np.percentile(tmax.dropna().values, threshold_percentile)
    above = tmax > threshold

    events: list[HeatWaveEvent] = []
    i = 0
    idx = tmax.index
    vals = tmax.values.astype(float)
    n = len(tmax)

    while i < n:
        if above.iloc[i]:
            start = i
            while i < n and above.iloc[i]:
                i += 1
            duration = i - start
            if duration >= min_duration:
                peak = float(np.max(vals[start:i]) - threshold)
                events.append(
                    HeatWaveEvent(
                        start=idx[start],
                        end=idx[i - 1],
                        duration=duration,
                        peak_intensity=peak,
                    )
                )
        else:
            i += 1

    if not events:
        return HeatWaveResult(
            n_events=0, max_duration=0, mean_duration=0.0, mean_intensity=0.0, events=[]
        )

    durations = [e.duration for e in events]
    intensities = [e.peak_intensity for e in events]

    return HeatWaveResult(
        n_events=len(events),
        max_duration=int(np.max(durations)),
        mean_duration=float(np.mean(durations)),
        mean_intensity=float(np.mean(intensities)),
        events=events,
    )


def consecutive_dry_days(
    precip: pd.Series,
    threshold_mm: float = 1.0,
) -> CDDResult:
    """Compute maximum consecutive dry days per year.

    Parameters
    ----------
    precip : pd.Series
        Daily precipitation (mm) with a ``DatetimeIndex``.
    threshold_mm : float
        Days with precipitation below this are "dry" (default 1.0 mm).

    Returns
    -------
    CDDResult
        Maximum and mean CDD, broken down by year.
    """
    dry = precip < threshold_mm
    by_year: dict[int, int] = {}

    for year, group in dry.groupby(dry.index.year):
        by_year[int(year)] = _max_consecutive(group.values)

    if not by_year:
        return CDDResult(max_cdd=0, mean_cdd=0.0, by_year={})

    vals = list(by_year.values())
    return CDDResult(
        max_cdd=int(np.max(vals)),
        mean_cdd=float(np.mean(vals)),
        by_year=by_year,
    )


def consecutive_wet_days(
    precip: pd.Series,
    threshold_mm: float = 1.0,
) -> CWDResult:
    """Compute maximum consecutive wet days per year.

    Parameters
    ----------
    precip : pd.Series
        Daily precipitation (mm) with a ``DatetimeIndex``.
    threshold_mm : float
        Days with precipitation at or above this are "wet" (default 1.0 mm).

    Returns
    -------
    CWDResult
        Maximum and mean CWD, broken down by year.
    """
    wet = precip >= threshold_mm
    by_year: dict[int, int] = {}

    for year, group in wet.groupby(wet.index.year):
        by_year[int(year)] = _max_consecutive(group.values)

    if not by_year:
        return CWDResult(max_cwd=0, mean_cwd=0.0, by_year={})

    vals = list(by_year.values())
    return CWDResult(
        max_cwd=int(np.max(vals)),
        mean_cwd=float(np.mean(vals)),
        by_year=by_year,
    )


def precipitation_concentration_index(precip_monthly: pd.Series) -> float:
    """Compute the Precipitation Concentration Index (Oliver, 1980).

    PCI = (Σ p_i²) / (Σ p_i)² × 100,  summed over 12 months.

    A PCI of ~8.3 indicates uniform distribution; values > 20 indicate
    strong seasonality.

    Parameters
    ----------
    precip_monthly : pd.Series
        Monthly precipitation totals.  If the series spans multiple
        years, only the **first 12 values** are used; for multi-year
        analysis, group by year and call per year.

    Returns
    -------
    float
        PCI value.

    Raises
    ------
    ValueError
        If fewer than 12 monthly values are supplied.
    """
    vals = precip_monthly.dropna().values.astype(float)
    if len(vals) < 12:
        raise ValueError(f"Need at least 12 monthly values, got {len(vals)}")

    p = vals[:12]
    total = p.sum()
    if total == 0:
        return 0.0

    return float(np.sum(p**2) / total**2 * 100)

@dataclass
class SPIResult:
    """Result of Standardized Precipitation Index computation.

    Attributes
    ----------
    spi : pd.Series
        SPI values on the same monthly index as the input precipitation.
    scale : int
        Accumulation window in months used for the SPI.
    drought_classes : pd.Series
        Categorical drought/wet classification for each month.
    """

    spi: pd.Series
    scale: int
    drought_classes: pd.Series


def drought_class(spi_value: float) -> str:
    """Map an SPI value to a McKee et al. (1993) drought/wet category.

    Parameters
    ----------
    spi_value : float
        A single SPI value.

    Returns
    -------
    str
        One of: ``'extremely_wet'``, ``'severely_wet'``,
        ``'moderately_wet'``, ``'near_normal'``,
        ``'moderately_dry'``, ``'severely_dry'``, ``'extremely_dry'``,
        or ``'nan'`` for missing values.

    References
    ----------
    McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship
        of drought frequency and duration to time scales. Proceedings of
        the 8th Conference on Applied Climatology, 17-22 January 1993,
        Anaheim, California. American Meteorological Society.
    """
    if np.isnan(spi_value):
        return "nan"
    if spi_value >= 2.0:
        return "extremely_wet"
    if spi_value >= 1.5:
        return "severely_wet"
    if spi_value >= 1.0:
        return "moderately_wet"
    if spi_value > -1.0:
        return "near_normal"
    if spi_value > -1.5:
        return "moderately_dry"
    if spi_value > -2.0:
        return "severely_dry"
    return "extremely_dry"


def spi(
    precipitation: pd.Series,
    scale: int = 3,
) -> SPIResult:
    """Compute the Standardized Precipitation Index (SPI).

    Aggregates precipitation over a rolling *scale*-month window, fits a
    two-parameter gamma distribution **per calendar month** (to remove
    seasonality), and transforms the cumulative probability to the
    standard normal — yielding the SPI value.

    Zero-precipitation months are handled via a mixed distribution:
    the probability of zero precipitation is estimated separately and
    combined with the gamma CDF before the normal transform, following
    the WMO (2012) SPI User Guide.

    Parameters
    ----------
    precipitation : pd.Series
        Monthly precipitation totals (mm) with a ``DatetimeIndex``.
        Must span at least ``scale + 1`` months.
    scale : int
        Accumulation window in months (e.g. 1, 3, 6, 12).  Default 3.

    Returns
    -------
    SPIResult
        SPI values, scale used, and drought classifications.

    Raises
    ------
    ValueError
        If *scale* < 1 or the series has fewer than ``scale + 1`` values.

    References
    ----------
    McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship
        of drought frequency and duration to time scales. Proceedings of
        the 8th Conference on Applied Climatology. AMS.
    World Meteorological Organization (2012). Standardized Precipitation
        Index User Guide. WMO-No. 1090. Geneva.
    """
    from scipy import stats

    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if len(precipitation) < scale + 1:
        raise ValueError(
            f"Need at least scale+1={scale + 1} values, got {len(precipitation)}"
        )

    # Rolling accumulation over the scale window.
    rolled = precipitation.rolling(window=scale, min_periods=scale).sum()

    spi_vals = np.full(len(rolled), np.nan)
    idx = rolled.index

    # Fit and transform per calendar month to remove seasonality.
    for month in range(1, 13):
        month_mask = idx.month == month
        month_positions = np.where(month_mask)[0]
        month_vals = rolled.iloc[month_positions].values.astype(float)

        # Drop NaNs introduced by the rolling window.
        valid_mask = ~np.isnan(month_vals)
        valid_vals = month_vals[valid_mask]
        valid_positions = month_positions[valid_mask]

        if len(valid_vals) < 4:
            # Too few data points to fit reliably — leave as NaN.
            continue

        # Mixed distribution: probability of zero + gamma CDF for positives.
        n_total = len(valid_vals)
        zero_mask = valid_vals == 0.0
        n_zeros = int(zero_mask.sum())
        prob_zero = n_zeros / n_total

        pos_vals = valid_vals[~zero_mask]

        if len(pos_vals) < 3:
            # Almost all zeros — can't fit gamma; leave as NaN.
            continue

        try:
            shape, loc, scale_param = stats.gamma.fit(pos_vals, floc=0)
        except Exception:
            continue

        # Combined CDF: P(X <= x) = prob_zero + (1 - prob_zero) * gamma_cdf(x)
        gamma_cdf = stats.gamma.cdf(valid_vals, shape, loc=loc, scale=scale_param)
        combined_prob = prob_zero + (1.0 - prob_zero) * gamma_cdf

        # Clip to avoid ±inf from the normal PPF at the boundaries.
        combined_prob = np.clip(combined_prob, 1e-6, 1 - 1e-6)

        spi_month = stats.norm.ppf(combined_prob)
        spi_vals[valid_positions] = spi_month

    spi_series = pd.Series(spi_vals, index=idx, name=f"SPI-{scale}")

    classes = spi_series.apply(
        lambda v: drought_class(v) if not np.isnan(v) else "nan"
    )
    classes.name = f"drought_class_SPI-{scale}"

    logger.info(
        "SPI-%d computed over %d months; %d valid values",
        scale,
        len(precipitation),
        int(np.sum(~np.isnan(spi_vals))),
    )

    return SPIResult(spi=spi_series, scale=scale, drought_classes=classes)
