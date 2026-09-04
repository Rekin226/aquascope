"""
Climate indices for hydro-meteorological analysis.

Provides implementations of commonly used climate and drought indices
including the Palmer Drought Severity Index, aridity index, heat-wave
detection, precipitation concentration metrics, and the standardised drought
indices SPI (McKee et al. 1993) and SPEI (Vicente-Serrano et al. 2010) with a
Thornthwaite PET helper.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
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


# ── Standardised indices: SPI and SPEI share one core ───────────────────
def _accumulate(series: pd.Series, scale: int, what: str) -> pd.Series:
    """``series`` summed over ``scale`` months, the first ``scale - 1`` months dropped."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{what} must have a DatetimeIndex.")
    if scale < 1:
        raise ValueError("scale must be >= 1 month.")
    s = series.sort_index().astype(float)
    acc = s.rolling(scale).sum().dropna()
    if acc.empty:
        raise ValueError("Series too short for the requested accumulation scale.")
    return acc


def _standardize(
    acc: pd.Series,
    *,
    name: str,
    per_month: bool,
    min_per_group: int,
    probabilities: Callable[[np.ndarray, int], np.ndarray | None],
) -> pd.Series:
    """The core SPI and SPEI share: a fitted CDF per calendar month, mapped to standard-normal scores.

    ``probabilities(values, min_per_group)`` returns the non-exceedance
    probability of each value under the distribution fitted to that group, or
    ``None`` when the group is too small to fit (left ``NaN``).
    """
    out = pd.Series(np.nan, index=acc.index, dtype=float, name=name)
    groups = range(1, 13) if per_month else [None]
    for g in groups:
        idx = acc.index if g is None else acc.index[acc.index.month == g]
        vals = acc.loc[idx].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        cdf = probabilities(vals, min_per_group)
        if cdf is None:
            logger.debug("%s group %s has too few observations (< %d); left NaN.", name, g, min_per_group)
            continue
        cdf = np.clip(np.asarray(cdf, dtype=float), 1e-6, 1 - 1e-6)
        out.loc[idx] = stats.norm.ppf(cdf)
    return out


def _gamma_with_zero_mass(vals: np.ndarray, min_per_group: int) -> np.ndarray | None:
    """SPI's fit: a point mass at zero (probability q) and a gamma on the positive values."""
    pos = vals[vals > 0]
    if len(pos) < min_per_group:
        return None
    q = float((vals == 0).mean())
    a, loc, scl = stats.gamma.fit(pos, floc=0.0)
    cdf = q + (1.0 - q) * stats.gamma.cdf(vals, a, loc=loc, scale=scl)
    return np.where(vals == 0, q / 2.0, cdf)  # zeros -> lower half of the mass


def standardized_precipitation_index(
    precip_monthly: pd.Series,
    scale: int = 3,
    per_month: bool = True,
    min_per_group: int = 10,
) -> pd.Series:
    """Standardized Precipitation Index (SPI), McKee et al. (1993).

    Monthly precipitation is accumulated over ``scale`` months, a gamma
    distribution is fitted (with explicit zero handling), and the cumulative
    probability is mapped to a standard-normal score. The result is unitless,
    centred on zero, with SPI < -1 indicating meteorological drought; it is
    directly comparable to the Standardised Groundwater Index for
    drought-propagation analysis.

    Parameters
    ----------
    precip_monthly:
        Monthly precipitation totals (mm) with a ``DatetimeIndex``.
    scale:
        Accumulation period in months (e.g. 3 -> SPI-3). Larger scales capture
        longer droughts that propagate to groundwater.
    per_month:
        When ``True`` (default), fit a separate gamma per calendar month, which
        removes the seasonal cycle (standard practice). When ``False``, fit one
        gamma to all accumulated values.
    min_per_group:
        Minimum positive values needed to fit a gamma for a group; groups with
        fewer yield ``NaN``.

    Returns
    -------
    pd.Series
        SPI indexed like the accumulated series, named ``"spi"``.
    """
    acc = _accumulate(precip_monthly, scale, "precip_monthly")
    return _standardize(acc, name="spi", per_month=per_month, min_per_group=min_per_group,
                        probabilities=_gamma_with_zero_mass)


def _sample_lmoments(values: np.ndarray) -> tuple[float, float, float]:
    """The first three sample L-moments (Hosking 1990) from unbiased probability-weighted moments."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1, dtype=float)
    b0 = float(x.mean())
    b1 = float(np.sum((i - 1) / (n - 1) * x) / n)
    b2 = float(np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n)
    return b0, 2.0 * b1 - b0, 6.0 * b2 - 6.0 * b1 + b0


def fit_generalized_logistic_lmoments(values: np.ndarray | pd.Series) -> tuple[float, float, float]:
    """Generalized logistic (Hosking 1990) fitted by L-moments: ``(xi, alpha, k)`` location, scale, shape.

    The distribution SPEI is fitted with in practice (the SPEI R package,
    Begueria et al. 2014): for negative shape it is the three-parameter
    log-logistic of Vicente-Serrano et al. (2010), for positive shape its
    mirror image, and for zero shape the logistic, so a calendar month whose
    water balance is left-skewed or symmetric is fitted as well as a
    right-skewed one. Raises ``ValueError`` on fewer than four values, no
    spread, or an L-skewness outside the distribution's range.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 4:
        raise ValueError("a generalized logistic fit needs at least four values")
    l1, l2, l3 = _sample_lmoments(x)
    if not np.isfinite(l2) or l2 <= 1e-9 * max(1.0, abs(l1)):
        raise ValueError("degenerate sample: no spread to fit")
    t3 = l3 / l2
    k = -t3
    if abs(k) < 1e-9:
        return l1, l2, 0.0
    if abs(k) >= 1.0:
        raise ValueError(f"L-skewness {t3:.3g} is outside the generalized logistic range")
    alpha = l2 * math.sin(k * math.pi) / (k * math.pi)
    xi = l1 - alpha * (1.0 / k - math.pi / math.sin(k * math.pi))
    return float(xi), float(alpha), float(k)


def fit_log_logistic_lmoments(values: np.ndarray | pd.Series) -> tuple[float, float, float]:
    """Three-parameter log-logistic by L-moments, in the ``(alpha, beta, gamma)`` of Vicente-Serrano et al. (2010).

    ``F(x) = [1 + (alpha / (x - gamma)) ** beta] ** -1``: scale, shape and
    origin, read off the generalized-logistic fit (``beta = -1/k``,
    ``alpha = alpha_glo * beta``, ``gamma = xi - alpha``). Raises
    ``ValueError`` when the sample is not right-skewed (``k >= 0``), where the
    log-logistic proper does not apply and
    :func:`fit_generalized_logistic_lmoments` is the fit to use.
    """
    xi, a, k = fit_generalized_logistic_lmoments(values)
    if k >= 0:
        raise ValueError(f"L-skewness {-k:.3g} is not positive: no log-logistic fits, use the generalized logistic")
    beta = -1.0 / k
    alpha = a * beta
    return float(alpha), float(beta), float(xi - alpha)


def _glo_cdf(x: np.ndarray, xi: float, alpha: float, k: float) -> np.ndarray:
    """Generalized logistic CDF; values beyond the bound map to 0 (below) or 1 (above)."""
    x = np.asarray(x, dtype=float)
    if k == 0.0:
        y = (x - xi) / alpha
    else:
        arg = 1.0 - k * (x - xi) / alpha
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(arg > 0, -np.log(np.where(arg > 0, arg, 1.0)) / k, -np.inf if k < 0 else np.inf)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-y))


def _log_logistic(vals: np.ndarray, min_per_group: int) -> np.ndarray | None:
    """SPEI's fit: the generalized logistic (log-logistic) by L-moments on the climatic water balance."""
    finite = vals[np.isfinite(vals)]
    if len(finite) < min_per_group:
        return None
    try:
        xi, alpha, k = fit_generalized_logistic_lmoments(finite)
    except ValueError as exc:
        logger.warning("SPEI: %s; normal scores used for this calendar month instead.", exc)
        sd = float(finite.std(ddof=1))
        if not sd:
            return None
        return stats.norm.cdf(vals, loc=float(finite.mean()), scale=sd)
    return _glo_cdf(vals, xi, alpha, k)


def standardized_precipitation_evapotranspiration_index(
    precip_monthly: pd.Series,
    pet_monthly: pd.Series,
    timescale: int = 3,
    *,
    per_month: bool = True,
    min_per_group: int = 10,
) -> pd.Series:
    """Standardized Precipitation-Evapotranspiration Index (SPEI), Vicente-Serrano et al. (2010).

    The SPI machinery applied to the climatic water balance ``D = P - PET``:
    ``D`` is accumulated over ``timescale`` months, a three-parameter
    log-logistic distribution is fitted per calendar month by L-moments
    (unbiased probability-weighted moments), and the cumulative probability
    is mapped to a standard-normal score. A gamma cannot be used because ``D``
    takes negative values; the log-logistic is the fit of the original paper
    (Vicente-Serrano et al. 2010) and, in its generalized-logistic form
    (:func:`fit_generalized_logistic_lmoments`, Hosking 1990), the one the
    SPEI R package settled on (Begueria et al. 2014), because it also covers
    a calendar month whose balance is symmetric or left-skewed. A month the
    fit cannot handle (no spread) falls back to normal scores with a warning.
    Values beyond the fitted bound map to the clip (about -4.75 or 4.75).

    SPEI sees what SPI cannot: a drought driven by evaporative demand under
    warming, with unchanged rainfall. Under a warming trend SPEI runs drier
    than SPI, which is the reason the drought playbook prefers it where a
    temperature or PET series exists.

    Parameters
    ----------
    precip_monthly:
        Monthly precipitation totals (mm) with a ``DatetimeIndex``.
    pet_monthly:
        Monthly potential evapotranspiration (mm) on the same calendar
        (:func:`thornthwaite_pet` from temperature, or a FAO-56 ET0 sum). Only
        months present in both series are used.
    timescale:
        Accumulation period in months (1, 3, 6, 12 are the usual set).
    per_month, min_per_group:
        As for :func:`standardized_precipitation_index`.

    Returns
    -------
    pd.Series
        SPEI indexed like the accumulated series, named ``"spei"``.
    """
    if not isinstance(precip_monthly.index, pd.DatetimeIndex) or not isinstance(pet_monthly.index, pd.DatetimeIndex):
        raise ValueError("precip_monthly and pet_monthly must have a DatetimeIndex.")
    p, e = precip_monthly.astype(float).align(pet_monthly.astype(float), join="inner")
    if p.empty:
        raise ValueError("precip_monthly and pet_monthly share no months.")
    balance = (p - e).dropna()
    acc = _accumulate(balance, timescale, "the climatic water balance")
    return _standardize(acc, name="spei", per_month=per_month, min_per_group=min_per_group,
                        probabilities=_log_logistic)


def thornthwaite_pet(temperature_monthly: pd.Series, latitude: float) -> pd.Series:
    """Monthly potential evapotranspiration (mm) after Thornthwaite (1948), from temperature alone.

    The formulation SPEI was introduced with (Vicente-Serrano et al. 2010):
    ``PET = 16 K (10 T / I) ** a`` for a month of mean temperature ``T`` (deg C,
    negative months give zero), with the annual heat index ``I`` summed over
    the record's twelve climatological monthly means, the exponent ``a`` a
    cubic in ``I``, and ``K`` the correction for the month's day length at
    ``latitude`` and its number of days. It needs nothing but temperature, so
    it runs on the ERA5 temperature :func:`aquascope.explore.anywhere` returns;
    it is a temperature-only approximation and FAO-56 Penman-Monteith
    (:func:`aquascope.agri.eto.penman_monteith_series`) is the better PET
    where humidity, wind and radiation exist.

    Parameters
    ----------
    temperature_monthly:
        Monthly mean air temperature (deg C) with a ``DatetimeIndex``.
    latitude:
        Degrees north (negative south), for the day-length correction.

    Returns
    -------
    pd.Series
        PET in mm per month, indexed like the input, named ``"pet"``.
    """
    if not isinstance(temperature_monthly.index, pd.DatetimeIndex):
        raise ValueError("temperature_monthly must have a DatetimeIndex.")
    t = temperature_monthly.sort_index().astype(float)
    clim = t.groupby(t.index.month).mean().clip(lower=0.0)
    heat = float(((clim / 5.0) ** 1.514).sum())
    if heat <= 0:
        return pd.Series(0.0, index=t.index, name="pet")
    a = 6.75e-7 * heat**3 - 7.71e-5 * heat**2 + 1.792e-2 * heat + 0.49239
    unadjusted = 16.0 * (10.0 * t.clip(lower=0.0) / heat) ** a
    mid = pd.DatetimeIndex([pd.Timestamp(year=ts.year, month=ts.month, day=15) for ts in t.index])
    declination = 0.4093 * np.sin(2.0 * np.pi * (mid.dayofyear.to_numpy() - 82) / 365.0)
    phi = math.radians(float(latitude))
    x = np.clip(-np.tan(phi) * np.tan(declination), -1.0, 1.0)
    day_hours = 24.0 * np.arccos(x) / np.pi
    k = (day_hours / 12.0) * (t.index.days_in_month.to_numpy() / 30.0)
    return pd.Series(unadjusted.to_numpy() * k, index=t.index, name="pet")


def drought_class(spi_value: float) -> str:
    """Classify an SPI value using the McKee et al. (1993) categories.

    Parameters
    ----------
    spi_value:
        Standardized Precipitation Index value.

    Returns
    -------
    str
        One of ``extremely_wet``, ``very_wet``, ``moderately_wet``,
        ``normal``, ``moderately_dry``, ``severely_dry``,
        ``extremely_dry``, or ``unknown`` for a missing value.
    """
    if pd.isna(spi_value):
        return "unknown"
    if spi_value >= 2.0:
        return "extremely_wet"
    if spi_value >= 1.5:
        return "very_wet"
    if spi_value >= 1.0:
        return "moderately_wet"
    if spi_value > -1.0:
        return "normal"
    if spi_value > -1.5:
        return "moderately_dry"
    if spi_value > -2.0:
        return "severely_dry"
    return "extremely_dry"
