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
