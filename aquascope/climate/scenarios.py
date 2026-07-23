"""
Climate scenario analysis tools.

Functions for analysing how extreme-event statistics shift under
different climate scenarios, including return-period analysis, IDF
curve adjustment, and drought / wet-spell frequency analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ── Result dataclasses ──────────────────────────────────────────────────
@dataclass
class ReturnPeriodShift:
    """How return periods change between historical and future climate.

    Attributes
    ----------
    return_periods : list[int]
        Original return periods (years).
    hist_quantiles : list[float]
        Quantile values for each return period under historical climate.
    future_quantiles : list[float]
        Quantile values under the future climate.
    shift_factors : list[float]
        Ratio of future to historical quantile for each return period.
    """

    return_periods: list[int]
    hist_quantiles: list[float]
    future_quantiles: list[float]
    shift_factors: list[float]


@dataclass
class DroughtStats:
    """Drought frequency and severity statistics.

    Attributes
    ----------
    n_events : int
        Number of drought events detected.
    mean_duration : float
        Mean drought duration (time steps).
    max_duration : int
        Maximum drought duration.
    mean_severity : float
        Mean cumulative deficit during drought events.
    total_deficit : float
        Total precipitation deficit across all drought events.
    """

    n_events: int
    mean_duration: float
    max_duration: int
    mean_severity: float
    total_deficit: float


@dataclass
class WetSpellStats:
    """Wet-spell frequency and duration statistics.

    Attributes
    ----------
    n_spells : int
        Number of wet spells detected.
    mean_duration : float
        Mean wet-spell duration (time steps).
    max_duration : int
        Maximum wet-spell duration.
    mean_intensity : float
        Mean precipitation intensity during wet spells.
    """

    n_spells: int
    mean_duration: float
    max_duration: int
    mean_intensity: float


# ── Helper ──────────────────────────────────────────────────────────────
def _run_lengths(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, length) pairs for each consecutive run of *True* in *mask*."""
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            start = i
            while i < n and mask[i]:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1
    return runs


# ── Public functions ────────────────────────────────────────────────────
def return_period_shift(
    hist_data: pd.Series,
    future_data: pd.Series,
    return_periods: list[int] | None = None,
) -> ReturnPeriodShift:
    """Analyse how return periods shift under a future climate scenario.

    Fits a GEV distribution to annual maxima of *hist_data* and
    *future_data*, then computes quantiles for the requested return periods.

    Parameters
    ----------
    hist_data : pd.Series
        Historical daily or sub-daily series with a ``DatetimeIndex``.
    future_data : pd.Series
        Future-scenario series with a ``DatetimeIndex``.
    return_periods : list[int] | None
        Return periods in years (default ``[2, 5, 10, 25, 50, 100]``).

    Returns
    -------
    ReturnPeriodShift
        Quantiles and shift factors for each return period.
    """
    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100]

    hist_am = hist_data.groupby(hist_data.index.year).max()
    future_am = future_data.groupby(future_data.index.year).max()

    # Fit GEV to annual maxima
    h_params = sp_stats.genextreme.fit(hist_am.values)
    f_params = sp_stats.genextreme.fit(future_am.values)

    hist_q: list[float] = []
    future_q: list[float] = []
    shifts: list[float] = []

    for rp in return_periods:
        exceedance = 1.0 / rp
        h_val = float(sp_stats.genextreme.isf(exceedance, *h_params))
        f_val = float(sp_stats.genextreme.isf(exceedance, *f_params))
        hist_q.append(h_val)
        future_q.append(f_val)
        shifts.append(f_val / h_val if h_val != 0 else float("nan"))

    return ReturnPeriodShift(
        return_periods=return_periods,
        hist_quantiles=hist_q,
        future_quantiles=future_q,
        shift_factors=shifts,
    )


def idf_adjustment(
    hist_intensities: np.ndarray,
    durations: np.ndarray,
    future_factor: float | np.ndarray,
) -> np.ndarray:
    """Scale IDF-curve intensities by a climate-change factor.

    Parameters
    ----------
    hist_intensities : np.ndarray
        Historical rainfall intensities (shape matches *durations*).
    durations : np.ndarray
        Storm durations corresponding to each intensity.
    future_factor : float | np.ndarray
        Multiplicative scaling factor(s).  A scalar applies uniformly;
        an array must match the shape of *hist_intensities*.

    Returns
    -------
    np.ndarray
        Adjusted intensities.
    """
    return np.asarray(hist_intensities) * np.asarray(future_factor)


def drought_frequency(
    precip: pd.Series,
    threshold_percentile: float = 20.0,
) -> DroughtStats:
    """Analyse drought frequency, duration, and severity.

    A drought event is a consecutive period where precipitation falls
    below the given percentile threshold.

    Parameters
    ----------
    precip : pd.Series
        Precipitation time series.
    threshold_percentile : float
        Percentile below which a time step is considered dry (default 20).

    Returns
    -------
    DroughtStats
        Summary statistics of drought events.
    """
    threshold = np.percentile(precip.dropna().values, threshold_percentile)
    below = (precip < threshold).values
    runs = _run_lengths(below)

    if not runs:
        return DroughtStats(
            n_events=0, mean_duration=0.0, max_duration=0, mean_severity=0.0, total_deficit=0.0
        )

    durations = [r[1] for r in runs]
    severities: list[float] = []
    for start, length in runs:
        deficit = float(np.sum(threshold - precip.iloc[start : start + length].values))
        severities.append(deficit)

    return DroughtStats(
        n_events=len(runs),
        mean_duration=float(np.mean(durations)),
        max_duration=int(np.max(durations)),
        mean_severity=float(np.mean(severities)),
        total_deficit=float(np.sum(severities)),
    )


def wet_spell_analysis(
    precip: pd.Series,
    threshold_mm: float = 1.0,
) -> WetSpellStats:
    """Analyse wet-spell frequency and duration.

    A wet spell is a consecutive period where daily precipitation
    exceeds *threshold_mm*.

    Parameters
    ----------
    precip : pd.Series
        Daily precipitation time series (mm).
    threshold_mm : float
        Minimum precipitation to count as a wet day (default 1.0 mm).

    Returns
    -------
    WetSpellStats
        Summary statistics of wet spells.
    """
    wet = (precip >= threshold_mm).values
    runs = _run_lengths(wet)

    if not runs:
        return WetSpellStats(n_spells=0, mean_duration=0.0, max_duration=0, mean_intensity=0.0)

    durations = [r[1] for r in runs]
    intensities: list[float] = []
    for start, length in runs:
        intensities.append(float(precip.iloc[start : start + length].mean()))

    return WetSpellStats(
        n_spells=len(runs),
        mean_duration=float(np.mean(durations)),
        max_duration=int(np.max(durations)),
        mean_intensity=float(np.mean(intensities)),
    )


def scenario_comparison(
    scenarios: dict[str, pd.Series],
    baseline: pd.Series,
    metric: str = "mean",
) -> pd.DataFrame:
    """Compare multiple SSP scenarios against a baseline.

    Parameters
    ----------
    scenarios : dict[str, pd.Series]
        Mapping of scenario name to its time series.
    baseline : pd.Series
        Baseline (historical) time series.
    metric : str
        Comparison metric — ``"mean"``, ``"median"``, ``"std"``,
        ``"max"``, ``"min"``.

    Returns
    -------
    pd.DataFrame
        One row per scenario with columns: *scenario*, *baseline_value*,
        *scenario_value*, *absolute_change*, *percent_change*.

    Raises
    ------
    ValueError
        If *metric* is not supported.
    """
    allowed = {"mean", "median", "std", "max", "min"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {allowed}, got {metric!r}")

    _agg = getattr(pd.Series, metric)
    base_val = float(_agg(baseline))

    rows: list[dict[str, object]] = []
    for name, series in scenarios.items():
        scen_val = float(_agg(series))
        abs_change = scen_val - base_val
        pct_change = (abs_change / base_val * 100) if base_val != 0 else float("nan")
        rows.append(
            {
                "scenario": name,
                "baseline_value": base_val,
                "scenario_value": scen_val,
                "absolute_change": abs_change,
                "percent_change": pct_change,
            }
        )
    return pd.DataFrame(rows)
