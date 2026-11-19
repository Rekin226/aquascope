"""Groundwater drought indices.

Implements the Standardised Groundwater level Index (SGI) of Bloomfield &
Marchant (2013, HESS 17:4769): a per-calendar-month, non-parametric normal-
scores transform of a groundwater-level series, merged into a continuous index.
SGI < -1 marks groundwater drought; the index is directly comparable to the
Standardized Precipitation Index (SPI) for drought-propagation analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DroughtEvent:
    """A continuous run of an index below a drought threshold.

    Attributes
    ----------
    start, end:
        First and last timestamps of the event.
    duration:
        Number of time steps in the event.
    severity:
        Sum of the (negative) index over the event (more negative = worse).
    peak:
        Most negative index value reached.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    duration: int
    severity: float
    peak: float


def standardised_groundwater_index(
    levels: pd.Series, min_per_month: int = 5
) -> pd.Series:
    """Standardised Groundwater Index (SGI), Bloomfield & Marchant (2013).

    For each calendar month the groundwater levels (across all years) are
    transformed to standard-normal scores via a non-parametric normal-scores
    (inverse-normal of plotting positions) transform; the twelve monthly series
    are then merged into one continuous monthly index. The result is unitless,
    centred on zero, with SGI < -1 indicating groundwater drought.

    Parameters
    ----------
    levels:
        Groundwater-level series with a ``DatetimeIndex`` (one value per month
        is expected; multiple values in a month are kept and standardised
        within that calendar month).
    min_per_month:
        Minimum number of observations a calendar month must have to be
        standardised. Months with fewer are returned as ``NaN`` (the record is
        too short to estimate that month's distribution).

    Returns
    -------
    pd.Series
        The SGI, indexed like the (sorted, non-null) input, named ``"sgi"``.
    """
    if not isinstance(levels.index, pd.DatetimeIndex):
        raise ValueError("levels must have a DatetimeIndex.")
    s = levels.dropna().sort_index()
    if len(s) < min_per_month:
        raise ValueError(
            f"Need at least {min_per_month} observations; got {len(s)}."
        )

    sgi = pd.Series(np.nan, index=s.index, dtype=float, name="sgi")
    month = s.index.month
    for m in range(1, 13):
        idx = s.index[month == m]
        if len(idx) == 0:
            continue
        vals = s.loc[idx]
        if len(vals) < min_per_month:
            logger.debug("Month %d has %d obs (< %d); left NaN.", m, len(vals), min_per_month)
            continue
        # Average ranks -> plotting positions -> inverse standard normal.
        ranks = stats.rankdata(vals.values, method="average")
        sgi.loc[idx] = stats.norm.ppf((ranks - 0.5) / len(vals))
    return sgi


def drought_events(index: pd.Series, threshold: float = -1.0) -> list[DroughtEvent]:
    """Identify drought events as runs of *index* at or below *threshold*.

    Missing index values and omitted calendar months in a ``DatetimeIndex``
    break a run. Observations within the same month may belong to one run.
    Returns events in time order, with duration in
    observations, accumulated severity, and peak (most negative) value.
    """
    s = index.sort_index(kind="stable")
    below = s.le(threshold).fillna(False)
    starts = ~below.shift(fill_value=False)
    if isinstance(s.index, pd.DatetimeIndex):
        months = pd.Series(s.index.year * 12 + s.index.month, index=s.index)
        starts |= months.diff().gt(1)
    groups = starts.cumsum()
    return [
        DroughtEvent(segment.index[0], segment.index[-1], len(segment), float(segment.sum()), float(segment.min()))
        for _, segment in s[below].groupby(groups[below], sort=False)
    ]


@dataclass
class PropagationResult:
    """How a meteorological index leads a groundwater index.

    Attributes
    ----------
    lag_months:
        The lag (months) at which the driver correlates best with the response.
    correlation:
        Pearson correlation at that lag.
    n:
        Overlapping months at that lag.
    correlations:
        Correlation at every lag tried, ``{lag: r}``.
    """

    lag_months: int
    correlation: float
    n: int
    correlations: dict[int, float]


def propagation_lag(
    driver: pd.Series, response: pd.Series, *, max_lag: int = 24, min_overlap: int = 24
) -> PropagationResult:
    """The lag at which a driver index (SPI-n) best explains a response index (SGI).

    Bloomfield and Marchant (2013) characterise drought propagation by the
    cross-correlation between SPI at several accumulation periods and the SGI:
    the accumulation period and lag that maximise it say how long a rainfall
    deficit takes to reach the water table. Both series are read on a monthly
    calendar (several values in a month are averaged); the driver is shifted
    forward by ``0 .. max_lag`` months and the Pearson correlation over the
    overlapping months is taken at each lag. Raises ``ValueError`` when no lag
    leaves ``min_overlap`` overlapping months.
    """
    if not isinstance(driver.index, pd.DatetimeIndex) or not isinstance(response.index, pd.DatetimeIndex):
        raise ValueError("driver and response must have a DatetimeIndex.")
    d = driver.dropna().astype(float)
    r = response.dropna().astype(float)
    d = d.groupby(d.index.to_period("M")).mean()
    r = r.groupby(r.index.to_period("M")).mean()
    correlations: dict[int, float] = {}
    counts: dict[int, int] = {}
    for lag in range(0, int(max_lag) + 1):
        shifted = d.copy()
        shifted.index = shifted.index + lag
        pair = pd.concat([shifted.rename("d"), r.rename("r")], axis=1, join="inner").dropna()
        if len(pair) < min_overlap:
            continue
        c = float(pair["d"].corr(pair["r"]))
        if np.isfinite(c):
            correlations[lag] = c
            counts[lag] = int(len(pair))
    if not correlations:
        raise ValueError(f"fewer than {min_overlap} overlapping months at every lag up to {max_lag}")
    best = max(correlations, key=lambda k: correlations[k])
    return PropagationResult(best, correlations[best], counts[best], correlations)
