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

    Works on any standardised index (SGI or SPI). Returns events ordered in
    time, each with duration, accumulated severity, and peak (most negative)
    value.
    """
    s = index.dropna().sort_index()
    below = s <= threshold
    events: list[DroughtEvent] = []
    run_start: pd.Timestamp | None = None
    prev: pd.Timestamp | None = None
    for ts, flag in below.items():
        if flag and run_start is None:
            run_start = ts
        elif not flag and run_start is not None:
            seg = s.loc[run_start:prev]
            events.append(DroughtEvent(run_start, prev, len(seg),
                                       float(seg.sum()), float(seg.min())))
            run_start = None
        prev = ts
    if run_start is not None:
        seg = s.loc[run_start:prev]
        events.append(DroughtEvent(run_start, prev, len(seg),
                                   float(seg.sum()), float(seg.min())))
    return events
