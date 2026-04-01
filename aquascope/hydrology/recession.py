"""Recession analysis for streamflow data.

Provides tools for:

- **Recession segment identification** — extract falling-limb segments
  from the hydrograph.
- **Master recession curve (MRC) fitting** — fit an exponential decay
  model to the ensemble of recession segments.
- **Storage–discharge relationship** — estimate the recession constant
  and aquifer storage coefficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RecessionSegment:
    """A single recession segment.

    Attributes
    ----------
    start:
        Start timestamp.
    end:
        End timestamp.
    discharge:
        Discharge values during the recession.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    discharge: np.ndarray


@dataclass
class RecessionResult:
    """Result of recession analysis.

    Attributes
    ----------
    segments:
        List of identified recession segments.
    recession_constant:
        Fitted exponential decay constant *k* in Q(t) = Q₀·e^(−t/k).
    r_squared:
        R² of the master recession curve fit.
    half_life_days:
        Time in days for discharge to halve (k·ln(2)).
    """

    segments: list[RecessionSegment] = field(default_factory=list)
    recession_constant: float = 0.0
    r_squared: float = 0.0
    half_life_days: float = 0.0


def identify_recessions(
    discharge: pd.Series,
    *,
    min_length: int = 5,
    min_decline_pct: float = 0.05,
) -> list[RecessionSegment]:
    """Identify recession segments in a daily discharge series.

    A recession is a continuous period where each day's discharge is
    less than the previous day's.  Very short segments or those with
    negligible total decline are excluded.

    Parameters
    ----------
    discharge:
        Daily discharge series with a DatetimeIndex.
    min_length:
        Minimum segment length in days.
    min_decline_pct:
        Minimum total decline as a fraction of the starting value.

    Returns
    -------
    List of :class:`RecessionSegment` instances.
    """
    q = discharge.dropna()
    if len(q) < min_length:
        return []

    segments: list[RecessionSegment] = []
    values = q.values
    dates = q.index

    i = 0
    while i < len(values) - 1:
        # Find start of recession (value decreases)
        if values[i + 1] >= values[i]:
            i += 1
            continue

        start_idx = i
        j = i + 1
        while j < len(values) and values[j] <= values[j - 1]:
            j += 1

        length = j - start_idx
        if length >= min_length:
            segment_vals = values[start_idx:j]
            decline = (segment_vals[0] - segment_vals[-1]) / segment_vals[0] if segment_vals[0] > 0 else 0
            if decline >= min_decline_pct:
                segments.append(RecessionSegment(
                    start=dates[start_idx],
                    end=dates[j - 1],
                    discharge=segment_vals,
                ))

        i = j

    logger.info("Found %d recession segments (min_length=%d)", len(segments), min_length)
    return segments


def fit_master_recession(
    segments: list[RecessionSegment],
) -> RecessionResult:
    """Fit a master recession curve to the identified segments.

    Uses least-squares fitting of ln(Q/Q₀) vs time to estimate the
    recession constant *k* in the exponential model Q(t) = Q₀·e^(−t/k).

    Parameters
    ----------
    segments:
        Recession segments from :func:`identify_recessions`.

    Returns
    -------
    A :class:`RecessionResult` with the fitted recession constant and
    goodness-of-fit metrics.

    Raises
    ------
    ValueError
        If no segments are provided.
    """
    if not segments:
        msg = "No recession segments provided"
        raise ValueError(msg)

    # Normalise and stack all segments
    all_t: list[float] = []
    all_log_q: list[float] = []

    for seg in segments:
        q0 = seg.discharge[0]
        if q0 <= 0:
            continue
        for day, qval in enumerate(seg.discharge):
            if qval > 0:
                all_t.append(float(day))
                all_log_q.append(np.log(qval / q0))

    if len(all_t) < 3:
        return RecessionResult(segments=segments)

    t_arr = np.array(all_t)
    log_q_arr = np.array(all_log_q)

    # Least-squares fit: ln(Q/Q0) = -t/k  →  slope = -1/k
    coeffs = np.polyfit(t_arr, log_q_arr, 1)
    slope = coeffs[0]

    if slope >= 0:
        logger.warning("Positive slope in recession fit — data may not contain true recessions")
        return RecessionResult(segments=segments)

    k = -1.0 / slope
    half_life = k * np.log(2)

    # R²
    predicted = np.polyval(coeffs, t_arr)
    ss_res = np.sum((log_q_arr - predicted) ** 2)
    ss_tot = np.sum((log_q_arr - log_q_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    logger.info("MRC fit: k=%.2f days, half_life=%.2f days, R²=%.4f", k, half_life, r2)
    return RecessionResult(
        segments=segments,
        recession_constant=k,
        r_squared=r2,
        half_life_days=half_life,
    )


def recession_analysis(
    discharge: pd.Series,
    *,
    min_length: int = 5,
    min_decline_pct: float = 0.05,
) -> RecessionResult:
    """Run full recession analysis: identify segments + fit MRC.

    Convenience function combining :func:`identify_recessions` and
    :func:`fit_master_recession`.

    Parameters
    ----------
    discharge:
        Daily discharge series with a DatetimeIndex.
    min_length:
        Minimum recession segment length in days.
    min_decline_pct:
        Minimum total decline fraction.

    Returns
    -------
    A :class:`RecessionResult`.
    """
    segments = identify_recessions(discharge, min_length=min_length, min_decline_pct=min_decline_pct)
    if not segments:
        logger.warning("No recession segments found")
        return RecessionResult()
    return fit_master_recession(segments)
