"""Flow duration curves and low-flow statistics.

Implements the standard hydrological tools every water engineer expects:

- **Flow Duration Curve (FDC)** — probability-of-exceedance vs discharge.
- **Percentile extraction** — Q5, Q10, Q50, Q75, Q90, Q95, Q99.
- **Low-flow statistics** — nQm (the minimum *n*-day average flow that
  occurs once every *m* years), e.g. 7Q10, 30Q5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FDCResult:
    """Result of a flow duration curve analysis.

    Attributes
    ----------
    exceedance:
        Exceedance probability array (0–100 %).
    discharge:
        Sorted discharge values (descending).
    percentiles:
        Mapping of exceedance % → discharge value (e.g. ``{95: 1.23}``).
    """

    exceedance: np.ndarray
    discharge: np.ndarray
    percentiles: dict[float, float] = field(default_factory=dict)


def flow_duration_curve(
    discharge: pd.Series,
    *,
    percentiles: list[float] | None = None,
) -> FDCResult:
    """Compute a flow duration curve.

    Parameters
    ----------
    discharge:
        Time-series of discharge values (any DatetimeIndex).
    percentiles:
        Exceedance percentiles to extract.  Defaults to
        ``[5, 10, 25, 50, 75, 90, 95, 99]``.

    Returns
    -------
    A :class:`FDCResult` containing sorted discharges and extracted
    percentile values.
    """
    if percentiles is None:
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]

    clean = discharge.dropna().values
    sorted_q = np.sort(clean)[::-1]
    n = len(sorted_q)
    exceedance = np.arange(1, n + 1) / n * 100

    pct_map: dict[float, float] = {}
    for p in percentiles:
        idx = min(int(p / 100 * n), n - 1)
        pct_map[p] = float(sorted_q[idx])

    logger.info("FDC computed: %d values, Q50=%.3f, Q95=%.3f", n, pct_map.get(50, 0), pct_map.get(95, 0))
    return FDCResult(exceedance=exceedance, discharge=sorted_q, percentiles=pct_map)


def low_flow_stat(
    discharge: pd.Series,
    *,
    n_day: int = 7,
    return_period: int = 10,
) -> float:
    """Compute nQm low-flow statistic (e.g. 7Q10).

    The nQm is the minimum *n*-day rolling average that occurs with
    a return period of *m* years, estimated using the Weibull plotting
    position.

    Parameters
    ----------
    discharge:
        Daily discharge series with a DatetimeIndex.
    n_day:
        Rolling window size in days.
    return_period:
        Return period in years.

    Returns
    -------
    The estimated nQm value in the same units as the input discharge.

    Raises
    ------
    ValueError
        If there are fewer than 3 complete water years.
    """
    rolling_min = discharge.rolling(window=n_day, min_periods=n_day).mean()

    annual_min = rolling_min.resample("YS").min().dropna()
    if len(annual_min) < 3:
        msg = f"Need ≥3 complete years; got {len(annual_min)}"
        raise ValueError(msg)

    sorted_vals = np.sort(annual_min.values)
    n = len(sorted_vals)
    # Weibull plotting position
    prob = np.arange(1, n + 1) / (n + 1)
    target_prob = 1.0 / return_period

    result = float(np.interp(target_prob, prob, sorted_vals))
    logger.info("%dQ%d = %.3f (from %d years)", n_day, return_period, result, n)
    return result
