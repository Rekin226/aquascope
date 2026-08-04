"""Mann-Kendall trend detection and Sen's slope estimation.

Provides statistical trend analysis for hydrological and climate time-series,
wrapping `pymannkendall` with NaN handling, validation, and standardized
dataclass outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from aquascope.utils.imports import require

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MannKendallResult:
    """Result of a Mann-Kendall trend test.

    Attributes
    ----------
    trend:
        Trend direction (``"increasing"``, ``"decreasing"``, or ``"no trend"``).
    h:
        ``True`` if a statistically significant trend exists at level *alpha*.
    p_value:
        Two-sided p-value of the test.
    z_stat:
        Normalized Z test statistic.
    tau:
        Kendall's tau correlation coefficient.
    s_stat:
        Mann-Kendall score S statistic.
    var_s:
        Variance of the S statistic.
    slope:
        Sen's slope estimate (magnitude of change per time step).
    intercept:
        Intercept for Sen's linear slope estimate.
    alpha:
        Significance level threshold used.
    method:
        Test variant used (e.g. ``"original"``, ``"hamed_rao"``).
    n_samples:
        Number of valid data points evaluated.
    n_dropped_nans:
        Number of NaN / null values dropped prior to testing.
    """

    trend: Literal["increasing", "decreasing", "no trend"]
    h: bool
    p_value: float
    z_stat: float
    tau: float
    s_stat: float
    var_s: float
    slope: float
    intercept: float
    alpha: float
    method: str
    n_samples: int
    n_dropped_nans: int


@dataclass
class SensSlopeResult:
    """Result of Sen's slope estimation.

    Attributes
    ----------
    slope:
        Median pairwise slope estimate (change per unit time).
    intercept:
        Intercept estimate corresponding to the median slope.
    n_samples:
        Number of valid data points evaluated.
    n_dropped_nans:
        Number of NaN / null values dropped prior to estimation.
    """

    slope: float
    intercept: float
    n_samples: int
    n_dropped_nans: int


_MK_METHODS = {
    "original",
    "hamed_rao",
    "yue_wang",
    "pre_whitening",
    "tfpw",
}


def mann_kendall(
    series: pd.Series | np.ndarray | list[float],
    alpha: float = 0.05,
    method: str = "original",
) -> MannKendallResult:
    """Perform a Mann-Kendall trend test on a time series.

    Delegates to :mod:`pymannkendall` with automatic boundary handling for
    missing values (NaNs), input validation, and standardized return structure.

    Parameters
    ----------
    series:
        Input data series (e.g. annual runoff, daily streamflow, or temperature).
    alpha:
        Significance level for the hypothesis test (default: ``0.05``).
    method:
        Mann-Kendall test variant. Supported values:
        - ``"original"``: Standard Mann-Kendall test (default).
        - ``"hamed_rao"``: Modified MK correcting for autocorrelation (Hamed & Rao 1998).
        - ``"yue_wang"``: Modified MK for autocorrelation (Yue & Wang 2004).
        - ``"pre_whitening"``: Pre-whitening modified MK (von Storch 1995).
        - ``"tfpw"``: Trend-free pre-whitening modified MK (Yue et al. 2002).

    Returns
    -------
    MannKendallResult
        Test statistics, p-value, trend classification, Sen's slope, and metadata.

    Raises
    ------
    ImportError
        If ``pymannkendall`` is not installed.
    ValueError
        If *method* is invalid or if fewer than 3 non-NaN values exist.
    """
    mk = require("pymannkendall", feature="Mann-Kendall trend analysis", group="ml")

    if method not in _MK_METHODS:
        msg = f"Unknown Mann-Kendall method {method!r}. Choose from {sorted(_MK_METHODS)}."
        raise ValueError(msg)

    import numpy as np
    import pandas as pd

    if isinstance(series, pd.Series):
        arr = series.to_numpy(dtype=float)
    elif isinstance(series, np.ndarray):
        arr = series.ravel().astype(float)
    else:
        arr = np.asarray(series, dtype=float)

    n_total = len(arr)
    valid_mask = ~np.isnan(arr)
    n_dropped_nans = int(n_total - np.count_nonzero(valid_mask))
    clean = arr[valid_mask]

    if n_dropped_nans > 0:
        logger.warning("Dropped %d NaN value(s) before performing Mann-Kendall test.", n_dropped_nans)

    if len(clean) < 3:
        msg = f"Insufficient valid data points for Mann-Kendall test (got {len(clean)}, need >= 3)."
        raise ValueError(msg)

    if method == "original":
        res = mk.original_test(clean, alpha=alpha)
    elif method == "hamed_rao":
        res = mk.hamed_rao_modification_test(clean, alpha=alpha)
    elif method == "yue_wang":
        res = mk.yue_wang_modification_test(clean, alpha=alpha)
    elif method == "pre_whitening":
        res = mk.pre_whitening_modification_test(clean, alpha=alpha)
    else:  # tfpw
        res = mk.trend_free_pre_whitening_modification_test(clean, alpha=alpha)

    return MannKendallResult(
        trend=str(res.trend),
        h=bool(res.h),
        p_value=float(res.p),
        z_stat=float(res.z),
        tau=float(res.Tau),
        s_stat=float(res.s),
        var_s=float(res.var_s),
        slope=float(res.slope),
        intercept=float(res.intercept),
        alpha=float(alpha),
        method=str(method),
        n_samples=len(clean),
        n_dropped_nans=n_dropped_nans,
    )


def sens_slope(
    series: pd.Series | np.ndarray | list[float],
) -> SensSlopeResult:
    """Estimate Sen's slope (median of pairwise slopes) for a time series.

    Parameters
    ----------
    series:
        Input data series.

    Returns
    -------
    SensSlopeResult
        Estimated slope, intercept, and sample counts.

    Raises
    ------
    ImportError
        If ``pymannkendall`` is not installed.
    ValueError
        If fewer than 2 non-NaN values exist.
    """
    mk = require("pymannkendall", feature="Sen's slope estimation", group="ml")

    import numpy as np
    import pandas as pd

    if isinstance(series, pd.Series):
        arr = series.to_numpy(dtype=float)
    elif isinstance(series, np.ndarray):
        arr = series.ravel().astype(float)
    else:
        arr = np.asarray(series, dtype=float)

    n_total = len(arr)
    valid_mask = ~np.isnan(arr)
    n_dropped_nans = int(n_total - np.count_nonzero(valid_mask))
    clean = arr[valid_mask]

    if n_dropped_nans > 0:
        logger.warning("Dropped %d NaN value(s) before computing Sen's slope.", n_dropped_nans)

    if len(clean) < 2:
        msg = f"Insufficient valid data points for Sen's slope (got {len(clean)}, need >= 2)."
        raise ValueError(msg)

    res = mk.sens_slope(clean)

    return SensSlopeResult(
        slope=float(res.slope),
        intercept=float(res.intercept),
        n_samples=len(clean),
        n_dropped_nans=n_dropped_nans,
    )
