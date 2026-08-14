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

    The default ``"original"`` method uses pure NumPy/SciPy without external
    dependencies. Autocorrelation-aware modified variants delegate to
    :mod:`pymannkendall`.

    Parameters
    ----------
    series:
        Input data series (e.g. annual runoff, daily streamflow, or temperature).
    alpha:
        Significance level for the hypothesis test (default: ``0.05``).
    method:
        Mann-Kendall test variant. Supported values:
        - ``"original"``: Standard Mann-Kendall test (default, pure SciPy).
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
        If a modified variant is requested but ``pymannkendall`` is not installed.
    ValueError
        If *method* is invalid or if fewer than 3 non-NaN values exist.
    """
    if method not in _MK_METHODS:
        msg = f"Unknown Mann-Kendall method {method!r}. Choose from {sorted(_MK_METHODS)}."
        raise ValueError(msg)

    import numpy as np
    import pandas as pd
    from scipy import stats

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
        n = len(clean)
        s = 0
        slopes_list: list[np.ndarray] = []
        for i in range(n - 1):
            d = clean[i + 1 :] - clean[i]
            s += int(np.sign(d).sum())
            slopes_list.append(d / np.arange(1, n - i))

        unique, counts = np.unique(clean, return_counts=True)
        tie_sum = sum(t * (t - 1) * (2 * t + 5) for t in counts if t > 1)
        var_s = (n * (n - 1) * (2 * n + 5) - tie_sum) / 18.0

        if var_s == 0:
            z = 0.0
        elif s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0

        p_val = float(2.0 * stats.norm.sf(abs(z)))

        if p_val < alpha:
            if z > 0:
                trend_str = "increasing"
            elif z < 0:
                trend_str = "decreasing"
            else:
                trend_str = "no trend"
        else:
            trend_str = "no trend"

        denom = 0.5 * n * (n - 1)
        tau = float(s / denom) if denom > 0 else 0.0

        if slopes_list:
            all_slopes = np.concatenate(slopes_list)
            slope_val = float(np.median(all_slopes))
        else:
            slope_val = 0.0

        intercept_val = float(np.median(clean) - np.median(np.arange(n)) * slope_val)
    else:
        mk = require("pymannkendall", feature="Mann-Kendall trend analysis", group="ml")
        if method == "hamed_rao":
            res = mk.hamed_rao_modification_test(clean, alpha=alpha)
        elif method == "yue_wang":
            res = mk.yue_wang_modification_test(clean, alpha=alpha)
        elif method == "pre_whitening":
            res = mk.pre_whitening_modification_test(clean, alpha=alpha)
        else:  # tfpw
            res = mk.trend_free_pre_whitening_modification_test(clean, alpha=alpha)

        trend_str = str(res.trend)
        p_val = float(res.p)
        z = float(res.z)
        tau = float(res.Tau)
        s = float(res.s)
        var_s = float(res.var_s)
        slope_val = float(res.slope)
        intercept_val = float(res.intercept)

    return MannKendallResult(
        trend=trend_str,  # type: ignore[arg-type]
        h=bool(p_val < alpha),
        p_value=p_val,
        z_stat=z,
        tau=tau,
        s_stat=float(s),
        var_s=float(var_s),
        slope=slope_val,
        intercept=intercept_val,
        alpha=float(alpha),
        method=str(method),
        n_samples=len(clean),
        n_dropped_nans=n_dropped_nans,
    )


def sens_slope(
    series: pd.Series | np.ndarray | list[float],
) -> SensSlopeResult:
    """Estimate Sen's slope (median of pairwise slopes) for a time series.

    Uses pure NumPy/SciPy without external dependencies.

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
    ValueError
        If fewer than 2 non-NaN values exist.
    """
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

    n = len(clean)
    slopes_list: list[np.ndarray] = []
    for i in range(n - 1):
        d = clean[i + 1 :] - clean[i]
        slopes_list.append(d / np.arange(1, n - i))

    if slopes_list:
        all_slopes = np.concatenate(slopes_list)
        slope_val = float(np.median(all_slopes))
    else:
        slope_val = 0.0

    intercept_val = float(np.median(clean) - np.median(np.arange(n)) * slope_val)

    return SensSlopeResult(
        slope=slope_val,
        intercept=intercept_val,
        n_samples=len(clean),
        n_dropped_nans=n_dropped_nans,
    )
