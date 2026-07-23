"""
Change-point detection module.

Implements multiple algorithms for detecting structural changes
in time series data: PELT, CUSUM, binary segmentation, Pettitt's
test, and Rodionov regime shift detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    """A detected change point.

    Attributes:
        index: Position in the data array.
        timestamp: Corresponding timestamp if data has a DatetimeIndex.
        statistic: Test statistic at this point.
        p_value: Significance level (None if not applicable).
        mean_before: Mean of the segment before the change point.
        mean_after: Mean of the segment after the change point.
        variance_before: Variance of the segment before the change point.
        variance_after: Variance of the segment after the change point.
    """

    index: int
    timestamp: datetime | pd.Timestamp | None
    statistic: float
    p_value: float | None
    mean_before: float
    mean_after: float
    variance_before: float
    variance_after: float


@dataclass
class ChangePointResult:
    """Result of change-point detection.

    Attributes:
        changepoints: List of detected change points.
        n_changepoints: Number of detected change points.
        method: Name of the detection algorithm used.
        penalty: Penalty value used (None if not applicable).
        segments: List of segment dictionaries with keys start, end, mean, variance.
    """

    changepoints: list[ChangePoint] = field(default_factory=list)
    n_changepoints: int = 0
    method: str = ""
    penalty: float | None = None
    segments: list[dict] = field(default_factory=list)


def _to_array(data: np.ndarray | pd.Series) -> tuple[np.ndarray, pd.DatetimeIndex | None]:
    """Convert input data to a numpy array, extracting DatetimeIndex if present.

    Parameters:
        data: Input time series as numpy array or pandas Series.

    Returns:
        Tuple of (1D float array, DatetimeIndex or None).
    """
    timestamps: pd.DatetimeIndex | None = None
    if isinstance(data, pd.Series):
        if isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index
        arr = data.to_numpy(dtype=float, na_value=np.nan)
    else:
        arr = np.asarray(data, dtype=float)
    arr = arr.ravel()
    return arr, timestamps


def _get_timestamp(timestamps: pd.DatetimeIndex | None, index: int) -> datetime | pd.Timestamp | None:
    """Retrieve timestamp for a given index, or None.

    Parameters:
        timestamps: DatetimeIndex extracted from the input Series.
        index: Position in the array.

    Returns:
        Timestamp at position *index*, or None.
    """
    if timestamps is not None and 0 <= index < len(timestamps):
        return timestamps[index]
    return None


def _compute_segments(arr: np.ndarray, cp_indices: list[int], timestamps: pd.DatetimeIndex | None) -> list[dict]:
    """Compute segment statistics between consecutive change points.

    Parameters:
        arr: 1D data array.
        cp_indices: Sorted list of change-point indices.
        timestamps: Optional DatetimeIndex for the data.

    Returns:
        List of dicts with keys: start, end, mean, variance.
    """
    boundaries = [0] + sorted(cp_indices) + [len(arr)]
    segments: list[dict] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        segment_data = arr[s:e]
        segments.append({
            "start": s,
            "end": e,
            "mean": float(np.mean(segment_data)),
            "variance": float(np.var(segment_data, ddof=1)) if len(segment_data) > 1 else 0.0,
        })
    return segments


def _build_changepoints(
    arr: np.ndarray,
    cp_indices: list[int],
    timestamps: pd.DatetimeIndex | None,
    statistics: list[float] | None = None,
    p_values: list[float | None] | None = None,
) -> list[ChangePoint]:
    """Build ChangePoint objects for each detected index.

    Parameters:
        arr: 1D data array.
        cp_indices: Sorted change-point indices.
        timestamps: Optional DatetimeIndex.
        statistics: Test statistic per change point.
        p_values: P-value per change point.

    Returns:
        List of ChangePoint instances.
    """
    cps: list[ChangePoint] = []
    for k, idx in enumerate(sorted(cp_indices)):
        before = arr[:idx] if idx > 0 else arr[:1]
        after = arr[idx:] if idx < len(arr) else arr[-1:]
        stat = statistics[k] if statistics is not None else 0.0
        pval = p_values[k] if p_values is not None else None
        cps.append(ChangePoint(
            index=idx,
            timestamp=_get_timestamp(timestamps, idx),
            statistic=float(stat),
            p_value=pval,
            mean_before=float(np.mean(before)),
            mean_after=float(np.mean(after)),
            variance_before=float(np.var(before, ddof=1)) if len(before) > 1 else 0.0,
            variance_after=float(np.var(after, ddof=1)) if len(after) > 1 else 0.0,
        ))
    return cps


# ---------------------------------------------------------------------------
# Cost functions for PELT
# ---------------------------------------------------------------------------

def _cost_normal(data: np.ndarray) -> float:
    """Negative log-likelihood cost under a normal model (mean + variance change).

    Parameters:
        data: Segment of observations.

    Returns:
        Cost value.
    """
    n = len(data)
    if n <= 1:
        return 0.0
    var = np.var(data, ddof=0)
    if var <= 0:
        var = 1e-12
    return n * np.log(var)


def _cost_mean(data: np.ndarray) -> float:
    """Sum-of-squares cost (only detect mean shifts, assume constant variance).

    Parameters:
        data: Segment of observations.

    Returns:
        Cost value.
    """
    n = len(data)
    if n <= 1:
        return 0.0
    return float(np.sum((data - np.mean(data)) ** 2))


_COST_FUNCTIONS = {
    "normal": _cost_normal,
    "mean": _cost_mean,
}


def pelt(
    data: np.ndarray | pd.Series,
    penalty: float | None = None,
    min_segment_length: int = 10,
    cost: str = "normal",
) -> ChangePointResult:
    """Pruned Exact Linear Time (PELT) change-point detection.

    PELT algorithm:
      1. For each time *t*, compute optimal segmentation of ``data[0:t]``.
      2. ``F(t) = min over s (F(s) + C(data[s+1:t]) + penalty)``
         where *C* is the cost function and *penalty* is per-changepoint.
      3. Pruning: discard *s* values that can never be optimal.

    Cost functions:
      - ``"normal"``: negative log-likelihood assuming normal distribution
        (mean + variance change).
      - ``"mean"``: only detect mean shifts (assume constant variance).

    If *penalty* is ``None``, use modified BIC: ``penalty = 2 * log(n)``.

    Parameters:
        data: 1D time series.
        penalty: Penalty per changepoint (higher → fewer changepoints).
            Default: ``2 * log(n)``.
        min_segment_length: Minimum observations between changepoints.
        cost: Cost function type (``"normal"`` or ``"mean"``).

    Returns:
        ChangePointResult with detected changepoints and segment statistics.

    Raises:
        ValueError: If *cost* is not a recognised cost function name.
    """
    arr, timestamps = _to_array(data)
    n = len(arr)

    if cost not in _COST_FUNCTIONS:
        raise ValueError(f"Unknown cost function '{cost}'. Choose from {list(_COST_FUNCTIONS)}")
    cost_fn = _COST_FUNCTIONS[cost]

    if penalty is None:
        penalty = 2.0 * np.log(n) if n > 1 else 1.0

    logger.debug("PELT: n=%d, penalty=%.4f, min_segment=%d, cost=%s", n, penalty, min_segment_length, cost)

    # F[t] = optimal cost for data[0:t]
    f = np.full(n + 1, np.inf)
    f[0] = -penalty  # so that F(0) + C(data[0:t]) + penalty = C(data[0:t])
    cp_map: dict[int, int] = {}  # last changepoint for each end position
    candidates: list[int] = [0]

    for t in range(min_segment_length, n + 1):
        best_cost = np.inf
        best_s = 0
        admissible: list[int] = []

        for s in candidates:
            seg_len = t - s
            if seg_len < min_segment_length:
                admissible.append(s)
                continue

            segment = arr[s:t]
            c = cost_fn(segment)
            total = f[s] + c + penalty

            if total < best_cost:
                best_cost = total
                best_s = s

            # Pruning criterion: keep s if f[s] + C(data[s:t]) <= F(t)
            if f[s] + c <= best_cost:
                admissible.append(s)

        f[t] = best_cost
        cp_map[t] = best_s
        # Add current time as candidate and keep admissible set
        admissible.append(t)
        candidates = admissible

    # Backtrack to recover changepoints
    cp_indices: list[int] = []
    pos = n
    while pos > 0:
        s = cp_map.get(pos, 0)
        if s > 0:
            cp_indices.append(s)
        pos = s

    cp_indices = sorted(cp_indices)
    logger.info("PELT detected %d changepoint(s)", len(cp_indices))

    segments = _compute_segments(arr, cp_indices, timestamps)
    changepoints = _build_changepoints(arr, cp_indices, timestamps)

    return ChangePointResult(
        changepoints=changepoints,
        n_changepoints=len(changepoints),
        method="pelt",
        penalty=penalty,
        segments=segments,
    )


def cusum(
    data: np.ndarray | pd.Series,
    threshold: float | None = None,
    drift: float = 0.0,
) -> ChangePointResult:
    """CUSUM (Cumulative Sum) change-point detection.

    Computes ``S_t = max(0, S_{t-1} + (x_t - target - drift))`` for upward
    shifts and ``T_t = max(0, T_{t-1} + (target - drift - x_t))`` for downward
    shifts.  A changepoint is flagged when either cumulative sum exceeds
    *threshold*.

    If *threshold* is ``None``, use ``4 * std(data)`` as default.
    Target is ``mean(data)``.

    Parameters:
        data: 1D time series.
        threshold: Detection threshold. Default: ``4 * std(data)``.
        drift: Allowable drift (slack) before accumulating evidence.

    Returns:
        ChangePointResult with detected changepoints and segment statistics.
    """
    arr, timestamps = _to_array(data)
    n = len(arr)
    target = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 1.0

    if threshold is None:
        threshold = 4.0 * std

    logger.debug("CUSUM: n=%d, target=%.4f, threshold=%.4f, drift=%.4f", n, target, threshold, drift)

    s_pos = 0.0
    s_neg = 0.0
    cp_indices: list[int] = []
    stat_values: list[float] = []
    detected: set[int] = set()

    for t in range(n):
        s_pos = max(0.0, s_pos + (arr[t] - target - drift))
        s_neg = max(0.0, s_neg + (target - drift - arr[t]))

        if s_pos > threshold and t not in detected:
            cp_indices.append(t)
            stat_values.append(float(s_pos))
            detected.add(t)
            s_pos = 0.0
        elif s_neg > threshold and t not in detected:
            cp_indices.append(t)
            stat_values.append(float(s_neg))
            detected.add(t)
            s_neg = 0.0

    cp_indices = sorted(cp_indices)
    logger.info("CUSUM detected %d changepoint(s)", len(cp_indices))

    segments = _compute_segments(arr, cp_indices, timestamps)
    changepoints = _build_changepoints(arr, cp_indices, timestamps, statistics=stat_values)

    return ChangePointResult(
        changepoints=changepoints,
        n_changepoints=len(changepoints),
        method="cusum",
        penalty=None,
        segments=segments,
    )


def mann_whitney_test(data: np.ndarray, split_point: int) -> tuple[float, float]:
    """Mann-Whitney U test for difference between two segments.

    Tests whether ``data[:split_point]`` and ``data[split_point:]`` come from
    the same distribution.

    Parameters:
        data: 1D array of observations.
        split_point: Index at which to split the array.

    Returns:
        Tuple of ``(U_statistic, p_value)``.

    Raises:
        ValueError: If *split_point* yields an empty segment.
    """
    if split_point <= 0 or split_point >= len(data):
        raise ValueError(f"split_point must be in (0, {len(data)}), got {split_point}")

    left = data[:split_point]
    right = data[split_point:]
    result = stats.mannwhitneyu(left, right, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def binary_segmentation(
    data: np.ndarray | pd.Series,
    max_changepoints: int = 5,
    min_segment_length: int = 10,
    significance: float = 0.05,
) -> ChangePointResult:
    """Binary segmentation for change-point detection.

    Recursive approach:
      1. Test entire series for one changepoint (find point maximising test
         statistic).
      2. If significant (Mann-Whitney U test ``p-value < significance``),
         split and recurse.
      3. Stop when *max_changepoints* reached or no more significant splits.

    Parameters:
        data: 1D time series.
        max_changepoints: Maximum number of changepoints to detect.
        min_segment_length: Minimum segment length to consider a split.
        significance: P-value threshold for the Mann-Whitney U test.

    Returns:
        ChangePointResult with detected changepoints and segment statistics.
    """
    arr, timestamps = _to_array(data)
    n = len(arr)

    logger.debug(
        "Binary segmentation: n=%d, max_cp=%d, min_seg=%d, sig=%.4f",
        n, max_changepoints, min_segment_length, significance,
    )

    cp_indices: list[int] = []
    stat_values: list[float] = []
    p_values: list[float | None] = []

    def _find_best_split(segment: np.ndarray, offset: int) -> tuple[int, float, float] | None:
        """Find the split maximising the Mann-Whitney U statistic.

        Parameters:
            segment: Data segment to search.
            offset: Global offset of this segment.

        Returns:
            Tuple of (global_index, U_statistic, p_value) or None.
        """
        seg_n = len(segment)
        if seg_n < 2 * min_segment_length:
            return None

        best_stat = -1.0
        best_idx = -1
        best_pval = 1.0

        for i in range(min_segment_length, seg_n - min_segment_length + 1):
            u_stat, p_val = mann_whitney_test(segment, i)
            if u_stat > best_stat:
                best_stat = u_stat
                best_idx = i
                best_pval = p_val

        if best_idx < 0 or best_pval >= significance:
            return None

        return offset + best_idx, best_stat, best_pval

    def _recurse(start: int, end: int) -> None:
        if len(cp_indices) >= max_changepoints:
            return
        segment = arr[start:end]
        result = _find_best_split(segment, start)
        if result is None:
            return

        idx, stat, pval = result
        cp_indices.append(idx)
        stat_values.append(stat)
        p_values.append(pval)

        if len(cp_indices) >= max_changepoints:
            return

        _recurse(start, idx)
        _recurse(idx, end)

    _recurse(0, n)

    # Sort by index
    order = sorted(range(len(cp_indices)), key=lambda k: cp_indices[k])
    cp_indices = [cp_indices[i] for i in order]
    stat_values = [stat_values[i] for i in order]
    p_values = [p_values[i] for i in order]

    logger.info("Binary segmentation detected %d changepoint(s)", len(cp_indices))

    segments = _compute_segments(arr, cp_indices, timestamps)
    changepoints = _build_changepoints(arr, cp_indices, timestamps, statistics=stat_values, p_values=p_values)

    return ChangePointResult(
        changepoints=changepoints,
        n_changepoints=len(changepoints),
        method="binary_segmentation",
        penalty=None,
        segments=segments,
    )


def pettitt_test(data: np.ndarray | pd.Series) -> ChangePoint | None:
    """Pettitt's test for a single change point.

    Non-parametric test based on the Mann-Whitney two-sample statistic.
    ``U_t = sum_{i=1}^{t} sum_{j=t+1}^{n} sign(x_i - x_j)``

    The changepoint is at ``t* = argmax(|U_t|)``.

    P-value approximation: ``p ≈ 2 * exp(-6 * K_n^2 / (n^3 + n^2))``
    where ``K_n = max(|U_t|)``.

    Parameters:
        data: 1D time series.

    Returns:
        ChangePoint if significant (``p < 0.05``), else ``None``.
    """
    arr, timestamps = _to_array(data)
    n = len(arr)

    if n < 4:
        logger.warning("Pettitt test requires at least 4 observations, got %d", n)
        return None

    # Compute U_t efficiently using ranks
    u_values = np.zeros(n)
    for t in range(1, n):
        count = 0.0
        for i in range(t):
            for j in range(t, n):
                count += np.sign(arr[i] - arr[j])
        u_values[t] = count

    abs_u = np.abs(u_values)
    t_star = int(np.argmax(abs_u))
    k_n = abs_u[t_star]

    # P-value approximation
    denom = n ** 3 + n ** 2
    p_value = 2.0 * np.exp(-6.0 * k_n ** 2 / denom) if denom > 0 else 1.0
    p_value = min(p_value, 1.0)

    logger.debug("Pettitt test: t*=%d, K_n=%.4f, p=%.6f", t_star, k_n, p_value)

    if p_value >= 0.05:
        logger.info("Pettitt test: no significant changepoint (p=%.4f)", p_value)
        return None

    before = arr[:t_star]
    after = arr[t_star:]
    cp = ChangePoint(
        index=t_star,
        timestamp=_get_timestamp(timestamps, t_star),
        statistic=float(k_n),
        p_value=float(p_value),
        mean_before=float(np.mean(before)) if len(before) > 0 else 0.0,
        mean_after=float(np.mean(after)) if len(after) > 0 else 0.0,
        variance_before=float(np.var(before, ddof=1)) if len(before) > 1 else 0.0,
        variance_after=float(np.var(after, ddof=1)) if len(after) > 1 else 0.0,
    )
    logger.info("Pettitt test: changepoint at index %d (p=%.4f)", t_star, p_value)
    return cp


def regime_shift_detector(
    data: np.ndarray | pd.Series,
    window_size: int = 30,
    threshold: float = 2.0,
) -> list[ChangePoint]:
    """Rodionov regime shift detection.

    Algorithm:
      1. Compute mean and std of window before potential shift.
      2. If new observation differs from mean by > ``threshold * std``,
         mark as potential shift.
      3. Confirm if next *window_size* observations maintain the new regime.

    Good for climate / ecological regime shifts in noisy data.

    Parameters:
        data: 1D time series.
        window_size: Number of observations in each regime window.
        threshold: Number of standard deviations to trigger a potential shift.

    Returns:
        List of ChangePoint instances for confirmed regime shifts.
    """
    arr, timestamps = _to_array(data)
    n = len(arr)
    changepoints: list[ChangePoint] = []

    if n < 2 * window_size:
        logger.warning("Not enough data for regime shift detection (n=%d, window=%d)", n, window_size)
        return changepoints

    logger.debug("Regime shift detector: n=%d, window=%d, threshold=%.2f", n, window_size, threshold)

    i = window_size
    while i < n - window_size:
        pre_window = arr[max(0, i - window_size):i]
        pre_mean = float(np.mean(pre_window))
        pre_std = float(np.std(pre_window, ddof=1))

        if pre_std < 1e-12:
            i += 1
            continue

        if abs(arr[i] - pre_mean) > threshold * pre_std:
            # Potential shift — check if the next window confirms it
            post_window = arr[i:min(i + window_size, n)]
            post_mean = float(np.mean(post_window))

            if abs(post_mean - pre_mean) > threshold * pre_std:
                stat = abs(post_mean - pre_mean) / pre_std
                cp = ChangePoint(
                    index=i,
                    timestamp=_get_timestamp(timestamps, i),
                    statistic=float(stat),
                    p_value=None,
                    mean_before=pre_mean,
                    mean_after=post_mean,
                    variance_before=float(np.var(pre_window, ddof=1)),
                    variance_after=float(np.var(post_window, ddof=1)),
                )
                changepoints.append(cp)
                logger.debug("Regime shift at index %d (stat=%.4f)", i, stat)
                i += window_size  # skip ahead past the confirmed regime
                continue

        i += 1

    logger.info("Regime shift detector found %d shift(s)", len(changepoints))
    return changepoints


def plot_changepoints(
    data: np.ndarray | pd.Series,
    result: ChangePointResult,
    title: str = "Change-Point Detection",
    save_path: str | None = None,
) -> None:
    """Plot data with detected changepoints marked as vertical lines and segment means.

    Uses matplotlib with lazy import.

    Parameters:
        data: Original 1D time series.
        result: ChangePointResult from any detection method.
        title: Plot title.
        save_path: If provided, save the figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt

    arr, timestamps = _to_array(data)
    x = timestamps if timestamps is not None else np.arange(len(arr))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, arr, color="steelblue", alpha=0.7, label="Data")

    # Draw segment means
    for seg in result.segments:
        seg_x = x[seg["start"]:seg["end"]]
        ax.hlines(seg["mean"], seg_x[0], seg_x[-1], colors="orange", linewidths=2, label="Segment mean")

    # Mark changepoints
    for cp in result.changepoints:
        ax.axvline(x[cp.index], color="red", linestyle="--", alpha=0.8, label="Changepoint")

    # Deduplicate legend labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    ax.set_title(title)
    ax.set_xlabel("Time" if timestamps is not None else "Index")
    ax.set_ylabel("Value")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)
