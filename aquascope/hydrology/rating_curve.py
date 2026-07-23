"""Stage-discharge rating curve fitting and analysis.

Implements power-law rating curve fitting ``Q = a * (H - H₀)^b``,
multi-segment curves with automatic breakpoint detection, uncertainty
estimation, temporal shift detection, and HEC-RAS export.

Key functions:

- :func:`fit_rating_curve` — single power-law fit
- :func:`fit_segmented_rating_curve` — multi-segment fit
- :func:`predict_discharge` / :func:`predict_stage` — forward and inverse prediction
- :func:`rating_curve_uncertainty` — prediction intervals
- :func:`detect_rating_shift` — temporal shift detection
- :func:`cross_validate_rating` — k-fold cross-validation
- :func:`export_hec_ras` — HEC-RAS compatible export
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)


@dataclass
class RatingSegment:
    """A segment of a segmented rating curve.

    Attributes:
        stage_min: Lower bound of the segment stage range.
        stage_max: Upper bound of the segment stage range.
        a: Power-law coefficient.
        b: Power-law exponent.
        h0: Stage offset (datum correction).
        r_squared: Coefficient of determination for the segment.
    """

    stage_min: float
    stage_max: float
    a: float
    b: float
    h0: float
    r_squared: float


@dataclass
class RatingCurveResult:
    """Result of rating curve fitting.

    Attributes:
        a: Power-law coefficient.
        b: Power-law exponent.
        h0: Stage offset (datum correction).
        r_squared: Coefficient of determination.
        rmse: Root mean squared error.
        n_points: Number of stage-discharge pairs used.
        residuals: Array of residuals (observed - predicted).
        stage_range: Tuple of (min_stage, max_stage).
        segments: Segment parameters for segmented curves, or ``None``.
    """

    a: float
    b: float
    h0: float
    r_squared: float
    rmse: float
    n_points: int
    residuals: np.ndarray
    stage_range: tuple[float, float]
    segments: list[RatingSegment] | None = None


def _validate_inputs(stage: np.ndarray, discharge: np.ndarray, min_points: int = 5) -> None:
    """Validate stage-discharge input arrays.

    Parameters:
        stage: Stage measurements.
        discharge: Discharge measurements.
        min_points: Minimum number of observations required.

    Raises:
        ValueError: If inputs are invalid.
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)

    if len(stage) != len(discharge):
        raise ValueError(f"Stage ({len(stage)}) and discharge ({len(discharge)}) arrays must have equal length.")

    if len(stage) < min_points:
        raise ValueError(f"At least {min_points} stage-discharge pairs required, got {len(stage)}.")

    if np.any(discharge < 0):
        raise ValueError("Discharge values must be non-negative.")

    if np.any(np.isnan(stage)) or np.any(np.isnan(discharge)):
        raise ValueError("Stage and discharge arrays must not contain NaN values.")


def _power_law(h: np.ndarray, a: float, b: float, h0: float) -> np.ndarray:
    """Compute Q = a * (H - h0)^b, clipping to zero where H <= h0."""
    effective = np.maximum(h - h0, 1e-12)
    return a * np.power(effective, b)


def _compute_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the coefficient of determination (R²)."""
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _fit_single_segment(
    stage: np.ndarray, discharge: np.ndarray, h0: float | None = None
) -> tuple[float, float, float, float, float, np.ndarray]:
    """Fit a single power-law segment, returning (a, b, h0, r2, rmse, residuals)."""
    if h0 is not None:
        # Fixed h0 — log-transform and solve linear regression
        effective = stage - h0
        mask = effective > 0
        if np.sum(mask) < 3:
            raise ValueError("Not enough points with stage > h0 for fitting.")
        log_eff = np.log(effective[mask])
        log_q = np.log(discharge[mask])
        slope, intercept, _, _, _ = stats.linregress(log_eff, log_q)
        a = np.exp(intercept)
        b = slope
    else:
        # Estimate h0 along with a and b using nonlinear least squares
        h0_init = np.min(stage) - 0.1 * (np.max(stage) - np.min(stage)) - 0.01
        a_init = np.median(discharge) / max((np.median(stage) - h0_init) ** 1.5, 1e-6)
        b_init = 1.5

        try:
            popt, _ = optimize.curve_fit(
                _power_law,
                stage,
                discharge,
                p0=[a_init, b_init, h0_init],
                bounds=([1e-10, 0.1, -np.inf], [np.inf, 10.0, np.min(stage) - 1e-6]),
                maxfev=10000,
            )
            a, b, h0 = popt
        except RuntimeError:
            logger.warning("Nonlinear fit failed; falling back to log-transform with h0 = min(stage) - 0.01.")
            h0 = np.min(stage) - 0.01
            effective = stage - h0
            log_eff = np.log(effective)
            log_q = np.log(discharge)
            slope, intercept, _, _, _ = stats.linregress(log_eff, log_q)
            a = np.exp(intercept)
            b = slope

    predicted = _power_law(stage, a, b, h0)
    residuals = discharge - predicted
    r2 = _compute_r_squared(discharge, predicted)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return a, b, h0, r2, rmse, residuals


def fit_rating_curve(stage: np.ndarray, discharge: np.ndarray, h0: float | None = None) -> RatingCurveResult:
    """Fit a power-law rating curve ``Q = a * (H - H₀)^b``.

    If *h0* is ``None``, it is optimised together with *a* and *b* using
    :func:`scipy.optimize.curve_fit`.

    Parameters:
        stage: Water level (stage) measurements.
        discharge: Corresponding discharge measurements.
        h0: Optional fixed stage offset.  If ``None``, estimated from data.

    Returns:
        :class:`RatingCurveResult` with fitted parameters and diagnostics.

    Raises:
        ValueError: If fewer than 5 stage-discharge pairs are provided,
            discharge contains negative values, or arrays contain NaN.
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    _validate_inputs(stage, discharge)

    a, b, h0_fit, r2, rmse, residuals = _fit_single_segment(stage, discharge, h0)

    logger.info("Rating curve fit: a=%.4f, b=%.4f, h0=%.4f, R²=%.4f, RMSE=%.4f", a, b, h0_fit, r2, rmse)

    return RatingCurveResult(
        a=a,
        b=b,
        h0=h0_fit,
        r_squared=r2,
        rmse=rmse,
        n_points=len(stage),
        residuals=residuals,
        stage_range=(float(np.min(stage)), float(np.max(stage))),
    )


def fit_segmented_rating_curve(
    stage: np.ndarray,
    discharge: np.ndarray,
    n_segments: int = 2,
    breakpoints: list[float] | None = None,
) -> RatingCurveResult:
    """Fit a multi-segment rating curve with breakpoints.

    Each segment is fitted independently as a power-law.  If *breakpoints*
    are not provided, optimal breakpoints are found by minimising total RMSE
    over a grid of candidate values.

    Parameters:
        stage: Water level (stage) measurements.
        discharge: Corresponding discharge measurements.
        n_segments: Number of segments (default 2).
        breakpoints: Explicit breakpoint stage values.  Length must equal
            ``n_segments - 1``.  If ``None``, breakpoints are optimised.

    Returns:
        :class:`RatingCurveResult` with per-segment parameters in *segments*.

    Raises:
        ValueError: If inputs are invalid or segments cannot be fitted.
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    _validate_inputs(stage, discharge)

    if breakpoints is not None:
        if len(breakpoints) != n_segments - 1:
            raise ValueError(f"Expected {n_segments - 1} breakpoints, got {len(breakpoints)}.")
        bp = sorted(breakpoints)
    else:
        # Grid-search for optimal breakpoints
        bp = _find_breakpoints(stage, discharge, n_segments)

    # Build segment boundaries
    edges = [float(np.min(stage))] + bp + [float(np.max(stage))]
    segments: list[RatingSegment] = []
    all_predicted = np.empty_like(discharge)
    all_residuals = np.empty_like(discharge)

    for i in range(n_segments):
        lo, hi = edges[i], edges[i + 1]
        if i < n_segments - 1:
            mask = (stage >= lo) & (stage < hi)
        else:
            mask = (stage >= lo) & (stage <= hi)

        seg_stage = stage[mask]
        seg_discharge = discharge[mask]

        if len(seg_stage) < 3:
            raise ValueError(f"Segment [{lo:.2f}, {hi:.2f}] has fewer than 3 points.")

        a, b, h0, r2, _, seg_residuals = _fit_single_segment(seg_stage, seg_discharge)
        segments.append(RatingSegment(stage_min=lo, stage_max=hi, a=a, b=b, h0=h0, r_squared=r2))
        all_predicted[mask] = _power_law(seg_stage, a, b, h0)
        all_residuals[mask] = seg_residuals

    overall_r2 = _compute_r_squared(discharge, discharge - all_residuals)
    overall_rmse = float(np.sqrt(np.mean(all_residuals**2)))

    logger.info("Segmented rating curve: %d segments, overall R²=%.4f, RMSE=%.4f", n_segments, overall_r2, overall_rmse)

    return RatingCurveResult(
        a=segments[0].a,
        b=segments[0].b,
        h0=segments[0].h0,
        r_squared=overall_r2,
        rmse=overall_rmse,
        n_points=len(stage),
        residuals=all_residuals,
        stage_range=(float(np.min(stage)), float(np.max(stage))),
        segments=segments,
    )


def _find_breakpoints(stage: np.ndarray, discharge: np.ndarray, n_segments: int) -> list[float]:
    """Find optimal breakpoints by grid search minimising total RMSE."""
    sorted_stage = np.sort(stage)
    n = len(sorted_stage)
    # Candidate breakpoints: percentile-based grid
    candidates = np.percentile(sorted_stage, np.linspace(15, 85, min(20, n // 3)))
    candidates = np.unique(candidates)

    if n_segments == 2:
        best_rmse = np.inf
        best_bp: list[float] = [float(np.median(stage))]
        for bp_val in candidates:
            try:
                rmse_total = _evaluate_breakpoints(stage, discharge, [float(bp_val)])
                if rmse_total < best_rmse:
                    best_rmse = rmse_total
                    best_bp = [float(bp_val)]
            except (ValueError, RuntimeError):
                continue
        return best_bp

    # General case: recursive grid search for n_segments > 2
    best_rmse = np.inf
    best_bp = [float(np.percentile(stage, 100 * (i + 1) / n_segments)) for i in range(n_segments - 1)]
    for combo in _breakpoint_combos(candidates, n_segments - 1):
        try:
            rmse_total = _evaluate_breakpoints(stage, discharge, list(combo))
            if rmse_total < best_rmse:
                best_rmse = rmse_total
                best_bp = list(combo)
        except (ValueError, RuntimeError):
            continue
    return best_bp


def _breakpoint_combos(candidates: np.ndarray, k: int):
    """Yield sorted k-combinations of candidate breakpoints."""
    from itertools import combinations

    for combo in combinations(candidates, k):
        yield sorted(combo)


def _evaluate_breakpoints(stage: np.ndarray, discharge: np.ndarray, breakpoints: list[float]) -> float:
    """Evaluate total RMSE for a given set of breakpoints."""
    edges = [float(np.min(stage))] + breakpoints + [float(np.max(stage))]
    n_segments = len(edges) - 1
    total_sse = 0.0
    total_n = 0

    for i in range(n_segments):
        lo, hi = edges[i], edges[i + 1]
        if i < n_segments - 1:
            mask = (stage >= lo) & (stage < hi)
        else:
            mask = (stage >= lo) & (stage <= hi)
        seg_stage = stage[mask]
        seg_discharge = discharge[mask]

        if len(seg_stage) < 3:
            raise ValueError("Segment too small.")

        _, _, _, _, _, residuals = _fit_single_segment(seg_stage, seg_discharge)
        total_sse += np.sum(residuals**2)
        total_n += len(seg_stage)

    return float(np.sqrt(total_sse / total_n))


def predict_discharge(result: RatingCurveResult, stage: np.ndarray | pd.Series) -> np.ndarray:
    """Predict discharge from stage using a fitted rating curve.

    For segmented curves the appropriate segment is selected for each
    stage value.

    Parameters:
        result: Fitted :class:`RatingCurveResult`.
        stage: Stage values to predict at.

    Returns:
        Predicted discharge array.
    """
    stage_arr = np.asarray(stage, dtype=float)

    if result.segments is None:
        return _power_law(stage_arr, result.a, result.b, result.h0)

    predicted = np.empty_like(stage_arr)
    for i, seg in enumerate(result.segments):
        if i < len(result.segments) - 1:
            mask = (stage_arr >= seg.stage_min) & (stage_arr < seg.stage_max)
        else:
            mask = (stage_arr >= seg.stage_min) & (stage_arr <= seg.stage_max)
        # Handle values outside the range by assigning to nearest segment
        if i == 0:
            mask |= stage_arr < seg.stage_min
        if i == len(result.segments) - 1:
            mask |= stage_arr > seg.stage_max
        predicted[mask] = _power_law(stage_arr[mask], seg.a, seg.b, seg.h0)

    return predicted


def predict_stage(result: RatingCurveResult, discharge: np.ndarray | pd.Series) -> np.ndarray:
    """Inverse prediction: compute stage from discharge.

    Solves ``H = (Q / a)^(1/b) + H₀`` for the primary (or first) segment.
    For segmented curves the first segment whose discharge range contains the
    query value is used.

    Parameters:
        result: Fitted :class:`RatingCurveResult`.
        discharge: Discharge values.

    Returns:
        Predicted stage array.
    """
    q_arr = np.asarray(discharge, dtype=float)

    if result.segments is None:
        return _invert_power_law(q_arr, result.a, result.b, result.h0)

    predicted = np.empty_like(q_arr)
    # Pre-compute discharge ranges per segment
    seg_q_ranges: list[tuple[float, float]] = []
    for seg in result.segments:
        q_lo = _power_law(np.array([seg.stage_min]), seg.a, seg.b, seg.h0)[0]
        q_hi = _power_law(np.array([seg.stage_max]), seg.a, seg.b, seg.h0)[0]
        seg_q_ranges.append((min(q_lo, q_hi), max(q_lo, q_hi)))

    assigned = np.zeros(len(q_arr), dtype=bool)
    for i, (seg, (q_lo, q_hi)) in enumerate(zip(result.segments, seg_q_ranges)):
        mask = (~assigned) & (q_arr >= q_lo) & (q_arr <= q_hi)
        if i == 0:
            mask |= (~assigned) & (q_arr < q_lo)
        if i == len(result.segments) - 1:
            mask |= (~assigned) & (q_arr > q_hi)
        predicted[mask] = _invert_power_law(q_arr[mask], seg.a, seg.b, seg.h0)
        assigned |= mask

    # Fallback: any unassigned points use the overall parameters
    if not np.all(assigned):
        predicted[~assigned] = _invert_power_law(q_arr[~assigned], result.a, result.b, result.h0)

    return predicted


def _invert_power_law(q: np.ndarray, a: float, b: float, h0: float) -> np.ndarray:
    """Solve H = (Q / a)^(1/b) + h0."""
    return np.power(q / a, 1.0 / b) + h0


def rating_curve_uncertainty(
    result: RatingCurveResult, stage: np.ndarray, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Compute prediction intervals for discharge estimates.

    Uses a residual-based approach: the standard error of the residuals is
    scaled by the appropriate *t*-distribution quantile.

    Parameters:
        result: Fitted :class:`RatingCurveResult`.
        stage: Stage values at which to compute intervals.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (lower_bound, upper_bound) discharge arrays.
    """
    stage_arr = np.asarray(stage, dtype=float)
    predicted = predict_discharge(result, stage_arr)

    se = float(np.std(result.residuals, ddof=2))
    dof = max(result.n_points - 3, 1)
    t_crit = stats.t.ppf((1 + confidence) / 2, dof)

    margin = t_crit * se
    lower = np.maximum(predicted - margin, 0.0)
    upper = predicted + margin

    return lower, upper


def detect_rating_shift(
    stage: np.ndarray,
    discharge: np.ndarray,
    timestamps: np.ndarray | pd.DatetimeIndex,
    window_size: int = 20,
) -> list[dict]:
    """Detect temporal shifts in the stage-discharge relationship.

    Fits rolling-window rating curves and compares the residual variance
    of successive windows using a chi-squared test.  Significant changes
    are flagged as rating shifts.

    Parameters:
        stage: Stage measurements.
        discharge: Corresponding discharge measurements.
        timestamps: Observation timestamps.
        window_size: Number of observations per rolling window.

    Returns:
        List of dicts, each containing ``'timestamp'``, ``'shift_magnitude'``,
        and ``'p_value'``.

    Raises:
        ValueError: If inputs are invalid.
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    timestamps = pd.DatetimeIndex(timestamps)

    if len(stage) != len(discharge) or len(stage) != len(timestamps):
        raise ValueError("stage, discharge, and timestamps must have equal length.")

    if len(stage) < 2 * window_size:
        logger.warning("Not enough data for shift detection (need >= %d points).", 2 * window_size)
        return []

    # Sort by timestamp
    order = np.argsort(timestamps)
    stage = stage[order]
    discharge = discharge[order]
    timestamps = timestamps[order]

    shifts: list[dict] = []
    step = max(window_size // 2, 1)

    prev_residuals: np.ndarray | None = None

    for start in range(0, len(stage) - window_size + 1, step):
        end = start + window_size
        win_stage = stage[start:end]
        win_discharge = discharge[start:end]
        win_ts = timestamps[start:end]

        try:
            _, _, _, _, _, residuals = _fit_single_segment(win_stage, win_discharge)
        except (ValueError, RuntimeError):
            continue

        centre = win_ts[window_size // 2]

        if prev_residuals is not None:
            var_prev = np.var(prev_residuals, ddof=1)
            var_curr = np.var(residuals, ddof=1)

            if var_prev > 0 and var_curr > 0:
                f_stat = max(var_curr, var_prev) / min(var_curr, var_prev)
                df1 = len(residuals) - 1
                df2 = len(prev_residuals) - 1
                p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))

                shift_magnitude = abs(var_curr - var_prev) / max(var_prev, 1e-12)

                if p_value < 0.05:
                    shifts.append({
                        "timestamp": centre,
                        "shift_magnitude": float(shift_magnitude),
                        "p_value": float(p_value),
                    })

        prev_residuals = residuals

    logger.info("Detected %d potential rating shifts.", len(shifts))
    return shifts


def export_hec_ras(result: RatingCurveResult, filepath: str | Path) -> None:
    """Export a rating curve to HEC-RAS compatible format.

    Writes a simple table of stage-discharge pairs at regular intervals
    spanning the fitted stage range.

    Parameters:
        result: Fitted :class:`RatingCurveResult`.
        filepath: Output file path.
    """
    filepath = Path(filepath)

    stage_min, stage_max = result.stage_range
    stages = np.linspace(stage_min, stage_max, 50)
    discharges = predict_discharge(result, stages)

    lines = [
        "# HEC-RAS Rating Curve Export",
        "# Generated by AquaScope",
        f"# Parameters: a={result.a:.6f}, b={result.b:.6f}, h0={result.h0:.6f}",
        f"# R²={result.r_squared:.4f}, RMSE={result.rmse:.4f}",
        "#",
        "# Stage(m)    Discharge(m3/s)",
    ]

    for h, q in zip(stages, discharges):
        lines.append(f"{h:12.4f}  {q:12.4f}")

    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Exported rating curve to %s", filepath)


def cross_validate_rating(stage: np.ndarray, discharge: np.ndarray, k_folds: int = 5) -> dict:
    """K-fold cross-validation of a rating curve fit.

    Parameters:
        stage: Stage measurements.
        discharge: Discharge measurements.
        k_folds: Number of folds (default 5).

    Returns:
        Dict with ``'mean_rmse'``, ``'std_rmse'``, ``'mean_r2'``, and
        ``'fold_results'`` (list of per-fold dicts).

    Raises:
        ValueError: If inputs are invalid.
    """
    stage = np.asarray(stage, dtype=float)
    discharge = np.asarray(discharge, dtype=float)
    _validate_inputs(stage, discharge)

    n = len(stage)
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    fold_size = n // k_folds
    fold_results: list[dict] = []

    for fold in range(k_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        if len(train_idx) < 5:
            continue

        try:
            a, b, h0, _, _, _ = _fit_single_segment(stage[train_idx], discharge[train_idx])
        except (ValueError, RuntimeError):
            continue

        pred = _power_law(stage[test_idx], a, b, h0)
        residuals = discharge[test_idx] - pred
        rmse = float(np.sqrt(np.mean(residuals**2)))
        r2 = _compute_r_squared(discharge[test_idx], pred)

        fold_results.append({"fold": fold, "rmse": rmse, "r2": r2, "n_test": len(test_idx)})

    rmses = [fr["rmse"] for fr in fold_results]
    r2s = [fr["r2"] for fr in fold_results]

    return {
        "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "std_rmse": float(np.std(rmses)) if rmses else float("nan"),
        "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
        "fold_results": fold_results,
    }
