"""Diagnostic plots for flood frequency analysis.

Provides Q-Q, P-P, return level, and density comparison plots to
assess the quality of fitted extreme-value distributions.

Supported distributions:

- **GEV** (``genextreme``)
- **LP3** (``pearson3``, applied in log10 space)
- **Gumbel** (``gumbel_r``)
- **GPD** (``genpareto``)
- **Weibull** (``weibull_min``)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from aquascope.viz.styles import (
    AQUA_PALETTE,
    MULTI_FIGSIZE,
    SQUARE_FIGSIZE,
    _save_or_show,
    apply_aqua_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from aquascope.hydrology.flood_frequency import FloodFreqResult

logger = logging.getLogger(__name__)

# ── Distribution mapping ────────────────────────────────────────────────
_DIST_MAP = {
    "gev": "genextreme",
    "lp3": "pearson3",
    "gumbel": "gumbel_r",
    "gpd": "genpareto",
    "weibull": "weibull_min",
}


def _get_scipy_dist(distribution: str):
    """Return the scipy.stats distribution object for *distribution*."""
    import scipy.stats as st

    key = distribution.lower()
    if key not in _DIST_MAP:
        msg = f"Unknown distribution {distribution!r}; choose from {list(_DIST_MAP)}"
        raise ValueError(msg)
    return getattr(st, _DIST_MAP[key])


def _prepare_data(observed, distribution: str):
    """Sort observed data and apply log10 transform for LP3."""
    data = np.sort(np.asarray(observed, dtype=np.float64))
    is_lp3 = distribution.lower() == "lp3"
    if is_lp3:
        data = np.log10(data[data > 0])
    return data, is_lp3


# ── Q-Q plot ────────────────────────────────────────────────────────────

def qq_plot(
    observed,
    distribution: str,
    params: tuple,
    *,
    ax: Axes | None = None,
    save_path: str | None = None,
    title: str | None = None,
) -> Figure:
    """Quantile-Quantile plot comparing observed data against a fitted distribution.

    Parameters
    ----------
    observed : array-like
        Observed data (e.g., annual maximum discharge).
    distribution : str
        Distribution name: ``"gev"``, ``"lp3"``, ``"gumbel"``, ``"weibull"``, ``"gpd"``.
    params : tuple
        Distribution parameters (shape, loc, scale) from ``scipy.stats`` fit.
    ax : Axes, optional
        Matplotlib axes to draw on.  A new figure is created when ``None``.
    save_path : str, optional
        If given, save the figure to this path.
    title : str, optional
        Plot title.  Defaults to ``"Q-Q Plot (<dist>)"``.

    Returns
    -------
    Figure
        The matplotlib ``Figure`` containing the Q-Q plot.
    """
    import matplotlib.pyplot as plt

    dist = _get_scipy_dist(distribution)
    data, is_lp3 = _prepare_data(observed, distribution)
    n = len(data)

    # Plotting positions (Cunnane)
    pp = (np.arange(1, n + 1) - 0.4) / (n + 0.2)
    theoretical = dist.ppf(pp, *params)

    apply_aqua_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    else:
        fig = ax.get_figure()

    ax.scatter(theoretical, data, color=AQUA_PALETTE["primary"], edgecolors="white", s=40, zorder=3)

    # 1:1 reference line
    lo = min(theoretical.min(), data.min())
    hi = max(theoretical.max(), data.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            color=AQUA_PALETTE["danger"], linestyle="--", linewidth=1.5, label="1:1 line")

    ylabel = "Observed (log₁₀)" if is_lp3 else "Observed"
    xlabel = f"Theoretical ({distribution.upper()})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Q-Q Plot ({distribution.upper()})")
    ax.legend()

    _save_or_show(fig, save_path)
    return fig


# ── P-P plot ────────────────────────────────────────────────────────────

def pp_plot(
    observed,
    distribution: str,
    params: tuple,
    *,
    ax: Axes | None = None,
    save_path: str | None = None,
    title: str | None = None,
) -> Figure:
    """Probability-Probability plot.

    Compares empirical CDFs with the theoretical CDF of the fitted distribution.

    Parameters
    ----------
    observed : array-like
        Observed data (e.g., annual maximum discharge).
    distribution : str
        Distribution name: ``"gev"``, ``"lp3"``, ``"gumbel"``, ``"weibull"``, ``"gpd"``.
    params : tuple
        Distribution parameters from ``scipy.stats`` fit.
    ax : Axes, optional
        Matplotlib axes to draw on.
    save_path : str, optional
        If given, save the figure to this path.
    title : str, optional
        Plot title.

    Returns
    -------
    Figure
        The matplotlib ``Figure`` containing the P-P plot.
    """
    import matplotlib.pyplot as plt

    dist = _get_scipy_dist(distribution)
    data, _ = _prepare_data(observed, distribution)
    n = len(data)

    # Empirical probabilities (Cunnane plotting position)
    empirical = (np.arange(1, n + 1) - 0.4) / (n + 0.2)
    theoretical = dist.cdf(data, *params)

    apply_aqua_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    else:
        fig = ax.get_figure()

    ax.scatter(theoretical, empirical, color=AQUA_PALETTE["primary"], edgecolors="white", s=40, zorder=3)
    ax.plot([0, 1], [0, 1], color=AQUA_PALETTE["danger"], linestyle="--", linewidth=1.5, label="1:1 line")

    ax.set_xlabel(f"Theoretical CDF ({distribution.upper()})")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title or f"P-P Plot ({distribution.upper()})")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    _save_or_show(fig, save_path)
    return fig


# ── Return level plot ───────────────────────────────────────────────────

def return_level_plot(
    result: FloodFreqResult,
    *,
    ci: bool = True,
    ax: Axes | None = None,
    save_path: str | None = None,
) -> Figure:
    """Return level plot with confidence intervals.

    Shows return period (x-axis, log-scale) versus discharge (y-axis)
    with optional confidence-interval bands.

    Parameters
    ----------
    result : FloodFreqResult
        Result from a flood frequency fit (e.g., ``fit_gev``).
    ci : bool
        Whether to draw the confidence-interval band (default ``True``).
    ax : Axes, optional
        Matplotlib axes to draw on.
    save_path : str, optional
        If given, save the figure to this path.

    Returns
    -------
    Figure
        The matplotlib ``Figure`` containing the return level plot.
    """
    import matplotlib.pyplot as plt

    apply_aqua_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    years = sorted(result.return_periods.keys())
    levels = [result.return_periods[y] for y in years]

    ax.plot(years, levels, "o-", color=AQUA_PALETTE["primary"], linewidth=2, markersize=7, label="Return level")

    if ci and result.confidence_intervals:
        lower = [result.confidence_intervals[y][0] for y in years if y in result.confidence_intervals]
        upper = [result.confidence_intervals[y][1] for y in years if y in result.confidence_intervals]
        ci_years = [y for y in years if y in result.confidence_intervals]
        ax.fill_between(ci_years, lower, upper, alpha=0.2, color=AQUA_PALETTE["accent"], label="90% CI")

    # Plot observed annual maxima as ticks
    if result.annual_max is not None:
        am = np.sort(result.annual_max.values)[::-1]
        n = len(am)
        obs_rp = (n + 1) / np.arange(1, n + 1)
        ax.scatter(obs_rp, am, marker="x", color=AQUA_PALETTE["warning"], s=30, zorder=4, label="Observed")

    ax.set_xscale("log")
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title(f"Return Level Plot ({result.distribution})")
    ax.legend()

    _save_or_show(fig, save_path)
    return fig


# ── Diagnostic panel ────────────────────────────────────────────────────

def diagnostic_panel(
    observed,
    distribution: str,
    params: tuple,
    result: FloodFreqResult | None = None,
    *,
    save_path: str | None = None,
) -> Figure:
    """4-panel diagnostic: Q-Q, P-P, return level, density comparison.

    Parameters
    ----------
    observed : array-like
        Observed data (e.g., annual maximum discharge).
    distribution : str
        Distribution name.
    params : tuple
        Distribution parameters from ``scipy.stats`` fit.
    result : FloodFreqResult, optional
        If provided, the return level panel is drawn using this result.
        Otherwise, the return level panel is replaced with a histogram.
    save_path : str, optional
        If given, save the composite figure to this path.

    Returns
    -------
    Figure
        The matplotlib ``Figure`` with four subplots.
    """
    import matplotlib.pyplot as plt

    apply_aqua_style()
    fig, axes = plt.subplots(2, 2, figsize=MULTI_FIGSIZE)

    # Top-left: Q-Q
    qq_plot(observed, distribution, params, ax=axes[0, 0])

    # Top-right: P-P
    pp_plot(observed, distribution, params, ax=axes[0, 1])

    # Bottom-left: return level or histogram
    if result is not None:
        return_level_plot(result, ax=axes[1, 0])
    else:
        _density_histogram(observed, distribution, params, ax=axes[1, 0])

    # Bottom-right: density comparison
    _density_histogram(observed, distribution, params, ax=axes[1, 1])

    _save_or_show(fig, save_path)
    return fig


def _density_histogram(
    observed,
    distribution: str,
    params: tuple,
    *,
    ax: Axes | None = None,
) -> None:
    """Overlay histogram of observed data with fitted PDF."""
    dist = _get_scipy_dist(distribution)
    data, is_lp3 = _prepare_data(observed, distribution)

    ax.hist(data, bins="auto", density=True, alpha=0.5, color=AQUA_PALETTE["accent"], edgecolor="white",
            label="Observed")

    x = np.linspace(data.min(), data.max(), 200)
    pdf = dist.pdf(x, *params)
    ax.plot(x, pdf, color=AQUA_PALETTE["danger"], linewidth=2, label=f"{distribution.upper()} PDF")

    xlabel = "Value (log₁₀)" if is_lp3 else "Value"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title("Density Comparison")
    ax.legend()
