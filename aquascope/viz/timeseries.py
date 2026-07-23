"""Time-series and forecast visualisation functions.

Provides publication-quality plots for:
- Single / multi-parameter time-series
- Forecast plots with confidence intervals
- Observed vs predicted comparison
- Model evaluation residual plots
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from aquascope.utils.imports import require
from aquascope.viz.styles import (
    AQUA_PALETTE,
    DEFAULT_FIGSIZE,
    SERIES_COLOURS,
    WIDE_FIGSIZE,
    _save_or_show,
    apply_aqua_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ── Single time-series ──────────────────────────────────────────────────


def plot_timeseries(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    title: str = "Time Series",
    ylabel: str = "Value",
    xlabel: str = "Date",
    colour: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Plot a single time-series from a DatetimeIndex DataFrame.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex and at least one value column.
    value_col:
        Column name containing the values to plot.
    title, ylabel, xlabel:
        Axis labels.
    colour:
        Line colour.  Defaults to AquaScope primary blue.
    ax:
        Optional pre-existing Axes to draw on.
    figsize:
        Figure size if creating a new figure.
    save_path:
        If provided, save figure to this path instead of showing.

    Returns
    -------
    The matplotlib Figure.
    """
    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt

    apply_aqua_style()
    colour = colour or AQUA_PALETTE["primary"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(df.index, df[value_col], color=colour, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()

    _save_or_show(fig, save_path)
    return fig


# ── Multi-parameter overlay ────────────────────────────────────────────


def plot_multi_param(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    title: str = "Multi-Parameter Time Series",
    ylabel: str = "Value",
    figsize: tuple[float, float] = WIDE_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Overlay multiple columns on the same axes.

    Parameters
    ----------
    df:
        DataFrame with DatetimeIndex and one or more numeric columns.
    columns:
        Subset of column names to plot.  ``None`` plots all numeric columns.
    title, ylabel:
        Axis labels.
    figsize:
        Figure size.
    save_path:
        Optional save path.

    Returns
    -------
    The matplotlib Figure.
    """
    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt

    apply_aqua_style()

    if columns is None:
        columns = df.select_dtypes("number").columns.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    for idx, col in enumerate(columns):
        ax.plot(
            df.index,
            df[col],
            color=SERIES_COLOURS[idx % len(SERIES_COLOURS)],
            linewidth=1.2,
            label=col,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.autofmt_xdate()

    _save_or_show(fig, save_path)
    return fig


# ── Forecast with confidence bands ─────────────────────────────────────


def plot_forecast(
    observed: pd.DataFrame | None = None,
    forecast: pd.DataFrame | None = None,
    *,
    obs_col: str = "value",
    pred_col: str = "yhat",
    lower_col: str = "yhat_lower",
    upper_col: str = "yhat_upper",
    title: str = "Forecast",
    ylabel: str = "Value",
    figsize: tuple[float, float] = WIDE_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Plot observed data with forecast and confidence interval bands.

    Parameters
    ----------
    observed:
        Historical DataFrame (DatetimeIndex + value column).
    forecast:
        Forecast DataFrame with ``yhat``, ``yhat_lower``, ``yhat_upper``.
    obs_col:
        Column name in *observed*.
    pred_col, lower_col, upper_col:
        Column names in *forecast*.
    title, ylabel:
        Axis labels.
    figsize:
        Figure size.
    save_path:
        Optional save path.

    Returns
    -------
    The matplotlib Figure.
    """
    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt

    apply_aqua_style()
    fig, ax = plt.subplots(figsize=figsize)

    if observed is not None and obs_col in observed.columns:
        ax.plot(
            observed.index,
            observed[obs_col],
            color=AQUA_PALETTE["primary"],
            linewidth=1.2,
            label="Observed",
        )

    if forecast is not None and pred_col in forecast.columns:
        ax.plot(
            forecast.index,
            forecast[pred_col],
            color=AQUA_PALETTE["danger"],
            linewidth=1.4,
            linestyle="--",
            label="Forecast",
        )
        if lower_col in forecast.columns and upper_col in forecast.columns:
            ax.fill_between(
                forecast.index,
                forecast[lower_col],
                forecast[upper_col],
                alpha=0.2,
                color=AQUA_PALETTE["danger"],
                label="95 % CI",
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.autofmt_xdate()

    _save_or_show(fig, save_path)
    return fig


# ── Observed vs Predicted scatter ──────────────────────────────────────


def plot_observed_vs_predicted(
    observed: pd.Series,
    predicted: pd.Series,
    *,
    metrics: dict | None = None,
    title: str = "Observed vs Predicted",
    figsize: tuple[float, float] = (7, 7),
    save_path: str | None = None,
) -> Figure:
    """Scatter plot of observed vs predicted values with 1:1 line.

    Parameters
    ----------
    observed:
        Observed values.
    predicted:
        Predicted values (same index as *observed*).
    metrics:
        Optional dict of evaluation metrics (NSE, KGE, …) to annotate.
    title:
        Plot title.
    figsize:
        Figure size.
    save_path:
        Optional save path.

    Returns
    -------
    The matplotlib Figure.
    """
    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt
    import numpy as np

    apply_aqua_style()
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(observed, predicted, alpha=0.5, s=20, color=AQUA_PALETTE["primary"])

    lo = min(observed.min(), predicted.min())
    hi = max(observed.max(), predicted.max())
    margin = (hi - lo) * 0.05
    line = np.linspace(lo - margin, hi + margin, 100)
    ax.plot(line, line, "--", color=AQUA_PALETTE["neutral"], linewidth=1, label="1:1")

    if metrics:
        text = "\n".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
        ax.text(0.05, 0.95, text, transform=ax.transAxes, va="top", fontsize=10,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": AQUA_PALETTE["light"], "alpha": 0.8})

    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()

    _save_or_show(fig, save_path)
    return fig


# ── Residual plot ──────────────────────────────────────────────────────


def plot_residuals(
    observed: pd.Series,
    predicted: pd.Series,
    *,
    title: str = "Residuals",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Plot residuals (observed − predicted) over the index.

    Parameters
    ----------
    observed:
        Observed values.
    predicted:
        Predicted values.
    title:
        Plot title.
    figsize:
        Figure size.
    save_path:
        Optional save path.

    Returns
    -------
    The matplotlib Figure.
    """
    require("matplotlib", feature="plotting")
    import matplotlib.pyplot as plt

    apply_aqua_style()
    residuals = observed - predicted

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))

    # Time-series of residuals
    axes[0].plot(residuals.index, residuals, color=AQUA_PALETTE["primary"], linewidth=0.8)
    axes[0].axhline(0, color=AQUA_PALETTE["danger"], linewidth=1, linestyle="--")
    axes[0].set_title("Residuals over Time")
    axes[0].set_ylabel("Residual")

    # Histogram
    axes[1].hist(residuals.dropna(), bins=30, color=AQUA_PALETTE["secondary"], edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color=AQUA_PALETTE["danger"], linewidth=1, linestyle="--")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")

    fig.suptitle(title, fontsize=14, y=1.02)
    _save_or_show(fig, save_path)
    return fig
