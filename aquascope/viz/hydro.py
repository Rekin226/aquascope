"""Hydrology-specific visualisation functions.

Standard plots every hydrologist expects:
- Flow duration curves (FDC)
- Hydrograph with baseflow separation
- SPI drought timeline
- Return period plots
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from aquascope.viz.styles import (
    AQUA_PALETTE,
    DEFAULT_FIGSIZE,
    SPI_COLOURS,
    WIDE_FIGSIZE,
    _save_or_show,
    apply_aqua_style,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ── Flow Duration Curve ────────────────────────────────────────────────


def plot_fdc(
    discharge: pd.Series,
    *,
    title: str = "Flow Duration Curve",
    ylabel: str = "Discharge (m³/s)",
    log_scale: bool = True,
    percentiles: list[float] | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Plot a flow duration curve.

    Parameters
    ----------
    discharge:
        Series of discharge values.
    title, ylabel:
        Axis labels.
    log_scale:
        If ``True``, use a log scale on the y-axis.
    percentiles:
        Exceedance percentiles to annotate (e.g. ``[5, 50, 95]``).
    figsize:
        Figure size.
    save_path:
        Optional save path.

    Returns
    -------
    The matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    apply_aqua_style()
    fig, ax = plt.subplots(figsize=figsize)

    sorted_q = np.sort(discharge.dropna().values)[::-1]
    n = len(sorted_q)
    exceedance = np.arange(1, n + 1) / n * 100

    ax.plot(exceedance, sorted_q, color=AQUA_PALETTE["primary"], linewidth=1.5)
    ax.fill_between(exceedance, 0, sorted_q, alpha=0.15, color=AQUA_PALETTE["accent"])

    if log_scale:
        ax.set_yscale("log")

    if percentiles is None:
        percentiles = [5, 10, 50, 90, 95]

    for pct in percentiles:
        idx = int(pct / 100 * n)
        idx = min(idx, n - 1)
        val = sorted_q[idx]
        ax.axvline(pct, color=AQUA_PALETTE["neutral"], linestyle=":", linewidth=0.8)
        ax.annotate(
            f"Q{pct}={val:.2f}",
            xy=(pct, val),
            xytext=(pct + 2, val),
            fontsize=8,
            color=AQUA_PALETTE["dark"],
        )

    ax.set_xlabel("Exceedance Probability (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 100)

    _save_or_show(fig, save_path)
    return fig


# ── Hydrograph ─────────────────────────────────────────────────────────


def plot_hydrograph(
    discharge: pd.DataFrame,
    *,
    total_col: str = "discharge",
    baseflow_col: str | None = "baseflow",
    precip_col: str | None = None,
    title: str = "Hydrograph",
    figsize: tuple[float, float] = WIDE_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Plot a hydrograph with optional baseflow and precipitation overlay.

    Parameters
    ----------
    discharge:
        DataFrame with DatetimeIndex and at least a total discharge column.
    total_col:
        Column name for total discharge.
    baseflow_col:
        Column name for baseflow (shaded underneath). ``None`` to skip.
    precip_col:
        Column for inverted precipitation bars on a secondary y-axis.
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
    import matplotlib.pyplot as plt

    apply_aqua_style()
    fig, ax1 = plt.subplots(figsize=figsize)

    # Total discharge
    ax1.plot(discharge.index, discharge[total_col], color=AQUA_PALETTE["primary"], linewidth=1.2, label="Total Q")

    # Baseflow
    if baseflow_col and baseflow_col in discharge.columns:
        ax1.fill_between(
            discharge.index, 0, discharge[baseflow_col],
            alpha=0.3, color=AQUA_PALETTE["secondary"], label="Baseflow",
        )
        discharge[total_col] - discharge[baseflow_col]
        ax1.fill_between(
            discharge.index, discharge[baseflow_col], discharge[total_col],
            alpha=0.2, color=AQUA_PALETTE["warning"], label="Quickflow",
        )

    ax1.set_ylabel("Discharge (m³/s)")
    ax1.set_xlabel("Date")

    # Precipitation (inverted on top)
    if precip_col and precip_col in discharge.columns:
        ax2 = ax1.twinx()
        ax2.bar(discharge.index, discharge[precip_col], color=AQUA_PALETTE["accent"], alpha=0.5, label="Precip")
        ax2.set_ylabel("Precipitation (mm)")
        ax2.invert_yaxis()
        ax2.set_ylim(ax2.get_ylim()[0], 0)
        ax2.legend(loc="upper right")

    ax1.set_title(title)
    ax1.legend(loc="upper left")
    fig.autofmt_xdate()

    _save_or_show(fig, save_path)
    return fig


# ── SPI Drought Timeline ──────────────────────────────────────────────


def plot_spi_timeline(
    spi_df: pd.DataFrame,
    *,
    spi_col: str = "spi_3",
    title: str = "SPI Drought Timeline",
    figsize: tuple[float, float] = WIDE_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Plot SPI values as a bar chart coloured by drought severity.

    Parameters
    ----------
    spi_df:
        DataFrame with DatetimeIndex and SPI column(s).
    spi_col:
        Which SPI column to plot.
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
    import matplotlib.pyplot as plt

    apply_aqua_style()
    fig, ax = plt.subplots(figsize=figsize)

    values = spi_df[spi_col].dropna()
    colours = []
    for v in values:
        if v >= 2.0:
            colours.append(SPI_COLOURS["extremely_wet"])
        elif v >= 1.5:
            colours.append(SPI_COLOURS["very_wet"])
        elif v >= 1.0:
            colours.append(SPI_COLOURS["moderately_wet"])
        elif v >= -1.0:
            colours.append(SPI_COLOURS["near_normal"])
        elif v >= -1.5:
            colours.append(SPI_COLOURS["moderately_dry"])
        elif v >= -2.0:
            colours.append(SPI_COLOURS["severely_dry"])
        else:
            colours.append(SPI_COLOURS["extremely_dry"])

    ax.bar(values.index, values.values, color=colours, width=20)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(-1, color=AQUA_PALETTE["warning"], linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-2, color=AQUA_PALETTE["danger"], linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(1, color=AQUA_PALETTE["secondary"], linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(2, color=AQUA_PALETTE["dark"], linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_ylabel("SPI")
    ax.set_title(title)
    fig.autofmt_xdate()

    _save_or_show(fig, save_path)
    return fig


# ── Return Period Plot ─────────────────────────────────────────────────


def plot_return_periods(
    return_periods: dict[int, float],
    *,
    observed_max: float | None = None,
    title: str = "Flood Return Periods",
    ylabel: str = "Discharge (m³/s)",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Plot return period estimates with optional observed maximum.

    Parameters
    ----------
    return_periods:
        Mapping of return period (years) to estimated discharge.
    observed_max:
        If given, draw a horizontal line at the observed maximum.
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
    import matplotlib.pyplot as plt

    apply_aqua_style()
    fig, ax = plt.subplots(figsize=figsize)

    years = sorted(return_periods.keys())
    values = [return_periods[y] for y in years]

    ax.plot(years, values, "o-", color=AQUA_PALETTE["primary"], linewidth=2, markersize=8, label="GEV estimate")

    if observed_max is not None:
        ax.axhline(observed_max, color=AQUA_PALETTE["danger"], linestyle="--", linewidth=1.5, label="Observed max")

    ax.set_xscale("log")
    ax.set_xlabel("Return Period (years)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    # Annotate each point
    for y, v in zip(years, values):
        ax.annotate(f"{v:.1f}", (y, v), textcoords="offset points", xytext=(0, 10), fontsize=9, ha="center")

    _save_or_show(fig, save_path)
    return fig
