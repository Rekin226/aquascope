"""Water-quality visualisation functions.

Provides plots for:
- Box plots by station or parameter
- Correlation heatmaps
- WHO guideline exceedance charts
- EDA report summary visualisations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from aquascope.viz.styles import (
    AQUA_PALETTE,
    MULTI_FIGSIZE,
    SERIES_COLOURS,
    WIDE_FIGSIZE,
    _save_or_show,
    apply_aqua_style,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


# ── Box plots ──────────────────────────────────────────────────────────


def plot_boxplot(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    group_col: str = "station_name",
    title: str = "Distribution by Group",
    ylabel: str = "Value",
    figsize: tuple[float, float] = WIDE_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Box plot of *value_col* grouped by *group_col*.

    Parameters
    ----------
    df:
        Long-format DataFrame with at least *value_col* and *group_col*.
    value_col:
        Column containing measurement values.
    group_col:
        Column to group by (station, parameter, etc.).
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

    groups = sorted(df[group_col].dropna().unique())
    data = [df.loc[df[group_col] == g, value_col].dropna().values for g in groups]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        tick_labels=groups,
        medianprops={"color": AQUA_PALETTE["danger"], "linewidth": 1.5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(AQUA_PALETTE["accent"])
        patch.set_alpha(0.7)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(groups) > 6:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    _save_or_show(fig, save_path)
    return fig


# ── Heatmap ────────────────────────────────────────────────────────────


def plot_heatmap(
    df: pd.DataFrame,
    *,
    title: str = "Correlation Heatmap",
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "RdYlBu_r",
    save_path: str | None = None,
) -> Figure:
    """Heatmap of the correlation matrix of numeric columns.

    Parameters
    ----------
    df:
        DataFrame with numeric columns.
    title:
        Plot title.
    figsize:
        Figure size.
    cmap:
        Colour map name.
    save_path:
        Optional save path.

    Returns
    -------
    The matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    apply_aqua_style()
    corr = df.select_dtypes("number").corr()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")

    labels = corr.columns.tolist()
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.values[i, j]
            colour = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=colour, fontsize=8)

    ax.set_title(title)
    _save_or_show(fig, save_path)
    return fig


# ── WHO exceedance chart ───────────────────────────────────────────────


def plot_who_exceedances(
    who_df: pd.DataFrame,
    *,
    title: str = "WHO Guideline Exceedances",
    figsize: tuple[float, float] = WIDE_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Horizontal bar chart of WHO guideline exceedance percentages.

    Parameters
    ----------
    who_df:
        DataFrame returned by ``WaterQualityChallenge.check_who_guidelines()``.
        Expected columns: ``variable``, ``pct_exceedances``, ``status``.
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

    who_df = who_df.sort_values("pct_exceedances", ascending=True)
    colours = [
        AQUA_PALETTE["success"] if s == "PASS" else AQUA_PALETTE["danger"]
        for s in who_df["status"]
    ]

    ax.barh(who_df["variable"], who_df["pct_exceedances"], color=colours, height=0.6)
    ax.axvline(10, color=AQUA_PALETTE["warning"], linestyle="--", linewidth=1, label="10 % threshold")
    ax.set_xlabel("Exceedance (%)")
    ax.set_title(title)
    ax.legend()

    _save_or_show(fig, save_path)
    return fig


# ── EDA summary dashboard ─────────────────────────────────────────────


def plot_eda_summary(
    report,  # EDAReport dataclass
    *,
    title: str = "EDA Summary",
    figsize: tuple[float, float] = MULTI_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Multi-panel summary of an EDAReport.

    Panels:
    1. Record count per parameter (bar)
    2. Missing data (bar)
    3. Outlier count (bar)
    4. Value ranges (error bar: mean ± std)

    Parameters
    ----------
    report:
        An ``EDAReport`` instance from ``aquascope.analysis.eda``.
    title:
        Super-title.
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
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    names = [p.name for p in report.parameters]
    counts = [p.count for p in report.parameters]
    missing = [p.missing for p in report.parameters]
    outliers = [p.outlier_count for p in report.parameters]
    means = [p.mean for p in report.parameters]
    stds = [p.std for p in report.parameters]

    # Record counts
    axes[0, 0].barh(names, counts, color=AQUA_PALETTE["primary"])
    axes[0, 0].set_title("Record Count")

    # Missing data
    axes[0, 1].barh(names, missing, color=AQUA_PALETTE["warning"])
    axes[0, 1].set_title("Missing Values")

    # Outliers
    axes[1, 0].barh(names, outliers, color=AQUA_PALETTE["danger"])
    axes[1, 0].set_title("Outliers (IQR)")

    # Mean ± std
    axes[1, 1].barh(names, means, xerr=stds, color=AQUA_PALETTE["secondary"], capsize=3)
    axes[1, 1].set_title("Mean ± Std Dev")

    fig.suptitle(title, fontsize=16)
    _save_or_show(fig, save_path)
    return fig


# ── Parameter comparison across stations ───────────────────────────────


def plot_param_comparison(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    param_col: str = "parameter",
    station_col: str = "station_name",
    title: str = "Parameter Comparison",
    figsize: tuple[float, float] = MULTI_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Grid of box plots — one per parameter, grouped by station.

    Parameters
    ----------
    df:
        Long-format DataFrame with measurements.
    value_col:
        Column with measurement values.
    param_col:
        Column with parameter names.
    station_col:
        Column with station names.
    title:
        Super-title.
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
    params = sorted(df[param_col].dropna().unique())
    n = len(params)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, param in enumerate(params):
        ax = axes[idx // ncols, idx % ncols]
        subset = df[df[param_col] == param]
        stations = sorted(subset[station_col].dropna().unique())
        data = [subset.loc[subset[station_col] == s, value_col].dropna().values for s in stations]

        if data:
            bp = ax.boxplot(data, patch_artist=True, tick_labels=stations)
            for patch, colour in zip(bp["boxes"], SERIES_COLOURS):
                patch.set_facecolor(colour)
                patch.set_alpha(0.6)

        ax.set_title(param, fontsize=11)
        if len(stations) > 4:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=16)
    _save_or_show(fig, save_path)
    return fig
