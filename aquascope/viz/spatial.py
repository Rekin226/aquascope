"""Spatial / map visualisations for station data.

Uses *folium* for interactive HTML maps and *matplotlib* for static
scatter maps.  Both rendering backends are lazily imported so the
module stays usable even when only one library is installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from aquascope.utils.imports import require
from aquascope.viz.styles import (
    AQUA_PALETTE,
    DEFAULT_FIGSIZE,
    RISK_COLOURS,
    _save_or_show,
    apply_aqua_style,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# ── Folium interactive map ─────────────────────────────────────────────


def plot_station_map(
    stations: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    label_col: str = "station_name",
    value_col: str | None = None,
    colour_col: str | None = None,
    title: str = "Station Map",
    save_path: str | None = None,
) -> object:
    """Create an interactive Folium map of monitoring stations.

    Parameters
    ----------
    stations:
        DataFrame with at least latitude/longitude columns.
    lat_col, lon_col:
        Column names for coordinates.
    label_col:
        Column used for popup labels.
    value_col:
        Optional column whose value is shown in tooltips.
    colour_col:
        Optional column for colour-coding markers (e.g. risk level).
        Values are mapped through ``RISK_COLOURS``; unrecognised values
        use AquaScope primary blue.
    title:
        Map title (shown in popup header).
    save_path:
        If provided, save the HTML map to this path.

    Returns
    -------
    A ``folium.Map`` object.
    """
    folium = require("folium", feature="interactive maps")

    center_lat = stations[lat_col].mean()
    center_lon = stations[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="cartodbpositron")

    for _, row in stations.iterrows():
        colour = AQUA_PALETTE["primary"]
        if colour_col and colour_col in row.index:
            colour = RISK_COLOURS.get(str(row[colour_col]), AQUA_PALETTE["primary"])

        popup_parts = [f"<b>{title}</b><br>{row.get(label_col, '')}"]
        if value_col and value_col in row.index:
            popup_parts.append(f"<br>Value: {row[value_col]:.3f}")

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=8,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            popup=folium.Popup("".join(popup_parts), max_width=250),
            tooltip=row.get(label_col, ""),
        ).add_to(m)

    if save_path:
        m.save(save_path)
        logger.info("Map saved to %s", save_path)

    return m


# ── Static matplotlib scatter map ──────────────────────────────────────


def plot_station_scatter(
    stations: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    label_col: str = "station_name",
    value_col: str | None = None,
    title: str = "Station Locations",
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    save_path: str | None = None,
) -> Figure:
    """Static scatter plot of station locations coloured by value.

    Parameters
    ----------
    stations:
        DataFrame with latitude/longitude and optional value column.
    lat_col, lon_col:
        Column names for coordinates.
    label_col:
        Column for point labels.
    value_col:
        If provided, colour-code by this column and add a colour bar.
    title:
        Plot title.
    cmap:
        Matplotlib colour map name (used when *value_col* is set).
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

    if value_col and value_col in stations.columns:
        sc = ax.scatter(
            stations[lon_col], stations[lat_col],
            c=stations[value_col], cmap=cmap, s=60, edgecolors="white", linewidths=0.5,
        )
        fig.colorbar(sc, ax=ax, shrink=0.8, label=value_col)
    else:
        ax.scatter(
            stations[lon_col], stations[lat_col],
            color=AQUA_PALETTE["primary"], s=60, edgecolors="white", linewidths=0.5,
        )

    for _, row in stations.iterrows():
        if label_col in row.index and pd.notna(row[label_col]):
            ax.annotate(
                str(row[label_col]),
                (row[lon_col], row[lat_col]),
                textcoords="offset points", xytext=(5, 5), fontsize=7,
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)

    _save_or_show(fig, save_path)
    return fig
