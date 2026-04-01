"""AquaScope visualisation module.

Provides publication-quality plots for water-quality analysis, hydrology,
forecasting, and spatial data.  All functions lazily import ``matplotlib``
so the module can be imported even when the ``viz`` optional dependency
group is not installed — an ``ImportError`` is raised only when a plot
function is actually called.

Quick start::

    from aquascope.viz import plot_timeseries, plot_forecast, plot_fdc

    plot_timeseries(df, title="Daily Discharge")
    plot_forecast(observed=train, forecast=pred, save_path="forecast.png")
    plot_fdc(discharge_series, save_path="fdc.svg")
"""

from __future__ import annotations

# Diagnostics
from aquascope.viz.diagnostics import (
    diagnostic_panel,
    pp_plot,
    qq_plot,
    return_level_plot,
)

# Hydrology
from aquascope.viz.hydro import (
    plot_fdc,
    plot_hydrograph,
    plot_return_periods,
    plot_spi_timeline,
)

# Water-quality
from aquascope.viz.quality import (
    plot_boxplot,
    plot_eda_summary,
    plot_heatmap,
    plot_param_comparison,
    plot_who_exceedances,
)

# Spatial
from aquascope.viz.spatial import plot_station_map, plot_station_scatter

# Styling
from aquascope.viz.styles import AQUA_PALETTE, apply_aqua_style

# Time-series & forecasting
from aquascope.viz.timeseries import (
    plot_forecast,
    plot_multi_param,
    plot_observed_vs_predicted,
    plot_residuals,
    plot_timeseries,
)

__all__ = [
    # diagnostics
    "qq_plot",
    "pp_plot",
    "return_level_plot",
    "diagnostic_panel",
    # timeseries
    "plot_timeseries",
    "plot_multi_param",
    "plot_forecast",
    "plot_observed_vs_predicted",
    "plot_residuals",
    # quality
    "plot_boxplot",
    "plot_heatmap",
    "plot_who_exceedances",
    "plot_eda_summary",
    "plot_param_comparison",
    # spatial
    "plot_station_map",
    "plot_station_scatter",
    # hydrology
    "plot_fdc",
    "plot_hydrograph",
    "plot_spi_timeline",
    "plot_return_periods",
    # styling
    "AQUA_PALETTE",
    "apply_aqua_style",
]
