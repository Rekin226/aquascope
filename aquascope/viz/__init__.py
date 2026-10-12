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

import importlib
from typing import Any

#: Each public name and the submodule that defines it. The submodules import matplotlib at their top, so they
#: load on first use (PEP 562) and ``import aquascope.viz.publication`` stays free of matplotlib until a figure
#: is drawn (the Studio imports it in the browser before the plotting package is there).
_EXPORTS: dict[str, str] = {
    # diagnostics
    "qq_plot": "diagnostics",
    "pp_plot": "diagnostics",
    "double_mass_plot": "diagnostics",
    "return_level_plot": "diagnostics",
    "diagnostic_panel": "diagnostics",
    # timeseries
    "plot_timeseries": "timeseries",
    "plot_multi_param": "timeseries",
    "plot_forecast": "timeseries",
    "plot_observed_vs_predicted": "timeseries",
    "plot_residuals": "timeseries",
    # quality
    "plot_boxplot": "quality",
    "plot_heatmap": "quality",
    "plot_who_exceedances": "quality",
    "plot_eda_summary": "quality",
    "plot_param_comparison": "quality",
    # spatial
    "plot_station_map": "spatial",
    "plot_station_scatter": "spatial",
    # hydrology
    "plot_budyko": "hydro",
    "plot_fdc": "hydro",
    "plot_hydrograph": "hydro",
    "plot_spi_timeline": "hydro",
    "plot_return_periods": "hydro",
    # styling
    "AQUA_PALETTE": "styles",
    "apply_aqua_style": "styles",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'aquascope.viz' has no attribute {name!r}")
    value = getattr(importlib.import_module(f"aquascope.viz.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
