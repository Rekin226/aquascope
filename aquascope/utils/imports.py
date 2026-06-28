"""Lazy import helpers with informative error messages."""
from __future__ import annotations

_INSTALL_MAP = {
    "sklearn": "ml",
    "xgboost": "ml",
    "statsmodels": "ml",
    "pymannkendall": "ml",
    "prophet": "forecast",
    "torch": "forecast",
    "matplotlib": "viz",
    "seaborn": "viz",
    "folium": "viz",
    "xarray": "scientific",
    "netCDF4": "scientific",
    "h5py": "scientific",
    "tables": "scientific",
    "rasterio": "spatial",
    "geopandas": "spatial",
    "shapely": "spatial",
    "pyproj": "spatial",
    "streamlit": "dashboard",
    "openai": "llm",
}


def require(module_name: str, *, feature: str = "", group: str | None = None) -> object:
    """Import a module, raising a helpful error if it's missing.

    Args:
        module_name: Importable module name (e.g. ``"xarray"``).
        feature: Human-readable feature name shown in the error.
        group: Override the suggested extras group. Defaults to the
            module's entry in ``_INSTALL_MAP``. Use this when a module is
            reachable from more than one extra (e.g. ``xarray`` ships in
            both ``scientific`` and ``interop``).
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError:
        group = group or _INSTALL_MAP.get(module_name, module_name)
        feat = f" ({feature})" if feature else ""
        msg = (
            f"Missing optional dependency '{module_name}'{feat}. "
            f"Install with: pip install 'aquascope[{group}]'"
        )
        raise ImportError(msg) from None
