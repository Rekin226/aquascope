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
    "streamlit": "dashboard",
    "openai": "llm",
}


def require(module_name: str, *, feature: str = "") -> object:
    """Import a module, raising a helpful error if it's missing."""
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError:
        group = _INSTALL_MAP.get(module_name, module_name)
        feat = f" ({feature})" if feature else ""
        msg = (
            f"Missing optional dependency '{module_name}'{feat}. "
            f"Install with: pip install 'aquascope[{group}]'"
        )
        raise ImportError(msg) from None
