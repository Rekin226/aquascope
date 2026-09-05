"""
AquaScope — Open-source water data aggregation and AI-powered research methodology recommender.

Collects water-quality, hydrology, and environmental data from Taiwan's open APIs
and global sources (USGS, UN SDG 6, GEMStat, WQP), then uses AI to suggest
suitable research methodologies for water-related studies.

Quick start::

    from aquascope import collect, find_stations, recommend, HydroAgent
    from aquascope.hydrology import flow_duration_curve, lyne_hollick
    from aquascope.viz import plot_timeseries, plot_fdc

"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.14.0"
__author__ = "AquaScope Contributors"
__license__ = "MIT"


def collect(source: str, **kwargs):
    """Convenience shortcut to create a collector and fetch data.

    Parameters
    ----------
    source:
        Data source name (e.g. ``"usgs"``, ``"openmeteo"``, ``"taiwan_moenv"``).
    **kwargs:
        Passed to the collector's ``fetch_raw()`` method.

    Returns
    -------
    List of normalised Pydantic schema objects.
    """
    source = source.lower()

    from aquascope.registry import SOURCES, build_collector

    if source not in SOURCES:
        msg = f"Unknown source: {source!r}.  Available: {sorted(SOURCES)}"
        raise ValueError(msg)

    params = dict(kwargs)
    ctor_kwargs = {k: params.pop(k) for k in ("mode", "data_type", "dataset_id") if k in params}
    collector = build_collector(source, api_key=params.pop("api_key", None), **ctor_kwargs)
    return collector.collect(**params)


def recommend(file: str | None = None, *, goal: str = "", top_k: int = 5, **kwargs):
    """Get AI methodology recommendations.

    Parameters
    ----------
    file:
        Optional path to a collected JSON data file.
    goal:
        Research goal (free text).
    top_k:
        Number of recommendations to return.

    Returns
    -------
    List of Recommendation dataclass instances.
    """
    import pandas as pd

    from aquascope.ai_engine.recommender import DatasetProfile
    from aquascope.ai_engine.recommender import recommend as recommend_methods
    from aquascope.analysis.eda import profile_dataset

    if file:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(file)
        if path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix!r}")
        profile = profile_dataset(df)
    else:
        profile = DatasetProfile()

    profile.research_goal = goal or profile.research_goal
    for key, value in kwargs.items():
        if hasattr(profile, key):
            setattr(profile, key, value)

    return recommend_methods(profile, top_k=top_k)


def __getattr__(name: str):
    """Lazy imports for convenience top-level access."""
    _lazy = {
        "HydroAgent": "aquascope.ai_engine.agent",
        "ChallengePlanner": "aquascope.ai_engine.planner",
        "ModelRecommender": "aquascope.ai_engine.model_recommender",
        "plan_irrigation": "aquascope.agri",
        "benchmark_aquastat": "aquascope.agri",
        "estimate_wapor_productivity": "aquascope.agri",
        # High-level convenience API (aquascope.api)
        "flood_analysis": "aquascope.api",
        "baseflow_analysis": "aquascope.api",
        "flow_duration": "aquascope.api",
        "compute_all_signatures": "aquascope.api",
        "detect_changepoints": "aquascope.api",
        "fit_copula": "aquascope.api",
        "bayesian_regression": "aquascope.api",
        "ensemble_forecast": "aquascope.api",
        "generate_report": "aquascope.api",
        "groundwater_analysis": "aquascope.api",
        "climate_downscale": "aquascope.api",
        "climate_indices": "aquascope.api",
        # Station catalog / registry (#187)
        "find_stations": "aquascope.registry",
        "station_catalogs": "aquascope.registry",
        "Station": "aquascope.schemas.station",
        # Key classes from new modules
        "GRACEProcessor": "aquascope.groundwater.grace",
        "CMIP6Processor": "aquascope.climate.cmip6",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'aquascope' has no attribute {name!r}")
