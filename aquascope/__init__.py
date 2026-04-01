"""
AquaScope — Open-source water data aggregation and AI-powered research methodology recommender.

Collects water-quality, hydrology, and environmental data from Taiwan's open APIs
and global sources (USGS, UN SDG 6, GEMStat, WQP), then uses AI to suggest
suitable research methodologies for water-related studies.

Quick start::

    from aquascope import collect, recommend, HydroAgent
    from aquascope.hydrology import flow_duration_curve, lyne_hollick
    from aquascope.viz import plot_timeseries, plot_fdc

"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.4.0"
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

    from aquascope.collectors import (
        AquastatCollector,
        CopernicusCollector,
        EUWFDCollector,
        GEMStatCollector,
        JapanMLITCollector,
        KoreaWAMISCollector,
        OpenMeteoCollector,
        SDG6Collector,
        TaiwanCivilIoTCollector,
        TaiwanMOENVCollector,
        TaiwanWRAReservoirCollector,
        TaiwanWRAWaterLevelCollector,
        USGSCollector,
        WaPORCollector,
        WQPCollector,
    )

    params = dict(kwargs)
    collector_map = {
        "taiwan_moenv": lambda p: TaiwanMOENVCollector(api_key=p.pop("api_key", "")),
        "taiwan_wra_level": lambda p: TaiwanWRAWaterLevelCollector(),
        "taiwan_wra_reservoir": TaiwanWRAReservoirCollector,
        "usgs": lambda p: USGSCollector(api_key=p.pop("api_key", "DEMO_KEY")),
        "sdg6": SDG6Collector,
        "gemstat": GEMStatCollector,
        "aquastat": AquastatCollector,
        "taiwan_civil_iot": TaiwanCivilIoTCollector,
        "wqp": WQPCollector,
        "openmeteo": lambda p: OpenMeteoCollector(mode=p.pop("mode", "weather")),
        "copernicus": CopernicusCollector,
        "wapor": WaPORCollector,
        "eu_wfd": EUWFDCollector,
        "japan_mlit": JapanMLITCollector,
        "korea_wamis": KoreaWAMISCollector,
    }

    factory = collector_map.get(source)
    if factory is None:
        msg = f"Unknown source: {source!r}.  Available: {sorted(collector_map)}"
        raise ValueError(msg)

    collector = factory(params) if callable(factory) and not isinstance(factory, type) else factory()  # type: ignore[misc]
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
        # Key classes from new modules
        "GRACEProcessor": "aquascope.groundwater.grace",
        "CMIP6Processor": "aquascope.climate.cmip6",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'aquascope' has no attribute {name!r}")
