"""Climate projections, downscaling, scenario analysis, and climate indices."""

from aquascope.climate.cmip6 import SSP, CMIP6Processor, EnsembleStats, TrendResult
from aquascope.climate.downscaling import (
    DownscalingMetrics,
    bias_correction,
    delta_method,
    evaluate_downscaling,
    quantile_delta_mapping,
    quantile_mapping,
)
from aquascope.climate.indices import (
    AridityResult,
    CDDResult,
    CWDResult,
    HeatWaveResult,
    aridity_index,
    consecutive_dry_days,
    consecutive_wet_days,
    heat_wave_index,
    palmer_drought_severity_index,
    precipitation_concentration_index,
)
from aquascope.climate.scenarios import (
    DroughtStats,
    ReturnPeriodShift,
    WetSpellStats,
    drought_frequency,
    idf_adjustment,
    return_period_shift,
    scenario_comparison,
    wet_spell_analysis,
)

__all__ = [
    # cmip6
    "CMIP6Processor",
    "EnsembleStats",
    "SSP",
    "TrendResult",
    # downscaling
    "DownscalingMetrics",
    "bias_correction",
    "delta_method",
    "evaluate_downscaling",
    "quantile_delta_mapping",
    "quantile_mapping",
    # indices
    "AridityResult",
    "CDDResult",
    "CWDResult",
    "HeatWaveResult",
    "aridity_index",
    "consecutive_dry_days",
    "consecutive_wet_days",
    "heat_wave_index",
    "palmer_drought_severity_index",
    "precipitation_concentration_index",
    # scenarios
    "DroughtStats",
    "ReturnPeriodShift",
    "WetSpellStats",
    "drought_frequency",
    "idf_adjustment",
    "return_period_shift",
    "scenario_comparison",
    "wet_spell_analysis",
]
