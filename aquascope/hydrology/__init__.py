"""AquaScope hydrology module.

Standard hydrological analysis tools:

- **Flow duration curves** and low-flow statistics (Q95, 7Q10, 30Q5)
- **Baseflow separation** (Lyne–Hollick, Eckhardt digital filters)
- **Recession analysis** (segment identification + MRC fitting)
- **Flood frequency analysis** (GEV, Log-Pearson Type III)
- **Stage-discharge rating curves** (power-law fit, segmented curves, shift detection)

Quick start::

    from aquascope.hydrology import flow_duration_curve, lyne_hollick, recession_analysis, fit_gev

    fdc = flow_duration_curve(discharge_series)
    print(f"Q95 = {fdc.percentiles[95]:.2f} m³/s")

    bf = lyne_hollick(discharge_series)
    print(f"BFI = {bf.bfi:.2f}")

    rec = recession_analysis(discharge_series)
    print(f"Recession constant = {rec.recession_constant:.1f} days")

    ffa = fit_gev(discharge_series)
    print(f"100-year flood = {ffa.return_periods[100]:.1f} m³/s")
"""

from __future__ import annotations

from aquascope.hydrology.baseflow import BaseflowResult, eckhardt, lyne_hollick
from aquascope.hydrology.flood_frequency import (
    EMAResult,
    FloodFreqResult,
    GoodnessOfFitResult,
    NonStationaryGEVResult,
    RegionalResult,
    anderson_darling_test,
    coverage_probability,
    cramer_von_mises_test,
    expected_moments_algorithm,
    fit_gev,
    fit_gev_lmoments,
    fit_gpd,
    fit_gumbel,
    fit_lp3,
    fit_nonstationary_gev,
    fit_weibull_min,
    grubbs_beck_test,
    leave_one_out_cv,
    lmoments_from_sample,
    probability_plot_correlation,
    regional_frequency_analysis,
    select_pot_threshold,
    weighted_skew,
)
from aquascope.hydrology.flow_duration import FDCResult, flow_duration_curve, low_flow_stat
from aquascope.hydrology.rating_curve import (
    RatingCurveResult,
    RatingSegment,
    cross_validate_rating,
    detect_rating_shift,
    export_hec_ras,
    fit_rating_curve,
    fit_segmented_rating_curve,
    predict_discharge,
    predict_stage,
    rating_curve_uncertainty,
)
from aquascope.hydrology.recession import (
    RecessionResult,
    RecessionSegment,
    fit_master_recession,
    identify_recessions,
    recession_analysis,
)
from aquascope.hydrology.signatures import (
    SignatureReport,
    baseflow_index_simple,
    compare_signatures,
    compute_signatures,
    flashiness_index,
    flow_elasticity,
    recession_constant,
    seasonality_index,
    similarity_score,
)

__all__ = [
    # flow duration
    "FDCResult",
    "flow_duration_curve",
    "low_flow_stat",
    # baseflow
    "BaseflowResult",
    "lyne_hollick",
    "eckhardt",
    # recession
    "RecessionResult",
    "RecessionSegment",
    "identify_recessions",
    "fit_master_recession",
    "recession_analysis",
    # flood frequency — original
    "FloodFreqResult",
    "fit_gev",
    "fit_lp3",
    # flood frequency — Bulletin 17C / EMA
    "EMAResult",
    "expected_moments_algorithm",
    "grubbs_beck_test",
    "weighted_skew",
    # flood frequency — new distributions
    "fit_gumbel",
    "fit_weibull_min",
    "fit_gpd",
    "select_pot_threshold",
    # flood frequency — L-moments
    "lmoments_from_sample",
    "fit_gev_lmoments",
    # flood frequency — non-stationary
    "NonStationaryGEVResult",
    "fit_nonstationary_gev",
    # flood frequency — regional
    "RegionalResult",
    "regional_frequency_analysis",
    # flood frequency — cross-validation
    "leave_one_out_cv",
    "coverage_probability",
    # flood frequency — goodness-of-fit
    "GoodnessOfFitResult",
    "anderson_darling_test",
    "cramer_von_mises_test",
    "probability_plot_correlation",
    # rating curve
    "RatingCurveResult",
    "RatingSegment",
    "fit_rating_curve",
    "fit_segmented_rating_curve",
    "predict_discharge",
    "predict_stage",
    "rating_curve_uncertainty",
    "detect_rating_shift",
    "export_hec_ras",
    "cross_validate_rating",
    # signatures
    "SignatureReport",
    "compute_signatures",
    "flashiness_index",
    "seasonality_index",
    "flow_elasticity",
    "baseflow_index_simple",
    "recession_constant",
    "compare_signatures",
    "similarity_score",
]
