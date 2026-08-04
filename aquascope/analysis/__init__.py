"""Data analysis, EDA, quality assessment, change-point detection, and copula modules."""

from aquascope.analysis.changepoint import (
    ChangePoint,
    ChangePointResult,
    binary_segmentation,
    cusum,
    mann_whitney_test,
    pelt,
    pettitt_test,
    plot_changepoints,
    regime_shift_detector,
)
from aquascope.analysis.copulas import (
    CopulaResult,
    JointProbability,
    compare_copulas,
    copula_density,
    copula_function,
    fit_copula,
    generate_copula_samples,
    generate_synthetic_data,
    joint_exceedance_probability,
    tail_dependence,
    to_pseudo_observations,
)
from aquascope.analysis.eda import (
    EDAReport,
    generate_eda_report,
    print_eda_report,
    profile_dataset,
)
from aquascope.analysis.quality import (
    QualityReport,
    assess_quality,
    preprocess,
    print_quality_report,
)
from aquascope.analysis.trends import (
    MannKendallResult,
    SensSlopeResult,
    mann_kendall,
    sens_slope,
)

__all__ = [
    "ChangePoint",
    "ChangePointResult",
    "CopulaResult",
    "EDAReport",
    "JointProbability",
    "MannKendallResult",
    "QualityReport",
    "SensSlopeResult",
    "assess_quality",
    "binary_segmentation",
    "compare_copulas",
    "copula_density",
    "copula_function",
    "cusum",
    "fit_copula",
    "generate_copula_samples",
    "generate_eda_report",
    "generate_synthetic_data",
    "joint_exceedance_probability",
    "mann_kendall",
    "mann_whitney_test",
    "pelt",
    "pettitt_test",
    "plot_changepoints",
    "preprocess",
    "print_eda_report",
    "print_quality_report",
    "profile_dataset",
    "regime_shift_detector",
    "sens_slope",
    "tail_dependence",
    "to_pseudo_observations",
]
