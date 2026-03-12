"""Data analysis, EDA, and quality assessment modules."""

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

__all__ = [
    "EDAReport",
    "QualityReport",
    "assess_quality",
    "generate_eda_report",
    "preprocess",
    "print_eda_report",
    "print_quality_report",
    "profile_dataset",
]
