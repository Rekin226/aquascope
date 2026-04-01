"""Automated report generation for AquaScope analysis results.

Provides :class:`ReportBuilder` for assembling Markdown and HTML reports
from DataFrames, matplotlib figures, metrics, and summaries.
"""

from __future__ import annotations

from aquascope.reporting.builder import ReportBuilder, ReportMetadata, ReportSection
from aquascope.reporting.templates import ACADEMIC_CSS, DEFAULT_CSS, get_css, html_template

__all__ = [
    "ACADEMIC_CSS",
    "DEFAULT_CSS",
    "ReportBuilder",
    "ReportMetadata",
    "ReportSection",
    "get_css",
    "html_template",
]
