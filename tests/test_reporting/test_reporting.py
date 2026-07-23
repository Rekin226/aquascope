"""Tests for aquascope.reporting module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from aquascope.reporting.builder import ReportBuilder, ReportMetadata
from aquascope.reporting.templates import DEFAULT_CSS, get_css, html_template

# ---------------------------------------------------------------------------
# Mock objects for AlertReport / EDAReport integration
# ---------------------------------------------------------------------------

@dataclass
class _MockAlert:
    sample_id: str = "S1"
    parameter: str = "pH"
    value: float = 9.5
    limit: float = 8.5
    severity: str = "HIGH"
    standard: str = "WHO"
    description: str = "pH exceeds limit"


@dataclass
class _MockAlertReport:
    alerts: list[_MockAlert] = field(
        default_factory=lambda: [_MockAlert(), _MockAlert(parameter="DO", severity="CRITICAL")],
    )
    total_samples: int = 100
    samples_with_alerts: int = 5
    parameters_checked: list[str] = field(default_factory=lambda: ["pH", "DO", "BOD5"])
    standards_used: list[str] = field(default_factory=lambda: ["WHO", "EPA"])
    summary: dict[str, int] = field(default_factory=lambda: {"CRITICAL": 1, "HIGH": 1})


@dataclass
class _MockParameterStats:
    name: str = "pH"
    count: int = 500
    missing: int = 10
    mean: float = 7.2
    std: float = 0.5
    min: float = 6.0
    q25: float = 6.9
    median: float = 7.2
    q75: float = 7.5
    max: float = 9.0
    outlier_count: int = 3


@dataclass
class _MockEDAReport:
    n_records: int = 5000
    n_stations: int = 12
    n_parameters: int = 8
    date_range: tuple[str, str] | None = ("2020-01-01", "2023-12-31")
    time_span_years: float = 4.0
    parameters: list[_MockParameterStats] = field(default_factory=lambda: [_MockParameterStats()])
    sources: list[str] = field(default_factory=lambda: ["taiwan_moenv"])
    completeness_pct: float = 95.2
    correlation_matrix: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({"Station": ["A", "B", "C"], "pH": [7.1, 6.8, 7.5], "DO": [8.2, 7.9, 8.5]})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReportBuilder:
    """Tests for the ReportBuilder class."""

    def test_builder_init(self) -> None:
        """Metadata is set correctly on initialisation."""
        rb = ReportBuilder("My Report", author="Alice", description="Test desc")
        assert rb.metadata.title == "My Report"
        assert rb.metadata.author == "Alice"
        assert rb.metadata.description == "Test desc"
        assert rb.metadata.version == "1.0"
        assert rb.metadata.date  # non-empty

    def test_add_heading(self) -> None:
        """Heading appears in rendered Markdown."""
        md = ReportBuilder("T").add_heading("Overview", level=2)._render_markdown()
        assert "## Overview" in md

    def test_add_paragraph(self) -> None:
        """Paragraph text appears in rendered Markdown."""
        md = ReportBuilder("T").add_paragraph("Hello world")._render_markdown()
        assert "Hello world" in md

    def test_add_dataframe(self) -> None:
        """DataFrame is rendered as a pipe-separated Markdown table."""
        df = _sample_df()
        md = ReportBuilder("T").add_dataframe(df, caption="Stations")._render_markdown()
        assert "| Station |" in md or "Station |" in md
        assert "|" in md
        assert "---" in md

    def test_add_figure_from_matplotlib(self) -> None:
        """Matplotlib figure is saved and referenced in Markdown."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        rb = ReportBuilder("T")
        rb.add_figure(fig, caption="Test Plot")
        md = rb._render_markdown()
        assert "Test Plot" in md
        assert "![" in md

    def test_add_figure_from_path(self) -> None:
        """String path is treated as an existing image reference."""
        rb = ReportBuilder("T")
        rb.add_figure("/images/plot.png", caption="Existing")
        md = rb._render_markdown()
        assert "/images/plot.png" in md
        assert "Existing" in md

    def test_add_metric(self) -> None:
        """Metric is formatted correctly in Markdown."""
        md = ReportBuilder("T").add_metric("NSE", 0.87)._render_markdown()
        assert "NSE" in md
        assert "0.87" in md

    def test_add_metric_exceeded(self) -> None:
        """Metric with exceeded threshold is flagged."""
        md = ReportBuilder("T").add_metric("Turbidity", 15.0, unit="NTU", threshold=10.0)._render_markdown()
        assert "EXCEEDED" in md

    def test_add_metrics_table(self) -> None:
        """Multiple metrics appear in a table."""
        metrics = {"NSE": 0.87, "KGE": 0.82, "RMSE": 1.23}
        md = ReportBuilder("T").add_metrics_table(metrics)._render_markdown()
        assert "NSE" in md
        assert "KGE" in md
        assert "RMSE" in md
        assert "0.87" in md

    def test_add_separator(self) -> None:
        """Horizontal rule appears in Markdown."""
        md = ReportBuilder("T").add_separator()._render_markdown()
        assert "---" in md

    def test_table_of_contents(self) -> None:
        """TOC contains links to headings."""
        rb = ReportBuilder("T")
        rb.add_heading("Introduction", level=2)
        rb.add_heading("Methods", level=2)
        rb.add_table_of_contents()
        md = rb._render_markdown()
        assert "Table of Contents" in md
        assert "[Introduction]" in md
        assert "[Methods]" in md

    def test_to_markdown_creates_file(self, tmp_path: Path) -> None:
        """Markdown export creates a file with content."""
        rb = ReportBuilder("File Test")
        rb.add_heading("Section 1")
        rb.add_paragraph("Content here.")
        out = rb.to_markdown(tmp_path / "report.md")
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "File Test" in text
        assert "Section 1" in text

    def test_to_html_creates_file(self, tmp_path: Path) -> None:
        """HTML export creates a file with valid HTML structure."""
        rb = ReportBuilder("HTML Test")
        rb.add_heading("Data")
        rb.add_paragraph("Some text.")
        out = rb.to_html(tmp_path / "report.html")
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in text
        assert "<html" in text
        assert "HTML Test" in text

    def test_to_html_embeds_images(self, tmp_path: Path) -> None:
        """Base64 image data appears in HTML img src."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])

        rb = ReportBuilder("Embed Test")
        rb.add_figure(fig, caption="Plot")
        out = rb.to_html(tmp_path / "report.html")
        text = out.read_text(encoding="utf-8")
        assert "data:image/png;base64," in text

    def test_to_html_default_style(self, tmp_path: Path) -> None:
        """Default style CSS is embedded in HTML output."""
        rb = ReportBuilder("Style Test")
        rb.add_paragraph("Text")
        out = rb.to_html(tmp_path / "report.html", style="default")
        text = out.read_text(encoding="utf-8")
        assert "<style>" in text
        assert "Segoe UI" in text

    def test_to_html_academic_style(self, tmp_path: Path) -> None:
        """Academic style CSS is embedded in HTML output."""
        rb = ReportBuilder("Academic Test")
        rb.add_paragraph("Text")
        out = rb.to_html(tmp_path / "report.html", style="academic")
        text = out.read_text(encoding="utf-8")
        assert "<style>" in text
        assert "Georgia" in text

    def test_chaining(self) -> None:
        """Fluent API: methods return self for chaining."""
        rb = ReportBuilder("Chain")
        result = rb.add_heading("H").add_paragraph("P").add_separator().add_metric("M", 1.0)
        assert result is rb

    def test_empty_report(self) -> None:
        """Empty report (no sections) still produces title and metadata."""
        md = ReportBuilder("Empty", author="Bot")._render_markdown()
        assert "# Empty" in md
        assert "Bot" in md

    def test_add_alert_summary(self) -> None:
        """Alert summary integrates mock AlertReport data."""
        alert_report = _MockAlertReport()
        rb = ReportBuilder("Alerts")
        rb.add_alert_summary(alert_report)
        md = rb._render_markdown()
        assert "Alert Summary" in md
        assert "Total alerts" in md
        assert "WHO" in md
        assert "CRITICAL" in md
        assert "pH" in md

    def test_add_eda_summary(self) -> None:
        """EDA summary integrates mock EDAReport data."""
        eda_report = _MockEDAReport()
        rb = ReportBuilder("EDA")
        rb.add_eda_summary(eda_report)
        md = rb._render_markdown()
        assert "EDA Summary" in md
        assert "12" in md  # n_stations
        assert "5000" in md  # n_records
        assert "95.2" in md  # completeness
        assert "taiwan_moenv" in md


class TestTemplates:
    """Tests for the templates module."""

    def test_default_css_content(self) -> None:
        """DEFAULT_CSS contains expected selectors."""
        assert "Segoe UI" in DEFAULT_CSS
        assert ".metric" in DEFAULT_CSS

    def test_get_css_default(self) -> None:
        """get_css returns default CSS."""
        css = get_css("default")
        assert "Segoe UI" in css

    def test_get_css_academic(self) -> None:
        """get_css returns academic CSS."""
        css = get_css("academic")
        assert "Georgia" in css

    def test_get_css_invalid(self) -> None:
        """get_css raises ValueError for unknown style."""
        try:
            get_css("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_html_template(self) -> None:
        """html_template wraps body in a full HTML document."""
        result = html_template("Title", "<p>Body</p>", "body { color: red; }")
        assert "<!DOCTYPE html>" in result
        assert "<title>Title</title>" in result
        assert "<p>Body</p>" in result
        assert "body { color: red; }" in result

    def test_html_template_escapes_title(self) -> None:
        """html_template escapes HTML entities in the title."""
        result = html_template("<script>alert(1)</script>", "<p>ok</p>", "")
        assert "<script>" not in result.split("<title>")[1].split("</title>")[0]
        assert "&lt;script&gt;" in result


class TestReportMetadata:
    """Tests for the ReportMetadata dataclass."""

    def test_defaults(self) -> None:
        """ReportMetadata has sensible defaults."""
        meta = ReportMetadata(title="Test")
        assert meta.title == "Test"
        assert meta.author == "AquaScope"
        assert meta.version == "1.0"
        assert re.match(r"\d{4}-\d{2}-\d{2}", meta.date)
