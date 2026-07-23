"""Report builder for structured analysis reports.

Provides :class:`ReportBuilder` which assembles Markdown and HTML reports
from headings, paragraphs, DataFrames, matplotlib figures, metrics, and
summaries produced by other AquaScope modules.
"""

from __future__ import annotations

import base64
import html as _html
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from aquascope.reporting.templates import get_css, html_template

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReportSection:
    """A section in the report.

    Parameters:
        title: Section heading text.
        content: Markdown-formatted body content.
        level: Heading level (1–4).
        figures: ``(path, caption)`` pairs for embedded images.
        tables: ``(caption, dataframe)`` pairs for tabular data.
    """

    title: str
    content: str
    level: int = 2
    figures: list[tuple[str, str]] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)


@dataclass
class ReportMetadata:
    """Report metadata.

    Parameters:
        title: Report title.
        author: Author name.
        date: Report date string (defaults to today).
        description: Short description of the report.
        data_sources: Names of data sources used.
        version: Report version identifier.
    """

    title: str
    author: str = "AquaScope"
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    description: str = ""
    data_sources: list[str] = field(default_factory=list)
    version: str = "1.0"


# ---------------------------------------------------------------------------
# Internal element types
# ---------------------------------------------------------------------------

@dataclass
class _Heading:
    text: str
    level: int


@dataclass
class _Paragraph:
    text: str


@dataclass
class _Table:
    markdown: str
    caption: str


@dataclass
class _Figure:
    path: str
    caption: str


@dataclass
class _Metric:
    name: str
    value: float
    unit: str
    exceeded: bool


@dataclass
class _Separator:
    pass


@dataclass
class _TOC:
    pass


@dataclass
class _RawMarkdown:
    text: str


# Union of all element types kept in the internal list
_Element = _Heading | _Paragraph | _Table | _Figure | _Metric | _Separator | _TOC | _RawMarkdown


# ---------------------------------------------------------------------------
# ReportBuilder
# ---------------------------------------------------------------------------

class ReportBuilder:
    """Build structured analysis reports in Markdown and HTML.

    Usage::

        report = ReportBuilder("Water Quality Analysis", author="Dr. Smith")
        report.add_heading("Data Overview", level=2)
        report.add_paragraph("This analysis covers 5 stations...")
        report.add_dataframe(df, caption="Station Summary")
        report.add_figure(fig, caption="Timeseries Plot")
        report.add_metric("NSE", 0.87)
        report.add_alert_summary(alert_report)
        report.add_eda_summary(eda_report)

        report.to_markdown("report.md")
        report.to_html("report.html")
    """

    def __init__(self, title: str, author: str = "AquaScope", description: str = "") -> None:
        """Initialise report with metadata.

        Parameters:
            title: Report title.
            author: Author name.
            description: Short description included in the report header.
        """
        self.metadata = ReportMetadata(title=title, author=author, description=description)
        self._elements: list[_Element] = []
        self._figure_dir: str | None = None
        self._figure_counter: int = 0

    # -- internal helpers ---------------------------------------------------

    def _get_figure_dir(self) -> str:
        """Return (and lazily create) a temporary directory for figure files."""
        if self._figure_dir is None:
            self._figure_dir = tempfile.mkdtemp(prefix="aquascope_report_")
        return self._figure_dir

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert heading text to a URL-safe anchor slug."""
        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s]+", "-", slug)
        return slug

    @staticmethod
    def _df_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
        """Render a DataFrame as a Markdown pipe-table."""
        if len(df) > max_rows:
            df = df.head(max_rows)

        headers = " | ".join(str(c) for c in df.columns)
        separators = " | ".join("---" for _ in df.columns)
        rows: list[str] = []
        for _, row in df.iterrows():
            rows.append(" | ".join(str(v) for v in row))

        return f"| {headers} |\n| {separators} |\n" + "\n".join(f"| {r} |" for r in rows)

    @staticmethod
    def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
        """Render a DataFrame as an HTML ``<table>``."""
        if len(df) > max_rows:
            df = df.head(max_rows)

        parts: list[str] = ["<table>", "<thead><tr>"]
        for col in df.columns:
            parts.append(f"  <th>{_html.escape(str(col))}</th>")
        parts.append("</tr></thead>")
        parts.append("<tbody>")
        for _, row in df.iterrows():
            parts.append("<tr>")
            for v in row:
                parts.append(f"  <td>{_html.escape(str(v))}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
        return "\n".join(parts)

    # -- public API ---------------------------------------------------------

    def add_heading(self, text: str, level: int = 2) -> ReportBuilder:
        """Add a heading.

        Parameters:
            text: Heading text.
            level: Heading level (1–4).

        Returns:
            Self, for method chaining.
        """
        self._elements.append(_Heading(text=text, level=level))
        return self

    def add_paragraph(self, text: str) -> ReportBuilder:
        """Add a paragraph of text.

        Parameters:
            text: Paragraph content (Markdown allowed).

        Returns:
            Self, for method chaining.
        """
        self._elements.append(_Paragraph(text=text))
        return self

    def add_dataframe(self, df: pd.DataFrame, caption: str = "", max_rows: int = 50) -> ReportBuilder:
        """Add a DataFrame as a formatted table.

        Parameters:
            df: The DataFrame to render.
            caption: Optional table caption.
            max_rows: Maximum rows to display (default 50).

        Returns:
            Self, for method chaining.
        """
        md = self._df_to_markdown(df, max_rows=max_rows)
        self._elements.append(_Table(markdown=md, caption=caption))
        return self

    def add_figure(self, fig: object, caption: str = "", filename: str | None = None) -> ReportBuilder:
        """Add a matplotlib figure or an existing image path.

        If *fig* is a ``str`` or :class:`~pathlib.Path` it is treated as a
        path to an existing image file.  Otherwise it must be a matplotlib
        ``Figure`` and will be saved to a temporary PNG file.

        Parameters:
            fig: A matplotlib ``Figure`` object, or a path to an image.
            caption: Optional figure caption.
            filename: Override filename when saving a matplotlib figure.

        Returns:
            Self, for method chaining.
        """
        if isinstance(fig, (str, Path)):
            path = str(fig)
        else:
            # Lazy-import matplotlib to avoid hard dependency
            import matplotlib.pyplot as plt  # noqa: F811

            self._figure_counter += 1
            if filename is None:
                filename = f"figure_{self._figure_counter}.png"
            path = str(Path(self._get_figure_dir()) / filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        self._elements.append(_Figure(path=path, caption=caption))
        return self

    def add_metric(self, name: str, value: float, unit: str = "", threshold: float | None = None) -> ReportBuilder:
        """Add a key metric.

        Parameters:
            name: Metric name (e.g. ``"NSE"``).
            value: Metric value.
            unit: Optional unit string.
            threshold: If provided the metric is flagged when *value* exceeds it.

        Returns:
            Self, for method chaining.
        """
        exceeded = threshold is not None and value > threshold
        self._elements.append(_Metric(name=name, value=value, unit=unit, exceeded=exceeded))
        return self

    def add_metrics_table(self, metrics: dict[str, float], title: str = "Performance Metrics") -> ReportBuilder:
        """Add a formatted metrics table.

        Parameters:
            metrics: Mapping of metric names to values (e.g. ``{"NSE": 0.87, "KGE": 0.82}``).
            title: Table heading.

        Returns:
            Self, for method chaining.
        """
        df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        self.add_heading(title, level=3)
        self.add_dataframe(df, caption=title)
        return self

    def add_alert_summary(self, alert_report: object) -> ReportBuilder:
        """Add summary from an :class:`~aquascope.alerts.checker.AlertReport`.

        Shows total alerts, breakdown by severity, and top violated parameters.

        Parameters:
            alert_report: An ``AlertReport`` instance.

        Returns:
            Self, for method chaining.
        """
        self.add_heading("Alert Summary", level=2)

        total = len(getattr(alert_report, "alerts", []))
        samples_with = getattr(alert_report, "samples_with_alerts", 0)
        total_samples = getattr(alert_report, "total_samples", 0)
        summary: dict[str, int] = getattr(alert_report, "summary", {})
        standards: list[str] = getattr(alert_report, "standards_used", [])

        lines = [
            f"- **Total alerts:** {total}",
            f"- **Samples with alerts:** {samples_with} / {total_samples}",
            f"- **Standards used:** {', '.join(standards) if standards else 'N/A'}",
        ]
        self.add_paragraph("\n".join(lines))

        if summary:
            self.add_heading("Alerts by Severity", level=3)
            df = pd.DataFrame(list(summary.items()), columns=["Severity", "Count"])
            self.add_dataframe(df, caption="Alerts by Severity")

        # Top violated parameters
        alerts = getattr(alert_report, "alerts", [])
        if alerts:
            param_counts: dict[str, int] = {}
            for alert in alerts:
                param = getattr(alert, "parameter", "unknown")
                param_counts[param] = param_counts.get(param, 0) + 1
            top_params = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            self.add_heading("Top Violated Parameters", level=3)
            df = pd.DataFrame(top_params, columns=["Parameter", "Alert Count"])
            self.add_dataframe(df, caption="Top Violated Parameters")

        return self

    def add_eda_summary(self, eda_report: object) -> ReportBuilder:
        """Add summary from an :class:`~aquascope.analysis.eda.EDAReport`.

        Shows station count, date range, completeness, and parameter stats.

        Parameters:
            eda_report: An ``EDAReport`` instance.

        Returns:
            Self, for method chaining.
        """
        self.add_heading("EDA Summary", level=2)

        n_stations = getattr(eda_report, "n_stations", 0)
        n_records = getattr(eda_report, "n_records", 0)
        n_parameters = getattr(eda_report, "n_parameters", 0)
        date_range = getattr(eda_report, "date_range", None)
        completeness = getattr(eda_report, "completeness_pct", 0.0)
        sources: list[str] = getattr(eda_report, "sources", [])

        date_str = f"{date_range[0]} to {date_range[1]}" if date_range else "N/A"
        lines = [
            f"- **Stations:** {n_stations}",
            f"- **Records:** {n_records}",
            f"- **Parameters:** {n_parameters}",
            f"- **Date range:** {date_str}",
            f"- **Completeness:** {completeness:.1f}%",
            f"- **Sources:** {', '.join(sources) if sources else 'N/A'}",
        ]
        self.add_paragraph("\n".join(lines))

        # Per-parameter stats
        parameters = getattr(eda_report, "parameters", [])
        if parameters:
            rows: list[dict[str, object]] = []
            for p in parameters:
                rows.append({
                    "Parameter": getattr(p, "name", ""),
                    "Count": getattr(p, "count", 0),
                    "Missing": getattr(p, "missing", 0),
                    "Mean": f"{getattr(p, 'mean', 0):.4f}",
                    "Std": f"{getattr(p, 'std', 0):.4f}",
                    "Min": f"{getattr(p, 'min', 0):.4f}",
                    "Max": f"{getattr(p, 'max', 0):.4f}",
                })
            df = pd.DataFrame(rows)
            self.add_heading("Parameter Statistics", level=3)
            self.add_dataframe(df, caption="Parameter Statistics")

        return self

    def add_separator(self) -> ReportBuilder:
        """Add a horizontal rule separator.

        Returns:
            Self, for method chaining.
        """
        self._elements.append(_Separator())
        return self

    def add_table_of_contents(self) -> ReportBuilder:
        """Insert a table-of-contents placeholder.

        The TOC is generated from headings at render time.

        Returns:
            Self, for method chaining.
        """
        self._elements.append(_TOC())
        return self

    # -- rendering ----------------------------------------------------------

    def _collect_headings(self) -> list[_Heading]:
        """Return all heading elements in order."""
        return [e for e in self._elements if isinstance(e, _Heading)]

    def _render_toc_markdown(self) -> str:
        """Render a Markdown table of contents from headings."""
        headings = self._collect_headings()
        if not headings:
            return ""
        lines = ["## Table of Contents\n"]
        for h in headings:
            indent = "  " * (h.level - 2) if h.level > 1 else ""
            slug = self._slugify(h.text)
            lines.append(f"{indent}- [{h.text}](#{slug})")
        return "\n".join(lines)

    def _render_toc_html(self) -> str:
        """Render an HTML table of contents from headings."""
        headings = self._collect_headings()
        if not headings:
            return ""
        parts = ['<div class="toc">', "<h2>Table of Contents</h2>", "<ul>"]
        for h in headings:
            slug = self._slugify(h.text)
            safe = _html.escape(h.text)
            indent_level = max(0, h.level - 2)
            prefix = "  " * indent_level
            parts.append(f'{prefix}<li><a href="#{slug}">{safe}</a></li>')
        parts.append("</ul></div>")
        return "\n".join(parts)

    def _render_markdown(self) -> str:
        """Render all sections to a single Markdown string."""
        parts: list[str] = []

        # Title + metadata header
        parts.append(f"# {self.metadata.title}\n")
        meta_lines = [f"**Author:** {self.metadata.author}  ", f"**Date:** {self.metadata.date}  "]
        if self.metadata.description:
            meta_lines.append(f"**Description:** {self.metadata.description}  ")
        if self.metadata.data_sources:
            meta_lines.append(f"**Data Sources:** {', '.join(self.metadata.data_sources)}  ")
        meta_lines.append(f"**Version:** {self.metadata.version}  ")
        parts.append("\n".join(meta_lines))
        parts.append("")

        for elem in self._elements:
            if isinstance(elem, _TOC):
                parts.append(self._render_toc_markdown())
                parts.append("")
            elif isinstance(elem, _Heading):
                prefix = "#" * elem.level
                parts.append(f"{prefix} {elem.text}\n")
            elif isinstance(elem, _Paragraph):
                parts.append(f"{elem.text}\n")
            elif isinstance(elem, _Table):
                if elem.caption:
                    parts.append(f"*{elem.caption}*\n")
                parts.append(elem.markdown)
                parts.append("")
            elif isinstance(elem, _Figure):
                parts.append(f"![{elem.caption}]({elem.path})")
                if elem.caption:
                    parts.append(f"*{elem.caption}*\n")
                else:
                    parts.append("")
            elif isinstance(elem, _Metric):
                exceeded_marker = " ⚠️ EXCEEDED" if elem.exceeded else ""
                unit_str = f" {elem.unit}" if elem.unit else ""
                parts.append(f"- **{elem.name}:** {elem.value}{unit_str}{exceeded_marker}")
            elif isinstance(elem, _Separator):
                parts.append("\n---\n")
            elif isinstance(elem, _RawMarkdown):
                parts.append(elem.text)
                parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _read_image_base64(path: str) -> tuple[str, str]:
        """Read an image and return ``(mime_type, base64_data)``."""
        p = Path(path)
        suffix = p.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml",
                    ".gif": "image/gif"}
        mime = mime_map.get(suffix, "image/png")
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return mime, data

    def _render_html(self, style: str) -> str:
        """Render to HTML with embedded CSS and base64 images."""
        css = get_css(style)
        parts: list[str] = []

        # Title + metadata
        parts.append(f"<h1>{_html.escape(self.metadata.title)}</h1>")
        meta_items = [
            f"<strong>Author:</strong> {_html.escape(self.metadata.author)}",
            f"<strong>Date:</strong> {_html.escape(self.metadata.date)}",
        ]
        if self.metadata.description:
            meta_items.append(f"<strong>Description:</strong> {_html.escape(self.metadata.description)}")
        if self.metadata.data_sources:
            sources_str = ", ".join(_html.escape(s) for s in self.metadata.data_sources)
            meta_items.append(f"<strong>Data Sources:</strong> {sources_str}")
        meta_items.append(f"<strong>Version:</strong> {_html.escape(self.metadata.version)}")
        meta_body = "<br>\n".join(meta_items)
        parts.append(f'<div class="report-meta">{meta_body}</div>')

        for elem in self._elements:
            if isinstance(elem, _TOC):
                parts.append(self._render_toc_html())
            elif isinstance(elem, _Heading):
                tag = f"h{elem.level}"
                slug = self._slugify(elem.text)
                parts.append(f'<{tag} id="{slug}">{_html.escape(elem.text)}</{tag}>')
            elif isinstance(elem, _Paragraph):
                # Convert markdown bold to HTML bold
                text = _html.escape(elem.text)
                text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
                text = text.replace("\n", "<br>\n")
                parts.append(f"<p>{text}</p>")
            elif isinstance(elem, _Table):
                # Re-parse the markdown table to produce HTML
                lines = elem.markdown.strip().split("\n")
                if len(lines) >= 2:
                    header_cells = [c.strip() for c in lines[0].strip("|").split("|")]
                    html_rows: list[str] = ["<table>", "<thead><tr>"]
                    for c in header_cells:
                        html_rows.append(f"  <th>{_html.escape(c)}</th>")
                    html_rows.append("</tr></thead><tbody>")
                    for row_line in lines[2:]:
                        cells = [c.strip() for c in row_line.strip("|").split("|")]
                        html_rows.append("<tr>")
                        for c in cells:
                            html_rows.append(f"  <td>{_html.escape(c)}</td>")
                        html_rows.append("</tr>")
                    html_rows.append("</tbody></table>")
                    if elem.caption:
                        parts.append(f'<p class="figure-caption">{_html.escape(elem.caption)}</p>')
                    parts.append("\n".join(html_rows))
            elif isinstance(elem, _Figure):
                try:
                    mime, data = self._read_image_base64(elem.path)
                    parts.append(f'<img src="data:{mime};base64,{data}" alt="{_html.escape(elem.caption)}">')
                except FileNotFoundError:
                    logger.warning("Figure not found: %s", elem.path)
                    parts.append(f"<p><em>[Figure not found: {_html.escape(elem.path)}]</em></p>")
                if elem.caption:
                    parts.append(f'<p class="figure-caption">{_html.escape(elem.caption)}</p>')
            elif isinstance(elem, _Metric):
                exceeded_cls = " metric-exceeded" if elem.exceeded else ""
                unit_html = f'<span class="metric-unit">{_html.escape(elem.unit)}</span>' if elem.unit else ""
                exceeded_html = ' <span class="metric-exceeded">⚠️</span>' if elem.exceeded else ""
                parts.append(
                    f'<div class="metric">'
                    f'<div class="metric-value{exceeded_cls}">{elem.value}{exceeded_html}</div>'
                    f'<div class="metric-name">{_html.escape(elem.name)}</div>'
                    f"{unit_html}</div>"
                )
            elif isinstance(elem, _Separator):
                parts.append("<hr>")
            elif isinstance(elem, _RawMarkdown):
                parts.append(f"<p>{_html.escape(elem.text)}</p>")

        body = "\n".join(parts)
        return html_template(self.metadata.title, body, css)

    # -- export -------------------------------------------------------------

    def to_markdown(self, path: str | Path) -> Path:
        """Export report as a Markdown file.

        Parameters:
            path: Destination file path.

        Returns:
            The resolved :class:`~pathlib.Path` of the written file.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = self._render_markdown()
        dest.write_text(content, encoding="utf-8")
        logger.info("Markdown report written to %s", dest)
        return dest

    def to_html(self, path: str | Path, style: str = "default") -> Path:
        """Export report as a styled HTML file.

        Figures are embedded as base64 data URIs so the HTML is self-contained.

        Parameters:
            path: Destination file path.
            style: CSS style — ``"default"`` (sans-serif, blue accents) or
                ``"academic"`` (serif, formal).

        Returns:
            The resolved :class:`~pathlib.Path` of the written file.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = self._render_html(style)
        dest.write_text(content, encoding="utf-8")
        logger.info("HTML report written to %s", dest)
        return dest
