"""CSS and HTML templates for report generation.

Provides pre-defined styles and the HTML wrapper function used by
:class:`aquascope.reporting.builder.ReportBuilder` when exporting to HTML.
"""

from __future__ import annotations

import html as _html

DEFAULT_CSS = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    color: #333;
    line-height: 1.6;
}
h1 {
    color: #1a5276;
    border-bottom: 2px solid #2980b9;
    padding-bottom: 10px;
}
h2 {
    color: #2471a3;
    margin-top: 30px;
}
h3 {
    color: #2e86c1;
    margin-top: 20px;
}
h4 {
    color: #3498db;
    margin-top: 15px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
}
th {
    background-color: #2980b9;
    color: white;
    padding: 10px;
    text-align: left;
}
td {
    padding: 8px;
    border-bottom: 1px solid #ddd;
}
tr:hover {
    background-color: #f5f5f5;
}
.metric {
    display: inline-block;
    background: #eaf2f8;
    border-radius: 8px;
    padding: 15px;
    margin: 5px;
    min-width: 150px;
    text-align: center;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #2980b9;
}
.metric-name {
    font-size: 12px;
    color: #666;
}
.metric-unit {
    font-size: 12px;
    color: #999;
}
.metric-exceeded {
    color: #e74c3c;
}
.alert-critical {
    color: #e74c3c;
    font-weight: bold;
}
.alert-high {
    color: #e67e22;
    font-weight: bold;
}
.alert-warning {
    color: #f39c12;
}
.alert-medium {
    color: #f39c12;
}
.alert-low {
    color: #27ae60;
}
.figure-caption {
    text-align: center;
    font-style: italic;
    color: #666;
    margin-top: 5px;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 10px auto;
}
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 30px 0;
}
.toc {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 5px;
    padding: 15px 25px;
    margin: 20px 0;
}
.toc ul {
    list-style-type: none;
    padding-left: 15px;
}
.toc > ul {
    padding-left: 0;
}
.toc a {
    color: #2471a3;
    text-decoration: none;
}
.toc a:hover {
    text-decoration: underline;
}
.report-meta {
    color: #666;
    font-size: 14px;
    margin-bottom: 20px;
}
"""

ACADEMIC_CSS = """
body {
    font-family: 'Georgia', 'Times New Roman', Times, serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 30px;
    color: #222;
    line-height: 1.8;
    font-size: 16px;
}
h1 {
    color: #1a1a1a;
    border-bottom: 1px solid #333;
    padding-bottom: 8px;
    font-size: 28px;
}
h2 {
    color: #333;
    margin-top: 35px;
    font-size: 22px;
}
h3 {
    color: #444;
    margin-top: 25px;
    font-size: 18px;
}
h4 {
    color: #555;
    margin-top: 18px;
    font-size: 16px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 14px;
}
th {
    background-color: #555;
    color: white;
    padding: 10px;
    text-align: left;
    font-weight: bold;
}
td {
    padding: 8px;
    border-bottom: 1px solid #ccc;
}
tr:hover {
    background-color: #f0f0f0;
}
.metric {
    display: inline-block;
    background: #f5f5f5;
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 12px;
    margin: 5px;
    min-width: 140px;
    text-align: center;
}
.metric-value {
    font-size: 22px;
    font-weight: bold;
    color: #333;
}
.metric-name {
    font-size: 12px;
    color: #666;
}
.metric-unit {
    font-size: 12px;
    color: #999;
}
.metric-exceeded {
    color: #c0392b;
}
.alert-critical {
    color: #c0392b;
    font-weight: bold;
}
.alert-high {
    color: #d35400;
    font-weight: bold;
}
.alert-warning {
    color: #e67e22;
}
.alert-medium {
    color: #e67e22;
}
.alert-low {
    color: #27ae60;
}
.figure-caption {
    text-align: center;
    font-style: italic;
    color: #555;
    margin-top: 8px;
    font-size: 14px;
}
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 15px auto;
}
hr {
    border: none;
    border-top: 1px solid #aaa;
    margin: 35px 0;
}
.toc {
    background: #fafafa;
    border: 1px solid #ddd;
    padding: 15px 25px;
    margin: 20px 0;
}
.toc ul {
    list-style-type: none;
    padding-left: 15px;
}
.toc > ul {
    padding-left: 0;
}
.toc a {
    color: #333;
    text-decoration: none;
}
.toc a:hover {
    text-decoration: underline;
}
.report-meta {
    color: #555;
    font-size: 14px;
    margin-bottom: 25px;
    font-style: italic;
}
"""

_STYLES: dict[str, str] = {
    "default": DEFAULT_CSS,
    "academic": ACADEMIC_CSS,
}


def get_css(style: str = "default") -> str:
    """Return the CSS string for the requested style.

    Parameters:
        style: Style name — ``"default"`` or ``"academic"``.

    Returns:
        CSS text.

    Raises:
        ValueError: If *style* is not recognised.
    """
    if style not in _STYLES:
        raise ValueError(f"Unknown style {style!r}. Choose from: {sorted(_STYLES)}")
    return _STYLES[style]


def html_template(title: str, body: str, css: str) -> str:
    """Wrap *body* in a full HTML5 document with ``<head>`` and embedded *css*.

    Parameters:
        title: Page ``<title>`` text.
        body: Pre-rendered HTML body content.
        css: CSS text to embed in a ``<style>`` block.

    Returns:
        Complete HTML document string.
    """
    safe_title = _html.escape(title)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
        f"  <title>{safe_title}</title>\n"
        f"  <style>{css}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )
