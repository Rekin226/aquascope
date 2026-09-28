"""The Markdown and HTML reports of a study, through :class:`aquascope.reporting.builder.ReportBuilder`.

Figures are referenced by their bundle path (``figures/...``) in Markdown and
embedded as data URIs in HTML, so ``report.html`` stands alone; tables come
from the CSV artifacts, capped at 50 rows. Nothing is written to disk: the
builder is subclassed to read the images from the workspace's bytes.
"""

from __future__ import annotations

import base64
import re
from typing import Any

from aquascope.reporting.builder import ReportBuilder
from aquascope.studio.deliverables import _common as c
from aquascope.studio.deliverables.tables import frame_of
from aquascope.studio.workspace import Workspace

MAX_TABLE_ROWS = 50


class _StudioReport(ReportBuilder):
    """A ReportBuilder whose figures live in memory (the artifact bytes by bundle path)."""

    def __init__(self, title: str, images: dict[str, bytes], **kw: Any) -> None:
        super().__init__(title, **kw)
        self._images = images

    def _read_image_base64(self, path: str) -> tuple[str, str]:  # type: ignore[override]
        data = self._images.get(path)
        if data is None:
            raise FileNotFoundError(path)
        mime = "image/svg+xml" if path.lower().endswith(".svg") else "image/png"
        return mime, base64.b64encode(data).decode("ascii")


def _blocks(text: str) -> list[str]:
    """Markdown text split on blank lines, so each block is one paragraph element."""
    return [b.strip() for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def _key_numbers_frame(ws: Workspace) -> Any:
    import pandas as pd

    rows = [{"Quantity": k.get("label"), "Value": k.get("value"), "Unit": k.get("unit") or "",
             "Step": k.get("step") or ""} for k in c.key_numbers(ws)]
    return pd.DataFrame(rows, columns=["Quantity", "Value", "Unit", "Step"])


def builder_for(ws: Workspace) -> ReportBuilder:
    """The report assembled from ``ws.report`` and the artifacts, ready to render either way."""
    images = {a.name: a.data for a in ws.figures()}
    rb = _StudioReport(c.title_of(ws), images, author="AquaScope Studio",
                       description=ws.brief.decision or ws.brief.problem or "")
    rb.metadata.date = c.date_of(ws) or rb.metadata.date
    rb.metadata.doi = c.CONCEPT_DOI
    sources = sorted({str(d.source) for d in (ws.inventory.datasets if ws.inventory else []) if d.source})
    rb.metadata.data_sources = sources
    site = c.site_text(ws)
    if site:
        rb.add_paragraph(f"**Site:** {site}")

    answer = c.answer_of(ws)
    if answer:
        rb.add_paragraph(f"**Answer.** {answer}")
    keys = c.key_numbers(ws)
    if keys:
        rb.add_dataframe(_key_numbers_frame(ws), caption="Key numbers")

    for block in c.blocks(ws):
        if block["kind"] == "list":
            rb.add_heading(block["title"], level=2)
            if block["key"] == "references":
                rb.add_paragraph("\n".join(f"{i}. {r}" for i, r in enumerate(block["items"], 1)))
            else:
                rb.add_paragraph(_bullets(block["items"]))
            continue
        rb.add_heading(block["title"] or block["id"], level=2)
        for para in _blocks(block["text"]):
            rb.add_paragraph(para)
        for fid in block["figures"]:
            fig = c.png_figure(ws, fid)
            if fig is not None:
                rb.add_figure(fig.name, caption=fig.caption or "")
        for tid in block["tables"]:
            tab, pointer = c.prose_table(ws, tid)
            if pointer:
                rb.add_paragraph(f"*{pointer}*")
            if tab is None:
                continue
            try:
                df = frame_of(tab)
                df = df.astype(object).where(df.notna(), "")
                rb.add_dataframe(df, caption=tab.caption or tab.id, max_rows=MAX_TABLE_ROWS)
            except Exception:  # noqa: BLE001 - a table that will not parse is named, not fatal
                rb.add_paragraph(f"*Table {tab.id} could not be rendered.*")
    rb.add_heading("Cite this software", level=2)
    rb.add_paragraph(c.citation())
    footer = c.footer_of(ws)
    if footer:
        rb.add_separator()
        rb.add_paragraph(f"*{footer}*")
    return rb


def report_markdown(ws: Workspace) -> str:
    """``report.md``: figures as relative ``figures/...`` paths, tables as Markdown tables (50 rows at most)."""
    return builder_for(ws)._render_markdown()


def report_html(ws: Workspace, *, style: str = "default") -> str:
    """``report.html``: self-contained, the figures embedded as data URIs."""
    return builder_for(ws)._render_html(style)
