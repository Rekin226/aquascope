"""The documents a study becomes: a technical report, a technical memorandum, or a study note.

:func:`build_report` and :func:`build_memo` compose a :class:`Document` from a
workspace; :func:`render_docx`, :func:`render_html` and :func:`render_markdown`
render it. :class:`HouseStyle` carries the organisation, the people and the
look. Nothing here needs a language model: the text is written from the
results, and a model's prose, when there is some, is quoted only where it
interprets.
"""

from __future__ import annotations

from aquascope.studio.document.compose import (
    build_memo,
    build_note,
    build_report,
    is_failed_study,
    terminal_summary,
)
from aquascope.studio.document.facts import Facts, facts_of
from aquascope.studio.document.model import Document
from aquascope.studio.document.render_html import render_html
from aquascope.studio.document.render_md import render_markdown
from aquascope.studio.document.style import HouseStyle, load_style

__all__ = ["Document", "Facts", "HouseStyle", "build_memo", "build_note", "build_report", "facts_of",
           "is_failed_study", "load_style", "render_docx", "render_html", "render_markdown", "terminal_summary"]


def render_docx(doc: Document, style: HouseStyle | None = None) -> bytes | None:
    """The document as Word bytes, or None when python-docx is not installed."""
    from aquascope.studio.document.word import render_docx as _render

    return _render(doc, style)
