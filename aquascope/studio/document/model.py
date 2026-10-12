"""The document a study becomes: a short list of typed blocks, rendered the same way to Word, HTML and Markdown.

The composer (:mod:`aquascope.studio.document.compose`) decides what the
reader sees and in which order; the renderers only decide how it looks. A
block is plain data, so a page, a test or a model can build and inspect a
document without any document library.

Cross-references are written in the text as ``{fig:frequency}`` or
``{tab:levels}`` and resolved at render time to "Figure 3" / "Table 2", in
the order the figures and tables appear, so the composer never counts.
Inline emphasis is ``**bold**`` and ``*italic*``; nothing else is markup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Bullets", "Callout", "Document", "Figure", "Heading", "KeyValue", "PageBreak", "Para", "Signoff",
           "Table"]


@dataclass
class Heading:
    text: str
    level: int = 1
    #: "1", "2.3": written by :meth:`Document.number_headings`, empty for an unnumbered heading.
    number: str = ""
    numbered: bool = True


@dataclass
class Para:
    text: str
    #: "lead" (larger, the opening of the summary), "small" (notes under a table), "" (body).
    style: str = ""


@dataclass
class Bullets:
    items: list[str]
    numbered: bool = False


@dataclass
class Figure:
    id: str
    png: bytes
    caption: str
    svg: bytes | None = None
    #: "full" (text width) or "half".
    width: str = "full"
    #: The file name the Markdown points at (``figures/s3_frequency_curve.png``).
    path: str = ""
    number: int = 0


@dataclass
class Table:
    id: str
    columns: list[str]
    rows: list[list[Any]]
    caption: str
    #: One alignment per column: "l", "r" or "c" (numbers right-aligned by default).
    align: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: A narrow table (a few columns of short text) is not stretched to the text width.
    compact: bool = False
    #: Row indexes set in bold (the design return period in a table of return levels).
    emphasis: list[int] = field(default_factory=list)
    #: False for a control table (revision history, review log): no "Table N." label, not counted.
    numbered: bool = True
    number: int = 0


@dataclass
class Callout:
    """A boxed statement: the answer, a warning. ``rows`` are (label, value) pairs under the title."""

    title: str
    body: str = ""
    rows: list[tuple[str, str]] = field(default_factory=list)
    #: "answer" (blue rule), "caution" (amber rule), "fail" (red rule).
    tone: str = "answer"


@dataclass
class KeyValue:
    """A two-column table without a caption: the document-control block, the memo header."""

    rows: list[tuple[str, str]]


@dataclass
class Signoff:
    """Prepared / checked / approved, with name, signature and date lines."""

    roles: list[tuple[str, str]]


@dataclass
class PageBreak:
    pass


Block = Heading | Para | Bullets | Figure | Table | Callout | KeyValue | Signoff | PageBreak

_REF = re.compile(r"\{(fig|tab):([A-Za-z0-9_.\-]+)\}")


@dataclass
class Document:
    """A titled list of blocks with the metadata a cover and a running header need."""

    title: str
    subtitle: str = ""
    #: "Technical report", "Technical memorandum", "Study note".
    kind: str = "Technical report"
    meta: dict[str, str] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)
    #: The status stamp ("DRAFT" until a person checks it, "" to omit).
    status: str = "DRAFT"

    def add(self, *blocks: Block | None) -> Document:
        for b in blocks:
            if b is not None:
                self.blocks.append(b)
        return self

    # ── numbering ───────────────────────────────────────────────────────────

    def finalize(self) -> Document:
        """Number the headings, the figures and the tables, and resolve every ``{fig:..}``/``{tab:..}``."""
        self.number_headings()
        figs: dict[str, int] = {}
        tabs: dict[str, int] = {}
        for b in self.blocks:
            if isinstance(b, Figure):
                figs[b.id] = b.number = len(figs) + 1
            elif isinstance(b, Table) and b.numbered:
                tabs[b.id] = b.number = len(tabs) + 1

        def swap(m: re.Match[str]) -> str:
            kind, key = m.group(1), m.group(2)
            n = (figs if kind == "fig" else tabs).get(key)
            if n is None:
                return "the figure" if kind == "fig" else "the table"
            return f"Figure {n}" if kind == "fig" else f"Table {n}"

        for b in self.blocks:
            if isinstance(b, Para):
                b.text = _REF.sub(swap, b.text)
            elif isinstance(b, Bullets):
                b.items = [_REF.sub(swap, i) for i in b.items]
            elif isinstance(b, Callout):
                b.body = _REF.sub(swap, b.body)
            elif isinstance(b, (Figure, Table)):
                b.caption = _REF.sub(swap, b.caption)
        return self

    def number_headings(self) -> None:
        counters = [0, 0, 0]
        for b in self.blocks:
            if not isinstance(b, Heading) or not b.numbered:
                continue
            lvl = max(1, min(3, b.level))
            counters[lvl - 1] += 1
            for i in range(lvl, 3):
                counters[i] = 0
            b.number = ".".join(str(c) for c in counters[:lvl])

    @property
    def figures(self) -> list[Figure]:
        return [b for b in self.blocks if isinstance(b, Figure)]

    @property
    def tables(self) -> list[Table]:
        return [b for b in self.blocks if isinstance(b, Table)]

    def outline(self) -> list[str]:
        """The numbered headings, for a contents list and for tests."""
        return [f"{b.number} {b.text}".strip() for b in self.blocks if isinstance(b, Heading)]
