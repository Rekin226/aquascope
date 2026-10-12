"""A document as Markdown: the same text, tables as pipe tables, figures as relative image links."""

from __future__ import annotations

from aquascope.studio.document.model import (
    Bullets,
    Callout,
    Document,
    Figure,
    Heading,
    KeyValue,
    PageBreak,
    Para,
    Signoff,
    Table,
)

__all__ = ["render_markdown"]


def _esc(v: object) -> str:
    return str("" if v is None else v).replace("|", "\\|").replace("\n", " ")


def render_markdown(doc: Document) -> str:
    out = [f"# {doc.title}", ""]
    if doc.subtitle:
        out += [f"**{doc.subtitle}**", ""]
    out += [f"*{doc.kind}*" + (f" · **{doc.status}**" if doc.status else ""), ""]
    for b in doc.blocks:
        if isinstance(b, Heading):
            out += [("#" * (b.level + 1)) + " " + (f"{b.number} " if b.number else "") + b.text, ""]
        elif isinstance(b, Para):
            out += [b.text, ""]
        elif isinstance(b, Bullets):
            out += [(f"{i}. " if b.numbered else "- ") + item for i, item in enumerate(b.items, 1)] + [""]
        elif isinstance(b, Figure):
            path = b.path or f"figures/{b.id}.png"
            out += [f"![Figure {b.number}]({path})", "", f"*Figure {b.number}. {b.caption}*", ""]
        elif isinstance(b, Table):
            out += [f"*Table {b.number}. {b.caption}*" if b.numbered and b.number else f"*{b.caption}*", ""]
            out.append("| " + " | ".join(_esc(c) for c in b.columns) + " |")
            out.append("| " + " | ".join("---:" if (b.align[i:i + 1] or ["l"])[0] == "r" else "---"
                                         for i in range(len(b.columns))) + " |")
            for ri, row in enumerate(b.rows):
                cells = [_esc(v) for v in row]
                if ri in b.emphasis:
                    cells = [f"**{c}**" if c else c for c in cells]
                out.append("| " + " | ".join(cells) + " |")
            out.append("")
            out += [f"{n}" for n in b.notes] + ([""] if b.notes else [])
        elif isinstance(b, Callout):
            out.append(f"> **{b.title}**" + (f": {b.body}" if b.body else ""))
            for k, v in b.rows:
                out.append(f"> - {k}: {v}")
            out.append("")
        elif isinstance(b, KeyValue):
            out += ["| | |", "| --- | --- |"] + [f"| {_esc(k)} | {_esc(v)} |" for k, v in b.rows] + [""]
        elif isinstance(b, Signoff):
            out += [" · ".join(f"{role}: {name or '________'}" for role, name in b.roles), ""]
        elif isinstance(b, PageBreak):
            out += ["---", ""]
    return "\n".join(out).rstrip() + "\n"
