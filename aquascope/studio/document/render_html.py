"""A document as one self-contained HTML page that reads like a typeset report on screen and prints to A4.

Serif body, sans headings, booktabs tables (three rules, no verticals),
figure captions below and table captions above, the answer in a ruled box,
a cover with the document-control table and the status stamp, page numbers
and a running header when printed (``@page`` margin boxes, which Chromium
prints; other browsers print without them). Figures are embedded as PNG data
URIs, so the file travels alone. Printing it from a browser is the PDF path.
"""

from __future__ import annotations

import base64
import html
import re

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
from aquascope.studio.document.style import HouseStyle

__all__ = ["render_html"]

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])")


def _inline(text: str) -> str:
    t = html.escape(str(text or ""), quote=False)
    t = _BOLD.sub(r"<strong>\1</strong>", t)
    return _ITALIC.sub(r"<em>\1</em>", t)


def _css(style: HouseStyle) -> str:
    accent = style.accent
    body = html.escape(style.body_font)
    head = html.escape(style.heading_font)
    paper = "A4" if style.paper.upper() == "A4" else "letter"
    return f"""
:root {{ --accent: {accent}; --ink: #1a1a1a; --muted: #5b6770; --rule: #1a1a1a; --tint: #f3f6f8;
  --caution: #b26a00; --fail: #b3261e; }}
* {{ box-sizing: border-box; }}
html {{ background: #e9ecef; }}
body {{ margin: 0; color: var(--ink); background: #e9ecef;
  font: {style.body_size}pt/1.45 "{body}", "STIX Two Text", "Times New Roman", Times, Georgia, serif;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; font-variant-numeric: lining-nums; }}
.page {{ max-width: 210mm; margin: 24px auto; background: #fff; padding: 22mm 24mm 24mm;
  box-shadow: 0 1px 3px rgba(0,0,0,.12), 0 8px 24px rgba(0,0,0,.06); }}
h1, h2, h3, .kicker, .cover-title, .cover-sub, table, .callout-title, .caption-label, .stamp, .doc-meta,
.signoff, .running {{ font-family: "{head}", "Helvetica Neue", Helvetica, Arial, sans-serif; }}
.cover {{ border-bottom: 1.5pt solid var(--rule); padding-bottom: 14pt; margin-bottom: 18pt; position: relative; }}
.cover .org {{ display: flex; align-items: center; gap: 12pt; color: var(--muted); font-size: 9pt;
  letter-spacing: .02em; font-family: "{head}", Arial, sans-serif; }}
.cover .org img {{ max-height: 34pt; }}
.kicker {{ text-transform: uppercase; letter-spacing: .14em; font-size: 8.5pt; color: var(--accent);
  font-weight: 700; margin: 26pt 0 6pt; }}
.cover-title {{ font-size: 22pt; line-height: 1.15; font-weight: 700; margin: 0; letter-spacing: -.01em; }}
.cover-sub {{ font-size: 13pt; color: var(--muted); margin: 6pt 0 0; font-weight: 400; }}
.stamp {{ position: absolute; top: 0; right: 0; border: 1.4pt solid var(--fail); color: var(--fail);
  font-weight: 700; font-size: 9pt; letter-spacing: .16em; padding: 3pt 8pt; transform: rotate(0deg); }}
.stamp.final {{ border-color: var(--accent); color: var(--accent); }}
.stamp.checked {{ border-color: var(--caution); color: var(--caution); }}
table.kv {{ width: 100%; border-collapse: collapse; font-size: 9pt; margin: 0 0 20pt; }}
table.kv th {{ text-align: left; font-weight: 600; color: var(--muted); width: 28%; padding: 2.5pt 10pt 2.5pt 0;
  vertical-align: top; border-bottom: .4pt solid #d5dade; }}
table.kv td {{ padding: 2.5pt 0; border-bottom: .4pt solid #d5dade; }}
h1 {{ font-size: 13.5pt; font-weight: 700; margin: 22pt 0 7pt; color: var(--ink); break-after: avoid; }}
h1 .num {{ color: var(--accent); margin-right: 8pt; }}
h2 {{ font-size: 11pt; font-weight: 700; margin: 15pt 0 5pt; break-after: avoid; }}
h2 .num {{ color: var(--accent); margin-right: 7pt; font-weight: 600; }}
p {{ margin: 0 0 7pt; text-align: justify; hyphens: auto; }}
p.lead {{ font-size: {style.body_size + 0.5}pt; line-height: 1.5; }}
p.small {{ font-size: {style.body_size - 2}pt; color: var(--muted); text-align: left; }}
ul, ol {{ margin: 0 0 8pt; padding-left: 18pt; }}
li {{ margin: 0 0 3.5pt; text-align: left; }}
.callout {{ border-left: 3pt solid var(--accent); background: var(--tint); padding: 9pt 12pt 8pt 13pt;
  margin: 4pt 0 12pt; break-inside: avoid; }}
.callout.caution {{ border-left-color: var(--caution); background: #fbf5ec; }}
.callout.fail {{ border-left-color: var(--fail); background: #fbefee; }}
.callout-title {{ font-size: 8pt; text-transform: uppercase; letter-spacing: .14em; color: var(--accent);
  font-weight: 700; margin-bottom: 3pt; }}
.callout.caution .callout-title {{ color: var(--caution); }}
.callout.fail .callout-title {{ color: var(--fail); }}
.callout-body {{ font-size: 15pt; font-weight: 700; line-height: 1.25; margin-bottom: 6pt;
  font-family: "{head}", Arial, sans-serif; }}
.callout table {{ border-collapse: collapse; font-size: 9pt; width: 100%; }}
.callout th {{ text-align: left; color: var(--muted); font-weight: 600; width: 24%; padding: 1.5pt 10pt 1.5pt 0;
  vertical-align: top; }}
.callout td {{ padding: 1.5pt 0; }}
figure {{ margin: 10pt 0 14pt; break-inside: avoid; }}
figure img {{ width: 100%; height: auto; display: block; }}
figure.half img {{ width: 50%; margin: 0 auto; }}
figcaption, .tcaption {{ font-size: {style.body_size - 1.5}pt; line-height: 1.4; text-align: left; }}
figcaption {{ margin-top: 5pt; }}
.caption-label {{ font-weight: 700; margin-right: 4pt; }}
.tablewrap {{ margin: 10pt 0 14pt; break-inside: avoid; overflow-x: auto; }}
.tcaption {{ margin-bottom: 4pt; }}
table.data {{ border-collapse: collapse; width: 100%; font-size: {style.body_size - 2}pt;
  font-variant-numeric: tabular-nums; border-top: 1.2pt solid var(--rule); border-bottom: 1.2pt solid var(--rule); }}
table.data.compact {{ width: auto; min-width: 55%; }}
table.data th {{ font-weight: 600; text-align: left; padding: 4pt 8pt 3pt; border-bottom: .6pt solid var(--rule);
  vertical-align: bottom; line-height: 1.25; }}
table.data td {{ padding: 2.6pt 8pt; vertical-align: top; line-height: 1.3; }}
table.data th.r, table.data td.r {{ text-align: right; }}
table.data th.c, table.data td.c {{ text-align: center; }}
table.data tr.em td {{ font-weight: 700; }}
.tnotes {{ font-size: {style.body_size - 2.5}pt; color: var(--muted); margin-top: 3pt; }}
.tnotes p {{ margin: 0 0 1pt; text-align: left; }}
.signoff {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16pt; margin: 4pt 0 22pt; font-size: 8.5pt;
  break-inside: avoid; }}
.signoff div {{ border-top: .6pt solid var(--rule); padding-top: 4pt; color: var(--muted); }}
.signoff b {{ display: block; color: var(--ink); font-weight: 600; margin-bottom: 14pt; }}
.pagebreak {{ break-after: page; }}
.footer {{ margin-top: 28pt; padding-top: 6pt; border-top: .4pt solid #c9cfd4; font-size: 8pt; color: var(--muted);
  font-family: "{head}", Arial, sans-serif; display: flex; justify-content: space-between; gap: 12pt; }}
@media screen and (max-width: 760px) {{ .page {{ padding: 18px 16px; margin: 0; box-shadow: none; }}
  .signoff {{ grid-template-columns: 1fr; }} .cover-title {{ font-size: 18pt; }} }}
@page {{ size: {paper}; margin: 20mm 22mm 22mm;
  @top-left {{ content: "{_css_str(style.author_line)}"; font: 7.5pt "{head}", Arial, sans-serif; color: #5b6770; }}
  @top-right {{ content: string(doctitle); font: 7.5pt "{head}", Arial, sans-serif; color: #5b6770; }}
  @bottom-right {{ content: "Page " counter(page) " of " counter(pages); font: 7.5pt "{head}", Arial, sans-serif;
    color: #5b6770; }}
  @bottom-left {{ content: "{_css_str(style.status or '')}"; font: 700 7.5pt "{head}", Arial, sans-serif;
    color: #b3261e; letter-spacing: .12em; }} }}
@page :first {{ @top-left {{ content: none; }} @top-right {{ content: none; }} }}
.cover-title {{ string-set: doctitle content(); }}
@media print {{ html, body {{ background: #fff; }} .page {{ box-shadow: none; margin: 0; padding: 0; max-width: none; }}
  .footer {{ display: none; }} a {{ color: inherit; text-decoration: none; }} }}
"""


def _css_str(s: str) -> str:
    return str(s or "").replace("\\", "\\\\").replace('"', '\\"')


def render_html(doc: Document, style: HouseStyle | None = None) -> str:
    """The document as a standalone HTML page."""
    style = style or HouseStyle()
    out: list[str] = []
    org = ""
    if style.logo:
        org += f'<img alt="" src="data:image/png;base64,{base64.b64encode(style.logo).decode("ascii")}">'
    org += f"<span>{html.escape(style.author_line)}</span>"
    stamp = ""
    if doc.status:
        cls = "stamp final" if doc.status.upper() in ("FINAL", "ISSUED") else \
            "stamp checked" if doc.status.upper() == "CHECKED" else "stamp"
        stamp = f'<div class="{cls}">{html.escape(doc.status.upper())}</div>'
    out.append(f'<header class="cover">{stamp}<div class="org">{org}</div>'
               f'<div class="kicker">{html.escape(doc.kind)}</div>'
               f'<h1 class="cover-title">{html.escape(doc.title)}</h1>'
               + (f'<p class="cover-sub">{html.escape(doc.subtitle)}</p>' if doc.subtitle else "") + "</header>")
    for b in doc.blocks:
        out.append(_block(b))
    footer = (f'<div class="footer"><span>{html.escape(doc.kind)} · {html.escape(doc.title)}</span>'
              f'<span>{html.escape(doc.meta.get("date", ""))}</span></div>')
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{html.escape(doc.title)}{' · ' + html.escape(doc.subtitle) if doc.subtitle else ''}</title>"
            f"<style>{_css(style)}</style></head><body><main class=\"page\">"
            + "\n".join(out) + footer + "</main></body></html>")


def _block(b: object) -> str:  # noqa: C901 - one branch per block type
    if isinstance(b, Heading):
        tag = "h1" if b.level <= 1 else "h2"
        num = f'<span class="num">{html.escape(b.number)}</span>' if b.number else ""
        return f"<{tag}>{num}{_inline(b.text)}</{tag}>"
    if isinstance(b, Para):
        cls = f' class="{b.style}"' if b.style else ""
        return f"<p{cls}>{_inline(b.text)}</p>"
    if isinstance(b, Bullets):
        tag = "ol" if b.numbered else "ul"
        return f"<{tag}>" + "".join(f"<li>{_inline(i)}</li>" for i in b.items) + f"</{tag}>"
    if isinstance(b, Figure):
        src = base64.b64encode(b.png).decode("ascii")
        cls = ' class="half"' if b.width == "half" else ""
        return (f'<figure{cls} id="fig-{html.escape(b.id)}"><img alt="{html.escape(b.caption[:120])}" '
                f'src="data:image/png;base64,{src}"><figcaption><span class="caption-label">Figure {b.number}.'
                f"</span>{_inline(b.caption)}</figcaption></figure>")
    if isinstance(b, Table):
        align = b.align or []
        head = "".join(f'<th class="{_al(align, i)}">{_inline(c)}</th>' for i, c in enumerate(b.columns))
        rows = []
        for ri, row in enumerate(b.rows):
            cls = ' class="em"' if ri in b.emphasis else ""
            cells = "".join(f'<td class="{_al(align, i)}">{_inline(_cell(v))}</td>' for i, v in enumerate(row))
            rows.append(f"<tr{cls}>{cells}</tr>")
        notes = ("<div class=\"tnotes\">" + "".join(f"<p>{_inline(n)}</p>" for n in b.notes) + "</div>"
                 if b.notes else "")
        compact = " compact" if b.compact else ""
        label = f'<span class="caption-label">Table {b.number}.</span>' if b.numbered and b.number else ""
        return (f'<div class="tablewrap" id="tab-{html.escape(b.id)}"><div class="tcaption">'
                f'{label}{_inline(b.caption)}</div>'
                f'<table class="data{compact}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody>'
                f"</table>{notes}</div>")
    if isinstance(b, Callout):
        rows = "".join(f"<tr><th>{_inline(k)}</th><td>{_inline(v)}</td></tr>" for k, v in b.rows)
        return (f'<div class="callout {b.tone}"><div class="callout-title">{html.escape(b.title)}</div>'
                + (f'<div class="callout-body">{_inline(b.body)}</div>' if b.body else "")
                + (f"<table>{rows}</table>" if rows else "") + "</div>")
    if isinstance(b, KeyValue):
        rows = "".join(f"<tr><th>{_inline(k)}</th><td>{_inline(v)}</td></tr>" for k, v in b.rows)
        return f'<table class="kv">{rows}</table>'
    if isinstance(b, Signoff):
        cells = "".join(f"<div><b>{html.escape(role)}</b>{html.escape(name) or '&nbsp;'}<br>Signature and date"
                        f"</div>" for role, name in b.roles)
        return f'<div class="signoff">{cells}</div>'
    if isinstance(b, PageBreak):
        return '<div class="pagebreak"></div>'
    return ""


def _al(align: list[str], i: int) -> str:
    return {"r": "r", "c": "c"}.get(align[i] if i < len(align) else "l", "")


def _cell(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        from aquascope.studio.document.text import num

        return num(v)
    return str(v)
