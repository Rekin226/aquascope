"""What the documents share: the title, the site, the date, the key numbers, the citation, the report's sections.

Read-only views over a :class:`aquascope.studio.workspace.Workspace`, so the
Markdown, HTML, Word, workbook and notebook makers say the same things in
the same words.
"""

from __future__ import annotations

import os
from typing import Any

from aquascope.studio.workspace import Artifact, Workspace

#: The concept DOI of the software (the one CITATION.cff carries), for the "how to cite" line.
CONCEPT_DOI = "10.5281/zenodo.21903143"
RELEASE_DOIS = {"0.18.0": "10.5281/zenodo.22787700"}

#: The report's sections in the order the Author writes them; anything else the Author adds goes after ``results``.
SECTION_ORDER = ("summary", "problem", "site_data", "methodology", "results", "limitations", "recommendations",
                 "references", "appendix")


def version() -> str:
    try:
        from aquascope import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def title_of(ws: Workspace) -> str:
    rep = ws.report or {}
    t = rep.get("title") or (ws.study or {}).get("title") or ws.brief.problem or "AquaScope study"
    return str(t).strip()


def answer_of(ws: Workspace) -> str:
    return str((ws.report or {}).get("answer") or "").strip()


def date_of(ws: Workspace) -> str:
    return str(ws.created or "")[:10]


def site_text(ws: Workspace) -> str:
    """``51.4123 N, 0.3123 W`` from the workspace's site (or the inventory's), else an empty string."""
    site = ws.site or (ws.inventory.site if ws.inventory else None) or {}
    lat, lon = site.get("lat", site.get("latitude")), site.get("lon", site.get("longitude"))
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return ""
    return f"{abs(la):.4f} {'N' if la >= 0 else 'S'}, {abs(lo):.4f} {'E' if lo >= 0 else 'W'}"


def model_line(ws: Workspace) -> str:
    if ws.model:
        return f"Model: {ws.model}" + (f" via {ws.provider}" if ws.provider else "") + f"; {ws.tokens:,} tokens."
    return "No model was used: the keyless path (rules, playbooks and templates) wrote this study."


def citation() -> str:
    release = version()
    doi = RELEASE_DOIS.get(release)
    text = f"Rekin226 and contributors. AquaScope Hydrology {release} [Software]. "
    if doi:
        text += f"Release DOI: https://doi.org/{doi}. "
    else:
        text += "Version DOI not recorded. "
    text += f"All versions: https://doi.org/{CONCEPT_DOI}."
    revision = os.environ.get("AQUASCOPE_REVISION")
    if revision:
        text += f" Analysis software revision: {revision}; the release DOI does not archive later code changes."
    return text


def key_numbers(ws: Workspace) -> list[dict[str, Any]]:
    rows = (ws.report or {}).get("key_numbers") or []
    return [dict(r) for r in rows if isinstance(r, dict)]


def sections(ws: Workspace) -> list[dict[str, Any]]:
    """The report's sections in the Author's order (they arrive ordered; anything malformed is dropped)."""
    out = []
    for s in (ws.report or {}).get("sections") or []:
        if not isinstance(s, dict):
            continue
        out.append({"id": str(s.get("id") or ""), "title": str(s.get("title") or s.get("id") or "").strip(),
                    "text": str(s.get("text") or ""), "figures": [str(f) for f in (s.get("figures") or [])],
                    "tables": [str(t) for t in (s.get("tables") or [])]})
    return out


def section_ids(ws: Workspace) -> set[str]:
    return {s["id"] for s in sections(ws)}


def list_of(ws: Workspace, key: str) -> list[str]:
    """A top-level list of the report (``not_established``, ``recommendations``, ``references``, ``caveats``)."""
    vals = (ws.report or {}).get(key) or []
    if isinstance(vals, str):
        vals = [vals]
    out = []
    for v in vals:
        if isinstance(v, dict):
            v = v.get("text") or v.get("citation") or v.get("detail") or " ".join(str(x) for x in v.values())
        if v is not None and str(v).strip():
            out.append(str(v).strip())
    return out


def footer_of(ws: Workspace) -> str:
    return str((ws.report or {}).get("footer") or "").strip()


def png_figure(ws: Workspace, artifact_id: str) -> Artifact | None:
    """The PNG figure artifact for an id (an ``-svg`` id is mapped to its PNG twin)."""
    a = ws.artifact(artifact_id)
    if a is None and artifact_id.endswith("-svg"):
        a = ws.artifact(artifact_id[:-4])
    if a is None or a.kind != "figure":
        return None
    if a.media_type != "image/png":
        twin = ws.artifact(a.id.removesuffix("-svg"))
        return twin if twin is not None and twin.media_type == "image/png" else None
    return a


def table_artifact(ws: Workspace, artifact_id: str) -> Artifact | None:
    a = ws.artifact(artifact_id)
    return a if a is not None and a.kind == "table" else None


#: Tables that are data, not evidence a reader checks in a document: the raw record and the raw samples. They
#: stay in the workbook and the notebook; the prose documents name where they live instead (#415).
BULK_TABLES = frozenset({"series", "samples"})


def prose_table(ws: Workspace, artifact_id: str) -> tuple[Artifact | None, str | None]:
    """The table to print in a document, or None with the one-line pointer to print instead when the table is
    bulk data (the workbook sheet that carries it)."""
    a = table_artifact(ws, artifact_id)
    if a is None:
        return None, None
    name = str((a.meta or {}).get("name") or "")
    if name in BULK_TABLES:
        sheet = f"{a.step}_{name}" if a.step else name
        rows = (a.meta or {}).get("rows")
        what = "The record" if name == "series" else "The samples"
        return None, (f"{what} ({rows} rows) is in the workbook (`workbook.xlsx`, sheet `{sheet}`) and the "
                      f"notebook, not printed here." if rows else
                      f"{what} is in the workbook (`workbook.xlsx`, sheet `{sheet}`) and the notebook.")
    return a, None


def steps_of(ws: Workspace) -> list[dict[str, Any]]:
    return [dict(s) for s in (ws.study or {}).get("steps") or [] if isinstance(s, dict)]


def results_of(ws: Workspace) -> list[dict[str, Any]]:
    """The run's per-step records (``id``, ``tool``, ``ok``, ``result``, ``gates``), in order."""
    run = ws.run or {}
    rows = run.get("results") or []
    return [dict(r) for r in rows if isinstance(r, dict)]


def gates_of(ws: Workspace) -> list[dict[str, Any]]:
    """Every gate outcome of the run with its step id (fallback gates as ``<id>.fallback``)."""
    out: list[dict[str, Any]] = []
    for r in results_of(ws):
        for g in r.get("gates") or []:
            if isinstance(g, dict):
                out.append({"step": r.get("id"), **g})
        fb = r.get("fallback")
        if isinstance(fb, dict):
            for g in fb.get("gates") or []:
                if isinstance(g, dict):
                    out.append({"step": f"{r.get('id')}.fallback", **g})
    run = ws.run or {}
    if not out and isinstance(run.get("gates"), list):
        out = [dict(g) for g in run["gates"] if isinstance(g, dict)]
    return out


def study_yaml(ws: Workspace) -> str:
    """The plan as ``study.yaml`` text (empty when the Methodologist has not written one)."""
    study = ws.study_obj()
    return study.to_yaml() if study is not None else ""


#: The top-level lists of the report and where they go among the sections: after ``after`` when that section
#: exists, else before the first of ``before``; never when a section of the same id already exists.
_EXTRAS: tuple[tuple[str, str, str | None, tuple[str, ...]], ...] = (
    ("not_established", "What this study does not establish", "limitations",
     ("recommendations", "references", "appendix")),
    ("caveats", "Caveats", "limitations", ("recommendations", "references", "appendix")),
    ("recommendations", "Recommendations", None, ("references", "appendix")),
    ("references", "References", None, ("appendix",)),
)


def blocks(ws: Workspace) -> list[dict[str, Any]]:
    """The report body in order: the Author's sections (``kind: section``) with the top-level lists
    (``kind: list`` with ``key``, ``title``, ``items``) slotted where they belong."""
    secs = sections(ws)
    ids = {s["id"] for s in secs}
    pending = [(key, title, after, before) for key, title, after, before in _EXTRAS
               if key not in ids and list_of(ws, key)]
    out: list[dict[str, Any]] = []

    def emit(key: str, title: str) -> None:
        out.append({"kind": "list", "key": key, "title": title, "items": list_of(ws, key)})

    for s in secs:
        for item in list(pending):
            key, title, _after, before = item
            if s["id"] in before or s["id"].split("-")[0] in before:
                emit(key, title)
                pending.remove(item)
        out.append({"kind": "section", **s})
        for item in list(pending):
            key, title, after, _before = item
            if after and s["id"] == after:
                emit(key, title)
                pending.remove(item)
    for key, title, _after, _before in pending:
        emit(key, title)
    return out


def frame_cells(df: Any) -> tuple[list[str], list[list[Any]]]:
    """A DataFrame as ``(columns, rows)`` with missing cells as None (never the string ``nan``)."""
    values = df.astype(object).where(df.notna(), None).values.tolist()
    return [str(col) for col in df.columns], values
