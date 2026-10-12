"""The documents of a study and the zip that carries everything.

:func:`build` adds the report (Markdown, HTML, Word), the workbook, the
notebook, ``study.yaml`` and ``workspace.json`` to the workspace as
artifacts, then the bundle zip with every artifact and a README. Nothing
here touches the filesystem except :func:`export`, which writes the
artifacts into a directory.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

from aquascope.studio.deliverables import _common as c
from aquascope.studio.deliverables.notebook import notebook_json
from aquascope.studio.deliverables.report_docx import memo_docx_bytes, report_docx_bytes
from aquascope.studio.deliverables.report_md import memo_html, report_html, report_markdown
from aquascope.studio.deliverables.workbook import workbook_bytes
from aquascope.studio.workspace import MEDIA_TYPES, Artifact, Workspace

logger = logging.getLogger(__name__)

#: The formats :func:`build` knows, in the order they are made. ``zip`` is the bundle of all the others.
FORMATS = ("md", "html", "docx", "memo", "memo-html", "xlsx", "ipynb", "yaml", "findings", "json", "zip")

_SPEC: dict[str, tuple[str, str, str, str]] = {
    # format: (artifact id, kind, file name, media type key)
    "md": ("report-md", "document", "report.md", "md"),
    "html": ("report-html", "document", "report.html", "html"),
    "docx": ("report-docx", "document", "report.docx", "docx"),
    "memo": ("memo-docx", "document", "memo.docx", "docx"),
    "memo-html": ("memo-html", "document", "memo.html", "html"),
    "xlsx": ("workbook", "workbook", "workbook.xlsx", "xlsx"),
    "ipynb": ("notebook", "notebook", "study.ipynb", "ipynb"),
    "yaml": ("study", "study", "study.yaml", "yaml"),
    "findings": ("findings", "data", "findings.json", "json"),
    "json": ("workspace", "data", "workspace.json", "json"),
    "zip": ("bundle", "bundle", "bundle.zip", "zip"),
}

_CAPTIONS = {
    "md": "The report in Markdown (figures as relative paths).",
    "html": "The technical report as one self-contained HTML page that prints to A4 (print it for a PDF).",
    "docx": "The technical report as a Word document: cover and document control, summary, data, method, "
            "results, checks, limitations, recommendations, references.",
    "memo": "The technical memorandum as a Word document: the answer, its basis and its conditions, in a few pages.",
    "memo-html": "The technical memorandum as one self-contained HTML page (print it for a PDF).",
    "xlsx": "The workbook: README, inventory, plan, gates, every table, the figure index, the ledger.",
    "ipynb": "The notebook that re-runs the study and redraws the figures.",
    "yaml": "The study file: aquascope run study.yaml replays it with no model.",
    "findings": "The Interpreter's findings: every claim with the result path it rests on, the decision block "
                "with its grade, the data the crew would ask for.",
    "json": "The workspace without the artifact bytes (resume with aquascope studio --resume).",
    "zip": "Everything above and every figure and table, with a README, and engineering/ inputs for HEC, SWMM, "
           "MODFLOW 6, Delft-FEWS and Raven when the study has a gauge record.",
}


def _artifact(fmt: str, data: bytes) -> Artifact:
    aid, kind, name, media = _SPEC[fmt]
    return Artifact(id=aid, kind=kind, name=name, data=data, media_type=MEDIA_TYPES[media], caption=_CAPTIONS[fmt])


#: The folder of engineering-tool inputs inside the bundle (#519).
ENGINEERING_DIR = "engineering"


def _main_record(ws: Workspace) -> dict[str, Any] | None:
    """The study's main gauge record: the first step whose result is a station series AquaScope can export,
    with its full observations (the retained ``series`` table, else the payload's own arrays)."""
    import pandas as pd

    from aquascope.io.engineering import kind_of
    from aquascope.studio.deliverables._payload import series_of
    from aquascope.studio.deliverables.tables import frame_of

    for r in c.results_of(ws):
        payload = r.get("result")
        if not r.get("ok", True) or not isinstance(payload, dict) or payload.get("error"):
            continue
        try:
            kind_of(payload.get("variable"))
        except ValueError:
            continue
        series = None
        art = next((a for a in ws.artifacts if a.id == f"tab-{r.get('id')}-series"), None)
        if art is not None:
            df = frame_of(art)
            if {"datetime", "value"} <= set(df.columns):
                series = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(),
                                   index=pd.to_datetime(df["datetime"], utc=True, errors="coerce"))
        if series is None:
            got = series_of(payload, "observations", preserve_time=True) or series_of(payload, preserve_time=True)
            if got:
                series = pd.Series(pd.to_numeric(pd.Series(got[1]), errors="coerce").to_numpy(),
                                   index=pd.to_datetime(pd.Series(got[0]), utc=True, errors="coerce"))
        if series is None or not series.notna().any():
            continue
        if series.index.tz is not None and (series.index == series.index.normalize()).all():
            series.index = series.index.tz_localize(None)  # plain dates: no time zone to keep
        return {"series": series[series.index.notna()], "payload": payload, "step": r.get("id")}
    return None


def engineering_files(ws: Workspace) -> tuple[dict[str, bytes], str | None]:
    """Inputs for every engineering tool that takes the study's main record, under ``engineering/``.

    Returns the files and the record's label (None, with no files, when the study has no exportable record).
    A tool that cannot take the record (too few years for HEC-SSP, say) is left out, not an error.
    """
    from aquascope.io import engineering as eng
    from aquascope.studio.deliverables._payload import record_name

    main = _main_record(ws)
    if main is None:
        return {}, None
    p = main["payload"]
    try:
        rec = eng.make_record(main["series"], variable=p.get("variable"), unit=p.get("unit"),
                              location=p.get("station_id") or main["step"], name=p.get("station_name") or p.get("name"),
                              source=p.get("source"))
        files, skipped = eng.bundle(rec)
    except Exception as exc:  # noqa: BLE001 - the exports are an extra; the study's documents stand without them
        logger.warning("engineering exports skipped: %s", exc)
        return {}, None
    for tool, why in skipped.items():
        logger.info("engineering export %s skipped: %s", tool, why)
    return {f"{ENGINEERING_DIR}/{k}": v for k, v in files.items()}, record_name(p)


def readme_text(ws: Workspace, *, engineering: tuple[dict[str, bytes], str | None] | None = None) -> str:
    """``README.txt`` for the bundle: what is in it and how to reproduce the study.

    ``engineering`` is :func:`engineering_files` when the caller already has it (it is computed otherwise).
    """
    lines = [c.title_of(ws), "=" * min(len(c.title_of(ws)), 78), ""]
    site = c.site_text(ws)
    if site:
        lines.append(f"Site: {site}")
    lines += [f"Date: {c.date_of(ws)}", f"AquaScope {c.version()}", c.model_line(ws), ""]
    answer = c.answer_of(ws)
    if answer:
        lines += ["Answer", "------", answer, ""]
    lines += ["Files", "-----"]
    for a in sorted(ws.artifacts, key=lambda x: (x.kind != "document", x.kind, x.name)):
        if a.kind == "bundle":
            continue
        lines.append(f"{a.name:40s} {a.caption or ''}".rstrip())
    eng_files, eng_record = engineering if engineering is not None else engineering_files(ws)
    if eng_files:
        tools = sorted({k.split("/")[1] for k in eng_files})
        lines.append(f"{ENGINEERING_DIR + '/':40s} Inputs for {', '.join(tools)} from {eng_record}; "
                     "each folder has a README.")
    lines += ["", "Reproduce", "---------",
              "pip install aquascope[studio]",
              "aquascope run study.yaml          # replays every step with its gates, no model needed",
              "jupyter notebook study.ipynb      # the same, with the figures redrawn",
              "aquascope studio --resume workspace.json   # picks the conversation up where it stopped",
              "", "Cite", "----", c.citation(), ""]
    return "\n".join(lines)


def bundle_bytes(ws: Workspace) -> bytes:
    """The zip of every artifact except the bundle itself, plus ``README.txt``."""
    buf = io.BytesIO()
    eng = engineering_files(ws)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", readme_text(ws, engineering=eng))
        seen: set[str] = set()
        for a in ws.artifacts:
            if a.kind == "bundle" or a.id == "bundle" or not a.name or a.name in seen:
                continue
            seen.add(a.name)
            zf.writestr(a.name, a.data)
        for name, data in eng[0].items():
            if name not in seen:
                zf.writestr(name, data)
    return buf.getvalue()


def build(ws: Workspace, *, formats: list[str] | tuple[str, ...] | None = None) -> list[Artifact]:
    """Add the documents (and the bundle) to the workspace; returns the artifacts added, in order.

    ``formats`` picks from :data:`FORMATS` (all by default). The Word report is skipped, with a note event,
    when python-docx is not installed; ``study.yaml`` is skipped when no plan exists.
    """
    wanted = [f for f in (formats or FORMATS) if f in _SPEC]
    added: list[Artifact] = []

    def add(fmt: str, data: bytes | str | None) -> None:
        if data is None:
            return
        raw = data.encode("utf-8") if isinstance(data, str) else data
        art = ws.add_artifact(_artifact(fmt, raw))
        added.append(art)
        ws.event("author", "artifact", f"{art.name} ({art.size:,} bytes)")

    if "md" in wanted:
        add("md", report_markdown(ws))
    if "html" in wanted:
        add("html", report_html(ws))
    if "docx" in wanted:
        add("docx", report_docx_bytes(ws))
    if "memo" in wanted:
        add("memo", memo_docx_bytes(ws))
    if "memo-html" in wanted:
        add("memo-html", memo_html(ws))
    if "xlsx" in wanted:
        add("xlsx", workbook_bytes(ws))
    if "ipynb" in wanted:
        add("ipynb", notebook_json(ws))
    if "yaml" in wanted:
        add("yaml", c.study_yaml(ws) or None)
    if "findings" in wanted and ws.findings:
        import json as _json

        add("findings", _json.dumps(ws.findings, ensure_ascii=False, indent=1, default=str))
    if "json" in wanted:
        add("json", ws.to_json(with_artifacts=False, indent=1))
    if "zip" in wanted:
        add("zip", bundle_bytes(ws))
    return added


def export(ws: Workspace, out_dir: str | Path) -> dict[str, str]:
    """Write every artifact under ``out_dir`` (``figures/`` and ``tables/`` as subdirectories) and the
    ``engineering/`` inputs; returns ``{artifact id: path}``."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for a in ws.artifacts:
        if not a.name:
            continue
        rel = Path(*[p for p in Path(a.name).parts if p not in ("..", "/", "")])
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(a.data)
        paths[a.id] = str(dest)
    from aquascope.io.engineering import write_files

    # The engineering inputs are written beside the artifacts (they are not artifacts, so not in ``paths``).
    write_files(engineering_files(ws)[0], root)
    return paths


def summary(ws: Workspace) -> dict[str, Any]:
    """What a face shows after a build: the documents by id with their sizes."""
    return {a.id: {"name": a.name, "bytes": a.size, "media_type": a.media_type}
            for a in ws.artifacts if a.kind in ("document", "workbook", "notebook", "study", "bundle", "data")}
