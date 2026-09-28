"""The notebook that re-runs a study and redraws its figures, written as plain nbformat-4 JSON.

No nbformat dependency: the structure is small and fixed (``nbformat`` 4,
``nbformat_minor`` 5, metadata, cells with an id, a type, a source and
metadata, code cells with outputs and an execution count).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from aquascope.studio.deliverables import _common as c
from aquascope.studio.workspace import Workspace


def _cell(kind: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"id": uuid.uuid4().hex[:8], "cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def rerun_workspace(study_path: str | Path, workspace_path: str | Path, *,
                    tools: dict[str, Any] | None = None) -> Workspace:
    """Run the saved plan through Studio in a fresh workspace, retaining uploaded inputs only.

    Agency-backed steps retrieve current data. The original results and files remain
    untouched; every new claim, gate, figure and input table belongs to this rerun.
    """
    from aquascope.studio.coordinator import Studio
    from aquascope.study import load

    original = Workspace.from_json(Path(workspace_path).read_text(encoding="utf-8"))
    ws = Workspace(site=deepcopy(original.site), brief=deepcopy(original.brief),
                   inventory=deepcopy(original.inventory), tables=dict(original.tables), status="review")
    study = load(study_path)
    study.results = {}
    ws.set_study(study)
    Studio(workspace=ws, tools=tools).approve()
    return ws


def notebook_cells(ws: Workspace) -> list[dict[str, Any]]:
    """The cells: a title, the run, one per step (summary, gates, figures), the workbook."""
    title = c.title_of(ws)
    answer = c.answer_of(ws)
    lines = [f"# {title}", ""]
    site = c.site_text(ws)
    if site:
        lines.append(f"**Site:** {site}  ")
    lines.append(f"**Date:** {c.date_of(ws)}  ")
    lines.append(f"**AquaScope:** {c.version()}  ")
    if answer:
        lines += ["", "**Original answer, before this rerun:**", "", answer]
    revision = os.environ.get("AQUASCOPE_REVISION", "")
    install = (f'aquascope[studio] @ git+https://github.com/Rekin226/aquascope.git@{revision}'
               if re.fullmatch(r"[0-9a-f]{7,40}", revision) else f"aquascope[studio]=={c.version()}")
    lines += ["", "This notebook reruns `study.yaml` through Studio with no language model. "
              "Agency steps request current data, so results may differ from the original. "
              "Uploaded tables are retained. Findings, gates, input tables and figures are rebuilt together "
              "in a fresh workspace; the original files are preserved. To inspect the original results "
              "without new requests, reopen the completed study in Explorer.", "",
              "Keep `study.yaml` and `workspace.json` next to this notebook. Install the recorded source "
              "revision when present, or the exact software release:", "", f'```bash\npip install "{install}"\n```']
    cells = [_cell("markdown", "\n".join(lines))]
    cells.append(_cell("code", "\n".join([
        "from pathlib import Path",
        "",
        "import io",
        "import matplotlib.pyplot as plt",
        "",
        "from aquascope.studio.deliverables.notebook import rerun_workspace",
        "",
        'ws = rerun_workspace("study.yaml", "workspace.json")',
        "run = ws.run or {}",
        'results = {r["id"]: r for r in run.get("results", [])}',
        'print("ok:", run.get("ok"), "| steps run:", len(results), "| stopped at:", run.get("stopped_at"),',
        '      run.get("stop_reason") or "")',
        'print("New answer:", (ws.report or {}).get("answer", "No report produced"))',
    ])))
    for i, step in enumerate(c.steps_of(ws), 1):
        sid = str(step.get("id") or f"s{i}")
        tool = str(step.get("tool") or "")
        rationale = step.get("rationale") or step.get("note") or ""
        md = [f"## Step {sid}: `{tool}`"]
        if rationale:
            md += ["", str(rationale)]
        if step.get("method"):
            md += ["", f"Method: `{step['method']}`"]
        cells.append(_cell("markdown", "\n".join(md)))
        cells.append(_cell("code", "\n".join([
            f"rec = results.get({sid!r})",
            "if rec is None:",
            f"    print({sid!r}, 'did not run')",
            "else:",
            "    print(rec['tool'], 'ok' if rec['ok'] else f\"failed: {rec.get('error')}\")",
            "    for g in rec.get('gates') or []:",
            "        status = 'skipped' if g.get('skipped') else 'passed' if g['passed'] else 'FAILED'",
            "        print('  gate', g['check'], status, '|', g.get('detail', ''))",
            "    payload = rec.get('result') or {}",
            "    for key in ('unit', 'years', 'start', 'end', 'n', 'status', 'verdict', 'text'):",
            "        if key in payload:",
            "            print('  ', key, '=', payload[key])",
            "    for artifact in ws.artifacts:",
            f"        if artifact.step == {sid!r} and artifact.media_type == 'image/png':",
            "            image = plt.imread(io.BytesIO(artifact.data), format='png')",
            "            fig, ax = plt.subplots(figsize=(8, 8 * image.shape[0] / image.shape[1]))",
            "            ax.imshow(image)",
            "            ax.axis('off')",
            "            plt.show()",
            "            plt.close(fig)",
            "            print(artifact.caption or artifact.name)",
        ])))
    cells.append(_cell("markdown", "## Save this run\n\nWrite the new report, workbook, full input tables, "
                                   "figures and workspace into a separate `rerun-<id>` directory. "
                                   "Check the new findings and limitations before using the results."))
    cells.append(_cell("code", "\n".join([
        "from aquascope.studio.deliverables import export",
        "",
        'out = Path(f"rerun-{ws.id}")',
        'paths = export(ws, out)',
        '(out / "workspace.json").write_text(ws.to_json(indent=1), encoding="utf-8")',
        'print("Saved", len(paths), "artifacts to", out.resolve())',
    ])))
    return cells


def notebook_dict(ws: Workspace) -> dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "aquascope": {"version": c.version(), "workspace": ws.id, "title": c.title_of(ws)},
        },
        "cells": notebook_cells(ws),
    }


def notebook_json(ws: Workspace) -> str:
    """``study.ipynb`` as text: a valid nbformat-4 notebook."""
    return json.dumps(notebook_dict(ws), ensure_ascii=False, indent=1) + "\n"
