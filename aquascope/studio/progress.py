"""The crew's events as a person reads them: one line per stage, not one per gate.

The workspace records every event the crew emits (a gate passing, a figure
drawn, a file written), which is right for the trace and wrong for a
terminal or a progress panel: sixty lines of ``reviewer s3: gate
max_return_period_factor: passed`` bury the answer. :class:`Narrator` turns
the stream into a short log::

    Reading the question
    Looking for data near the site: 9 datasets (USGS 01013500, 123 years of discharge)
    Planned 4 analyses
    ✓ Catchment description
    ✓ Record analysis                5 of 5 checks passed
    ✓ Flood frequency analysis       5 of 5 checks passed
    ✓ Reanalysis and GloFAS          1 passed, 1 could not be run
    Interpreting the results: grade indicative
    Writing the documents: report.docx, memo.docx, report.html, workbook.xlsx and 5 more

A step's line is printed when the step finishes, with its checks folded in;
an error or a failed check is always shown in full. Faces call
:meth:`Narrator.feed` with each event and print what it returns, then
:meth:`Narrator.flush` at the end. The raw stream stays in
``workspace.events`` (and behind ``--verbose`` in the CLI).
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["Narrator"]

_TOOL_WORDS = {
    "describe_catchment": "Catchment description",
    "analyze_station": "Record analysis",
    "flood_frequency": "Flood frequency analysis",
    "anywhere": "Reanalysis climate and GloFAS check",
    "similar_basins": "Donor catchment search",
    "regionalize_signatures": "Transfer from similar catchments",
    "supply_reliability": "Supply reliability",
    "low_flow_context": "Low-flow analysis",
    "drought_indices": "Drought indices",
    "drought_propagation": "Drought propagation",
    "sgi_drought": "Groundwater drought index",
    "wqi": "Water quality index",
    "who_screen": "WHO guideline screening",
    "crop_water_demand": "Crop water demand",
    "climate_projection": "Climate projection",
    "catchment_model": "Rainfall-runoff model",
    "regional_flood": "Regional flood frequency",
    "change_points": "Change-point tests",
}

_TOOL_IN_DETAIL = re.compile(r"^([a-z_]+)\(")


class Narrator:
    """Stateful translation of crew events into short lines (see the module docstring)."""

    def __init__(self, *, width: int = 34) -> None:
        self.width = width
        self._open: dict[str, dict[str, Any]] = {}
        self._said: set[str] = set()

    def _once(self, key: str, line: str) -> list[str]:
        if key in self._said:
            return []
        self._said.add(key)
        return [line]

    def _step_line(self, sid: str) -> list[str]:
        st = self._open.pop(sid, None)
        if st is None:
            return []
        gates = st["gates"]
        passed = sum(1 for g in gates if g == "passed")
        failed = sum(1 for g in gates if g == "failed")
        skipped = sum(1 for g in gates if g == "skipped")
        mark = "✗" if st["error"] or failed else "✓"
        name = _TOOL_WORDS.get(st["tool"], st["tool"].replace("_", " ").capitalize() or sid)
        bits = []
        if gates:
            if not failed and not skipped:
                bits.append(f"{passed} of {len(gates)} checks passed")
            else:
                parts = [f"{passed} passed"] + ([f"{failed} failed"] if failed else []) + \
                        ([f"{skipped} could not be run"] if skipped else [])
                bits.append(", ".join(parts))
        if st["fallback"]:
            bits.append("fell back to " + st["fallback"])
        if st["error"]:
            bits.append(st["error"][:120])
        return [f"{mark} {name:<{self.width}} {'; '.join(bits)}".rstrip()]

    def feed(self, event: dict[str, Any]) -> list[str]:  # noqa: C901 - one branch per event kind
        if not isinstance(event, dict):
            return []
        role, kind = str(event.get("role") or ""), str(event.get("event") or "")
        sid = str(event.get("step") or "")
        detail = " ".join(str(event.get("detail") or "").split())
        out: list[str] = []

        if kind in ("error",) and role not in ("runner",):
            return [f"! {role}: {detail[:200]}"]
        if role == "consultant" and kind in ("brief", "checklist"):
            return self._once("brief", "Reading the question")
        if role == "scout" and kind == "inventory":
            m = re.match(r"(\d+) dataset", detail)
            n = m.group(1) if m else "?"
            return self._once("inventory", f"Looking for data near the site: {n} datasets found")
        if role == "scout" and kind == "gauge":
            return [f"  {detail[:160]}"]
        if role == "methodologist" and kind == "plan":
            m = re.search(r"(\d+) step", detail)
            return self._once("plan", f"Planned {m.group(1)} analyses" if m else "Planned the analyses")
        if role == "methodologist" and kind == "data_request":
            return [f"? The crew would like: {detail[:200]}"]
        if role == "runner":
            if kind == "start":
                out += [line for other in list(self._open) for line in self._step_line(other)]
                tm = _TOOL_IN_DETAIL.match(detail)
                self._open[sid] = {"tool": tm.group(1) if tm else "", "gates": [], "error": "", "fallback": ""}
                return out
            if kind == "reused":
                tm = _TOOL_IN_DETAIL.match(detail)
                name = _TOOL_WORDS.get(tm.group(1), tm.group(1).replace("_", " ").capitalize()) if tm else sid
                return [f"· {name:<{self.width}} reused from the last run"]
            st = self._open.setdefault(sid, {"tool": "", "gates": [], "error": "", "fallback": ""})
            if kind in ("error", "failed", "stop"):
                st["error"] = detail
            elif kind == "fallback":
                tm = re.search(r"running ([a-z_]+)\(", detail)
                st["fallback"] = _TOOL_WORDS.get(tm.group(1), tm.group(1)) if tm else "another method"
            elif kind == "skipped":
                st["error"] = "skipped: " + detail
            return []
        if role == "reviewer" and kind == "gate":
            st = self._open.setdefault(sid, {"tool": "", "gates": [], "error": "", "fallback": ""})
            # "max_return_period_factor: passed, T = 100 years ..." / "cross_check_ratio: passed, skipped: ..."
            rest = detail.split(": ", 1)[1] if ": " in detail else detail
            if rest.startswith("passed, skipped") or rest.startswith("skipped"):
                verdict = "skipped"
            elif rest.startswith("passed"):
                verdict = "passed"
            else:
                verdict = "failed"
            st["gates"].append(verdict)
            if verdict == "failed":
                out.append(f"  ✗ {detail[:200]}")
            return out
        if role == "analyst" and kind == "gates":
            out += [line for other in list(self._open) for line in self._step_line(other)]
            return out
        if role == "interpreter" and kind == "findings":
            m = re.search(r"grade ([a-z_ ]+)", detail)
            grade = m.group(1).replace("_", " ").strip() if m else ""
            return self._once("interpret", "Interpreting the results" + (f": grade {grade}" if grade else ""))
        if role == "critic" and kind == "checks":
            return self._once("critic", f"Checking the draft: {detail}")
        if role == "author" and kind == "deliverables":
            return self._once("docs", "Writing the documents")
        if role == "coordinator" and kind == "status" and detail == "done":
            out += [line for other in list(self._open) for line in self._step_line(other)]
            return out
        return out

    def flush(self) -> list[str]:
        return [line for sid in list(self._open) for line in self._step_line(sid)]
