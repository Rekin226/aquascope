"""Data requests: what the crew asks for instead of declining, and what it changes (#419).

A playbook's own decline rules name the data they miss ("bring the pumping
data", "a scheme with a reservoir needs the capacity and the rule"). Until
now a matching rule ended the study. Here the same rule becomes a request:
what the crew wants, why, what it would change, and what happens when the
user continues without it (an intake change that plans the study at a
lower grade, or nothing, in which case the decline stands).

The rules are a small table, keyed by playbook, matched against the decline
sentence and the intake; the Interpreter's post-run requests
(``aquascope.studio.roles.interpreter``) are the other half, for what the
run itself could not establish.
"""

from __future__ import annotations

import re
from typing import Any

from aquascope.studio.workspace import Workspace

__all__ = ["RULES", "continue_without", "data_request_for", "is_continue"]

#: One rule: the playbook it belongs to (None: any), the pattern over the decline sentence, the intake flag that
#: also triggers it, what to ask for, why, the effect on the answer, and the intake to apply when the user
#: continues without the data (None: the decline stands).
RULES: list[dict[str, Any]] = [
    {"playbook": "groundwater_decline", "pattern": r"abstraction|pumping", "flag": "attribute_cause",
     "what": "abstraction (pumping) records for the wells around the site, as a table (date, well, volume)",
     "why": "a decline can only be attributed to pumping, drought or land use with the abstraction history",
     "effect": "with them the cause can be tested; without them the study reports the trend, the drought "
               "index and the recharge, and stops at what the levels show (screening for the cause)",
     "continue_without": {"attribute_cause": False}, "grade_without": "screening"},
    {"playbook": "supply_reliability", "pattern": r"reservoir|storage", "flag": "storage",
     "what": "the reservoir's capacity and its operating rule (draw-down, releases, the dead storage)",
     "why": "a stored supply needs a storage-yield analysis; a run-of-river screening cannot stand in for it",
     "effect": "with them a storage-yield analysis becomes possible; without them the study screens the "
               "run-of-river reliability against the flow-duration curve (indicative for a stored scheme)",
     "continue_without": {"storage": False}, "grade_without": "indicative"},
    {"playbook": "water_quality", "pattern": r"no samples|samples? within reach|sampled", "flag": None,
     "what": "a table of your own water-quality samples (date, parameter, value, unit)",
     "why": "no agency samples are within reach of the site; an index needs measured parameters",
     "effect": "with your samples the index and the screening run on them; without them there is nothing to "
               "index", "continue_without": None, "grade_without": None},
    {"playbook": None, "pattern": r"one or two donors|too few donors|no gauge with|no usable record",
     "flag": None,
     "what": "a discharge record you hold for this stream or a nearby one (date, value, unit)",
     "why": "the regional transfer rests on too few donor gauges to quote a number",
     "effect": "with your record the study runs on it at indicative grade; without it the study cannot quote a "
               "flow", "continue_without": None, "grade_without": None},
    {"playbook": None, "pattern": r"\b(bring|upload|provide|supply)\b\s+(?:the |your |a )?([^.;]{4,80})", "flag": None,
     "what": None, "why": "the playbook names it as the missing input", "effect": "see the playbook's note",
     "continue_without": None, "grade_without": None},
]

_CONTINUE = re.compile(r"^\s*(continue|proceed|go on|carry on|without( it| them| the data)?|skip( it)?|just go|"
                       r"no data|none|go)\b", re.I)


def is_continue(text: str) -> bool:
    """Whether a reply at a data request means "continue without the data"."""
    return bool(_CONTINUE.match(text or ""))


def data_request_for(ws: Workspace, reason: str | None) -> dict[str, Any] | None:
    """The request a decline turns into, or None when the decline is not about data the user could bring."""
    pb = str(ws.brief.playbook or "")
    intake = ws.brief.intake or {}
    text = str(reason or "")
    for rule in RULES:
        if rule["playbook"] not in (None, pb):
            continue
        flag = rule.get("flag")
        m = re.search(rule["pattern"], text, re.I) if text else None
        if not m and not (flag and intake.get(flag)):
            continue
        what = rule.get("what") or (m.group(2).strip() if m and m.lastindex and m.lastindex >= 2 else None)
        if not what:
            continue
        return {"what": what, "why": rule["why"], "effect": rule["effect"], "reason": text or None,
                "continue_without": rule.get("continue_without"), "grade_without": rule.get("grade_without"),
                "playbook": pb or None,
                "can_continue": rule.get("continue_without") is not None}
    return None


def continue_without(ws: Workspace, request: dict[str, Any]) -> bool:
    """Apply what continuing without the data means: the intake changes the rule names, recorded as an
    assumption. Returns False when the request allows no continuation (the decline stands)."""
    changes = request.get("continue_without")
    if not changes:
        return False
    for k, v in changes.items():
        ws.brief.intake[k] = v
    note = (f"{request.get('what')} was not provided; the study goes on without it"
            + (f" ({request.get('grade_without')} for that part)" if request.get("grade_without") else ""))
    if note not in ws.brief.assumptions:
        ws.brief.assumptions.append(note)
    ws.event("consultant", "continue_without", note)
    return True
