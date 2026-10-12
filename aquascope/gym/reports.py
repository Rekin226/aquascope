"""HydroGym: score the report, not only the plan (#382).

Phase 2 (:mod:`aquascope.gym.plans`) asks whether the *plan* an agent writes
is the one a hydrologist would write. It never reads the document a person
actually receives: a plan can be perfect and the prose built on it can still
invent a number, drop a failed step from the limitations, or call a p = 0.302
trend "downward". This module scores that document, deterministically and
with no model, over a finished bundle (a ``workspace.json`` dict): the
Author's report, the Critic's own checks, and the run it was built on.

:func:`score_report` returns six dimensions in ``[0, 1]`` (``None`` when the
run gives nothing to judge it on, left out of the mean rather than scored
zero, the same convention :mod:`aquascope.gym.plans` uses for a part its
reference cannot judge):

* ``traceability``: the share of the report's numeric claims that trace to a
  tool result (its own arguments counted, as :mod:`aquascope.ai_engine.verify`
  does) within 2 percent, or the prose's own rounding.
* ``not_established_completeness``: the share of what :func:`aquascope.
  studio.roles.critic.not_established` lists (a failed gate, a step that did
  not run, the stop reason) that the report's own prose (not the mechanical
  checklist a bundle appends) actually names.
* ``no_filled_holes``: no number quoted near a step that did not establish
  its result, and no direction word (increasing, declining, ...) asserted
  where the evidence does not support it.
* ``interval_discipline``: a return level, or any result gated by
  ``ci_finite``, is not quoted alone; its interval comes with it.
* ``citations``: the registry citations of the methods actually used
  (``aquascope.methods.METHODS[...].citation``, carried on the run's results
  as ``methods[].citation``) are traceable in the report's own reference
  list.
* ``recommendation_support``: the recommendations section adds no new
  untraced number and no attribution ("because of", "due to") the run did
  not establish.

Checks are the Critic's own: :func:`aquascope.studio.roles.critic.
tool_results`, :func:`aquascope.studio.roles.critic.not_established` and
:func:`aquascope.ai_engine.verify.verify` are reused rather than
reimplemented, over a :class:`aquascope.studio.workspace.Workspace` rebuilt
with ``Workspace.from_dict``.

## Two literature metrics (2026-09-14)

A `citation check <https://arxiv.org/abs/2609.06192>`_ dated 2026-09-14
("SciRigor") scores a claim by the weakest link of its evidence chain and
reports an evidence-chain precision and recall plus the earliest layer a
chain breaks at (data, transform, result, claim); a second
(`arXiv:2608.25336 <https://arxiv.org/abs/2608.25336>`_, "claim-locked
reporting") checks the numbers a report shows for cross-run agreement and
flags direction inversions, a claim saying "increasing" or "above" where the
result says the opposite, or "significant" against p > 0.05; a third
(`arXiv:2606.16603 <https://arxiv.org/abs/2606.16603>`_, "VeriGraph") reports
a grounding rate, the share of atomic claims recoverable from the evidence.
``score_report``'s ``metrics`` carries the three: ``grounding_rate`` (the
same fraction ``traceability`` scores), ``evidence_chain`` (``precision``,
``recall``, ``earliest_failure_layer``) and ``direction_inversions`` (which
also lowers ``no_filled_holes``, since an inverted claim is worse than a
merely untraced one). Cross-run agreement itself needs more than one run of
the same case and is not computed here; a caller comparing repeated runs of
:func:`score_report` can diff their ``metrics.numbers_without_evidence`` and
``dimensions`` for that.

## The reference schema

A reference (``aquascope/gym/reports/<id>.yaml``, the rules in
``_authoring.yaml``) is a small set of assertions a faithful report of one
recorded study must and must not make, written by a hydrologist from the
recorded ``workspace.json``: ``must_say`` and ``must_not_say`` are lists of
:class:`Assertion` (a number with a tolerance, a phrase, a failed step that
must be named under a section, a forbidden number near a keyword, a grade
phrase), ``decision`` is the headline answer (value, unit, tolerance, grade),
and ``findings`` is optional, for the shape `#417 <https://github.com/
Rekin226/aquascope/issues/417>`_ adds to a workspace.

## Findings (once #417 lands)

When ``ws_dict["findings"]`` is present, :func:`score_report` scores it too,
every part optional (absent means skipped, not zero): ``basis_resolution_rate``
(the share of a finding's ``basis`` paths that resolve against the run's
results with :func:`aquascope.gates.resolve_path`, the path's first segment
read as the step id), ``wrong_but_valid_rate`` (of the resolving bases, the
share whose value is not within 2 percent of any number in the claim),
``grade_agreement`` and ``decision_agreement`` with a reference grade (exact
match, and value-within-tolerance with a grade at least as cautious,
respectively: established, indicative, screening, not_established, in that
order of caution), and ``asked_when_reference_asks`` (the reference's data
request keywords found among the findings' own).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "DIMENSIONS", "GRADES", "REPORTS_DIR", "SHOWCASE_DIR", "Assertion", "Reference", "ReportResult",
    "list_references", "load_reference", "load_references", "load_report_results", "report_leaderboard",
    "run_report_bench", "score_report", "score_study_dir", "summarize_reports",
]

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"
#: The Explorer's recorded studies (#382 scores these twelve as the first row): a directory per case, each
#: ``workspace.json``, ``report.md``, ``study.yaml`` and ``meta.json``.
SHOWCASE_DIR = Path(__file__).resolve().parents[2] / "explorer" / "showcase" / "studies"

#: The six dimensions :func:`score_report` scores, in the order it reports them.
DIMENSIONS = ("traceability", "not_established_completeness", "no_filled_holes", "interval_discipline",
              "citations", "recommendation_support")

#: A findings grade, least cautious (most confident) to most cautious, matching #417's shape.
GRADES = ("established", "indicative", "screening", "not_established")
_GRADE_ORDER = {g: i for i, g in enumerate(GRADES)}
#: Phrases a report's own prose uses for each grade, for the ``grade`` assertion kind and a reference's
#: ``decision.grade``. Approximate by design: a hydrologist writing a reference picks the phrase that is
#: actually in the recorded prose, so this need only recognise it, not anticipate every wording.
_GRADE_PHRASES: dict[str, tuple[str, ...]] = {
    "established": ("established", "measured", "confirmed", "well established"),
    "indicative": ("indicative", "planning estimate", "approximate", "provisional estimate"),
    "screening": ("screening", "screening-level", "screening estimate"),
    "not_established": ("not established", "could not be established", "not determined", "is not established"),
}


# ── number and text helpers (self-contained: verify.py's own helpers are private) ──────────────────────────


#: The optional exponent matters: without it "7.6e-05" leaves a bare "-05" behind, read as -5.
_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")
_DATE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")  # an ISO date is prose, not three claimed numbers
_PERCENT = re.compile(r"-?\d[\d,]*\.?\d*\s*%")
#: Q95, T100, SPI-12, last-30 (day): a label, not a claim. The optional hyphen matters: without it "SPI-12"
#: leaves "-12" behind, a false negative claim from a drought index's own name.
_LABEL = re.compile(r"\b[A-Za-z]+-?\d+[A-Za-z\d]*\b")
_RANGE = re.compile(r"(?<=\d)\s*-\s*(?=\d)")  # "297-348" is a range, not a negative number
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTED = re.compile(r"'([^']{2,})'")
_STEP_REF = re.compile(r"^Step (\S+)")
_GATE_REF = re.compile(r"gate (\w+)")
#: A decimal, or a bare integer of two digits or more (never a lone "1" or a digit inside a step id like "s4"):
#: a length gate's own detail ("9.7 years of record, 10 needed") is usually echoed in prose by its numbers
#: rather than by the internal step id or check name.
_NUM_TOKEN = re.compile(r"\b\d+\.\d+\b|\b\d{2,}\b")
#: Exact inflected forms, not stems: a stem match ("fall") would also catch "fallback", a step's own recovery
#: path and not a claim about direction at all.
_UP_WORDS = frozenset({"increasing", "increase", "increased", "increases", "rising", "rise", "rises", "rose",
                       "upward", "growing", "grows", "grew", "uptrend", "higher"})
_DOWN_WORDS = frozenset({"decreasing", "decrease", "decreased", "decreases", "declining", "decline", "declined",
                         "declines", "falling", "fell", "falls", "fall", "downward", "dropping", "dropped", "drops",
                         "downtrend", "lower"})
_NEGATORS = {"no", "not", "n't", "without", "non", "isn't", "doesn't", "don't", "wasn't", "hasn't"}
#: A sentence carrying one of these is posing the question or reporting that a test did not run, not asserting
#: a direction: "the decision is whether ... a declining trend", "the question of decline is not supported".
_SENTENCE_HEDGES = ("whether ", "question of ", "not supported", "not established", "could not", "cannot",
                    "can't", "unable to", "is not", "was not", "were not", "are not")
_TREND_CONTEXT = ("trend", "mann-kendall", "mann kendall", "sen's slope", "sen slope")


def _claim_numbers(text: str | None) -> list[float]:
    """Numbers a reader would take as a claim: no dates, percentages, labels (Q95, T100) or bare years."""
    if not text:
        return []
    t = _RANGE.sub(" ; ", _LABEL.sub(" ", _PERCENT.sub(" ", _DATE.sub(" ", text))))
    out = []
    for tok in _NUM.findall(t):
        try:
            v = float(tok.replace(",", ""))
        except ValueError:
            continue
        if abs(v) < 0.001:
            continue
        if 1800 <= v <= 2100 and float(v).is_integer():
            continue
        out.append(v)
    return out


def _payload_numbers(payload: Any) -> list[float]:
    """Every number anywhere in a tool result or its arguments."""
    found: list[float] = []
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            found.append(float(item))
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, str):
            found.extend(_claim_numbers(item))
    return found


def _pool_numbers(ok_results: list[dict[str, Any]]) -> list[float]:
    pool: list[float] = []
    for r in ok_results:
        pool.extend(_payload_numbers(r.get("payload")))
        pool.extend(_payload_numbers(r.get("arguments")))
    return pool


def _close(value: float, pool: list[float], *, rel: float = 0.02) -> bool:
    """Is ``value`` in ``pool``, allowing for rounding in the prose (a relative tolerance)?"""
    for other in pool:
        if other == value:
            return True
        scale = max(abs(value), abs(other), 1e-9)
        if abs(other - value) / scale <= rel:
            return True
    return False


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(text or "") if s.strip()]


def _asserts_any(sentence_low: str, words: frozenset[str]) -> bool:
    """Does the sentence assert one of ``words``, unnegated and not merely posing the question?

    "no declining trend" and "not a decline" do not count: the four tokens
    before the match are checked for a negation cue. Nor does "the decision
    is whether levels show a declining trend" or "the question of decline is
    not supported": :data:`_SENTENCE_HEDGES` anywhere in the sentence voids
    it (a hedge far from the word is still the same sentence's point).
    """
    if any(h in sentence_low for h in _SENTENCE_HEDGES):
        return False
    tokens = re.findall(r"[a-z']+", sentence_low)
    for i, tok in enumerate(tokens):
        if tok not in words:
            continue
        if any(p in _NEGATORS for p in tokens[max(0, i - 4):i]):
            continue
        return True
    return False


#: A percentage's own value, for a reference's "number" assertion: the traceability dimension deliberately
#: drops percentages (they are usually arithmetic the prose did on two numbers that are each already a claim,
#: and flagging the ratio as unsupported would discredit a correct answer), but a reference that names 96.5 as
#: the number to find should still match "96.5%" in the prose.
_PERCENT_VALUE = re.compile(r"-?\d[\d,]*\.?\d*(?=\s*%)")


def _numbers_incl_percent(text: str) -> list[float]:
    nums = _claim_numbers(text)
    for tok in _PERCENT_VALUE.findall(text or ""):
        try:
            nums.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return nums


def _mean(values: Any) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


# ── the report as text and as a claim pool ──────────────────────────────────


def _report_text(ws: Any) -> str:
    """The report's own prose: the answer and every section but the appendix and the reference list, the same
    scope :func:`aquascope.studio.roles.critic.critique` reads (its own ``_draft`` is private, so this is a
    small, independent reimplementation, not an import of it)."""
    report = ws.report or {}
    parts = [str(report.get("answer") or "")]
    for s in report.get("sections") or []:
        if s.get("id") in ("appendix", "references"):
            continue
        parts.append(str(s.get("text") or ""))
    return "\n\n".join(p for p in parts if p)


def _section_text(ws: Any, section_id: str | None) -> str | None:
    if section_id is None:
        return None
    if section_id == "answer":
        return str((ws.report or {}).get("answer") or "")
    for s in (ws.report or {}).get("sections") or []:
        if s.get("id") == section_id:
            return str(s.get("text") or "")
    return None


def _recommendation_text(ws: Any) -> str:
    return _section_text(ws, "recommendations") or ""


def _step_tool(ws: Any, step_id: str) -> str | None:
    for s in ((ws.study or {}).get("steps")) or []:
        if isinstance(s, dict) and s.get("id") == step_id:
            return s.get("tool")
    return None


def _effective_failed_steps(ws: Any) -> list[dict[str, Any]]:
    """Steps that did not establish a usable result: ``run.failed_steps`` when the run carries it (added
    2026-09-08, #412), else derived the same way, from the raw results ``StudyRun.failed_steps`` itself reads
    (the tool failed or was skipped, or neither the step nor its fallback passed every gate)."""
    run = ws.run or {}
    if run.get("failed_steps"):
        return [dict(f) for f in run["failed_steps"] if isinstance(f, dict)]
    out: list[dict[str, Any]] = []
    for r in run.get("results") or []:
        if r.get("skipped"):
            out.append({"id": r.get("id"), "tool": r.get("tool"), "reason": r.get("error") or "skipped",
                        "skipped": True})
            continue
        if not r.get("ok"):
            out.append({"id": r.get("id"), "tool": r.get("tool"), "reason": r.get("error") or "the tool failed",
                        "skipped": False})
            continue
        if r.get("gates_passed", True):
            continue
        fb = r.get("fallback")
        if r.get("fallback_used") and isinstance(fb, dict) and fb.get("ok") and fb.get("gates_passed"):
            continue
        out.append({"id": r.get("id"), "tool": r.get("tool"), "reason": "a gate failed", "skipped": False})
    return out


def _step_keywords(ws: Any, step_id: str) -> set[str]:
    """Distinctive words to look for when checking a step is *named*: its id and tool, and any quoted token
    (a field name such as 'climate') in the failed gates or the derived reason attached to it."""
    kws = {step_id.lower()}
    tool = _step_tool(ws, step_id)
    if tool:
        kws.add(tool.lower())
    run = ws.run or {}
    for g in run.get("failed_gates") or []:
        if str(g.get("step") or "").split(".")[0] == step_id:
            kws |= {q.lower() for q in _QUOTED.findall(str(g.get("detail") or ""))}
    for f in _effective_failed_steps(ws):
        if str(f.get("id") or "") == step_id:
            kws |= {q.lower() for q in _QUOTED.findall(str(f.get("reason") or ""))}
    return {k for k in kws if len(k) > 1}


def _untraced_numbers_near(text: str, keyword: str, pool: list[float]) -> list[tuple[float, str]]:
    kw = keyword.lower()
    out = []
    for sentence in _split_sentences(text):
        if kw not in sentence.lower():
            continue
        for n in _claim_numbers(sentence):
            if not _close(n, pool):
                out.append((n, sentence.strip()[:200]))
    return out


def _item_keywords(item: str, ws: Any) -> set[str]:
    """Distinctive words in one ``not_established(ws)`` line: the step id and its tool, a gate's check name,
    and any single-quoted token (gate details quote the field, ``"nothing at 'climate'"``)."""
    kws: set[str] = set()
    m = _STEP_REF.match(item)
    if m:
        sid = m.group(1).rstrip(",").split(".")[0]
        kws.add(sid.lower())
        tool = _step_tool(ws, sid)
        if tool:
            kws.add(tool.lower())
    m2 = _GATE_REF.search(item)
    if m2:
        kws.add(m2.group(1).lower())
    kws |= {q.lower() for q in _QUOTED.findall(item)}
    kws |= set(_NUM_TOKEN.findall(item))
    return {k for k in kws if len(k) > 1}


def _trend_dicts(payload: Any) -> list[dict[str, Any]]:
    """Every dict anywhere in a payload that looks like a trend test's result: a ``trend`` classification and
    a ``p_value``, whatever tool produced it."""
    found: list[dict[str, Any]] = []
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if isinstance(item.get("trend"), str) and "p_value" in item:
                found.append(item)
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return found


def _direction_inversions(results: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Sentences that assert a trend direction the run's own trend test does not support: a p = 0.302 result
    called a "downward trend", or an increasing series called a decline. Scoped to sentences that mention the
    trend (so an unrelated "prices are falling" elsewhere is never touched)."""
    classifications = {
        str(t.get("trend")).lower() for r in results if r.get("ok") for t in _trend_dicts(r.get("payload"))
        if str(t.get("trend") or "").lower() in ("increasing", "decreasing", "no trend")
    }
    if not classifications:
        return []
    out = []
    for sentence in _split_sentences(text):
        low = sentence.lower()
        if not any(k in low for k in _TREND_CONTEXT):
            continue
        up = _asserts_any(low, _UP_WORDS)
        down = _asserts_any(low, _DOWN_WORDS)
        if not up and not down:
            continue
        supported = (up and "increasing" in classifications) or (down and "decreasing" in classifications)
        if not supported:
            out.append({"sentence": sentence.strip()[:220], "claimed": "up" if up else "down",
                        "classifications": sorted(classifications)})
    return out


def _filled_holes(failed_steps: list[dict[str, Any]], text: str, pool: list[float],
                  keyword_lookup: Any) -> list[dict[str, Any]]:
    """A number quoted near a step that did not establish its result, and does not trace to the run."""
    out = []
    for f in failed_steps:
        if f.get("skipped"):
            continue
        step_id = str(f.get("id") or "")
        if not step_id:
            continue
        kws = keyword_lookup(step_id) | {q.lower() for q in _QUOTED.findall(str(f.get("reason") or ""))}
        kws = {k for k in kws if len(k) > 2}
        if not kws:
            continue
        for sentence in _split_sentences(text):
            low = sentence.lower()
            if not any(k in low for k in kws):
                continue
            for n in _claim_numbers(sentence):
                if not _close(n, pool):
                    out.append({"step": step_id, "tool": f.get("tool"), "number": n,
                               "sentence": sentence.strip()[:220]})
    return out


# ── per-dimension scorers ────────────────────────────────────────────────────


def _score_traceability(claimed: list[float], untraced: list[float]) -> tuple[float | None, list[str]]:
    if not claimed:
        return None, []
    score = round((len(claimed) - len(untraced)) / len(claimed), 4)
    return score, [f"not in any tool result or its arguments: {n:g}" for n in untraced]


def _score_not_established(ws: Any, missing: list[str], text: str) -> tuple[float | None, list[str]]:
    if not missing:
        return None, []
    low = text.lower()
    unreflected = []
    for item in missing:
        kws = _item_keywords(item, ws)
        if kws and not any(k in low for k in kws):
            unreflected.append(item)
    score = round((len(missing) - len(unreflected)) / len(missing), 4)
    return score, unreflected


def _score_no_filled_holes(filled: list[dict[str, Any]], inversions: list[dict[str, Any]], n_failed: int,
                           n_trend: int) -> tuple[float, list[str]]:
    fh = 1.0 if not filled else max(0.0, 1.0 - len(filled) / max(1, n_failed))
    inv = 1.0 if not inversions else max(0.0, 1.0 - len(inversions) / max(1, n_trend))
    score = round((fh + inv) / 2, 4)
    evid = [f"an untraced number {h['number']:g} appears near failed step {h['step']} ({h['tool']}): {h['sentence']}"
            for h in filled]
    evid += [f"claims {i['claimed']} where the test found {', '.join(i['classifications'])}: {i['sentence']}"
             for i in inversions]
    return score, evid


_INTERVAL_RE = re.compile(r"\bbetween\b|\bCI\b|confidence|interval|\[.*,.*\]|±", re.I)


def _score_interval_discipline(ws: Any, text: str, checks: Any) -> tuple[float | None, list[str]]:
    parts: list[float] = []
    evid: list[str] = []
    flood_check = next((c for c in checks.checks if c.name == "flood_estimate_carries_uncertainty"), None) \
        if checks is not None else None
    if flood_check is not None:
        parts.append(1.0 if flood_check.passed else 0.0)
        if not flood_check.passed:
            evid.append(flood_check.detail)
    ci_gates = [g for r in (ws.run or {}).get("results") or [] for g in (r.get("gates") or [])
                if g.get("check") == "ci_finite"]
    if any(g.get("passed") for g in ci_gates):
        has_interval = bool(_INTERVAL_RE.search(text))
        parts.append(1.0 if has_interval else 0.0)
        if not has_interval:
            evid.append("a result gated by ci_finite passed, but no interval language appears in the report")
    if not parts:
        return None, []
    return round(sum(parts) / len(parts), 4), evid


_CITE_YEAR = re.compile(r"\((\d{4})[a-z]?\)")
_CITE_AUTHOR = re.compile(r"([A-Z][A-Za-z'\-]+)")


def _split_citation(text: str) -> list[str]:
    if text.count(";") and text.count("(") >= 2 and "; " in text:
        return [p.strip() for p in text.split(";") if p.strip()]
    return [text.strip()] if text.strip() else []


def _citation_present(part: str, refs_low: str) -> bool:
    year = _CITE_YEAR.search(part)
    author = _CITE_AUTHOR.search(part)
    if year and author:
        return author.group(1).lower() in refs_low and year.group(1) in refs_low
    if author:
        return author.group(1).lower() in refs_low
    return " ".join(part.lower().split())[:24] in refs_low


def _score_citations(ws: Any) -> tuple[float | None, list[str]]:
    run = ws.run or {}
    needed: dict[str, str] = {}
    for r in run.get("results") or []:
        if not r.get("ok"):
            continue
        payloads = [r.get("result")]
        fb = r.get("fallback")
        if isinstance(fb, dict) and fb.get("ok"):
            payloads.append(fb.get("result"))
        for p in payloads:
            if not isinstance(p, dict):
                continue
            for m in p.get("methods") or []:
                if isinstance(m, dict) and m.get("citation"):
                    for part in _split_citation(str(m["citation"])):
                        needed[part.lower()[:60]] = part
    if not needed:
        return None, []
    refs_low = " ".join(str(x) for x in (ws.report or {}).get("references") or []).lower()
    missing = [part for part in needed.values() if not _citation_present(part, refs_low)]
    score = round((len(needed) - len(missing)) / len(needed), 4)
    return score, [f"not in the reference list: {m[:120]}" for m in missing]


_CAUSE_PHRASES = ("because of", "due to", "caused by", "as a result of", "attributable to", "is driving",
                  "is responsible for", "the cause is")


def _score_recommendation_support(ws: Any, pool: list[float]) -> tuple[float | None, list[str]]:
    rec = _recommendation_text(ws)
    if not rec.strip():
        return None, []
    claimed = _claim_numbers(rec)
    untraced = [n for n in claimed if not _close(n, pool)]
    trace_part = 1.0 if not claimed else round((len(claimed) - len(untraced)) / len(claimed), 4)
    low = rec.lower()
    cause_hits = [p for p in _CAUSE_PHRASES if p in low]
    cause_part = 0.0 if cause_hits else 1.0
    score = round((trace_part + cause_part) / 2, 4)
    evid = [f"a recommendation number not traced to a result: {n:g}" for n in untraced]
    evid += [f"attribution language the run does not establish: {p!r}" for p in cause_hits]
    return score, evid


# ── the two literature metrics (evidence chain, earliest failure layer) ─────


_FETCH_TOOLS = frozenset({"get_timeseries", "find_stations", "describe_catchment", "assess_site",
                          "water_quality_samples", "anywhere"})


def _earliest_failure_layer(failed_steps: list[dict[str, Any]], ws: Any, untraced: list[float],
                            inversions: list[dict[str, Any]]) -> str | None:
    """SciRigor's layer a claim's evidence chain first breaks at: ``data`` (a fetch tool failed), ``transform``
    (an analysis tool failed), ``result`` (every step ran but a gate failed on what it returned), ``claim`` (the
    run is clean but the prose has an untraced number or a direction inversion), or ``None``."""
    for f in failed_steps:
        if f.get("skipped"):
            continue
        if str(f.get("tool") or "") in _FETCH_TOOLS:
            return "data"
    if failed_steps:
        return "transform"
    if (ws.run or {}).get("failed_gates"):
        return "result"
    if untraced or inversions:
        return "claim"
    return None


def _evidence_chain(pool: list[float], claimed: list[float], untraced: list[float]) -> dict[str, float | None]:
    precision = None if not claimed else round((len(claimed) - len(untraced)) / len(claimed), 4)
    uniq_pool = sorted({round(p, 6) for p in pool})
    recall = None if not uniq_pool else round(sum(1 for p in uniq_pool if _close(p, claimed)) / len(uniq_pool), 4)
    return {"precision": precision, "recall": recall}


# ── the reference schema ─────────────────────────────────────────────────────


_ASSERTION_KEYS = frozenset({"kind", "any", "value", "tol", "unit", "step", "keyword", "grade", "section", "label"})


@dataclass
class Assertion:
    """One thing a faithful report of a case must (``must_say``) or must not (``must_not_say``) do.

    ``kind`` is one of:

    * ``phrase``: ``any`` (a list of strings) — at least one appears (``must_say``) or none does
      (``must_not_say``), case-insensitive, optionally scoped to ``section``.
    * ``number``: ``value`` and ``tol`` (relative, default 0.02) — a claimed number within tolerance appears
      (``must_say``) or does not (``must_not_say``), optionally scoped to ``section``.
    * ``failed_step_named``: ``step`` — the step (its id, its tool, or a quoted field from its failed gate) is
      named within ``section`` (default ``limitations``).
    * ``no_number_near``: ``keyword`` — no number that fails to trace to the run appears in a sentence
      mentioning ``keyword`` (a fabricated cross-check, when the tool that would have produced it failed).
    * ``grade``: ``grade`` — a phrase for that grade (:data:`GRADES`) appears in the report.
    """

    kind: str
    any: list[str] = field(default_factory=list)
    value: float | None = None
    tol: float = 0.02
    unit: str | None = None
    step: str | None = None
    keyword: str | None = None
    grade: str | None = None
    section: str | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Assertion:
        return cls(kind=str(d.get("kind") or ""), any=[str(x) for x in (d.get("any") or [])],
                   value=(float(d["value"]) if d.get("value") is not None else None),
                   tol=float(d.get("tol", 0.02)), unit=d.get("unit"), step=d.get("step"), keyword=d.get("keyword"),
                   grade=d.get("grade"), section=d.get("section"), label=d.get("label"))


@dataclass
class Reference:
    """One case: a recorded study, and the assertions a faithful report of it must and must not make."""

    id: str
    study: str
    title: str | None = None
    rationale: str | None = None
    tags: list[str] = field(default_factory=list)
    must_say: list[Assertion] = field(default_factory=list)
    must_not_say: list[Assertion] = field(default_factory=list)
    #: ``{"value", "unit", "tolerance", "grade"}``: the headline answer a faithful report reaches.
    decision: dict[str, Any] = field(default_factory=dict)
    #: ``{"decision": {"grade", "value", "unit", "tolerance"}, "data_requests": [keyword, ...]}``, once #417's
    #: findings block exists on a workspace; absent, findings scoring against this reference is skipped.
    findings: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "study": self.study, "title": self.title, "rationale": self.rationale,
            "tags": list(self.tags), "must_say": [a.to_dict() for a in self.must_say],
            "must_not_say": [a.to_dict() for a in self.must_not_say], "decision": dict(self.decision),
            "findings": dict(self.findings),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, path: str | None = None) -> Reference:
        return cls(
            id=str(d.get("id") or (Path(path).stem if path else "")), study=str(d.get("study") or ""),
            title=d.get("title"), rationale=" ".join(str(d.get("rationale") or "").split()) or None,
            tags=[str(t) for t in (d.get("tags") or [])],
            must_say=[Assertion.from_dict(a) for a in (d.get("must_say") or []) if isinstance(a, dict)],
            must_not_say=[Assertion.from_dict(a) for a in (d.get("must_not_say") or []) if isinstance(a, dict)],
            decision=dict(d.get("decision") or {}), findings=dict(d.get("findings") or {}), path=path,
        )


def _case_files(reports_dir: str | Path | None = None) -> list[Path]:
    root = Path(reports_dir) if reports_dir else REPORTS_DIR
    return sorted(p for p in root.glob("*.yaml") if not p.name.startswith("_"))


def load_reference(case: str | Path, reports_dir: str | Path | None = None) -> Reference:
    """A reference by id (``kingston-flood``) or by file path."""
    from aquascope.study import _parse_yaml

    path = Path(case)
    if not path.suffix:
        root = Path(reports_dir) if reports_dir else REPORTS_DIR
        path = root / f"{path.name}.yaml"
    if not path.exists():
        known = ", ".join(p.stem for p in _case_files(reports_dir))
        raise FileNotFoundError(f"no report reference {str(case)!r}; known: {known}")
    data = _parse_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a report reference is a YAML mapping")
    return Reference.from_dict(data, path=str(path))


def load_references(reports_dir: str | Path | None = None) -> list[Reference]:
    return [load_reference(p, reports_dir) for p in _case_files(reports_dir)]


def list_references(reports_dir: str | Path | None = None) -> list[dict[str, Any]]:
    out = []
    for ref in load_references(reports_dir):
        out.append({"id": ref.id, "study": ref.study, "title": ref.title, "must_say": len(ref.must_say),
                    "must_not_say": len(ref.must_not_say), "tags": list(ref.tags)})
    return out


# ── assertion checking and reference scoring ─────────────────────────────────


def _scope_text(ws: Any, full_text: str, section: str | None) -> str:
    if not section:
        return full_text
    t = _section_text(ws, section)
    return t if t is not None else ""


def _check_assertion(ws: Any, full_text: str, pool: list[float], a: Assertion) -> tuple[bool, str]:
    """Whether the condition ``a`` describes holds. ``must_say`` wants ``True``; ``must_not_say`` treats a
    ``True`` here as the violation (the forbidden thing was found)."""
    scope = _scope_text(ws, full_text, a.section)
    low = scope.lower()
    if a.kind == "phrase":
        hit = next((p for p in a.any if p.lower() in low), None)
        where = f" in {a.section}" if a.section else ""
        return (hit is not None), (f"found {hit!r}{where}" if hit else f"none of {a.any!r} found{where}")
    if a.kind == "number":
        nums = _numbers_incl_percent(scope)
        found = _close(a.value, nums, rel=a.tol) if a.value is not None else False
        where = f" in {a.section}" if a.section else ""
        return found, (f"found a number within {a.tol:.0%} of {a.value:g}{where}" if found
                       else f"no number within {a.tol:.0%} of {a.value:g}{where}")
    if a.kind == "failed_step_named":
        section = a.section or "limitations"
        scope2 = _scope_text(ws, full_text, section)
        low2 = (scope2 or "").lower()
        kws = _step_keywords(ws, str(a.step)) if a.step else set()
        hit = next((k for k in kws if k in low2), None)
        return (hit is not None), (f"{a.step} named via {hit!r} in {section}" if hit
                                   else f"{a.step} not named in {section}")
    if a.kind == "no_number_near":
        # True means the forbidden thing (a fabricated number near the keyword) was found, matching every other
        # kind's contract (must_say wants True, must_not_say treats a True as the violation).
        text_for = full_text if not a.section else scope
        hits = _untraced_numbers_near(text_for, str(a.keyword), pool)
        return bool(hits), (f"an untraced number near {a.keyword!r}: " + "; ".join(f"{n:g}" for n, _s in hits[:4])
                            if hits else "no untraced number near " + str(a.keyword))
    if a.kind == "grade":
        phrases = _GRADE_PHRASES.get(str(a.grade), ())
        hit = next((p for p in phrases if p in low), None)
        return (hit is not None), (f"found {hit!r}" if hit else f"no phrase for grade {a.grade!r}")
    return False, f"unknown assertion kind {a.kind!r}"


def _score_reference(ws: Any, reference: Reference | dict[str, Any], full_text: str,
                     pool: list[float]) -> dict[str, Any]:
    ref = reference if isinstance(reference, Reference) else Reference.from_dict(reference)
    must_say_fail = []
    for a in ref.must_say:
        ok, note = _check_assertion(ws, full_text, pool, a)
        if not ok:
            must_say_fail.append(f"{a.label or a.kind}: {note}")
    must_not_say_hit = []
    for a in ref.must_not_say:
        ok, note = _check_assertion(ws, full_text, pool, a)
        if ok:
            must_not_say_hit.append(f"{a.label or a.kind}: {note}")
    decision = None
    if ref.decision:
        value = ref.decision.get("value")
        grade = ref.decision.get("grade")
        value_ok = None
        if value is not None:
            tol = float(ref.decision.get("tolerance", 0.05))
            value_ok = _close(float(value), _numbers_incl_percent(full_text), rel=tol)
        grade_ok = None
        if grade:
            grade_ok = any(p in full_text.lower() for p in _GRADE_PHRASES.get(str(grade), ()))
        decision = {"value_ok": value_ok, "grade_ok": grade_ok}
    n_checks = len(ref.must_say) + len(ref.must_not_say)
    n_passed = (len(ref.must_say) - len(must_say_fail)) + (len(ref.must_not_say) - len(must_not_say_hit))
    if decision:
        for key in ("value_ok", "grade_ok"):
            if decision[key] is not None:
                n_checks += 1
                n_passed += int(bool(decision[key]))
    return {
        "reference_id": ref.id, "study": ref.study,
        "must_say": {"total": len(ref.must_say), "passed": len(ref.must_say) - len(must_say_fail),
                    "failed": must_say_fail},
        "must_not_say": {"total": len(ref.must_not_say), "passed": len(ref.must_not_say) - len(must_not_say_hit),
                         "violated": must_not_say_hit},
        "decision": decision, "score": round(n_passed / n_checks, 4) if n_checks else None,
    }


# ── findings (#417) ──────────────────────────────────────────────────────────


def _resolve_basis(ws: Any, basis_path: str) -> Any:
    """A basis path's value: the first segment is a step id, the rest a payload path (:func:`aquascope.
    gates.resolve_path`) into that step's own result (or its fallback's, when the step's own is missing)."""
    from aquascope.gates import resolve_path

    step_id, _, rest = basis_path.partition(".")
    result = next((r for r in (ws.run or {}).get("results") or [] if r.get("id") == step_id), None)
    if result is None:
        return None
    payload = result.get("result")
    if payload is None and isinstance(result.get("fallback"), dict):
        payload = result["fallback"].get("result")
    return resolve_path(payload, rest) if rest else payload


def _score_findings(ws: Any, findings: dict[str, Any], reference: Reference | dict[str, Any] | None) -> dict[str, Any]:
    items = findings.get("findings") if isinstance(findings.get("findings"), list) else []
    bases_total = bases_resolved = resolved_numeric = wrong_but_valid = 0
    unresolved: list[str] = []
    wrong: list[str] = []
    for f in items:
        if not isinstance(f, dict):
            continue
        claim_nums = _claim_numbers(str(f.get("claim") or ""))
        for basis in f.get("basis") or []:
            bases_total += 1
            try:
                val = _resolve_basis(ws, str(basis))
            except Exception as exc:  # noqa: BLE001 - an unresolvable basis is data, not a crash
                val, unresolved_note = None, f"{basis}: {type(exc).__name__}"
                unresolved.append(unresolved_note)
                continue
            if val is None:
                unresolved.append(str(basis))
                continue
            bases_resolved += 1
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                resolved_numeric += 1
                if claim_nums and not _close(float(val), claim_nums, rel=0.02):
                    wrong_but_valid += 1
                    wrong.append(f"{f.get('id')}: basis {basis} resolves to {val:g}, not within 2% of the claim")
    out: dict[str, Any] = {
        "basis_resolution_rate": round(bases_resolved / bases_total, 4) if bases_total else None,
        "wrong_but_valid_rate": round(wrong_but_valid / resolved_numeric, 4) if resolved_numeric else None,
        "grade_agreement": None, "decision_agreement": None, "asked_when_reference_asks": None,
        "evidence": {"unresolved_basis": unresolved[:20], "wrong_but_valid": wrong[:20], "n_findings": len(items)},
    }
    if reference is not None:
        ref = reference if isinstance(reference, Reference) else Reference.from_dict(reference)
        rdec = ref.findings.get("decision") or {}
        fdec = findings.get("decision") if isinstance(findings.get("decision"), dict) else {}
        if rdec.get("grade"):
            out["grade_agreement"] = float(str(fdec.get("grade") or "") == str(rdec["grade"]))
        if rdec.get("value") is not None:
            value = fdec.get("value")
            tol = float(rdec.get("tolerance", 0.05))
            value_ok = value is not None and _close(float(rdec["value"]), [float(value)], rel=tol)
            grade_ok = True
            if rdec.get("grade") and fdec.get("grade"):
                grade_ok = _GRADE_ORDER.get(str(fdec["grade"]), 99) >= _GRADE_ORDER.get(str(rdec["grade"]), 0)
            out["decision_agreement"] = float(bool(value_ok) and bool(grade_ok))
        wants = [str(w) for w in (ref.findings.get("data_requests") or [])]
        if wants:
            have = " ".join(f"{d.get('what', '')} {d.get('why', '')}" for d in findings.get("data_requests") or []
                            if isinstance(d, dict)).lower()
            out["asked_when_reference_asks"] = round(sum(1 for w in wants if w.lower() in have) / len(wants), 4)
    return out


# ── score_report ──────────────────────────────────────────────────────────────


def score_report(ws_dict: dict[str, Any], reference: Reference | dict[str, Any] | None = None) -> dict[str, Any]:
    """Score the report a finished study bundle carries, deterministically, with no model.

    ``ws_dict`` is a ``workspace.json`` dict (:class:`aquascope.studio.workspace.Workspace` is rebuilt with
    ``Workspace.from_dict`` and never mutated). ``reference`` is a :class:`Reference` or its dict, scored under
    ``"reference"``; a ``findings`` key on ``ws_dict`` (the shape #417 adds) is scored under ``"findings"``. See
    the module docstring for ``dimensions`` and ``metrics``.
    """
    from aquascope.ai_engine.verify import verify
    from aquascope.studio.roles.critic import not_established, tool_results
    from aquascope.studio.workspace import Workspace

    ws = Workspace.from_dict(dict(ws_dict))
    results = tool_results(ws)
    ok_results = [r for r in results if r.get("ok")]
    pool = _pool_numbers(ok_results)
    text = _report_text(ws)
    missing = not_established(ws)
    checks = verify(text, results, question=ws.brief.problem)
    claimed = _claim_numbers(text)
    untraced = [n for n in claimed if not _close(n, pool)]

    failed_steps = _effective_failed_steps(ws)
    inversions = _direction_inversions(results, text)
    filled = _filled_holes(failed_steps, text, pool, lambda sid: _step_keywords(ws, sid))
    n_trend = len(_trend_dicts([r.get("payload") for r in ok_results])) or len(
        [t for r in ok_results for t in _trend_dicts(r.get("payload"))])

    dims: dict[str, float | None] = {}
    evid: dict[str, list[str]] = {}
    dims["traceability"], evid["traceability"] = _score_traceability(claimed, untraced)
    dims["not_established_completeness"], evid["not_established_completeness"] = _score_not_established(
        ws, missing, text)
    dims["no_filled_holes"], evid["no_filled_holes"] = _score_no_filled_holes(
        filled, inversions, len(failed_steps), max(1, n_trend))
    dims["interval_discipline"], evid["interval_discipline"] = _score_interval_discipline(ws, text, checks)
    dims["citations"], evid["citations"] = _score_citations(ws)
    dims["recommendation_support"], evid["recommendation_support"] = _score_recommendation_support(ws, pool)

    out: dict[str, Any] = {
        "status": ws.status,
        "dimensions": dims,
        "mean": _mean(dims.values()),
        "evidence": evid,
        "metrics": {
            "grounding_rate": dims["traceability"],
            "numbers_without_evidence": untraced,
            "direction_inversions": inversions,
            "evidence_chain": _evidence_chain(pool, claimed, untraced),
            "earliest_failure_layer": _earliest_failure_layer(failed_steps, ws, untraced, inversions),
        },
    }
    if reference is not None:
        out["reference"] = _score_reference(ws, reference, text, pool)
    findings = ws_dict.get("findings") if isinstance(ws_dict, dict) else None
    if isinstance(findings, dict) and findings:
        out["findings"] = _score_findings(ws, findings, reference)
    return out


# ── the bench over recorded studies ──────────────────────────────────────────


@dataclass
class ReportResult:
    """One recorded study, scored."""

    study_id: str
    model: str | None = None
    provider: str | None = None
    dimensions: dict[str, float | None] = field(default_factory=dict)
    mean: float | None = None
    evidence: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    reference: dict[str, Any] | None = None
    findings: dict[str, Any] | None = None
    cost_usd: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float | None = None
    finished: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReportResult:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def score_study_dir(study_dir: str | Path, reference: Reference | dict[str, Any] | None = None) -> ReportResult:
    """Score the ``workspace.json`` under a recorded study's directory; cost, model and seconds come from
    ``meta.json`` when it sits alongside (the recorded ledger), else from the workspace's own."""
    d = Path(study_dir)
    with (d / "workspace.json").open(encoding="utf-8") as fh:
        ws_dict = json.load(fh)
    scored = score_report(ws_dict, reference=reference)
    meta: dict[str, Any] = {}
    meta_path = d / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    tokens = meta.get("tokens") or {}
    return ReportResult(
        study_id=d.name, model=meta.get("model") or ws_dict.get("model"),
        provider=meta.get("provider") or ws_dict.get("provider"),
        dimensions=scored["dimensions"], mean=scored["mean"], evidence=scored["evidence"],
        metrics=scored["metrics"], reference=scored.get("reference"), findings=scored.get("findings"),
        cost_usd=meta.get("usd"), prompt_tokens=int(tokens.get("prompt") or 0),
        completion_tokens=int(tokens.get("completion") or 0), seconds=meta.get("seconds"), finished=_now(),
    )


def run_report_bench(studies_dir: str | Path | None = None, references: list[Reference] | None = None, *,
                     study_ids: list[str] | None = None, out: str | Path | None = None,
                     on_event: Any = None) -> list[ReportResult]:
    """Score every recorded study under ``studies_dir`` (default :data:`SHOWCASE_DIR`) against the matching
    reference, when one names it (``reference.study == study_id``). Results are appended to ``out`` as JSONL as
    they come, like :func:`aquascope.gym.plans.run_plan_bench`."""
    say = on_event or (lambda _m: None)
    root = Path(studies_dir) if studies_dir else SHOWCASE_DIR
    refs = {r.study: r for r in (references if references is not None else load_references())}
    dirs = (sorted(p for p in root.iterdir() if p.is_dir() and (p / "workspace.json").exists())
            if root.is_dir() else [])
    if study_ids:
        wanted = set(study_ids)
        dirs = [p for p in dirs if p.name in wanted]
    out_path = Path(out) if out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[ReportResult] = []
    for d in dirs:
        say(f"scoring {d.name}")
        try:
            res = score_study_dir(d, reference=refs.get(d.name))
        except Exception as exc:  # noqa: BLE001 - one broken study is a row, not the end of the run
            res = ReportResult(study_id=d.name, error=f"{type(exc).__name__}: {exc}"[:400], finished=_now())
            logger.warning("study %s failed to score: %s", d.name, res.error)
        results.append(res)
        say(f"    {d.name}: mean {res.mean}" + (f", reference {res.reference['score']}" if res.reference else "")
            + (f", error {res.error}" if res.error else ""))
        if out_path:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(res.to_dict(), ensure_ascii=False, default=str) + "\n")
    return results


def load_report_results(paths: list[str | Path]) -> list[ReportResult]:
    out: list[ReportResult] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get("study_id") and "dimensions" in row:
                    out.append(ReportResult.from_dict(row))
    return out


def summarize_reports(results: list[ReportResult]) -> list[dict[str, Any]]:
    """One row per (model, provider): the mean over the six dimensions, the overall mean, the mean reference
    score where a reference scored the study, cost, and errors."""
    groups: dict[tuple[str, str], list[ReportResult]] = {}
    for r in results:
        groups.setdefault((r.model or "", r.provider or ""), []).append(r)
    rows = []
    for (model, provider), rs in sorted(groups.items()):
        ok = [r for r in rs if not r.error]
        refd = [r for r in ok if r.reference is not None]
        row: dict[str, Any] = {
            "model": model or None, "provider": provider or None, "n": len(rs),
            "mean": _mean(r.mean for r in ok),
        }
        for dim in DIMENSIONS:
            row[dim] = _mean((r.dimensions or {}).get(dim) for r in ok)
        row["n_reference"] = len(refd)
        row["reference_score"] = _mean(r.reference["score"] for r in refd) if refd else None
        row["cost_usd"] = round(sum(r.cost_usd or 0.0 for r in rs), 4) if any(r.cost_usd for r in rs) else None
        row["errors"] = sum(1 for r in rs if r.error)
        rows.append(row)
    return rows


def report_leaderboard(results: list[ReportResult], *, out: str | Path | None = None,
                       title: str | None = None) -> str:
    """The Markdown leaderboard of report quality, next to :func:`aquascope.gym.plans.plan_leaderboard`."""
    rs = list(results)
    rows = summarize_reports(rs)
    lines = [
        f"## {title or 'HydroGym report-quality leaderboard'}", "",
        f"{len(rs)} studies, {len(rows)} model(s). Each dimension is the mean over the studies it could be "
        "judged on (a study with nothing to check on a dimension leaves it out, not zero): traceability, "
        "how much of the report's numeric claims trace to a tool result; not_established_completeness, how "
        "much of what the run did not establish the report's own prose actually names; no_filled_holes, no "
        "number quoted near a step that failed and no direction claim the result does not support; "
        "interval_discipline, an interval-gated result is not quoted alone; citations, the methods used are "
        "traceable in the reference list; recommendation_support, the recommendations add no new untraced "
        "number or unestablished cause. reference_score is the mean pass rate against a hand-written "
        "reference, for the studies one names.", "",
        "| model | studies | mean | traceability | not-established | no filled holes | intervals | citations "
        "| recommendations | reference | cost USD | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{100 * x:.0f} %"

    for r in rows:
        ref_cell = f"{pct(r['reference_score'])} ({r['n_reference']})" if r["n_reference"] else "-"
        cost_cell = "-" if r["cost_usd"] is None else f"{r['cost_usd']:.3f}"
        lines.append(
            f"| {r['model'] or 'none'} | {r['n']} | {pct(r['mean'])} | {pct(r['traceability'])} "
            f"| {pct(r['not_established_completeness'])} | {pct(r['no_filled_holes'])} "
            f"| {pct(r['interval_discipline'])} | {pct(r['citations'])} | {pct(r['recommendation_support'])} "
            f"| {ref_cell} | {cost_cell} | {r['errors']} |"
        )
    text = "\n".join(lines) + "\n"
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
    return text
