"""Which trend a report quotes: the annual maxima for a flood question, the annual mean otherwise.

``aquascope.explore.analyze_series`` computes two Mann-Kendall tests on a
discharge record: ``trend`` on the annual means and ``ffa.amax_trend`` on the
annual maxima (#416). "Is the 100-year flood getting worse?" is a question
about the maxima; the mean flow can stay flat while the peaks rise. A supply,
water-balance or groundwater question is about the mean. These plain
functions make that choice once, so the key numbers, the template sentences
and the trend figure all quote the same series.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["FLOOD_KINDS", "is_flood_question", "mark_reported_trend", "reported_trend"]

#: Problem kinds and playbooks whose trend question is about the peaks.
FLOOD_KINDS = frozenset({"flood_risk", "flood", "flood_frequency", "design_flood", "extremes"})
#: Kinds whose trend question is about the volume: never switched to the maxima, whatever the words say.
_MEAN_KINDS = frozenset({"supply_reliability", "irrigation", "irrigation_feasibility", "groundwater_decline",
                         "drought", "drought_status", "water_quality", "ungauged_flow", "water_balance"})
_FLOOD_WORDS = re.compile(
    r"\bflood|\bpeak|\bmaxim|\bextreme|\bhigh[- ]flows?\b|\breturn (level|period)|\b\d+[- ]year (flood|flow|event)",
    re.I,
)


def is_flood_question(kind: str | None = None, question: str | None = None, playbook: str | None = None) -> bool:
    """True when the trend asked about is the trend in the floods: a flood kind or playbook, or, when neither
    names a kind, a question that talks about floods, peaks, maxima or extremes."""
    names = {str(x) for x in (kind, playbook) if x}
    if names & FLOOD_KINDS:
        return True
    if names & _MEAN_KINDS:
        return False
    return bool(question and _FLOOD_WORDS.search(str(question)))


def _amax_trend(payload: dict[str, Any]) -> dict[str, Any] | None:
    ffa = payload.get("ffa")
    tr = ffa.get("amax_trend") if isinstance(ffa, dict) else None
    if isinstance(tr, dict) and tr.get("p_value") is not None:
        return {**tr, "on": tr.get("on") or "annual maxima"}
    return None


def reported_trend(payload: dict[str, Any] | None, *, flood: bool | None = None) -> dict[str, Any] | None:
    """The trend block a report quotes for this payload. ``flood`` True picks the annual-maxima test when the
    payload has one; False the annual-mean test; None reads the choice :func:`mark_reported_trend` stored
    (``trend_reported``), else the annual mean. The block carries ``on`` (``annual maxima`` or ``annual
    mean``) so the sentence and the figure can say which series it is."""
    if not isinstance(payload, dict):
        return None
    if flood is None and isinstance(payload.get("trend_reported"), dict):
        return payload["trend_reported"]
    if flood:
        amax = _amax_trend(payload)
        if amax is not None:
            return amax
    tr = payload.get("trend")
    if isinstance(tr, dict) and tr.get("p_value") is not None:
        return {**tr, "on": tr.get("on") or "annual mean"}
    return None


def mark_reported_trend(payload: dict[str, Any] | None, *, flood: bool) -> bool:
    """Store the chosen trend on the payload as ``trend_reported`` (a flood question with an annual-maxima
    test only; the annual mean stays the default otherwise). Returns True when the payload was marked."""
    if not flood or not isinstance(payload, dict):
        return False
    amax = _amax_trend(payload)
    if amax is None:
        return False
    payload["trend_reported"] = amax
    return True
