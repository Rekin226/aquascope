"""Check an answer against the tool results it was supposed to come from.

The Analyst's strongest property is that the Data and Methods sections are
assembled from tool output rather than written by the model. The prose in
between is still the model's, and that is where a wrong number can appear: a
return level quoted without its interval, a figure that is not in any result, a
"no significant trend" when the tool reported p = 0.03.

These are deterministic checks over the recorded tool calls and the answer text.
They do not ask a model to grade another model. Each returns a `Check` with a
verdict, and the loop shows every unmet one under the answer rather than
quietly hoping.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Check", "normalise", "verify"]


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class Verification:
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [c.to_dict() for c in self.checks]}

    def to_markdown(self) -> str:
        if self.ok:
            return ""
        lines = ["", "## What this answer does not establish", ""]
        lines += [f"- {c.detail or c.name}" for c in self.failed]
        return "\n".join(lines) + "\n"


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

# A date is prose, not a claim: "1986-08-21 to 2026-08-19" should not put 8, 21
# and 19 up for checking against the tool output.
_DATE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")

# A percentage is nearly always a ratio the model worked out from two numbers
# that *are* in the results ("the interval is about 7 % of the median"). Asking
# for it verbatim in a tool result would flag correct arithmetic.
_PERCENT = re.compile(r"-?\d[\d,]*\.?\d*\s*%")

#: Dashes and minus signs a model reaches for, all meaning ASCII "-".
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")

#: Digit grouping with a space: "14 555" is one number, not 14 and 555. Only
#: before a group of exactly three digits, which is what grouping means.
_GROUPED = re.compile(r"(?<=\d)[\s\u00a0\u202f\u2009](?=\d{3}\b)")


def normalise(text: str) -> str:
    """Fold a well-typeset answer onto the plain ASCII the checks compare against.

    Good models write good typography: ``m³ s⁻¹`` for the unit, a non-breaking
    hyphen inside a station id, a narrow space grouping ``14 555``. Every one of
    those made a check fail on an answer that was correct, which is worse than
    having no check at all, so the text is folded first: NFKC turns the
    superscripts into digits, the dashes become ASCII, and grouped digits join up.
    """
    folded = unicodedata.normalize("NFKC", text or "").translate(_DASHES)
    return _GROUPED.sub("", folded)


def _numbers(text: str, *, claims_only: bool = False) -> list[float]:
    """Numbers in the text. ``claims_only`` drops the dates and percentages."""
    text = text or ""
    if claims_only:
        text = _PERCENT.sub(" ", _DATE.sub(" ", text))
    out = []
    for token in _NUMBER.findall(text):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def _walk(payload: Any) -> list[float]:
    """Every number anywhere in a tool result."""
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
            stack.extend(k for k in item if isinstance(k, (int, float)))
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, str):
            found.extend(_numbers(item))
    return found


def _close(value: float, pool: list[float], *, rel: float = 0.02) -> bool:
    """Is this number in the tool output, allowing for rounding in the prose?"""
    for other in pool:
        if other == value:
            return True
        scale = max(abs(value), abs(other), 1e-9)
        if abs(other - value) / scale <= rel:
            return True
    return False


def units_in(ok_results: list[dict[str, Any]]) -> set[str]:
    """The units the successful tool results reported."""
    return {r["payload"].get("unit") for r in ok_results
            if isinstance(r.get("payload"), dict) and r["payload"].get("unit")} - {None, ""}


def _unit_spellings(unit: str) -> set[str]:
    """The ways an answer may legitimately write a unit like ``m3/s``.

    After :func:`normalise`, ``m³ s⁻¹`` reads as ``m3 s-1``, which is the same
    unit and has to count. So does ``m^3/s``, and ``cumec`` for discharge.
    """
    u = normalise(unit).lower().replace(" ", "")
    forms = {u, u.replace("/", ""), u.replace("^", "")}
    if "/" in u:
        top, _, bottom = u.partition("/")
        forms |= {f"{top}{bottom}-1", f"{top} {bottom}-1", f"{top}per{bottom}"}
    if u in {"m3/s", "m^3/s"}:
        forms |= {"cumec", "cubicmetrespersecond", "cubicmeterspersecond"}
    return {f for f in forms if f}


def _unit_named(unit: str, answer: str) -> bool:
    flat = answer.lower().replace(" ", "")
    return any(form.replace(" ", "") in flat for form in _unit_spellings(unit))


def verify(answer: str, tool_results: list[dict[str, Any]], *, question: str = "") -> Verification:
    """Run the deterministic checks over an answer and the results behind it.

    ``tool_results`` is a list of ``{"name": ..., "arguments": {...}, "payload": ...,
    "ok": bool}``, which is what the loop records as it goes.
    """
    v = Verification()
    answer = normalise(answer or "")
    ok_results = [r for r in tool_results if r.get("ok")]

    # 1. Did anything actually run?
    v.checks.append(Check(
        "tools_were_used", bool(ok_results),
        "" if ok_results else "No tool call succeeded, so nothing in this answer comes from data.",
    ))

    # 2. Do the numbers in the prose appear in the tool output?
    pool: list[float] = []
    for r in ok_results:
        pool.extend(_walk(r.get("payload")))
    # Years and small integers are usually prose, not claims; check the rest.
    # Strip the units before reading numbers: "m3 s-1" is a unit, not a claim
    # of 3 and -1, and flagging those would discredit the whole check.
    prose = answer
    for unit in units_in(ok_results):
        for form in _unit_spellings(unit):
            prose = re.sub(re.escape(form).replace(r"\ ", r"\s*"), " ", prose, flags=re.I)
    claimed = [n for n in _numbers(prose, claims_only=True)
               if abs(n) >= 0.001 and not (1800 <= n <= 2100 and float(n).is_integer())]
    unsupported = [n for n in claimed if not _close(n, pool)]
    if pool:
        v.checks.append(Check(
            "numbers_come_from_tools", not unsupported,
            "" if not unsupported else
            f"These numbers are not in any tool result: {', '.join(str(n) for n in unsupported[:6])}.",
        ))

    # 3. A return level should come with its uncertainty.
    mentions_flood = re.search(r"\b(\d+)[- ]?year (flood|return)|return level|return period", answer, re.I)
    if mentions_flood:
        has_interval = bool(re.search(r"\b(CI|confidence|interval|between .* and |\[.*,.*\]|±|to )\b", answer, re.I))
        v.checks.append(Check(
            "flood_estimate_carries_uncertainty", has_interval,
            "" if has_interval else
            "A return level is quoted without its confidence interval, which the tools do provide.",
        ))

    # 4. A trend claim should agree with the tool's significance.
    trend_payloads = [r for r in ok_results if isinstance(r.get("payload"), dict) and "trend" in str(r.get("payload"))]
    if trend_payloads and re.search(r"\btrend|increasing|decreasing|drier|wetter\b", answer, re.I):
        significance = None
        for r in trend_payloads:
            payload = r["payload"]
            trend = payload.get("trend") if isinstance(payload, dict) else None
            if isinstance(trend, dict) and "p_value" in trend:
                significance = trend
                break
        if significance is not None:
            p = significance.get("p_value")
            says_significant = bool(re.search(r"\bsignificant\b(?!ly no)", answer, re.I)) and not re.search(
                r"\bno significant|not significant\b", answer, re.I)
            actually = p is not None and p < 0.05
            agrees = says_significant == actually or not re.search(r"\bsignificant\b", answer, re.I)
            v.checks.append(Check(
                "trend_matches_the_test", agrees,
                "" if agrees else
                f"The answer's wording about significance does not match the test (p = {p}).",
            ))

    # 5. Units: if every result carries a unit, the answer should name one.
    units = units_in(ok_results)
    if units and re.search(r"\b\d", answer):
        named = any(u and _unit_named(u, answer) for u in units)
        listed = ", ".join(sorted(u for u in units if u))
        v.checks.append(Check(
            "units_are_named", named,
            "" if named else f"The answer quotes numbers without a unit; the records are in {listed}.",
        ))

    # 6. Did the answer name the record it used?
    stations = []
    for r in ok_results:
        payload = r.get("payload")
        if isinstance(payload, dict):
            for key in ("station_id", "name"):
                if isinstance(payload.get(key), str):
                    stations.append(payload[key])
    if stations:
        named = any(s and s.lower()[:12] in answer.lower() for s in stations)
        v.checks.append(Check(
            "record_is_named", named,
            "" if named else "The answer does not say which station or record the numbers come from.",
        ))

    return v
