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
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Check", "verify"]


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


def _numbers(text: str) -> list[float]:
    out = []
    for token in _NUMBER.findall(text or ""):
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


def verify(answer: str, tool_results: list[dict[str, Any]], *, question: str = "") -> Verification:
    """Run the deterministic checks over an answer and the results behind it.

    ``tool_results`` is a list of ``{"name": ..., "arguments": {...}, "payload": ...,
    "ok": bool}``, which is what the loop records as it goes.
    """
    v = Verification()
    answer = answer or ""
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
    claimed = [n for n in _numbers(answer) if abs(n) >= 0.001 and not (1800 <= n <= 2100 and float(n).is_integer())]
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
    units = {r["payload"].get("unit") for r in ok_results
             if isinstance(r.get("payload"), dict) and r["payload"].get("unit")}
    if units and re.search(r"\b\d", answer):
        named = any(u and u.lower().replace("³", "3") in answer.lower().replace("³", "3") for u in units)
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
