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

#: A unit with a negative exponent, once folded: "mm yr-1", "m3 s-1", "W m-2".
#: Without this the exponent reads as a claim of -1, repeatedly.
#: A unit exponent is glued to its unit and is one digit: "m s-1", "s-1", "kg-1".
#: With an optional space it also matched "slope -0" in "slope -0.0029", and the
#: orphaned ".0029" read as 29.
_EXPONENT_UNIT = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]{0,5}-\d\b(?![.,]\d)")

#: And a positive one, written closed up: "km2", "m3", "km3". No space allowed
#: here, or "gauge 3" would be swallowed along with it.
_SQUARED_UNIT = re.compile(r"\b(?:k?m|ha|km|mi|ft|in)[23]\b")

#: "68.6 W" is the longitude -68.6: the compass letter carries the sign, and the
#: prose writes the magnitude. Read the sign off the letter rather than accepting
#: either one, so a hemisphere error is still caught.
_COMPASS = re.compile(r"(\d+\.?\d*)\s*°?\s*([NSEW])\b")

#: Quantities a trend claim may be about that are *not* what analyze_station
#: tests. Its Mann-Kendall runs on annual means, so "no significant trend in
#: low flow" is a statement about something the test never measured.
_OTHER_SERIES = re.compile(r"\b(low[- ]flow|q95|q05|q90|q10|baseflow|base flow|peak|maxim|minim|"
                           r"groundwater|rainfall|precipitation)\b", re.I)

#: Conventional thresholds, not claims about anyone's data.
_CONVENTIONS = {0.05, 0.01, 0.1, 0.9, 0.95, 0.99}

#: Digit grouping with a space: "14 555" is one number, not 14 and 555. Only
#: before groups of exactly three digits, which is what grouping means, and
#: only when the leading digits stand on their own: in "Q2 325" the 2 belongs
#: to the label, and joining it read the return level as 2325 (#324).
_GROUPED = re.compile(r"(?<![\w.])(\d{1,3})((?:[\s\u00a0\u202f\u2009]\d{3})+)\b")
_GROUP_SPACE = re.compile(r"[\s\u00a0\u202f\u2009]")

#: A dash between two numbers is a range ("297-348", "453 - 525"), never a
#: sign. Read as one, every interval's upper bound came out negative (#324).
_RANGE = re.compile(r"(?<=\d)\s*-\s*(?=\d)")

#: A label with digits in it: Q2, Q95, T100, ET0, GR4J. Not a number, and not
#: to be glued onto the number that follows it (#324).
_LABEL = re.compile(r"\b[A-Za-z]+\d+[A-Za-z\d]*\b")

#: A calendar year in the prose: 1800 to 2100, standing on its own (not part of
#: a longer number, nor a decimal such as "2000.5 m3/s").
_YEAR = re.compile(r"(?<![\d.,])(1[89]\d\d|20\d\d|2100)(?!\d|[.,]\d)")

#: The label the Analyst is told to put on a fact that came from its memory
#: rather than from a tool; a sentence carrying it is not held to the data.
_GENERAL_KNOWLEDGE = re.compile(r"general knowledge", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n")


def normalise(text: str) -> str:
    """Fold a well-typeset answer onto the plain ASCII the checks compare against.

    Good models write good typography: ``m³ s⁻¹`` for the unit, a non-breaking
    hyphen inside a station id, a narrow space grouping ``14 555``. Every one of
    those made a check fail on an answer that was correct, which is worse than
    having no check at all, so the text is folded first: NFKC turns the
    superscripts into digits, the dashes become ASCII, and grouped digits join up.
    """
    folded = unicodedata.normalize("NFKC", text or "").translate(_DASHES)
    return _GROUPED.sub(lambda m: m.group(1) + _GROUP_SPACE.sub("", m.group(2)), folded)


def _numbers(text: str, *, claims_only: bool = False) -> list[float]:
    """Numbers in the text. ``claims_only`` drops the dates, percentages and units.

    A dash between two numbers separates a range, so "297-348" is 297 and 348
    rather than 297 and -348; a real sign is kept ("skew = -0.864"). Labels
    such as Q2 or T100 are dropped, so "Q2 325" is 325 alone.
    """
    text = (text or "").translate(_DASHES)
    if claims_only:
        text = _SQUARED_UNIT.sub(" ", _EXPONENT_UNIT.sub(" ", _PERCENT.sub(" ", _DATE.sub(" ", text))))
    text = _LABEL.sub(" ", _RANGE.sub(" ; ", text))
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


def _identifiers(payload: Any) -> list[str]:
    """Every station id and name anywhere in a result.

    A flood-frequency payload carries the id and no name; the search that found
    it carries the name. Looking only at the top level of each payload meant an
    answer saying "Kingston" was reported as not naming its record.
    """
    found: list[str] = []
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key in ("station_id", "name", "label"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    found.append(value)
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return found


def _decimals(value: float) -> int | None:
    """How many decimals the prose wrote: 0.004 has three, 29.0 none; None for exponent forms."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        return None
    frac = text.split(".")[1].rstrip("0") if "." in text else ""
    return len(frac)


def _close(value: float, pool: list[float], *, rel: float = 0.02) -> bool:
    """Is this number in the tool output, allowing for rounding in the prose?

    Two tolerances: a relative one for big numbers written approximately, and
    the rounding the prose itself did (tau = -0.0037 in the result, "-0.004"
    in the answer, is the same number said to three decimals).
    """
    decimals = _decimals(value)
    half_unit = 0.5 * 10 ** (-decimals) + 1e-12 if decimals is not None else None
    for other in pool:
        if other == value:
            return True
        scale = max(abs(value), abs(other), 1e-9)
        if abs(other - value) / scale <= rel:
            return True
        if half_unit is not None and abs(other - value) <= half_unit:
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
        # The arguments count as well: a coordinate the answer repeats back was
        # given to a tool that accepted it, so it is not an invented number.
        pool.extend(_walk(r.get("arguments")))
    # Years and small integers are usually prose, not claims; check the rest.
    # Strip the units before reading numbers: "m3 s-1" is a unit, not a claim
    # of 3 and -1, and flagging those would discredit the whole check.
    prose = answer
    for unit in units_in(ok_results):
        for form in _unit_spellings(unit):
            prose = re.sub(re.escape(form).replace(r"\ ", r"\s*"), " ", prose, flags=re.I)
    claimed = [n for n in _numbers(prose, claims_only=True)
               if abs(n) >= 0.001 and not (1800 <= n <= 2100 and float(n).is_integer())]
    # A coordinate written "68.6 W" is -68.6 in the arguments the tool was given.
    signed = {float(m.group(1)): (float(m.group(1)) if m.group(2) in "NE" else -float(m.group(1)))
              for m in _COMPASS.finditer(answer)}
    unsupported = [n for n in claimed
                   if n not in _CONVENTIONS
                   and not _close(n, pool)
                   and not (n in signed and _close(signed[n], pool))]
    if pool:
        v.checks.append(Check(
            "numbers_come_from_tools", not unsupported,
            "" if not unsupported else
            f"These numbers are not in any tool result: {', '.join(str(n) for n in unsupported[:6])}.",
        ))

    # 2b. Are the years in the prose traceable to a result? "The 2014 winter
    # floods" came from the model's memory in a live answer (#324). A year that
    # appears in no tool result (payload values, dates and period strings
    # included) and not in the question is listed, unless its sentence says it
    # is general knowledge, which is the label the Analyst is asked to use.
    if ok_results:
        known = {int(n) for n in pool if float(n).is_integer() and 1800 <= n <= 2100}
        known |= {int(m.group(1)) for m in _YEAR.finditer(normalise(question or ""))}
        untraced: list[int] = []
        for sentence in _SENTENCE.split(answer):
            if _GENERAL_KNOWLEDGE.search(sentence):
                continue
            for m in _YEAR.finditer(sentence):
                year = int(m.group(1))
                if year not in known and year not in untraced:
                    untraced.append(year)
        v.checks.append(Check(
            "years_traceable", not untraced,
            "" if not untraced else
            f"These years are in no tool result: {', '.join(str(y) for y in untraced[:6])}. "
            "If they are from memory, the sentence should say so: from general knowledge, not from the data.",
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
            # Only weigh sentences that are about the series the test ran on.
            # An answer that says "no significant trend in low flow" while the
            # tool tested annual means is being precise, not contradictory, and
            # flagging it taught the reader to distrust the checks.
            claims = [s for s in re.split(r"(?<=[.!?])\s+|\n", answer)
                      if re.search(r"\bsignificant\b", s, re.I) and not _OTHER_SERIES.search(s)]
            said = " ".join(claims)
            says_significant = bool(said) and not re.search(r"\bno significant|not significant\b", said, re.I)
            actually = p is not None and p < 0.05
            agrees = not said or says_significant == actually
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
        stations.extend(_identifiers(r.get("payload")))
    if stations:
        named = any(s and s.lower()[:12] in answer.lower() for s in stations)
        v.checks.append(Check(
            "record_is_named", named,
            "" if named else "The answer does not say which station or record the numbers come from.",
        ))

    return v
