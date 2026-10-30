"""What if, on the map (#565): a scenario run for a basin and painted on the rivers downstream.

"What if it rains 20 % less?" should change the rivers on the map, not only a table. This module is the one
engine behind the Explorer's "What if…" on the map card, the MCP tool ``whatif_map`` and ``aquascope whatif``:

* :func:`whatif` takes a clicked river reach (or a gauge), finds the basin that drains to it from the GEOGLOWS
  v2 routing tables (the ones :func:`aquascope.rivers.upstream_area` and :func:`aquascope.rivers.trace_downstream`
  read), runs :func:`aquascope.advanced.catchment_model` (GR4J, with a snow store where it snows, ERA5 forcing at
  the basin's centre, calibrated on the gauge or on the reach's GEOGLOWS simulated flow, validated on years it
  never saw) without and with the scenario (rain change in %, warming in degrees C, optionally a demand in m3/s
  taken from the river), and returns the change in mean flow, low flow (Q95) and a T-year flood at the outlet,
  each with a range from the parameter sets that did as well on the validation years. Then it carries the change
  down the river to the next few major confluences by area weighting (:data:`SIMPLIFICATION` says what that
  assumes).
* :func:`parse_scenario` is the keyless reading of one line ("20% less rain and 2 degrees warmer"), and
  :func:`scenario_prompt`, :func:`parse_scenario_reply` and :func:`model_scenario` the model path. The model
  parses, the code reads (#590, round 3): for each clause the model names a change kind (:data:`CHANGES`), the
  number as the words write it (an enum of the words' own number tokens, :func:`number_tokens`), a unit
  (:data:`UNITS`) and a part of the year (:data:`SEASONS`), all enums of :func:`scenario_schema`. It writes no
  value: code takes the number from the words, checks the unit against the one the words give, converts it
  ("draws 7.5 l/s" runs as 0.0075 m³/s), sets the sign and holds it to :data:`LIMITS`. Every number on the map
  comes from the code here.

Everything is plain Python over numpy, scipy and pandas, so the Explorer's Pyodide worker runs it unchanged.
Model output throughout: GEOGLOWS v2 is CC BY 4.0, the river network TDX-Hydro (CC BY-SA 4.0), ERA5 through
Open-Meteo (CC BY 4.0).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

__all__ = [
    "CHANGES", "CONFLUENCES", "FLOOD_RP", "LIMITS", "MEMBERS", "SEASONS", "SIMPLIFICATION", "UNITS", "check_scenario",
    "downstream_shares", "model_scenario", "number_tokens", "parse_scenario", "parse_scenario_reply",
    "scenario_label", "scenario_prompt", "scenario_schema", "whatif",
]

#: The bounds a scenario is held to: rain and evaporation change in %, warming in degrees C, demand in m3/s.
LIMITS: dict[str, tuple[float, float]] = {
    "dp_pct": (-80.0, 80.0), "dt_c": (-5.0, 8.0), "dpet_pct": (-50.0, 100.0), "demand_cms": (0.0, 100_000.0),
}
#: The flood reported by default (years).
FLOOD_RP = 10.0
#: Parameter sets tried for the range (see :func:`aquascope.advanced.catchment_model`).
MEMBERS = 24
#: How many major confluences the change is carried past.
CONFLUENCES = 3
#: A tributary at least this share of the main stem's drainage area above it makes a confluence "major".
MAJOR_SHARE = 0.25
#: Below this share of the water coming from the basin modelled, the change is lost and the walk stops.
MIN_SHARE = 0.02
#: The most reaches walked downstream.
MAX_PATH = 3000
#: The most basin reaches listed for the map (the largest drainage first, like upstream_ids).
BASIN_IDS = 4000
#: Above this the single ERA5 cell at the centre cannot stand for the basin's weather: pick a smaller river.
MAX_BASIN_KM2 = 100_000.0

SIMPLIFICATION = (
    "Downstream of the basin modelled the change is carried by area weighting: the water from the rest of the "
    "drainage area is assumed unchanged and to run off at the same rate per km2, so at each reach the change is "
    "the outlet's change times the share of the drainage area the basin makes up. It ignores that other "
    "tributaries may be wetter or drier, that a dam or a lake can hold the change back, and that floods from "
    "different tributaries do not peak together: the flood's change downstream is the roughest of the three. "
    "The scenario applies only to the basin modelled; the same rain change over the tributaries would not fade.")

AREA_METHOD = {
    "name": "Area-weighted routing of a change downstream",
    "text": "The relative change at the basin outlet times the basin's share of the drainage area at each reach "
            "downstream (GEOGLOWS v2 unit catchments, rapid_connect topology).",
    "citation": "A first-order dilution assumption; see the notes for what it ignores.",
}

_UNITS = {"dp_pct": "%", "dt_c": "°C", "dpet_pct": "%", "demand_cms": "m³/s"}


# ── the scenario: checked, said, read from words ─────────────────────────────


def _num(x: Any) -> float | None:
    if x is None or isinstance(x, bool) or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def check_scenario(sc: Any) -> tuple[dict[str, Any], list[str]]:
    """A scenario dict held to :data:`LIMITS`: ``{"dp_pct", "dt_c", "dpet_pct", "demand_cms"}`` (``dpet_pct`` None
    means "from the warming", the catchment model's own rule). Returns the checked scenario and what was refused
    or clipped, in words."""
    sc = sc if isinstance(sc, dict) else {}
    out: dict[str, Any] = {"dp_pct": 0.0, "dt_c": 0.0, "dpet_pct": None, "demand_cms": 0.0}
    errors: list[str] = []
    for key, (lo, hi) in LIMITS.items():
        raw = sc.get(key)
        if raw is None or raw == "":
            continue
        v = _num(raw)
        if v is None:
            errors.append(f"{key} is not a number: {raw!r}")
            continue
        if v < lo or v > hi:
            errors.append(f"{key} {v:g} is outside {lo:g} to {hi:g}; held at the limit")
            v = min(hi, max(lo, v))
        # A demand keeps its small values: 7.5 l/s is 0.0075 m³/s, not 0.007 (#590).
        out[key] = round(v, 6) if key == "demand_cms" else round(v, 3)
    return out, errors


def _g(v: float) -> str:
    """``|v|`` to three significant figures, whole numbers from 100: a value that is not nought never reads
    as "0" (0.02 m³/s taken, #590). The Explorer's ``whatif-core.js`` writes it the same way."""
    from decimal import ROUND_HALF_UP, Decimal

    a = abs(float(v))
    if not a:
        return "0"
    places = max(0, 2 - math.floor(math.log10(a)))
    # Half up on the float's exact value, as JavaScript's toFixed rounds (1234.5 is 1235 on both faces).
    out = str(Decimal(a).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
    return out.rstrip("0").rstrip(".") if "." in out else out


def scenario_label(sc: dict[str, Any]) -> str:
    """The scenario in a few words, the same on the map, in the log and in the CLI: "20% less rain, 2 °C warmer"."""
    bits = []
    dp, dt = (sc.get("dp_pct") or 0.0), (sc.get("dt_c") or 0.0)
    de, dm = sc.get("dpet_pct"), (sc.get("demand_cms") or 0.0)
    if dp:
        bits.append(f"{_g(dp)}% {'less' if dp < 0 else 'more'} rain")
    if dt:
        bits.append(f"{_g(dt)} °C {'cooler' if dt < 0 else 'warmer'}")
    if de:
        bits.append(f"{_g(de)}% {'less' if de < 0 else 'more'} evaporation")
    if dm:
        bits.append(f"{_g(dm)} m³/s taken")
    return ", ".join(bits) or "no change"


_FRACTIONS = {"a tenth": 10.0, "one tenth": 10.0, "a fifth": 20.0, "one fifth": 20.0, "a quarter": 25.0,
              "one quarter": 25.0, "a third": 100.0 / 3, "one third": 100.0 / 3, "half": 50.0}
_NEG = r"\b(?:less|fewer|lower|drier|dryer|down|minus|decrease[sd]?|reduc(?:e|ed|tion)|cut|drop|falls?|cooler|colder|" \
       r"cooling)\b"
_POS = r"\b(?:more|higher|wetter|up|plus|increase[sd]?|rise[sn]?|warmer|hotter|warming|heating)\b"
_KINDS = (
    ("dpet_pct", r"\b(?:evapo\w*|pet|et0|evaporation)\b"),
    ("dp_pct", r"\b(?:rain\w*|precip\w*|wetter|drier|dryer|snowfall)\b"),
    ("dt_c", r"(?:\bwarm\w*|\bhot\w*|\bcool\w*|\bcold\w*|\btemperature\w*|\bdegrees?\b|\d\s*(?:°|deg\b|c\b))"),
    ("demand_cms", r"\b(?:demand|abstract\w*|withdraw\w*|irrigat\w*|pump\w*|tak(?:e|en|ing)|extract\w*|divert\w*|"
                   r"draw\w*|use[sd]?)\b|m3/s|cumecs|\bl/s\b"),
)
# A number, not the digit of a unit or a name ("m3/s", "et0").
_NUMBER = re.compile(r"(?<![a-z\d.])(?P<sign>[+-])?\s*(?P<n>\d+(?:\.\d+)?)")
_FILLER = frozenset("""what if a an the of in on by with and than to for at this basin river catchment here there
    it is be was were would if please scenario per cent percent c degree degrees celsius m3 s cumecs l year years
    every each all over across more less climate future scenario world catchments flows flow gets get got becomes
    become turns goes go rains rained""".split())


# Numbers in words, two to twenty and the tens: "three degrees" is 3 degrees. "One" counts only before a unit
# ("one degree"): elsewhere it is mostly a pronoun ("a hotter one"), and "one fifth" is read by _FRACTIONS.
_WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split()) if w != "one"}
_WORD_NUMBERS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
                      "ninety": 90, "hundred": 100, "a hundred": 100, "one hundred": 100,
                      "one": 1})
_WORD_NUMBER = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, (w for w in _WORD_NUMBERS if w != "one")),
                                                     key=len, reverse=True))
                          + r"|one(?=\s*(?:degrees?\b|°|%|m3/s|l/s)))\b(?!\s+(?:tenths?|fifths?|quarters?|thirds?)\b)")
# Litres (or liters) per second, however written, is l/s; cubic metres per second is m3/s.
_PER_SECOND = re.compile(r"\b(?:litres?|liters?|l)\s*(?:per|a|/)\s*(?:second|sec|s)\b|\blps\b")
_CUMECS = re.compile(r"\bcubic\s+met(?:re|er)s?\s*(?:per|a|/)\s*(?:second|sec|s)\b")


# A part of the year: the model runs a change over the whole year, so a line that names one says so (#590).
_SEASON = re.compile(
    r"\b(?:summers?|winters?|autumns?|springtime|monsoons?|(?:dry|wet|rainy|growing|melt|snowmelt)\s+seasons?|"
    r"seasons?|seasonal(?:ly)?|january|february|march|april|june|july|august|september|october|november|december|"
    r"(?:in|during|each|every|this|next|over)\s+(?:the\s+)?(?:spring|fall|may))\b", re.IGNORECASE)


def season_words(text: str) -> list[str]:
    """The words of ``text`` that name a part of the year ("summers", "in July", "the dry season"). A scenario is
    run over the whole year, so the line under the box says these were not applied rather than widening them in
    silence (#590: "summers get much hotter, maybe three degrees" ran as a year-round warming)."""
    return list(dict.fromkeys(re.sub(r"\s+", " ", m.group(0)).strip() for m in _SEASON.finditer(str(text or ""))))


def _normalise(text: str) -> str:
    t = str(text or "").lower()
    for a, b in (("−", "-"), ("–", "-"), ("m³/s", " m3/s"), ("m^3/s", " m3/s"), ("º", "°"), ("℃", "°c"),
                 ("percent", "%"), ("per cent", "%")):
        t = t.replace(a, b)
    t = _CUMECS.sub(" m3/s", t)
    t = _PER_SECOND.sub(" l/s", t)
    # "one and a half degrees" is 1.5 degrees (#590): the half goes with its number before either is read.
    t = re.sub(r"\b(" + _WORD_RE + r")\b", lambda m: _g(_word_value(m.group(1))) if m.group(1).endswith(
        "and a half") else m.group(1), t)
    t = re.sub(r"\b(\d+(?:\.\d+)?) and a half\b", lambda m: _g(float(m.group(1)) + 0.5), t)
    t = _WORD_NUMBER.sub(lambda m: str(_WORD_NUMBERS[re.sub(r"\s+", " ", m.group(0))]), t)
    return re.sub(r"\s+", " ", t).strip(" .!?")


def _clause(raw: str) -> tuple[str | None, float | None, str]:
    """One clause: (which parameter, its value, what was left unread)."""
    kind = None
    for k, pattern in _KINDS:
        if re.search(pattern, raw):
            kind = k
            break
    if kind is None:
        return None, None, raw
    value = None
    converted = False
    for words, v in _FRACTIONS.items():
        if re.search(rf"\b{words}\b", raw):
            value = v
            # "a half" names the same fraction as "half", but the token keeps the article.
            token = next((t for t in number_tokens(raw) if t["token"] in (words, f"a {words}")), None)
            if token and not token["share"] and token["unit"] in UNITS:
                dimension, factor = UNITS[token["unit"]]
                if dimension == _UNIT_OF_KEY[kind]:
                    value = token["value"] * factor
                    converted = True
            break
    sign_given = False
    if value is None:
        m = _NUMBER.search(raw)
        if m:
            value = float(m.group("n"))
            if m.group("sign"):
                sign_given = True
                value = -value if m.group("sign") == "-" else value
    if value is None:
        return kind, None, raw
    if kind == "demand_cms" and not converted and re.search(r"\bl/s\b", raw):
        value /= 1000.0
    if not sign_given and kind != "demand_cms":
        if re.search(_NEG, raw):
            value = -abs(value)
        elif not re.search(_POS, raw) and kind != "dt_c":
            return kind, None, raw   # "20% rain": more or less?
    left = re.sub(r"[+-]?\d+(?:\.\d+)?|%|°c?|m3/s|l/s|" + _NEG + "|" + _POS, " ", raw)
    for _k, pattern in _KINDS:
        left = re.sub(pattern, " ", left)
    for words in _FRACTIONS:
        left = left.replace(words, " ")
    unknown = [w for w in re.findall(r"[a-z]+", left) if w not in _FILLER]
    return kind, value, " ".join(unknown)


def parse_scenario(text: str) -> dict[str, Any]:
    """Read one line ("20% less rain and 2 degrees warmer", "take 5 m3/s for irrigation") into a scenario,
    with no model and no network.

    Returns ``{"text", "matched", "scenario", "said", "rules", "unknown", "errors", "season"}``: ``matched`` only
    when every clause was read and at least one parameter changed; ``said`` is :func:`scenario_label` of the
    result; ``season`` the words that name a part of the year (:func:`season_words`), which the run does not
    apply.
    """
    norm = _normalise(text)
    out: dict[str, Any] = {"text": str(text or "").strip(), "matched": False, "scenario": None, "said": "",
                           "rules": [], "unknown": [], "errors": [], "season": season_words(text)}
    if not norm:
        return out
    norm = re.sub(r"^(?:what if|what happens if|if|suppose|imagine)\s+", "", norm)
    parts = [p.strip() for p in re.split(r"\s*(?:;|,|\band\b|\bwith\b|\bplus\b(?=\s+\D)|&)\s*", norm) if p.strip()]
    found: dict[str, float] = {}
    for part in parts:
        kind, value, left = _clause(part)
        if kind is None or value is None:
            out["unknown"].append(part)
            continue
        found[kind] = value
        out["rules"].append(kind)
        if left:
            out["unknown"].append(left)
    if not found:
        return out
    checked, errors = check_scenario(found)
    out.update(scenario=checked, errors=errors, said=scenario_label(checked))
    changed = any(checked[k] for k in ("dp_pct", "dt_c", "demand_cms")) or bool(checked["dpet_pct"])
    out["matched"] = changed and not out["unknown"]
    return out


# ── the model path: the model parses, the code reads the numbers and converts (#590, round 3) ─────────

#: What a model may say a clause changes, and the parameter and sign each sets.
CHANGES: dict[str, tuple[str, int]] = {
    "less_rain": ("dp_pct", -1), "more_rain": ("dp_pct", 1), "warmer": ("dt_c", 1), "cooler": ("dt_c", -1),
    "less_evaporation": ("dpet_pct", -1), "more_evaporation": ("dpet_pct", 1), "take_water": ("demand_cms", 1),
}
#: The units a model may name, and what each converts to (code converts; a model never does): a change in rain or
#: evaporation in percent, a warming in degrees C, a demand in m3/s.
UNITS: dict[str, tuple[str, float]] = {
    "percent": ("pct", 1.0), "fraction": ("pct", 100.0),
    "degC": ("deg", 1.0), "K": ("deg", 1.0), "degF": ("deg", 5.0 / 9.0),
    "m3/s": ("flow", 1.0), "l/s": ("flow", 1e-3), "m3/day": ("flow", 1.0 / 86400.0),
    "ML/day": ("flow", 1000.0 / 86400.0), "mm": ("depth", 1.0), "none": ("", 1.0),
}
_UNIT_OF_KEY = {"dp_pct": "pct", "dpet_pct": "pct", "dt_c": "deg", "demand_cms": "flow"}
#: The parts of the year a model may name; the run is over the whole year, so any but "year" is said, not applied.
SEASONS = ("year", "winter", "spring", "summer", "autumn", "wet season", "dry season", "some months")

# What follows a number in the words, as the unit it is in. Degrees with no letter are degrees C.
_UNIT_AFTER: tuple[tuple[str, str], ...] = (
    ("percent", r"%|per ?cent\b|pct\b"),
    ("degF", r"°\s*f\b|deg(?:rees?)?\s*f(?:ahrenheit)?\b|fahrenheit\b"),
    ("K", r"kelvins?\b|k\b"),
    ("degC", r"°\s*c?|deg(?:rees?)?(?:\s*(?:c|celsius|centigrade)\b)?|celsius\b|centigrade\b|c\b"),
    ("m3/s", r"m3/s\b|m3 ?s-1\b|cumecs\b|cubic met(?:re|er)s?\s*(?:per|a|/)\s*(?:second|sec|s)\b"),
    ("l/s", r"l/s\b|lps\b|lit(?:re|er)s?\s*(?:per|a|an|/|each)\s*(?:second|sec|s)\b"),
    ("m3/day", r"m3/d(?:ay)?\b|cubic met(?:re|er)s?\s*(?:per|a|/)\s*day\b"),
    ("ML/day", r"ml/d(?:ay)?\b|megalit(?:re|er)s?\s*(?:per|a|/)\s*day\b"),
    ("mm", r"mm\b|millimet(?:re|er)s?\b"),
)
_SMALL = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
          "seventeen eighteen nineteen").split()
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
         "ninety": 90}
_SHARES = {"tenth": 0.1, "fifth": 0.2, "quarter": 0.25, "third": 1 / 3, "half": 0.5}
_WORD_RE = (r"(?:(?:a|one)\s+hundred|hundred|(?:" + "|".join(_TENS) + r")(?:[\s-](?:" + "|".join(_SMALL[1:10])
            + r"))?|" + "|".join(sorted(_SMALL, key=len, reverse=True)) + r")(?:\s+and\s+a\s+half)?")
_TOKEN = re.compile(
    r"(?P<digits>(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?![\d]))"
    r"|(?P<share>\b(?:a|one)\s+(?:tenth|fifth|quarter|third)\b|\b(?:a\s+)?half\b)"
    r"|(?P<word>\b" + _WORD_RE + r"\b)")


def _word_value(words: str) -> float | None:
    """A number in words as a value: "three" 3, "twenty-five" 25, "one and a half" 1.5, "a hundred" 100."""
    w = re.sub(r"[\s-]+", " ", str(words or "").lower()).strip()
    half = w.endswith(" and a half")
    w = w[: -len(" and a half")] if half else w
    if w in ("a hundred", "one hundred", "hundred"):
        v = 100.0
    elif w in _SMALL:
        v = float(_SMALL.index(w))
    else:
        parts = w.split(" ")
        if parts[0] not in _TENS:
            return None
        v = float(_TENS[parts[0]]) + (float(_SMALL.index(parts[1])) if len(parts) > 1 and parts[1] in _SMALL else 0)
    return v + (0.5 if half else 0.0)


def _token_value(token: str) -> tuple[float, bool] | None:
    """A number token's value and whether it is a share ("a fifth" is 0.2 of something), or None."""
    t = re.sub(r"\s+", " ", str(token or "").strip().lower())
    if re.fullmatch(r"\d+(?:,\d{3})*(?:\.\d+)?", t):
        return float(t.replace(",", "")), False
    share = re.fullmatch(r"(?:(?:a|one) )?(tenth|fifth|quarter|third|half)", t)
    if share:
        return _SHARES[share.group(1)], True
    v = _word_value(t)
    return (v, False) if v is not None else None


def _lower(text: str) -> str:
    t = str(text or "").lower()
    for a, b in (("−", "-"), ("–", "-"), ("m³/s", "m3/s"), ("m^3/s", "m3/s"), ("m³/d", "m3/d"), ("º", "°"),
                 ("℃", "°c"), ("℉", "°f")):
        t = t.replace(a, b)
    return t


def number_tokens(text: str) -> list[dict[str, Any]]:
    """The numbers in the reader's words, as written, with the unit the words give each (what follows it):
    ``[{"token", "value", "share", "unit"}]``. "7.5 l/s" is ``7.5`` in ``l/s``, "one and a half degrees" ``1.5``
    in ``degC``, "a fifth" the share ``0.2`` (unit ``fraction``). ``unit`` is None when the words give none."""
    low = _lower(text)
    out = []
    for m in _TOKEN.finditer(low):
        token = re.sub(r"\s+", " ", m.group(0)).strip()
        got = _token_value(token)
        if got is None:
            continue
        value, share = got
        unit = "fraction" if share else None
        rest = low[m.end():m.end() + 40]
        rest = re.sub(r"^\s*(?:more|less|fewer|of|extra|higher|lower|warmer|cooler)?\s*", "", rest) \
            if not re.match(r"\s*(?:%|°)", rest) else rest
        # "half a litre" is a quantity, while "half the rain" is a share.
        # Strip an article only when testing for an explicitly named unit.
        if share:
            rest = re.sub(r"^(?:a|an)\s+", "", rest)
        for name, pattern in _UNIT_AFTER:
            if re.match(r"\s*(?:" + pattern + ")", rest):
                unit, share = name, False
                break
        out.append({"token": token, "value": value, "share": share, "unit": unit})
    return out


def scenario_schema(text: str = "") -> dict[str, Any]:
    """The JSON Schema a model's reading is held to: up to four clauses, each a change kind (:data:`CHANGES`),
    the number as written in the reader's words (an enum of ``text``'s own number tokens, or "none"), a unit
    (:data:`UNITS`) and a part of the year (:data:`SEASONS`). Nothing in it is free text."""
    from aquascope.map_pick import choice

    # Offer a digit alias only if the matcher can read it back as the same value. _g is for display and
    # can lose precision (a third becomes 0.333); conversion must use the original token's value.
    tokens = []
    for t in number_tokens(text):
        tokens.append(t["token"])
        if not t["share"]:
            alias = _g(t["value"])
            parsed = _token_value(alias)
            if parsed and math.isclose(t["value"], parsed[0], rel_tol=1e-9):
                tokens.append(alias)
    number = choice([*tokens, "none"]) if text else {"type": "string", "maxLength": 40}
    return {"type": "object", "required": ["clauses"], "additionalProperties": False, "properties": {
        "clauses": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "required": ["change", "number", "unit", "season"], "additionalProperties": False,
            "properties": {"change": choice(CHANGES), "number": number, "unit": choice(UNITS),
                           "season": choice(SEASONS)}}}}}


def scenario_prompt(context: str = "", text: str = "") -> dict[str, Any]:
    """The system prompt and schema for a model that parses a what-if line (``text``, when given, sets the
    number enum to the line's own number tokens). The model parses; the code reads each number from the words,
    converts its unit and checks its direction (:func:`parse_scenario_reply`)."""
    shots = (
        ("20% less rain and 2 degrees warmer",
         [("less_rain", "20", "percent", "year"), ("warmer", "2", "degC", "year")]),
        ("take 300 litres per second for irrigation", [("take_water", "300", "l/s", "year")]),
        ("summers a fifth drier", [("less_rain", "a fifth", "fraction", "summer")]),
    )
    lines = [
        "You parse a 'what if' line about a river basin into clauses for a rainfall-runoff model. You do not "
        "work anything out: code reads each number from the line and converts its unit.",
        'Reply with ONE JSON object and nothing else: {"clauses": [{"change", "number", "unit", "season"}]}.',
        f"change: one of {', '.join(CHANGES)}. number: the number exactly as the line writes it (\"7.5\", "
        "\"twenty\", \"one and a half\", \"a fifth\"), or \"none\" when the line gives none. unit: the unit the line "
        f"gives that number in, one of {', '.join(UNITS)} (\"a fifth\" is a fraction). season: the part of the "
        f"year the line names, one of {', '.join(SEASONS)} (\"year\" when it names none).",
        "One clause per change the line asks for. If the line is not a what-if about water, reply with no clauses.",
        "",
    ]
    for line, clauses in shots:
        reply = {"clauses": [dict(zip(("change", "number", "unit", "season"), c)) for c in clauses]}
        lines.append(f"Line: {line}\nReply: {json.dumps(reply)}")
    if text:
        toks = [t["token"] for t in number_tokens(text)]
        lines += ["", "The numbers in this line: " + (", ".join(f'"{t}"' for t in toks) if toks else "none") + "."]
    system = "\n".join(lines)
    if context:
        system += f"\n\nWhere the reader is: {context}"
    return {"system": system, "schema": scenario_schema(text)}


def _season_said(season: str, words: list[str]) -> bool:
    stems = {"wet season": ("wet", "rainy", "monsoon"), "dry season": ("dry",), "some months": (
        "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
        "november", "december", "month", "season")}.get(season, (season.split(" ")[0][:5],))
    return any(any(s in w.lower() for s in stems) for w in words)


def parse_scenario_reply(reply: Any, text: str = "") -> dict[str, Any]:
    """A model's parse of the reader's ``text`` (its reply, text or parsed JSON) as a checked scenario:
    ``{"scenario", "errors", "said", "season", "read"}``.

    The reply is held to :func:`scenario_schema` first: a change, number, unit or season outside its enums (a
    number that is not one of the words' own number tokens) refuses the reply whole. Then each clause, read by
    code: its number must be a number token of the words, used no more often than the words say it, in the unit
    the words give it (the model's unit must agree: "draws 7.5 l/s" is 0.0075 m³/s, never 7.5); code converts
    the unit (percent or a share for rain and evaporation, degrees C, K or F for warming, m3/s, l/s, m3/day or
    ML/day for a demand), sets the sign from the change, refuses a direction the keyless rules read the other
    way round ("20% more rain" is never -20) and holds the value to :data:`LIMITS`. A refused clause is said in
    ``errors`` and left out. ``season`` is the words that name a part of the year (:func:`season_words`): the run
    is over the whole year, so the line says those were not applied. ``read`` says each clause as code read it.
    """
    from aquascope.map_pick import pick

    if not str(text or "").strip():
        return {"scenario": None, "errors": ["no words to read"], "said": "", "season": [], "read": []}
    data, errors = pick(reply, scenario_schema(text))
    if errors:
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        return {"scenario": None, "errors": [f"the reply was refused: {errors[0]}{more}"], "said": "",
                "season": season_words(text), "read": []}
    tokens = number_tokens(text)
    left = list(range(len(tokens)))
    rules = parse_scenario(text).get("scenario") or {}
    found: dict[str, float] = {}
    errs: list[str] = []
    read: list[str] = []
    seasons = season_words(text)
    for c in data["clauses"]:
        key, sign = CHANGES[c["change"]]
        words = c["change"].replace("_", " ")
        if c["number"] == "none":
            errs.append(f"{words}: the words give no number for it")
            continue
        want = _token_value(c["number"])
        hit = next((i for i in left if tokens[i]["token"] == c["number"]), None)
        if hit is None and want:
            hit = next((i for i in left if math.isclose(tokens[i]["value"], want[0], rel_tol=1e-9)), None)
        if hit is None:
            errs.append(f"{words} {c['number']}: that number is not in the words as often as it is used")
            continue
        tok = tokens[hit]
        said_unit, given = tok["unit"], c["unit"]
        same = given == said_unit or (said_unit == "degC" and given == "K")
        if said_unit is None:
            errs.append(f"{words} {tok['token']}: the words give no unit for it")
            continue
        if not same:
            errs.append(f"{words} {tok['token']}: the words say {said_unit}, not {given}")
            continue
        dim, factor = UNITS[given]
        if dim != _UNIT_OF_KEY[key]:
            why = ("a change in mm cannot be run: the model takes rain as a percent" if dim == "depth"
                   else f"{given} is not a unit for {words}")
            errs.append(f"{words} {tok['token']} {given}: {why}")
            continue
        if key in found:
            errs.append(f"{words}: said twice")
            continue
        value = sign * abs(tok["value"]) * factor
        ruled = _num(rules.get(key))
        if ruled and (ruled > 0) != (value > 0):
            errs.append(f"{words} {tok['token']}: the words say {scenario_label({key: ruled})}")
            continue
        lo, hi = LIMITS[key]
        if not lo <= value <= hi:
            errs.append(f"{words} {tok['token']}: {_g(value)} is outside {lo:g} to {hi:g}")
            continue
        if c["season"] != "year" and not _season_said(c["season"], seasons):
            errs.append(f"{words}: the words name no {c['season']}, so none is said")
        left.remove(hit)
        found[key] = round(value, 6)
        read.append(f"{tok['token']} {given}" if given not in ("fraction", "none") else tok["token"])
    if not found:
        return {"scenario": None, "errors": errs or ["no change was read from the line"], "said": "",
                "season": seasons, "read": read}
    sc, more = check_scenario(found)
    return {"scenario": sc, "errors": errs + more, "said": scenario_label(sc), "season": seasons, "read": read}


def model_scenario(text: str, *, provider: str | None = None, model: str | None = None, api_key: str | None = None,
                   base_url: str | None = None, context: str = "", client: Any = None) -> dict[str, Any]:
    """Ask the reader's own model to parse the line, and read its reply (:func:`parse_scenario_reply`). Raises
    with no key and no client."""
    from aquascope.ai_engine.analyst import resolve_llm
    from aquascope.ai_engine.llm_transport import make_client

    if client is None:
        cfg = resolve_llm(provider, model, api_key, base_url)
        client = make_client(cfg["api_key"], cfg["base_url"], provider=cfg["provider"])
        model, provider = cfg["model"], cfg["provider"]
    prompt = scenario_prompt(context, text)
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=300,
        messages=[{"role": "system", "content": prompt["system"]}, {"role": "user", "content": str(text)}],
    )
    out = parse_scenario_reply(resp.choices[0].message.content or "", text)
    out["model"] = f"{model} via {provider}"
    return out


# ── the basin and the way down ───────────────────────────────────────────────


def _too_large(basin_km2: float) -> str | None:
    """Why a basin of ``basin_km2`` is too large for :func:`whatif`, or None (:data:`MAX_BASIN_KM2`)."""
    if not basin_km2 > MAX_BASIN_KM2:
        return None
    return (f"{basin_km2:,.0f} km2 drain to this reach: one ERA5 cell at the centre cannot stand for the weather of "
            f"a basin that size (the limit here is {MAX_BASIN_KM2:,.0f} km2). Pick a reach on a tributary or further "
            "upstream.")


def _basin_check(river_id: int | str, lat: float | None = None, lon: float | None = None) -> dict[str, Any]:
    """The area that drains to a reach and whether :func:`whatif` can run there, before any calibration:
    ``{"river_id", "area_km2", "max_km2", "ok", "error"}``. The Explorer asks this when "What if..." opens, so a
    reach over the limit says why at once instead of failing on Run (#590). It reads the same routing tables."""
    from aquascope import rivers

    rid = rivers._river_id(river_id)
    net = rivers._network_for(rid, lat, lon)
    i = net.index(rid)
    if i is None:
        raise LookupError(f"reach {rid} is not in its routing tables")
    km2 = round(_drainage_m2(net, i) / 1e6, 2)
    why = _too_large(km2)
    return {"river_id": rid, "area_km2": km2, "max_km2": MAX_BASIN_KM2, "ok": why is None, "error": why}


def _drainage_m2(net: Any, i: int) -> float:
    """The drainage area (m2) of reach index ``i``: its unit catchment and everything upstream."""
    import numpy as np

    from aquascope.rivers import _upstream_tree

    order, _parent, _starts = _upstream_tree(net, i)
    a = net.area_m2[order].astype(float)
    return float(np.nansum(a))


def downstream_shares(net: Any, i: int, *, confluences: int = CONFLUENCES, max_reaches: int = MAX_PATH,
                      min_share: float = MIN_SHARE) -> dict[str, Any]:
    """Walk down from reach index ``i`` and give each reach the share of its drainage area the basin of ``i`` is.

    The walk adds each tributary that joins (its whole drainage area) and stops at the reach where the
    ``confluences``-th major tributary joins (one at least :data:`MAJOR_SHARE` of the main stem's area above it),
    at the outlet, after ``max_reaches``, or once the share falls under ``min_share``. Returns ``{"basin_km2",
    "reaches": [{"index", "river_id", "area_km2", "share", "confluence"}], "stopped"}``, the basin's own reach
    first with share 1.
    """
    basin = _drainage_m2(net, i)
    reaches = [{"index": int(i), "river_id": int(net.ids[i]), "area_km2": round(basin / 1e6, 2), "share": 1.0,
                "confluence": None}]
    acc = basin
    cur = i
    majors = 0
    seen = {int(i)}
    stopped = "outlet"
    while True:
        if len(reaches) >= max_reaches:
            stopped = "max_reaches"
            break
        nxt = net.index(int(net.ds[cur]))
        if nxt is None or nxt in seen:
            stopped = "outlet"
            break
        seen.add(nxt)
        side = [j for j in net.children(nxt) if j != cur]
        side_m2 = sum(_drainage_m2(net, j) for j in side)
        unit = float(net.area_m2[nxt]) if math.isfinite(float(net.area_m2[nxt])) else 0.0
        major = side_m2 >= MAJOR_SHARE * acc
        acc = acc + side_m2 + unit
        share = basin / acc if acc > 0 else 0.0
        conf = None
        if major:
            majors += 1
            conf = {"n": majors, "tributary_km2": round(side_m2 / 1e6, 1)}
        reaches.append({"index": int(nxt), "river_id": int(net.ids[nxt]), "area_km2": round(acc / 1e6, 2),
                        "share": round(share, 4), "confluence": conf})
        cur = nxt
        if majors >= confluences:
            stopped = "confluences"
            break
        if share < min_share:
            stopped = "faded"
            break
    return {"basin_km2": round(basin / 1e6, 2), "reaches": reaches, "stopped": stopped, "majors": majors}


def _centroid(net: Any, i: int) -> tuple[float, float, list[float]] | None:
    """The area-weighted centre of the basin upstream of reach index ``i`` (lat, lon) and the box around its unit
    catchments' centres ([west, south, east, north]), from the routing tables."""
    import numpy as np

    from aquascope.rivers import _upstream_tree

    order, _p, _s = _upstream_tree(net, i)
    a = net.area_m2[order].astype(float)
    lon, lat = net.lon[order].astype(float), net.lat[order].astype(float)
    ok = np.isfinite(a) & np.isfinite(lon) & np.isfinite(lat) & (a > 0)
    if not ok.any():
        return None
    w = a[ok]
    box = [round(float(lon[ok].min()), 4), round(float(lat[ok].min()), 4), round(float(lon[ok].max()), 4),
           round(float(lat[ok].max()), 4)]
    return float(np.average(lat[ok], weights=w)), float(np.average(lon[ok], weights=w)), box


# ── the run ──────────────────────────────────────────────────────────────────


def _scaled(v: float | None, share: float) -> float | None:
    return None if v is None else round(float(v) * share, 2)


def _span(r: list[float] | None, share: float) -> list[float] | None:
    if not r or r[0] is None or r[1] is None:
        return None
    a, b = float(r[0]) * share, float(r[1]) * share
    return [round(min(a, b), 2), round(max(a, b), 2)]


def _pct_words(v: float | None, r: list[float] | None) -> str:
    if v is None:
        return "n/a"
    s = f"{v:+.0f}%"
    if r:
        s += f" ({r[0]:+.0f} to {r[1]:+.0f}%)"
    return s


def whatif(
    river_id: int | str | None = None,
    *,
    lat: float | None = None,
    lon: float | None = None,
    source: str | None = None,
    station_id: str | None = None,
    area_km2: float | None = None,
    dp_pct: float | None = None,
    dt_c: float | None = None,
    dpet_pct: float | None = None,
    demand_cms: float | None = None,
    text: str | None = None,
    flood_rp: float = FLOOD_RP,
    members: int = MEMBERS,
    confluences: int = CONFLUENCES,
    years: int | None = None,
    maxiter: int = 25,
    snow: str | bool = "auto",
    basin_ids: int = BASIN_IDS,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """A what-if scenario for the basin draining to a river reach, and its change carried down the river.

    Give the reach (``river_id``, or ``lat``/``lon`` to snap), or a gauge (``source`` and ``station_id`` with its
    ``lat``/``lon``, and ``area_km2`` when known). The scenario is ``dp_pct`` (rain, %), ``dt_c`` (warming, degrees
    C), ``dpet_pct`` (evaporation, %; by default 3 % per degree of warming) and ``demand_cms`` (taken from the
    river, m3/s), or ``text`` read by :func:`parse_scenario`.

    GR4J is calibrated on the gauge's record, or on the reach's GEOGLOWS v2 simulated flow when there is no gauge,
    with ERA5 forcing at the basin's area-weighted centre (:func:`aquascope.advanced.catchment_model`). The
    answer's ``outlet`` holds the mean flow, the low flow (Q95) and the ``flood_rp``-year flood before and after,
    each change with its ``range`` from the ``members`` parameter sets that did as well on the validation years;
    ``reaches`` holds each reach downstream to the ``confluences``-th major confluence with its share of the
    basin's water and the changes scaled by it (:data:`SIMPLIFICATION`); ``basin.ids`` lists the basin's reaches
    for the map. Every number is computed here; ``verdict`` says them in one paragraph. ``params`` (the
    ``model.params`` of an earlier answer for the same reach) skips the calibration, so a second scenario on the
    same basin is quick; the fit is seeded, so the numbers are the ones a fresh run gives.
    """
    from aquascope import advanced, rivers

    out: dict[str, Any] = {"ok": False, "modelled": True, "attribution": rivers.ATTRIBUTION,
                           "licence": {"flow": "CC BY 4.0 (GEOGLOWS v2)", "network": "CC BY-SA 4.0 (TDX-Hydro)",
                                       "forcing": "CC BY 4.0 (ERA5 via Open-Meteo)"},
                           "simplification": SIMPLIFICATION, "notes": []}
    # the scenario
    given = {"dp_pct": dp_pct, "dt_c": dt_c, "dpet_pct": dpet_pct, "demand_cms": demand_cms}
    if text and not any(_num(v) for v in given.values()):
        parsed = parse_scenario(text)
        if not parsed["scenario"]:
            out["error"] = (f"could not read a scenario from {text!r}; say for example '20% less rain and 2 degrees "
                            "warmer', or give the numbers")
            return out
        given = parsed["scenario"]
        out["read"] = {"text": parsed["text"], "by": "rules", "unknown": parsed["unknown"]}
    sc, errors = check_scenario(given)
    out["notes"] += errors
    if not any(sc[k] for k in ("dp_pct", "dt_c", "demand_cms")) and not sc["dpet_pct"]:
        out["error"] = "nothing to change: give a rain change, a warming, an evaporation change or a demand"
        return out
    sc["label"] = scenario_label(sc)
    out["scenario"] = sc
    # the reach and its basin
    gauge = bool(source and station_id and source != advanced.GEOGLOWS_SOURCE)
    snap = None
    if river_id is None:
        if lat is None or lon is None:
            out["error"] = "give a river_id, or lat and lon to snap to the nearest reach"
            return out
        snap = rivers.snap_to_river(float(lat), float(lon), area_km2=area_km2 if gauge else None)
        if not snap.get("snapped"):
            out["error"] = snap.get("message") or "no river reach here"
            out["snap"] = snap
            return out
        river_id = snap["river_id"]
    try:
        rid = rivers._river_id(river_id)
        net = rivers._network_for(rid, lat, lon)
    except (ValueError, LookupError) as exc:
        out["error"] = str(exc)
        return out
    i = net.index(rid)
    assert i is not None
    centre = _centroid(net, i)
    down = downstream_shares(net, i, confluences=max(1, int(confluences)))
    basin_km2 = float(down["basin_km2"])
    out["reach"] = {"river_id": rid, "lat": _round(snap["snap_lat"] if snap else lat, 5),
                    "lon": _round(snap["snap_lon"] if snap else lon, 5)}
    if snap:
        out["snap"] = {k: snap.get(k) for k in ("distance_m", "strahler_order", "choice", "message")}
    too_large = _too_large(basin_km2)
    if too_large:
        out["error"] = too_large
        out["basin"] = {"area_km2": basin_km2}
        return out
    if centre is None:
        out["error"] = "the routing tables give no area for this basin"
        return out
    ups = rivers.upstream_ids(rid, max_n=max(1, int(basin_ids)), lat=lat, lon=lon)
    out["basin"] = {"area_km2": basin_km2, "centre": [round(centre[0], 4), round(centre[1], 4)], "bbox": centre[2],
                    "n_reaches": ups["n_upstream"], "ids": ups["ids"], "truncated": ups["truncated"],
                    "min_area_km2": ups["min_area_km2"]}
    # the model, without and with the scenario
    model_area = float(area_km2) if gauge and area_km2 else basin_km2
    cm = advanced.catchment_model(
        source if gauge else advanced.GEOGLOWS_SOURCE, str(station_id) if gauge else str(rid),
        centre[0], centre[1], area_km2=model_area, years=int(years or advanced.MODEL_YEARS),
        scenarios=[{"label": sc["label"], "dp_pct": sc["dp_pct"], "dt_c": sc["dt_c"], "dpet_pct": sc["dpet_pct"],
                    "demand_cms": sc["demand_cms"]}],
        maxiter=int(maxiter), snow=snow, flood_rp=float(flood_rp), members=int(members), params=params)
    if cm.get("error"):
        out["error"] = f"the catchment model could not run: {cm['error']}"
        out["model"] = {"error": cm["error"]}
        return out
    run = cm["scenarios"][0]
    base = cm.get("baseline_simulated") or {}
    span = run.get("range") or {}
    out["model"] = {
        "calibrated_on": "gauge" if gauge else "modelled",
        "source": source if gauge else advanced.GEOGLOWS_SOURCE, "station_id": str(station_id) if gauge else str(rid),
        "area_km2": model_area, "forcing_at": [round(centre[0], 4), round(centre[1], 4)],
        "calibration": cm.get("calibration"), "validation": cm.get("validation"), "skill_grade": cm.get("skill_grade"),
        "band": cm.get("band"), "snow": cm.get("snow"), "params": cm.get("params"), "ensemble": cm.get("ensemble"),
        "dpet_pct": run.get("dpet_pct"), "verdict": cm.get("verdict"), "record": cm.get("record"),
    }
    rp = float(cm.get("flood_rp") or flood_rp)
    out["outlet"] = {
        "river_id": rid, "area_km2": basin_km2,
        "mean": {"label": "mean flow", "before": base.get("mean"), "after": run.get("mean_flow"),
                 "change_pct": run.get("mean_flow_change_pct"), "range": span.get("mean_flow_change_pct"),
                 "unit": "m3/s"},
        "low": {"label": "low flow (Q95, exceeded 95% of days)", "before": base.get("q95"), "after": run.get("q95"),
                "change_pct": run.get("q95_change_pct"), "range": span.get("q95_change_pct"), "unit": "m3/s"},
        "flood": {"label": f"{rp:g}-year flood", "rp": rp, "before": base.get("flood"), "after": run.get("flood"),
                  "change_pct": run.get("flood_change_pct"), "range": span.get("flood_change_pct"), "unit": "m3/s"},
    }
    o = out["outlet"]
    reaches = []
    for r in down["reaches"]:
        s = float(r["share"])
        j = r["index"]
        lo_, la_ = float(net.lon[j]), float(net.lat[j])
        reaches.append({
            "river_id": r["river_id"], "area_km2": r["area_km2"], "share": r["share"],
            "lat": round(la_, 4) if math.isfinite(la_) else None, "lon": round(lo_, 4) if math.isfinite(lo_) else None,
            "mean_change_pct": _scaled(o["mean"]["change_pct"], s), "mean_range": _span(o["mean"]["range"], s),
            "q95_change_pct": _scaled(o["low"]["change_pct"], s), "q95_range": _span(o["low"]["range"], s),
            "flood_change_pct": _scaled(o["flood"]["change_pct"], s), "flood_range": _span(o["flood"]["range"], s),
            "confluence": r["confluence"],
        })
    out["reaches"] = reaches
    out["downstream"] = {"n_reaches": len(reaches), "stopped": down["stopped"], "major_confluences": down["majors"],
                         "rule": (f"to the {int(confluences)} next major confluences (a tributary at least "
                                  f"{MAJOR_SHARE:.0%} of the main stem's area above it), the outlet, or where the "
                                  f"basin's share falls under {MIN_SHARE:.0%}")}
    out["methods"] = [*(cm.get("methods") or []), AREA_METHOD]
    out["notes"] += list(cm.get("notes") or [])
    ens = cm.get("ensemble") or {}
    if ens.get("note"):
        out["notes"].append(ens["note"])
    out["verdict"] = _verdict(out, run)
    out["ok"] = True
    return _clean(out)


def _round(v: Any, nd: int) -> float | None:
    x = _num(v)
    return None if x is None else round(x, nd)


def _clean(x: Any) -> Any:
    from aquascope.advanced import _clean as clean

    return clean(x)


def _verdict(out: dict[str, Any], run: dict[str, Any]) -> str:
    """One paragraph from the numbers: what the scenario does at the outlet and how it fades downstream."""
    sc, o, m = out["scenario"], out["outlet"], out["model"]
    who = "the gauge's record" if m["calibrated_on"] == "gauge" else "the reach's GEOGLOWS simulated flow"
    kv = (m.get("validation") or {}).get("kge")
    head = (f"With {sc['label']}, GR4J (calibrated on {who}, validation KGE "
            f"{kv:.2f}, {m.get('skill_grade') or 'ungraded'}) gives at river reach {o['river_id']} "
            f"({o['area_km2']:,.0f} km2): mean flow {_pct_words(o['mean']['change_pct'], o['mean']['range'])}, "
            f"low flow (Q95) {_pct_words(o['low']['change_pct'], o['low']['range'])}, "
            f"{o['flood']['label']} {_pct_words(o['flood']['change_pct'], o['flood']['range'])}." if kv is not None
            else f"With {sc['label']}, at river reach {o['river_id']}: mean flow "
                 f"{_pct_words(o['mean']['change_pct'], o['mean']['range'])}.")
    conf = [r for r in out["reaches"] if r.get("confluence")]
    if conf:
        bits = [f"{r['mean_change_pct']:+.0f}% after confluence {r['confluence']['n']} "
                f"({r['confluence']['tributary_km2']:,.0f} km2 joins)"
                for r in conf if r["mean_change_pct"] is not None]
        if bits:
            head += " Downstream the mean-flow change fades as other rivers join: " + "; ".join(bits) + "."
    elif len(out["reaches"]) > 1:
        last = out["reaches"][-1]
        if last["mean_change_pct"] is not None:
            head += (f" {len(out['reaches']) - 1} reaches downstream, the mean-flow change is "
                     f"{last['mean_change_pct']:+.0f}% where the walk stops.")
    return head
