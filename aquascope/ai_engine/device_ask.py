"""Ask on the reader's device (#590): what a small model may choose, and the sentences the package writes.

The Explorer's on-device Ask runs Chrome's built-in model (or WebLLM) through a short loop. Since round 3 of
#590 the model picks and the package writes:

* **The model decides** which tool to run next, with which gauge id, point or search phrase, and then which
  facts answer the question and in what order, by key, plus a few connecting sentences by id.
* **The model cannot** write anything the reader sees: no number, date, id, unit, place or claim. Each step's
  reply is held to a schema of enums built here (:func:`step_schema`): the tools still open, the gauge ids,
  sources and search phrases the reader's words or an earlier result hold, the keys of :func:`fact_table`.
  :func:`check_call` refuses a call outside them and does not run a repeat; :func:`render_answer` writes the
  answer from the keys, each sentence naming its gauge, label, value, unit and method, and refuses anything else
  before display, so the page shows the facts with one short line saying why.

Why: three checks on the device held the model's free text to the results with patterns, and each round the
model found a new way past them ("significant trend in annual maxima with a p = 0.0689" where the test says
not significant) while the patterns refused right answers ("100-year" read as a gauge id).

:func:`compact_result` is what a tool result is cut down to: the record, the statistics, the flow-duration
points, the return levels and the trend, and no series. :func:`lock_answer`, the free-text lock of rounds 1
and 2, stays as a second line of defence on the package's own sentences.

Internal: only the Explorer's worker imports this module. The reader's own key runs the full Analyst
(:mod:`aquascope.ai_engine.analyst`), not this loop.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from aquascope.ai_engine.verify import (
    _COMPASS,
    _CONVENTIONS,
    _DATE,
    _EXPONENT_UNIT,
    _SQUARED_UNIT,
    _close,
    _numbers,
    normalise,
)

#: Tools whose result is one station's (or one model cell's) analysis.
_ANALYSIS_TOOLS = ("analyze_station", "flood_frequency")

#: Keys that never help a small model answer: citations, licences, hashes.
_DROP_KEYS = frozenset({"methods", "citation", "license", "attribution", "url", "data_snapshot", "archive_revision",
                        "software_revision", "eligibility", "sampling", "series", "series_downsampled",
                        "annual_max", "annual_max_excluded", "requested", "params"})

#: The most a compact result may weigh, as JSON; past it, lists are cut harder.
MAX_CHARS = 4000

_P_SIGNIFICANT = 0.05


def _round(x: Any, digits: int = 4) -> Any:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return x
    if x != x or x in (float("inf"), float("-inf")):
        return None
    return round(float(x), digits) if isinstance(x, float) else x


def _short(text: Any, limit: int) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    s = re.sub(r"\s+", " ", text).strip()
    return s if len(s) <= limit else s[: limit - 3].rstrip() + "..."


def _shrink(value: Any, *, depth: int = 0, list_limit: int = 8) -> Any:
    """A generic compact copy: no citations, no long numeric series, short strings, a few levels deep."""
    if depth > 4:
        return None
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in _DROP_KEYS:
                continue
            small = _shrink(v, depth=depth + 1, list_limit=list_limit)
            if small is not None and small != {} and small != []:
                out[k] = small
        return out
    if isinstance(value, (list, tuple)):
        if len(value) > list_limit and all(isinstance(v, (int, float)) or v is None for v in value):
            return None   # a series: the statistics carry what the answer needs
        return [s for s in (_shrink(v, depth=depth + 1, list_limit=list_limit) for v in value[:list_limit])
                if s is not None]
    if isinstance(value, str):
        return _short(value, 200)
    return _round(value)


def _trend(block: Any, unit: str | None) -> dict[str, Any] | None:
    """A Mann-Kendall block with its reading in words, which the model is asked to copy."""
    if not isinstance(block, dict) or block.get("p_value") is None:
        return None
    p = float(block["p_value"])
    slope = block.get("sens_slope_per_year")
    out: dict[str, Any] = {k: _round(block[k]) for k in ("on", "p_value", "sens_slope_per_year", "n_years")
                           if block.get(k) is not None}
    out["significant"] = p < _P_SIGNIFICANT
    on = block.get("on") or "the series"
    slope_txt = ""
    if isinstance(slope, (int, float)):
        slope_txt = f"; Sen's slope {slope:+.4g}{' ' + unit if unit else ''} per year"
    if p < _P_SIGNIFICANT:
        direction = block.get("trend") if block.get("trend") in ("increasing", "decreasing") else (
            "increasing" if isinstance(slope, (int, float)) and slope > 0 else "decreasing")
        out["direction"] = direction
        out["reading"] = f"a significant {direction} trend in the {on} (Mann-Kendall p = {p:.3g}{slope_txt})"
    else:
        out["reading"] = f"no significant trend in the {on} (Mann-Kendall p = {p:.3g}{slope_txt})"
    return out


def _return_levels(ffa: Any) -> dict[str, Any] | None:
    if not isinstance(ffa, dict):
        return None
    fits = ffa.get("fits") if isinstance(ffa.get("fits"), dict) else {}
    gev = (fits.get("gev_lmoments") or {}).get("q_by_T") or {}
    lp3 = fits.get("lp3") or {}
    lp3_q = lp3.get("q_by_T") or {}
    periods = ffa.get("return_periods") or sorted({int(float(t)) for t in [*gev, *lp3_q]})
    ci = lp3.get("ci") if isinstance(lp3.get("ci"), list) else []
    rows = []
    for i, t in enumerate(periods):
        key = str(int(t)) if float(t).is_integer() else str(t)
        row: dict[str, Any] = {"T_years": _round(t)}
        if gev.get(key) is not None:
            row["gev"] = _round(gev[key], 2)
        if lp3_q.get(key) is not None:
            row["lp3"] = _round(lp3_q[key], 2)
        if i < len(ci) and isinstance(ci[i], (list, tuple)) and len(ci[i]) == 2:
            row["lp3_90ci"] = [_round(ci[i][0], 2), _round(ci[i][1], 2)]
        if len(row) > 1:
            rows.append(row)
    out: dict[str, Any] = {}
    if ffa.get("n_years") is not None:
        out["n_years"] = ffa["n_years"]
    if rows:
        out["return_levels"] = rows
    rec = ffa.get("record_max")
    if isinstance(rec, dict) and rec.get("value") is not None:
        out["record_max"] = {k: _round(rec[k], 2) for k in ("value", "year") if rec.get(k) is not None}
    return out or None


def compact_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """One analysis (analyze_station, flood_frequency, a GloFAS cell): record, statistics, FDC, floods, trend."""
    if not isinstance(payload, dict):
        return {}
    if payload.get("error"):
        return {"error": _short(str(payload["error"]), 300)}
    unit = payload.get("unit")
    out: dict[str, Any] = {k: _round(payload[k]) for k in (
        "source", "station_id", "name", "river", "agency", "variable", "unit", "start", "end", "years", "n", "modelled",
        "comparable") if payload.get(k) is not None}
    stats = payload.get("stats")
    if isinstance(stats, dict):
        # min and max are of the daily values, not of annual ones; the key says so.
        out["daily_values"] = {k: _round(v) for k, v in stats.items() if isinstance(v, (int, float))}
    fdc = payload.get("fdc")
    if isinstance(fdc, dict):
        pts = {k: _round(fdc[k]) for k in ("q95", "q50", "q10") if fdc.get(k) is not None}
        if pts:
            out["flow_duration"] = {**pts, "note": "q95 is exceeded 95 % of the time, q10 10 % of the time"}
    floods = _return_levels(payload.get("ffa"))
    if floods:
        # Two fits, two numbers per period: the model is told which is which, and that the page's own figures
        # are the GEV (#590: a 100-year answer gave the LP3 value and named no method).
        floods["method"] = ("gev: GEV fitted by L-moments, the method the page's figures use; lp3: Log-Pearson III, "
                            "with lp3_90ci its 90 % interval. Name the method with the number.")
        out["flood_frequency"] = floods
    trend = _trend(payload.get("trend"), unit)
    if trend:
        out["trend_annual_mean"] = trend
    ffa = payload.get("ffa") if isinstance(payload.get("ffa"), dict) else {}
    amax = _trend(ffa.get("amax_trend"), unit)
    if amax:
        out["trend_annual_max"] = amax
    for key in ("fetch_note", "note"):
        if _short(payload.get(key), 200):
            out[key] = _short(payload.get(key), 200)
    notes = [_short(n, 160) for n in (payload.get("notes") or []) if isinstance(n, str)]
    if any(notes):
        out["notes"] = [n for n in notes if n][:3]
    return out


def _compact_stations(payload: dict[str, Any]) -> dict[str, Any]:
    fields = ("source", "station_id", "name", "river", "country", "latitude", "longitude", "period_start",
              "period_end")
    rows = []
    for st in (payload.get("stations") or [])[:10]:
        if isinstance(st, dict):
            rows.append({k: _round(st[k]) for k in fields if st.get(k) not in (None, "")})
    return {"n_returned": payload.get("n_returned", len(rows)), "stations": rows}


def _compact_anywhere(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: _round(payload[k]) for k in ("latitude", "longitude", "start", "end", "years")
                           if payload.get(k) is not None}
    climate = payload.get("climate")
    if isinstance(climate, dict):
        out["climate"] = {k: _round(climate[k]) for k in (
            "source", "precipitation_mm_per_year", "et0_mm_per_year", "temperature_mean_c", "aridity_index",
            "aridity_class", "wettest_day_mm") if climate.get(k) is not None}
    glofas = payload.get("glofas")
    if isinstance(glofas, dict):
        out["glofas"] = compact_analysis(glofas)
    notes = [_short(n, 160) for n in (payload.get("notes") or []) if isinstance(n, str)]
    if any(notes):
        out["notes"] = [n for n in notes if n][:3]
    return out


def compact_result(name: str, payload: Any) -> dict[str, Any]:
    """What the on-device model is shown of a tool result: the facts an answer needs, no series, no citations.

    Small enough for a small model's context (:data:`MAX_CHARS` as JSON) and never cut in the middle: a
    result that is still too big loses list items, not its trend.
    """
    if not isinstance(payload, dict):
        return {"value": _shrink(payload)}
    if payload.get("error"):
        return {"error": _short(str(payload["error"]), 300)}
    if name in _ANALYSIS_TOOLS:
        out = compact_analysis(payload)
    elif name == "find_stations":
        out = _compact_stations(payload)
    elif name == "anywhere":
        out = _compact_anywhere(payload)
    else:
        out = _shrink(payload) or {}
    limit = 8
    while len(json.dumps(out, default=str)) > MAX_CHARS and limit > 1:
        limit = max(1, limit // 2)
        out = _shrink(out, list_limit=limit) or {}
    return out


# ── the facts the page shows when no answer can be written ──────────────────


def _fmt(x: Any) -> str:
    return display(x)


def facts(name: str, compact: Any) -> list[str]:
    """Plain lines from one compact result, written here: what the page shows when the model picks nothing
    it may (the summary keys of :func:`fact_table`)."""
    if not isinstance(compact, dict):
        return []
    if compact.get("error"):
        return [f"{name} failed: {compact['error']}"]
    table = fact_table([{"name": name, "arguments": {}, "compact": compact}])
    lines = _write([f for f in table["facts"] if f.get("default")], table["gauges"])
    if lines or name in _ANALYSIS_TOOLS or name in ("find_stations", "anywhere", "describe_catchment"):
        return lines
    flat = []
    for k, v in compact.items():
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            flat.append(f"{k.replace('_', ' ')}: {display(v)}")
    return flat[:6]


# ── the lock ───────────────────────────────────────────────────────────────────

#: A gauge id with letters and digits ("E02102A", "3400TH"); pure digit ids are numbers and checked as numbers.
#: "100-year" and "7-day" are a period, not an id (check3: "the id 100-year is in no result" refused two right
#: summaries).
_ALNUM_ID = re.compile(r"\b(?!\d+-[A-Za-z]+\b)(?=[A-Za-z0-9_-]*[A-Za-z])(?=(?:[A-Za-z_-]*\d){3})"
                       r"[A-Za-z0-9][A-Za-z0-9_-]{4,}\b")
_Q_LABEL = re.compile(r"\bQ ?(\d{1,2})\b", re.I)
_LIST_MARK = re.compile(r"(?m)^\s*(?:\d{1,2}[.)]|[-*•])\s+")

_TREND_TOPIC = re.compile(r"\btrend|mann|kendall|sen'?s|over (?:the )?(?:time|years|record|decades)|getting|becoming|"
                          r"per year|/yr\b|a year\b", re.I)
_UP = re.compile(r"\b(?:increas\w*|upward|rising|wetter)\b", re.I)
_DOWN = re.compile(r"\b(?:decreas\w*|downward|falling|declin\w*|drier|dryer)\b", re.I)
_NOT_SIG = re.compile(r"\b(?:not|no|non|isn'?t|aren'?t|wasn'?t)[- ](?:statistically )?significant|\binsignificant|"
                      r"\bno (?:clear |real |detectable |meaningful |statistically significant |significant )?trend|"
                      r"\bno evidence of (?:a )?trend|\bnot a (?:significant |real )?trend", re.I)
_NOT_DIR = re.compile(r"\b(?:not|isn'?t|aren'?t|no longer) (?:getting |becoming |really )?"
                      r"(?P<w>increas\w*|decreas\w*|drier|dryer|wetter|rising|falling|declin\w*)", re.I)
_SIG = re.compile(r"\bsignificant", re.I)
_MAXIMA = re.compile(r"\b(?:peak|maxim|flood|annual max|amax)\w*", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n")


def _strings(payload: Any) -> list[str]:
    found: list[str] = []
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, str):
            found.append(item)
    return found


def _pool(payloads: list[Any], words: str) -> list[float]:
    """Every number in the results and in the reader's own words."""
    pool: list[float] = []
    stack = list(payloads)
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            pool.append(float(item))
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, str):
            pool.extend(_numbers(item))
    pool.extend(_numbers(words))
    return pool


def _stations(compacts: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, str]]:
    """(station_id, name) for every gauge in the results."""
    out = []
    for name, c in compacts:
        rows = c.get("stations") if name == "find_stations" else [c]
        for r in rows or []:
            if isinstance(r, dict) and r.get("station_id"):
                out.append((str(r["station_id"]), str(r.get("name") or "")))
    return out


def _swapped_ids(answer: str, stations: list[tuple[str, str]]) -> list[str]:
    """Ids written next to another gauge's name: the nearest name before an id (else after it) must be its own."""
    names = sorted({n for _, n in stations if len(n) >= 4}, key=len, reverse=True)
    if len({sid for sid, _ in stations}) < 2 or not names:
        return []
    own: dict[str, set[str]] = {}
    for sid, n in stations:
        own.setdefault(sid.lower(), set()).add(n.lower())
    wrong = []
    for line in re.split(r"\n|;|(?<=[.!?])\s+", answer):
        low = line.lower()
        spans: list[tuple[int, int, str]] = []
        for n in names:
            for m in re.finditer(re.escape(n.lower()), low):
                if not any(a <= m.start() < b for a, b, _ in spans):
                    spans.append((m.start(), m.end(), n.lower()))
        if not spans:
            continue
        for sid in own:
            for m in re.finditer(r"(?<![\w-])" + re.escape(sid) + r"(?![\w-])", low):
                before = [s for s in spans if s[1] <= m.start()]
                after = [s for s in spans if s[0] >= m.end()]
                if before:
                    near = max(before, key=lambda s: s[1])
                elif after:
                    near = min(after, key=lambda s: s[0])
                else:
                    continue
                if near[2] not in own[sid]:
                    their = next(iter(own[sid]), "").upper() or "another gauge"
                    wrong.append(f"{sid.upper()} is the id of {their}, not {near[2].upper()}")
    return wrong


def _trend_check(answer: str, compacts: list[tuple[str, dict[str, Any]]]) -> list[str]:
    tests: dict[str, dict[str, Any]] = {}
    for _, c in compacts:
        for holder in (c, c.get("glofas") if isinstance(c.get("glofas"), dict) else None):
            if not isinstance(holder, dict):
                continue
            for key in ("trend_annual_mean", "trend_annual_max"):
                if isinstance(holder.get(key), dict) and key not in tests:
                    tests[key] = holder[key]
    refused = []
    # The package's own connective sentences say how a trend is tested ("a p-value below 0.05 is called
    # significant"), never what a test found: they are no claim to hold (check4 refused a right answer for it).
    for line in CONNECTIVES.values():
        answer = answer.replace(normalise(line), " ")
    for sentence in _SENTENCE.split(answer):
        if not _TREND_TOPIC.search(sentence):
            continue
        not_sig = bool(_NOT_SIG.search(sentence))
        not_dir = _NOT_DIR.search(sentence)
        rest = _NOT_DIR.sub(" ", sentence)
        up, down = bool(_UP.search(rest)), bool(_DOWN.search(rest))
        says_sig = bool(_SIG.search(sentence)) and not not_sig
        if not (not_sig or not_dir or up or down or says_sig):
            continue
        key = "trend_annual_max" if _MAXIMA.search(sentence) and "trend_annual_max" in tests else "trend_annual_mean"
        test = tests.get(key) or next(iter(tests.values()), None)
        if test is None:
            refused.append("the answer states a trend, but no trend was computed for this question")
            continue
        sig = bool(test.get("significant"))
        slope = test.get("sens_slope_per_year")
        actual = test.get("direction") or ("increasing" if isinstance(slope, (int, float)) and slope > 0 else
                                           "decreasing" if isinstance(slope, (int, float)) and slope < 0 else None)
        said = "increasing" if up and not down else "decreasing" if down and not up else None
        p = test.get("p_value")
        if not_sig:
            ok = not sig and (said is None or actual is None or said == actual)
        elif not_dir:
            w = not_dir.group("w").lower()
            denied = "increasing" if _UP.match(w) else "decreasing"
            ok = not (sig and actual == denied)
        else:
            ok = sig and (said is None or said == actual)
        if not ok:
            refused.append(f"its trend wording does not match the test ({test.get('reading') or f'p = {p}'})")
    return refused


# ── labelled numbers: each label held to its own value (#590) ─────────────────
#
# The number check above asks only whether a number is somewhere in the results. On the device that let a right
# number through under a wrong label: "the Potomac's q95 of 3.43 m3/s" (3.43 is its daily minimum; its Q95 is
# 35.7) and "the return period for a flood of magnitude 2 m3/s is 2 years" (2 is a return period, not a flow).
# Here a number that a label claims (Q95, a mean, a minimum, an N-year flood, a record's dates) must be that
# label's value for the gauge the sentence is about, and a number written as a flow must be a flow.

#: A flow written after a number: m3/s and its spellings (NFKC has already turned m³ into m3).
_FLOW_UNIT = re.compile(r"\s*(?:m3/s(?:ec)?|m3 ?s-1|m\^3/s|cumecs|cubic met(?:re|er)s? (?:per|a) second)", re.I)
#: Any other unit after a number: rain, temperature, shares, areas, periods. Such a number is not a flow statistic.
_OTHER_UNIT = re.compile(r"\s*(?:mm\b|°|deg|%|percent|km|kilomet|years?\b|yrs?\b|days?\b|people|inhabitants)",
                         re.I)
#: A number as the prose writes it, with its place in the text: not a label's digits (Q95, m3) nor an id's.
_NUM_AT = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w])")
#: Where one claim ends and the next begins: sentence ends, semicolons, and "while", "whereas", "but", "and".
_CLAUSE_BREAK = re.compile(r"[.!?](?=\s|$)|;|\n|,?\s+(?:while|whereas|but|and)\s+", re.I)
#: The labels a statistic is claimed under, and what each may be.
_STAT_LABELS = (
    ("q", re.compile(r"\bq ?(95|50|10)\b", re.I)),
    ("mean", re.compile(r"\b(?:mean|average)\b", re.I)),
    ("median", re.compile(r"\bmedian\b", re.I)),
    ("min", re.compile(r"\b(?:minimum|min|lowest)\b", re.I)),
    ("max", re.compile(r"\b(?:maximum|max|highest)\b", re.I)),
)
_LABEL_WORDS = {"mean": "mean", "median": "median", "min": "minimum", "max": "maximum"}
#: What may stand between a label and its number: "a minimum of 3.43", "mean flow: 320.8", "maximum recorded
#: value is 12100". Anything else ("the annual mean (Mann-Kendall p = 0.82") does not tie the two.
_CONNECTIVE = re.compile(r"[\s:=(,]*(?:(?:daily|recorded|flow|discharge|value|values|is|was|of|about|around|"
                         r"approximately|roughly|at|day|on record|ever|a|an|the)[\s:=(,]+)*", re.I)
#: An N-year flood, said four ways.
_PERIODS = (
    re.compile(r"\b(\d{1,4})[- ](?:year|yr)\b(?=[- ](?:flood|return|event|peak|design|flow|discharge|level|"
               r"storm|high|max))", re.I),
    re.compile(r"\b1[- ]in[- ](\d{1,4})\b", re.I),
    re.compile(r"\breturn period (?:of |is |= ?)?(?:about |around )?(\d{1,4}) ?(?:years?|yrs?)\b", re.I),
    re.compile(r"\breturn period\b[^.;\n]*?\bis (\d{1,4}) ?(?:years?|yrs?)\b", re.I),
)
#: After a number: a rate per year (Sen's slope), not a flow.
_RATE = re.compile(r"\s*(?:m3/s(?:ec)?|m3 ?s-1|cumecs)?\s*(?:per year|a year|/ ?yr|/ ?year)\b", re.I)
#: Before a label in its clause: the number that follows is a test's, not the label's.
_TEST_WORDS = re.compile(r"p[- ]?values?|\bp\s*=|\bslope|\btau\b|mann|kendall", re.I)
_LP3 = re.compile(r"\b(?:lp3|lp-3|lp ?iii|log[- ]?pearson)", re.I)
_GEV = re.compile(r"\bgev\b|generali[sz]ed extreme", re.I)
_DATE_RANGE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?:to|until|through|-|and)\s*(\d{4}-\d{2}-\d{2})", re.I)
#: Words in a gauge's name too common to say which gauge a sentence is about.
_GENERIC = frozenset({"river", "near", "above", "below", "creek", "brook", "station", "stream", "gauge", "pump",
                      "with", "from", "upon", "lake", "canal", "water", "little", "great", "north", "south", "east",
                      "west", "upper", "lower", "the", "and", "at"})
#: Numbers in a result that are never flows: periods, counts, tests, places, climate.
_NOT_FLOW = frozenset({"T_years", "n_years", "years", "n", "p_value", "tau", "latitude", "longitude", "year",
                       "n_returned", "n_catalog", "limit", "aridity_index", "precipitation_mm_per_year",
                       "et0_mm_per_year", "temperature_mean_c", "wettest_day_mm"})


def _flow_pool(payloads: list[Any]) -> list[float]:
    """The numbers in the results that are flows (or rates of flow), not periods, counts or test statistics."""
    pool: list[float] = []
    stack: list[tuple[str, Any]] = [("", p) for p in payloads]
    while stack:
        key, item = stack.pop()
        if key in _NOT_FLOW or isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            pool.append(float(item))
        elif isinstance(item, dict):
            stack.extend((str(k), v) for k, v in item.items())
        elif isinstance(item, (list, tuple)):
            stack.extend((key, v) for v in item)
    return pool


def _gauges(compacts: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Every analysed record in the results, with what names it in prose and its labelled values."""
    out = []
    for name, c in compacts:
        if name in _ANALYSIS_TOOLS:
            g, whole = c, c
        elif name == "anywhere" and isinstance(c.get("glofas"), dict):
            g, whole = c["glofas"], c
        else:
            continue
        if g.get("error"):
            continue
        label = " ".join(str(g[k]) for k in ("name", "station_id") if g.get(k)) or (
            "the GloFAS cell" if name == "anywhere" else "the record")
        tokens = {str(g["station_id"]).lower()} if g.get("station_id") else set()
        for field in ("name", "river"):
            tokens |= {w for w in re.findall(r"[a-z]{4,}", str(g.get(field) or "").lower()) if w not in _GENERIC}
        if name == "anywhere":
            tokens.add("glofas")
        daily = g.get("daily_values") or {}
        fd = g.get("flow_duration") or {}
        ff = g.get("flood_frequency") or {}
        rec = (ff.get("record_max") or {}).get("value")
        values = {
            "q95": [fd.get("q95")], "q50": [fd.get("q50"), daily.get("median")], "q10": [fd.get("q10")],
            "mean": [daily.get("mean")], "median": [daily.get("median"), fd.get("q50")],
            "min": [daily.get("min")], "max": [daily.get("max"), rec],
        }
        levels = {}
        for r in ff.get("return_levels") or []:
            if isinstance(r, dict) and r.get("T_years") is not None:
                ci = r.get("lp3_90ci") if isinstance(r.get("lp3_90ci"), list) else []
                levels[float(r["T_years"])] = {"gev": r.get("gev"), "lp3": r.get("lp3"), "ci": ci}
        pairs = set()
        for d in [whole, g]:
            stack = [d]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    if isinstance(item.get("start"), str) and isinstance(item.get("end"), str):
                        pairs.add((item["start"][:10], item["end"][:10]))
                    stack.extend(item.values())
        out.append({"label": label, "tokens": tokens, "unit": g.get("unit") or "",
                    "values": {k: [float(v) for v in vs if isinstance(v, (int, float)) and not isinstance(v, bool)]
                               for k, vs in values.items()},
                    "levels": levels, "pairs": pairs,
                    "strings": " ".join(s.lower() for s in _strings(whole))})
    # A word two gauges share ("thames" for two gauges on it) names neither.
    for g in out:
        g["tokens"] = {t for t in g["tokens"] if not any(t in h["tokens"] for h in out if h is not g)}
    return out


def _clauses(text: str) -> list[tuple[int, int]]:
    bounds, start = [], 0
    for m in _CLAUSE_BREAK.finditer(text):
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(text)))
    return [b for b in bounds if b[1] > b[0]]


class _Who:
    """Which gauge a place in the answer is about: the one its clause names, else the last one named before."""

    def __init__(self, text: str, gauges: list[dict[str, Any]]):
        low = text.lower()
        self.gauges = gauges
        self.mentions: list[tuple[int, int]] = []
        for i, g in enumerate(gauges):
            for t in g["tokens"]:
                for m in re.finditer(r"(?<![\w-])" + re.escape(t) + r"(?![\w-])", low):
                    self.mentions.append((m.start(), i))
        self.mentions.sort()
        self.clauses = _clauses(text)

    def at(self, pos: int) -> int | None:
        if len(self.gauges) == 1:
            return 0
        a, b = next(((a, b) for a, b in self.clauses if a <= pos < b), (pos, pos + 1))
        inside = {i for p, i in self.mentions if a <= p < b}
        if len(inside) == 1:
            return inside.pop()
        if inside:
            return None    # two gauges in one clause: either may be meant
        before = [i for p, i in self.mentions if p < a]
        return before[-1] if before else None

    def clause(self, pos: int) -> tuple[int, int]:
        return next(((a, b) for a, b in self.clauses if a <= pos < b), (pos, pos + 1))


def _num(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _fmt_vals(vals: list[float], unit: str) -> str:
    return " or ".join(f"{_fmt(v)}" for v in vals) + (f" {unit}" if unit else "")


def _label_check(text: str, compacts: list[tuple[str, dict[str, Any]]], words: str) -> list[str]:
    """Each labelled number against its own value, for the gauge its sentence is about (see the section note)."""
    gauges = _gauges(compacts)
    refused: list[str] = []
    blank = _DATE.sub(lambda m: " " * len(m.group(0)), text)
    numbers = [(m.start(), m.end(), _num(m.group(0))) for m in _NUM_AT.finditer(blank)]
    # A gauge id of digits ("USGS-01646500") is a name here, not a number a label could claim.
    low = blank.lower()
    ids = {sid.lower() for sid, _ in _stations(compacts)}
    id_spans = [(m.start(), m.end()) for sid in ids
                for m in re.finditer(r"(?<![\w])" + re.escape(sid) + r"(?![\w])", low)]
    numbers = [(a, b, v) for a, b, v in numbers
               if v is not None and not any(x <= a < y for x, y in id_spans)]
    # "-0.0365 m3/s per year" is a Sen's slope, a rate: not a flow, and no label's value (check3: it was read as the
    # daily mean and a right trend answer refused).
    rates = {a for a, b, _ in numbers if _RATE.match(blank, b)}
    numbers = [n for n in numbers if n[0] not in rates]
    flows = {a for a, b, _ in numbers if _FLOW_UNIT.match(blank, b)}

    # A number written as a flow must be a flow in the results (or the reader's own): "2 m3/s" where 2 is a
    # return period is refused.
    flow_pool = _flow_pool([c for _, c in compacts]) + _numbers(words)
    if compacts:
        for a, b, v in numbers:
            if a in flows and abs(v) > 1e-9 and not _close(v, flow_pool):
                refused.append(f"{v:g} m3/s is not a flow in the results")
    if not gauges:
        return refused
    who = _Who(text, gauges)

    def name(i: int | None) -> str:
        return gauges[i]["label"] if i is not None else "any gauge in the results"

    def next_number(pos: int, end: int) -> tuple[int, int, float] | None:
        return next(((a, b, v) for a, b, v in numbers if pos <= a < end), None)

    # Statistics under a label: Q95, mean, median, minimum, maximum.
    labels = []
    for kind, rx in _STAT_LABELS:
        for m in rx.finditer(blank):
            labels.append((m.start(), m.end(), f"q{m.group(1)}" if kind == "q" else kind))
    labels.sort()
    for j, (ls, le, kind) in enumerate(labels):
        _, cend = who.clause(ls)
        nxt = next_number(le, cend)
        if nxt is None:
            continue
        a, b, v = nxt
        if any(le <= o[0] < a for o in labels[j + 1:]) or a - le > 80:
            continue   # another label sits between (the number is that one's), or the two are far apart
        filler = blank[le:a]
        if a not in flows and (_OTHER_UNIT.match(blank, b) or not _CONNECTIVE.fullmatch(filler)):
            continue
        cstart, _ = who.clause(ls)
        if a not in flows and _TEST_WORDS.search(blank[cstart:ls]):
            continue   # "the p-value for the annual mean is 0.953": the test's number, not the mean
        i = who.at(a)
        cands = [i] if i is not None else list(range(len(gauges)))
        have = [k for k in cands if gauges[k]["values"].get(kind)]
        if not have:
            continue
        if any(_close(v, gauges[k]["values"][kind]) for k in have):
            continue
        said = kind.upper() if kind.startswith("q") else f"daily {_LABEL_WORDS[kind]}"
        others = [k for k, g in enumerate(gauges) if k not in cands and _close(v, g["values"].get(kind) or [])]
        if i is not None and others:
            refused.append(f"{v:g} is the {said} of {name(others[0])}, not of {name(i)}")
        elif i is not None:
            refused.append(f"{v:g} is not the {said} of {name(i)} "
                           f"({_fmt_vals(gauges[i]['values'][kind][:1], gauges[i]['unit'])})")
        else:
            refused.append(f"{v:g} is not the {said} of any gauge in the results")

    # N-year floods: the flow tied to a period must be that period's return level, by the method named.
    for rx in _PERIODS:
        for m in rx.finditer(blank):
            t = float(m.group(1))
            cstart, cend = who.clause(m.start())
            after = [n for n in numbers if m.end() <= n[0] < cend and n[0] in flows]
            before = [n for n in numbers if cstart <= n[0] < m.start() and n[0] in flows]
            bound = after[0] if after else (before[-1] if before else None)
            if bound is None:
                continue
            a, b, v = bound
            i = who.at(a)
            cands = [i] if i is not None else list(range(len(gauges)))
            rows = [(k, gauges[k]["levels"].get(t)) for k in cands]
            rows = [(k, r) for k, r in rows if r]
            if not rows:
                if not any(gauges[k]["levels"] for k in cands):
                    continue
                refused.append(f"there is no {t:g}-year flood in the results for {name(i)}")
                continue
            sentence = next((s for s in _SENTENCE.split(text) if text.find(s) <= a < text.find(s) + len(s)), text)
            says_lp3, says_gev = bool(_LP3.search(sentence)), bool(_GEV.search(sentence))
            fits = []
            for k, r in rows:
                if r.get("gev") is not None and _close(v, [r["gev"]]):
                    fits.append("gev")
                if r.get("lp3") is not None and _close(v, [r["lp3"]]):
                    fits.append("lp3")
                if any(_close(v, [x]) for x in r.get("ci") or [] if isinstance(x, (int, float))):
                    fits.append("ci")
            k0, r0 = rows[0]
            unit = gauges[k0]["unit"]
            if not fits:
                elsewhere = [f"{tt:g}" for tt, rr in gauges[k0]["levels"].items()
                             if any(isinstance(x, (int, float)) and _close(v, [x])
                                    for x in (rr.get("gev"), rr.get("lp3")))]
                if elsewhere:
                    refused.append(f"{v:g} is the {elsewhere[0]}-year flood, not the {t:g}-year")
                else:
                    gev = f"GEV {_fmt(r0['gev'])}" if r0.get("gev") is not None else ""
                    lp3 = f"LP3 {_fmt(r0['lp3'])}" if r0.get("lp3") is not None else ""
                    refused.append(f"{v:g} is not the {t:g}-year flood of {name(k0)} "
                                   f"({', '.join(x for x in (gev, lp3) if x)} {unit})".replace(" )", ")"))
            elif "gev" not in fits and "lp3" in fits and not says_lp3:
                refused.append(f"{v:g} is the LP3 {t:g}-year flood, and the answer does not say LP3 (the page's "
                               f"figure is the GEV, {_fmt(r0['gev'])} {unit})" if r0.get("gev") is not None else
                               f"{v:g} is the LP3 {t:g}-year flood, and the answer does not say LP3")
            elif "lp3" not in fits and "gev" in fits and says_lp3 and not says_gev:
                refused.append(f"{v:g} is the GEV {t:g}-year flood, not the LP3")

    # Record dates: a date said of one gauge must be that gauge's, and "from A to B" its start and end.
    for m in _DATE_RANGE.finditer(text):
        i = who.at(m.start())
        pair = (m.group(1), m.group(2))
        cands = [i] if i is not None else list(range(len(gauges)))
        if any(pair in gauges[k]["pairs"] for k in cands):
            continue
        others = [k for k, g in enumerate(gauges) if k not in cands and pair in g["pairs"]]
        if i is not None and others:
            refused.append(f"{pair[0]} to {pair[1]} is the record of {name(others[0])}, not of {name(i)}")
        elif i is not None and all(d in gauges[i]["strings"] for d in pair) and gauges[i]["pairs"]:
            s0, e0 = sorted(gauges[i]["pairs"])[0]
            refused.append(f"the record of {name(i)} runs from {s0} to {e0}, not {pair[0]} to {pair[1]}")
    for m in _DATE.finditer(text):
        i = who.at(m.start())
        d = m.group(0)
        if i is None or d in gauges[i]["strings"]:
            continue
        others = [k for k, g in enumerate(gauges) if k != i and d in g["strings"]]
        if others:
            refused.append(f"{d} is in the record of {name(others[0])}, not of {name(i)}")
    return refused


_LIST_COUNT = re.compile(r"\b(?P<n>two|three|four|five|six|seven|eight|nine|ten|\d{1,2})\s+(?:[a-z]+\s+){0,2}"
                         r"(?:gauges|stations|sites)\b", re.I)
_LIST_WORDS = re.compile(r"\b(?:list|which|all)\b[^.?]*\b(?:gauges|stations)\b", re.I)
_COUNT_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def wanted_list(question: str) -> int:
    """How many gauges a question asks to be listed ("the five gauges nearest Kingston" is 5); 0 for none."""
    m = _LIST_COUNT.search(question or "")
    if m:
        n = m.group("n").lower()
        return _COUNT_WORDS.get(n) or int(n)
    return 2 if _LIST_WORDS.search(question or "") else 0


def _list_check(text: str, compacts: list[tuple[str, dict[str, Any]]], words: str) -> list[str]:
    """A question that asks for several gauges is answered with several: "five gauges nearest Kingston" with one
    analysed gauge is refused, and the page shows the search's list instead."""
    question = (words or "").split("\n\nContext:")[0]
    want = wanted_list(question)
    rows = [r for name, c in compacts if name == "find_stations" for r in c.get("stations") or []
            if isinstance(r, dict)]
    if want < 2 or not rows:
        return []
    low = text.lower()
    named = sum(1 for r in rows if (r.get("station_id") and str(r["station_id"]).lower() in low)
                or (len(str(r.get("name") or "")) >= 4 and str(r["name"]).lower() in low))
    need = min(want, len(rows))
    if named < need:
        return [f"the question asks for {want} gauges and the answer names {named} of those the search found"]
    return []


def lock_answer(answer: str, results: list[dict[str, Any]], words: str = "") -> dict[str, Any]:
    """Hold an on-device answer to this question's tool results.

    ``results`` are ``{"name", "arguments", "compact", "ok"}``, one per tool call made for this question (and
    only this one: a fresh session per question means nothing else could have been seen). ``words`` is the
    question with the context line the page added.

    Every number, date, flow-duration label (Q95, Q10) and gauge id in the answer must be in the results or in
    the words, allowing for rounding; a gauge id must sit next to its own gauge's name; every trend or
    significance word must agree with the Mann-Kendall result. A labelled number must be that label's value for
    the gauge its clause is about (:func:`_label_check`: Q95 its flow-duration Q95, an N-year flood its return
    level for N by the method named, a mean or a record date its own field), and a number written in m3/s a
    flow. A question asking for several gauges must be answered with several (:func:`_list_check`).

    Returns ``{"ok", "refused", "facts", "notes"}``: ``refused`` says why in short phrases, ``facts`` are lines
    built from the results (what the page shows when the answer is refused), ``notes`` say what the reader
    should know either way.
    """
    text = normalise(answer or "")
    words_n = normalise(words or "")
    ok_results = [r for r in results or [] if r.get("ok") and isinstance(r.get("compact"), dict)]
    compacts = [(str(r.get("name")), r["compact"]) for r in ok_results]
    refused: list[str] = []

    # Dates, whole: "1930-03-01" must be a date one of the results (or the reader) wrote.
    strings = [s.lower() for _, c in compacts for s in _strings(c)] + [words_n.lower()]
    for d in dict.fromkeys(_DATE.findall(text)):
        if not any(d in s for s in strings):
            refused.append(f"the date {d} is in no result for this question")
    body = _LIST_MARK.sub(" ", _DATE.sub(" ", text))

    # Flow-duration labels: Q10 is a point of the flow-duration curve, and only one the results hold.
    fdc_keys = {k for _, c in compacts for holder in (c, c.get("glofas") or {}) if isinstance(holder, dict)
                for k in (holder.get("flow_duration") or {})}
    for m in dict.fromkeys(_Q_LABEL.findall(body)):
        if f"q{m}" not in fdc_keys and f"q{m}" not in words_n.lower():
            refused.append(f"Q{m} is in no result for this question")

    # Gauge ids with letters in them, which the number check cannot see.
    for token in dict.fromkeys(_ALNUM_ID.findall(body)):
        if not any(token.lower() in s for s in strings):
            refused.append(f"the id {token} is in no result for this question")
    refused += _swapped_ids(text, _stations(compacts))

    # Numbers: the results and the reader's words, with rounding; a percentage may be a fraction in the result.
    pool = _pool([c for _, c in compacts], words_n)
    plain = _SQUARED_UNIT.sub(" ", _EXPONENT_UNIT.sub(" ", body))
    percents = {float(m.replace(",", "")) for m in re.findall(r"(-?\d[\d,]*\.?\d*)\s*%", plain)}
    # "68.6 W" is the longitude -68.6: the compass letter carries the sign.
    compass = {float(m.group(1)) for m in _COMPASS.finditer(plain) if m.group(2) in "SW"}
    unsupported = []
    for n in _numbers(plain):
        if n in _CONVENTIONS or abs(n) < 1e-9 or (n in percents and n / 100 in _CONVENTIONS):
            continue   # p < 0.05, a 90 % interval: conventions, not claims about the data
        if float(n).is_integer() and 1800 <= n <= 2100:
            ok = n in pool
        else:
            ok = (_close(n, pool) or (n in compass and _close(-n, pool))
                  or (n in percents and _close(n / 100, pool)))
        if not ok and n not in unsupported:
            unsupported.append(n)
    if unsupported:
        listed = ", ".join(f"{n:g}" for n in unsupported[:5])
        refused.append(f"{listed} {'is' if len(unsupported) == 1 else 'are'} in no result for this question"
                       if compacts else f"no tool ran for this question, so {listed} cannot come from data")

    refused += _trend_check(text, compacts)
    refused += _label_check(text, compacts, words_n)
    refused += _list_check(text, compacts, words_n)

    notes = []
    for r in ok_results:
        near = (r.get("arguments") or {}).get("near")
        if r.get("name") == "find_stations" and isinstance(near, (list, tuple)) and len(near) == 2:
            given = _numbers(words_n)
            if not all(isinstance(v, (int, float)) and _close(float(v), given) for v in near):
                notes.append(f"The search was near {near[0]}, {near[1]}, a point the model chose; your words did "
                             "not give it, so 'nearest' is nearest to that point.")
    lines = [line for name, c in compacts for line in facts(name, c)]
    return {"ok": not refused, "refused": list(dict.fromkeys(refused))[:6], "facts": lines[:12], "notes": notes}


# ── round 3: the model picks, the code writes (#590) ──────────────────────────
#
# Three on-device checks held the model's free text to the results with patterns, and each round's locks caught
# the last round's errors while Gemini Nano found new ones ("significant trend in annual maxima with a p =
# 0.0689", "100-year" read as a gauge id). So the model no longer writes the answer. Each step's reply is one of
# two shapes, held by a responseConstraint schema whose values are enums built here from this question:
#
# * ``{"tool": <name>, "arguments": {...}}``: a tool from :data:`DEVICE_TOOLS`, with a gauge id, a source, a
#   search phrase or a point that the reader's words or an earlier result hold (:func:`check_call` refuses any
#   other, and a call already made is not run again);
# * ``{"answer": [<fact key>, ...]}``: keys of :func:`fact_table`, the sentences this module writes from the
#   compact results ("USGS-01646500.fdc.q95", "USGS-01646500.gev.100", "USGS-01646500.trend.mean"), plus a few
#   connective sentences by id (:data:`CONNECTIVES`).
#
# :func:`render_answer` writes the answer from the keys. Every sentence names its gauge, label, value, unit and
# method, and a trend is this module's own reading of the test. Nothing the model writes reaches the page.

#: The tools the device may call, and the arguments each takes.
DEVICE_TOOLS: dict[str, tuple[str, ...]] = {
    "find_stations": ("query", "near", "limit"),
    "analyze_station": ("source", "station_id"),
    "anywhere": ("lat", "lon"),
    "describe_catchment": ("lat", "lon"),
}

#: Sentences the model may add between facts, by id. Each says how the facts were made, never what they show.
CONNECTIVES: dict[str, str] = {
    "say.daily": "Mean, median, lowest and highest are of the daily values.",
    "say.floods": "Flood figures are fitted to the annual maxima: GEV by L-moments, and LP3 (Log-Pearson III) "
                  "with its 90 % interval.",
    "say.trend": "Trends are Mann-Kendall tests on the annual series with Sen's slope; a p-value below 0.05 is "
                 "called significant.",
    "say.modelled": "GloFAS discharge is modelled by a global model, not measured at a gauge.",
}

#: The most facts one answer may pick.
MAX_PICKS = 10


def display(x: Any) -> str:
    """A number as the Explorer shows it: whole with commas from 100 up, else three significant figures, never
    in exponent form (11247.39 is "11,247", 0.0365 is "0.0365"; the old fact lines said "1.125e+04")."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return str(x)
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        return "unknown"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if v == 0:
        return "0"
    decimals = max(0, 2 - int(math.floor(math.log10(abs(v)))))
    out = f"{round(v, decimals):.{decimals}f}"
    return out.rstrip("0").rstrip(".") if "." in out else out


def coord(x: Any) -> str:
    """A latitude or longitude as written: up to four decimals, no trailing zeros (121.55, not "122")."""
    try:
        return f"{round(float(x), 4):g}"
    except (TypeError, ValueError):
        return str(x)


def _p_text(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {display(p)}"


def _unit_text(unit: str) -> str:
    return f" {unit}" if unit else ""


def _point_key(lat: Any, lon: Any) -> str:
    return f"{float(lat):g},{float(lon):g}"


def _gauge_names(c: dict[str, Any], label: str | None = None) -> tuple[str, str]:
    """The long and short names of a record: "POTOMAC ... (usgs USGS-01646500)" first, then "USGS-01646500"."""
    if label:
        return label, label
    sid = str(c.get("station_id") or "")
    name = str(c.get("name") or "")
    src = str(c.get("source") or "")
    if name and sid:
        return f"{name} ({src + ' ' if src else ''}{sid})", sid
    return (name or sid or "the record"), (sid or name or "the record")


def _analysis_facts(c: dict[str, Any], prefix: str, gauge: str, *, modelled: bool = False) -> list[dict[str, Any]]:
    """The sentences for one analysis, each with its key, the gauge it names and whether the page's summary
    (shown when no answer is picked) includes it."""
    u = str(c.get("unit") or "")
    out: list[dict[str, Any]] = []

    def add(key: str, text: str, default: bool = False) -> None:
        out.append({"key": f"{prefix}.{key}", "gauge": gauge, "text": text, "default": default})

    if c.get("error"):
        add("error", "{who}: the tool failed (" + str(c["error"]) + ").", True)
        return out
    var = str(c.get("variable") or ("modelled discharge" if modelled else "record"))
    if c.get("start") and c.get("end"):
        years = f", {display(c['years'])} years" if isinstance(c.get("years"), (int, float)) else ""
        add("record", f"{{who}}: {var} from {c['start']} to {c['end']}{years}.", True)
    daily = c.get("daily_values") or {}
    words = {"mean": "The mean of the daily values", "median": "The median of the daily values",
             "min": "The lowest daily value", "max": "The highest daily value"}
    for k in ("mean", "median", "min", "max"):
        if isinstance(daily.get(k), (int, float)):
            add(k, f"{words[k]} at {{who}} is {display(daily[k])}{_unit_text(u)}.", k == "mean")
    fd = c.get("flow_duration") or {}
    share = {"q95": "95 % of the time", "q50": "half the time", "q10": "10 % of the time"}
    for k in ("q95", "q50", "q10"):
        if isinstance(fd.get(k), (int, float)):
            add(f"fdc.{k}", f"{k.upper()} at {{who}} is {display(fd[k])}{_unit_text(u)}, the flow exceeded "
                            f"{share[k]} (flow-duration curve).", k == "q95")
    ff = c.get("flood_frequency") or {}
    n = ff.get("n_years")
    of_n = f" to {n} annual maxima" if isinstance(n, (int, float)) else ""
    for r in ff.get("return_levels") or []:
        if not isinstance(r, dict) or not isinstance(r.get("T_years"), (int, float)):
            continue
        t = display(r["T_years"])
        if isinstance(r.get("gev"), (int, float)):
            add(f"gev.{t}", f"The {t}-year flood at {{who}} is {display(r['gev'])}{_unit_text(u)} "
                            f"(GEV fitted by L-moments{of_n}).", t == "100")
        if isinstance(r.get("lp3"), (int, float)):
            ci = r.get("lp3_90ci") if isinstance(r.get("lp3_90ci"), list) and len(r["lp3_90ci"]) == 2 else None
            ci_txt = f", 90 % interval {display(ci[0])} to {display(ci[1])}{_unit_text(u)}" if ci else ""
            fitted = f", fitted{of_n}" if of_n else ""
            add(f"lp3.{t}", f"The {t}-year flood at {{who}} is {display(r['lp3'])}{_unit_text(u)} by Log-Pearson "
                            f"III (LP3{fitted}){ci_txt}.")
    rec = ff.get("record_max") or {}
    if isinstance(rec.get("value"), (int, float)):
        year = f", in {rec['year']}" if rec.get("year") is not None else ""
        add("amax", f"The largest annual maximum at {{who}} is {display(rec['value'])}{_unit_text(u)}{year}.")
    for key, short in (("trend_annual_mean", "trend.mean"), ("trend_annual_max", "trend.max")):
        t = c.get(key)
        if isinstance(t, dict) and isinstance(t.get("p_value"), (int, float)):
            add(short, "At {who} there is " + _reading(t, u) + ".", True)
    return out


def _reading(t: dict[str, Any], unit: str) -> str:
    """The test's verdict in words, from its own p-value and slope: the only trend wording the page shows."""
    p = float(t["p_value"])
    on = t.get("on") or "the series"
    slope = t.get("sens_slope_per_year")
    slope_txt = f"; Sen's slope {display(slope)}{_unit_text(unit)} per year" if isinstance(slope, (int, float)) else ""
    if p < _P_SIGNIFICANT:
        direction = t.get("direction") or ("increasing" if isinstance(slope, (int, float)) and slope > 0
                                           else "decreasing")
        return f"a significant {direction} trend in the {on} (Mann-Kendall {_p_text(p)}{slope_txt})"
    return f"no significant trend in the {on} (Mann-Kendall {_p_text(p)}{slope_txt})"


def _climate_facts(c: dict[str, Any], key: str, where: str) -> list[dict[str, Any]]:
    cl = c.get("climate") or {}
    src = cl.get("source") or "ERA5"
    span = f", {c['start']} to {c['end']}" if c.get("start") and c.get("end") else ""
    rows = []
    for field, k, text in (
        ("precipitation_mm_per_year", "rain", "Mean rainfall at {w} is {v} mm a year"),
        ("et0_mm_per_year", "et0", "Reference evapotranspiration at {w} is {v} mm a year"),
        ("temperature_mean_c", "temp", "Mean air temperature at {w} is {v} °C"),
        ("wettest_day_mm", "wettest", "The wettest day at {w} had {v} mm"),
    ):
        if isinstance(cl.get(field), (int, float)):
            rows.append({"key": f"{key}.{k}", "gauge": None, "default": k in ("rain", "temp"),
                         "text": text.format(w=where, v=display(cl[field])) + f" ({src}{span})."})
    if isinstance(cl.get("aridity_index"), (int, float)):
        cls = f" ({cl['aridity_class']})" if cl.get("aridity_class") else ""
        rows.append({"key": f"{key}.aridity", "gauge": None, "default": True,
                     "text": f"The aridity index at {where} is {display(cl['aridity_index'])}{cls} ({src})."})
    return rows


def _catchment_facts(c: dict[str, Any], key: str, where: str) -> list[dict[str, Any]]:
    attrs = c.get("attributes") if isinstance(c.get("attributes"), dict) else {}
    rows = []
    if isinstance(attrs.get("area_km2"), (int, float)):
        rows.append({"key": f"{key}.area", "gauge": None, "default": True,
                     "text": f"The catchment upstream of {where} covers {display(attrs['area_km2'])} km2 "
                             "(BasinATLAS)."})
    for field, entry in attrs.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("value"), (int, float)) or not entry.get("label"):
            continue
        unit = str(entry.get("unit") or "")
        rows.append({"key": f"{key}.{field}", "gauge": None, "default": False,
                     "text": f"The {entry['label']} of the catchment of {where} is {display(entry['value'])}"
                             f"{_unit_text(unit)} (BasinATLAS)."})
    return rows


def _search_facts(c: dict[str, Any], args: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = [r for r in c.get("stations") or [] if isinstance(r, dict)]
    near = args.get("near") if isinstance(args.get("near"), (list, tuple)) and len(args["near"]) == 2 else None
    if near:
        how = f"near {coord(near[0])}, {coord(near[1])}, nearest first"
    elif args.get("query"):
        how = f"for “{args['query']}”"
    else:
        how = ""
    n = c.get("n_returned") if isinstance(c.get("n_returned"), int) else len(rows)
    count = "no gauges" if not n else ("1 gauge" if n == 1 else f"{n} gauges")
    if n and n > len(rows):
        count += f"; the first {len(rows)} are listed"
    out = [{"key": f"{key}.count", "gauge": None, "default": True,
            "text": f"The catalogue search {how} returned {count}.".replace("search  ", "search ")}]
    for r in rows:
        sid = r.get("station_id")
        if not sid:
            continue
        bits = [f"{r.get('name') or sid} ({r.get('source')} {sid})"]
        if r.get("river"):
            river = str(r["river"])
            bits.append(f"on {river}" if river.lower().startswith("the ") else f"on the {river}")
        if isinstance(r.get("latitude"), (int, float)) and isinstance(r.get("longitude"), (int, float)):
            bits.append(f"at {coord(r['latitude'])}, {coord(r['longitude'])}")
        if r.get("period_start") and r.get("period_end"):
            bits.append(f"record {r['period_start']} to {r['period_end']}")
        out.append({"key": f"{sid}.where", "gauge": None, "default": True, "text": ", ".join(bits) + "."})
    return out


def fact_table(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Every sentence this question's results allow, by key: ``{"facts": [{key, gauge, text, default}],
    "gauges": {gauge: [long name, short name]}}``.

    ``text`` holds ``{who}`` where the gauge's name goes: its long name the first time an answer names it, its
    id after that. ``default`` marks the summary the page shows when the model picks nothing it may.
    """
    facts_out: list[dict[str, Any]] = []
    gauges: dict[str, list[str]] = {}
    searches = 0
    for r in results or []:
        if not isinstance(r, dict) or not isinstance(r.get("compact"), dict):
            continue
        name, c, args = str(r.get("name")), r["compact"], r.get("arguments") or {}
        if name in _ANALYSIS_TOOLS:
            prefix = str(c.get("station_id") or args.get("station_id") or f"record{len(gauges) + 1}")
            if prefix not in gauges:
                gauges[prefix] = list(_gauge_names(c))
            facts_out += _analysis_facts(c, prefix, prefix)
        elif name == "anywhere" and not c.get("error"):
            lat, lon = c.get("latitude", args.get("lat")), c.get("longitude", args.get("lon"))
            if not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
                continue
            pk = _point_key(lat, lon)
            where = f"{coord(lat)}, {coord(lon)}"
            facts_out += _climate_facts(c, f"climate@{pk}", where)
            if isinstance(c.get("glofas"), dict) and not c["glofas"].get("error"):
                gk = f"glofas@{pk}"
                label = f"the GloFAS cell at {where} (modelled)"
                gauges[gk] = [label, label]
                facts_out += _analysis_facts(c["glofas"], gk, gk, modelled=True)
        elif name == "describe_catchment" and not c.get("error"):
            lat, lon = c.get("latitude", args.get("lat")), c.get("longitude", args.get("lon"))
            if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                facts_out += _catchment_facts(c, f"catchment@{_point_key(lat, lon)}",
                                              f"{coord(lat)}, {coord(lon)}")
        elif name == "find_stations" and not c.get("error"):
            searches += 1
            facts_out += _search_facts(c, args, "search" if searches == 1 else f"search{searches}")
    seen: set[str] = set()
    unique = []
    for f in facts_out:
        if f["key"] not in seen:
            seen.add(f["key"])
            unique.append(f)
    return {"facts": unique, "gauges": gauges}


def _write(picked: list[dict[str, Any]], gauges: dict[str, list[str]], *, short: bool = False) -> list[str]:
    named: set[str] = set(gauges) if short else set()
    lines = []
    for f in picked:
        g = f.get("gauge")
        who = ""
        if g:
            long, short = gauges.get(g, [g, g])
            who = short if g in named else long
            named.add(g)
        line = f["text"].replace("{who}", who)
        lines.append(line[:1].upper() + line[1:])
    return lines


# ── what the model may pass to a tool ─────────────────────────────────────────

_COORD_PAIR = re.compile(r"(?<![\w.])(-?\d{1,2}(?:\.\d+)?)\s*°?\s*[NS]?\s*,\s*(-?\d{1,3}(?:\.\d+)?)(?![\w.])")
_CTX_GAUGE = re.compile(r"source (?P<src>[\w-]+), id (?P<id>[^,\s)]+)")
_SLASH_GAUGE = re.compile(r"\((?P<src>[a-z][\w-]*) ?/ ?(?P<id>[^)\s]+)\)", re.I)
#: A gauge id in the reader's own words: letters and digits mixed ("E02102A", "USGS-01646500"), or six digits
#: or more. "100-year", "100yr", "1930s", "7-day" and dates are not ids.
_WORD_ID = re.compile(r"(?<![\w-])(?!\d+-[A-Za-z]+\b)(?!\d+(?:yrs?|years?|s|th|st|nd|rd|km2?|mm|cm|m3s?)\b)"
                      r"(?:[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*|"
                      r"\d+[A-Za-z][A-Za-z0-9_-]*|\d{6,})(?![\w-])")
_STOP = frozenset("""a an and are as at be by can do does find for from gauge gauges give has have how i in is it its
me my near nearest of on or show station stations tell than that the their there these this to us was what
when where which with you your five four three two six seven eight nine ten summarise summarize compare record
flow flows mean median river rivers flood floods discharge rain rainfall trend trends climate catchment water level
levels low high year years period frequency peak peaks drier wetter dry wet getting bigger smaller annual daily
modelled list use using please name get check describe explain why will would should could did had""".split())
#: Where a search phrase stops: punctuation, and anything that is not a word of letters (a number, an id's digits).
_PHRASE_BREAK = re.compile(r"[^A-Za-z'\s-]+|\s-+\s")


def _question(words: str) -> str:
    return (words or "").split("\n\nContext:")[0]


def _known(results: list[dict[str, Any]], words: str) -> dict[str, Any]:
    """What the model may name in a call: gauges (id -> source or None), points, and search phrases."""
    gauges: dict[str, str | None] = {}
    points: list[tuple[float, float]] = []
    text = normalise(words or "")
    for m in _CTX_GAUGE.finditer(text):
        gauges.setdefault(m.group("id"), m.group("src"))
    for m in _SLASH_GAUGE.finditer(text):
        gauges.setdefault(m.group("id"), m.group("src"))
    for m in _WORD_ID.finditer(_DATE.sub(" ", _question(text))):
        if len(m.group(0)) >= 5 and sum(ch.isdigit() for ch in m.group(0)) >= 3:
            gauges.setdefault(m.group(0), None)
    for m in _COORD_PAIR.finditer(text):
        if re.search(r"centred on\s*$|centered on\s*$", text[max(0, m.start() - 14):m.start()]):
            continue   # the map's centre is where the view is, not a place the reader asked about
        lat, lon = float(m.group(1)), float(m.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            points.append((lat, lon))
    for r in results or []:
        c = r.get("compact") if isinstance(r, dict) else None
        if not isinstance(c, dict):
            continue
        rows = c.get("stations") if r.get("name") == "find_stations" else [c]
        for row in rows or []:
            if isinstance(row, dict) and row.get("station_id"):
                gauges[str(row["station_id"])] = str(row.get("source") or "") or gauges.get(str(row["station_id"]))
            if isinstance(row, dict) and isinstance(row.get("latitude"), (int, float)) \
                    and isinstance(row.get("longitude"), (int, float)):
                points.append((float(row["latitude"]), float(row["longitude"])))
    # Search phrases: runs of the question's words, names first ("Kingston upon Thames", "Kingston", "Thames");
    # all of them only when the question names nothing with a capital and no gauge is known yet. A run never
    # crosses punctuation, a gauge id or a number, so an id's letters are no phrase (check4 offered "USGS-" and
    # "Potomac USGS-"), and an opening verb is a stop word, not a name ("List", "Use").
    phrases: list[str] = []
    for named in ((True,) if gauges else (True, False)):
        for chunk in _PHRASE_BREAK.split(_WORD_ID.sub(" , ", _DATE.sub(" , ", _question(text)))):
            q = [w.strip("'-") for w in chunk.split() if w.strip("'-")]
            for n in (4, 3, 2, 1):
                for i in range(len(q) - n + 1):
                    span = q[i:i + n]
                    if span[0].lower() in _STOP or span[-1].lower() in _STOP or len(" ".join(span)) < 3:
                        continue
                    if named and not (span[0][0].isupper() and span[-1][0].isupper()):
                        continue
                    p = " ".join(span)
                    if p not in phrases:
                        phrases.append(p)
        if phrases:
            break
    analysed = {str(r["compact"].get("station_id")).lower() for r in results or []
                if isinstance(r, dict) and r.get("name") in _ANALYSIS_TOOLS and isinstance(r.get("compact"), dict)
                and r["compact"].get("station_id")}
    analysed |= {str((r.get("arguments") or {}).get("station_id")).lower() for r in results or []
                 if isinstance(r, dict) and r.get("name") == "analyze_station"}
    return {"gauges": gauges, "points": list(dict.fromkeys(points)), "phrases": phrases[:40],
            "analysed": analysed}


def _same_point(a: tuple[float, float], points: list[tuple[float, float]]) -> tuple[float, float] | None:
    return next((p for p in points if abs(p[0] - a[0]) <= 0.01 and abs(p[1] - a[1]) <= 0.01), None)


def _call_key(name: str, args: dict[str, Any]) -> str:
    return json.dumps([name, {k: args[k] for k in sorted(args)}], default=str)


def check_call(reply: Any, results: list[dict[str, Any]], words: str = "") -> dict[str, Any]:
    """Hold a tool call to this question: ``{"ok", "name", "arguments", "refused", "repeat"}``.

    The tool must be one of :data:`DEVICE_TOOLS`; a gauge id must be in the reader's words, the context line
    or an earlier result (its source is then the one that names it, not the model's); a point must be a pair
    the words or a result give; a search phrase must be the reader's own words. Other arguments are dropped.
    A call already made for this question (``repeat``) is not run again: the next step answers from what the
    results hold (#590: "analyze_station" three times with the same arguments, then no answer).
    """
    if not isinstance(reply, dict):
        return {"ok": False, "refused": "the reply was not a tool call"}
    name = reply.get("tool")
    if name not in DEVICE_TOOLS:
        return {"ok": False, "refused": f"{str(name)[:40]!r} is not one of the tools ({', '.join(DEVICE_TOOLS)})"}
    raw = reply.get("arguments") if isinstance(reply.get("arguments"), dict) else {}
    known = _known(results, words)
    args: dict[str, Any] = {}
    if name == "analyze_station":
        sid = str(raw.get("station_id") or "").strip()
        match = next((g for g in known["gauges"] if g.lower() == sid.lower()), None)
        if not match:
            return {"ok": False, "name": name, "refused": f"the gauge id {sid[:40]!r} is in neither your words nor "
                                                          "a result; search for the gauge first"}
        source = known["gauges"][match] or str(raw.get("source") or "").strip()
        if not source:
            return {"ok": False, "name": name, "refused": f"no source is known for {match}"}
        args = {"source": source, "station_id": match}
    elif name in ("anywhere", "describe_catchment"):
        try:
            pt = (float(raw.get("lat")), float(raw.get("lon")))
        except (TypeError, ValueError):
            return {"ok": False, "name": name, "refused": f"{name} needs lat and lon"}
        hit = _same_point(pt, known["points"])
        if not hit:
            return {"ok": False, "name": name, "refused": f"{coord(pt[0])}, {coord(pt[1])} is in neither your "
                                                          "words nor a result"}
        args = {"lat": hit[0], "lon": hit[1]}
    else:   # find_stations
        query = str(raw.get("query") or "").strip()
        near = raw.get("near")
        if query:
            if not any(p.lower() == query.lower() for p in known["phrases"]) and \
                    not re.search(r"(?<![\w])" + re.escape(query.lower()) + r"(?![\w])", _question(words).lower()):
                return {"ok": False, "name": name, "refused": f"the search {query[:40]!r} is not in your words"}
            args["query"] = query
        if isinstance(near, (list, tuple)) and len(near) == 2:
            try:
                hit = _same_point((float(near[0]), float(near[1])), known["points"])
            except (TypeError, ValueError):
                hit = None
            if not hit:
                return {"ok": False, "name": name, "refused": "the search point is in neither your words nor a result"}
            args["near"] = [hit[0], hit[1]]
        if not args:
            return {"ok": False, "name": name, "refused": "a search needs words from the question or a point"}
        try:
            limit = int(raw.get("limit") or 5)
        except (TypeError, ValueError):
            limit = 5
        args["limit"] = max(1, min(10, limit))
    key = _call_key(name, args)
    if any(_call_key(str(r.get("name")), r.get("arguments") or {}) == key for r in results or []
           if isinstance(r, dict)):
        return {"ok": False, "repeat": True, "name": name, "arguments": args,
                "refused": f"{name} already ran with these arguments; its result is in the facts"}
    return {"ok": True, "name": name, "arguments": args}


# ── each step: what the model may reply ───────────────────────────────────────

_COMPARE = re.compile(r"\b(?:compare[ds]?|comparing|comparison|versus|vs\.?|than|both|each|difference between)\b",
                      re.I)


def step_budget(words: str) -> int:
    """How many replies a question gets. One gauge or one place (a summary, a trend, a flood): a tool, the
    answer and one spare, 3. A comparison or a list: a search, two analyses, the answer and one spare, 5. The
    last reply may only be an answer, so a question never ends on a tool call (#590: 5 of 14 did)."""
    q = _question(words)
    return 5 if wanted_list(q) >= 2 or _COMPARE.search(q) else 3


def _tools_open(results: list[dict[str, Any]], known: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The tools a step may still call, each with the values it may take; a call already made is not offered."""
    done = {_call_key(str(r.get("name")), r.get("arguments") or {}) for r in results or [] if isinstance(r, dict)}
    out: dict[str, dict[str, Any]] = {}
    ids = [g for g in known["gauges"] if g.lower() not in known["analysed"]]
    if ids:
        out["analyze_station"] = {"station_id": ids}
    for tool in ("anywhere", "describe_catchment"):
        pts = [p for p in known["points"] if _call_key(tool, {"lat": p[0], "lon": p[1]}) not in done]
        if pts:
            out[tool] = {"points": pts}
    if known["phrases"] or known["points"]:
        out["find_stations"] = {"query": known["phrases"], "points": known["points"]}
    return out


def _arguments_schema(tools: dict[str, dict[str, Any]], known: dict[str, Any]) -> dict[str, Any]:
    """The ``arguments`` a step's tool call may hold: closed, and only the arguments the open tools take
    (:data:`DEVICE_TOOLS`), each an enum of what this question gives. A point is never a free number: ``lat``,
    ``lon`` and ``near`` take only the coordinates of the points the words or the results name, and are left out
    when there are none. ``limit`` is a whole number from 1 to 10, as an enum.

    The fourth check (2026-10-26) found the open object used: made-up search points ([51.0, -0.3], [51.47, 0.5]),
    extra keys ("limit__", "limittypicode"), and a number written for 176 s until Chrome's output limit. One flat
    object for every open tool, which a small model follows better than a ``oneOf`` and Chrome accepts as it
    accepted the others; :func:`check_call` then keeps only the arguments of the tool named, and holds a point
    to a pair the words or results give.
    """
    props: dict[str, Any] = {}
    ids = list(tools.get("analyze_station", {}).get("station_id", []))
    if ids:
        props["station_id"] = {"type": "string", "enum": ids}
        sources = sorted({s for s in known["gauges"].values() if s})
        if sources:
            props["source"] = {"type": "string", "enum": sources}
    search = tools.get("find_stations")
    if search and search["query"]:
        props["query"] = {"type": "string", "enum": list(search["query"])}
    pts = list(dict.fromkeys(p for t in ("anywhere", "describe_catchment") for p in tools.get(t, {}).get("points", [])))
    if pts:
        props["lat"] = {"type": "number", "enum": list(dict.fromkeys(p[0] for p in pts))}
        props["lon"] = {"type": "number", "enum": list(dict.fromkeys(p[1] for p in pts))}
    if search and search["points"]:
        coords = list(dict.fromkeys(v for p in search["points"] for v in p))
        props["near"] = {"type": "array", "items": {"type": "number", "enum": coords}, "minItems": 2, "maxItems": 2}
    if search:
        props["limit"] = {"type": "integer", "enum": list(range(1, 11))}
    return {"type": "object", "properties": props, "additionalProperties": False}


def step_schema(results: list[dict[str, Any]], words: str = "", *, step: int = 1, budget: int | None = None,
                force_answer: bool = False, notes: list[str] | None = None) -> dict[str, Any]:
    """The reply this step may give: ``{"mode", "schema", "prompt", "budget", "keys"}``.

    ``mode`` is "tool" on the first step (there are no facts yet), "answer" on the last, after a repeated call,
    or when no tool is left, and "either" between. ``schema`` is the responseConstraint, with every value an
    enum built here: the tool names still open, the gauge ids and sources and search phrases the words and
    results hold, and the fact keys of :func:`fact_table` with :data:`CONNECTIVES`. ``prompt`` is the step's
    message: the question, what has run, and the facts by key with the sentence each one writes. "none" means
    no tool can run and nothing can be answered (no gauge, place or point in the words). ``notes`` are told
    to the model (a call refused at the step before, a repeat not run).
    """
    budget = budget or step_budget(words)
    known = _known(results, words)
    table = fact_table(results)
    keys = [f["key"] for f in table["facts"]]
    tools = _tools_open(results, known)
    if not keys and not tools:
        mode = "none"
    elif force_answer or step >= budget or not tools:
        mode = "answer" if keys else "none"
    elif not keys:
        mode = "tool"
    else:
        mode = "either"
    props: dict[str, Any] = {}
    if mode in ("tool", "either"):
        props["tool"] = {"type": "string", "enum": list(tools)}
        props["arguments"] = _arguments_schema(tools, known)
    if mode in ("answer", "either"):
        props["answer"] = {"type": "array", "items": {"type": "string", "enum": keys + list(CONNECTIVES)},
                           "minItems": 1, "maxItems": MAX_PICKS}
    required = ["answer"] if mode == "answer" else ["tool", "arguments"] if mode == "tool" else []
    schema = {"type": "object", "properties": props, "required": required, "additionalProperties": False}
    return {"mode": mode, "schema": schema, "budget": budget, "keys": keys,
            "prompt": _step_prompt(words, results, table, tools, mode, notes or [])}


def _step_prompt(words: str, results: list[dict[str, Any]], table: dict[str, Any],
                 tools: dict[str, dict[str, Any]], mode: str, notes: list[str]) -> str:
    lines = [f"Question: {_question(words).strip()}"]
    ctx = (words or "").split("\n\nContext:", 1)
    if len(ctx) == 2 and ctx[1].strip():
        lines.append(f"Context: {ctx[1].strip()}")
    ran = [f"{r.get('name')}({json.dumps(r.get('arguments') or {})})" for r in results or [] if isinstance(r, dict)]
    if ran:
        lines.append("Already run, do not call again: " + "; ".join(ran) + ".")
    lines += [f"Note: {n}." for n in notes[-2:]]
    if mode in ("tool", "either") and tools:
        offer = []
        for name, vals in tools.items():
            if name == "analyze_station":
                offer.append(f"analyze_station with station_id one of {', '.join(vals['station_id'][:12])}")
            elif name == "find_stations":
                bits = []
                if vals["query"]:
                    bits.append("query: words from the question")
                if vals["points"]:
                    pts = "; ".join(f"[{a:g}, {b:g}]" for a, b in vals["points"][:6])
                    bits.append(f"near one of {pts}")
                offer.append(f"find_stations ({' or '.join(bits)}; limit up to 10)")
            else:
                pts = "; ".join(f"{a:g}, {b:g}" for a, b in vals["points"][:6])
                offer.append(f"{name} with lat, lon one of: {pts}")
        lines.append("Tools you may call: " + "; ".join(offer) + ".")
    if table["facts"] and mode in ("answer", "either"):
        lines.append("Facts you may answer with, by key (AquaScope writes each sentence):")
        lines += [f"{f['key']}: {_write([f], table['gauges'], short=True)[0]}" for f in table["facts"]]
        lines += [f"{k}: {v}" for k, v in CONNECTIVES.items()]
    if mode == "answer":
        lines.append('Reply now with {"answer": [the keys of the facts that answer the question, in order]}.')
    elif mode == "tool":
        lines.append('Reply with {"tool": "<name>", "arguments": {...}}.')
    else:
        lines.append('Reply with {"answer": [keys]} if the facts answer the question, else with one tool call '
                     '{"tool": "<name>", "arguments": {...}}.')
    return "\n".join(lines)


# ── the answer, written here from the keys the model picked ───────────────────


def render_answer(reply: Any, results: list[dict[str, Any]], words: str = "", *,
                  why: str | None = None) -> dict[str, Any]:
    """Write the answer from the fact keys the model picked: ``{"ok", "text", "sentences", "picked", "refused",
    "facts", "notes"}``.

    ``reply`` is the model's ``{"answer": [keys]}``. Every key must be in :func:`fact_table` for these results
    or in :data:`CONNECTIVES`; anything else (a key that is not there, free text, a tool call when only an
    answer may come) is refused before display, and the page then shows ``facts`` (the summary lines) with the
    refusal in one short line. ``why`` refuses with that reason (the steps ran out, the model stopped). The
    written text then goes through :func:`lock_answer` as a second line of defence.
    """
    table = fact_table(results)
    by_key = {f["key"]: f for f in table["facts"]}
    summary = _write([f for f in table["facts"] if f.get("default")], table["gauges"])[:12]
    notes = _search_notes(results, words)
    base = {"facts": summary, "notes": notes, "sentences": [], "picked": [], "text": ""}
    if why:
        return {**base, "ok": False, "refused": [why]}
    picks = reply.get("answer") if isinstance(reply, dict) else None
    if isinstance(picks, str):
        return {**base, "ok": False, "refused": ["the model wrote free text instead of picking facts by key"]}
    if not isinstance(picks, list) or not picks:
        what = "a tool call" if isinstance(reply, dict) and reply.get("tool") else "no facts"
        return {**base, "ok": False, "refused": [f"the model gave {what} where its answer should be"]}
    bad = [str(k)[:60] for k in picks if not isinstance(k, str) or (k not in by_key and k not in CONNECTIVES)]
    if bad:
        return {**base, "ok": False, "refused": [f"{', '.join(repr(b) for b in bad[:3])} "
                                                 f"{'is' if len(bad) == 1 else 'are'} not among the facts"]}
    ordered = list(dict.fromkeys(picks))[:MAX_PICKS]
    if not any(k in by_key for k in ordered):
        return {**base, "ok": False, "refused": ["the model picked no fact, only connecting sentences"]}
    sentences = []
    picked_facts: list[dict[str, Any]] = []
    for k in ordered:
        if k in by_key:
            picked_facts.append(by_key[k])
        else:
            picked_facts.append({"key": k, "gauge": None, "text": CONNECTIVES[k]})
    sentences = _write(picked_facts, table["gauges"])
    text = " ".join(sentences)
    second = lock_answer(text, results, words)
    if not second["ok"]:
        # The package's own sentences failed its own lock: a bug here, not the model's. Shown as refused.
        return {**base, "ok": False, "refused": ["AquaScope's own check stopped the answer: "
                                                 + "; ".join(second["refused"][:2])]}
    return {**base, "ok": True, "refused": [], "sentences": sentences, "picked": ordered, "text": text}


def _search_notes(results: list[dict[str, Any]], words: str) -> list[str]:
    notes = []
    given = _numbers(normalise(words or ""))
    for r in results or []:
        near = (r.get("arguments") or {}).get("near") if isinstance(r, dict) else None
        if r.get("name") == "find_stations" and isinstance(near, (list, tuple)) and len(near) == 2:
            if not all(isinstance(v, (int, float)) and _close(float(v), given) for v in near):
                notes.append(f"The search was near {coord(near[0])}, {coord(near[1])}, a point from an earlier "
                             "result, not from your words.")
    return notes
