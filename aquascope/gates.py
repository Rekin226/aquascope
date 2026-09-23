"""Gates: the checks a study step must pass before its result is worth quoting.

A gate is data, not code: ``{"check": "min_years", "value": 20, "path": "years"}``
written into a study step's ``expects`` list, evaluated by the runner against
the tool's payload after the step ran. The vocabulary is small and hydrologic,
and the thresholds come from the same registry (:mod:`aquascope.methods`)
the reconnaissance step quotes, so the plan and the report say the same
numbers.

Every check takes a ``path``: dotted into the payload (``ffa.fits.lp3.ci``),
with list indexes (``stations.0.name`` or ``stations[0].name``) and a selector
over a list of dicts (``sufficiency[method=gr4j_calibration].status``). A
check over a per-return-period list may carry ``return_period`` instead of an
index: the runner looks the index up in the payload's own ``return_periods``.

The result is a list of ``{"check", "passed", "detail"}`` so a report can print
every outcome, passed or not. An unknown check fails rather than passing
quietly: a typo in a gate must not read as a green light.
"""

from __future__ import annotations

import math
import re
from typing import Any

__all__ = ["CHECKS", "evaluate", "plain", "resolve_path"]

#: The check vocabulary (v1) and one line on each, for docs and the validator.
CHECKS: dict[str, str] = {
    "min_years": "the number at path is at least value (years of record)",
    "max_return_period_factor": "return_period is at most value times the years at path",
    "ci_finite": "the confidence interval(s) at path are finite numbers",
    "spread_within": "the relative spread between the numbers at paths is at most value",
    "nse_min": "the Nash-Sutcliffe efficiency at path is at least value",
    "kge_min": "the Kling-Gupta efficiency at path is at least value",
    "not_empty": "the value at path exists and is not empty",
    "unit_present": "a non-empty unit string sits at path (default: unit)",
    "max_area_km2": "the catchment area at path is at most value km2",
    "min_donors": "the donor count (or list) at path has at least value entries",
    "status_is": "the status at path equals value (or is in the list value)",
    "min_samples": "every sample count at path (a number, a list, or a dict of counts per parameter) is at least value",
    "fit_envelopes_max": "the fit's quantile at the record maximum's empirical return period (ffa at path) is "
                         "within value (relative) of the observed maximum",
    "sampling_density": "the record at path (a sampling block: n, span_years, per_year) is sampled densely "
                        "enough for the resolution value names (daily, weekly, monthly) or the rate value gives",
    "trend_on_series": "the Mann-Kendall p-value at path (a trend block) is at least value: no significant trend "
                       "in the series the test was run on",
    "cross_check_ratio": "the number at path against reference (a number, or a dict by return period), as a ratio "
                         "within 1 +/- value; skipped when the compared block says comparable: false",
}

_DEFAULT_PATH = {
    "min_years": "years",
    "max_return_period_factor": "years",
    "unit_present": "unit",
    "max_area_km2": "area_km2",
    "min_samples": "sample_counts",
    "fit_envelopes_max": "ffa",
    "sampling_density": "sampling",
    "trend_on_series": "ffa.amax_trend",
}

_MISSING = object()

_SELECTOR = re.compile(r"^([^\[\]]*)\[([^\]=]+)=([^\]]*)\]$")
_INDEX = re.compile(r"^([^\[\]]*)\[(-?\d+)\]$")


#: Observations a year each resolution label implies (mirrors aquascope.explore.RESOLUTION_PER_YEAR without the
#: import, so the gates stay dependency-free).
_RESOLUTION_PER_YEAR = {"daily": 365.0, "weekly": 52.0, "monthly": 12.0, "quarterly": 4.0, "annual": 1.0}


def _segments(path: str) -> list[str]:
    """Split a dotted path, keeping ``[...]`` selectors attached to their key."""
    out: list[str] = []
    buf = ""
    depth = 0
    for ch in path:
        if ch == "." and depth == 0:
            if buf:
                out.append(buf)
            buf = ""
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        buf += ch
    if buf:
        out.append(buf)
    return out


def _step_into(node: Any, seg: str) -> Any:
    """One path segment: a key, a list index or a ``key[field=value]`` selector."""
    m = _SELECTOR.match(seg)
    if m:
        key, field, want = m.group(1), m.group(2).strip(), m.group(3).strip().strip('"').strip("'")
        node = _step_into(node, key) if key else node
        if isinstance(node, list):
            for item in node:
                if isinstance(item, dict) and str(item.get(field)) == want:
                    return item
        return _MISSING
    m = _INDEX.match(seg)
    if m:
        key, index = m.group(1), int(m.group(2))
        node = _step_into(node, key) if key else node
        if isinstance(node, (list, tuple)):
            try:
                return node[index]
            except IndexError:
                return _MISSING
        return _MISSING
    if isinstance(node, dict):
        return node.get(seg, _MISSING)
    if isinstance(node, (list, tuple)) and re.fullmatch(r"-?\d+", seg):
        try:
            return node[int(seg)]
        except IndexError:
            return _MISSING
    return _MISSING


def resolve_path(payload: Any, path: str | None) -> Any:
    """The value at ``path`` in ``payload``, or ``None`` when the path leads nowhere."""
    if path in (None, "", "."):
        return payload
    node = payload
    for seg in _segments(str(path)):
        node = _step_into(node, seg)
        if node is _MISSING:
            return None
    return node


def _find_key(payload: Any, key: str) -> Any:
    """Depth-first search for ``key`` anywhere in a payload (``return_periods`` lives under ``ffa``)."""
    stack = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if key in item:
                return item[key]
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return None


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _number(x: Any) -> float | None:
    if _is_number(x):
        return float(x)
    if isinstance(x, str):
        try:
            v = float(x)
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    return None


def _at_return_period(value: Any, payload: Any, gate: dict[str, Any]) -> tuple[Any, str]:
    """Index a per-return-period list by the gate's ``return_period``; a note says which."""
    rp = gate.get("return_period")
    if rp is None or not isinstance(value, (list, tuple)):
        return value, ""
    periods = _find_key(payload, "return_periods")
    if not isinstance(periods, (list, tuple)):
        return None, f"no return_periods list in the payload to look up T = {rp:g}"
    try:
        idx = [float(p) for p in periods].index(float(rp))
    except ValueError:
        return None, f"T = {rp:g} years is not among the fitted return periods {list(periods)}"
    return (value[idx] if idx < len(value) else None), f"at T = {rp:g} years"


def _fmt(x: Any) -> str:
    if _is_number(x):
        return f"{float(x):,.4g}"
    return str(x)


def evaluate(expects: list[dict[str, Any]] | None, payload: Any) -> list[dict[str, Any]]:
    """Evaluate every gate in ``expects`` against ``payload``; one ``{check, passed, detail}`` each."""
    out: list[dict[str, Any]] = []
    for gate in expects or []:
        if not isinstance(gate, dict) or not gate.get("check"):
            out.append({"check": str(gate), "passed": False, "detail": "a gate must be a dict with a check"})
            continue
        name = str(gate["check"])
        try:
            passed, detail = _run_check(name, gate, payload)
        except Exception as exc:  # noqa: BLE001 - a broken gate is a failed gate, said out loud
            passed, detail = False, f"gate could not be evaluated: {type(exc).__name__}: {exc}"
        # None is a skip: the check does not apply to this payload (a model cell that is not the gauge's river)
        row = {"check": name, "passed": True if passed is None else bool(passed), "detail": detail}
        if passed is None:
            row["skipped"] = True
        if gate.get("path") is not None:
            row["path"] = gate["path"]
        if gate.get("paths") is not None:
            row["paths"] = list(gate["paths"])
        if gate.get("value") is not None:
            row["value"] = gate["value"]
        out.append(row)
    return out


def _not_comparable(payload: Any, path: str | None) -> str | None:
    """The note of the first block along ``path`` that says ``comparable: false``; None when none does."""
    if not path:
        return None
    segs = _segments(str(path))
    for i in range(1, len(segs) + 1):
        node = resolve_path(payload, ".".join(segs[:i]))
        if isinstance(node, dict) and node.get("comparable") is False:
            why = node.get("note") or (node.get("cell") or {}).get("why")
            return str(why or "the compared block is marked not comparable")
        if node is None:
            return None
    return None


def _run_check(name: str, gate: dict[str, Any], payload: Any) -> tuple[bool | None, str]:
    path = gate.get("path", _DEFAULT_PATH.get(name))
    value = gate.get("value")
    if isinstance(payload, dict) and payload.get("error") and name != "status_is":
        return False, f"the step returned an error: {payload['error']}"

    if name == "min_years":
        years = _number(resolve_path(payload, path))
        need = _number(value)
        if years is None or need is None:
            return False, f"no record length at {path!r}"
        ok = years >= need
        return ok, f"{years:g} years of record, {need:g} needed" + ("" if ok else ": too short")

    if name == "max_return_period_factor":
        years = _number(resolve_path(payload, path))
        factor = _number(value)
        rp = _number(gate.get("return_period"))
        if years is None or factor is None:
            return False, f"no record length at {path!r} to compare the return period with"
        if rp is None:
            return True, "no return period asked, nothing to compare (not applicable)"
        cap = factor * years
        ok = rp <= cap
        return ok, (
            f"T = {rp:g} years against a cap of about {cap:.0f} years ({factor:g} times {years:g} years of record)"
            + ("" if ok else ": beyond the cap, an extrapolation")
        )

    if name == "ci_finite":
        got, note = _at_return_period(resolve_path(payload, path), payload, gate)
        if got is None:
            return False, f"no confidence interval at {path!r}" + (f" ({note})" if note else "")
        pairs = got if (isinstance(got, (list, tuple)) and got and isinstance(got[0], (list, tuple))) else [got]
        bad = [p for p in pairs if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(_is_number(x) for x in p))]
        if bad:
            return False, f"confidence interval at {path!r} is not finite" + (f" ({note})" if note else "")
        widths = [f"[{_fmt(p[0])}, {_fmt(p[1])}]" for p in pairs[:3]]
        return True, "finite interval " + ", ".join(widths) + (f" {note}" if note else "")

    if name == "spread_within":
        paths = list(gate.get("paths") or ([path] if path else []))
        if len(paths) == 1 and "," in str(paths[0]):
            # a model often writes the two paths in one string, as the catalogue lists them
            paths = [p.strip() for p in str(paths[0]).split(",") if p.strip()]
        limit = _number(value)
        if limit is None:
            limit = 0.25  # the tolerance the flood playbook uses when a plan names none
        if len(paths) < 2:
            return False, "spread_within needs two or more paths (paths: [...], or one string with a comma)"
        nums: list[float] = []
        notes: list[str] = []
        for p in paths:
            got, note = _at_return_period(resolve_path(payload, p), payload, gate)
            n = _number(got)
            if n is None:
                return False, f"no number at {p!r}" + (f" ({note})" if note else "")
            nums.append(n)
            if note and note not in notes:
                notes.append(note)
        mean = sum(nums) / len(nums)
        if mean == 0:
            return False, "the values average to zero, the spread is undefined"
        spread = (max(nums) - min(nums)) / abs(mean)
        ok = spread <= limit
        return ok, (
            f"spread {spread:.0%} between {', '.join(_fmt(n) for n in nums)} "
            f"({limit:.0%} allowed)" + (f" {notes[0]}" if notes else "") + ("" if ok else ": the fits disagree")
        )

    if name in ("nse_min", "kge_min"):
        score = _number(resolve_path(payload, path))
        need = _number(value)
        label = "NSE" if name == "nse_min" else "KGE"
        if score is None or need is None:
            return False, f"no {label} at {path!r}"
        ok = score >= need
        return ok, f"{label} = {score:.2f}, {need:g} needed" + ("" if ok else ": the model does not beat the threshold")

    if name == "not_empty":
        got = resolve_path(payload, path)
        empty = got is None or (isinstance(got, (list, tuple, dict, str)) and len(got) == 0)
        if isinstance(got, float) and math.isnan(got):
            empty = True
        return (not empty), (f"{path!r} is present" if not empty else f"nothing at {path!r}")

    if name == "unit_present":
        got = resolve_path(payload, path)
        ok = isinstance(got, str) and bool(got.strip())
        return ok, (f"unit {got}" if ok else f"no unit at {path!r}")

    if name == "max_area_km2":
        area = _number(resolve_path(payload, path))
        ceiling = _number(value)
        if area is None or ceiling is None:
            return False, f"no area at {path!r}"
        ok = area <= ceiling
        return ok, (
            f"catchment of {area:,.0f} km2 against a ceiling of {ceiling:,.0f} km2"
            + ("" if ok else ": above the ceiling for a lumped model")
        )

    if name == "min_donors":
        got = resolve_path(payload, path)
        need = _number(value)
        count = float(len(got)) if isinstance(got, (list, tuple, dict)) else _number(got)
        if count is None or need is None:
            return False, f"no donor count at {path!r}"
        ok = count >= need
        return ok, f"{count:g} donors, {need:g} needed" + ("" if ok else ": too few for a transfer")

    if name == "status_is":
        got = resolve_path(payload, path)
        allowed = [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]
        ok = got is not None and str(got) in allowed
        return ok, f"status {got!r}, wanted {' or '.join(allowed)}"

    if name == "min_samples":
        got = resolve_path(payload, path)
        need = _number(value)
        counts: dict[str, float | None]
        if isinstance(got, dict):
            counts = {str(k): (float(len(v)) if isinstance(v, (list, tuple, dict)) else _number(v))
                      for k, v in got.items()}
        elif isinstance(got, (list, tuple)):
            counts = {str(i): (float(len(v)) if isinstance(v, (list, tuple, dict)) else _number(v))
                      for i, v in enumerate(got)}
        else:
            n = _number(got)
            counts = {"samples": n} if n is not None else {}
        known = {k: v for k, v in counts.items() if v is not None}
        if not known or need is None:
            return False, f"no sample counts at {path!r}"
        thin = [f"{k} ({v:g})" for k, v in known.items() if v < need]
        ok = not thin
        return ok, (f"{len(known)} parameter(s) with at least {need:g} samples each" if ok else
                    f"{need:g} samples per parameter needed, too few for {', '.join(thin[:6])}")

    if name == "fit_envelopes_max":
        ffa = resolve_path(payload, path)
        tol = _number(value)
        tol = 0.25 if tol is None else tol
        if not isinstance(ffa, dict) or not isinstance(ffa.get("record_max"), dict):
            return False, f"no record maximum at {path!r} (the flood payload carries ffa.record_max)"
        rec = ffa["record_max"]
        observed = _number(rec.get("value"))
        t_max = _number(rec.get("empirical_return_period"))
        fits = ffa.get("fits") if isinstance(ffa.get("fits"), dict) else {}
        wanted = gate.get("fit")
        names = [str(wanted)] if wanted else ["gev_lmoments", "lp3", "gev_bootstrap"]
        fit_name = next((f for f in names if isinstance(fits.get(f), dict)
                         and _number(fits[f].get("at_record_max")) is not None), None)
        if observed is None or fit_name is None:
            return False, "no fit evaluated at the record maximum's return period"
        fitted = float(_number(fits[fit_name]["at_record_max"]))
        if fitted <= 0:
            return False, f"the {fit_name} fit gives a non-positive quantile at the record maximum"
        ratio = observed / fitted
        ok = ratio <= 1.0 + tol
        where = f"T about {t_max:g} years" if t_max else "its empirical return period"
        return ok, (
            f"record maximum {_fmt(observed)} ({rec.get('year')}, {where}) against the {fit_name} fit's "
            f"{_fmt(fitted)} there: ratio {ratio:.2f} ({1 + tol:.2f} allowed)"
            + ("" if ok else ": the largest observed event sits above the fit")
        )

    if name == "sampling_density":
        block = resolve_path(payload, path)
        if not isinstance(block, dict) or _number(block.get("per_year")) is None:
            return False, f"no sampling block at {path!r} (n, span_years, per_year)"
        per_year = float(_number(block["per_year"]))
        label = str(value).strip().lower() if isinstance(value, str) else None
        need = _number(value) if label is None else _RESOLUTION_PER_YEAR.get(label)
        if need is None:
            return False, ("sampling_density needs a resolution (daily, weekly, monthly) or a rate a year, "
                           f"not {value!r}")
        floor = 0.55 * need if label else float(need)
        ok = per_year >= floor
        claimed = f"{label} claimed" if label else f"{need:g} a year needed"
        return ok, (
            f"{block.get('n')} observations in {block.get('span_years')} years: {per_year:g} a year, "
            f"about {block.get('inferred_resolution')}; {claimed}"
            + ("" if ok else ": the record is sparser than the resolution assumed")
        )

    if name == "trend_on_series":
        block = resolve_path(payload, path)
        threshold = _number(value)
        threshold = 0.05 if threshold is None else threshold
        if not isinstance(block, dict) or _number(block.get("p_value")) is None:
            return False, f"no trend test at {path!r} (p_value, tau)"
        p_value = float(_number(block["p_value"]))
        tau = _number(block.get("tau"))
        series = block.get("on") or path
        ok = p_value >= threshold
        p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3g}"
        return ok, (
            f"Mann-Kendall on the {series}: {p_text}" + (f", tau = {tau:.2f}" if tau is not None else "")
            + (f": no trend at the {threshold:g} level" if ok
               else f": a significant trend in the {series} ({threshold:g} level); a stationary estimate needs a "
                    "caveat")
        )

    if name == "cross_check_ratio":
        skip = _not_comparable(payload, path)
        if skip:
            return None, f"skipped: {skip}"
        got, note = _at_return_period(resolve_path(payload, path), payload, gate)
        ref = gate.get("reference")
        rp = gate.get("return_period")
        if isinstance(got, dict) and rp is not None:
            got = got.get(f"{float(rp):g}", got.get(str(rp)))
            note = f"at T = {float(rp):g} years"
        if isinstance(ref, dict) and rp is not None:
            ref = ref.get(f"{float(rp):g}", ref.get(str(rp)))
        elif isinstance(ref, (list, tuple)):
            ref, _ = _at_return_period(ref, payload, gate)
        a, b = _number(got), _number(ref)
        tol = _number(value)
        tol = 0.5 if tol is None else tol
        if a is None:
            return False, f"no number at {path!r}" + (f" ({note})" if note else "")
        if b is None:
            return False, "no reference number to compare with (the gate's reference did not resolve)"
        if b == 0:
            return False, "the reference is zero, the ratio is undefined"
        ratio = a / b
        ok = (1.0 / (1.0 + tol)) <= ratio <= (1.0 + tol)
        return ok, (
            f"{_fmt(a)} against the reference {_fmt(b)}" + (f" {note}" if note else "") + f": ratio {ratio:.2f} "
            f"(within a factor {1 + tol:.2f} allowed)" + ("" if ok else ": the cross-check disagrees")
        )

    return False, f"unknown check {name!r}; known: {', '.join(CHECKS)}"


# ── the plan in plain words ─────────────────────────────────────────────────

#: What a payload path holds, in words, for ``not_empty`` (the last segment is looked up first).
_PATH_WORDS = {
    "points": "observations", "series": "a series", "stations": "at least one gauge", "sub_basin": "a catchment",
    "indices": "drought indices", "spi": "an SPI value", "spei": "an SPEI value", "n": "some rows",
    "trend": "a trend test", "percentiles": "flow percentiles", "estimates": "signature estimates",
    "skill": "a skill score", "reliability": "a reliability figure", "fdc": "a flow-duration curve",
    "climate": "climate data", "glofas": "GloFAS discharge", "sgi": "a groundwater index", "score": "a score",
    "gross_irrigation_mm": "an irrigation depth", "peak_month_m3s": "a peak-month flow", "months": "a season",
    "n_records": "some rows", "k": "donor gauges",
}

#: The tolerance a check uses when the gate names none (as in :func:`_run_check`).
_DEFAULT_VALUE = {"spread_within": 0.25, "fit_envelopes_max": 0.25, "trend_on_series": 0.05,
                  "cross_check_ratio": 0.5}


def _words_for(path: Any) -> str:
    text = str(path or "").strip()
    if not text:
        return "a result"
    last = re.sub(r"\[.*\]$", "", _segments(text)[-1])
    return _PATH_WORDS.get(last) or last.replace("_", " ")


def _num(x: Any) -> str:
    """A threshold as a person writes it: 20, 0.5, 1,000."""
    n = _number(x)
    if n is None:
        return str(x)
    return f"{int(n):,}" if float(n).is_integer() else f"{n:g}"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%" if abs(x * 100 - round(x * 100)) < 1e-9 else f"{x * 100:g}%"


def plain(gate: dict[str, Any] | Any) -> str:
    """One gate as one plain sentence, for a plan a person reads: ``{"check": "min_years", "value": 20}`` reads
    "needs at least 20 years of record". The raw gate stays the reference; this is only its wording. An
    unknown check reads as its name and value, so nothing in a plan is hidden by the translation."""
    if not isinstance(gate, dict) or not gate.get("check"):
        return str(gate)
    name = str(gate["check"])
    value = gate.get("value")
    if value is None:
        value = _DEFAULT_VALUE.get(name)
    rp = _number(gate.get("return_period"))
    at_t = f" at the {_num(rp)}-year level" if rp is not None else ""
    v = _number(value)

    if name == "min_years":
        return f"needs at least {_num(value)} years of record"
    if name == "max_return_period_factor":
        if rp is not None:
            return f"the {_num(rp)}-year estimate may not exceed {_num(value)}x the record length"
        return f"a return period may not exceed {_num(value)}x the record length"
    if name == "ci_finite":
        return "the confidence interval must be a finite range" + at_t
    if name == "spread_within":
        return (f"the fitted distributions must agree within {_pct(v)}" if v is not None
                else "the fitted distributions must agree") + at_t
    if name == "nse_min":
        return f"the model must reach a Nash-Sutcliffe efficiency of at least {_num(value)}"
    if name == "kge_min":
        return f"the model must reach a Kling-Gupta efficiency of at least {_num(value)}"
    if name == "not_empty":
        return f"must return {_words_for(gate.get('path'))}"
    if name == "unit_present":
        return "must report its unit"
    if name == "max_area_km2":
        return f"the catchment may be at most {_num(value)} km2 for a lumped model"
    if name == "min_donors":
        return f"needs at least {_num(value)} donor gauges"
    if name == "status_is":
        allowed = [str(x) for x in value] if isinstance(value, (list, tuple)) else [str(value)]
        return f"the status must be {' or '.join(a.replace('_', ' ') for a in allowed)}"
    if name == "min_samples":
        return f"needs at least {_num(value)} samples per parameter"
    if name == "fit_envelopes_max":
        return (f"the fitted curve must reach the largest flood on record (within {_pct(v)})" if v is not None
                else "the fitted curve must reach the largest flood on record")
    if name == "sampling_density":
        if isinstance(value, str) and value.strip():
            return f"needs a record sampled about {value.strip().lower()}"
        return f"needs at least {_num(value)} observations a year"
    if name == "trend_on_series":
        series = "flood peaks" if "amax" in str(gate.get("path") or "ffa.amax_trend") else "series"
        return (f"the {series} must show no significant trend (at the {_pct(v)} level)" if v is not None
                else f"the {series} must show no significant trend")
    if name == "cross_check_ratio":
        return (f"must agree with the earlier estimate within a factor of {1 + v:g}" if v is not None
                else "must agree with the earlier estimate") + at_t
    return f"{name} {_fmt(value)}" if value is not None else name
