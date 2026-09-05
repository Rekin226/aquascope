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

__all__ = ["CHECKS", "evaluate", "resolve_path"]

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
}

_DEFAULT_PATH = {
    "min_years": "years",
    "max_return_period_factor": "years",
    "unit_present": "unit",
    "max_area_km2": "area_km2",
    "min_samples": "sample_counts",
}

_MISSING = object()

_SELECTOR = re.compile(r"^([^\[\]]*)\[([^\]=]+)=([^\]]*)\]$")
_INDEX = re.compile(r"^([^\[\]]*)\[(-?\d+)\]$")


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
        row = {"check": name, "passed": bool(passed), "detail": detail}
        if gate.get("path") is not None:
            row["path"] = gate["path"]
        if gate.get("paths") is not None:
            row["paths"] = list(gate["paths"])
        if gate.get("value") is not None:
            row["value"] = gate["value"]
        out.append(row)
    return out


def _run_check(name: str, gate: dict[str, Any], payload: Any) -> tuple[bool, str]:
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
            return False, "the gate names no return_period"
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
        limit = _number(value)
        if len(paths) < 2 or limit is None:
            return False, "spread_within needs two or more paths and a value"
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

    return False, f"unknown check {name!r}; known: {', '.join(CHECKS)}"
