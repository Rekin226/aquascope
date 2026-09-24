"""Study links: an approved plan, small enough to travel in a URL, checked before it runs again.

A study made in the Explorer ends with a bundle. A link is the lighter thing
to send: it carries the plan (the question, the site, the intake and the
steps with their registered tools, arguments, methods and gates), never the
results, so whoever opens it reruns the numbers in their own browser with no
key. The same plan can arrive as a ``study.yaml`` (a bundle's, or the CLI's).

Both arrive from someone else, so both are untrusted. :func:`open_link` and
:func:`open_study_yaml` decode, cap the size and the step count, and check
every step against the method catalogue (registered tools only, their own
arguments, known gates and registry methods) before a page shows the plan.
The Studio checks the plan again when it is approved, so a step that passes
here and fails at the site is still pruned there, never run.

The token is ``z1.<base64url of raw deflate of compact JSON>`` (``j1.`` for
uncompressed JSON). Plain functions over dicts, stdlib only: the browser
worker imports this module.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import zlib
from typing import Any

__all__ = [
    "LINK_VERSION", "MAX_LINK_CHARS", "MAX_STEPS", "MAX_STUDY_BYTES",
    "compact_from_study", "compact_from_workspace", "decode", "encode", "expand",
    "link_from_workspace", "open_link", "open_study_yaml", "studio_op", "validate",
]

LINK_VERSION = 1
#: The token's length in characters, the part after ``study=``. Chat apps and mail clients keep links far
#: longer than this, and a twelve-step plan compresses to about a third of it.
MAX_LINK_CHARS = 8000
#: The same cap as the Methodologist's: a plan longer than a model may write is not a plan the crew runs.
MAX_STEPS = 12
#: A fetched ``study.yaml`` (and a decompressed token) larger than this is refused before it is parsed.
MAX_STUDY_BYTES = 200_000
_MAX_TEXT = 2000
_MAX_RATIONALE = 400
_MAX_INTAKE_KEYS = 24

_PREFIX_DEFLATE = "z1."
_PREFIX_JSON = "j1."


class LinkError(ValueError):
    """A link or a study file that cannot be opened, in words a reader can act on."""


# ── the compact plan ────────────────────────────────────────────────────────


def _step(raw: dict[str, Any], i: int, *, rationale: bool = True) -> dict[str, Any]:
    """One step with only what reruns it: the tool, its arguments, the method, the gates, the fallback and the
    dependencies. Outputs are left out (the Methodologist derives them again), results always."""
    out: dict[str, Any] = {"id": str(raw.get("id") or f"s{i}"), "tool": str(raw.get("tool") or "")}
    args = raw.get("arguments")
    if isinstance(args, dict) and args:
        out["arguments"] = dict(args)
    for key in ("method",):
        if raw.get(key):
            out[key] = str(raw[key])
    if raw.get("expects"):
        out["expects"] = [dict(g) for g in raw["expects"] if isinstance(g, dict)]
    fb = raw.get("fallback")
    if fb == "stop" or (isinstance(fb, dict) and isinstance(fb.get("step"), dict)):
        out["fallback"] = fb
    if raw.get("depends_on"):
        out["depends_on"] = [str(d) for d in raw["depends_on"] if d]
    if rationale and raw.get("rationale"):
        out["rationale"] = str(raw["rationale"])[:_MAX_RATIONALE]
    return out


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _site(lat: Any, lon: Any, name: Any = None) -> dict[str, Any]:
    site: dict[str, Any] = {"lat": lat, "lon": lon}
    if name:
        site["name"] = str(name)[:120]
    return site


def compact_from_study(study: dict[str, Any], *, lat: float | None = None, lon: float | None = None,
                       text: str | None = None, intake: dict[str, Any] | None = None,
                       name: str | None = None) -> dict[str, Any]:
    """The compact plan of a study dict (a ``study.yaml`` read, or a workspace's ``study``). The site, the
    question and the intake come from the study's ``problem`` block unless given."""
    problem = _dict(study.get("problem"))
    site = _dict(problem.get("site"))
    plan = _dict(study.get("plan"))
    compact: dict[str, Any] = {
        "v": LINK_VERSION,
        "q": str(text if text is not None else (problem.get("text") or study.get("question") or "")),
        "site": _site(lat if lat is not None else site.get("lat"), lon if lon is not None else site.get("lon"), name),
        "steps": [_step(s, i) for i, s in enumerate(study.get("steps") or [], 1) if isinstance(s, dict)],
    }
    given = intake if intake is not None else problem.get("params")
    if isinstance(given, dict):
        kept = {str(k): v for k, v in given.items() if v is not None}
        if kept:
            compact["intake"] = kept
    for key in ("objective", "decision", "playbook"):
        if plan.get(key):
            compact.setdefault("plan", {})[key] = str(plan[key])[:_MAX_TEXT]
    return compact


def compact_from_workspace(ws: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
    """The compact plan of a Studio workspace dict (the page's copy): its approved study at its site, with the
    brief's problem and intake so the recipient's Consultant reads the same brief."""
    study = ws.get("study")
    if not isinstance(study, dict) or not study.get("steps"):
        raise LinkError("this study has no plan to share yet")
    site = _dict(ws.get("site"))
    brief = _dict(ws.get("brief"))
    compact = compact_from_study(study, lat=site.get("lat"), lon=site.get("lon"),
                                 text=str(brief.get("problem") or study.get("question") or ""),
                                 intake=brief.get("intake") if isinstance(brief.get("intake"), dict) else None,
                                 name=name)
    if ws.get("tables"):
        compact["tables"] = sorted(str(k) for k in ws["tables"])[:8]
    return compact


# ── the token ───────────────────────────────────────────────────────────────


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    try:
        return base64.b64decode(text + "=" * (-len(text) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        raise LinkError("the link is damaged: it is not valid base64") from None


def _deflate(data: bytes) -> bytes:
    c = zlib.compressobj(9, zlib.DEFLATED, -15)
    return c.compress(data) + c.flush()


def _inflate(data: bytes) -> bytes:
    d = zlib.decompressobj(-15)
    try:
        out = d.decompress(data, MAX_STUDY_BYTES + 1)
    except zlib.error:
        raise LinkError("the link is damaged: it does not decompress") from None
    if len(out) > MAX_STUDY_BYTES or d.unconsumed_tail:
        raise LinkError("the link unpacks to more than a study plan can hold")
    return out


def _dumps(compact: dict[str, Any]) -> bytes:
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def encode(compact: dict[str, Any], *, max_chars: int = MAX_LINK_CHARS) -> str:
    """The token for a compact plan. Rationales are dropped when the link would otherwise be too long; a plan
    that is still too long raises :class:`LinkError`."""
    for attempt in (compact, {**compact, "steps": [{k: v for k, v in s.items() if k != "rationale"}
                                                   for s in compact.get("steps") or []]}):
        token = _PREFIX_DEFLATE + _b64(_deflate(_dumps(attempt)))
        if len(token) <= max_chars:
            return token
    raise LinkError(f"this plan is too long for a link ({len(token)} characters, at most {max_chars}); "
                    "share the bundle's study.yaml instead")


def decode(token: str, *, max_chars: int = MAX_LINK_CHARS) -> dict[str, Any]:
    """The compact plan in a token. Raises :class:`LinkError` on anything that is not one."""
    token = str(token or "").strip()
    if not token:
        raise LinkError("the link carries no study")
    if len(token) > max_chars:
        raise LinkError(f"the link is too long ({len(token)} characters, at most {max_chars})")
    if token.startswith(_PREFIX_DEFLATE):
        raw = _inflate(_unb64(token[len(_PREFIX_DEFLATE):]))
    elif token.startswith(_PREFIX_JSON):
        raw = _unb64(token[len(_PREFIX_JSON):])
    else:
        raise LinkError("the link is not a study link this version of AquaScope reads")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LinkError("the link is damaged: it does not hold a study plan") from None
    if not isinstance(data, dict):
        raise LinkError("the link does not hold a study plan")
    return data


# ── the check ───────────────────────────────────────────────────────────────


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def validate(compact: dict[str, Any]) -> list[str]:
    """Every reason the compact plan cannot be offered for a rerun, in plain words. Empty means it can.

    Checks the shape (version, question, site, intake, steps), the caps, and every step against the method
    catalogue: a registered tool, its own arguments, known gates, a registry method the tool applies,
    references only to earlier steps. The catalogue check runs without repairs, so what is shown is what
    was sent."""
    errors: list[str] = []
    if not isinstance(compact, dict):
        return ["the study is not a mapping"]
    v = compact.get("v")
    if v != LINK_VERSION:
        errors.append(f"the link is version {v!r}; this AquaScope reads version {LINK_VERSION}")
    q = compact.get("q")
    if not isinstance(q, str) or not q.strip():
        errors.append("the study has no question")
    elif len(q) > _MAX_TEXT:
        errors.append(f"the question is longer than {_MAX_TEXT} characters")
    site = compact.get("site")
    lat = _number(site.get("lat")) if isinstance(site, dict) else None
    lon = _number(site.get("lon")) if isinstance(site, dict) else None
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        errors.append("the study has no valid site (a latitude and a longitude)")
    intake = compact.get("intake")
    if intake is not None and (not isinstance(intake, dict) or len(intake) > _MAX_INTAKE_KEYS):
        errors.append("the intake is not a short mapping")
    plan = compact.get("plan")
    if plan is not None and not isinstance(plan, dict):
        errors.append("the plan block is not a mapping")
    steps = compact.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("the study has no steps")
        return errors
    if len(steps) > MAX_STEPS:
        errors.append(f"the study has {len(steps)} steps; at most {MAX_STEPS}")
        return errors
    if not all(isinstance(s, dict) for s in steps):
        errors.append("a step is not a mapping")
        return errors
    from aquascope.studio import catalogue

    checked = json.loads(json.dumps(steps))   # the catalogue check must not touch what the page shows
    errors += catalogue.validate_plan(checked, repair=False)
    return errors


def expand(compact: dict[str, Any]) -> dict[str, Any]:
    """What a page needs to rerun a checked plan: where, the question, the intake, and the plan in the shape
    ``Studio.approve(plan=...)`` takes (the objective and decision, the steps). Only known keys pass."""
    site = compact.get("site") or {}
    plan_block = _dict(compact.get("plan"))
    steps = [_step(s, i) for i, s in enumerate(compact.get("steps") or [], 1)]
    plan: dict[str, Any] = {"steps": steps}
    for key in ("objective", "decision"):
        if plan_block.get(key):
            plan[key] = str(plan_block[key])
    return {
        "text": str(compact.get("q") or ""),
        "lat": float(site["lat"]), "lon": float(site["lon"]),
        "name": str(site.get("name") or "") or None,
        "intake": dict(compact.get("intake") or {}),
        "playbook": str(plan_block.get("playbook") or "") or None,
        "tables": [str(t) for t in (compact.get("tables") or []) if isinstance(t, str)][:8],
        "plan": plan,
    }


def _opened(compact: dict[str, Any]) -> dict[str, Any]:
    errors = validate(compact)
    if errors:
        return {"ok": False, "errors": errors[:12]}
    study = expand(compact)
    notes = []
    if study["tables"]:
        notes.append(f"This study read a table of its own ({', '.join(study['tables'])}); the link does not "
                     "carry it, so the crew will ask for one.")
    return {"ok": True, "errors": [], "notes": notes, "study": study}


def open_link(token: str) -> dict[str, Any]:
    """``{"ok", "errors", "notes", "study"}`` for a token from a URL: decoded, capped and checked."""
    try:
        compact = decode(token)
    except LinkError as exc:
        return {"ok": False, "errors": [str(exc)]}
    return _opened(compact)


def open_study_yaml(text: str) -> dict[str, Any]:
    """The same for a ``study.yaml`` fetched from a URL: capped, parsed without PyYAML's full loader, checked."""
    from aquascope.study import parse_block_yaml

    text = str(text or "")
    if len(text.encode("utf-8")) > MAX_STUDY_BYTES:
        return {"ok": False, "errors": [f"the study file is larger than {MAX_STUDY_BYTES // 1000} kB"]}
    try:
        data = parse_block_yaml(text)
    except Exception as exc:  # noqa: BLE001 - untrusted text: any parse failure is a refusal, never a crash
        return {"ok": False, "errors": [f"the study file is not readable YAML: {exc}"]}
    if not isinstance(data, dict):
        return {"ok": False, "errors": ["the study file is not a study (a YAML mapping)"]}
    try:
        compact = compact_from_study(data)
    except (TypeError, AttributeError, ValueError) as exc:
        return {"ok": False, "errors": [f"the study file is not a study: {exc}"]}
    return _opened(compact)


def link_from_workspace(ws: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
    """``{"ok", "token", "chars", "errors"}``: the link for a Studio workspace, checked as a recipient would
    check it, so a link that would not open is never handed out."""
    try:
        compact = compact_from_workspace(ws, name=name)
        token = encode(compact)
    except LinkError as exc:
        return {"ok": False, "errors": [str(exc)]}
    opened = open_link(token)
    if not opened["ok"]:
        return {"ok": False, "errors": opened["errors"]}
    return {"ok": True, "token": token, "chars": len(token), "errors": []}


def studio_op(a: dict[str, Any]) -> dict[str, Any]:
    """The worker's face: ``{"op": "link", "workspace", "name"}`` or ``{"op": "open_link", "token" | "yaml"}``."""
    op = a.get("op")
    if op == "link":
        return link_from_workspace(a.get("workspace") or {}, name=a.get("name") or None)
    if op == "open_link":
        if a.get("yaml") is not None:
            return open_study_yaml(str(a.get("yaml")))
        return open_link(str(a.get("token") or ""))
    return {"ok": False, "errors": [f"unknown op {op!r}"]}
