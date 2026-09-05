"""Playbooks: the method-selection scaffold as data (#307).

A playbook is a YAML file that says, for one class of problem, which method
chain to run for the data that actually exists at a site: branches over the
reconnaissance dict (``assess_site``), study-v2 steps with gates and
fallbacks, the sentences it prints when it declines, the caveats every
report carries verbatim, and the citations. The tree runs with no model at
all: ``plan(playbook, recon, intake)`` returns a :class:`aquascope.study.Study`
the runner executes, and the plan-first Analyst (``aquascope solve``) uses the
same tree to constrain a model when one is present.

Preconditions are not repeated here: a step that names its ``method`` is
checked against :mod:`aquascope.methods` at plan time, so a method the
registry calls not defensible at this site is refused before anything runs
(the #273 failure class, a lumped model on a 100,000 km2 catchment).

Placeholders in step arguments and prose: ``{{ intake.<field> }}``,
``{{ station.source }}``, ``{{ station.station_id }}``, ``{{ station.name }}``,
``{{ station.years }}``, ``{{ site.lat }}``, ``{{ site.lon }}`` and
``{{ derived.<key> }}``, all resolved when the plan is filled. A step may also
take a number an earlier step computed: ``{{ result.<step id>.<dotted path> }}``
is left in the study and resolved by the runner against that step's payload
(an irrigation demand feeding a supply check). Conditions (``when``) are
evaluated over the recon dict extended with ``intake``, ``station``, ``site``
and ``derived`` (record lengths, the return-period cap from the registry,
donors, dams, whether temperature is reachable). A step names the variable
its station carries with ``station_variable`` when it differs from the
branch's (a well next to a rain gauge).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aquascope.study import Step, Study

__all__ = [
    "Branch",
    "Declined",
    "Playbook",
    "PlaybookError",
    "as_json",
    "coerce_intake",
    "describe",
    "evaluation_context",
    "fill_intake",
    "list_playbooks",
    "load",
    "plan",
    "select_branch",
    "validate",
]

PLAYBOOK_DIR = Path(__file__).parent

OPERATORS = ("==", "!=", ">=", "<=", ">", "<", "in", "exists")
INTAKE_TYPES = ("int", "float", "str", "bool", "choice", "list")
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_]+)\.([A-Za-z0-9_]+(?:\.[A-Za-z0-9_\[\]=-]+)*)\s*\}\}")
_NAMESPACES = ("intake", "station", "site", "derived")
#: Namespaces the plan leaves in place for the runner (``{{ result.s2.demand.mean_m3s }}``).
_DEFERRED = ("result",)


class PlaybookError(ValueError):
    """A playbook file that does not follow the schema."""


class Declined(Exception):  # noqa: N818 - a decline is a verdict the playbook prints, not an error
    """The playbook refuses this problem at this site, with the sentence it prints.

    ``kind`` is ``declined`` (a decline rule matched), ``no_branch`` (no branch
    applies) or ``refused`` (the registry calls a required method not
    defensible here).
    """

    def __init__(self, reason: str, *, kind: str = "declined", playbook: str | None = None,
                 branch: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind
        self.playbook = playbook
        self.branch = branch


class Condition(BaseModel):
    path: str
    op: str = "=="
    value: Any = None


class IntakeField(BaseModel):
    name: str
    label: str | None = None
    type: str = "str"
    default: Any = None
    options: list[Any] = Field(default_factory=list)
    required: bool = False
    help: str | None = None
    #: Bounds for an int or float field (inclusive); a value outside them is out of range.
    min: float | None = None
    max: float | None = None


class StepTemplate(BaseModel):
    id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
    method: str | None = None
    expects: list[dict[str, Any]] = Field(default_factory=list)
    fallback: dict[str, Any] | str | None = None
    depends_on: list[str] = Field(default_factory=list)
    #: Dropped (with a note) rather than refused when the registry says not defensible.
    optional: bool = False
    #: The variable this step's station must carry, when it differs from the branch's (a well beside a rain gauge).
    station_variable: str | None = None


class Branch(BaseModel):
    id: str
    when: list[Condition] = Field(default_factory=list)
    rationale: str | None = None
    #: The variable the branch's station must carry (default: the playbook's).
    station_variable: str | None = None
    steps: list[StepTemplate]


class Decline(BaseModel):
    when: list[Condition]
    say: str


class Caveat(BaseModel):
    say: str
    when: list[Condition] = Field(default_factory=list)


class Playbook(BaseModel):
    id: str
    title: str
    problem: str
    description: str | None = None
    #: The variable the problem is mostly about (picks the station for placeholders).
    variable: str | None = None
    version: int = 1
    intake: list[IntakeField] = Field(default_factory=list)
    branches: list[Branch]
    declines: list[Decline] = Field(default_factory=list)
    caveats: list[str | Caveat] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    def branch(self, branch_id: str) -> Branch | None:
        return next((b for b in self.branches if b.id == branch_id), None)


# ── loading ─────────────────────────────────────────────────────────────────


def _files() -> list[Path]:
    return sorted(p for p in PLAYBOOK_DIR.glob("*.yaml") if not p.name.startswith("_"))


def list_playbooks() -> list[dict[str, Any]]:
    """Every playbook shipped with the package: id, title, problem, branches and intake fields."""
    out = []
    for path in _files():
        try:
            pb = load(path)
        except (PlaybookError, OSError) as exc:  # a broken file is listed, not hidden
            out.append({"id": path.stem, "error": str(exc), "file": str(path)})
            continue
        out.append({
            "id": pb.id, "title": pb.title, "problem": pb.problem,
            "description": pb.description,
            "branches": [b.id for b in pb.branches],
            "intake": [f.name for f in pb.intake],
            "declines": len(pb.declines),
            "file": path.name,
        })
    return out


def as_json() -> str:
    """The playbooks with their intake fields, as the JSON the Explorer ships (``explorer/playbooks.json``).

    The page draws its problem chips and intake inputs from this file before
    Python has booted in the browser; ``python -m aquascope.playbooks`` writes
    it, and a test keeps it in step with the YAML files.
    """
    rows = []
    for row in list_playbooks():
        if "error" in row:
            continue
        pb = load(row["id"])
        rows.append({
            "id": pb.id, "title": pb.title, "problem": pb.problem, "description": pb.description,
            "variable": pb.variable,
            "intake": [f.model_dump() for f in pb.intake],
            "branches": [b.id for b in pb.branches],
        })
    return json.dumps({"playbooks": rows}, indent=2, ensure_ascii=False) + "\n"


def load(playbook: str | Path | dict[str, Any] | Playbook) -> Playbook:
    """A playbook by id (``flood_risk``), file path, dict or instance."""
    if isinstance(playbook, Playbook):
        return playbook
    if isinstance(playbook, dict):
        data = playbook
    else:
        path = Path(playbook)
        if not path.suffix:
            path = PLAYBOOK_DIR / f"{path.name}.yaml"
        if not path.exists():
            known = ", ".join(p.stem for p in _files())
            raise PlaybookError(f"no playbook {str(playbook)!r}; known: {known}")
        from aquascope.study import _parse_yaml

        data = _parse_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PlaybookError("a playbook must be a YAML mapping")
    try:
        return Playbook.model_validate(data)
    except Exception as exc:  # noqa: BLE001 - pydantic's message, in our exception type
        raise PlaybookError(f"playbook {data.get('id', '?')!r} does not follow the schema: {exc}") from None


def describe(playbook: str | Playbook) -> dict[str, Any]:
    """The playbook as plain dicts, for MCP, the CLI and the page."""
    return load(playbook).model_dump()


# ── validation ──────────────────────────────────────────────────────────────


def validate(playbook: str | Playbook | dict[str, Any]) -> list[str]:
    """Every problem with a playbook beyond its shape, as sentences (empty when it is sound)."""
    from aquascope.gates import CHECKS
    from aquascope.methods import METHODS
    from aquascope.study import tool_names

    pb = load(playbook)
    errors: list[str] = []
    tools = set(tool_names())
    intake_names = {f.name for f in pb.intake}
    for f in pb.intake:
        if f.type not in INTAKE_TYPES:
            errors.append(f"intake {f.name}: type {f.type!r} is not one of {INTAKE_TYPES}")
        if f.type == "choice" and not f.options:
            errors.append(f"intake {f.name}: a choice needs options")
        if f.type == "choice" and f.default is not None and f.default not in f.options:
            errors.append(f"intake {f.name}: default {f.default!r} is not among its options")
        if f.type == "list" and f.default is not None and not isinstance(f.default, (list, tuple, str)):
            errors.append(f"intake {f.name}: a list default is a list or a comma-separated string")
        if (f.min is not None or f.max is not None) and f.type not in ("int", "float"):
            errors.append(f"intake {f.name}: min/max apply to int and float fields only")
        if f.min is not None and f.max is not None and f.min > f.max:
            errors.append(f"intake {f.name}: min {f.min!r} is above max {f.max!r}")
    if not pb.branches:
        errors.append("a playbook needs at least one branch")
    seen_branches: set[str] = set()
    for b in pb.branches:
        if b.id in seen_branches:
            errors.append(f"branch {b.id}: duplicate id")
        seen_branches.add(b.id)
        errors += [f"branch {b.id}: {e}" for e in _check_conditions(b.when)]
        ids: list[str] = []
        for s in b.steps:
            if s.id in ids:
                errors.append(f"branch {b.id}, step {s.id}: duplicate id")
            if s.tool not in tools:
                errors.append(f"branch {b.id}, step {s.id}: unknown tool {s.tool!r}")
            if s.method and s.method not in METHODS:
                errors.append(f"branch {b.id}, step {s.id}: unknown method {s.method!r}")
            for d in s.depends_on:
                if d not in ids:
                    errors.append(f"branch {b.id}, step {s.id}: depends_on {d!r} is not an earlier step")
            for g in s.expects:
                if g.get("check") not in CHECKS:
                    errors.append(f"branch {b.id}, step {s.id}: unknown check {g.get('check')!r}")
            if isinstance(s.fallback, dict):
                if "step" in s.fallback:
                    fs = s.fallback["step"]
                    if not isinstance(fs, dict) or fs.get("tool") not in tools:
                        errors.append(f"branch {b.id}, step {s.id}: fallback step names no known tool")
                elif "branch" in s.fallback:
                    if s.fallback["branch"] not in {x.id for x in pb.branches}:
                        errors.append(f"branch {b.id}, step {s.id}: fallback branch {s.fallback['branch']!r} unknown")
                else:
                    errors.append(f"branch {b.id}, step {s.id}: a fallback is {{step: ...}}, {{branch: ...}} or stop")
            elif s.fallback not in (None, "stop"):
                errors.append(f"branch {b.id}, step {s.id}: a fallback is {{step: ...}}, {{branch: ...}} or stop")
            for ns, key in _placeholders(s.model_dump()):
                if ns in _DEFERRED:
                    ref = key.split(".", 1)[0]
                    if ref not in ids:
                        errors.append(f"branch {b.id}, step {s.id}: placeholder result.{key} names no earlier step")
                    elif ref not in s.depends_on:
                        errors.append(f"branch {b.id}, step {s.id}: reads result.{ref}, so depends_on must list {ref}")
                elif ns not in _NAMESPACES:
                    errors.append(f"branch {b.id}, step {s.id}: unknown placeholder namespace {ns!r}")
                elif ns == "intake" and key not in intake_names:
                    errors.append(f"branch {b.id}, step {s.id}: placeholder intake.{key} is not an intake field")
            ids.append(s.id)
    for i, d in enumerate(pb.declines):
        errors += [f"decline {i + 1}: {e}" for e in _check_conditions(d.when)]
        if not d.say.strip():
            errors.append(f"decline {i + 1}: says nothing")
    for i, c in enumerate(pb.caveats):
        if isinstance(c, Caveat):
            errors += [f"caveat {i + 1}: {e}" for e in _check_conditions(c.when)]
    return errors


def _check_conditions(conds: list[Condition]) -> list[str]:
    return [f"condition on {c.path!r}: unknown operator {c.op!r}" for c in conds if c.op not in OPERATORS]


def _placeholders(obj: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found += [(m.group(1), m.group(2)) for m in _PLACEHOLDER.finditer(item)]
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return found


# ── the evaluation context ──────────────────────────────────────────────────


def fill_intake(pb: Playbook, intake: dict[str, Any] | None) -> dict[str, Any]:
    """The intake with defaults applied and values coerced to the field types."""
    out: dict[str, Any] = dict(intake or {})
    for f in pb.intake:
        raw = out.get(f.name, f.default)
        if raw is None:
            if f.required:
                raise Declined(f"intake field {f.name!r} ({f.label or f.name}) is required", kind="intake",
                               playbook=pb.id)
            out[f.name] = None
            continue
        try:
            out[f.name] = _coerce(raw, f)
        except (TypeError, ValueError) as exc:
            raise Declined(f"intake field {f.name!r}: {exc}", kind="intake", playbook=pb.id) from None
    return out


def coerce_intake(pb: str | Playbook | dict[str, Any], values: dict[str, Any] | None) -> dict[str, Any]:
    """The intake a model (or any untrusted source) wrote, made safe: the lenient twin of :func:`fill_intake`.

    Every field of the playbook comes back: a value the field can take is
    coerced to its type, a field the values do not name gets its default, and
    so does a value the field cannot take (the wrong type, a choice outside
    the options, a number outside ``min``/``max``, a non-finite number).
    Fields the playbook does not have are dropped. Nothing raises here: the
    Explorer's on-device model fills the intake through this, and a small
    model's mistake should cost a default, not the plan.
    """
    pb = load(pb)
    out: dict[str, Any] = {}
    for f in pb.intake:
        raw = values.get(f.name) if isinstance(values, dict) else None
        if raw is None:
            out[f.name] = f.default
            continue
        try:
            out[f.name] = _coerce(raw, f)
        except (TypeError, ValueError, OverflowError):
            out[f.name] = f.default
    return out


def _coerce(raw: Any, f: IntakeField) -> Any:
    if f.type in ("int", "float"):
        if isinstance(raw, bool):
            raise ValueError(f"{raw!r} is not a number")
        number = float(raw)
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(f"{raw!r} is not a finite number")
        if f.min is not None and number < f.min:
            raise ValueError(f"{raw!r} is below the minimum {f.min:g}")
        if f.max is not None and number > f.max:
            raise ValueError(f"{raw!r} is above the maximum {f.max:g}")
        return int(number) if f.type == "int" else number
    if f.type == "bool":
        if isinstance(raw, str):
            if raw.strip().lower() in ("true", "yes", "y", "1", "on"):
                return True
            if raw.strip().lower() in ("false", "no", "n", "0", "off", ""):
                return False
            raise ValueError(f"{raw!r} is not a yes/no value")
        return bool(raw)
    if f.type == "choice":
        text = str(raw).strip()
        for opt in f.options:
            if text.lower() == str(opt).lower():
                return opt
        raise ValueError(f"{raw!r} is not one of {f.options}")
    if f.type == "list":
        items = raw if isinstance(raw, (list, tuple)) else [x for x in str(raw).replace(";", ",").split(",")]
        out: list[Any] = []
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            try:
                num = float(text)
                out.append(int(num) if num.is_integer() else num)
            except ValueError:
                out.append(text)
        if not out:
            raise ValueError("an empty list")
        return out
    return str(raw)


def _pick_station(recon: dict[str, Any], variable: str | None) -> dict[str, Any] | None:
    stations = [s for s in (recon.get("stations") or []) if isinstance(s, dict)]
    if variable:
        for s in stations:
            if variable in (s.get("variables") or []):
                return s
        return None
    return stations[0] if stations else None


def _derived(recon: dict[str, Any], intake: dict[str, Any]) -> dict[str, Any]:
    from aquascope.methods import METHODS

    context = recon.get("context") or {}
    years = context.get("years_by_variable") or {}
    catchment = recon.get("catchment") or {}
    discharge_years = float(years.get("discharge") or 0.0)
    factor = METHODS["at_site_flood_frequency"].max_return_period_factor or 3.0
    rp = intake.get("return_period")
    try:
        rp = float(rp) if rp is not None else None
    except (TypeError, ValueError):
        rp = None
    cap = factor * discharge_years if discharge_years else None
    dams = catchment.get("dams")
    if dams is None:
        reg = catchment.get("degree_of_regulation_pct")
        if reg is None:
            reg = ((catchment.get("attributes") or {}).get("degree_of_regulation_pct") or {})
        if isinstance(reg, dict):
            reg = reg.get("value")
        dams = 1 if isinstance(reg, (int, float)) and reg > 0 else 0
    elif isinstance(dams, bool):
        dams = int(dams)
    elif isinstance(dams, (list, tuple)):
        dams = len(dams)
    donors = context.get("donors")
    available = context.get("available")
    return {
        "has_temperature": ("temperature" in available) if isinstance(available, (list, tuple, set)) else True,
        "discharge_years": discharge_years,
        "groundwater_years": float(years.get("groundwater_level") or 0.0),
        "precipitation_years": float(years.get("precipitation") or 0.0),
        "ungauged": bool(context.get("ungauged", not years)),
        "donors": donors,
        "has_regional": bool(isinstance(donors, (int, float)) and donors >= 3),
        "dams": dams if isinstance(dams, (int, float)) else 0,
        "return_period": rp,
        "return_period_cap": cap,
        "return_period_beyond_cap": bool(rp is not None and cap is not None and rp > cap),
        "area_km2": catchment.get("upstream_area_km2") or catchment.get("area_km2") or context.get("area_km2"),
    }


def evaluation_context(pb: Playbook, recon: dict[str, Any], intake: dict[str, Any] | None = None,
                       *, station_variable: str | None = None) -> dict[str, Any]:
    """The dict conditions and placeholders read: recon plus intake, station, site and derived."""
    recon = dict(recon or {})
    intake = fill_intake(pb, intake)
    point = recon.get("point") or {}
    ctx = dict(recon)
    ctx["intake"] = intake
    ctx["site"] = {"lat": point.get("lat"), "lon": point.get("lon")}
    station = dict(_pick_station(recon, station_variable or pb.variable) or {})
    if station and not station.get("name"):
        station["name"] = station.get("station_id")  # an unnamed gauge reads as its id in the prose
    ctx["station"] = station
    ctx["derived"] = _derived(recon, intake)
    return ctx


def _resolve_value(value: Any, ctx: dict[str, Any]) -> Any:
    if isinstance(value, str) and _PLACEHOLDER.search(value):
        return _fill(value, ctx)
    return value


def _holds(cond: Condition, ctx: dict[str, Any]) -> bool:
    from aquascope.gates import resolve_path

    got = resolve_path(ctx, cond.path)
    want = _resolve_value(cond.value, ctx)
    op = cond.op
    if op == "exists":
        present = got is not None
        return present if (want is None or want is True) else not present
    if op == "==":
        return got == want
    if op == "!=":
        return got != want
    if op == "in":
        if want is None:
            return False
        if isinstance(want, (list, tuple, set)):
            return got in want
        return got is not None and str(got) in str(want)
    if got is None or want is None:
        return False
    try:
        a, b = float(got), float(want)
    except (TypeError, ValueError):
        return False
    return {">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b}.get(op, False)


def _all_hold(conds: list[Condition], ctx: dict[str, Any]) -> bool:
    return all(_holds(c, ctx) for c in conds)


def _fill(obj: Any, ctx: dict[str, Any]) -> Any:
    """Resolve placeholders; a string that is one placeholder keeps the value's type.

    A ``result.*`` placeholder is the runner's to resolve and is left as it is.
    """
    if isinstance(obj, str):
        whole = _PLACEHOLDER.fullmatch(obj.strip())
        if whole:
            if whole.group(1) in _DEFERRED:
                return obj
            return _lookup(whole.group(1), whole.group(2), ctx)

        def sub(m: re.Match[str]) -> str:
            if m.group(1) in _DEFERRED:
                return m.group(0)
            v = _lookup(m.group(1), m.group(2), ctx)
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            return "" if v is None else str(v)

        return _PLACEHOLDER.sub(sub, obj)
    if isinstance(obj, dict):
        return {k: _fill(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, ctx) for v in obj]
    return obj


def _lookup(ns: str, key: str, ctx: dict[str, Any]) -> Any:
    if ns not in _NAMESPACES:
        raise PlaybookError(f"unknown placeholder namespace {ns!r}")
    space = ctx.get(ns) or {}
    if key not in space:
        if ns == "station":
            raise PlaybookError(f"placeholder station.{key}: no station with the needed variable within reach")
        raise PlaybookError(f"placeholder {ns}.{key} has no value")
    return space[key]


# ── the tree ────────────────────────────────────────────────────────────────


def select_branch(pb: str | Playbook, recon: dict[str, Any], intake: dict[str, Any] | None = None) -> Branch | None:
    """The first branch whose conditions all hold over the recon (None when none does)."""
    pb = load(pb)
    for b in pb.branches:
        ctx = evaluation_context(pb, recon, intake, station_variable=b.station_variable)
        if _all_hold(b.when, ctx):
            return b
    return None


def declines_for(pb: str | Playbook, recon: dict[str, Any], intake: dict[str, Any] | None = None) -> list[str]:
    """The decline sentences whose conditions hold."""
    pb = load(pb)
    ctx = evaluation_context(pb, recon, intake)
    return [d.say for d in pb.declines if d.when and _all_hold(d.when, ctx)]


def caveats_for(pb: Playbook, ctx: dict[str, Any]) -> list[str]:
    out = []
    for c in pb.caveats:
        if isinstance(c, str):
            out.append(_fill(c, ctx))
        elif _all_hold(c.when, ctx):
            out.append(_fill(c.say, ctx))
    return out


def _method_status(method: str, recon: dict[str, Any], intake: dict[str, Any]) -> dict[str, Any]:
    """The registry's verdict on a method at this site, computed from the recon's context.

    What the reconnaissance could not find out is not held against the method:
    a donor count of ``None`` or an absent ``available`` set means unknown, and
    the run-time gate (``min_donors``, ``not_empty``) is where that gets
    settled. Record length, resolution and catchment area are always applied.
    """
    from aquascope.methods import METHODS, SiteContext, assess_method

    if method not in METHODS:
        return {"method": method, "status": "not_defensible", "reason": f"unknown method {method!r}"}
    pre = METHODS[method]
    c = recon.get("context")
    if not isinstance(c, dict):
        for row in recon.get("sufficiency") or []:
            if isinstance(row, dict) and row.get("method") == method:
                return row
        c = {}
    rp = intake.get("return_period")
    donors = c.get("donors")
    if donors is None and pre.min_donors:
        donors = pre.min_donors
    available = set(c["available"]) if c.get("available") is not None else set(pre.needs)
    catchment = recon.get("catchment") or {}
    ctx = SiteContext(
        years_by_variable={k: float(v) for k, v in (c.get("years_by_variable") or {}).items() if v is not None},
        resolution_by_variable=dict(c.get("resolution_by_variable") or {}),
        area_km2=c.get("area_km2") or catchment.get("upstream_area_km2") or catchment.get("area_km2"),
        return_period=float(rp) if isinstance(rp, (int, float)) else None,
        donors=donors,
        available=available,
    )
    return assess_method(pre, ctx)


def plan(
    pb: str | Playbook | dict[str, Any],
    recon: dict[str, Any],
    intake: dict[str, Any] | None = None,
    *,
    branch: str | None = None,
    problem_text: str | None = None,
) -> Study:
    """Fill a study from the tree alone: no model, and every placeholder resolved.

    Raises :class:`Declined` when a decline rule matches, when no branch
    applies, or when the registry calls a required step's method not
    defensible at this site (an optional step is dropped with a note instead).
    """
    pb = load(pb)
    recon = dict(recon or {})
    intake = fill_intake(pb, intake)
    base = evaluation_context(pb, recon, intake)
    for d in pb.declines:
        if d.when and _all_hold(d.when, base):
            raise Declined(_fill(d.say, base), kind="declined", playbook=pb.id)
    if branch:
        chosen = pb.branch(branch)
        if chosen is None:
            raise PlaybookError(f"playbook {pb.id} has no branch {branch!r}")
    else:
        chosen = select_branch(pb, recon, intake)
        if chosen is None:
            years = (recon.get("context") or {}).get("years_by_variable") or {}
            have = ", ".join(f"{k} {v:g} years" for k, v in years.items()) or "no usable record"
            raise Declined(
                f"No branch of the {pb.title.lower()} playbook applies to this site ({have}).",
                kind="no_branch", playbook=pb.id,
            )
    ctx = evaluation_context(pb, recon, intake, station_variable=chosen.station_variable)
    notes: list[str] = []
    steps: list[Step] = []
    dropped: set[str] = set()
    for t in chosen.steps:
        step_ctx = ctx
        if t.station_variable and t.station_variable != (chosen.station_variable or pb.variable):
            step_ctx = evaluation_context(pb, recon, intake, station_variable=t.station_variable)
        if t.method:
            verdict = _method_status(t.method, recon, intake)
            if verdict.get("status") == "not_defensible":
                reason = f"{t.tool} ({t.method}) is not defensible here: {verdict.get('reason')}"
                if t.optional:
                    notes.append(f"step {t.id} dropped: {reason}")
                    dropped.add(t.id)
                    continue
                raise Declined(reason, kind="refused", playbook=pb.id, branch=chosen.id)
        if any(d in dropped for d in t.depends_on):
            notes.append(f"step {t.id} dropped: it depends on a dropped step")
            dropped.add(t.id)
            continue
        steps.append(Step(
            tool=t.tool,
            arguments=_fill(dict(t.arguments), step_ctx),
            id=t.id,
            rationale=_fill(t.rationale, step_ctx) if t.rationale else None,
            method=t.method,
            expects=_fill([dict(g) for g in t.expects], step_ctx),
            fallback=_fill(t.fallback, step_ctx) if isinstance(t.fallback, dict) else t.fallback,
            depends_on=list(t.depends_on),
        ))
    site = ctx["site"]
    station = ctx.get("station") or {}
    where = f"{site['lat']}, {site['lon']}" if site.get("lat") is not None else "the site"
    question = problem_text or f"{pb.title} at {where}"
    plan_block: dict[str, Any] = {
        "playbook": pb.id,
        "branch": chosen.id,
        "rationale": _fill(chosen.rationale, ctx) if chosen.rationale else None,
        "caveats": caveats_for(pb, ctx),
        "citations": list(pb.citations),
    }
    if station:
        plan_block["station"] = {k: station.get(k) for k in ("source", "station_id", "name", "years", "distance_km")}
    if notes:
        plan_block["notes"] = notes
    if recon.get("notes"):
        plan_block["recon_notes"] = [str(n) for n in recon["notes"]]
    return Study(
        question=question,
        title=f"{pb.title}: {where}",
        steps=steps,
        author="playbook",
        version=2,
        problem={k: v for k, v in {"kind": pb.problem, "site": dict(site), "params": dict(intake),
                                   "text": problem_text}.items() if v is not None},
        plan=plan_block,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
