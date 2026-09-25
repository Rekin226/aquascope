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
    "COMPANIONS",
    "companions",
    "Branch",
    "ChecklistItem",
    "Declined",
    "Playbook",
    "PlaybookError",
    "as_json",
    "checklist_open",
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


class ChecklistOption(BaseModel):
    """One answer a checklist question offers: the intake value, the words shown, and the words in the
    client's own text that pick it without asking."""

    value: Any
    label: str | None = None
    match: str | None = None


class ChecklistItem(BaseModel):
    """One thing the study must know before it plans, asked only when the text, the site and the earlier
    answers leave it open. ``field`` is the intake field it fills; ``why`` is the one line that says what the
    answer changes; ``when`` (conditions over ``intake.*``) says when the item applies at all."""

    field: str
    ask: str
    why: str | None = None
    options: list[ChecklistOption] = Field(default_factory=list)
    when: list[Condition] = Field(default_factory=list)

    def label_of(self, value: Any) -> str | None:
        for o in self.options:
            if o.value == value or str(o.value).lower() == str(value).lower():
                return o.label or str(o.value)
        return None

    def pick(self, text: Any) -> ChecklistOption | None:
        """The option a reply names (its label, its value, or its ``match`` words), else None."""
        low = str(text if text is not None else "").strip().lower()
        if not low:
            return None
        for o in self.options:
            if low in (str(o.value).lower(), str(o.label or "").lower()):
                return o
        for o in self.options:
            if o.match and re.search(o.match, str(text), re.I):
                return o
        return None


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
    #: What the study must know before it plans, in the order it is asked (aquascope.studio's Consultant).
    checklist: list[ChecklistItem] = Field(default_factory=list)
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
    for item in pb.checklist:
        field = next((f for f in pb.intake if f.name == item.field), None)
        if field is None:
            errors.append(f"checklist {item.field}: not an intake field")
            continue
        errors += [f"checklist {item.field}: {e}" for e in _check_conditions(item.when)]
        for o in item.options:
            try:
                if o.value is not None:  # None: the field's own "no value" (the whole record)
                    _coerce(o.value, field)
            except (TypeError, ValueError, OverflowError):
                errors.append(f"checklist {item.field}: option {o.value!r} is not a value the field takes")
            if o.match:
                try:
                    re.compile(o.match)
                except re.error as exc:
                    errors.append(f"checklist {item.field}: match {o.match!r} is not a pattern ({exc})")
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
    if isinstance(raw, str) and "{{" in raw and "result." in raw:
        # A quantity an earlier step of the plan computes (a companion playbook reads the primary plan's
        # derived values this way); the runner resolves it, the coercion cannot.
        return raw
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


def checklist_open(pb: str | Playbook | dict[str, Any], intake: dict[str, Any] | None,
                   stated: list[str] | set[str] | None = None) -> list[ChecklistItem]:
    """The checklist items still open, in order: those whose ``when`` holds over ``intake`` and whose field the
    client has not stated (``stated``: the fields read off the text or answered). A default in ``intake`` does
    not close an item; only the client's words do."""
    pb = load(pb)
    done = set(stated or ())
    ctx = {"intake": dict(intake or {})}
    return [item for item in pb.checklist if item.field not in done and _all_hold(item.when, ctx)]


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


#: A compound brief asks for more than one playbook's answer. Each row is a playbook that can be a companion
#: to another, the branch to take for it (None: whichever the site supports) and the pattern that names its
#: intent outright. These are stronger than the keyword rules that pick the primary playbook: a companion is
#: only added when the brief plainly asks for it, and only when the brief asks more than one thing.
COMPANIONS: list[tuple[str, str | None, str]] = [
    ("ungauged_flow", "regional",
     r"regional(?:i[sz](?:ed|ation))?\s+(?:estimate|transfer|figure|value)|similar (?:gauged )?(?:catchments|basins)|"
     r"donor (?:catchments|basins|gauges)|transferred from"),
    ("groundwater_decline", None,
     r"borehole|piezometer|groundwater levels?|water[- ]table|observation well|monitored well"),
    ("supply_reliability", None,
     r"\breliab|run-of-river|\bbe met\b|percent of (?:the )?(?:days|time)"),
    ("irrigation_feasibility", None, r"\bhectares?\b|crop water (?:demand|requirement|need)|planted in"),
    ("drought_status", None, r"\bin drought\b|rainfall deficit|\bspi\b|\bspei\b|drought (?:status|index|indices)"),
    ("flood_risk", None, r"\bflood\b|design (?:flow|flood)|\d+\s*-?\s*year\s+(?:flow|flood|design|return)"),
    ("water_quality", None, r"water quality|drinking-water guidelines|\bwqi\b"),
]

_ASKS = re.compile(r"\?|[,;]\s+(?:and\s+)?(?:what|how|is|are|does|do|would|where|which|can)\b|"
                   r"\b(?:and|also)\s+(?:what|how|is|are|does|do|would|where|which|can)\b", re.I)
_COMPOUND = re.compile(r"two (?:independent |separate |different )?(?:estimates|answers|figures|methods)|\bboth\b|"
                       r"\bcompare\b|\bagree\b|side by side|as well as|in addition", re.I)
MAX_COMPANIONS = 2


def companions(problem_text: str | None, primary: str | None) -> list[tuple[str, str | None]]:
    """The playbooks a compound brief asks for besides ``primary``, with the branch to take (None: the site's).

    A brief is compound when it asks more than one thing (two question clauses, or a comparison cue) and names
    another playbook's intent outright (``COMPANIONS``). At most ``MAX_COMPANIONS``, in table order.
    """
    text = problem_text or ""
    if not text.strip():
        return []
    if len(_ASKS.findall(text)) < 2 and not _COMPOUND.search(text):
        return []
    out: list[tuple[str, str | None]] = []
    for pid, branch, pattern in COMPANIONS:
        if pid == primary:
            continue
        if re.search(pattern, text, re.I):
            out.append((pid, branch))
    return out[:MAX_COMPANIONS]


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
    compose: bool = True,
) -> Study:
    """Fill a study from the tree alone: no model, and every placeholder resolved.

    Raises :class:`Declined` when a decline rule matches, when no branch
    applies, or when the registry calls a required step's method not
    defensible at this site (an optional step is dropped with a note instead).

    With ``compose`` (the default) and a ``problem_text`` that asks more than
    one thing, the branches of the companion playbooks the brief names
    (:func:`companions`) are planned too and their new steps appended, so a
    compound brief gets every part of its answer (#383).
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
    study = Study(
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
    if compose and problem_text:
        for cid, cbranch in companions(problem_text, pb.id):
            _add_companion(study, cid, cbranch, recon, intake)
    return study


_REF = re.compile(r"\{\{\s*result\.([A-Za-z0-9_]+)\.")


def _rename_refs(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _REF.sub(lambda m: m.group(0).replace(m.group(1), mapping.get(m.group(1), m.group(1))), value)
    if isinstance(value, dict):
        return {k: _rename_refs(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_rename_refs(v, mapping) for v in value]
    return value


def _add_companion(study: Study, cid: str, branch: str | None, recon: dict[str, Any], intake: dict[str, Any]) -> None:
    """Append what the companion playbook's branch adds to ``study``: the steps whose tool and method the plan
    does not have yet (optional steps and framing steps the plan already has are left out), re-numbered after
    the plan's own, references renamed, and the plan block saying what was added and why. A companion that
    declines or has no branch at this site adds a note instead."""
    plan_block = study.plan if isinstance(study.plan, dict) else {}
    # What the primary plan derives counts as given for the companion: a demand the crop step computes is a
    # demand the supply playbook may screen against, so its "state the demand" rule does not fire.
    derived = dict(intake)
    for s in study.steps:
        for k, v in (s.arguments or {}).items():
            if isinstance(v, str) and _REF.search(v) and k not in derived:
                derived[k] = v
    try:
        other = plan(cid, recon, derived, branch=branch, compose=False)
    except (Declined, PlaybookError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        plan_block.setdefault("notes", []).append(f"the brief also asks for {cid.replace('_', ' ')}: {reason}")
        return
    have_tools = {s.tool for s in study.steps}
    have_pairs = {(s.tool, s.method) for s in study.steps}
    have_methods = {s.method for s in study.steps if s.method}
    other_ids = {s.id for s in other.steps if s.id}
    template = load(cid).branch(str((other.plan or {}).get("branch") or ""))
    optional_ids = {t.id for t in (template.steps if template else []) if t.optional}
    # Which of the companion's steps feed which: a helper step with no method of its own (a series fetch) is
    # only worth adding when a step that is added needs it.
    needs: dict[str, set[str]] = {}
    for t in other.steps:
        wants = set(t.depends_on)
        src = (t.arguments or {}).get("from_step")
        if src:
            wants.add(str(src))
        for v in (t.arguments or {}).values():
            if isinstance(v, str):
                wants.update(m.group(1) for m in _REF.finditer(v))
        for w in wants:
            needs.setdefault(w, set()).add(t.id or "")
    chosen: list[Step] = []
    for t in other.steps:
        if t.tool == "describe_catchment" and "describe_catchment" in have_tools:
            continue
        if (t.tool, t.method) in have_pairs or (t.method and t.method in have_methods):
            continue
        if t.id in optional_ids:
            continue
        chosen.append(t)
    # A helper step (one another companion step reads) is only worth adding while a reader is added too.
    changed = True
    while changed:
        changed = False
        ids = {t.id for t in chosen}
        for t in list(chosen):
            readers = needs.get(t.id or "")
            if readers and not t.method and not (readers & ids):
                chosen.remove(t)
                changed = True
    mapping: dict[str, str] = {}
    added: list[Step] = []
    n = len(study.steps)
    for t in chosen:
        n += 1
        new_id = f"s{n}"
        mapping[t.id or new_id] = new_id
        added.append(Step(
            tool=t.tool, arguments=dict(t.arguments), id=new_id, rationale=t.rationale, method=t.method,
            expects=[dict(g) for g in t.expects], fallback=t.fallback,
            depends_on=[d for d in t.depends_on],
        ))
    if not added:
        return
    kept = {s.id for s in added} | mapping.keys()
    for s in added:
        s.arguments = _rename_refs(s.arguments, mapping)
        s.expects = _rename_refs(s.expects, mapping)
        s.fallback = _rename_refs(s.fallback, mapping) if isinstance(s.fallback, dict) else s.fallback
        s.depends_on = [mapping.get(d, d) for d in s.depends_on if d in kept or d in mapping or d not in other_ids]
        s.depends_on = [d for d in s.depends_on if d in {a.id for a in added} | {p.id for p in study.steps}]
    study.steps.extend(added)
    other_plan = other.plan or {}
    tools = ", ".join(s.tool for s in added)
    plan_block.setdefault("companions", []).append({
        "playbook": cid, "branch": other_plan.get("branch"), "steps": [s.id for s in added],
        "rationale": other_plan.get("rationale"),
    })
    plan_block["compound"] = True
    lead = (plan_block.get("rationale") or "").rstrip()
    plan_block["rationale"] = (lead + " " if lead else "") + (
        f"The brief also asks for what the {cid.replace('_', ' ')} playbook establishes, so its "
        f"{other_plan.get('branch') or 'own'} branch adds {tools}; the answers are set side by side in the report."
    )
    for key in ("caveats", "citations"):
        merged = list(dict.fromkeys([*(plan_block.get(key) or []), *(other_plan.get(key) or [])]))
        if merged:
            plan_block[key] = merged
    for note in other_plan.get("notes") or []:
        plan_block.setdefault("notes", []).append(f"{cid}: {note}")
    if study.problem is not None:
        params = dict(study.problem.get("params") or {})
        for k, v in ((other.problem or {}).get("params") or {}).items():
            if v is None or (isinstance(v, str) and _REF.search(v)):
                continue  # unset, or a quantity the primary plan derives rather than a parameter
            params.setdefault(k, v)
        study.problem["params"] = params
