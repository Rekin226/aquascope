"""Steerable steps: the few parameters of a method a person may change after seeing its result, and the rerun.

A plan is written once, but a reader looks at a result and wants to push back
on it: fit LP3 rather than GEV, ask for the 200-year flood, call a drought at
SPI -1.5, use only the last 30 years. Each catalogue entry declares which of
its arguments are steerable, with a type and the values allowed
(:func:`steerable`, surfaced as ``Entry.steer`` in :mod:`aquascope.studio.catalogue`).
A change is validated against that declaration, applied to the step, and
the step reruns with every step that depends on it (by ``depends_on``,
``from_step`` or a ``{{ result.<id>... }}`` reference); every other result
is reused as it was. The change is recorded in the study's plan
(``plan.steering``), and the step's arguments carry the new value, so
``study.yaml`` and the notebook that re-runs it reproduce the steered study.

The plan also reads in plain words here: :func:`plain_plan` turns each gate
into one sentence (:func:`aquascope.gates.plain`) and lists each step's
controls with their current values, for the page to draw.

Everything is a plain function over a study and dicts, and runs in the
browser worker (numpy, scipy, pandas, pydantic, httpx only). No model is
involved: a steered rerun is the runner's, with the same gates.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from aquascope.study import Step, Study, StudyRun, _referenced_steps, load, loads, run_study

__all__ = ["Control", "apply_change", "controls_for", "declared", "dependants", "plain_plan", "rerun_step",
           "steerable", "validate_change"]

#: The return periods a flood step reports when its plan names none (mirrors aquascope.explore.RETURN_PERIODS,
#: which is not imported here to keep this module light).
_DEFAULT_PERIODS = (2, 5, 10, 25, 50, 100)


@dataclass(frozen=True)
class Control:
    """One steerable parameter of a tool.

    ``argument`` is the tool argument it sets; the special ``return_period``
    control (``argument`` None) sets the design T: it is added to the step's
    ``return_periods`` and written into every gate of the step and its
    dependants that carries a ``return_period``. ``optional`` means an empty
    value removes the argument, so the tool's own default applies.
    ``methods`` limits the control to steps applying one of those registry
    methods (a step with no method keeps it).
    """

    param: str
    label: str
    #: choice | number | integer | boolean
    type: str
    choices: tuple[Any, ...] = ()
    min: float | None = None
    max: float | None = None
    default: Any = None
    optional: bool = False
    argument: str | None = None
    methods: tuple[str, ...] = ()
    help: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"param": self.param, "label": self.label, "type": self.type}
        for key in ("min", "max", "default"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.choices:
            out["choices"] = list(self.choices)
        if self.optional:
            out["optional"] = True
        if self.help:
            out["help"] = self.help
        return out


def _return_period(methods: tuple[str, ...] = ()) -> Control:
    return Control("return_period", "Design return period (years)", "choice",
                   choices=(10, 20, 25, 50, 100, 200, 500, 1000), methods=methods,
                   help="the T the report quotes; the gates check it against the record length")


def _years(label: str = "Use only the last N years", *, default: int | None = None, lo: int = 5,
           hi: int = 200, optional: bool = True) -> Control:
    return Control("years", label, "integer", min=lo, max=hi, default=default, optional=optional, argument="years",
                   help="leave empty for the full record" if optional else "")


def _declarations() -> dict[str, tuple[Control, ...]]:
    """The steerable parameters per catalogue tool. The closed sets come from the constants the tools check
    against, so a control never offers a value the tool would refuse."""
    from aquascope import workbench

    flood = ("at_site_flood_frequency",)
    return {
        "flood_frequency": (
            _return_period(),
            _years(),
            Control("bootstrap_ci", "Bootstrap confidence band", "boolean", default=False, argument="bootstrap_ci"),
        ),
        "analyze_station": (_return_period(flood), _years()),
        "return_periods": (
            Control("distribution", "Distribution", "choice", choices=tuple(workbench.DISTRIBUTIONS), default="gev",
                    argument="distribution", help="GEV, Log-Pearson III or Gumbel fitted to the annual maxima"),
            Control("confidence_level", "Confidence level", "choice", choices=(0.8, 0.9, 0.95, 0.99), default=0.95,
                    argument="confidence_level"),
        ),
        "drought_indices": (
            Control("threshold", "Drought threshold (index value)", "number", min=-3.0, max=0.0, default=-1.0,
                    argument="threshold", help="a month at or below this SPI or SPEI counts as drought"),
            Control("pet", "Evapotranspiration for SPEI", "choice", choices=("thornthwaite", "fao56", "none"),
                    default="thornthwaite", argument="pet"),
            _years("Years of ERA5 record", default=40, lo=10, hi=80, optional=False),
        ),
        "low_flow_context": (_years(),),
        "get_timeseries": (_years(),),
        "supply_reliability": (
            Control("share", "Largest share of the flow taken", "number", min=0.01, max=1.0, default=0.1,
                    argument="share"),
            Control("reserve", "Flow kept in the river", "choice", choices=("q95", "none"), default="q95",
                    argument="reserve"),
        ),
        "similar_basins": (
            Control("k", "Donor gauges", "integer", min=3, max=50, default=10, argument="k"),
            Control("method", "Donor selection", "choice", choices=("similarity", "proximity", "combined"),
                    default="combined", argument="method"),
        ),
        "regionalize_signatures": (
            Control("k", "Donor gauges", "integer", min=3, max=50, default=10, argument="k"),
            Control("method", "Transfer method", "choice", choices=("similarity", "regression", "both"),
                    default="similarity", argument="method"),
        ),
        "baseflow": (
            Control("method", "Separation filter", "choice", choices=tuple(workbench.BASEFLOW_METHODS),
                    default="lyne_hollick", argument="method"),
        ),
        "wqi": (
            Control("use", "Water use", "choice", choices=tuple(workbench.WQI_USES), default="drinking",
                    argument="use"),
        ),
        "crop_water_demand": (
            Control("efficiency", "Irrigation efficiency", "number", min=0.3, max=1.0, default=0.7,
                    argument="efficiency"),
        ),
    }


_CACHE: dict[str, dict[str, tuple[Control, ...]]] = {}


def steerable() -> dict[str, tuple[Control, ...]]:
    """Every tool's controls by tool id (built once; the workbench constants are read on first use)."""
    if "decl" not in _CACHE:
        _CACHE["decl"] = _declarations()
    return _CACHE["decl"]


def declared(tool: str) -> list[dict[str, Any]]:
    """The declaration of a tool's steerable parameters, as dicts (empty when it has none)."""
    return [c.to_dict() for c in steerable().get(str(tool), ())]


def controls_for(step: Step) -> tuple[Control, ...]:
    """The controls that apply to one step (its tool's, limited by the step's method where a control says so)."""
    out = []
    for c in steerable().get(step.tool, ()):
        if c.methods and step.method and step.method not in c.methods:
            continue
        out.append(c)
    return tuple(out)


# ── current values ──────────────────────────────────────────────────────────


def _current(step: Step, control: Control, study: Study | None = None) -> Any:
    if control.param == "return_period" and control.argument is None:
        for g in step.expects or []:
            if isinstance(g, dict) and isinstance(g.get("return_period"), (int, float)):
                return g["return_period"]
        rp = ((study.problem or {}).get("params") or {}).get("return_period") if study is not None else None
        return rp if isinstance(rp, (int, float)) and not isinstance(rp, bool) else None
    return (step.arguments or {}).get(control.argument or control.param, control.default)


# ── validation ──────────────────────────────────────────────────────────────


def _is_template(value: Any) -> bool:
    return isinstance(value, str) and "{{" in value


def _coerce(control: Control, value: Any) -> Any:
    """The value in the control's type, or ValueError with the reason in plain words."""
    label = control.label.lower()
    if value is None or (isinstance(value, str) and not value.strip()):
        if control.optional:
            return None
        raise ValueError(f"{label} needs a value")
    if control.type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "on", "off"):
            return value.strip().lower() in ("true", "yes", "on")
        raise ValueError(f"{label} is yes or no, not {value!r}")
    if control.type == "choice":
        for choice in control.choices:
            if isinstance(choice, str) and isinstance(value, str) and value.strip().lower() == choice.lower():
                return choice
            if not isinstance(choice, str) and not isinstance(value, bool):
                try:
                    if float(value) == float(choice):
                        return choice
                except (TypeError, ValueError):
                    pass
        shown = ", ".join(str(c) for c in control.choices)
        raise ValueError(f"{label} must be one of {shown}, not {value!r}")
    if isinstance(value, bool):
        raise ValueError(f"{label} is a number, not {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} is a number, not {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    if control.type == "integer":
        if not number.is_integer():
            raise ValueError(f"{label} is a whole number, not {value!r}")
        number = int(number)
    if control.min is not None and number < control.min:
        raise ValueError(f"{label} must be at least {control.min:g}, not {number:g}")
    if control.max is not None and number > control.max:
        raise ValueError(f"{label} must be at most {control.max:g}, not {number:g}")
    return number


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a == b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def validate_change(step: Step, changes: dict[str, Any], study: Study | None = None) -> tuple[dict[str, Any],
                                                                                             list[str]]:
    """``(clean, errors)``: each change checked against the step's declared controls and coerced to its type.
    A parameter the step does not declare, a value outside the allowed set or range, an argument that is taken
    from another step's result, and a change that changes nothing are refused with the reason."""
    errors: list[str] = []
    clean: dict[str, Any] = {}
    if not isinstance(changes, dict) or not changes:
        return clean, ["no change was given"]
    controls = {c.param: c for c in controls_for(step)}
    if not controls:
        return clean, [f"step {step.id} ({step.tool}) has no parameters that can be adjusted"]
    for param, value in changes.items():
        control = controls.get(str(param))
        if control is None:
            errors.append(f"{param} cannot be adjusted on step {step.id}; it takes {', '.join(sorted(controls))}")
            continue
        if control.argument and _is_template((step.arguments or {}).get(control.argument)):
            errors.append(f"{control.label.lower()} on step {step.id} is taken from another step's result; "
                          "adjust that step instead")
            continue
        try:
            new = _coerce(control, value)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        old = _current(step, control, study)
        same = (control.argument not in (step.arguments or {})) if new is None else _same(new, old)
        if same:
            errors.append(f"{control.label.lower()} is already {old if old is not None else 'the default'}")
            continue
        clean[control.param] = new
    return clean, errors


# ── dependants ──────────────────────────────────────────────────────────────


def _ids(study: Study) -> None:
    """Every step gets the id the runner would give it (``s<i>``), so a change can name it."""
    for i, s in enumerate(study.steps, 1):
        if not s.id:
            s.id = f"s{i}"


def dependants(study: Study, step_id: str) -> list[str]:
    """``step_id`` and every step that depends on it, directly or through another: by ``depends_on``, by
    ``from_step`` or by a ``{{ result.<id>... }}`` reference in its arguments or gates. In plan order."""
    dirty = {str(step_id)}
    for s in study.steps:
        sid = str(s.id)
        if sid in dirty:
            continue
        reads = set(s.depends_on or []) | set(_referenced_steps(s))
        if reads & dirty:
            dirty.add(sid)
    return [str(s.id) for s in study.steps if str(s.id) in dirty]


# ── applying a change ───────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _t(x: Any) -> Any:
    return int(x) if isinstance(x, float) and x.is_integer() else x


def _words(value: Any) -> str:
    if value is None:
        return "the default"
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(_t(value))


def apply_change(study: Study, step_id: str, changes: dict[str, Any], *, at: str | None = None) -> dict[str, Any]:
    """A copy of ``study`` with the change applied to step ``step_id``, and what it takes to rerun it.

    Returns ``{"study", "change", "dirty", "text"}``: the new :class:`~aquascope.study.Study`, the record
    appended to ``plan.steering`` (``{"step", "tool", "changes": {param: {"from", "to"}}, "reran", "at"}``),
    the ids to rerun (the step and its dependants, in order) and one line in words. Raises ``ValueError``
    with every reason when the change is not accepted; the study passed in is never modified.
    """
    new = copy.deepcopy(study)
    _ids(new)
    step = new.step_by_id(str(step_id))
    if step is None:
        raise ValueError(f"there is no step {step_id!r} in this study")
    clean, errors = validate_change(step, changes, new)
    if errors:
        raise ValueError("; ".join(errors))
    dirty = dependants(new, str(step.id))
    controls = {c.param: c for c in controls_for(step)}
    record: dict[str, Any] = {"step": step.id, "tool": step.tool, "changes": {}, "reran": dirty, "at": at or _now()}
    for param, value in clean.items():
        control = controls[param]
        old = _current(step, control, new)
        record["changes"][param] = {"from": _t(old), "to": _t(value)}
        if control.param == "return_period" and control.argument is None:
            _set_return_period(new, step, float(value), dirty)
        elif value is None:
            step.arguments.pop(control.argument or param, None)
        else:
            step.arguments[control.argument or param] = _t(value)
    new.version = max(2, int(new.version or 1))
    new.plan = dict(new.plan or {})
    new.plan["steering"] = [*list(new.plan.get("steering") or []), record]
    bits = [f"{controls[p].label.lower()} {_words(v['from'])} to {_words(v['to'])}"
            for p, v in record["changes"].items()]
    text = f"Step {step.id}: {', '.join(bits)}"
    return {"study": new, "change": record, "dirty": dirty, "text": text}


def _set_return_period(study: Study, step: Step, t: float, dirty: list[str]) -> None:
    """The design T into the step's ``return_periods`` and into every T-carrying gate of the dirty steps; the
    study's problem says the new T too, so the report's headline quotes it."""
    value = _t(t)
    periods = [float(x) for x in (step.arguments.get("return_periods") or _DEFAULT_PERIODS)
               if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if t not in periods:
        periods.append(t)
    step.arguments["return_periods"] = [_t(p) for p in sorted(set(periods))]
    for s in study.steps:
        if str(s.id) not in dirty:
            continue
        for g in s.expects or []:
            if isinstance(g, dict) and "return_period" in g:
                g["return_period"] = value
    if study.problem:
        study.problem = {**study.problem, "params": {**dict(study.problem.get("params") or {}),
                                                     "return_period": value}}


# ── the rerun ───────────────────────────────────────────────────────────────


def _study_and_prior(state: Any) -> tuple[Study, StudyRun | None]:
    """A study and its last run from what a face holds: a StudyRun, a Study (or its dict, or YAML text, or a
    path), or ``{"study": ..., "results": [...]}`` / ``{"study": ..., "run": {"results": [...]}}``."""
    if isinstance(state, StudyRun):
        return state.study, state
    if isinstance(state, Study):
        return state, None
    if isinstance(state, dict) and "study" in state:
        study = _as_study(state["study"])
        results = state.get("results")
        if results is None and isinstance(state.get("run"), dict):
            results = state["run"].get("results")
        if isinstance(results, list) and results:
            rows = [dict(r) for r in results if isinstance(r, dict)]
            return study, StudyRun(study=copy.deepcopy(study), results=rows)
        return study, None
    return _as_study(state), None


def _as_study(value: Any) -> Study:
    if isinstance(value, Study):
        return value
    if isinstance(value, dict):
        return Study.from_dict(value)
    if isinstance(value, str) and ("\n" in value or ":" in value.split("\n", 1)[0]):
        return loads(value)
    return load(value)


def rerun_step(study_state: Any, step_id: str, changes: dict[str, Any], *, tools: dict[str, Any] | None = None,
               on_event: Any = None) -> dict[str, Any]:
    """Change a steerable parameter of one step and rerun that step and its dependants only.

    ``study_state`` is the study with its last run (see :func:`_study_and_prior` for the forms). The change is
    validated against the tool's declared controls; on success the step and every step that depends on it run
    again with :func:`aquascope.study.run_study`, and every other step's result is reused as it was, passed or
    failed. Returns ``{"ok", "errors", "change", "rerun", "reused", "study", "study_yaml", "summary", "gates",
    "results"}``; ``study_yaml`` carries the change in ``plan.steering`` and in the step's arguments, so
    ``aquascope run`` and the notebook reproduce it. A refused change returns ``ok`` false and the reasons,
    and nothing runs.
    """
    study, prior = _study_and_prior(study_state)
    try:
        out = apply_change(study, step_id, changes)
    except ValueError as exc:
        return {"ok": False, "errors": str(exc).split("; "), "rerun": [], "reused": []}
    dirty = set(out["dirty"])
    keep = [str(r.get("id")) for r in (prior.results if prior else []) if str(r.get("id")) not in dirty]
    new: Study = out["study"]
    run = run_study(new, on_event=on_event, prior=prior, tools=tools, reuse=keep)
    return {
        "ok": True, "errors": [], "change": out["change"], "text": out["text"],
        "rerun": out["dirty"], "reused": keep,
        "study": new.to_dict(), "study_yaml": new.to_yaml(),
        "summary": run.summary, "gates": run.gates, "results": run.results, "run_ok": run.ok,
    }


# ── the plan in plain words ─────────────────────────────────────────────────


def plain_plan(study: Study | dict[str, Any] | None) -> dict[str, Any]:
    """The plan as a person reads it: per step, its gates as sentences (the raw gates kept beside them) and its
    controls with their current values. ``{"steps": [{"id", "tool", "checks": [str], "gates": [dict],
    "controls": [dict]}], "steering": [...]}``; an empty dict for no study."""
    from aquascope.gates import plain

    if study is None:
        return {}
    st = study if isinstance(study, Study) else Study.from_dict(study)
    st = copy.deepcopy(st)
    _ids(st)
    steps = []
    for s in st.steps:
        controls = []
        for c in controls_for(s):
            row = c.to_dict()
            value = _current(s, c, st)
            row["value"] = _t(value) if not isinstance(value, bool) else value
            row["locked"] = bool(c.argument and _is_template((s.arguments or {}).get(c.argument)))
            controls.append(row)
        steps.append({"id": s.id, "tool": s.tool, "checks": [plain(g) for g in s.expects or []],
                      "gates": [dict(g) for g in s.expects or []], "controls": controls})
    return {"steps": steps, "steering": list((st.plan or {}).get("steering") or [])}
