"""HydroGym Phase 2: plan quality, scored against expert reference plans (#367, epic #363).

Phase 1 (:mod:`aquascope.gym.tasks`, :mod:`aquascope.gym.bench`) asks whether
an agent lands on the playbook branch the tree selects. Phase 2 asks a harder
question: given a brief at a real site, is the *plan* an agent writes the one
a hydrologist would write? The key is a set of reference plans
(``aquascope/gym/plans/*.yaml``), each an ordered list of steps with the tool,
the registry method, the gates that must be present, the steps that are
optional, the tools and methods that are not defensible at that site, or a
decline when the data cannot carry the question. Every case carries the
reconnaissance of its site, saved once (``plans/recon/*.json``), so a plan can
be produced and scored with no network: the tree needs none, the Methodologist
needs the model call only, and a plan produced elsewhere (a device model, a
notebook) is scored from its JSON.

The score of a candidate plan against a reference (:func:`score_plan`) is a
weighted sum of five parts, the weights in :data:`WEIGHTS`:

* ``coverage_tools``: the fraction of the reference's required tools the plan
  uses (a required step may name ``alternatives``, other tools that do the
  same job; any one of them covers the step);
* ``coverage_methods``: the fraction of its required registry methods the
  plan names (an alternative's method covers the step's);
* ``coverage_gates``: the fraction of its required ``(tool, check, path)``
  gates the plan carries on a step with that tool (on an alternative tool the
  check alone counts);
* ``parsimony``, ``1 - extraneous``: ``extraneous`` is the fraction of the
  plan's steps whose tool is neither in the reference (required or optional)
  nor a framing tool (:data:`FRAMING_TOOLS`);
* ``clean``: 1 when no step uses a forbidden tool or method, else 0.

A part the reference cannot judge (a reference with no method, or no gate)
is left out and the other weights are renormalised. A plan with no steps
scores 0 on every part; a plan that declines a solvable case scores 0. On a
case whose reference declines, the score is 1 when the candidate declines
and 0 otherwise, and the coverage parts are not computed. The
``validator_errors_first_try`` field counts the errors the Studio's validator
raised on the model's first plan (0 for the tree, and for a model plan that
passed at once); it is reported, not scored.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aquascope.gym.bench import PRICES_NOTE, _with_timeout, estimate_cost

logger = logging.getLogger(__name__)

__all__ = [
    "AGENTS",
    "FRAMING_TOOLS",
    "PLANS_DIR",
    "RECON_DIR",
    "WEIGHTS",
    "PlanResult",
    "Reference",
    "ReferenceStep",
    "candidate_from",
    "list_references",
    "load_plan_results",
    "load_reference",
    "load_references",
    "methodologist_candidate",
    "plan_leaderboard",
    "recon_for",
    "rescore_plans",
    "run_plan_bench",
    "score_plan",
    "summarize_plans",
    "tree_candidate",
    "validate_reference",
]

PLANS_DIR = Path(__file__).parent / "plans"
RECON_DIR = PLANS_DIR / "recon"
AGENTS = ("tree", "methodologist", "file")
DEFAULT_TIMEOUT = 300.0

#: The parts of the score and their weights; they sum to one. See the module docstring.
WEIGHTS: dict[str, float] = {"tools": 0.30, "methods": 0.25, "gates": 0.20, "clean": 0.15, "parsimony": 0.10}
#: Steps that frame a study without being an analysis; never extraneous, never required unless a reference says so.
FRAMING_TOOLS = frozenset({"describe_catchment", "find_stations", "assess_site"})
#: The reference-file keys a case may carry (anything else is an error the validator reports).
_CASE_KEYS = frozenset({"id", "playbook", "title", "brief", "intake", "site", "expected_branch", "steps", "forbidden",
                        "decline", "decline_kind", "rationale", "tags"})
_STEP_KEYS = frozenset({"tool", "method", "gates", "optional", "note", "alternatives"})


# ── the reference plans ─────────────────────────────────────────────────────


@dataclass
class ReferenceStep:
    """One step of an expert plan: the tool, the registry method it applies, the gates it must carry."""

    tool: str
    method: str | None = None
    #: ``[{check, path}]``; the path is the payload path the catalogue lists for the check on this tool.
    gates: list[dict[str, Any]] = field(default_factory=list)
    optional: bool = False
    note: str | None = None
    #: ``[{tool, method}]``: other steps that do this step's job (the SGI from the propagation tool rather than
    #: from a series and the table tool); any one of them covers the step.
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tools(self) -> frozenset[str]:
        return frozenset({self.tool} | {str(a["tool"]) for a in self.alternatives if a.get("tool")})

    @property
    def methods(self) -> frozenset[str]:
        found = {self.method} | {a.get("method") for a in self.alternatives}
        return frozenset(m for m in found if m)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], False)}


@dataclass
class Reference:
    """One case: a brief at a real site with the reconnaissance saved, and the expert plan (or a decline)."""

    id: str
    playbook: str
    brief: str
    site: dict[str, Any]
    intake: dict[str, Any] = field(default_factory=dict)
    title: str | None = None
    expected_branch: str | None = None
    steps: list[ReferenceStep] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    forbidden_methods: list[str] = field(default_factory=list)
    decline: bool = False
    #: ``declined`` (a playbook rule), ``no_branch``, ``refused`` (the registry), ``intake``.
    decline_kind: str | None = None
    rationale: str | None = None
    tags: list[str] = field(default_factory=list)
    path: str | None = None

    @property
    def required_steps(self) -> list[ReferenceStep]:
        return [s for s in self.steps if not s.optional]

    @property
    def lat(self) -> float:
        return float(self.site["lat"])

    @property
    def lon(self) -> float:
        return float(self.site["lon"])

    @property
    def recon_name(self) -> str:
        return str(self.site.get("recon") or self.id)

    def to_dict(self, *, with_recon: bool = False) -> dict[str, Any]:
        d = {
            "id": self.id, "playbook": self.playbook, "title": self.title, "brief": self.brief, "site": dict(self.site),
            "intake": dict(self.intake), "expected_branch": self.expected_branch,
            "steps": [s.to_dict() for s in self.steps],
            "forbidden": {"tools": list(self.forbidden_tools), "methods": list(self.forbidden_methods)},
            "decline": self.decline, "decline_kind": self.decline_kind, "rationale": self.rationale,
            "tags": list(self.tags),
        }
        if with_recon:
            d["recon"] = recon_for(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, path: str | None = None) -> Reference:
        steps = []
        for raw in d.get("steps") or []:
            if not isinstance(raw, dict):
                continue
            gates = []
            for g in raw.get("gates") or []:
                if isinstance(g, dict) and g.get("check"):
                    gates.append({"check": str(g["check"]), "path": _gate_path(g)})
            alternatives = [{"tool": str(a.get("tool") or ""), "method": a.get("method") or None}
                            for a in (raw.get("alternatives") or []) if isinstance(a, dict) and a.get("tool")]
            steps.append(ReferenceStep(tool=str(raw.get("tool") or ""), method=raw.get("method") or None, gates=gates,
                                       optional=bool(raw.get("optional")), note=raw.get("note"),
                                       alternatives=alternatives))
        forbidden = d.get("forbidden") or {}
        return cls(
            id=str(d.get("id") or (Path(path).stem if path else "")), playbook=str(d.get("playbook") or ""),
            brief=" ".join(str(d.get("brief") or "").split()), site=dict(d.get("site") or {}),
            intake=dict(d.get("intake") or {}), title=d.get("title"), expected_branch=d.get("expected_branch"),
            steps=steps, forbidden_tools=[str(t) for t in (forbidden.get("tools") or [])],
            forbidden_methods=[str(m) for m in (forbidden.get("methods") or [])],
            decline=bool(d.get("decline")), decline_kind=d.get("decline_kind"),
            rationale=" ".join(str(d.get("rationale") or "").split()) or None,
            tags=[str(t) for t in (d.get("tags") or [])], path=path,
        )


def _gate_path(g: dict[str, Any]) -> str | None:
    """A gate's path as one string: ``path``, or ``paths`` joined as the catalogue writes them."""
    if g.get("path"):
        return str(g["path"])
    paths = g.get("paths")
    if isinstance(paths, (list, tuple)) and paths:
        return ", ".join(str(p) for p in paths)
    if isinstance(paths, str):
        return paths
    return None


def _case_files(plans_dir: str | Path | None = None) -> list[Path]:
    root = Path(plans_dir) if plans_dir else PLANS_DIR
    return sorted(p for p in root.glob("*.yaml") if not p.name.startswith("_"))


def load_reference(case: str | Path, plans_dir: str | Path | None = None) -> Reference:
    """A reference by id (``flood_at_site_potomac``) or by file path."""
    from aquascope.study import _parse_yaml

    path = Path(case)
    if not path.suffix:
        root = Path(plans_dir) if plans_dir else PLANS_DIR
        path = root / f"{path.name}.yaml"
    if not path.exists():
        known = ", ".join(p.stem for p in _case_files(plans_dir))
        raise FileNotFoundError(f"no reference plan {str(case)!r}; known: {known}")
    data = _parse_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a reference plan is a YAML mapping")
    return Reference.from_dict(data, path=str(path))


def load_references(plans_dir: str | Path | None = None) -> list[Reference]:
    """Every reference plan in the directory (the package's by default), in file order."""
    return [load_reference(p, plans_dir) for p in _case_files(plans_dir)]


def list_references(plans_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """One row per case: id, playbook, site, whether it declines, the step count, the tags."""
    out = []
    for ref in load_references(plans_dir):
        out.append({
            "id": ref.id, "playbook": ref.playbook, "title": ref.title,
            "site": ref.site.get("name") or f"{ref.lat:.4f}, {ref.lon:.4f}", "decline": ref.decline,
            "steps": len(ref.required_steps), "optional": len(ref.steps) - len(ref.required_steps),
            "expected_branch": ref.expected_branch, "tags": list(ref.tags),
        })
    return out


_RECON_CACHE: dict[str, Any] = {}


def _load_recon_file(name: str, plans_dir: str | Path | None = None) -> dict[str, Any]:
    root = (Path(plans_dir) / "recon") if plans_dir else RECON_DIR
    path = root / f"{name}.json"
    key = str(path)
    if key not in _RECON_CACHE:
        with path.open(encoding="utf-8") as fh:
            _RECON_CACHE[key] = json.load(fh)
    return _RECON_CACHE[key]


def recon_for(ref: Reference, plans_dir: str | Path | None = None) -> dict[str, Any]:
    """The saved reconnaissance of the case's site, as the Studio's Scout would hold it for this playbook.

    The file holds ``assess_site(lat, lon)`` with the full sufficiency table;
    the Scout asks for the playbook's problem kind, which narrows the table
    to that problem's methods, so the same narrowing is applied here. A
    deep copy, so a caller may edit it.
    """
    from aquascope import playbooks as pbk
    from aquascope.methods import METHODS

    raw = _load_recon_file(ref.recon_name, plans_dir)
    recon = json.loads(json.dumps(raw.get("recon") if isinstance(raw.get("recon"), dict) else raw))
    try:
        problem = pbk.load(ref.playbook).problem
    except pbk.PlaybookError:
        problem = None
    if problem:
        recon["sufficiency"] = [r for r in recon.get("sufficiency") or []
                                if r.get("method") in METHODS and problem in METHODS[r["method"]].problems]
    return recon


def _playbook_pairs() -> set[tuple[str, str]]:
    """The (tool, method) pairs the playbooks themselves use; a reference may use them even where the catalogue
    lists the method under another tool (the supply_reliability tool transferring signatures at an ungauged
    point)."""
    from aquascope import playbooks as pbk

    if "pairs" not in _RECON_CACHE:
        pairs: set[tuple[str, str]] = set()
        for row in pbk.list_playbooks():
            if "error" in row:
                continue
            for b in pbk.load(row["id"]).branches:
                pairs |= {(st.tool, st.method) for st in b.steps if st.method}
        _RECON_CACHE["pairs"] = pairs
    return _RECON_CACHE["pairs"]


def validate_reference(ref: Reference, plans_dir: str | Path | None = None) -> list[str]:
    """Every problem with a reference plan, as sentences (empty when it is sound).

    Checks: the keys are known; the playbook loads and the intake fills; the
    reconnaissance file exists; every tool is in the catalogue and every
    method in the registry and one the tool applies; every gate is a known
    check with the path the catalogue lists for it on that tool (a gate the
    catalogue does not list for the tool is accepted with its own path);
    no required step uses a forbidden tool or method, or a method the saved
    reconnaissance calls not defensible at the site; a decline names its
    kind; a solvable case has at least one required analysis step.
    """
    from aquascope import playbooks as pbk
    from aquascope.gates import CHECKS
    from aquascope.methods import METHODS
    from aquascope.studio import catalogue

    errors: list[str] = []
    if not ref.id:
        errors.append("no id")
    try:
        pb = pbk.load(ref.playbook)
        pbk.fill_intake(pb, ref.intake)
    except pbk.PlaybookError as exc:
        errors.append(f"playbook: {exc}")
    except pbk.Declined as exc:
        errors.append(f"intake: {exc.reason}")
    if ref.site.get("lat") is None or ref.site.get("lon") is None:
        errors.append("site needs lat and lon")
    try:
        recon = recon_for(ref, plans_dir)
    except (OSError, ValueError) as exc:
        errors.append(f"recon {ref.recon_name!r}: {type(exc).__name__}: {exc}")
        recon = {}
    if not ref.brief:
        errors.append("no brief")
    if ref.decline:
        if ref.decline_kind not in ("declined", "no_branch", "refused", "intake"):
            errors.append(f"decline_kind {ref.decline_kind!r} is not one of declined, no_branch, refused, intake")
        if ref.steps:
            errors.append("a declining case lists no steps")
        return errors
    point = recon.get("point") or {}
    if recon and ref.site.get("lat") is not None and point.get("lat") is not None and (
            abs(float(point["lat"]) - ref.lat) > 0.01 or abs(float(point["lon"]) - ref.lon) > 0.01):
        errors.append(f"site ({ref.lat}, {ref.lon}) is not the reconnaissance's point ({point['lat']}, {point['lon']})")
    status = {r.get("method"): r.get("status") for r in recon.get("sufficiency") or [] if isinstance(r, dict)}
    pairs = _playbook_pairs()
    for i, s in enumerate(ref.steps, 1):
        entry = catalogue.get(s.tool)
        where = f"step {i} ({s.tool})"
        if entry is None:
            errors.append(f"{where}: unknown tool")
            continue
        if s.method:
            if s.method not in METHODS:
                errors.append(f"{where}: unknown method {s.method!r}")
            elif entry.methods and s.method not in entry.methods and (s.tool, s.method) not in pairs:
                errors.append(f"{where}: {s.tool} does not apply {s.method!r} (it applies {entry.methods})")
            elif not s.optional and status.get(s.method) in ("not_defensible", "not defensible"):
                errors.append(f"{where}: the registry calls {s.method!r} not defensible at this site")
        for g in s.gates:
            if g["check"] not in CHECKS:
                errors.append(f"{where}: unknown check {g['check']!r}")
                continue
            listed = [_gate_path(x) for x in entry.gates if x.get("check") == g["check"]]
            if listed and g.get("path") and g["path"] not in listed:
                errors.append(f"{where}: gate {g['check']} reads {g['path']!r}; the catalogue lists {listed}")
        if not s.optional and (s.tool in ref.forbidden_tools or (s.method and s.method in ref.forbidden_methods)):
            errors.append(f"{where}: a required step uses a forbidden tool or method")
        for a in s.alternatives:
            alt = catalogue.get(str(a.get("tool") or ""))
            if alt is None:
                errors.append(f"{where}: alternative tool {a.get('tool')!r} is not in the catalogue")
                continue
            m = a.get("method")
            if m and m not in METHODS:
                errors.append(f"{where}: alternative method {m!r} is not in the registry")
            elif m and alt.methods and m not in alt.methods and (alt.id, m) not in pairs:
                errors.append(f"{where}: {alt.id} does not apply {m!r} (it applies {alt.methods})")
            elif m and not s.optional and status.get(m) in ("not_defensible", "not defensible"):
                errors.append(f"{where}: the registry calls the alternative {m!r} not defensible at this site")
            if alt.id in ref.forbidden_tools or (m and m in ref.forbidden_methods):
                errors.append(f"{where}: an alternative uses a forbidden tool or method")
    for t in ref.forbidden_tools:
        if catalogue.get(t) is None:
            errors.append(f"forbidden tool {t!r} is not in the catalogue")
    for m in ref.forbidden_methods:
        if m not in METHODS:
            errors.append(f"forbidden method {m!r} is not in the registry")
    if not any(s.tool not in FRAMING_TOOLS for s in ref.required_steps):
        errors.append("a solvable case needs at least one required analysis step")
    return errors


# ── candidates ──────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """A plan to score, in one shape: its steps (tool, method, gates) or a decline."""

    steps: list[dict[str, Any]] = field(default_factory=list)
    declined: bool = False
    reason: str | None = None
    kind: str | None = None
    branch: str | None = None
    author: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_from(obj: Any) -> Candidate:
    """A candidate from a study (object or ``to_dict``), a workspace dict, or ``{"declined": true, ...}``.

    A workspace whose status is ``declined`` (or that has no study) is a
    decline; a study's steps become ``{id, tool, method, gates, fallback}``
    rows, a gate as ``{check, path}``.
    """
    if hasattr(obj, "to_dict") and not isinstance(obj, dict):
        obj = obj.to_dict()
    if not isinstance(obj, dict):
        return Candidate()
    if obj.get("declined") is True or obj.get("status") == "declined":
        return Candidate(declined=True, reason=obj.get("reason") or obj.get("declined_reason"),
                         kind=obj.get("kind") or obj.get("decline_kind"))
    if "study" in obj and "steps" not in obj:
        study = obj.get("study")
        if not isinstance(study, dict):
            return Candidate(declined=bool(obj.get("declined_reason")), reason=obj.get("declined_reason"))
        obj = study
    plan = obj.get("plan") if isinstance(obj.get("plan"), dict) else {}
    steps: list[dict[str, Any]] = []
    for i, raw in enumerate(obj.get("steps") or [], 1):
        if not isinstance(raw, dict):
            continue
        fb = raw.get("fallback")
        fb_step = fb.get("step") if isinstance(fb, dict) and isinstance(fb.get("step"), dict) else None
        steps.append({
            "id": str(raw.get("id") or f"s{i}"), "tool": str(raw.get("tool") or ""),
            "method": raw.get("method") or None,
            "gates": [{"check": g.get("check"), "path": _gate_path(g)} for g in (raw.get("expects") or [])
                      if isinstance(g, dict) and g.get("check")],
            "fallback": {"tool": fb_step.get("tool"), "method": fb_step.get("method")} if fb_step else None,
        })
    return Candidate(steps=steps, branch=plan.get("branch"), author=plan.get("author") or obj.get("author"))


def tree_candidate(ref: Reference, plans_dir: str | Path | None = None) -> tuple[Candidate, dict[str, Any]]:
    """The playbook tree on the saved reconnaissance: the baseline every reference is measured against."""
    from aquascope import playbooks as pbk

    recon = recon_for(ref, plans_dir)
    try:
        study = pbk.plan(ref.playbook, recon, ref.intake, problem_text=ref.brief)
    except pbk.Declined as exc:
        return Candidate(declined=True, reason=exc.reason, kind=exc.kind), {"kind": exc.kind}
    cand = candidate_from(study)
    return cand, {"branch": cand.branch, "notes": list((study.plan or {}).get("notes") or [])}


def _inventory_from_recon(ws: Any, recon: dict[str, Any]) -> None:
    """The Scout's inventory from a saved reconnaissance, with no call to ``assess_site`` (mirrors ``scout.scout``)."""
    from aquascope.studio.roles.scout import ERA5_START, _station_datasets
    from aquascope.studio.workspace import Dataset, Inventory

    site = dict(ws.site or {})
    ctx = recon.get("context") or {}
    inv = Inventory(site={"lat": float(site["lat"]), "lon": float(site["lon"])}, recon=recon,
                    catchment=recon.get("catchment") if isinstance(recon.get("catchment"), dict) else None,
                    donors=ctx.get("donors") if isinstance(ctx.get("donors"), int) else None,
                    notes=[str(n) for n in (recon.get("notes") or [])])
    inv.datasets += _station_datasets(recon)
    catchment = inv.catchment or {}
    if catchment and not catchment.get("error"):
        area = catchment.get("upstream_area_km2") or catchment.get("area_km2")
        inv.datasets.append(Dataset(
            id="catchment", kind="catchment", source=str(catchment.get("source") or "BasinATLAS"),
            name=str(catchment.get("sub_basin") or "the catchment of the point"),
            note=f"upstream area {area:,.0f} km2" if isinstance(area, (int, float)) else None))
    if inv.donors is not None:
        inv.datasets.append(Dataset(id="donors", kind="donors", source="similar_basins", n=int(inv.donors),
                                    name=f"{inv.donors} donor gauges by catchment similarity"))
    today = datetime.now(timezone.utc).date()
    inv.datasets.append(Dataset(
        id="era5", kind="reanalysis", variable="climate", source="ERA5 via Open-Meteo", name="ERA5 cell",
        lat=inv.site["lat"], lon=inv.site["lon"], start=ERA5_START, end=today.isoformat(),
        years=round((today - datetime(1940, 1, 1).date()).days / 365.25, 1), resolution="daily",
        note="precipitation, temperature and FAO-56 ET0 for a 9 km cell, any point on land; GloFAS modelled "
             "discharge for the same point, indicative",
    ))
    ws.inventory = inv
    ws.event("scout", "inventory", f"{len(inv.datasets)} dataset(s) from the saved reconnaissance")


def workspace_for(ref: Reference, plans_dir: str | Path | None = None) -> Any:
    """A Studio workspace at the case's site with the brief taken as the case states it (problem text, playbook,
    intake) and the inventory built from the saved reconnaissance: what the Methodologist plans on."""
    from aquascope import playbooks as pbk
    from aquascope.studio.workspace import Workspace

    ws = Workspace()
    ws.site = {"lat": ref.lat, "lon": ref.lon}
    pb = pbk.load(ref.playbook)
    b = ws.brief
    b.problem = ref.brief
    b.playbook = pb.id
    b.kind = pb.problem
    b.intake = pbk.fill_intake(pb, ref.intake)
    b.decision = ref.title
    b.ready = True
    b.source = "reference"
    _inventory_from_recon(ws, recon_for(ref, plans_dir))
    return ws


_ERRORS_IN_EVENT = re.compile(r"; ")


def methodologist_candidate(
    ref: Reference,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client: Any | None = None,
    plans_dir: str | Path | None = None,
) -> tuple[Candidate, dict[str, Any]]:
    """The Studio Methodologist on a model, planning on the saved reconnaissance: one model call (a second for a
    repair when the validator objects), no network otherwise. Keyless (no model named) it is the tree.

    The detail carries the usage, the validator's errors on the first plan,
    whether the plan that stands is the model's or the tree's fallback, and
    the Methodologist's events.
    """
    from aquascope.studio.model import Model
    from aquascope.studio.roles import methodologist

    ws = workspace_for(ref, plans_dir)
    llm = Model.resolve(ws, provider=provider, model=model, api_key=api_key, base_url=base_url, client=client)
    study = methodologist.plan(ws, llm)
    events = [e for e in ws.events if e.get("role") == "methodologist" and e.get("event") != "model_call"]
    usage = dict(ws.ledger.get("methodologist") or {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    first_invalid = next((e for e in events if e.get("event") == "invalid"), None)
    fallback = next((e for e in events if e.get("event") == "fallback"), None)
    if first_invalid is not None:
        n_errors = len([x for x in _ERRORS_IN_EVENT.split(str(first_invalid.get("detail") or "")) if x])
    elif fallback is not None:
        n_errors = len([x for x in _ERRORS_IN_EVENT.split(str(fallback.get("detail") or "").split(": ", 1)[-1]) if x])
    else:
        n_errors = 0
    detail: dict[str, Any] = {
        "usage": usage, "model": ws.model, "provider": ws.provider, "validator_errors_first_try": n_errors,
        "first_try_valid": first_invalid is None and fallback is None,
        "events": [f"{e.get('event')}: {str(e.get('detail') or '')[:160]}" for e in events][:12],
    }
    if study is None:
        detail["kind"] = "declined"
        return Candidate(declined=True, reason=ws.declined_reason, kind="declined"), detail
    cand = candidate_from(study)
    plan = study.plan or {}
    detail["author"] = plan.get("author")
    detail["fallback_to_tree"] = bool(llm) and plan.get("author") == "playbook"
    detail["branch"] = plan.get("branch")
    detail["notes"] = list(plan.get("notes") or [])[:6]
    detail["model_plan_rejected"] = list(plan.get("model_plan_rejected") or [])[:6]
    return cand, detail


def file_candidate(ref: Reference, candidates_dir: str | Path) -> tuple[Candidate, dict[str, Any]]:
    """A plan produced elsewhere: ``<candidates_dir>/<case id>.json``, a study, a workspace or a decline object."""
    path = Path(candidates_dir) / f"{ref.id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no candidate plan for {ref.id} at {path}")
    with path.open(encoding="utf-8") as fh:
        obj = json.load(fh)
    cand = candidate_from(obj)
    detail: dict[str, Any] = {"path": str(path), "author": cand.author, "branch": cand.branch}
    if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
        detail["usage"] = dict(obj["usage"])
    if isinstance(obj, dict) and obj.get("model"):
        detail["model"] = obj.get("model")
        detail["provider"] = obj.get("provider")
    return cand, detail


# ── the scorer ──────────────────────────────────────────────────────────────


def _group(names: Iterable[str]) -> str:
    """``a`` or ``a (or b, c)`` for a set of interchangeable tools or methods."""
    items = sorted(names)
    return items[0] + (f" (or {', '.join(items[1:])})" if len(items) > 1 else "")


def _gate_matches(ref_path: str | None, got_path: str | None) -> bool:
    if not ref_path:
        return True
    if not got_path:
        return False
    return got_path == ref_path or got_path.startswith(ref_path + ".")


def score_plan(reference: Reference | dict[str, Any], candidate: Any) -> dict[str, Any]:
    """Score a candidate plan against a reference; see the module docstring for the formula.

    ``reference`` is a :class:`Reference` or its dict; ``candidate`` is
    anything :func:`candidate_from` reads (a study, a workspace, a decline).
    Returns ``coverage_tools``, ``coverage_methods``, ``coverage_gates``,
    ``extraneous``, ``forbidden_used`` (a count), ``decline_correct``,
    ``validator_errors_first_try`` (None here; the bench fills it in),
    ``score`` and ``explain`` (one sentence per finding).
    """
    ref = reference if isinstance(reference, Reference) else Reference.from_dict(reference)
    cand = candidate if isinstance(candidate, Candidate) else candidate_from(candidate)
    out: dict[str, Any] = {
        "coverage_tools": None, "coverage_methods": None, "coverage_gates": None, "extraneous": None,
        "forbidden_used": 0, "decline_correct": None, "validator_errors_first_try": None, "score": 0.0,
        "explain": [],
    }
    explain: list[str] = out["explain"]
    if ref.decline:
        out["decline_correct"] = bool(cand.declined)
        if cand.declined:
            explain.append(f"declined, as the reference does ({ref.decline_kind}): {str(cand.reason or '')[:160]}")
            if ref.decline_kind and cand.kind and cand.kind != ref.decline_kind:
                explain.append(f"the decline kind is {cand.kind!r}; the reference expects {ref.decline_kind!r}")
        else:
            explain.append(f"the reference declines ({ref.decline_kind}); the plan has {len(cand.steps)} step(s)")
        out["score"] = 1.0 if cand.declined else 0.0
        return out
    out["decline_correct"] = not cand.declined
    if cand.declined:
        explain.append(f"declined a solvable case: {str(cand.reason or '')[:160]}")
        return out
    required = ref.required_steps
    req_tools = {s.tools for s in required}
    req_methods = {s.methods for s in required if s.methods}
    req_gates = {(s.tool, s.tools, g["check"], g.get("path")) for s in required for g in s.gates}
    allowed = set().union(*(s.tools for s in ref.steps)) | FRAMING_TOOLS if ref.steps else set(FRAMING_TOOLS)
    steps = cand.steps
    tools = [st["tool"] for st in steps]
    methods = {st["method"] for st in steps if st.get("method")}
    got_gates = [(st["tool"], g.get("check"), g.get("path")) for st in steps for g in st.get("gates") or []]

    if not steps:
        explain.append("the plan has no steps")
        out.update({"coverage_tools": 0.0, "coverage_methods": 0.0 if req_methods else None,
                    "coverage_gates": 0.0 if req_gates else None, "extraneous": 0.0})
        return out

    have = set(tools)
    missing_tools = sorted((g for g in req_tools if not g & have), key=sorted)
    out["coverage_tools"] = (len(req_tools) - len(missing_tools)) / len(req_tools) if req_tools else None
    explain += [f"missing tool {_group(g)}" for g in missing_tools]
    if req_methods:
        missing_methods = sorted((g for g in req_methods if not g & methods), key=sorted)
        out["coverage_methods"] = (len(req_methods) - len(missing_methods)) / len(req_methods)
        explain += [f"missing method {_group(g)}" for g in missing_methods]
    if req_gates:
        hit = 0
        for tool, group, check, path in sorted(req_gates, key=str):
            if any(t in group and c == check and (_gate_matches(path, p) if t == tool else True)
                   for t, c, p in got_gates):
                hit += 1
            else:
                explain.append(f"missing gate {check} on {tool}" + (f" ({path})" if path else ""))
        out["coverage_gates"] = hit / len(req_gates)
    extra = [st for st in steps if st["tool"] not in allowed]
    out["extraneous"] = len(extra) / len(steps)
    explain += [f"extraneous step {st['id']} ({st['tool']}" + (f", {st['method']}" if st.get("method") else "") + ")"
                for st in extra]
    forbidden = 0
    for st in steps:
        hits = []
        if st["tool"] in ref.forbidden_tools:
            hits.append(f"tool {st['tool']}")
        if st.get("method") and st["method"] in ref.forbidden_methods:
            hits.append(f"method {st['method']}")
        fb = st.get("fallback") or {}
        if fb.get("tool") in ref.forbidden_tools:
            hits.append(f"fallback tool {fb['tool']}")
        if fb.get("method") and fb["method"] in ref.forbidden_methods:
            hits.append(f"fallback method {fb['method']}")
        if hits:
            forbidden += 1
            explain.append(f"forbidden in step {st['id']}: " + ", ".join(hits))
    out["forbidden_used"] = forbidden
    parts = {
        "tools": out["coverage_tools"], "methods": out["coverage_methods"], "gates": out["coverage_gates"],
        "clean": 1.0 if forbidden == 0 else 0.0, "parsimony": 1.0 - out["extraneous"],
    }
    present = {k: v for k, v in parts.items() if v is not None}
    total = sum(WEIGHTS[k] for k in present)
    out["score"] = round(sum(WEIGHTS[k] * v for k, v in present.items()) / total, 4) if total else 0.0
    if not explain:
        explain.append("the plan covers the reference")
    return out


# ── the bench ───────────────────────────────────────────────────────────────


@dataclass
class PlanResult:
    """One agent on one case, scored against the reference plan."""

    case_id: str
    playbook: str
    decline_expected: bool
    agent: str
    model: str | None
    provider: str | None
    repeat: int = 0
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    coverage_tools: float | None = None
    coverage_methods: float | None = None
    coverage_gates: float | None = None
    extraneous: float | None = None
    forbidden_used: int = 0
    decline_correct: bool | None = None
    validator_errors_first_try: int | None = None
    first_try_valid: bool | None = None
    fallback_to_tree: bool = False
    declined: bool = False
    declined_reason: str | None = None
    branch: str | None = None
    author: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    explain: list[str] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    seconds: float = 0.0
    error: str | None = None
    finished: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    phase: str = "plans"

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlanResult:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class _Config:
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    client: Any | None = None
    candidates_dir: str | Path | None = None
    plans_dir: str | Path | None = None

    @property
    def wants_model(self) -> bool:
        return self.client is not None or any((self.provider, self.model, self.api_key, self.base_url))


def _run(ref: Reference, agent: str, cfg: _Config) -> tuple[Candidate, dict[str, Any]]:
    if agent == "tree":
        return tree_candidate(ref, cfg.plans_dir)
    if agent == "methodologist":
        return methodologist_candidate(ref, provider=cfg.provider, model=cfg.model, api_key=cfg.api_key,
                                       base_url=cfg.base_url, client=cfg.client, plans_dir=cfg.plans_dir)
    if agent == "file":
        if not cfg.candidates_dir:
            raise ValueError("the file agent needs candidates_dir (a folder of <case id>.json plans)")
        return file_candidate(ref, cfg.candidates_dir)
    raise ValueError(f"unknown agent {agent!r}; one of {AGENTS}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _result(ref: Reference, agent: str, cfg: _Config, repeat: int, cand: Candidate, detail: dict[str, Any],
            seconds: float) -> PlanResult:
    scored = score_plan(ref, cand)
    usage = detail.get("usage") or {}
    model = detail.get("model") if detail.get("model") is not None else cfg.model
    provider = detail.get("provider") if detail.get("provider") is not None else cfg.provider
    res = PlanResult(
        case_id=ref.id, playbook=ref.playbook, decline_expected=ref.decline, agent=agent,
        model=model if agent != "tree" else None, provider=provider if agent != "tree" else None, repeat=repeat,
        tags=list(ref.tags), score=float(scored["score"]),
        coverage_tools=scored["coverage_tools"], coverage_methods=scored["coverage_methods"],
        coverage_gates=scored["coverage_gates"], extraneous=scored["extraneous"],
        forbidden_used=int(scored["forbidden_used"]), decline_correct=scored["decline_correct"],
        validator_errors_first_try=(int(detail["validator_errors_first_try"])
                                    if detail.get("validator_errors_first_try") is not None else (0 if agent == "tree"
                                                                                                  else None)),
        first_try_valid=detail.get("first_try_valid") if agent != "tree" else True,
        fallback_to_tree=bool(detail.get("fallback_to_tree")), declined=cand.declined, declined_reason=cand.reason,
        branch=cand.branch or detail.get("branch"), author=cand.author or detail.get("author"),
        steps=[{k: v for k, v in st.items() if k != "fallback" or v} for st in cand.steps],
        explain=list(scored["explain"]),
        calls=int(usage.get("calls", 0) or 0), prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0), seconds=round(seconds, 2), finished=_now(),
        detail={k: v for k, v in detail.items() if k not in ("usage", "model", "provider")},
    )
    res.cost_usd = estimate_cost(res.model, res.prompt_tokens, res.completion_tokens)
    return res


def _error_result(ref: Reference, agent: str, cfg: _Config, repeat: int, error: str, seconds: float) -> PlanResult:
    return PlanResult(case_id=ref.id, playbook=ref.playbook, decline_expected=ref.decline, agent=agent,
                      model=cfg.model if agent != "tree" else None, provider=cfg.provider if agent != "tree" else None,
                      repeat=repeat, tags=list(ref.tags), error=error, seconds=round(seconds, 2), finished=_now(),
                      cost_usd=0.0)


def select_references(refs: Iterable[Reference], *, limit: int | None = None,
                      case_ids: Iterable[str] | None = None) -> list[Reference]:
    pool = list(refs)
    if case_ids:
        wanted = set(case_ids)
        return [r for r in pool if r.id in wanted]
    return pool[:limit] if limit is not None else pool


def run_plan_bench(
    references: Iterable[Reference] | str | Path | None,
    agent: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    client: Any | None = None,
    candidates_dir: str | Path | None = None,
    limit: int | None = None,
    case_ids: Iterable[str] | None = None,
    repeats: int = 1,
    out: str | Path | None = None,
    resume: bool = False,
    timeout: float | None = DEFAULT_TIMEOUT,
    on_event: Callable[[str], None] | None = None,
) -> list[PlanResult]:
    """Play ``agent`` (``tree``, ``methodologist`` or ``file``) on the reference cases and score every plan.

    ``references`` is a list, a plans directory, or None for the package's.
    ``repeats`` plays every case that many times (a model's spread); the
    tree is deterministic and plays once. Results are appended to ``out`` as
    JSONL as they come; ``resume`` skips the (case, repeat) pairs ``out``
    already holds a finished row for. The event line after each case
    carries the spend so far.
    """
    if agent not in AGENTS:
        raise ValueError(f"unknown agent {agent!r}; one of {AGENTS}")
    if agent == "file" and not candidates_dir:
        raise ValueError("the file agent needs candidates_dir (a folder of <case id>.json plans)")
    say = on_event or (lambda _m: None)
    plans_dir = references if isinstance(references, (str, Path)) else None
    pool = load_references(plans_dir) if (references is None or plans_dir) else list(references)
    chosen = select_references(pool, limit=limit, case_ids=case_ids)
    cfg = _Config(provider=provider, model=model, api_key=api_key, base_url=base_url, client=client,
                  candidates_dir=candidates_dir, plans_dir=plans_dir)
    if agent == "methodologist" and cfg.wants_model and cfg.client is None:
        from aquascope.ai_engine.analyst import resolve_llm

        resolved = resolve_llm(provider, model, api_key, base_url)
        cfg = _Config(provider=str(resolved["provider"]), model=str(resolved["model"]), api_key=resolved["api_key"],
                      base_url=resolved["base_url"], candidates_dir=candidates_dir, plans_dir=plans_dir)
    n_repeats = max(1, int(repeats)) if agent == "methodologist" and cfg.wants_model else 1
    out_path = Path(out) if out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = set()
    if resume and out_path and out_path.exists():
        done = {(r.case_id, r.repeat) for r in load_plan_results([out_path])
                if r.agent == agent and (r.model or "") == (cfg.model or "") and not r.error}
        if done:
            say(f"resuming: {len(done)} of {len(chosen) * n_repeats} runs already have a row in {out_path}")
    results: list[PlanResult] = []
    spent = 0.0
    total = len(chosen) * n_repeats
    i = 0
    for repeat in range(n_repeats):
        for ref in chosen:
            i += 1
            if (ref.id, repeat) in done:
                continue
            say(f"[{i}/{total}] {ref.id}" + (f" (repeat {repeat})" if n_repeats > 1 else "")
                + f" ({'decline' if ref.decline else ref.expected_branch or 'off-tree'}) {ref.brief[:70]}")
            t0 = time.time()
            try:
                cand, detail = _with_timeout(lambda: _run(ref, agent, cfg), timeout)
                res = _result(ref, agent, cfg, repeat, cand, detail, time.time() - t0)
            except Exception as exc:  # noqa: BLE001 - one failed case is a row, not the end of the run
                res = _error_result(ref, agent, cfg, repeat, f"{type(exc).__name__}: {exc}"[:400], time.time() - t0)
                logger.warning("case %s failed: %s", ref.id, res.error)
            results.append(res)
            spent += res.cost_usd or 0.0
            say(f"    score {res.score:.2f}: tools {_pct(res.coverage_tools)}, methods {_pct(res.coverage_methods)}, "
                f"gates {_pct(res.coverage_gates)}, extraneous {_pct(res.extraneous)}, forbidden {res.forbidden_used}, "
                f"declined {res.declined}" + (f", errors first try {res.validator_errors_first_try}"
                                              if res.validator_errors_first_try else "")
                + (", tree fallback" if res.fallback_to_tree else "") + f", {res.tokens} tokens, {res.seconds} s"
                + (f", error {res.error}" if res.error else "")
                + (f"; spent {spent:.3f} USD so far" if cfg.wants_model else ""))
            if out_path:
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res.to_dict(), ensure_ascii=False, default=str) + "\n")
    return results


def rescore_plans(results: Iterable[PlanResult], references: Iterable[Reference] | None = None,
                  plans_dir: str | Path | None = None) -> list[PlanResult]:
    """Score stored rows again from the plans they carry (the steps, or the decline); the agent is not run again.

    For a change in a reference or in the formula after a run: ``score``,
    the coverage parts, ``extraneous``, ``forbidden_used``,
    ``decline_correct``, ``explain``, ``tags`` and ``decline_expected`` are
    recomputed; a row with an error, or whose case is not among the
    references, is returned as it is. Rows keep their order.
    """
    refs = {r.id: r for r in (references if references is not None else load_references(plans_dir))}
    out: list[PlanResult] = []
    for r in results:
        ref = refs.get(r.case_id)
        if ref is None or r.error:
            out.append(r)
            continue
        cand = Candidate(steps=[dict(st, gates=list(st.get("gates") or []), fallback=st.get("fallback"))
                                for st in r.steps], declined=r.declined, reason=r.declined_reason,
                         branch=r.branch, author=r.author)
        scored = score_plan(ref, cand)
        r.score = float(scored["score"])
        for key in ("coverage_tools", "coverage_methods", "coverage_gates", "extraneous", "decline_correct"):
            setattr(r, key, scored[key])
        r.forbidden_used = int(scored["forbidden_used"])
        r.explain = list(scored["explain"])
        r.tags = list(ref.tags)
        r.decline_expected = ref.decline
        out.append(r)
    return out


# ── the leaderboard ─────────────────────────────────────────────────────────


def load_plan_results(paths: Iterable[str | Path], *, latest: bool = True) -> list[PlanResult]:
    """Plan-bench rows from JSONL files (rows of another shape, Phase 1 results among them, are skipped);
    ``latest`` keeps the last row per (agent, model, provider, case, repeat)."""
    out: list[PlanResult] = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get("case_id") and row.get("agent"):
                    out.append(PlanResult.from_dict(row))
    if not latest:
        return out
    last: dict[tuple[str, str, str, str, int], PlanResult] = {}
    for r in out:
        last[(r.agent, r.model or "", r.provider or "", r.case_id, int(r.repeat or 0))] = r
    return list(last.values())


def _mean(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize_plans(results: Iterable[PlanResult]) -> list[dict[str, Any]]:
    """One row per (agent, model): the mean score over all cases and over the solvable ones, the spread across
    repeats, the decline rates, the coverage means, extraneous and forbidden rates, first-try validity, tree
    fallbacks, tokens, seconds, cost, errors, and the mean score per playbook."""
    groups: dict[tuple[str, str, str], list[PlanResult]] = {}
    for r in results:
        groups.setdefault((r.agent, r.model or "", r.provider or ""), []).append(r)
    rows = []
    for (agent, model, provider), rs in sorted(groups.items()):
        solvable = [r for r in rs if not r.decline_expected]
        hard = [r for r in rs if r.decline_expected]
        off_tree = [r for r in solvable if "off_tree" in (r.tags or [])]
        by_repeat: dict[int, list[float]] = {}
        for r in rs:
            by_repeat.setdefault(int(r.repeat or 0), []).append(r.score if not r.error else 0.0)
        run_means = [sum(v) / len(v) for v in by_repeat.values() if v]
        by_pb: dict[str, list[float]] = {}
        for r in rs:
            by_pb.setdefault(r.playbook, []).append(r.score if not r.error else 0.0)
        cost = [r.cost_usd if r.cost_usd is not None or r.tokens else 0.0 for r in rs]
        rows.append({
            "agent": agent, "model": model or None, "provider": provider or None,
            "n": len(rs), "n_cases": len({r.case_id for r in rs}), "n_solvable": len({r.case_id for r in solvable}),
            "n_decline": len({r.case_id for r in hard}),
            "repeats": len(by_repeat),
            "score": _mean([r.score if not r.error else 0.0 for r in rs]),
            "score_solvable": _mean([r.score if not r.error else 0.0 for r in solvable]),
            "score_off_tree": _mean([r.score if not r.error else 0.0 for r in off_tree]),
            "score_min_run": round(min(run_means), 3) if run_means else None,
            "score_max_run": round(max(run_means), 3) if run_means else None,
            "decline_rate": _mean([bool(r.decline_correct) for r in hard if not r.error]),
            "false_decline_rate": _mean([r.declined and not r.error for r in solvable]),
            "coverage_tools": _mean([r.coverage_tools for r in solvable if not r.error]),
            "coverage_methods": _mean([r.coverage_methods for r in solvable if not r.error]),
            "coverage_gates": _mean([r.coverage_gates for r in solvable if not r.error]),
            "extraneous": _mean([r.extraneous for r in solvable if not r.error]),
            "forbidden_rate": _mean([r.forbidden_used > 0 for r in solvable if not r.error]),
            "first_try_valid": _mean([r.first_try_valid for r in rs if not r.error and r.first_try_valid is not None]),
            "fallback_rate": _mean([r.fallback_to_tree for r in rs if not r.error]),
            "tokens_per_case": _mean([r.tokens for r in rs]),
            "prompt_tokens": sum(r.prompt_tokens for r in rs),
            "completion_tokens": sum(r.completion_tokens for r in rs),
            "seconds_per_case": _mean([r.seconds for r in rs]),
            "cost_usd": round(sum(c for c in cost if c is not None), 4) if all(c is not None for c in cost) else None,
            "errors": sum(1 for r in rs if r.error),
            "by_playbook": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_pb.items()) if v},
        })
    return rows


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f} %"


def _num(x: float | None, digits: int = 0) -> str:
    return "-" if x is None else f"{x:,.{digits}f}"


def _label(r: dict[str, Any]) -> str:
    return r["model"] or ("keyless" if r["agent"] == "methodologist" else "none")


def plan_leaderboard(results: Iterable[PlanResult], *, out: str | Path | None = None,
                     title: str | None = None) -> str:
    """The Markdown leaderboard of plan quality (any agents, any models), written to ``out`` when given."""
    rs = list(results)
    rows = summarize_plans(rs)
    n_cases = len({r.case_id for r in rs})
    lines = [
        f"## {title or 'HydroGym plan-quality leaderboard'}", "",
        f"{n_cases} cases, {len(rs)} runs, {len(rows)} agent-model pairs. Score is the mean over every case of the "
        "plan-quality score (weights " + ", ".join(f"{k} {v:g}" for k, v in WEIGHTS.items())
        + "; a declining case scores 1 for a decline and 0 for a plan; an error scores 0); solvable is the mean over "
        "the cases the reference does not decline, off-tree over the cases no single playbook branch covers; spread "
        "is the range of the per-run means when a model was played more than once; declined is the share of "
        "declining cases the agent refused; false declines are solvable cases it refused; tools, methods and gates "
        "are the mean coverage of the reference's required tools, methods and gates; extraneous is the mean share "
        "of a plan's steps the reference does not name; forbidden is the share of plans that use a tool or method "
        "not defensible at the site; valid first try is the share of model plans the validator accepted at once; "
        "tree fallback is the share of plans that are the tree's because the model's did not pass.", "",
        "| agent | model | cases (solvable + decline) | score | solvable | off-tree | spread | declined "
        "| false declines | tools | methods | gates | extraneous | forbidden | valid first try | tree fallback "
        "| tokens/case | s/case | cost USD | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        spread = ("-" if r["repeats"] < 2 or r["score_min_run"] is None
                  else f"{r['score_min_run']:.2f} to {r['score_max_run']:.2f}")
        lines.append(
            f"| {r['agent']} | {_label(r)} | {r['n_cases']} ({r['n_solvable']} + {r['n_decline']})"
            + (f" x{r['repeats']}" if r["repeats"] > 1 else "")
            + f" | {_num(r['score'], 2)} | {_num(r['score_solvable'], 2)} | {_num(r['score_off_tree'], 2)} | {spread} "
            f"| {_pct(r['decline_rate'])} | {_pct(r['false_decline_rate'])} | {_pct(r['coverage_tools'])} "
            f"| {_pct(r['coverage_methods'])} | {_pct(r['coverage_gates'])} | {_pct(r['extraneous'])} "
            f"| {_pct(r['forbidden_rate'])} | {_pct(r['first_try_valid'])} | {_pct(r['fallback_rate'])} "
            f"| {_num(r['tokens_per_case'])} | {_num(r['seconds_per_case'], 1)} | {_num(r['cost_usd'], 3)} "
            f"| {r['errors']} |"
        )
    playbooks = sorted({p for r in rows for p in r["by_playbook"]})
    if playbooks:
        lines += ["", "Mean score by playbook:", "", "| agent | model | " + " | ".join(playbooks) + " |",
                  "|---|---|" + "---|" * len(playbooks)]
        for r in rows:
            cells = [f"{r['by_playbook'][p]:.2f}" if p in r["by_playbook"] else "-" for p in playbooks]
            lines.append(f"| {r['agent']} | {_label(r)} | " + " | ".join(cells) + " |")
    lines += ["", PRICES_NOTE, ""]
    text = "\n".join(lines)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
    return text
