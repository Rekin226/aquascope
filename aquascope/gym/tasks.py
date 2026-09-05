"""HydroGym Phase 1: a verifiable benchmark of hydrology agents on real sites, generated from the playbooks (#175).

A task is a problem at a place: a playbook, a site (a catalog gauge or a bare
point), the intake, and a snapshot of the reconnaissance (``assess_site``)
taken when the task was made. The scoring key is what the playbook's tree
says for that reconnaissance with no model in the loop: the branch it selects,
the gates that branch expects, the tools it would call, or the sentence it
prints when it declines. A task whose playbook declines is *unsolvable*: the
right answer is to refuse, and an agent that quotes a number instead is wrong.

Two kinds of unsolvable task exist. The data-driven ones arise on their own
(an ungauged point with fewer than three donor gauges, a return period far
beyond the record with no donors). The *probes* are read off each playbook's
own decline rules: every rule whose conditions are all over ``intake.*``
fields (inundation extent, attributing the cause of a groundwater decline)
becomes an intake that triggers it, so the suite always has out-of-scope asks
that a well-behaved agent has to turn down.

Sites are held out by a deterministic hash of the site (one in four goes to
``test``), so an agent tuned on ``train`` can be checked on sites it never
saw. Tasks serialise to JSONL; ``aquascope gym tasks`` writes them and
``aquascope gym bench`` (:mod:`aquascope.gym.bench`) plays agents on them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "SITE_KINDS",
    "SPLITS",
    "Task",
    "decline_probes",
    "on_land_basinatlas",
    "problem_text",
    "read_tasks",
    "scoring_key",
    "site_key",
    "split_for",
    "suggest_sites",
    "tasks_from_playbooks",
    "write_tasks",
]

SPLITS = ("train", "test")
#: One site in this many is held out.
TEST_EVERY = 4
SITE_KINDS = ("gauged_long", "gauged_short", "groundwater", "ungauged")
#: Record lengths that make a catalog row a long or a short gauge (the playbooks' own thresholds).
LONG_YEARS = 20.0
SHORT_YEARS = 5.0
WELL_YEARS = 10.0
#: An ungauged point needs catalog gauges in this many of the four quadrants around it (within about 0.8
#: degrees): a crude proxy for being on land, which an island cluster on one side of the point fails.
MIN_QUADRANTS = 3

_CONTINENT_BY_COUNTRY = {
    "north_america": {"USA", "CAN", "MEX"},
    "south_america": {"BRA", "CHL", "ARG", "PER", "COL", "URY", "PRY", "BOL", "ECU", "VEN"},
    "europe": {"GBR", "FRA", "DEU", "IRL", "ESP", "ITA", "PRT", "NLD", "BEL", "CHE", "AUT", "NOR", "SWE", "FIN",
               "DNK", "POL", "CZE"},
    "asia": {"TWN", "JPN", "KOR", "IND", "CHN", "IDN", "THA", "VNM", "MYS", "PHL"},
    "oceania": {"AUS", "NZL"},
    "africa": {"ZAF", "KEN", "NGA", "EGY", "MAR", "ETH", "TZA", "GHA", "BFA"},
}


@dataclass
class Task:
    """One benchmark task: a playbook at a site with an intake, and the tree's verdict as the scoring key."""

    id: str
    playbook: str
    #: ``{lat, lon, source, station_id, name, kind, country, continent, years}``; source and id are None for a point.
    site: dict[str, Any]
    intake: dict[str, Any]
    #: The problem in plain language, what an agent is given.
    problem: str
    #: The reconnaissance the key was computed on (``assess_site`` at generation time).
    recon: dict[str, Any]
    #: ``{branch, gates: [{step, check, path}], tools: [...], declined: bool, decline_reason, decline_kind, notes}``.
    expected: dict[str, Any]
    split: str = "train"
    created: str = ""
    #: Which decline rule this task probes, when it is a probe (else None).
    probe: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def unsolvable(self) -> bool:
        return bool(self.expected.get("declined"))

    @property
    def lat(self) -> float:
        return float(self.site["lat"])

    @property
    def lon(self) -> float:
        return float(self.site["lon"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# ── sites ────────────────────────────────────────────────────────────────────


def site_key(site: dict[str, Any]) -> str:
    """What identifies a site for the split and the task id: the gauge, else the rounded position."""
    if site.get("source") and site.get("station_id"):
        return f"{site['source']}/{site['station_id']}"
    return f"{float(site['lat']):.4f},{float(site['lon']):.4f}"


def split_for(site: dict[str, Any]) -> str:
    """``test`` for one site in :data:`TEST_EVERY` by a hash of the site, ``train`` otherwise. Deterministic."""
    digest = hashlib.sha1(site_key(site).encode("utf-8")).hexdigest()
    return "test" if int(digest[:8], 16) % TEST_EVERY == 0 else "train"


def _continent(country: str | None, lat: float, lon: float) -> str:
    for name, codes in _CONTINENT_BY_COUNTRY.items():
        if country in codes:
            return name
    if lat < -10 and 110 <= lon <= 180:
        return "oceania"
    if -35 <= lat <= 38 and -20 <= lon <= 52:
        return "africa"
    if lat > 35 and -12 <= lon <= 45:
        return "europe"
    if lon < -30:
        return "north_america" if lat > 12 else "south_america"
    return "asia"


def _parse_date(text: Any) -> date | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text)[:10]).date()
    except ValueError:
        return None


def _span_years(start: Any, end: Any, today: date) -> float | None:
    """Record length from the catalog span; an open end (a station still listed as open) runs to today.

    The same reading as the reconnaissance (``explore._span_years``), so the
    kind the sampler gives a gauge is the record the key will be computed on.
    The collectors leave ``period_end`` empty for an open station (uk_ea
    ``dateClosed``, hubeau ``date_fermeture_station``); a row with no start
    at all has no span.
    """
    a = _parse_date(start)
    if a is None:
        return None
    b = min(_parse_date(end) or today, today)
    if b < a:
        return None
    return round((b - a).days / 365.25, 1)


def _classify(row: dict[str, Any], today: date) -> tuple[str, float] | None:
    """The site kind a catalog row makes, with its years, or None when it makes none."""
    years = _span_years(row.get("period_start"), row.get("period_end"), today)
    if years is None or row.get("latitude") is None or row.get("longitude") is None:
        return None
    variables = set(row.get("variables") or [])
    if "discharge" in variables:
        if years >= LONG_YEARS:
            return "gauged_long", years
        if years >= SHORT_YEARS:
            return "gauged_short", years
        return None
    if "groundwater_level" in variables and years >= WELL_YEARS:
        return "groundwater", years
    return None


def _site_from_row(row: dict[str, Any], kind: str, years: float) -> dict[str, Any]:
    lat, lon = float(row["latitude"]), float(row["longitude"])
    return {
        "lat": round(lat, 5), "lon": round(lon, 5), "source": row.get("source"), "station_id": row.get("station_id"),
        "name": row.get("name"), "kind": kind, "country": row.get("country"),
        "continent": _continent(row.get("country"), lat, lon), "years": years,
        "variables": list(row.get("variables") or []),
    }


def suggest_sites(
    n: int = 12,
    *,
    seed: int = 0,
    sources: list[str] | None = None,
    catalog: list[dict[str, Any]] | None = None,
    ungauged_share: float = 0.25,
    today: date | None = None,
    on_land: Callable[[float, float], bool] | None = None,
) -> list[dict[str, Any]]:
    """``n`` sites for a task suite, sampled from the station catalog with no agency call.

    Gauged sites come in two lengths (20 years and more, 5 to 20 years) and
    wells with ten years of levels make a third kind; the sampler draws round
    robin over (kind, continent, source) so a suite spans the sources and the
    continents the catalog covers rather than the source with the most rows.
    The rest (``ungauged_share``) are bare points: a random offset of 0.5 to
    0.9 degrees from a gauge in a sparse part of the catalog, so the point is
    on land near measured rivers but usually beyond the 50 km radius of any
    gauge. A point must have gauges around it in three of the four quadrants
    (so an island's cluster does not put it at sea) and pass ``on_land`` when
    one is given (:func:`on_land_basinatlas` asks BasinATLAS, which is what
    the reconnaissance consults). Reproducible for a ``seed``.
    """
    if n <= 0:
        return []
    rng = random.Random(seed)
    today = today or datetime.now(timezone.utc).date()
    if catalog is None:
        from aquascope.archive.catalog import load_stations

        catalog = load_stations()
    wanted = set(sources or [])
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in catalog:
        if wanted and row.get("source") not in wanted:
            continue
        verdict = _classify(row, today)
        if verdict is None:
            continue
        kind, years = verdict
        site = _site_from_row(row, kind, years)
        buckets.setdefault((kind, site["continent"], str(site["source"])), []).append(site)
        if kind != "groundwater":
            cells.setdefault((int(site["lat"] // 1), int(site["lon"] // 1)), []).append(site)
    for group in buckets.values():
        rng.shuffle(group)
    n_ungauged = min(n, int(round(n * ungauged_share))) if cells else 0
    n_gauged = n - n_ungauged

    # Round robin: every kind in turn, and within a kind the (continent, source) buckets in turn.
    out: list[dict[str, Any]] = []
    by_kind: dict[str, list[list[dict[str, Any]]]] = {}
    for key in sorted(buckets):
        by_kind.setdefault(key[0], []).append(buckets[key])
    for groups in by_kind.values():
        rng.shuffle(groups)
    kinds = [k for k in SITE_KINDS if k in by_kind]
    cursor = dict.fromkeys(kinds, 0)
    while len(out) < n_gauged and any(any(g for g in by_kind[k]) for k in kinds):
        for kind in kinds:
            if len(out) >= n_gauged:
                break
            groups = by_kind[kind]
            for _ in range(len(groups)):
                i = cursor[kind] % len(groups)
                cursor[kind] += 1
                if groups[i]:
                    out.append(groups[i].pop())
                    break

    # Ungauged points: offsets from gauges in sparse 1-degree cells, sparse first. A candidate must be surrounded
    # by gauges (three quadrants) and pass on_land; an anchor whose candidates all fail is skipped.
    sparse = sorted(cells.items(), key=lambda kv: (len(kv[1]), kv[0]))
    picks = [g for _, group in sparse[: max(4 * n_ungauged, 1)] for g in group]
    picks += [g for _, group in sparse[max(4 * n_ungauged, 1):] for g in group]
    rng.shuffle(picks)
    points: list[dict[str, Any]] = []
    for anchor in picks:
        if len(points) >= n_ungauged:
            break
        for _ in range(12):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(0.5, 0.9)
            lat = max(-89.0, min(89.0, anchor["lat"] + dist * math.sin(angle)))
            lon = ((anchor["lon"] + dist * math.cos(angle) + 180) % 360) - 180
            if not _surrounded(cells, lat, lon):
                continue
            if on_land is not None and not on_land(lat, lon):
                continue
            points.append({
                "lat": round(lat, 4), "lon": round(lon, 4), "source": None, "station_id": None, "name": None,
                "kind": "ungauged", "country": anchor.get("country"), "continent": anchor.get("continent"),
                "years": None, "variables": [],
                "anchor": f"{anchor['source']}/{anchor['station_id']}",
            })
            break
    out += points
    # Short of points: top up with gauged sites so the suite keeps its size.
    while len(out) < n and any(g for groups in by_kind.values() for g in groups):
        for kind in kinds:
            group = next((g for g in by_kind[kind] if g), None)
            if group and len(out) < n:
                out.append(group.pop())
    return out[:n]


def _surrounded(cells: dict[tuple[int, int], list[dict[str, Any]]], lat: float, lon: float,
                radius_deg: float = 0.8, quadrants: int = MIN_QUADRANTS) -> bool:
    """Whether catalog gauges lie within ``radius_deg`` of a point in at least ``quadrants`` of its four quadrants."""
    seen: set[tuple[bool, bool]] = set()
    ci, cj = int(lat // 1), int(lon // 1)
    for i in (ci - 1, ci, ci + 1):
        for j in (cj - 1, cj, cj + 1):
            for g in cells.get((i, j), ()):
                dlat, dlon = g["lat"] - lat, g["lon"] - lon
                if abs(dlat) <= radius_deg and abs(dlon) <= radius_deg and (dlat or dlon):
                    seen.add((dlat >= 0, dlon >= 0))
                    if len(seen) >= quadrants:
                        return True
    return False


def on_land_basinatlas(lat: float, lon: float) -> bool:
    """Whether BasinATLAS has a sub-basin at the point (what the reconnaissance consults); False at sea."""
    try:
        from aquascope.mcp_server import describe_catchment

        return not describe_catchment(lat, lon, upstream=False).get("error")
    except Exception as exc:  # noqa: BLE001 - unreachable BasinATLAS is not a verdict on the point
        logger.info("BasinATLAS check skipped at %.3f, %.3f: %s", lat, lon, exc)
        return True


# ── the problem text and the intake probes ──────────────────────────────────

_PURPOSE = {
    "irrigation offtake": "an irrigation offtake",
    "environmental flow": "an environmental flow assessment",
    "hydropower screening": "a micro-hydro screening",
    "water supply": "a water supply intake",
}


def problem_text(playbook: str, intake: dict[str, Any] | None = None) -> str:
    """The problem in plain language for a playbook and an intake: what an agent is given.

    Written so the Coordinator's keyword rules and ``intake_hints`` read the
    same playbook and intake off it that the key was computed with.
    """
    intake = dict(intake or {})
    if playbook == "flood_risk":
        rp = intake.get("return_period") or 100
        decision = str(intake.get("decision") or "design flow")
        if decision == "inundation extent":
            return (f"Which streets and fields around this point flood in a {rp}-year event, and how deep? "
                    "Map the inundation extent.")
        if decision == "risk screening":
            return f"Is this site at risk of flooding? Screen the {rp}-year flood at this point."
        if decision == "insurance":
            return f"Flood risk at this point for insurance: the {rp}-year return period flow."
        return f"Design flow for a road crossing at this point, {rp}-year return period."
    if playbook == "ungauged_flow":
        purpose = _PURPOSE.get(str(intake.get("purpose") or ""), "")
        head = "What flow can this ungauged stream give" + (f" {purpose}" if purpose else "")
        stat = str(intake.get("statistic") or "all")
        tail = {"mean": "the mean flow", "Q95": "the Q95 low flow", "Q05": "the Q05 high flow"}.get(
            stat, "mean flow, Q95 and Q05")
        return f"{head}: {tail}?"
    if playbook == "groundwater_decline":
        if intake.get("attribute_cause"):
            return "Is the water table under a well at this point falling, and why: is it pumping?"
        return "Is the water table under a well at this point falling, and how fast?"
    if playbook == "drought_status":
        if intake.get("flash_drought"):
            return "Is a flash drought setting in at this point this week? Drought status over the last weeks."
        concern = _CONCERN.get(str(intake.get("drought_concern") or ""), "")
        return f"Is this point in drought{concern}? The drought status by SPI and SPEI."
    if playbook == "supply_reliability":
        if intake.get("demand_ml_day") is not None and intake.get("demand_m3s") is None:
            demand = f"{intake['demand_ml_day']:g} ML/day"
        else:
            demand = f"{intake.get('demand_m3s') or 2:g} m3/s"
        use = _USE.get(str(intake.get("use") or ""), "")
        if intake.get("storage"):
            return f"Can a reservoir on this river supply {demand}{use}? How reliable is the supply?"
        return f"Can the river at this point supply {demand}{use}? How reliable is the supply?"
    if playbook == "irrigation_feasibility":
        crop = str(intake.get("crop") or "maize").replace("_", " ")
        area = intake.get("area_ha") or 10
        month = _MONTH_NAMES[int(intake.get("planting_month") or 4) - 1]
        if str(intake.get("decision") or "") == "daily schedule":
            return (f"Give me the daily irrigation schedule for {area:g} ha of {crop} planted in {month} at this "
                    "point: when to irrigate and how much each day.")
        return f"Can I irrigate {area:g} ha of {crop} planted in {month} at this point from the river?"
    from aquascope import playbooks as pbk

    try:
        return f"{pbk.load(playbook).title} at this point."
    except Exception:  # noqa: BLE001 - an unknown playbook still gets a sentence
        return f"{playbook.replace('_', ' ')} at this point."


_CONCERN = {"agriculture": " for the crops", "water supply": " for the water supply",
            "groundwater": " for the groundwater"}
_USE = {"municipal": " to a town", "irrigation": " to an irrigation scheme", "industrial": " to a factory"}
_MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December")


def decline_probes(playbook: Any) -> list[dict[str, Any]]:
    """Intakes that trigger a playbook's decline rules whose conditions are all over ``intake.*`` fields.

    Each is ``{"intake": {...}, "rule": "<the first words of the sentence>"}``.
    A rule that also reads the reconnaissance (a return period beyond the
    record) is not a probe: whether it fires depends on the site.
    """
    from aquascope import playbooks as pbk

    pb = pbk.load(playbook)
    out: list[dict[str, Any]] = []
    for rule in pb.declines:
        intake: dict[str, Any] = {}
        for cond in rule.when:
            if not cond.path.startswith("intake.") or cond.op != "==":
                intake = {}
                break
            intake[cond.path.split(".", 1)[1]] = cond.value
        if intake:
            out.append({"intake": intake, "rule": " ".join(rule.say.split()[:6])})
    return out


# ── the scoring key ─────────────────────────────────────────────────────────


def scoring_key(playbook: Any, recon: dict[str, Any], intake: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the tree says for this reconnaissance and intake, as the key a bench scores against."""
    from aquascope import playbooks as pbk

    pb = pbk.load(playbook)
    try:
        study = pbk.plan(pb, recon, intake)
    except pbk.Declined as exc:
        return {"branch": None, "gates": [], "tools": [], "declined": True, "decline_reason": exc.reason,
                "decline_kind": exc.kind, "notes": []}
    gates = [{"step": s.id, "check": g.get("check"), "path": g.get("path") or g.get("paths")}
             for s in study.steps for g in s.expects if isinstance(g, dict)]
    plan = study.plan or {}
    return {
        "branch": plan.get("branch"), "gates": gates, "tools": [s.tool for s in study.steps],
        "declined": False, "decline_reason": None, "decline_kind": None, "notes": list(plan.get("notes") or []),
        "station": plan.get("station"),
    }


def _task_id(playbook: str, site: dict[str, Any], intake: dict[str, Any]) -> str:
    raw = f"{playbook}|{site_key(site)}|{json.dumps(intake, sort_keys=True, default=str)}"
    return f"{playbook}-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"


def tasks_from_playbooks(
    sites: Iterable[dict[str, Any]],
    playbooks: Iterable[Any] | None = None,
    *,
    recon: Callable[..., dict[str, Any]] | None = None,
    probes: int | None = 1,
    on_event: Callable[[str], None] | None = None,
    skip_unreachable: bool = True,
    skipped: list[dict[str, Any]] | None = None,
) -> list[Task]:
    """Tasks for every site and playbook: the tree run on the reconnaissance gives the scoring key.

    ``recon(lat, lon)`` is called once per site (``assess_site`` by default,
    which reads the catalog, BasinATLAS and the donor search but no agency
    record). ``probes`` is how many decline probes a site gets on top of the
    base task per playbook: they rotate through every playbook's intake-only
    decline rules across the sites, so the suite exercises each rule; ``None``
    gives every probe at every site, ``0`` none. A task whose playbook declines
    (a probe, or a site the data cannot support) is unsolvable.

    A site whose reconnaissance raises (the network, a source that is down)
    is skipped with a note (``skip_unreachable``; the site and the error go
    to ``skipped`` when a list is given), because a key computed on an empty
    snapshot would call a gauged site ungauged. With ``skip_unreachable=False``
    the site keeps its tasks on the empty snapshot, the failure noted in
    ``recon.notes``. A task whose key the tree cannot compute (a
    ``PlaybookError``, neither a branch nor a decline) is skipped the same way,
    with the playbook named in the ``skipped`` entry.
    """
    from aquascope import playbooks as pbk

    say = on_event or (lambda _m: None)
    if recon is None:
        from aquascope.explore import assess_site as recon  # type: ignore[assignment]
    ids = [p["id"] for p in pbk.list_playbooks() if "error" not in p] if playbooks is None else list(playbooks)
    loaded = [pbk.load(p) for p in ids]
    all_probes = [(pb, probe) for pb in loaded for probe in decline_probes(pb)]
    tasks: list[Task] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i, site in enumerate(sites):
        lat, lon = float(site["lat"]), float(site["lon"])
        try:
            snapshot = recon(lat, lon)
        except Exception as exc:  # noqa: BLE001 - a site the reconnaissance cannot see is skipped or noted
            note = f"reconnaissance unavailable: {type(exc).__name__}: {exc}"
            if skip_unreachable:
                logger.warning("site %s skipped: %s", site_key(site), note)
                say(f"site {i + 1} {site_key(site)}: skipped, {note[:120]}")
                if skipped is not None:
                    skipped.append({"site": dict(site), "error": note})
                continue
            snapshot = {"point": {"lat": lat, "lon": lon}, "stations": [], "catchment": {},
                        "context": {"years_by_variable": {}, "resolution_by_variable": {}, "ungauged": True},
                        "sufficiency": [], "notes": [note]}
        years = (snapshot.get("context") or {}).get("years_by_variable") or {}
        say(f"site {i + 1} {site_key(site)}: " + (", ".join(f"{k} {v:g} yr" for k, v in years.items()) or "no record"))
        split = split_for(site)
        wanted: list[tuple[Any, dict[str, Any], str | None]] = [(pb, {}, None) for pb in loaded]
        if probes is None:
            wanted += [(pb, dict(probe["intake"]), probe["rule"]) for pb, probe in all_probes]
        elif probes > 0 and all_probes:
            for j in range(probes):
                pb, probe = all_probes[(i * probes + j) % len(all_probes)]
                wanted.append((pb, dict(probe["intake"]), probe["rule"]))
        for pb, raw_intake, rule in wanted:
            intake = pbk.fill_intake(pb, raw_intake)
            try:
                key = scoring_key(pb, snapshot, intake)
            except pbk.PlaybookError as exc:
                # Neither a branch nor a decline: the tree itself failed on this snapshot. No key, no task.
                note = f"no key: {type(exc).__name__}: {exc}"
                logger.warning("task %s at %s skipped: %s", pb.id, site_key(site), note)
                say(f"  {pb.id}" + (f" [{rule}]" if rule else "") + f": skipped, {note[:100]}")
                if skipped is not None:
                    skipped.append({"site": dict(site), "playbook": pb.id, "intake": intake, "error": note})
                continue
            tasks.append(Task(
                id=_task_id(pb.id, site, intake), playbook=pb.id, site=dict(site), intake=intake,
                problem=problem_text(pb.id, intake), recon=snapshot, expected=key, split=split, created=now,
                probe=rule,
            ))
            say(f"  {pb.id}" + (f" [{rule}]" if rule else "") + ": "
                + ("declined, " + str(key["decline_reason"])[:60] if key["declined"] else f"branch {key['branch']}"))
    return tasks


# ── JSONL ────────────────────────────────────────────────────────────────────


def write_tasks(tasks: Iterable[Task], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False, default=str) + "\n")
    return path


def read_tasks(path: str | Path) -> list[Task]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Task.from_dict(json.loads(line)))
    return out
