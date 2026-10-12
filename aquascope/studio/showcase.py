"""Recorded studies: the crew's worked examples, run once with a model and replayed keyless in the Explorer.

The Study drawer is keyless by default, so a visitor sees the playbook tree's
plan and template prose. What the crew does with a model (a composed
methodology, a Critic with opinions, an Author who writes) needs a model
once, not at every visit. So the maintainer records a dozen studies with a
model and commits the bundles; the page offers them as chips, re-runs the
numbers live in the browser with no key, and shows the recorded prose
labelled with the model and the date.

Per case the recording is a directory under ``out_dir``::

    <id>/workspace.json   the workspace without the artifact bytes (resume, replay)
    <id>/report.md        the report as the Author wrote it
    <id>/study.yaml       the plan; aquascope run study.yaml replays it with no model
    <id>/figures/*.png    the figures the Analysts drew
    <id>/meta.json        model, date, tokens, USD, seconds, gates, the headline

and ``index.json`` at the root lists them. Re-runs skip cases fresher than
``fresh_for_days``; a run stops at ``max_usd``. Like :mod:`aquascope.showcase`
(the Ask examples), this is a maintenance command::

    aquascope studio-showcase record --out explorer/showcase/studies [--only id] [--max-usd 15]
    aquascope studio-showcase list --out explorer/showcase/studies
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aquascope import __version__
from aquascope.ai_engine.providers import PRICES as _PROVIDER_PRICES

logger = logging.getLogger(__name__)

__all__ = ["CASES", "Case", "PRICES", "already_recorded", "diagnose", "headline", "load_index", "load_meta",
           "record", "synthetic_flows", "usd_for", "write_index"]

#: USD per million tokens (prompt, completion) the ledger's estimate uses, per model: the package's one price
#: table (:data:`aquascope.ai_engine.providers.PRICES`). The estimate is written into every meta.json with the
#: rate, so a reader can recompute it.
PRICES: dict[str, tuple[float, float]] = dict(_PROVIDER_PRICES)
DEFAULT_PRICE = (3.0, 15.0)

#: A recorded workspace above this many bytes has its result lists trimmed (the page holds every recording).
MAX_WORKSPACE_BYTES = 1_500_000
TRIM_TO = 50

#: The synthetic tables a case may attach, by file name.
TABLE_NAME = "my_flows.csv"


@dataclass
class Case:
    """One study to record: where, what was asked, what it shows."""

    id: str
    title: str
    lat: float
    lon: float
    problem: str
    #: The problem kind (the playbook id the brief should map to).
    kind: str
    #: A short place line for the index ("Thames at Kingston, London, UK").
    site: str
    shows: str = ""
    intake: dict[str, Any] = field(default_factory=dict)
    #: A synthetic table attached as ``upload:<name>`` (the "your own table" case).
    upload: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Twelve studies: flood, drought, supply, groundwater, ungauged flow, water quality, irrigation and a user's
#: table; gauged sites and bare points; Europe, North America, Asia, Africa and Oceania. Every gauged site was
#: checked with ``aquascope assess`` before it went on this list.
CASES: list[Case] = [
    Case(
        id="kingston-flood", title="Design flood of the Thames at Kingston", lat=51.415, lon=-0.308,
        problem="Design flow for a new road bridge over the Thames at Kingston: the 100-year flood with its "
                "uncertainty band.",
        kind="flood_risk", site="Thames at Kingston, London, UK",
        shows="A gauge with 140 years of daily discharge: two frequency fits, the spread between them, the band.",
        intake={"return_period": 100, "decision": "design flow"},
    ),
    Case(
        id="potomac-supply", title="Can the Potomac supply the city in a dry year?", lat=38.95, lon=-77.13,
        problem="Can the Potomac at Little Falls reliably supply 8 m3/s to the city's water works, run of river, "
                "in a dry year?",
        kind="supply_reliability", site="Potomac at Little Falls, Washington DC, USA",
        shows="Supply screening against the flow-duration curve of a 95-year USGS record.",
        intake={"demand_m3s": 8, "use": "municipal"},
    ),
    Case(
        id="taichung-drought", title="Is Taichung in drought?", lat=24.15, lon=120.68,
        problem="Is Taichung in drought now, and how does this dry spell compare with the worst on record? "
                "It matters for the city's water supply.",
        kind="drought_status", site="Taichung, Taiwan",
        shows="SPI and SPEI on a 130-year CWA rain gauge with ERA5 temperature.",
        intake={"timescales": [3, 12], "drought_concern": "water supply"},
    ),
    Case(
        id="cambridge-groundwater", title="Groundwater decline in the Chalk near Cambridge", lat=52.20, lon=0.12,
        problem="Are groundwater levels in the Chalk near Cambridge declining over the last ten years compared "
                "with the full record? Public supply depends on the boreholes.",
        kind="groundwater_decline", site="Chalk boreholes near Cambridge, UK",
        shows="A 48-year borehole record: trend, the SGI and a recharge estimate.",
        intake={"horizon": 10, "concern": "supply"},
    ),
    Case(
        id="bamako-ungauged", title="Flow to expect in the Niger at Bamako", lat=12.6, lon=-8.0,
        problem="No gauge I can reach on the Niger at Bamako: what mean flow and Q95 should a water-supply "
                "offtake expect?",
        kind="ungauged_flow", site="Niger at Bamako, Mali (no gauge in reach)",
        shows="Prediction in an ungauged basin: donors by similarity, transferred signatures, the GloFAS check.",
        intake={"purpose": "water supply", "statistic": "all"},
    ),
    Case(
        id="nairobi-drought", title="Meteorological drought at Nairobi", lat=-1.29, lon=36.82,
        problem="Is Nairobi in a meteorological drought this season, for the smallholder farms around the city? "
                "No rain gauge is in the catalog here.",
        kind="drought_status", site="Nairobi, Kenya (bare point)",
        shows="A bare point: SPI and SPEI on the ERA5 cell, and what that does and does not say.",
        intake={"timescales": [3, 6, 12], "drought_concern": "agriculture"},
    ),
    Case(
        id="sintra-supply", title="A village supply from an ungauged stream near Lisbon", lat=38.80, lon=-9.38,
        problem="A stream in the Sintra hills near Lisbon with no gauge: can it supply a village with 4 ML/day "
                "run of river, and how reliably?",
        kind="supply_reliability", site="Sintra hills near Lisbon, Portugal (no gauge in reach)",
        shows="Ungauged supply: the flow-duration curve transferred from donors, with its band and skill.",
        intake={"demand_ml_day": 4, "use": "municipal"},
    ),
    Case(
        id="wagga-flood", title="Design flood of the Murrumbidgee at Wagga Wagga", lat=-35.10, lon=147.37,
        problem="Design flood for a levee upgrade on the Murrumbidgee at Wagga Wagga: the 100-year flow, and how "
                "far the evidence can be trusted.",
        kind="flood_risk", site="Murrumbidgee at Wagga Wagga, NSW, Australia",
        shows="A BoM gauge the catalog lists but cannot read yet: the regional path and its honest caveats.",
        intake={"return_period": 100, "decision": "design flow"},
    ),
    Case(
        id="little-falls-water-quality", title="Water quality of the Potomac at Little Falls", lat=38.95, lon=-77.13,
        problem="Screen the last five years of water-quality samples of the Potomac at Little Falls against the "
                "drinking-water guidelines: which parameters exceed, and what is the index?",
        kind="water_quality", site="Potomac at Little Falls, Washington DC, USA",
        shows="WQP samples at a USGS station: the WHO screen and the water-quality index over what was sampled.",
        intake={"use": "drinking", "years": 5},
    ),
    Case(
        id="own-table-flood", title="The 50-year flood from your own table", lat=40.2, lon=-8.0,
        problem="Use my attached table of daily flows for a culvert design on an ungauged stream: the 50-year "
                "flood, with two fits and their spread.",
        kind="flood_risk", site="An ungauged stream in central Portugal (your own table)",
        shows="A table you bring: the ingest mapping and QA, then the frequency fits on your annual maxima.",
        intake={"return_period": 50, "decision": "design flow"},
        upload=TABLE_NAME,
    ),
    Case(
        id="toulouse-irrigation", title="Irrigating maize from the Garonne at Toulouse", lat=43.53, lon=1.40,
        problem="Irrigating 40 ha of maize from the Garonne at Portet-sur-Garonne, south of Toulouse, planted in "
                "April: what is the seasonal water demand and can the river meet it run of river?",
        kind="irrigation_feasibility", site="Garonne at Portet-sur-Garonne, Toulouse, France",
        shows="FAO-56 crop demand from the ERA5 climate, then a supply screening on a 116-year Hub'Eau gauge.",
        intake={"crop": "maize", "area_ha": 40, "planting_month": 4},
    ),
    Case(
        id="orleans-drought", title="Hydrological drought of the Loire at Orleans", lat=47.9, lon=1.9,
        problem="Is the Loire at Orleans in hydrological drought, and how does the river's low flow follow the "
                "rainfall deficit?",
        kind="drought_status", site="Loire at Orleans, France",
        shows="Drought propagation: the ERA5 indices, the gauge's low-flow context and the lag between them.",
        intake={"timescales": [3, 12], "drought_concern": "water supply"},
    ),
]


# ── the synthetic table ─────────────────────────────────────────────────────


def synthetic_flows(*, years: int = 30, seed: int = 7, start: str = "1994-01-01") -> Any:
    """Thirty years of daily flow (m3/s) for the "your own table" case: a seasonal baseflow with recession,
    lognormal storm peaks and a small trend. Deterministic, so the recording and the replay see one table."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=int(365.25 * years), freq="D")
    doy = dates.dayofyear.to_numpy()
    season = 6.0 + 4.5 * np.cos(2 * np.pi * (doy - 30) / 365.25)
    storms = rng.random(len(dates)) < (0.06 + 0.05 * np.cos(2 * np.pi * (doy - 15) / 365.25))
    peaks = np.where(storms, rng.lognormal(mean=2.6, sigma=0.75, size=len(dates)), 0.0)
    flow = np.empty(len(dates))
    level = season[0]
    for i in range(len(dates)):
        level = 0.93 * level + 0.07 * season[i] + peaks[i]
        flow[i] = level * (1.0 + 0.08 * rng.standard_normal())
    flow = np.clip(flow, 0.2, None) * (1.0 - 0.003 * np.arange(len(dates)) / 365.25)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "flow_m3s": np.round(flow, 3)})


TABLES: dict[str, Callable[[], Any]] = {TABLE_NAME: synthetic_flows}


# ── what is on disk ──────────────────────────────────────────────────────────


def load_meta(out_dir: str | Path, case_id: str) -> dict[str, Any] | None:
    path = Path(out_dir) / case_id / "meta.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _metas(out_dir: str | Path) -> list[dict[str, Any]]:
    out = Path(out_dir)
    if not out.is_dir():
        return []
    rows = []
    for path in sorted(out.glob("*/meta.json")):
        meta = load_meta(out, path.parent.name)
        if meta and meta.get("id"):
            rows.append(meta)
    return rows


def already_recorded(out_dir: str | Path, *, fresh_for_days: float) -> set[str]:
    """Ids with a recording newer than ``fresh_for_days`` that produced a report or an honest decline, which a
    top-up run leaves alone (a dozen studies cost real money, so a run that fails halfway tops up)."""
    if fresh_for_days <= 0:
        return set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=fresh_for_days)
    fresh: set[str] = set()
    for meta in _metas(out_dir):
        try:
            when = datetime.fromisoformat(str(meta.get("recorded")))
        except (TypeError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when > cutoff and meta.get("status") in ("done", "declined") and not meta.get("error"):
            fresh.add(str(meta["id"]))
    return fresh


def load_index(out_dir: str | Path) -> dict[str, Any]:
    path = Path(out_dir) / "index.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"studies": []}
    return data if isinstance(data, dict) else {"studies": []}


# ── the ledger ───────────────────────────────────────────────────────────────


def usd_for(ledger: dict[str, dict[str, int]], model: str | None, price: tuple[float, float] | None = None) -> float:
    """The estimate for a workspace's ledger at the model's rate (USD per million prompt and completion tokens)."""
    from aquascope.ai_engine.providers import CACHE_READ_PRICES, CACHE_WRITE_MULTIPLIER

    rate = price or PRICES.get(str(model or ""), DEFAULT_PRICE)

    def total(key: str) -> int:
        return sum(int(v.get(key) or 0) for v in ledger.values())

    read_rate = CACHE_READ_PRICES.get(str(model or ""), rate[0] * 0.1) if price is None else rate[0] * 0.1
    usd = (total("prompt_tokens") * rate[0] + total("completion_tokens") * rate[1]
           + total("cache_read_tokens") * read_rate + total("cache_write_tokens") * rate[0] * CACHE_WRITE_MULTIPLIER)
    return round(usd / 1e6, 4)


def headline(text: str | None, *, limit: int = 240) -> str:
    """The first sentence of an answer, without Markdown emphasis, for the index."""
    if not text:
        return ""
    # Emphasis markers go; an underscore inside a word (uk_ea, hubeau_hydrometrie) stays.
    clean = re.sub(r"\*+|`+|^#+\s*|(?<!\w)_+|_+(?!\w)", "", str(text).strip())
    clean = clean.split("\n", 1)[0].strip()
    m = re.search(r"^(.+?[.!?])(?:\s+[A-Z(\"']|$)", clean)
    first = m.group(1) if m else clean
    return first if len(first) <= limit else first[:limit - 3].rstrip() + "..."


# ── the run ──────────────────────────────────────────────────────────────────


def _default_factory(**kwargs: Any) -> Any:
    from aquascope.studio import Studio

    return Studio(**kwargs)


def _drive(studio: Any, problem: str, *, max_questions: int = 3) -> Any:
    """Say the problem, answer any questions with the defaults, approve the plan; the last reply comes back."""
    reply = studio.say(problem)
    rounds = 0
    while reply.kind == "questions" and rounds < max_questions:
        rounds += 1
        reply = studio.say("just go")
    if reply.kind == "plan":
        reply = studio.approve()
    return reply


def _trim(value: Any, notes: list[str], where: str) -> Any:
    """Lists longer than TRIM_TO cut to their head, recursively; every cut is noted."""
    if isinstance(value, dict):
        return {k: _trim(v, notes, f"{where}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > TRIM_TO:
            notes.append(f"{where}: {len(value)} entries kept to {TRIM_TO}")
            value = value[:TRIM_TO]
        return [_trim(v, notes, where) for v in value]
    return value


def _workspace_json(ws_dict: dict[str, Any]) -> tuple[str, list[str]]:
    """The workspace as JSON; when it exceeds MAX_WORKSPACE_BYTES the run's result lists are trimmed."""
    text = json.dumps(ws_dict, ensure_ascii=False, default=str, indent=1)
    if len(text.encode("utf-8")) <= MAX_WORKSPACE_BYTES:
        return text, []
    notes: list[str] = []
    run = ws_dict.get("run") or {}
    for r in run.get("results") or []:
        sid = str(r.get("id") or "?")
        if isinstance(r.get("result"), (dict, list)):
            r["result"] = _trim(r["result"], notes, f"run.results[{sid}].result")
        fb = r.get("fallback")
        if isinstance(fb, dict) and isinstance(fb.get("result"), (dict, list)):
            fb["result"] = _trim(fb["result"], notes, f"run.results[{sid}].fallback.result")
    study = ws_dict.get("study") or {}
    if isinstance(study, dict) and isinstance(study.get("results"), list):
        study["results"] = _trim(study["results"], notes, "study.results")
    text = json.dumps(ws_dict, ensure_ascii=False, default=str, indent=1)
    if len(text.encode("utf-8")) > MAX_WORKSPACE_BYTES:
        notes.append(f"still {len(text.encode('utf-8')):,} bytes after trimming")
    return text, notes


def _write_case(ws: Any, case: Case, case_dir: Path, *, seconds: float, price: tuple[float, float] | None,
                error: str | None) -> dict[str, Any]:
    """The five files of a recording; returns the meta written."""
    case_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = case_dir / "figures"
    if figures_dir.is_dir():
        shutil.rmtree(figures_dir)
    files: list[str] = []

    text, trimmed = _workspace_json(ws.to_dict(with_artifacts=False))
    (case_dir / "workspace.json").write_text(text + "\n", encoding="utf-8")
    files.append("workspace.json")

    report_md = _report_markdown(ws)
    if report_md:
        (case_dir / "report.md").write_text(report_md, encoding="utf-8")
        files.append("report.md")
    study = ws.study_obj()
    if study is not None:
        (case_dir / "study.yaml").write_text(study.to_yaml(), encoding="utf-8")
        files.append("study.yaml")

    figures: list[dict[str, Any]] = []
    for a in ws.figures():
        if a.media_type != "image/png" or not a.data:
            continue
        name = Path(a.name).name
        figures_dir.mkdir(parents=True, exist_ok=True)
        (figures_dir / name).write_bytes(a.data)
        files.append(f"figures/{name}")
        figures.append({"name": name, "id": a.id, "step": a.step, "caption": a.caption, "bytes": len(a.data)})

    run = ws.run or {}
    critique = ws.critique or {}
    checks = critique.get("checks") or []
    ledger = {k: dict(v) for k, v in ws.ledger.items()}
    rate = price or PRICES.get(str(ws.model or ""), DEFAULT_PRICE)
    report = ws.report or {}
    meta = {
        "id": case.id, "title": case.title, "kind": case.kind, "shows": case.shows,
        "site": {"lat": case.lat, "lon": case.lon, "name": case.site},
        "problem": case.problem, "intake": dict(ws.brief.intake or {}), "upload": case.upload,
        "status": ws.status, "declined_reason": ws.declined_reason, "error": error,
        "model": ws.model, "provider": ws.provider,
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(float(seconds), 1),
        "tokens": {"prompt": sum(int(v.get("prompt_tokens") or 0) for v in ledger.values()),
                   "completion": sum(int(v.get("completion_tokens") or 0) for v in ledger.values()),
                   "calls": sum(int(v.get("calls") or 0) for v in ledger.values()), "by_role": ledger},
        "usd": usd_for(ledger, ws.model, rate), "usd_per_mtoken": list(rate),
        "playbook": (ws.study or {}).get("plan", {}).get("playbook") if ws.study else ws.brief.playbook,
        "branch": (ws.study or {}).get("plan", {}).get("branch") if ws.study else None,
        "plan_author": (ws.study or {}).get("plan", {}).get("author") if ws.study else None,
        "steps": len((ws.study or {}).get("steps") or []),
        "gates": {"passed": len(run.get("gates") or []) - len(run.get("failed_gates") or []),
                  "total": len(run.get("gates") or [])},
        "checks": {"passed": sum(1 for c in checks if c.get("passed")), "total": len(checks)},
        "replans": int(run.get("replans") or 0), "stop_reason": run.get("stop_reason"),
        "headline": headline(report.get("answer")) if ws.status == "done" else headline(ws.declined_reason),
        "not_established": list(report.get("not_established") or []),
        "figures": figures, "files": files, "trimmed": trimmed,
        "aquascope_version": __version__,
    }
    meta["tokens"]["total"] = meta["tokens"]["prompt"] + meta["tokens"]["completion"]
    (case_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str) + "\n",
                                        encoding="utf-8")
    return meta


def _report_markdown(ws: Any) -> str:
    """The Markdown report from the deliverables package, or the Author's plain version, or nothing (declined)."""
    if ws.status != "done":
        return ""
    try:
        from aquascope.studio.deliverables.report_md import report_markdown

        return report_markdown(ws)
    except Exception as exc:  # noqa: BLE001 - the plain version is the fallback
        logger.info("report_markdown unavailable (%s); writing the plain report", exc)
    from aquascope.studio.roles.author import to_markdown

    return to_markdown(ws)


def record(
    cases: list[Case] | None = None,
    out_dir: str | Path = "explorer/showcase/studies",
    *,
    provider: str | None = "anthropic",
    model: str | None = "claude-sonnet-5-5",
    api_key: str | None = None,
    base_url: str | None = None,
    max_usd: float = 15.0,
    fresh_for_days: float = 30.0,
    only: list[str] | set[str] | None = None,
    on_event: Callable[[str], None] | None = None,
    studio_factory: Callable[..., Any] | None = None,
    price: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Run each case with the crew and write its recording; returns the meta rows written this run.

    ``only`` re-records those ids whatever their age; otherwise cases fresher than ``fresh_for_days`` are
    skipped. The run stops before a case when the run's spend reaches ``max_usd``. ``studio_factory`` builds
    the Studio (the constructor by default; a fake in the tests). The index is rewritten after every case, so
    a run stopped halfway leaves a consistent set.
    """
    say = on_event or (lambda m: logger.info("%s", m))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    make = studio_factory or _default_factory
    wanted = cases if cases is not None else CASES
    if only:
        chosen = {str(i).strip() for i in only}
        wanted = [c for c in wanted if c.id in chosen]
        fresh: set[str] = set()
    else:
        fresh = already_recorded(out, fresh_for_days=fresh_for_days)
    if fresh:
        say("already recorded and still fresh, skipping: " + ", ".join(sorted(fresh)))
    spent = 0.0
    written: list[dict[str, Any]] = []
    for case in wanted:
        if case.id in fresh:
            continue
        if spent >= max_usd:
            say(f"budget reached ({spent:.2f} USD of {max_usd:.2f}); stopping before {case.id}")
            break
        say(f"recording {case.id}: {case.problem}")
        data = {case.upload: TABLES[case.upload]()} if case.upload and case.upload in TABLES else None
        events: list[str] = []

        def relay(event: dict[str, Any], _events: list[str] = events) -> None:
            line = (f"{event.get('role', '')}{' ' + str(event['step']) if event.get('step') else ''}: "
                    f"{event.get('event', '')} {event.get('detail', '')}").strip()
            _events.append(line)
            say(f"  · {line}")

        started = time.monotonic()
        error: str | None = None
        try:
            studio = make(lat=case.lat, lon=case.lon, provider=provider, model=model, api_key=api_key,
                          base_url=base_url, data=data, intake=dict(case.intake or {}), on_event=relay)
            _drive(studio, case.problem)
        except Exception as exc:  # noqa: BLE001 - one failed case must not lose the rest
            error = f"{type(exc).__name__}: {exc}"
            say(f"  failed: {error}")
            if "studio" not in locals():
                written.append({"id": case.id, "status": "error", "error": error, "usd": 0.0})
                continue
        seconds = time.monotonic() - started
        meta = _write_case(studio.workspace, case, out / case.id, seconds=seconds, price=price, error=error)
        spent += float(meta.get("usd") or 0.0)
        written.append(meta)
        say(f"  {case.id}: {meta['status']}, {meta['steps']} step(s), gates {meta['gates']['passed']}/"
            f"{meta['gates']['total']}, {meta['tokens']['total']:,} tokens, {meta['usd']:.2f} USD, "
            f"{meta['seconds']:.0f} s (run total {spent:.2f} USD)")
        write_index(out)
    return written


# ── the index and the table ──────────────────────────────────────────────────


def write_index(out_dir: str | Path) -> Path:
    """``index.json`` from every ``*/meta.json`` on disk: what a page needs to offer the chips."""
    out = Path(out_dir)
    rows = []
    for meta in _metas(out):
        rows.append({
            "id": meta["id"], "title": meta.get("title"), "kind": meta.get("kind"), "site": meta.get("site"),
            "shows": meta.get("shows"), "status": meta.get("status"), "model": meta.get("model"),
            "provider": meta.get("provider"), "date": str(meta.get("recorded") or "")[:10],
            "recorded": meta.get("recorded"), "usd": meta.get("usd"), "seconds": meta.get("seconds"),
            "steps": meta.get("steps"), "gates": meta.get("gates"), "checks": meta.get("checks"),
            "headline": meta.get("headline"), "files": meta.get("files") or [],
            "figures": [f.get("name") for f in (meta.get("figures") or []) if isinstance(f, dict)],
            "upload": meta.get("upload"),
        })
    index = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aquascope_version": __version__,
        "note": "Recorded studies: the prose was written once by the model named in each entry, on the date "
                "given. The plan re-runs live in the page with no key; the recorded numbers are the run's.",
        "studies": rows,
    }
    path = out / "index.json"
    path.write_text(json.dumps(index, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def diagnose(out_dir: str | Path) -> str:
    """One line per recording (status, steps, gates, tokens, USD, seconds, date) and the totals, as text."""
    metas = _metas(out_dir)
    if not metas:
        return f"no recordings under {out_dir}"
    head = f"{'id':28s} {'status':9s} {'steps':>5s} {'gates':>7s} {'tokens':>8s} {'usd':>6s} {'sec':>5s}  date"
    lines = [head, "-" * len(head)]
    total_usd = 0.0
    total_tokens = 0
    for m in metas:
        gates = m.get("gates") or {}
        tokens = (m.get("tokens") or {}).get("total") or 0
        total_usd += float(m.get("usd") or 0.0)
        total_tokens += int(tokens)
        status = str(m.get("status") or "?")
        if m.get("error"):
            status = "error"
        lines.append(f"{m['id'][:28]:28s} {status:9s} {m.get('steps') or 0:5d} "
                     f"{gates.get('passed', 0):3d}/{gates.get('total', 0):<3d} {int(tokens):8,d} "
                     f"{float(m.get('usd') or 0):6.2f} {float(m.get('seconds') or 0):5.0f}  "
                     f"{str(m.get('recorded') or '')[:10]}")
    lines.append("-" * len(head))
    lines.append(f"{len(metas)} recording(s), {total_tokens:,} tokens, {total_usd:.2f} USD")
    problems = [m for m in metas if m.get("error") or m.get("status") not in ("done", "declined")]
    for m in problems:
        lines.append(f"  {m['id']}: {m.get('error') or m.get('status')}")
    return "\n".join(lines)
