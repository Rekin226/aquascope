"""The Studio workspace: the one object every role of the crew reads and writes.

A study is a conversation that ends in a bundle. Between those two ends the
crew needs a shared, serialisable memory: the brief the Consultant wrote, the
inventory the Scout built, the plan the Methodologist proposed (a version-3
:class:`aquascope.study.Study`), the run and its gates, the Critic's findings,
the report the Author assembled, every artifact as bytes, the messages, the
events and the ledger per role (calls, tokens and, when the model is priced,
USD). That is this module.

It is plain dataclasses with ``to_dict`` / ``from_dict`` (bytes travel as
base64), so the browser can hold it between worker calls, the CLI can write
``workspace.json`` and resume, and the MCP tools can pass it in and out and
stay stateless. Nothing here imports matplotlib, a document library or a
model client: the workspace is data.
"""

from __future__ import annotations

import base64
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

#: The Coordinator's states, in the order a study normally passes through them.
STATES = ("intake", "scouting", "planning", "waiting", "review", "running", "critique", "authoring", "done",
          "declined")

#: Who writes into the workspace. The names are the ones the events and the ledger use.
ROLES = ("consultant", "scout", "methodologist", "analyst", "critic", "author", "coordinator", "runner", "reviewer")

WORKSPACE_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── the brief ───────────────────────────────────────────────────────────────


@dataclass
class Question:
    """One thing the Consultant could not infer and asks the user."""

    id: str
    text: str
    options: list[str] | None = None
    default: Any = None
    answer: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        return cls(id=str(d.get("id") or ""), text=str(d.get("text") or ""),
                   options=list(d["options"]) if d.get("options") else None,
                   default=d.get("default"), answer=d.get("answer"))


@dataclass
class Brief:
    """What the crew is asked to do, structured. The user's words are kept in ``problem``."""

    problem: str = ""
    #: What will be decided with the answer ("size a culvert", "renew an abstraction licence").
    decision: str | None = None
    #: The numbers wanted, in words ("the 100-year flow with a band", "SPI-12 classes since 2000").
    quantities: list[str] = field(default_factory=list)
    period: str | None = None
    horizon: str | None = None
    constraints: list[str] = field(default_factory=list)
    deliverables: list[str] = field(default_factory=lambda: ["report", "workbook", "figures", "notebook"])
    #: The problem kind the registry knows (flood_risk, drought, ...) and the playbook it maps to, when one does.
    kind: str | None = None
    playbook: str | None = None
    #: Every playbook the brief asks for, the primary first: a compound brief names more than one and the tree
    #: composes their branches (#383).
    kinds: list[str] = field(default_factory=list)
    #: The playbook's intake fields, when a playbook applies (return_period, timescales, crop, ...).
    intake: dict[str, Any] = field(default_factory=dict)
    #: What the Consultant assumed rather than asked.
    assumptions: list[str] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    ready: bool = False
    #: Who wrote it: "rules" (keyword rules and intake hints), "device" (an on-device model), "model".
    source: str = "rules"

    @property
    def open_questions(self) -> list[Question]:
        return [q for q in self.questions if q.answer is None]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["questions"] = [q.to_dict() for q in self.questions]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Brief:
        d = dict(d or {})
        qs = [Question.from_dict(q) for q in (d.pop("questions", None) or []) if isinstance(q, dict)]
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        b = cls(**known)
        b.questions = qs
        return b


# ── the inventory ───────────────────────────────────────────────────────────


@dataclass
class Dataset:
    """One row of the data inventory: a gauge, a well, a rain gauge, the ERA5 cell, the catchment, a user's table."""

    id: str
    #: station | upload | reanalysis | catchment | donors | samples
    kind: str
    variable: str | None = None
    source: str | None = None
    station_id: str | None = None
    name: str | None = None
    lat: float | None = None
    lon: float | None = None
    distance_km: float | None = None
    start: str | None = None
    end: str | None = None
    years: float | None = None
    resolution: str | None = None
    n: int | None = None
    #: The ingest QA verdict for an upload, the catalog's notes for a station.
    quality: dict[str, Any] | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Dataset:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        known.setdefault("id", str(d.get("id") or ""))
        known.setdefault("kind", str(d.get("kind") or "station"))
        return cls(**known)


@dataclass
class Inventory:
    """What exists at the site and in the user's hands. ``recon`` is the raw ``assess_site`` dict."""

    site: dict[str, float] = field(default_factory=dict)
    datasets: list[Dataset] = field(default_factory=list)
    recon: dict[str, Any] = field(default_factory=dict)
    catchment: dict[str, Any] | None = None
    donors: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def sufficiency(self) -> list[dict[str, Any]]:
        """The registry's verdict per method at this site, as ``assess_site`` returned it."""
        return list(self.recon.get("sufficiency") or [])

    @property
    def years_by_variable(self) -> dict[str, float]:
        return dict((self.recon.get("context") or {}).get("years_by_variable") or {})

    def dataset(self, dataset_id: str) -> Dataset | None:
        return next((d for d in self.datasets if d.id == dataset_id), None)

    def uploads(self) -> list[Dataset]:
        return [d for d in self.datasets if d.kind == "upload"]

    def to_dict(self) -> dict[str, Any]:
        return {"site": dict(self.site), "datasets": [d.to_dict() for d in self.datasets], "recon": self.recon,
                "catchment": self.catchment, "donors": self.donors, "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> Inventory | None:
        if not d:
            return None
        return cls(site=dict(d.get("site") or {}),
                   datasets=[Dataset.from_dict(x) for x in (d.get("datasets") or []) if isinstance(x, dict)],
                   recon=dict(d.get("recon") or {}), catchment=d.get("catchment"), donors=d.get("donors"),
                   notes=[str(n) for n in (d.get("notes") or [])])


# ── artifacts and messages ──────────────────────────────────────────────────

MEDIA_TYPES = {
    "png": "image/png",
    "svg": "image/svg+xml",
    "csv": "text/csv",
    "md": "text/markdown",
    "html": "text/html",
    "json": "application/json",
    "geojson": "application/geo+json",
    "yaml": "application/yaml",
    "ipynb": "application/x-ipynb+json",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "zip": "application/zip",
    "txt": "text/plain",
}


@dataclass
class Artifact:
    """A file the crew produced: a figure, a table, a document, the workbook, the notebook, the bundle."""

    id: str
    #: figure | table | document | workbook | notebook | study | bundle | data
    kind: str
    #: The path inside the bundle (``figures/s3_frequency_curve.png``).
    name: str
    data: bytes = b""
    media_type: str = "application/octet-stream"
    caption: str | None = None
    step: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.data)

    def to_dict(self, *, with_data: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "name": self.name, "media_type": self.media_type,
                               "caption": self.caption, "step": self.step, "meta": dict(self.meta),
                               "size": self.size}
        if with_data:
            out["data"] = base64.b64encode(self.data).decode("ascii")
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Artifact:
        raw = d.get("data")
        data = base64.b64decode(raw) if isinstance(raw, str) else (raw if isinstance(raw, bytes) else b"")
        return cls(id=str(d.get("id") or ""), kind=str(d.get("kind") or "data"), name=str(d.get("name") or ""),
                   data=data, media_type=str(d.get("media_type") or "application/octet-stream"),
                   caption=d.get("caption"), step=d.get("step"), meta=dict(d.get("meta") or {}))


@dataclass
class Message:
    """One turn of the conversation. ``kind`` tells a face how to show it (text, questions, plan, report)."""

    role: str
    text: str
    at: str = field(default_factory=now)
    kind: str = "text"
    payload: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(role=str(d.get("role") or "coordinator"), text=str(d.get("text") or ""),
                   at=str(d.get("at") or now()), kind=str(d.get("kind") or "text"), payload=d.get("payload"))


# ── the workspace ───────────────────────────────────────────────────────────


@dataclass
class Workspace:
    """Everything the crew shares. See the module docstring."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: str = field(default_factory=now)
    status: str = "intake"
    site: dict[str, float] | None = None
    brief: Brief = field(default_factory=Brief)
    inventory: Inventory | None = None
    #: ``Study.to_dict()`` of the plan (version 3). Rebuild the object with :meth:`study`.
    study: dict[str, Any] | None = None
    #: The run: ``{"ok", "results": [...], "gates": [...], "stopped_at", "stop_reason", "started", "finished"}``.
    run: dict[str, Any] | None = None
    #: The Critic's findings: ``{"checks": [...], "issues": [...], "not_established": [...]}``.
    #: The Interpreter's findings: claims with basis paths, consistency, the decision block, data requests.
    findings: dict[str, Any] | None = None
    #: The data request the study is waiting on (status ``waiting``): what, why, effect, continue_without.
    pending_request: dict[str, Any] | None = None
    critique: dict[str, Any] | None = None
    #: The Author's report: ``{"title", "answer", "key_numbers": [...], "sections": [...], "not_established": [...],
    #: "references": [...], "footer"}``. A section is ``{"id", "title", "text", "figures": [ids], "tables": [ids]}``.
    report: dict[str, Any] | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    #: Calls and tokens per role: ``{"consultant": {"calls", "prompt_tokens", "completion_tokens"}, ...}``, plus
    #: ``cost_usd`` when the model is in the price table (:data:`aquascope.ai_engine.providers.PRICES`).
    ledger: dict[str, dict[str, Any]] = field(default_factory=dict)
    model: str | None = None
    provider: str | None = None
    #: The spend ceiling, once one was reached: ``{"max_usd", "spent_usd", "role", "at"}``. The roles run
    #: keyless from that point; the event with the same words is in ``events``.
    budget: dict[str, Any] | None = None
    #: The user's tables as CSV text by dataset id, so they round-trip with the workspace.
    tables: dict[str, str] = field(default_factory=dict)
    #: Follow-ups after the report: ``{"text", "at", "kind": "question" | "change", "steps": [...]}``.
    follow_ups: list[dict[str, Any]] = field(default_factory=list)
    declined_reason: str | None = None
    version: int = WORKSPACE_VERSION
    #: A face's callback for every event as it happens (the Coordinator sets it); not serialised.
    listener: Any = field(default=None, repr=False, compare=False)

    # ── conversation and events ──

    def say(self, role: str, text: str, *, kind: str = "text", payload: dict[str, Any] | None = None) -> Message:
        m = Message(role=role, text=text, kind=kind, payload=payload)
        self.messages.append(m)
        return m

    def event(self, role: str, event: str, detail: str, *, step: str | None = None) -> dict[str, Any]:
        e = {"role": role, "step": step, "event": event, "detail": detail, "at": now()}
        self.events.append(e)
        if self.listener is not None:
            try:
                self.listener(e)
            except Exception:  # noqa: BLE001 - a face's printing must not stop the study
                pass
        return e

    def set_status(self, status: str) -> None:
        if status not in STATES:
            raise ValueError(f"unknown status {status!r}; one of {STATES}")
        self.status = status
        self.event("coordinator", "status", status)

    # ── artifacts ──

    def add_artifact(self, artifact: Artifact) -> Artifact:
        """Add or replace (by id) an artifact."""
        self.artifacts = [a for a in self.artifacts if a.id != artifact.id]
        self.artifacts.append(artifact)
        return artifact

    def artifact(self, artifact_id: str) -> Artifact | None:
        return next((a for a in self.artifacts if a.id == artifact_id), None)

    def figures(self, step: str | None = None) -> list[Artifact]:
        return [a for a in self.artifacts if a.kind == "figure" and (step is None or a.step == step)]

    def artifacts_of(self, kind: str) -> list[Artifact]:
        return [a for a in self.artifacts if a.kind == kind]

    # ── the plan as an object ──

    def study_obj(self) -> Any:
        """The plan as a :class:`aquascope.study.Study` (None before the Methodologist wrote one)."""
        if not self.study:
            return None
        from aquascope.study import Study

        return Study.from_dict(self.study)

    def set_study(self, study: Any) -> None:
        self.study = study.to_dict() if hasattr(study, "to_dict") else dict(study)

    # ── the user's tables ──

    def frames(self) -> dict[str, Any]:
        """The user's tables as DataFrames, by dataset id (pandas is imported here, not at module import)."""
        import io

        import pandas as pd

        return {k: pd.read_csv(io.StringIO(v)) for k, v in self.tables.items()}

    def add_table(self, dataset_id: str, frame_or_csv: Any) -> str:
        """Keep a user's table (a DataFrame or CSV text) under ``dataset_id``; returns the id."""
        if isinstance(frame_or_csv, str):
            self.tables[dataset_id] = frame_or_csv
        else:
            self.tables[dataset_id] = frame_or_csv.to_csv(index=False)
        return dataset_id

    # ── the ledger ──

    def charge(self, role: str, prompt_tokens: int = 0, completion_tokens: int = 0,
               cost_usd: float | None = None) -> None:
        entry = self.ledger.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        entry["calls"] += 1
        entry["prompt_tokens"] += int(prompt_tokens or 0)
        entry["completion_tokens"] += int(completion_tokens or 0)
        if cost_usd is not None:
            self.charge_usd(role, cost_usd)

    def charge_usd(self, role: str, cost_usd: float) -> None:
        """Add ``cost_usd`` to the role's line (the tokens were counted by the transport)."""
        entry = self.ledger.setdefault(role, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        entry["cost_usd"] = round(float(entry.get("cost_usd") or 0.0) + float(cost_usd), 6)

    @property
    def tokens(self) -> int:
        return sum(v.get("prompt_tokens", 0) + v.get("completion_tokens", 0) for v in self.ledger.values())

    @property
    def total_usd(self) -> float | None:
        """The USD spent across the roles, or None when no line is priced (an unknown model: tokens only)."""
        priced = [float(v["cost_usd"]) for v in self.ledger.values() if v.get("cost_usd") is not None]
        return round(sum(priced), 6) if priced else None

    # ── (de)serialisation ──

    def to_dict(self, *, with_artifacts: bool = True) -> dict[str, Any]:
        """The whole workspace as JSON-able data. ``with_artifacts=False`` keeps the artifact list but drops the bytes
        (a face that only needs to show the state)."""
        return {
            "version": self.version,
            "id": self.id,
            "created": self.created,
            "status": self.status,
            "site": dict(self.site) if self.site else None,
            "brief": self.brief.to_dict(),
            "inventory": self.inventory.to_dict() if self.inventory else None,
            "study": self.study,
            "run": self.run,
            "findings": self.findings,
            "pending_request": self.pending_request,
            "critique": self.critique,
            "report": self.report,
            "artifacts": [a.to_dict(with_data=with_artifacts) for a in self.artifacts],
            "messages": [m.to_dict() for m in self.messages],
            "events": list(self.events),
            "ledger": {k: dict(v) for k, v in self.ledger.items()},
            "model": self.model,
            "provider": self.provider,
            "budget": dict(self.budget) if self.budget else None,
            "tables": dict(self.tables),
            "follow_ups": list(self.follow_ups),
            "declined_reason": self.declined_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Workspace:
        ws = cls(
            id=str(d.get("id") or uuid.uuid4().hex[:12]),
            created=str(d.get("created") or now()),
            status=str(d.get("status") or "intake"),
            site=dict(d["site"]) if d.get("site") else None,
            brief=Brief.from_dict(d.get("brief")),
            inventory=Inventory.from_dict(d.get("inventory")),
            study=dict(d["study"]) if isinstance(d.get("study"), dict) else None,
            run=dict(d["run"]) if isinstance(d.get("run"), dict) else None,
            findings=dict(d["findings"]) if isinstance(d.get("findings"), dict) else None,
            pending_request=dict(d["pending_request"]) if isinstance(d.get("pending_request"), dict) else None,
            critique=dict(d["critique"]) if isinstance(d.get("critique"), dict) else None,
            report=dict(d["report"]) if isinstance(d.get("report"), dict) else None,
            artifacts=[Artifact.from_dict(a) for a in (d.get("artifacts") or []) if isinstance(a, dict)],
            messages=[Message.from_dict(m) for m in (d.get("messages") or []) if isinstance(m, dict)],
            events=[dict(e) for e in (d.get("events") or []) if isinstance(e, dict)],
            ledger={str(k): dict(v) for k, v in (d.get("ledger") or {}).items() if isinstance(v, dict)},
            model=d.get("model"),
            provider=d.get("provider"),
            budget=dict(d["budget"]) if isinstance(d.get("budget"), dict) else None,
            tables={str(k): str(v) for k, v in (d.get("tables") or {}).items()},
            follow_ups=[dict(f) for f in (d.get("follow_ups") or []) if isinstance(f, dict)],
            declined_reason=d.get("declined_reason"),
            version=int(d.get("version") or WORKSPACE_VERSION),
        )
        return ws

    def to_json(self, *, with_artifacts: bool = True, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(with_artifacts=with_artifacts), ensure_ascii=False, default=str, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Workspace:
        return cls.from_dict(json.loads(text))

    # ── a compact view for a model or a face ──

    def summary(self) -> dict[str, Any]:
        """What a face shows in one glance, without the bytes."""
        return {
            "id": self.id, "status": self.status, "site": self.site,
            "brief": {"problem": self.brief.problem, "decision": self.brief.decision, "kind": self.brief.kind,
                      "playbook": self.brief.playbook, "ready": self.brief.ready,
                      "open_questions": [q.to_dict() for q in self.brief.open_questions]},
            "datasets": len(self.inventory.datasets) if self.inventory else 0,
            "steps": len((self.study or {}).get("steps") or []),
            "artifacts": [a.to_dict(with_data=False) for a in self.artifacts],
            "tokens": self.tokens, "usd": self.total_usd, "model": self.model, "provider": self.provider,
            "budget": self.budget, "declined_reason": self.declined_reason,
        }
