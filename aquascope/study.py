"""A study: the steps behind an answer, written down so they can be run again.

An LLM answer is not reproducible. The tools that produced it are: every one is
a named function with JSON arguments. So the Analyst writes down what it ran,
in order, and that file is the reproducible unit:

    aquascope run study.yaml

The model writes the study; the engine runs the study. Re-running it calls the
same tools with the same arguments and writes the same report, with no model in
the loop at all, which is what makes an answer checkable by someone else.

This is the declarative runner of #54, with a provenance manifest: the aquascope
version, when it ran, and a hash of each step's result, so a re-run that drifts
is visible rather than silent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aquascope import __version__

__all__ = ["Study", "Step", "StudyRun", "load", "run_study"]


@dataclass
class Step:
    """One tool call: a name, its arguments, and why it is here."""

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tool": self.tool, "arguments": self.arguments}
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class Study:
    """A question and the steps that answer it."""

    question: str
    steps: list[Step] = field(default_factory=list)
    title: str | None = None
    created: str | None = None
    aquascope_version: str = __version__
    #: What produced it, for honesty: "analyst" (a model wrote it) or "hand".
    author: str = "analyst"
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title or self.question[:80],
            "question": self.question,
            "created": self.created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "aquascope_version": self.aquascope_version,
            "author": self.author,
            "model": self.model,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_yaml(self) -> str:
        """Serialise without a YAML dependency: this subset is small and fixed."""
        d = self.to_dict()
        lines = [
            "# An AquaScope study: the steps behind an answer, so it can be run again.",
            "#   aquascope run study.yaml",
            f"title: {_scalar(d['title'])}",
            f"question: {_scalar(d['question'])}",
            f"created: {_scalar(d['created'])}",
            f"aquascope_version: {_scalar(d['aquascope_version'])}",
            f"author: {_scalar(d['author'])}",
        ]
        if d.get("model"):
            lines.append(f"model: {_scalar(d['model'])}")
        lines.append("steps:")
        for step in d["steps"]:
            lines.append(f"  - tool: {_scalar(step['tool'])}")
            if step.get("note"):
                lines.append(f"    note: {_scalar(step['note'])}")
            args = step.get("arguments") or {}
            if args:
                lines.append("    arguments:")
                for k, v in args.items():
                    lines.append(f"      {k}: {_scalar(v)}")
            else:
                lines.append("    arguments: {}")
        return "\n".join(lines) + "\n"


def _scalar(value: Any) -> str:
    """A YAML scalar for the value types a study carries (str, number, bool, list)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, dict):
        return json.dumps(value)
    return json.dumps(str(value))          # always quoted: safe for colons, hashes and unicode


@dataclass
class StudyRun:
    """What happened when a study was run."""

    study: Study
    results: list[dict[str, Any]] = field(default_factory=list)
    started: str = ""
    finished: str = ""
    ok: bool = True

    def manifest(self) -> dict[str, Any]:
        return {
            "aquascope_version": __version__,
            "started": self.started,
            "finished": self.finished,
            "ok": self.ok,
            "question": self.study.question,
            "steps": [
                {
                    "tool": r["tool"],
                    "arguments": r["arguments"],
                    "ok": r["ok"],
                    "result_sha256": r.get("sha256"),
                    "error": r.get("error"),
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.study.title or self.study.question}", "", self.study.question, ""]
        lines.append("## Steps")
        for i, r in enumerate(self.results, 1):
            args = ", ".join(f"{k}={v!r}" for k, v in (r["arguments"] or {}).items())
            state = "ok" if r["ok"] else f"failed: {r.get('error')}"
            lines.append(f"{i}. `{r['tool']}({args})`: {state}")
        methods: list[str] = []
        for r in self.results:
            for m in (r.get("methods") or []):
                # A tool may hand back method dicts or already-formatted strings.
                if isinstance(m, dict):
                    line = f"**{m.get('name')}.** {m.get('text')} {m.get('citation', '')}".strip()
                else:
                    line = str(m).strip()
                if line and line not in methods:
                    methods.append(line)
        if methods:
            lines += ["", "## Methods and citations", ""]
            lines += [f"{i}. {m}" for i, m in enumerate(methods, 1)]
        lines += ["", "---", "",
                  f"Produced by aquascope {__version__} from a study file, {self.finished}. "
                  "No model was involved in this run: the steps are the ones recorded in the study."]
        return "\n".join(lines) + "\n"


# ── reading a study ─────────────────────────────────────────────────────────


def load(path: str | Path) -> Study:
    """Read a study from YAML (PyYAML when available, else the subset writer's own format)."""
    text = Path(path).read_text(encoding="utf-8")
    return loads(text)


def loads(text: str) -> Study:
    data = _parse_yaml(text)
    steps = [
        Step(tool=str(s.get("tool")), arguments=dict(s.get("arguments") or {}), note=s.get("note"))
        for s in (data.get("steps") or [])
    ]
    return Study(
        question=str(data.get("question") or data.get("title") or ""),
        steps=steps,
        title=data.get("title"),
        created=data.get("created"),
        aquascope_version=str(data.get("aquascope_version") or __version__),
        author=str(data.get("author") or "hand"),
        model=data.get("model"),
    )


def _parse_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml  # noqa: PLC0415 - optional
    except ImportError:
        return _parse_subset(text)
    return yaml.safe_load(text) or {}


def _parse_subset(text: str) -> dict[str, Any]:
    """A tiny reader for the shape :meth:`Study.to_yaml` writes, so PyYAML stays optional."""
    out: dict[str, Any] = {"steps": []}
    step: dict[str, Any] | None = None
    in_args = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0:
            in_args = False
            if stripped == "steps:":
                continue
            key, _, value = stripped.partition(":")
            out[key.strip()] = _value(value.strip())
        elif stripped.startswith("- tool:"):
            step = {"tool": _value(stripped.split(":", 1)[1].strip()), "arguments": {}}
            out["steps"].append(step)
            in_args = False
        elif step is not None and stripped.startswith("note:"):
            step["note"] = _value(stripped.split(":", 1)[1].strip())
        elif step is not None and stripped.startswith("arguments:"):
            rest = stripped.split(":", 1)[1].strip()
            in_args = rest in ("", "{}") and rest != "{}"
            if rest == "{}":
                step["arguments"] = {}
        elif step is not None and in_args:
            key, _, value = stripped.partition(":")
            step["arguments"][key.strip()] = _value(value.strip())
    return out


def _value(token: str) -> Any:
    token = token.strip()
    if token in ("null", "~", ""):
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    try:
        return json.loads(token)
    except json.JSONDecodeError:
        return token


# ── running a study ─────────────────────────────────────────────────────────


def _tools() -> dict[str, Any]:
    from aquascope.ai_engine.analyst import _tool_specs

    return {s.name: s.func for s in _tool_specs()}


def run_study(study: Study | str | Path, *, on_event: Any = None) -> StudyRun:
    """Run every step in order and collect the results. No model, no network beyond the tools'."""
    if not isinstance(study, Study):
        study = load(study)
    say = on_event or (lambda _m: None)
    tools = _tools()
    run = StudyRun(study=study, started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    for step in study.steps:
        func = tools.get(step.tool)
        say(f"step {step.tool}({', '.join(f'{k}={v!r}' for k, v in step.arguments.items())})")
        if func is None:
            run.results.append({"tool": step.tool, "arguments": step.arguments, "ok": False,
                                "error": f"unknown tool {step.tool!r}"})
            run.ok = False
            continue
        try:
            payload = func(**step.arguments)
            ok = not (isinstance(payload, dict) and payload.get("error"))
            blob = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
            run.results.append({
                "tool": step.tool, "arguments": step.arguments, "ok": ok,
                "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
                "methods": (payload or {}).get("methods") if isinstance(payload, dict) else None,
                "result": payload,
                "error": (payload or {}).get("error") if isinstance(payload, dict) else None,
            })
            run.ok = run.ok and ok
        except Exception as exc:  # noqa: BLE001 - a failed step is a result, not a crash
            run.results.append({"tool": step.tool, "arguments": step.arguments, "ok": False,
                                "error": f"{type(exc).__name__}: {exc}"})
            run.ok = False
    run.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return run


def write_outputs(run: StudyRun, out_dir: str | Path) -> dict[str, str]:
    """Write ``report.md``, ``manifest.json`` and ``results.json`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(run.to_markdown(), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(run.manifest(), indent=2) + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "methods"} for r in run.results],
                   indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return {name: str(out / name) for name in ("report.md", "manifest.json", "results.json")}


def study_from_calls(question: str, calls: list[Any], *, model: str | None = None) -> Study:
    """Build a study from an AskResult's tool calls (what the model actually ran)."""
    steps = [
        Step(tool=c.name, arguments=dict(c.arguments or {}))
        for c in calls
        if getattr(c, "ok", True) and c.name not in {"describe_methods", "list_sources", "list_analyses"}
    ]
    return Study(question=question, steps=steps, author="analyst", model=model,
                 created=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def asdict_study(study: Study) -> dict[str, Any]:
    return asdict(study)
