"""A study: the steps behind an answer, written down so they can be run again.

An LLM answer is not reproducible. The tools that produced it are: every one is
a named function with JSON arguments. So the Analyst writes down what it ran,
in order, and that file is the reproducible unit:

    aquascope run study.yaml

The model writes the study; the engine runs the study. Re-running it calls the
same tools with the same arguments and writes the same report, with no model in
the loop at all, which is what makes an answer checkable by someone else.

Version 2 (#308) turns the receipt into a plan. A step carries an ``id``, a
``rationale``, the gates it must pass (``expects``, evaluated by
:mod:`aquascope.gates` after the tool ran), a ``fallback`` for when a gate
fails, and ``depends_on``. The study also names the ``problem`` it answers and
the ``plan`` (playbook, branch, rationale, caveats) it follows, and the runner
writes its ``results`` back into the same document, one entry per step, so the
plan and the receipt are one file. Version-1 studies (a list of steps, no
``version``) load and run as before.

The provenance manifest stays: the aquascope version, when it ran, and a hash
of each step's result, so a re-run that drifts is visible rather than silent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aquascope import __version__

__all__ = ["Step", "Study", "StudyRun", "load", "loads", "parse_block_yaml", "run_study", "write_outputs"]

#: Tools a study may name beyond the Analyst's own: the reconnaissance step.
EXTRA_TOOLS = ("assess_site",)


@dataclass
class Step:
    """One tool call: a name, its arguments, and why it is here.

    Version-2 fields: ``id`` (how results and ``depends_on`` refer to it),
    ``rationale`` (why this step, in one sentence), ``expects`` (gates, see
    :mod:`aquascope.gates`), ``fallback`` (``{"step": {...}}`` to run once when
    a gate fails, ``{"branch": "regional"}`` to ask for a replan, or ``"stop"``),
    ``depends_on`` (step ids that must have succeeded first) and ``method``
    (the id in :mod:`aquascope.methods` this step applies, for the registry
    check at plan time).
    """

    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    note: str | None = None
    id: str | None = None
    rationale: str | None = None
    expects: list[dict[str, Any]] = field(default_factory=list)
    fallback: dict[str, Any] | str | None = None
    depends_on: list[str] = field(default_factory=list)
    method: str | None = None

    @property
    def is_v2(self) -> bool:
        return bool(self.id or self.rationale or self.expects or self.fallback or self.depends_on or self.method)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tool": self.tool, "arguments": self.arguments}
        if self.id:
            out["id"] = self.id
        if self.note:
            out["note"] = self.note
        if self.rationale:
            out["rationale"] = self.rationale
        if self.method:
            out["method"] = self.method
        if self.expects:
            out["expects"] = [dict(e) for e in self.expects]
        if self.fallback:
            out["fallback"] = self.fallback
        if self.depends_on:
            out["depends_on"] = list(self.depends_on)
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Step:
        return cls(
            tool=str(d.get("tool") or ""),
            arguments=dict(d.get("arguments") or {}),
            note=d.get("note"),
            id=(str(d["id"]) if d.get("id") is not None else None),
            rationale=d.get("rationale"),
            expects=[dict(e) for e in (d.get("expects") or []) if isinstance(e, dict)],
            fallback=d.get("fallback"),
            depends_on=[str(x) for x in (d.get("depends_on") or [])],
            method=d.get("method"),
        )


@dataclass
class Study:
    """A question and the steps that answer it (version 2: and the plan and the results)."""

    question: str
    steps: list[Step] = field(default_factory=list)
    title: str | None = None
    created: str | None = None
    aquascope_version: str = __version__
    #: What produced it, for honesty: "analyst" (a model wrote it), "playbook" (a tree) or "hand".
    author: str = "analyst"
    model: str | None = None
    version: int = 1
    #: ``{"kind": "flood_risk", "site": {"lat", "lon"}, "params": {...}}``
    problem: dict[str, Any] | None = None
    #: ``{"playbook", "branch", "rationale", "caveats", "citations", ...}``
    plan: dict[str, Any] | None = None
    #: Written by the runner, one entry per step id.
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def is_v2(self) -> bool:
        return self.version >= 2 or bool(self.problem or self.plan or self.results) or any(s.is_v2 for s in self.steps)

    def step_by_id(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "title": self.title or self.question[:80],
            "question": self.question,
            "created": self.created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "aquascope_version": self.aquascope_version,
            "author": self.author,
            "model": self.model,
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.is_v2:
            out = {"version": 2, **out}
            if self.problem:
                out["problem"] = self.problem
            if self.plan:
                out["plan"] = self.plan
            if self.results:
                out["results"] = self.results
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Study:
        steps = [Step.from_dict(s) for s in (data.get("steps") or []) if isinstance(s, dict)]
        version = data.get("version")
        try:
            version = int(version) if version is not None else 1
        except (TypeError, ValueError):
            version = 1
        return cls(
            question=str(data.get("question") or data.get("title") or ""),
            steps=steps,
            title=data.get("title"),
            created=data.get("created"),
            aquascope_version=str(data.get("aquascope_version") or __version__),
            author=str(data.get("author") or "hand"),
            model=data.get("model"),
            version=version,
            problem=dict(data["problem"]) if isinstance(data.get("problem"), dict) else None,
            plan=dict(data["plan"]) if isinstance(data.get("plan"), dict) else None,
            results={str(k): dict(v) for k, v in (data.get("results") or {}).items() if isinstance(v, dict)},
        )

    def to_yaml(self) -> str:
        """Serialise without a YAML dependency: this subset is small and fixed.

        Scalars are JSON (always quoted strings, so colons, hashes and unicode
        are safe); nested values are JSON flow collections, which YAML reads
        and :func:`parse_block_yaml` reads back without PyYAML.
        """
        d = self.to_dict()
        if not self.is_v2:
            return self._v1_yaml(d)
        lines = [
            "# An AquaScope study (version 2): the plan behind an answer, its gates, and what happened.",
            "#   aquascope run study.yaml",
            "version: 2",
            f"title: {_scalar(d['title'])}",
            f"question: {_scalar(d['question'])}",
            f"created: {_scalar(d['created'])}",
            f"aquascope_version: {_scalar(d['aquascope_version'])}",
            f"author: {_scalar(d['author'])}",
        ]
        if d.get("model"):
            lines.append(f"model: {_scalar(d['model'])}")
        for block in ("problem", "plan"):
            if d.get(block):
                lines.append(f"{block}:")
                for k, v in d[block].items():
                    lines.append(f"  {k}: {_scalar(v)}")
        lines.append("steps:")
        for step in d["steps"]:
            lines.append(f"  - tool: {_scalar(step['tool'])}")
            for key in ("id", "note", "rationale", "method"):
                if step.get(key):
                    lines.append(f"    {key}: {_scalar(step[key])}")
            args = step.get("arguments") or {}
            if args:
                lines.append("    arguments:")
                for k, v in args.items():
                    lines.append(f"      {k}: {_scalar(v)}")
            else:
                lines.append("    arguments: {}")
            if step.get("expects"):
                lines.append("    expects:")
                for gate in step["expects"]:
                    lines.append(f"      - {_scalar(gate)}")
            if step.get("fallback"):
                lines.append(f"    fallback: {_scalar(step['fallback'])}")
            if step.get("depends_on"):
                lines.append(f"    depends_on: {_scalar(step['depends_on'])}")
        if d.get("results"):
            lines.append("results:")
            for k, v in d["results"].items():
                lines.append(f"  {k}: {_scalar(v)}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _v1_yaml(d: dict[str, Any]) -> str:
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
    """A YAML scalar for the value types a study carries (str, number, bool, list, dict)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), default=str)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return json.dumps(str(value))          # always quoted: safe for colons, hashes and unicode


@dataclass
class StudyRun:
    """What happened when a study was run."""

    study: Study
    results: list[dict[str, Any]] = field(default_factory=list)
    started: str = ""
    finished: str = ""
    ok: bool = True
    #: Set when a failed gate stopped the run: the step id and the reason.
    stopped_at: str | None = None
    stop_reason: str | None = None
    #: A ``{"branch": ...}`` fallback the runner cannot take on its own (the team can).
    replan: dict[str, Any] | None = None

    @property
    def gates(self) -> list[dict[str, Any]]:
        """Every gate outcome of the run, with its step id."""
        out = []
        for r in self.results:
            for g in r.get("gates") or []:
                out.append({"step": r.get("id"), **g})
            fb = r.get("fallback")
            if isinstance(fb, dict):
                for g in fb.get("gates") or []:
                    out.append({"step": f"{r.get('id')}.fallback", **g})
        return out

    @property
    def failed_gates(self) -> list[dict[str, Any]]:
        return [g for g in self.gates if not g.get("passed")]

    def manifest(self) -> dict[str, Any]:
        return {
            "aquascope_version": __version__,
            "started": self.started,
            "finished": self.finished,
            "ok": self.ok,
            "question": self.study.question,
            "stopped_at": self.stopped_at,
            "stop_reason": self.stop_reason,
            "steps": [
                {
                    "id": r.get("id"),
                    "tool": r["tool"],
                    "arguments": r["arguments"],
                    "ok": r["ok"],
                    "result_sha256": r.get("sha256"),
                    "error": r.get("error"),
                    "gates": r.get("gates") or [],
                    "fallback_used": bool(r.get("fallback_used")),
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [f"# {self.study.title or self.study.question}", "", self.study.question, ""]
        plan = self.study.plan or {}
        if plan:
            head = ", ".join(f"{k} {plan[k]}" for k in ("playbook", "branch") if plan.get(k))
            lines += ["## Plan", ""]
            if head:
                lines.append(f"{head.capitalize()}." if head else "")
            if plan.get("rationale"):
                lines.append(str(plan["rationale"]))
            lines.append("")
        lines.append("## Steps")
        for i, r in enumerate(self.results, 1):
            args = ", ".join(f"{k}={v!r}" for k, v in (r["arguments"] or {}).items())
            state = "ok" if r["ok"] else f"failed: {r.get('error')}"
            label = f"{r['id']}: " if r.get("id") else ""
            lines.append(f"{i}. {label}`{r['tool']}({args})`: {state}")
            if r.get("rationale"):
                lines.append(f"   {r['rationale']}")
            for g in r.get("gates") or []:
                lines.append(f"   - gate {g['check']}: {'passed' if g['passed'] else 'FAILED'}, {g.get('detail', '')}")
            fb = r.get("fallback")
            if r.get("fallback_used") and isinstance(fb, dict):
                fargs = ", ".join(f"{k}={v!r}" for k, v in (fb.get("arguments") or {}).items())
                fstate = "ok" if fb.get("ok") else f"failed: {fb.get('error')}"
                lines.append(f"   - fallback `{fb.get('tool')}({fargs})`: {fstate}")
                for g in fb.get("gates") or []:
                    verdict = "passed" if g["passed"] else "FAILED"
                    lines.append(f"     - gate {g['check']}: {verdict}, {g.get('detail', '')}")
        if self.stop_reason:
            lines += ["", f"**Stopped at {self.stopped_at}:** {self.stop_reason}"]
        for key, title in (("caveats", "Caveats"),):
            items = plan.get(key) or []
            if items:
                lines += ["", f"## {title}", ""] + [f"- {c}" for c in items]
        methods: list[str] = []
        for r in self.results:
            payloads = [r] + ([r["fallback"]] if isinstance(r.get("fallback"), dict) else [])
            for p in payloads:
                for m in (p.get("methods") or []):
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
        if plan.get("citations"):
            lines += ["", "## Playbook citations", ""] + [f"- {c}" for c in plan["citations"]]
        lines += ["", "---", "",
                  f"Produced by aquascope {__version__} from a study file, {self.finished}. "
                  "No model was involved in this run: the steps are the ones recorded in the study."]
        return "\n".join(lines) + "\n"


# ── reading a study ─────────────────────────────────────────────────────────


def load(path: str | Path) -> Study:
    """Read a study from YAML (PyYAML when available, else the subset reader)."""
    text = Path(path).read_text(encoding="utf-8")
    return loads(text)


def loads(text: str) -> Study:
    data = _parse_yaml(text)
    if not isinstance(data, dict):
        raise ValueError("a study file must be a YAML mapping")
    return Study.from_dict(data)


def _parse_yaml(text: str) -> Any:
    try:
        import yaml  # noqa: PLC0415 - optional
    except ImportError:
        return parse_block_yaml(text)
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"not valid YAML: {exc}") from None


# ── a YAML subset reader, so PyYAML stays optional (the browser worker has none) ──


def parse_block_yaml(text: str) -> Any:
    """Read the block-style YAML subset the studies and playbooks are written in.

    Supported: nested mappings and sequences by indentation, ``- key: value``
    items, scalars as JSON (numbers, ``true``/``false``/``null``, quoted
    strings, one-line ``[...]`` and ``{...}`` flow collections), bare strings,
    single-quoted strings, ``|`` and ``>`` block scalars, and ``#`` comments.
    Not supported: anchors, tags, multi-line flow collections, complex keys.
    """
    rows: list[list[Any]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        rows.append([len(raw) - len(raw.lstrip(" ")), raw.strip()])
    if not rows:
        return {}
    value, _ = _parse_node(rows, 0, rows[0][0])
    return value


def _parse_node(rows: list[list[Any]], i: int, indent: int) -> tuple[Any, int]:
    if i >= len(rows):
        return None, i
    if rows[i][1] == "-" or rows[i][1].startswith("- "):
        return _parse_sequence(rows, i, indent)
    return _parse_mapping(rows, i, indent)


def _split_key(content: str) -> tuple[str, str] | None:
    """``key: rest`` or ``key:`` for a bare key; None when the line is not a mapping entry."""
    if content.startswith(("'", '"', "[", "{")):
        return None
    for j, ch in enumerate(content):
        if ch == ":" and (j + 1 == len(content) or content[j + 1] == " "):
            return content[:j].strip(), content[j + 1:].strip()
        if ch in " \t" and ":" not in content[j:]:
            return None
    return None


def _parse_mapping(rows: list[list[Any]], i: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while i < len(rows) and rows[i][0] == indent:
        content = rows[i][1]
        if content == "-" or content.startswith("- "):
            break
        kv = _split_key(content)
        if kv is None:
            raise ValueError(f"cannot read line {content!r} as a mapping entry")
        key, rest = kv
        if rest in ("|", "|-", ">", ">-"):
            value, i = _block_scalar(rows, i + 1, indent, folded=rest.startswith(">"), keep=not rest.endswith("-"))
            out[key] = value
            continue
        if rest == "":
            nxt = i + 1
            deeper = nxt < len(rows) and rows[nxt][0] > indent
            same_level_list = nxt < len(rows) and rows[nxt][0] == indent and rows[nxt][1].startswith("- ")
            if deeper or same_level_list:
                value, i = _parse_node(rows, nxt, rows[nxt][0])
            else:
                value, i = None, nxt
            out[key] = value
            continue
        out[key] = _value(rest)
        i += 1
    return out, i


def _parse_sequence(rows: list[list[Any]], i: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    while i < len(rows) and rows[i][0] == indent and (rows[i][1] == "-" or rows[i][1].startswith("- ")):
        content = rows[i][1]
        rest = content[1:].lstrip()
        if rest == "":
            nxt = i + 1
            if nxt < len(rows) and rows[nxt][0] > indent:
                value, i = _parse_node(rows, nxt, rows[nxt][0])
            else:
                value, i = None, nxt
            out.append(value)
            continue
        if rest in ("|", "|-", ">", ">-"):
            value, i = _block_scalar(rows, i + 1, indent, folded=rest.startswith(">"), keep=not rest.endswith("-"))
            out.append(value)
            continue
        if _split_key(rest) is not None:
            # "- key: value": a mapping whose first line sits after the dash.
            offset = len(content) - len(rest)
            rows[i] = [indent + offset, rest]
            value, i = _parse_mapping(rows, i, indent + offset)
            out.append(value)
            continue
        out.append(_value(rest))
        i += 1
    return out, i


def _block_scalar(rows: list[list[Any]], i: int, indent: int, *, folded: bool, keep: bool) -> tuple[str, int]:
    parts: list[str] = []
    while i < len(rows) and rows[i][0] > indent:
        parts.append(rows[i][1])
        i += 1
    text = (" " if folded else "\n").join(parts)
    return (text + ("\n" if keep and text else "")), i


def _value(token: str) -> Any:
    token = token.strip()
    if token in ("null", "~", ""):
        return None
    if token in ("true", "True"):
        return True
    if token in ("false", "False"):
        return False
    if token.startswith("'") and token.endswith("'") and len(token) >= 2:
        return token[1:-1].replace("''", "'")
    try:
        return json.loads(token)
    except json.JSONDecodeError:
        pass
    if " #" in token and not token.startswith('"'):
        token = token.split(" #", 1)[0].rstrip()
    # A one-line flow collection with bare words: [a b, c] or {k: v, k2: v2}.
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        return [_value(part) for part in _split_flow(inner)] if inner else []
    if token.startswith("{") and token.endswith("}"):
        inner = token[1:-1].strip()
        out: dict[str, Any] = {}
        for part in _split_flow(inner):
            kv = _split_key(part.strip())
            if kv is None:
                raise ValueError(f"cannot read {part!r} as a key: value pair in {token!r}")
            out[kv[0]] = _value(kv[1])
        return out
    return token


def _split_flow(inner: str) -> list[str]:
    """Split a flow collection's body on top-level commas (quotes and nesting respected)."""
    parts: list[str] = []
    buf = ""
    depth = 0
    quote: str | None = None
    for ch in inner:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
            continue
        buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


# ── running a study ─────────────────────────────────────────────────────────


def _workbench_tool(name: str) -> Callable[..., Any]:
    def run(df: Any = None, **params: Any) -> dict[str, Any]:
        from aquascope import workbench

        out = workbench.run(name, df, **params)
        if isinstance(out, dict):
            out.pop("frame", None)
        return out

    return run


def _assess_site_tool(**kwargs: Any) -> dict[str, Any]:
    from aquascope.explore import assess_site

    return assess_site(**kwargs)


def _tools() -> dict[str, Callable[..., Any]]:
    """Every tool a study may name: the Analyst's, the workbench analyses and the reconnaissance."""
    from aquascope import workbench
    from aquascope.ai_engine.analyst import _tool_specs

    tools: dict[str, Callable[..., Any]] = {name: _workbench_tool(name) for name in workbench.TOOLS}
    tools.update({s.name: s.func for s in _tool_specs()})
    tools["assess_site"] = _assess_site_tool
    return tools


def tool_names() -> list[str]:
    """The names a study step may use as ``tool``."""
    return sorted(_tools())


def _frame_from(payload: Any) -> Any:
    """A DataFrame from a tool payload that carries a series (datetime, value) or tidy ``samples`` rows."""
    import pandas as pd

    if not isinstance(payload, dict):
        raise ValueError("the referenced step returned no table")
    if isinstance(payload.get("samples"), list):
        rows = [r for r in payload["samples"] if isinstance(r, dict)]
        if rows:
            return pd.DataFrame(rows)
    if isinstance(payload.get("points"), list):
        rows = [p for p in payload["points"] if isinstance(p, (list, tuple)) and len(p) == 2]
        return pd.DataFrame({"datetime": [r[0] for r in rows], "value": [r[1] for r in rows]})
    series = payload.get("series")
    if isinstance(series, dict) and "t" in series and "v" in series:
        return pd.DataFrame({"datetime": list(series["t"]), "value": list(series["v"])})
    raise ValueError("the referenced step returned no points or series to analyse")


_RESULT_REF = re.compile(r"\{\{\s*result\.([A-Za-z0-9_]+)\.([A-Za-z0-9_.\[\]=-]+)\s*\}\}")


def _resolve_results(args: Any, done: dict[str, dict[str, Any]]) -> Any:
    """Fill ``{{ result.<step>.<path> }}`` in a step's arguments from the payloads of the steps already run.

    A string that is one reference keeps the value's type (a number stays a
    number); a reference to a step that has not run, that failed, or whose
    payload has nothing at the path fails the step with the reason.
    """
    from aquascope.gates import resolve_path

    def lookup(step_id: str, path: str) -> Any:
        rec = done.get(step_id)
        if rec is None:
            raise ValueError(f"result.{step_id}.{path}: step {step_id!r} has not run")
        if not rec.get("ok"):
            raise ValueError(f"result.{step_id}.{path}: step {step_id!r} failed, so its result cannot be used")
        value = resolve_path(rec.get("result"), path)
        if value is None:
            raise ValueError(f"result.{step_id}.{path}: nothing at {path!r} in the result of step {step_id!r}")
        return value

    if isinstance(args, str):
        whole = _RESULT_REF.fullmatch(args.strip())
        if whole:
            return lookup(whole.group(1), whole.group(2))
        return _RESULT_REF.sub(lambda m: str(lookup(m.group(1), m.group(2))), args)
    if isinstance(args, dict):
        return {k: _resolve_results(v, done) for k, v in args.items()}
    if isinstance(args, list):
        return [_resolve_results(v, done) for v in args]
    return args


def _summarise(payload: Any) -> str:
    """One line about a payload, deterministic, for the results block and the timeline."""
    if not isinstance(payload, dict):
        return str(payload)[:120]
    if payload.get("error"):
        return f"error: {str(payload['error'])[:140]}"
    bits = []
    for key in ("source", "station_id", "name", "variable", "unit", "years", "start", "end", "k", "n_donors",
                "n_returned", "sub_basin", "current", "value_mm_per_year", "method"):
        v = payload.get(key)
        if isinstance(v, (str, int, float)) and not isinstance(v, bool):
            bits.append(f"{key}={v}")
    if not bits:
        bits = [f"{k}={v}" for k, v in payload.items() if isinstance(v, (str, int, float))][:5]
    return ", ".join(bits)[:200]


def _run_tool(func: Callable[..., Any], args: dict[str, Any]) -> tuple[Any, bool, str | None]:
    try:
        payload = func(**args)
    except Exception as exc:  # noqa: BLE001 - a failed step is a result, not a crash
        return None, False, f"{type(exc).__name__}: {exc}"
    ok = not (isinstance(payload, dict) and payload.get("error"))
    error = (payload or {}).get("error") if isinstance(payload, dict) else None
    return payload, ok, (str(error) if error else None)


def _record(step: Step, step_id: str, payload: Any, ok: bool, error: str | None, gates: list[dict[str, Any]]) -> dict:
    blob = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True) if payload is not None else ""
    return {
        "id": step_id, "tool": step.tool, "arguments": step.arguments, "ok": ok,
        "rationale": step.rationale,
        "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16] if blob else None,
        "methods": (payload or {}).get("methods") if isinstance(payload, dict) else None,
        "result": payload,
        "error": error,
        "gates": gates,
        "gates_passed": all(g["passed"] for g in gates),
        "fallback_used": False,
    }


def run_study(
    study: Study | str | Path,
    *,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    prior: StudyRun | None = None,
    tools: dict[str, Callable[..., Any]] | None = None,
) -> StudyRun:
    """Run every step in order, evaluate its gates, and collect the results.

    No model, no network beyond the tools'. ``on_event`` receives dicts
    ``{"role", "step", "event", "detail"}`` as the run goes. ``prior`` is an
    earlier run of the same study whose successful, gate-passing steps are
    reused rather than fetched again (the team's replan uses it). ``tools``
    adds to or replaces the registry's tools by name: the browser worker,
    where BasinATLAS cannot be read, serves ``describe_catchment`` from what
    the page already holds.
    """
    from aquascope.gates import evaluate

    if not isinstance(study, Study):
        study = load(study)
    say = on_event or (lambda _m: None)
    tools = {**_tools(), **(tools or {})}
    run = StudyRun(study=study, started=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    done: dict[str, dict[str, Any]] = {}
    # A version-1 study stays a version-1 file: results are written back only into a plan.
    write_back = study.is_v2
    reusable = {r["id"]: r for r in (prior.results if prior else []) if r.get("ok") and r.get("gates_passed")}

    for i, step in enumerate(study.steps, 1):
        step_id = step.id or f"s{i}"
        args_text = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())

        missing = [d for d in step.depends_on if d not in done or not done[d].get("ok")]
        if missing:
            detail = f"depends on {', '.join(missing)}, which did not succeed"
            say({"role": "runner", "step": step_id, "event": "skipped", "detail": detail})
            rec = _record(step, step_id, None, False, detail, [])
            rec["skipped"] = True
            run.results.append(rec)
            run.ok = False
            done[step_id] = rec
            continue

        old = reusable.get(step_id)
        if old is not None and old["tool"] == step.tool and old["arguments"] == step.arguments:
            say({"role": "runner", "step": step_id, "event": "reused", "detail": f"{step.tool}({args_text})"})
            rec = dict(old)
            run.results.append(rec)
            done[step_id] = rec
            if write_back:
                study.results[step_id] = _result_entry(rec)
            continue

        func = tools.get(step.tool)
        say({"role": "runner", "step": step_id, "event": "start", "detail": f"{step.tool}({args_text})"})
        if func is None:
            error = f"unknown tool {step.tool!r}"
            say({"role": "runner", "step": step_id, "event": "error", "detail": error})
            rec = _record(step, step_id, None, False, error, [])
            run.results.append(rec)
            run.ok = False
            done[step_id] = rec
            if write_back:
                study.results[step_id] = _result_entry(rec)
            continue

        args = dict(step.arguments)
        src = args.pop("from_step", None)
        try:
            args = _resolve_results(args, done)
        except ValueError as exc:
            args = None  # type: ignore[assignment]
            payload, ok, error = None, False, str(exc)
        if args is not None and src is not None:
            try:
                args["df"] = _frame_from((done.get(str(src)) or {}).get("result"))
            except Exception as exc:  # noqa: BLE001
                args = None  # type: ignore[assignment]
                payload, ok, error = None, False, f"from_step {src}: {exc}"
        if args is not None:
            payload, ok, error = _run_tool(func, args)
        gates = evaluate(step.expects, payload)
        rec = _record(step, step_id, payload, ok, error, gates)
        say({"role": "runner", "step": step_id, "event": "done" if ok else "error",
             "detail": error or _summarise(payload)})
        for g in gates:
            say({"role": "reviewer", "step": step_id, "event": "gate",
                 "detail": f"{g['check']}: {'passed' if g['passed'] else 'FAILED'}, {g['detail']}"})
        run.ok = run.ok and ok

        failed = [g for g in gates if not g["passed"]]
        if failed:
            reason = "; ".join(f"{g['check']} ({g['detail']})" for g in failed)
            fb = step.fallback
            if isinstance(fb, dict) and isinstance(fb.get("step"), dict):
                fstep = Step.from_dict(fb["step"])
                fid = f"{step_id}.fallback"
                fargs = ", ".join(f"{k}={v!r}" for k, v in fstep.arguments.items())
                say({"role": "runner", "step": fid, "event": "fallback",
                     "detail": f"gate failed ({reason}); running {fstep.tool}({fargs})"})
                ffunc = tools.get(fstep.tool)
                if ffunc is None:
                    fpayload, fok, ferror = None, False, f"unknown tool {fstep.tool!r}"
                else:
                    fpayload, fok, ferror = _run_tool(ffunc, dict(fstep.arguments))
                fgates = evaluate(fstep.expects, fpayload)
                frec = _record(fstep, fid, fpayload, fok, ferror, fgates)
                say({"role": "runner", "step": fid, "event": "done" if fok else "error",
                     "detail": ferror or _summarise(fpayload)})
                for g in fgates:
                    say({"role": "reviewer", "step": fid, "event": "gate",
                         "detail": f"{g['check']}: {'passed' if g['passed'] else 'FAILED'}, {g['detail']}"})
                rec["fallback_used"] = True
                rec["fallback"] = frec
                if not fok or not frec["gates_passed"]:
                    run.stopped_at, run.stop_reason = step_id, (
                        f"gate failed: {reason}; the fallback {fstep.tool} "
                        + ("failed too: " + (ferror or "") if not fok else "did not pass its own gates")
                    )
            elif isinstance(fb, dict) and fb.get("branch"):
                run.replan = {"step": step_id, "branch": str(fb["branch"]), "reason": reason}
                run.stopped_at, run.stop_reason = step_id, (
                    f"gate failed: {reason}; the plan asks for a replan on branch {fb['branch']!r}"
                )
            else:
                run.stopped_at, run.stop_reason = step_id, f"gate failed: {reason}"
        run.results.append(rec)
        done[step_id] = rec
        if write_back:
            study.results[step_id] = _result_entry(rec)
        if run.stop_reason:
            run.ok = False
            say({"role": "runner", "step": step_id, "event": "stop", "detail": run.stop_reason})
            break
    run.finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if run.stop_reason is None and not run.results and study.steps:
        run.ok = False
    return run


def _result_entry(rec: dict[str, Any]) -> dict[str, Any]:
    """The compact per-step entry written back into the study (no payloads)."""
    out: dict[str, Any] = {
        "ok": bool(rec.get("ok")),
        "gates": [{k: g[k] for k in ("check", "passed", "detail") if k in g} for g in (rec.get("gates") or [])],
        "summary": _summarise(rec.get("result")) if rec.get("result") is not None else (rec.get("error") or ""),
        "fallback_used": bool(rec.get("fallback_used")),
    }
    if rec.get("sha256"):
        out["sha256"] = rec["sha256"]
    if rec.get("error"):
        out["error"] = rec["error"]
    fb = rec.get("fallback")
    if isinstance(fb, dict):
        out["fallback"] = {
            "tool": fb.get("tool"), "arguments": fb.get("arguments"), "ok": bool(fb.get("ok")),
            "gates": [{k: g[k] for k in ("check", "passed", "detail") if k in g} for g in (fb.get("gates") or [])],
            "summary": _summarise(fb.get("result")) if fb.get("result") is not None else (fb.get("error") or ""),
        }
    return out


def write_outputs(run: StudyRun, out_dir: str | Path) -> dict[str, str]:
    """Write ``report.md``, ``manifest.json``, ``results.json`` (and ``study.yaml`` for v2) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(run.to_markdown(), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(run.manifest(), indent=2) + "\n", encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps([{k: v for k, v in r.items() if k != "methods"} for r in run.results],
                   indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    paths = {name: str(out / name) for name in ("report.md", "manifest.json", "results.json")}
    if run.study.is_v2:
        (out / "study.yaml").write_text(run.study.to_yaml(), encoding="utf-8")
        paths["study.yaml"] = str(out / "study.yaml")
    return paths


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
