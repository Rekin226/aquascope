"""HydroGym Phase 1: play an agent on the tasks and score it against the playbooks' own keys (#175).

Three agents, in order of how much they know about the scaffold:

* ``tree``: the playbook alone, no model, on the task's reconnaissance
  snapshot. It is the key's own baseline and scores 100 percent by
  construction; it is there to prove the harness and to time it.
* ``team``: :func:`aquascope.ai_engine.team.solve`, the plan-first Analyst,
  given the problem text, the coordinates and the intake, with the review
  auto-approved and the task's reconnaissance snapshot so the key and the
  run see the same catalog. Keyless when no model is named; with a model the
  Coordinator, the Specialist and the Narrator use it.
* ``ask``: the older tool loop (:func:`aquascope.ai_engine.analyst.ask`) given
  only the problem text and the coordinates. It has no playbook, so its
  branch is inferred from the tools it called, and a refusal is read off the
  answer (a heuristic, said so in the docs).

Per task: ``branch_match``, ``gates_respected`` (the fraction of the expected
gates the run evaluated), ``tools_matched``, ``declined_correctly`` on the
unsolvable tasks, ``answer_present``, tokens (per role for the team), seconds
and the error if any. Results go to JSONL as each task finishes, so an
interrupted run keeps what it did; :func:`leaderboard` aggregates any number
of result files into a Markdown table with a cost estimate from a small price
table (:data:`PRICES_USD_PER_MTOK`, list prices that change).

Tool calls in a bench run fetch from the agencies like any other run: that is
the point. A per-task timeout keeps one slow agency from stalling the suite.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aquascope.gym.tasks import Task, read_tasks

logger = logging.getLogger(__name__)

__all__ = [
    "AGENTS",
    "PRICES_USD_PER_MTOK",
    "Result",
    "estimate_cost",
    "infer_branch",
    "leaderboard",
    "load_results",
    "read_refusal",
    "rescore_ask",
    "run_bench",
    "select_tasks",
    "summarize",
]

AGENTS = ("tree", "team", "ask")
DEFAULT_TIMEOUT = 900.0
#: The ``ask`` agent's conversation budget in characters (about 10k tokens) and step cap; both bound spend.
ASK_CONTEXT_CHARS = 40_000
ASK_MAX_STEPS = 8

#: USD per million tokens (input, output), Anthropic's list prices as published in mid-2026. Prices change and
#: other providers are not listed; a model absent here gets no cost estimate rather than a guess. Cache reads and
#: batch discounts are not modelled.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
PRICES_NOTE = ("Cost is estimated from the tokens the provider reported and a small table of list prices "
               "(aquascope.gym.bench.PRICES_USD_PER_MTOK, mid-2026); prices change, cache and batch discounts are "
               "not modelled, and a model not in the table gets no estimate.")

#: How the ``ask`` agent's refusal is read off its answer. A heuristic: the loop has no decline verdict of its
#: own. Two lists. An *explicit* refusal ("out of scope", "cannot answer", "I decline") in the opening of an answer
#: is a refusal of the question; the same words at the end of an answer that quoted its numbers are a caveat, so on
#: a solvable task only the opening counts. A *status* phrase ("not defensible", "too short to", "insufficient
#: record") is what a good answer says about the method it did not use before giving another estimate, so on a
#: solvable task it counts only when the loop called no tool of any branch. On an unsolvable task a refusal of
#: either kind anywhere is the right behaviour: the playbook's own decline for "why is the well falling" is to
#: report the trend and refuse the cause. The bare stem "declin" is not a refusal: on the groundwater decline
#: playbook "not a decline" is the finding.
_REFUSAL_EXPLICIT = re.compile(
    r"(out of scope|outside (?:of )?(?:what|the scope)|"
    r"cannot (?:be )?(?:answer|estimat|attribut|map|determin|provid|quot|give|say|tell|confirm|verif|establish|"
    r"conclud|rule out)|"
    r"can(?:no|')t (?:answer|estimate|attribute|map|determine|provide|quote|give|say|tell|confirm|verify|establish|"
    r"conclude|rule out)|"
    r"unable to|hydraulic model|do(?:es)? not (?:run|produce|cover|map|report|carry|include|hold)|"
    r"\bI (?:must |have to |will )?decline\b|\bdeclin(?:e|ed|ing) to\b|\brefus)",
    re.I,
)
_REFUSAL_STATUS = re.compile(
    r"(not (?:possible|able|defensible|feasible)|would be an extrapolation|beyond (?:about )?three times|"
    r"insufficient (?:data|record)|too short (?:a|to|for)|not enough (?:data|years|record)|"
    r"no (?:usable|suitable) (?:gauge|record|data))",
    re.I,
)
#: Both lists in one, for a reading that does not distinguish them (an unsolvable task).
_REFUSAL = re.compile(f"{_REFUSAL_EXPLICIT.pattern}|{_REFUSAL_STATUS.pattern}", re.I)
#: The opening of an answer, where a refusal of the question is stated (a caveat comes after the numbers).
OPENING_CHARS = 600
#: How much of an answer a result keeps.
ANSWER_CHARS = 6_000
_OUT_OF_STEPS = "I ran out of tool-call steps"


# ── the result ───────────────────────────────────────────────────────────────


@dataclass
class Result:
    """One agent on one task, scored against the task's key."""

    task_id: str
    playbook: str
    split: str
    unsolvable: bool
    agent: str
    model: str | None
    provider: str | None
    branch_expected: str | None
    branch_chosen: str | None = None
    playbook_chosen: str | None = None
    branch_match: bool = False
    gates_expected: int = 0
    gates_evaluated: int = 0
    gates_passed: int = 0
    #: Fraction of the expected (step, check) gates the run evaluated (0 for an agent without gates); None when
    #: the key expects none.
    gates_respected: float | None = None
    tools_expected: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    #: Fraction of the expected tools the agent called; None when the key expects none.
    tools_matched: float | None = None
    declined: bool = False
    #: On an unsolvable task: did the agent decline. None on a solvable task (see ``declined`` there).
    declined_correctly: bool | None = None
    declined_reason: str | None = None
    answer_present: bool = False
    answer: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    seconds: float = 0.0
    error: str | None = None
    finished: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def correct(self) -> bool:
        """Declined an unsolvable task, or picked the expected branch on a solvable one."""
        if self.error:
            return False
        if self.unsolvable:
            return bool(self.declined_correctly)
        return self.branch_match and not self.declined

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["correct"] = self.correct
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Result:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def estimate_cost(model: str | None, prompt_tokens: int, completion_tokens: int) -> float | None:
    """USD from the price table, or None when the model is not listed or nothing was spent."""
    if not model or model not in PRICES_USD_PER_MTOK:
        return None if (prompt_tokens or completion_tokens) else 0.0
    p_in, p_out = PRICES_USD_PER_MTOK[model]
    return round((prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000, 6)


# ── agents ───────────────────────────────────────────────────────────────────


@dataclass
class _Config:
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    client: Any | None = None
    max_steps: int = ASK_MAX_STEPS
    context_chars: int | None = ASK_CONTEXT_CHARS

    @property
    def wants_model(self) -> bool:
        return self.client is not None or any((self.provider, self.model, self.api_key, self.base_url))


def _run_tree(task: Task, cfg: _Config) -> dict[str, Any]:
    from aquascope import playbooks as pbk

    try:
        study = pbk.plan(task.playbook, task.recon, task.intake, problem_text=task.problem)
    except pbk.Declined as exc:
        return {"playbook": task.playbook, "declined": True, "declined_reason": exc.reason, "answer": exc.reason,
                "detail": {"kind": exc.kind}}
    plan = study.plan or {}
    return {
        "playbook": task.playbook, "branch": plan.get("branch"),
        "gates": [{"step": s.id, "check": g.get("check"), "passed": None} for s in study.steps
                  for g in s.expects if isinstance(g, dict)],
        "tools": [s.tool for s in study.steps], "answer": str(plan.get("rationale") or ""), "declined": False,
        "detail": {"notes": plan.get("notes") or []},
    }


def _run_team(task: Task, cfg: _Config) -> dict[str, Any]:
    from aquascope.ai_engine.team import solve

    res = solve(task.problem, lat=task.lat, lon=task.lon, intake=dict(task.intake), recon=task.recon,
                provider=cfg.provider, model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url,
                client=cfg.client, review=lambda study: study, max_replans=1)
    plan = res.study.plan or {}
    tools: list[str] = []
    if res.run is not None:
        for r in res.run.results:
            if not r.get("skipped"):
                tools.append(str(r["tool"]))
            fb = r.get("fallback")
            if r.get("fallback_used") and isinstance(fb, dict) and fb.get("tool"):
                tools.append(str(fb["tool"]))
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for entry in res.cost.values():
        for k in usage:
            usage[k] += int(entry.get(k, 0) or 0)
    return {
        "playbook": plan.get("playbook"), "branch": plan.get("branch"),
        "gates": [{"step": g.get("step"), "check": g.get("check"), "passed": g.get("passed")} for g in res.gates],
        "tools": tools, "answer": res.answer, "declined": res.declined, "declined_reason": res.declined_reason,
        "usage": usage, "model": res.model, "provider": res.provider,
        "detail": {
            "cost_by_role": res.cost, "ok": res.ok,
            "stop_reason": res.run.stop_reason if res.run else None,
            "not_established": res.not_established[:8],
            "checks_failed": [c.get("name") for c in res.checks if not c.get("passed")],
            "events": [f"{e.get('role')}:{e.get('event')}" for e in res.timeline][:40],
            "replans": plan.get("replans"),
        },
    }


def _run_ask(task: Task, cfg: _Config) -> dict[str, Any]:
    from aquascope.ai_engine.analyst import ask

    question = f"{task.problem} The point is at latitude {task.lat:.4f}, longitude {task.lon:.4f}."
    res = ask(question, provider=cfg.provider, model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url,
              client=cfg.client, max_steps=cfg.max_steps, context_chars=cfg.context_chars)
    tools = [c.name for c in res.tool_calls]
    branch = infer_branch(task.playbook, tools)
    text = res.answer or ""
    out_of_steps = text.startswith(_OUT_OF_STEPS)
    declined, reason = read_refusal(text, unsolvable=task.unsolvable, branch=branch)
    return {
        "playbook": task.playbook, "branch": branch, "gates": [], "tools": tools,
        "answer": "" if out_of_steps else text, "declined": declined, "declined_reason": reason,
        "usage": dict(res.usage), "model": res.model, "provider": res.provider,
        "detail": {"steps": res.steps, "verified": res.verified, "out_of_steps": out_of_steps,
                   "checks_failed": [c.get("name") for c in res.checks if not c.get("passed")],
                   "tool_calls": [{"name": c.name, "ok": c.ok} for c in res.tool_calls]},
    }


_AGENT_FUNCS: dict[str, Callable[[Task, _Config], dict[str, Any]]] = {
    "tree": _run_tree, "team": _run_team, "ask": _run_ask,
}


def read_refusal(text: str, *, unsolvable: bool, branch: str | None) -> tuple[bool, str | None]:
    """Whether an answer without a decline verdict of its own refused, and why (see ``_REFUSAL_EXPLICIT``).

    On an unsolvable task any refusal wording anywhere counts. On a solvable
    one an explicit refusal counts in the opening of the answer, and a
    method-status phrase (or a refusal after the numbers) only when
    ``branch`` is None, that is when no tool of any branch was called.
    """
    if unsolvable:
        m = _REFUSAL.search(text)
        return (True, f"refusal wording in the answer: {m.group(0)!r}") if m else (False, None)
    m = _REFUSAL_EXPLICIT.search(text[:OPENING_CHARS])
    if m:
        return True, f"refusal wording in the opening of the answer: {m.group(0)!r}"
    if branch is None:
        m = _REFUSAL.search(text)
        if m:
            return True, f"refusal wording in the answer, and no tool of any branch was called: {m.group(0)!r}"
    return False, None


def rescore_ask(results: Iterable[Result], tasks: Iterable[Task]) -> list[Result]:
    """Re-read the decline of stored ``ask`` rows from their answers (the model is not run again).

    For a change in the refusal lists after a run: ``declined``,
    ``declined_reason``, ``declined_correctly`` and ``answer_present`` are
    recomputed from the stored answer, the inferred branch and the task; a
    row with an error, or of another agent, is returned as it is. Rows keep
    their order.
    """
    by_id = {t.id: t for t in tasks}
    out: list[Result] = []
    for r in results:
        task = by_id.get(r.task_id)
        if r.agent != "ask" or r.error or task is None:
            out.append(r)
            continue
        declined, reason = read_refusal(r.answer, unsolvable=task.unsolvable, branch=r.branch_chosen)
        r.declined, r.declined_reason = declined, reason
        r.declined_correctly = declined if task.unsolvable else None
        r.answer_present = bool(r.answer.strip()) and not declined
        out.append(r)
    return out


def infer_branch(playbook: str, tools_called: Iterable[str]) -> str | None:
    """The playbook branch whose tools the calls cover best (ties to the earlier branch, as the tree would).

    None when no tool of any branch was called. For an agent with no plan of
    its own (``ask``), this is how its branch is read.
    """
    from aquascope import playbooks as pbk

    called = set(tools_called)
    try:
        pb = pbk.load(playbook)
    except pbk.PlaybookError:
        return None
    best: tuple[float, str] | None = None
    for b in pb.branches:
        mine = {s.tool for s in b.steps}
        if not mine:
            continue
        score = len(mine & called) / len(mine)
        if score > 0 and (best is None or score > best[0]):
            best = (score, b.id)
    return best[1] if best else None


# ── scoring ──────────────────────────────────────────────────────────────────


def _score(task: Task, agent: str, cfg: _Config, outcome: dict[str, Any], seconds: float) -> Result:
    expected = task.expected
    exp_gates = {(g.get("step"), g.get("check")) for g in expected.get("gates") or []}
    got_gates = outcome.get("gates") or []
    got_pairs = {(g.get("step"), g.get("check")) for g in got_gates}
    exp_tools = list(expected.get("tools") or [])
    called = list(outcome.get("tools") or [])
    usage = outcome.get("usage") or {}
    model = outcome.get("model") if outcome.get("model") is not None else cfg.model
    provider = outcome.get("provider") if outcome.get("provider") is not None else cfg.provider
    declined = bool(outcome.get("declined"))
    answer = str(outcome.get("answer") or "")
    res = Result(
        task_id=task.id, playbook=task.playbook, split=task.split, unsolvable=task.unsolvable, agent=agent,
        model=model if agent != "tree" else None, provider=provider if agent != "tree" else None,
        branch_expected=expected.get("branch"), branch_chosen=outcome.get("branch"),
        playbook_chosen=outcome.get("playbook"),
        branch_match=(not task.unsolvable and outcome.get("branch") is not None
                      and outcome.get("branch") == expected.get("branch")
                      and (outcome.get("playbook") in (None, task.playbook))),
        gates_expected=len(exp_gates), gates_evaluated=len(got_gates),
        gates_passed=sum(1 for g in got_gates if g.get("passed") is True),
        gates_respected=(len(exp_gates & got_pairs) / len(exp_gates)) if exp_gates else None,
        tools_expected=exp_tools, tools_called=called,
        tools_matched=(len(set(exp_tools) & set(called)) / len(set(exp_tools))) if exp_tools else None,
        declined=declined, declined_correctly=declined if task.unsolvable else None,
        declined_reason=outcome.get("declined_reason"),
        answer_present=bool(answer.strip()) and not declined, answer=answer[:ANSWER_CHARS],
        calls=int(usage.get("calls", 0) or 0), prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        seconds=round(seconds, 2), finished=_now(), detail=dict(outcome.get("detail") or {}),
    )
    res.cost_usd = estimate_cost(res.model, res.prompt_tokens, res.completion_tokens)
    return res


def _error_result(task: Task, agent: str, cfg: _Config, error: str, seconds: float) -> Result:
    return Result(task_id=task.id, playbook=task.playbook, split=task.split, unsolvable=task.unsolvable, agent=agent,
                  model=cfg.model if agent != "tree" else None, provider=cfg.provider if agent != "tree" else None,
                  branch_expected=task.expected.get("branch"), tools_expected=list(task.expected.get("tools") or []),
                  gates_expected=len(task.expected.get("gates") or []), error=error, seconds=round(seconds, 2),
                  finished=_now(), cost_usd=0.0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _with_timeout(fn: Callable[[], Any], seconds: float | None) -> Any:
    """Run ``fn`` in a thread and give up after ``seconds`` (the thread is left to finish on its own)."""
    if not seconds:
        return fn()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise TimeoutError(f"no result within {seconds:g} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ── the run ──────────────────────────────────────────────────────────────────


def select_tasks(tasks: Iterable[Task], *, limit: int | None = None, unsolvable: int | None = None,
                 task_ids: Iterable[str] | None = None, spread: bool = False) -> list[Task]:
    """The tasks a run plays: by id, or the first ``limit`` with at most ``unsolvable`` unsolvable among them.

    A tasks file is site-major (every task of site 1, then site 2, ...), so
    "the first N" is the first few sites. ``spread`` takes them round robin
    over the sites instead (the first task of every site, then the second,
    ...), so a subset covers the suite's sites and branches evenly; the order
    of the file is kept in the result either way.
    """
    pool = list(tasks)
    if task_ids:
        wanted = set(task_ids)
        return [t for t in pool if t.id in wanted]
    if limit is None:
        return pool
    order = _spread_over_sites(pool) if spread else pool
    if unsolvable is None:
        chosen = {t.id for t in order[:limit]}
        return [t for t in pool if t.id in chosen]
    hard = [t for t in order if t.unsolvable][:unsolvable]
    easy = [t for t in order if not t.unsolvable][: max(0, limit - len(hard))]
    chosen = {t.id for t in hard + easy}
    return [t for t in pool if t.id in chosen]


def _spread_over_sites(pool: list[Task]) -> list[Task]:
    """The tasks round robin over their sites, in file order within a site."""
    from aquascope.gym.tasks import site_key

    by_site: dict[str, list[Task]] = {}
    for t in pool:
        by_site.setdefault(site_key(t.site), []).append(t)
    out: list[Task] = []
    rounds = max((len(v) for v in by_site.values()), default=0)
    for i in range(rounds):
        out += [group[i] for group in by_site.values() if i < len(group)]
    return out


def run_bench(
    tasks: Iterable[Task] | str | Path,
    agent: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    limit: int | None = None,
    out: str | Path | None = None,
    unsolvable: int | None = None,
    task_ids: Iterable[str] | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,
    client: Any | None = None,
    max_steps: int = ASK_MAX_STEPS,
    context_chars: int | None = ASK_CONTEXT_CHARS,
    on_event: Callable[[str], None] | None = None,
    spread: bool = False,
    resume: bool = False,
) -> list[Result]:
    """Play ``agent`` (``tree``, ``team`` or ``ask``) on the tasks and score each against its key.

    ``tasks`` is a list or a JSONL path. ``limit`` takes the first N tasks
    (with at most ``unsolvable`` unsolvable ones among them when given;
    ``spread`` takes them round robin over the sites, see
    :func:`select_tasks`); ``task_ids`` picks tasks by id instead. A model is
    used only when one is named (``provider``, ``model``, ``api_key``,
    ``base_url`` or ``client``): ``team`` is keyless otherwise and ``ask``
    needs one. Results are appended to ``out`` as JSONL as they come, and
    ``resume`` skips the tasks ``out`` already holds a finished row for (a
    row with an error, a timeout among them, is played again and the newer
    row wins when the file is read back). The event line after each task
    carries the spend so far, from the price table.
    """
    if agent not in _AGENT_FUNCS:
        raise ValueError(f"unknown agent {agent!r}; one of {AGENTS}")
    say = on_event or (lambda _m: None)
    pool = read_tasks(tasks) if isinstance(tasks, (str, Path)) else list(tasks)
    chosen = select_tasks(pool, limit=limit, unsolvable=unsolvable, task_ids=task_ids, spread=spread)
    cfg = _Config(provider=provider, model=model, api_key=api_key, base_url=base_url, client=client,
                  max_steps=max_steps, context_chars=context_chars)
    if agent == "ask" and not cfg.wants_model:
        raise ValueError("the ask agent needs a model: pass provider/model (or api_key, base_url, client)")
    if agent == "team" and cfg.wants_model and cfg.client is None:
        # Resolve once so every result carries the model that actually answered.
        from aquascope.ai_engine.analyst import resolve_llm

        resolved = resolve_llm(provider, model, api_key, base_url)
        cfg = _Config(provider=str(resolved["provider"]), model=str(resolved["model"]),
                      api_key=resolved["api_key"], base_url=resolved["base_url"], client=None,
                      max_steps=max_steps, context_chars=context_chars)
    fn = _AGENT_FUNCS[agent]
    out_path = Path(out) if out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, Result] = {}
    if resume and out_path and out_path.exists():
        done = {r.task_id: r for r in load_results([out_path])
                if r.agent == agent and (r.model or "") == (cfg.model or "") and not r.error}
        if done:
            say(f"resuming: {len(done)} of {len(chosen)} tasks already have a row in {out_path}")
    results: list[Result] = []
    spent = 0.0
    for i, task in enumerate(chosen, 1):
        if task.id in done:
            continue
        say(f"[{i}/{len(chosen)}] {task.id} ({'unsolvable' if task.unsolvable else task.expected.get('branch')}) "
            f"{task.problem[:70]}")
        t0 = time.time()
        try:
            outcome = _with_timeout(lambda: fn(task, cfg), timeout)
            res = _score(task, agent, cfg, outcome, time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - one failed task is a row, not the end of the run
            res = _error_result(task, agent, cfg, f"{type(exc).__name__}: {exc}"[:400], time.time() - t0)
            logger.warning("task %s failed: %s", task.id, res.error)
        results.append(res)
        spent += res.cost_usd or 0.0
        say(f"    {'correct' if res.correct else 'wrong'}: branch {res.branch_chosen} (expected {res.branch_expected}),"
            f" declined {res.declined}, gates {res.gates_respected}, {res.tokens} tokens, {res.seconds} s"
            + (f", error {res.error}" if res.error else "")
            + (f"; spent {spent:.3f} USD so far" if cfg.wants_model else ""))
        if out_path:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(res.to_dict(), ensure_ascii=False, default=str) + "\n")
    return results


# ── the leaderboard ──────────────────────────────────────────────────────────


def load_results(paths: Iterable[str | Path], *, latest: bool = True) -> list[Result]:
    """Results from JSONL files; rows that are not results (a tasks file in the same folder) are skipped.

    ``latest`` keeps one row per (agent, model, provider, task): the last one
    read, so a task played again after an error or a timeout (a resumed run)
    counts once. ``latest=False`` returns every row.
    """
    out: list[Result] = []
    for p in paths:
        skipped = 0
        with Path(p).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict) and row.get("agent") and row.get("task_id"):
                    out.append(Result.from_dict(row))
                else:
                    skipped += 1
        if skipped:
            logger.info("%s: %d rows are not bench results, skipped", p, skipped)
    if not latest:
        return out
    last: dict[tuple[str, str, str, str], Result] = {}
    for r in out:
        last[(r.agent, r.model or "", r.provider or "", r.task_id)] = r
    return list(last.values())


def _mean(values: Iterable[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(results: Iterable[Result]) -> list[dict[str, Any]]:
    """One row per (agent, model): accuracy on the solvable tasks, the decline rate on the unsolvable ones,
    false declines, gates and tools respected, tokens, seconds, cost, errors, and accuracy per expected branch."""
    groups: dict[tuple[str, str, str], list[Result]] = {}
    for r in results:
        groups.setdefault((r.agent, r.model or "", r.provider or ""), []).append(r)
    rows = []
    for (agent, model, provider), rs in sorted(groups.items()):
        solvable = [r for r in rs if not r.unsolvable]
        hard = [r for r in rs if r.unsolvable]
        test = [r for r in solvable if r.split == "test"]
        by_branch: dict[str, list[int]] = {}
        for r in solvable:
            b = by_branch.setdefault(str(r.branch_expected), [0, 0])
            b[0] += int(r.correct)
            b[1] += 1
        # A row that spent no token (an error, a timeout) costs nothing whatever the model; only a priced token
        # count that the table cannot price leaves the total unknown.
        cost = [r.cost_usd if r.cost_usd is not None or r.tokens else 0.0 for r in rs]
        rows.append({
            "agent": agent, "model": model or None, "provider": provider or None,
            "n": len(rs), "n_solvable": len(solvable), "n_unsolvable": len(hard), "n_test": len(test),
            "accuracy": _mean([r.correct for r in solvable]),
            "accuracy_test": _mean([r.correct for r in test]),
            "decline_rate_unsolvable": _mean([r.declined for r in hard]),
            "false_decline_rate": _mean([r.declined and not r.error for r in solvable]),
            "gates_respected": _mean([r.gates_respected for r in solvable]),
            "tools_matched": _mean([r.tools_matched for r in solvable]),
            "answer_rate": _mean([r.answer_present for r in solvable]),
            "tokens_per_task": _mean([r.tokens for r in rs]),
            "prompt_tokens": sum(r.prompt_tokens for r in rs),
            "completion_tokens": sum(r.completion_tokens for r in rs),
            "seconds_per_task": _mean([r.seconds for r in rs]),
            "cost_usd": round(sum(c for c in cost if c is not None), 4) if all(c is not None for c in cost) else None,
            "errors": sum(1 for r in rs if r.error),
            "timeouts": sum(1 for r in rs if r.error and r.error.startswith("TimeoutError")),
            "by_branch": {k: {"correct": v[0], "n": v[1]} for k, v in sorted(by_branch.items())},
        })
    return rows


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f} %"


def _num(x: float | None, digits: int = 0) -> str:
    return "-" if x is None else f"{x:,.{digits}f}"


def leaderboard(results: Iterable[Result], *, out: str | Path | None = None, title: str | None = None) -> str:
    """The Markdown leaderboard for a set of results (any agents, any models), written to ``out`` when given."""
    rs = list(results)
    rows = summarize(rs)
    n_tasks = len({r.task_id for r in rs})
    lines = [f"## {title or 'HydroGym leaderboard'}", "",
             f"{n_tasks} tasks, {len(rs)} runs, {len(rows)} agent-model pairs. "
             "Accuracy is the share of solvable tasks on which the agent picked the expected playbook branch "
             "(a keyless decline counts as wrong); declined is the share of unsolvable tasks the agent refused; "
             "false declines are solvable tasks it refused; gates and tools are the fractions of the key's gates "
             "evaluated and tools called (means over solvable tasks); accuracy on test is over the solvable tasks "
             "of the held-out split (its size in brackets); an error or a timeout counts as wrong.", "",
             "| agent | model | tasks (solvable + unsolvable) | accuracy | accuracy on test | declined unsolvable "
             "| false declines | gates | tools | tokens/task | s/task | cost USD | errors | timeouts |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        model = r["model"] or ("keyless" if r["agent"] == "team" else "none")
        lines.append(
            f"| {r['agent']} | {model} | {r['n']} ({r['n_solvable']} + {r['n_unsolvable']}) | {_pct(r['accuracy'])} "
            f"| {_pct(r['accuracy_test'])} ({r['n_test']}) | {_pct(r['decline_rate_unsolvable'])} "
            f"| {_pct(r['false_decline_rate'])} | {_pct(r['gates_respected'])} | {_pct(r['tools_matched'])} "
            f"| {_num(r['tokens_per_task'])} | {_num(r['seconds_per_task'], 1)} "
            f"| {_num(r['cost_usd'], 3)} | {r['errors']} | {r['timeouts']} |"
        )
    branches = sorted({b for r in rows for b in r["by_branch"]})
    if branches:
        lines += ["", "Correct on solvable tasks by expected branch (correct / n):", "",
                  "| agent | model | " + " | ".join(branches) + " |", "|---|---|" + "---|" * len(branches)]
        for r in rows:
            model = r["model"] or ("keyless" if r["agent"] == "team" else "none")
            cells = [f"{r['by_branch'][b]['correct']} / {r['by_branch'][b]['n']}" if b in r["by_branch"] else "-"
                     for b in branches]
            lines.append(f"| {r['agent']} | {model} | " + " | ".join(cells) + " |")
    lines += ["", PRICES_NOTE, ""]
    text = "\n".join(lines)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
    return text
