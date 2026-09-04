#!/usr/bin/env python3
"""Drive aquascope's team from a LangGraph StateGraph.

The team behind ``aquascope solve`` is a set of plain functions sharing one
study: the Scout (``aquascope.explore.assess_site``), the Coordinator and the
tree (``aquascope.ai_engine.team.solve(execute=False)``), your review, the
runner with its gates and the Reviewer and Narrator
(``aquascope.ai_engine.team.run_reviewed``). The package needs no agent
framework to run them (the browser runs the same functions in a worker), so
they drop into whatever orchestrator your stack already has. This file is
that mapping for LangGraph, one node per role, with the review as a
human-in-the-loop interrupt:

    START -> scout -> plan -> review -> run -> report -> END
                        |        |
                        +--------+--> report (declined)

Run it keyless at the Thames at Kingston::

    pip install langgraph langchain-core          # not aquascope dependencies
    python examples/langgraph_team.py --lat 51.415 --lon -0.308 --playbook flood_risk \\
        --problem "Design flow for a road crossing, 100-year return period" --yes

Drop ``--yes`` to be asked ``y/N`` at the review interrupt. The same four
functions map onto CrewAI-style roles just as directly: scout, planner,
reviewer and runner agents whose tools are these functions, in that order.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Callable
from typing import Any, TypedDict

LANGGRAPH_MISSING = (
    "This example drives aquascope from LangGraph, which aquascope does not depend on. "
    "Install it next to aquascope with:  pip install langgraph langchain-core"
)

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    HAVE_LANGGRAPH = True
except ImportError:  # the nodes below still import and run; only build_graph() needs LangGraph
    HAVE_LANGGRAPH = False
    Command = interrupt = InMemorySaver = StateGraph = START = END = None  # type: ignore[assignment]


class TeamState(TypedDict, total=False):
    """The blackboard: what the roles read and write."""

    problem: str
    lat: float
    lon: float
    playbook: str | None
    intake: dict[str, Any]
    recon: dict[str, Any]          # written by scout
    study: dict[str, Any]          # written by plan (the Study as a dict), edited by review
    declined: str | None           # a reason, when the plan or the review said no
    approved: bool                 # written by review
    result: dict[str, Any]         # written by run (SolveResult.to_dict())
    report: str                    # written by report


#: When LangGraph's ``interrupt`` is not available (or the graph runs without
#: a checkpointer), the review node calls this instead: it gets the study dict
#: and returns the study to run (edited or not), or None to decline.
REVIEW: Callable[[dict[str, Any]], dict[str, Any] | None] = lambda study: study  # noqa: E731


# ── the nodes: one per role, each a plain call into the package ─────────────


def scout_node(state: TeamState) -> dict[str, Any]:
    """Scout: what exists at the place and what the record supports."""
    from aquascope.explore import assess_site

    rp = (state.get("intake") or {}).get("return_period")
    recon = assess_site(float(state["lat"]), float(state["lon"]), problem=state.get("playbook"),
                        return_period=float(rp) if isinstance(rp, (int, float)) else None)
    return {"recon": recon}


def plan_node(state: TeamState) -> dict[str, Any]:
    """Coordinator: the playbook branch for the data that exists, filled into a study. Nothing runs."""
    from aquascope.ai_engine.team import solve

    res = solve(state.get("problem") or "", lat=float(state["lat"]), lon=float(state["lon"]),
                playbook=state.get("playbook"), intake=state.get("intake"), recon=state.get("recon"),
                execute=False)
    return {"study": res.study.to_dict(), "declined": res.declined_reason if res.declined else None}


def review_node(state: TeamState) -> dict[str, Any]:
    """You: the plan before anything runs. An interrupt when LangGraph can pause, a callback otherwise.

    Resume the interrupt with the study dict (edited or not) or ``True`` to run
    it, with ``None`` or ``False`` to decline.
    """
    study = state.get("study") or {}
    decision: Any
    if HAVE_LANGGRAPH and interrupt is not None:
        try:
            decision = interrupt({"question": "Run this plan?", "plan": plan_text(study), "study": study})
        except RuntimeError:  # no checkpointer configured: interrupts cannot pause, ask the callback
            decision = REVIEW(study)
    else:
        decision = REVIEW(study)
    if decision is None or decision is False:
        return {"approved": False, "declined": "The plan was declined at review."}
    if isinstance(decision, dict) and decision.get("steps"):
        return {"approved": True, "study": decision}
    return {"approved": True}


def run_node(state: TeamState, *, tools: dict[str, Callable[..., Any]] | None = None) -> dict[str, Any]:
    """Runner, Reviewer and Narrator: the gates per step, one bounded replan, the report."""
    from aquascope.ai_engine.team import run_reviewed

    res = run_reviewed(state["study"], recon=state.get("recon"), tools=tools)
    return {"result": res.to_dict()}


def report_node(state: TeamState) -> dict[str, Any]:
    """The answer with what it does not establish; or the decline, in the playbook's own words."""
    result = state.get("result")
    if result:
        return {"report": str(result.get("report") or result.get("answer") or "")}
    plan = (state.get("study") or {}).get("plan") or {}
    reason = state.get("declined") or plan.get("declined") or "no plan"
    return {"report": f"Declined: {reason}\n"}


def _after_plan(state: TeamState) -> str:
    return "report" if state.get("declined") else "review"


def _after_review(state: TeamState) -> str:
    return "run" if state.get("approved") else "report"


# ── the graph ───────────────────────────────────────────────────────────────


def build_graph(*, checkpointer: Any | None = None, tools: dict[str, Callable[..., Any]] | None = None) -> Any:
    """The compiled StateGraph. ``checkpointer`` (default: in memory) lets the review interrupt pause."""
    if not HAVE_LANGGRAPH:
        raise ImportError(LANGGRAPH_MISSING)
    g = StateGraph(TeamState)
    g.add_node("scout", scout_node)
    g.add_node("plan", plan_node)
    g.add_node("review", review_node)
    g.add_node("run", functools.partial(run_node, tools=tools))
    g.add_node("report", report_node)
    g.add_edge(START, "scout")
    g.add_edge("scout", "plan")
    g.add_conditional_edges("plan", _after_plan, {"review": "review", "report": "report"})
    g.add_conditional_edges("review", _after_review, {"run": "run", "report": "report"})
    g.add_edge("run", "report")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


def plan_text(study: dict[str, Any]) -> str:
    """The plan as the CLI prints it: branch, rationale, a numbered step list with gates."""
    plan = study.get("plan") or {}
    steps = study.get("steps") or []
    lines = [f"Plan: playbook {plan.get('playbook')}, branch {plan.get('branch')}, {len(steps)} step(s)"]
    if plan.get("rationale"):
        lines.append(f"  {plan['rationale']}")
    for i, s in enumerate(steps, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in (s.get("arguments") or {}).items())
        lines.append(f"  {i}. {s.get('tool')}({args})")
        for g in s.get("expects") or []:
            value = f" {g['value']}" if g.get("value") is not None else ""
            lines.append(f"     gate {g.get('check')}{value} on {g.get('path') or ','.join(g.get('paths') or [])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="aquascope's team as a LangGraph StateGraph")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--problem", default="Design flow for a road crossing, 100-year return period")
    ap.add_argument("--playbook", default=None,
                    help="flood_risk, ungauged_flow, groundwater_decline (default: the keyword rules)")
    ap.add_argument("--intake", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--yes", action="store_true", help="approve the plan without asking")
    ap.add_argument("--thread", default="kingston", help="the checkpoint thread id")
    args = ap.parse_args(argv)
    if not HAVE_LANGGRAPH:
        print(LANGGRAPH_MISSING, file=sys.stderr)
        return 2

    intake: dict[str, Any] = {}
    for item in args.intake:
        key, _, value = item.partition("=")
        try:
            intake[key] = json.loads(value)
        except json.JSONDecodeError:
            intake[key] = value

    graph = build_graph()
    config = {"configurable": {"thread_id": args.thread}}
    state = graph.invoke({"problem": args.problem, "lat": args.lat, "lon": args.lon,
                          "playbook": args.playbook, "intake": intake}, config=config)
    pauses = state.get("__interrupt__") or []
    if pauses:
        payload = pauses[0].value
        print(payload["plan"])
        if args.yes:
            answer = "y"
        else:
            try:
                answer = input("Run this plan? [y/N] ").strip().lower()
            except EOFError:
                answer = "n"
        state = graph.invoke(Command(resume=(payload["study"] if answer == "y" else None)), config=config)
    print()
    print(state.get("report") or "no report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
