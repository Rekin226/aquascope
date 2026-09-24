"""The Coordinator: the state machine over the workspace, and the API every face is thin over.

    intake -> scouting -> planning -> [waiting] -> review -> running -> interpreting -> critique -> authoring -> done
                                        |                                          |
                                     declined                                  follow-up

:class:`Studio` drives it: :meth:`Studio.say` takes the client's messages
until the brief is ready, then runs the Scout and the Methodologist and
returns the plan; :meth:`Studio.approve` runs the Analysts, the Critic and
the Author to the bundle; :meth:`Studio.follow_up` answers a question from
the workspace or plans, runs and re-authors a change; :meth:`Studio.export`
writes the files. Every method returns a :class:`Reply` and leaves the
workspace consistent, so a face can stop anywhere and resume with
:meth:`Studio.from_dict`. A role's exception never kills the study: it is an
event, and a decline (intake, planning) or a report of what happened (the
run).

The Critic's verdict is acted on, not filed: its ``fix`` issues and every
failed deterministic check go to the Author for one fix round (keyless, the
template repair drops the sentences the checks refuse), the second critique
keeps the model when there is one, and a report whose critique is still not
ok opens with one line naming the failed checks (``report["notice"]``,
``report["critique_ok"]`` and the reply payload's ``critique_ok``). A spend
ceiling (``max_usd``) stops the model calls, not the crew: past it the roles
run keyless, the event and ``ws.budget`` say so, and the footer shows the
USD spent.

Bring your own model: a face that runs a model of its own (the Explorer's
on-device model, any client) hands the crew what that model wrote and the
crew treats it exactly like its own model's output. ``say(text,
proposed={"brief": ...})`` merges a brief with the Consultant's coercion;
``approve(plan=...)`` takes a plan through the Methodologist's validator
(repair, pruning, the tree as the fall-back); ``narrate(sections)`` replaces
the report's prose after the Critic's deterministic checks (a sentence whose
numbers are in no result is dropped). :meth:`Studio.consultant_context`,
:meth:`Studio.methodologist_context` and :meth:`Studio.author_context` are
the prompts and the compact contexts the roles would send, so the same
prompts run anywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aquascope.studio.model import Model
from aquascope.studio.workspace import Artifact, Workspace, now

__all__ = ["Reply", "Studio"]

_APPROVE = re.compile(r"^\s*(approve|approved|run( it| this| the plan)?|go|yes|y|ok(ay)?|looks good|lgtm)\s*[.!]?\s*$",
                      re.I)


@dataclass
class Reply:
    """What a face gets back: ``kind`` is questions, data_request, plan, report, answer or declined."""

    kind: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def questions(self) -> list[dict[str, Any]]:
        return list(self.payload.get("questions") or [])

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "payload": self.payload}


class Studio:
    """A study at a place. See the module docstring and ``docs/studio-design.md``."""

    def __init__(
        self,
        lat: float | None = None,
        lon: float | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        data: dict[str, Any] | None = None,
        on_event: Any = None,
        on_artifact: Any = None,
        max_replans: int = 1,
        workspace: dict[str, Any] | Workspace | None = None,
        tools: dict[str, Any] | None = None,
        intake: dict[str, Any] | None = None,
        max_usd: float | None = None,
    ):
        if isinstance(workspace, Workspace):
            self.ws = workspace
        elif workspace:
            self.ws = Workspace.from_dict(workspace)
        else:
            self.ws = Workspace()
        if lat is not None and lon is not None:
            self.ws.site = {"lat": float(lat), "lon": float(lon)}
        if self.ws.site is None:
            raise ValueError("a study needs a site: pass lat and lon")
        self.on_event = on_event
        self.on_artifact = on_artifact
        self.max_replans = int(max_replans)
        self._tools = dict(tools or {})
        self._frames: dict[str, Any] = {}
        self.ws.listener = self._relay
        for key, value in (data or {}).items():
            dataset_id = key if str(key).startswith("upload:") else f"upload:{key}"
            self.ws.add_table(dataset_id, value)
            if not isinstance(value, str):
                self._frames[dataset_id] = value
        if intake:
            self.ws.brief.intake.update({k: v for k, v in intake.items() if v is not None})
        self.max_usd = float(max_usd) if max_usd is not None else None
        self.model = Model.resolve(self.ws, provider=provider, model=model, api_key=api_key, base_url=base_url,
                                   client=client, max_usd=self.max_usd)
        if self.model:
            self.ws.event("coordinator", "model", f"{self.ws.model} via {self.ws.provider}"
                          + (f", ceiling {self.max_usd:.2f} USD" if self.max_usd is not None else ""))

    # ── plumbing ──

    @property
    def workspace(self) -> Workspace:
        return self.ws

    def _relay(self, event: dict[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def _decline(self, reason: str, *, role: str = "coordinator") -> Reply:
        self.ws.declined_reason = reason
        if self.ws.status != "declined":
            self.ws.set_status("declined")
        self.ws.event(role, "declined", reason)
        if not self.ws.messages or self.ws.messages[-1].kind != "declined":
            self.ws.say(role, f"Declined: {reason}", kind="declined", payload={"reason": reason})
        return Reply("declined", f"Declined: {reason}", {"reason": reason})

    def _plan_reply(self) -> Reply:
        from aquascope.studio.roles.methodologist import plan_text

        text = plan_text(self.ws.study)
        return Reply("plan", text, {"study": self.ws.study, "plan": (self.ws.study or {}).get("plan") or {},
                                    "brief": self.ws.brief.to_dict()})

    def _questions_reply(self, text: str) -> Reply:
        return Reply("questions", text, {"questions": [q.to_dict() for q in self.ws.brief.open_questions],
                                         "brief": self.ws.brief.to_dict()})

    def _report_reply(self) -> Reply:
        report = self.ws.report or {}
        return Reply("report", str(report.get("answer") or ""), {
            "report": report, "artifacts": [a.to_dict(with_data=False) for a in self.ws.artifacts],
            "not_established": report.get("not_established") or [], "status": self.ws.status,
            "critique_ok": bool(report.get("critique_ok", True)), "notice": report.get("notice"),
            "dropped": int(report.get("dropped") or 0), "cost_usd": self.ws.total_usd, "budget": self.ws.budget,
            "findings": (self.ws.findings or {}).get("findings") or [],
            "decision": (self.ws.findings or {}).get("decision"),
            "data_requests": (self.ws.findings or {}).get("data_requests") or [],
            "grade": ((self.ws.findings or {}).get("decision") or {}).get("grade"),
        })

    def _apply_verdict(self) -> None:
        """Carry the Critic's verdict into the report: what is not established, the checks and the issues, and
        when the critique is not ok, the one-line notice the answer opens with (so the bundle's README and every
        document open with it too)."""
        from aquascope.studio.roles.critic import failed_checks, notice

        ws = self.ws
        if ws.report is None or ws.critique is None:
            return
        report, critique = ws.report, ws.critique
        issues = critique.get("issues") or []
        ok = bool(critique.get("ok", not failed_checks(critique) and not any(i.get("severity") == "fix"
                                                                              for i in issues)))
        report["not_established"] = list(critique.get("not_established") or [])
        report["critique"] = {"ok": ok, "issues": issues, "failed": failed_checks(critique),
                              "checks_passed": sum(1 for c in critique.get("checks") or [] if c.get("passed")),
                              "checks": len(critique.get("checks") or [])}
        report["critique_ok"] = ok
        answer = str(report.get("answer") or "")
        old = report.get("notice")
        if isinstance(old, str) and old and answer.startswith(old):
            answer = answer[len(old):].lstrip()
        line = notice(critique)
        if line:
            report["notice"] = line
            report["answer"] = f"{line}\n\n{answer}" if answer else line
            ws.event("critic", "notice", line)
        else:
            report.pop("notice", None)
            report["answer"] = answer

    # ── the conversation ──

    def say(self, text: str, *, proposed: dict[str, Any] | None = None) -> Reply:
        """The client's message. Intake until the brief is ready, then scouting and planning; at review, an
        approval word runs the plan and anything else changes the brief and plans again; after the report,
        a follow-up. ``proposed`` is ``{"brief": {...}, "source": "device"}``: a brief a model of the caller's
        own wrote from the text (decision, quantities, period, horizon, constraints, kind, playbook, intake,
        assumptions, questions), merged with the coercion a model reply gets before the flow goes on."""
        from aquascope.studio.roles.consultant import consult

        ws = self.ws
        if ws.status == "declined":
            return Reply("declined", f"Declined: {ws.declined_reason}", {"reason": ws.declined_reason})
        if ws.status == "done":
            return self.follow_up(text)
        if ws.status in ("running", "critique", "authoring"):
            return Reply("answer", "The crew is running; the report comes next.", {"status": ws.status})
        if ws.status == "waiting":
            return self._answer_request(text)
        if ws.status == "review":
            if _APPROVE.match(text or ""):
                return self.approve()
            try:
                consult(ws, self.model, text, tables=self._frames, proposed=proposed)
            except Exception as exc:  # noqa: BLE001
                ws.event("consultant", "error", f"{type(exc).__name__}: {exc}")
                return self._plan_reply()
            ws.event("coordinator", "replan", "the brief changed at review")
            return self._plan()
        try:
            msg = consult(ws, self.model, text, tables=self._frames, proposed=proposed)
        except Exception as exc:  # noqa: BLE001 - an intake failure is a decline, not a crash
            ws.event("consultant", "error", f"{type(exc).__name__}: {exc}")
            return self._decline(f"the Consultant could not take the brief: {exc}", role="consultant")
        if not ws.brief.ready:
            return self._questions_reply(msg.text)
        return self._scout_and_plan()

    def _request_reply(self) -> Reply:
        from aquascope.studio.roles.methodologist import request_text

        request = dict(self.ws.pending_request or {})
        return Reply("data_request", request_text(request) if request else "The crew is waiting for data.",
                     {"request": request, "brief": self.ws.brief.to_dict(), "status": self.ws.status})

    def _answer_request(self, text: str) -> Reply:
        """A reply while the study waits for data: "continue without" plans at the lower grade the request
        named (or declines when it allows no continuation); anything else repeats the request. A table
        arrives through :meth:`add_table`, which plans again on its own."""
        from aquascope.studio.requests import continue_without, is_continue

        ws = self.ws
        request = dict(ws.pending_request or {})
        ws.say("user", text)
        if not is_continue(text):
            return self._request_reply()
        if not continue_without(ws, request):
            ws.pending_request = None
            return self._decline(str(request.get("reason") or f"{request.get('what')} was not provided"),
                                 role="methodologist")
        ws.pending_request = None
        ws.brief.intake["_no_request"] = True
        try:
            return self._plan()
        finally:
            ws.brief.intake.pop("_no_request", None)

    def add_table(self, name: str, frame_or_csv: Any) -> Reply:
        """A table of the user's at any point of the study. At intake it is kept for the brief; while the study
        waits for data or sits at review it is inventoried and the plan is written again; after the report it
        is a follow-up change ("use the new table"), run and re-authored. The reply is the next one the study
        would give."""
        from aquascope.studio.roles.scout import scout

        ws = self.ws
        dataset_id = name if str(name).startswith("upload:") else f"upload:{name}"
        ws.add_table(dataset_id, frame_or_csv)
        if not isinstance(frame_or_csv, str):
            self._frames[dataset_id] = frame_or_csv
        ws.event("coordinator", "table", f"{dataset_id} added at {ws.status}")
        if ws.status in ("waiting", "review"):
            ws.pending_request = None
            ws.set_status("scouting")
            try:
                scout(ws)
            except Exception as exc:  # noqa: BLE001
                ws.event("scout", "error", f"{type(exc).__name__}: {exc}")
                return self._decline(f"the Scout could not read the table: {exc}", role="scout")
            return self._plan()
        if ws.status == "done":
            return self.follow_up(f"use the new table {dataset_id} and redo the steps it serves")
        return Reply("answer", f"{dataset_id} is attached; the crew will use it.", {"table": dataset_id,
                                                                                   "status": ws.status})

    def _scout_and_plan(self) -> Reply:
        from aquascope.studio.roles.scout import scout

        ws = self.ws
        ws.set_status("scouting")
        try:
            scout(ws)
        except Exception as exc:  # noqa: BLE001
            ws.event("scout", "error", f"{type(exc).__name__}: {exc}")
            return self._decline(f"the Scout could not build the inventory: {exc}", role="scout")
        return self._plan()

    def _plan(self) -> Reply:
        from aquascope.studio.roles.methodologist import plan

        ws = self.ws
        ws.set_status("planning")
        try:
            study = plan(ws, self.model)
        except Exception as exc:  # noqa: BLE001
            ws.event("methodologist", "error", f"{type(exc).__name__}: {exc}")
            return self._decline(f"the Methodologist failed: {exc}", role="methodologist")
        if study is None:
            if ws.status == "waiting":
                return self._request_reply()
            if ws.status != "declined":
                return self._decline(ws.declined_reason or "no plan", role="methodologist")
            return Reply("declined", f"Declined: {ws.declined_reason}", {"reason": ws.declined_reason})
        ws.set_status("review")
        return self._plan_reply()

    # ── the run ──

    def approve(self, edits: dict[str, Any] | list[dict[str, Any]] | None = None, *,
                plan: dict[str, Any] | None = None) -> Reply:
        """Approve the plan (with the user's edits, revalidated) and run the crew to the report.

        ``plan`` is a plan a model of the caller's own wrote, in the Methodologist's reply shape (objective,
        decision, methodology, steps with id, tool, arguments, rationale, method, expects, fallback,
        depends_on, outputs; assumptions, alternatives, limitations_expected, citations) plus an optional
        ``source`` (default ``device``). It goes the way a model plan goes: the validator with its repairs, a
        wrong method dropped, the invalid steps pruned, the tree when nothing valid remains. The report's
        payload then carries ``plan_errors`` (the validator's findings) and ``plan_used`` (``proposed`` or
        ``tree``), and the plan says who wrote it.
        """
        from aquascope.studio.roles.methodologist import adopt, revise

        ws = self.ws
        if ws.status == "declined":
            return Reply("declined", f"Declined: {ws.declined_reason}", {"reason": ws.declined_reason})
        if ws.status != "review" or not ws.study:
            return Reply("answer", f"There is no plan to approve (status {ws.status}).", {"status": ws.status})
        extra: dict[str, Any] = {}
        if plan:
            source = str(plan.get("source") or "device") if isinstance(plan, dict) else "device"
            try:
                _, errors, used = adopt(ws, plan, source=source)
            except Exception as exc:  # noqa: BLE001 - a broken proposal runs the plan at review
                ws.event("methodologist", "error", f"{source} plan: {type(exc).__name__}: {exc}")
                errors, used = [f"{type(exc).__name__}: {exc}"], "tree"
            extra = {"plan_errors": list(errors), "plan_used": used}
            ws.event("coordinator", "review", f"the {source} plan was {'adopted' if used == 'proposed' else 'refused'}"
                     + (f" ({len(errors)} validator finding(s))" if errors else ""))
        if edits:
            try:
                revise(ws, self.model, edits)
            except ValueError as exc:
                reply = self._plan_reply()
                reply.text = f"The edit was not accepted: {exc}\n\n" + reply.text
                reply.payload["errors"] = str(exc).split("; ")
                reply.payload.update(extra)
                return reply
        ws.event("coordinator", "review", "the plan was approved" + (" with edits" if edits else ""))
        reply = self._run_to_report()
        reply.payload.update(extra)
        return reply

    def _run_to_report(self, *, prior: Any = None, reuse: list[str] | None = None) -> Reply:
        from aquascope.studio.roles.analysts import run
        from aquascope.studio.roles.author import author_report
        from aquascope.studio.roles.critic import critique, fixes_for
        from aquascope.studio.roles.interpreter import interpret

        ws = self.ws
        ws.set_status("running")
        try:
            run(ws, self.model, tools=self._tools, on_artifact=self.on_artifact, max_replans=self.max_replans,
                prior=prior, reuse=reuse)
        except Exception as exc:  # noqa: BLE001 - the report says what happened
            reason = f"the run failed: {type(exc).__name__}: {exc}"
            ws.event("analyst", "error", reason)
            ws.run = {"ok": False, "results": [], "gates": [], "failed_gates": [], "stopped_at": None,
                      "stop_reason": reason, "started": now(), "finished": now(), "replans": 0}
        ws.set_status("critique")
        try:
            interpret(ws, self.model)
        except Exception as exc:  # noqa: BLE001 - the findings are an aid; the report goes on without them
            ws.event("interpreter", "error", f"{type(exc).__name__}: {exc}")
            ws.findings = None
        try:
            author_report(ws, self.model)
        except Exception as exc:  # noqa: BLE001
            ws.event("author", "error", f"{type(exc).__name__}: {exc}")
            ws.report = {"title": ws.brief.problem, "answer": f"The report could not be written: {exc}",
                         "key_numbers": [], "sections": [], "not_established": [str(exc)], "references": [],
                         "footer": {"model": ws.model, "provider": ws.provider}}
        try:
            critique(ws, self.model)
        except Exception as exc:  # noqa: BLE001
            ws.event("critic", "error", f"{type(exc).__name__}: {exc}")
            ws.critique = {"ok": False, "checks": [], "issues": [], "not_established": [f"the Critic failed: {exc}"]}
        ws.set_status("authoring")
        fixes = fixes_for(ws.critique)
        if fixes:
            checks = [str(i["check"]) for i in fixes if i.get("check")]
            ws.event("critic", "fixes", f"{len(fixes)} fix(es) for the Author"
                     + (f", failed checks: {', '.join(checks)}" if checks else ""))
            try:
                author_report(ws, self.model, issues=fixes)
                critique(ws, self.model)
            except Exception as exc:  # noqa: BLE001
                ws.event("author", "error", f"{type(exc).__name__}: {exc}")
        self._apply_verdict()
        self._build_deliverables()
        ws.set_status("done")
        report = ws.report or {}
        ws.say("author", str(report.get("answer") or ""), kind="report",
               payload={"title": report.get("title"), "key_numbers": report.get("key_numbers"),
                        "not_established": report.get("not_established"),
                        "artifacts": [a.to_dict(with_data=False) for a in ws.artifacts]})
        return self._report_reply()

    def _build_deliverables(self) -> None:
        ws = self.ws
        try:
            from aquascope.studio.deliverables import build
        except ImportError as exc:
            ws.event("author", "deliverables_unavailable", f"{exc}")
            return
        try:
            made = build(ws)
            n = len(made) if isinstance(made, (list, dict)) else len(ws.artifacts)
            ws.event("author", "deliverables", f"{n} artifact(s)")
        except Exception as exc:  # noqa: BLE001
            ws.event("author", "error", f"deliverables: {type(exc).__name__}: {exc}")

    # ── after the report ──

    def follow_up(self, text: str) -> Reply:
        """A question is answered from the workspace; a change is planned, run and re-authored."""
        from aquascope.studio.roles.analysts import prior_run
        from aquascope.studio.roles.consultant import classify_follow_up
        from aquascope.studio.roles.methodologist import change

        ws = self.ws
        if ws.status != "done":
            return Reply("answer", f"The study is not finished (status {ws.status}); a follow-up comes after the "
                         "report.", {"status": ws.status})
        ws.say("user", text)
        try:
            verdict = classify_follow_up(ws, self.model, text)
        except Exception as exc:  # noqa: BLE001
            ws.event("consultant", "error", f"{type(exc).__name__}: {exc}")
            verdict = {"kind": "question", "answer": str((ws.report or {}).get("answer") or "")}
        entry: dict[str, Any] = {"text": text, "at": now(), "kind": verdict["kind"]}
        if verdict["kind"] == "question":
            answer = str(verdict.get("answer") or "")
            entry["answer"] = answer
            ws.follow_ups.append(entry)
            ws.event("consultant", "follow_up", "question answered from the workspace")
            ws.say("consultant", answer, kind="text")
            return Reply("answer", answer, {"kind": "question"})
        ws.event("consultant", "follow_up", f"change: {verdict.get('request') or text}")
        prior = prior_run(ws)
        ws.set_status("planning")
        try:
            study = change(ws, self.model, str(verdict.get("request") or text), intake=verdict.get("intake"))
        except Exception as exc:  # noqa: BLE001
            ws.event("methodologist", "error", f"{type(exc).__name__}: {exc}")
            study = None
        if study is None:
            ws.set_status("done")
            last = next((e["detail"] for e in reversed(ws.events)
                         if e["role"] == "methodologist" and e["event"] in ("declined", "error")), "no plan")
            answer = f"The change could not be planned: {last}"
            entry["answer"] = answer
            ws.follow_ups.append(entry)
            ws.say("methodologist", answer, kind="text")
            return Reply("answer", answer, {"kind": "change", "planned": False})
        entry["steps"] = [s.get("id") for s in (ws.study or {}).get("steps") or []]
        entry["intake"] = dict(verdict.get("intake") or {})
        ws.follow_ups.append(entry)
        ws.set_status("review")
        ws.event("coordinator", "review", "a follow-up change runs without a second approval")
        return self._run_to_report(prior=prior)

    # ── steering one step (aquascope.studio.steering) ──

    def steer(self, step_id: str, changes: dict[str, Any]) -> Reply:
        """Change a steerable parameter of one step after the report: validated against the catalogue's
        declaration, then that step and its dependants rerun (every other result stands as it was), the change
        is recorded in the plan's ``steering`` list so study.yaml and the notebook reproduce it, and the report
        is written again. A refused change leaves the study as it was and says why."""
        from aquascope.studio.roles.analysts import prior_run
        from aquascope.studio.steering import apply_change

        ws = self.ws
        if ws.status != "done" or not ws.study:
            return Reply("answer", f"A step can be adjusted once the report is out (status {ws.status}).",
                         {"kind": "steer", "status": ws.status})
        try:
            out = apply_change(ws.study_obj(), str(step_id), dict(changes or {}))
        except ValueError as exc:
            text = f"The change was not accepted: {exc}"
            ws.say("coordinator", text)
            return Reply("answer", text, {"kind": "steer", "errors": str(exc).split("; ")})
        prior = prior_run(ws)
        keep = [str(r.get("id")) for r in (prior.results if prior else []) if str(r.get("id")) not in out["dirty"]]
        ws.say("user", out["text"])
        ws.follow_ups.append({"text": out["text"], "at": now(), "kind": "steer", "steps": out["dirty"],
                              "change": out["change"]})
        ws.set_study(out["study"])
        ws.event("coordinator", "steer", f"{out['text']}; rerunning {', '.join(out['dirty'])}")
        return self._run_to_report(prior=prior, reuse=keep)

    def narrate(self, sections: dict[str, str] | list[dict[str, Any]], *, source: str = "device") -> Reply:
        """Prose a model of the caller's own wrote for the report, after the crew's checks (only after the
        report). ``sections`` maps section ids (as in ``ws.report["sections"]``, plus ``answer`` and
        ``recommendations``) to text, or lists ``{"id", "text"}``. A sentence whose numbers are in no tool
        result is dropped and counted; the report says which source wrote which section
        (``ws.report["written_by"]``, the footer); the deliverables are rebuilt. The reply is the report with
        ``dropped`` and ``written_by`` in its payload. Sections not given keep their text."""
        from aquascope.studio.roles.author import narrate
        from aquascope.studio.roles.critic import critique

        ws = self.ws
        if ws.status != "done" or not ws.report:
            return Reply("answer", f"There is no report to narrate yet (status {ws.status}).", {"status": ws.status})
        given: dict[str, str] = {}
        rows = sections.items() if isinstance(sections, dict) else \
            [(r.get("id"), r.get("text")) for r in sections if isinstance(r, dict)]
        for sid, text in rows:
            if sid and isinstance(text, str) and text.strip():
                given[str(sid)] = text
        outcome = narrate(ws, given, source=source)
        try:
            critique(ws, None)
        except Exception as exc:  # noqa: BLE001
            ws.event("critic", "error", f"{type(exc).__name__}: {exc}")
        self._apply_verdict()
        self._build_deliverables()
        ws.say("author", str(ws.report.get("answer") or ""), kind="report",
               payload={"title": ws.report.get("title"), "key_numbers": ws.report.get("key_numbers"),
                        "not_established": ws.report.get("not_established"), "written_by": outcome["written_by"],
                        "artifacts": [a.to_dict(with_data=False) for a in ws.artifacts]})
        reply = self._report_reply()
        reply.payload.update(outcome)
        return reply

    # ── bring your own model: the prompts and the contexts the roles would send ──

    def consultant_context(self, text: str) -> dict[str, Any]:
        """The Consultant's context for ``text`` (the brief's, or the answers' while questions are open) with
        the system prompt under ``system``; a page runs the same prompt on a model of its own and hands the
        reply to :meth:`say` as ``proposed``."""
        from aquascope.studio.roles.consultant import brief_context

        system, context = brief_context(self.ws, text, self._frames)
        return {**context, "system": system}

    def methodologist_context(self, request: str | None = None) -> dict[str, Any]:
        """The Methodologist's context at this point (the plan's; with ``request``, or after the report, the
        change's) with the system prompt under ``system``; the reply goes to :meth:`approve` as ``plan``."""
        from aquascope.studio.roles.methodologist import change_context, plan_context

        ws = self.ws
        if request is not None or ws.status == "done":
            system, context = change_context(ws, request or "")
        else:
            system, context = plan_context(ws)
        return {**context, "system": system}

    def author_context(self, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """The Author's context at this point (the report's; with ``issues``, the fix round's) with the system
        prompt under ``system``; the reply's sections go to :meth:`narrate`."""
        from aquascope.studio.roles.author import report_context

        system, context = report_context(self.ws, issues=issues)
        return {**context, "system": system}

    # ── files and checkpoints ──

    def export(self, out_dir: str | Path) -> dict[str, str]:
        """Write the bundle's files into ``out_dir`` and return their paths by name."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            from aquascope.studio.deliverables import export as export_bundle
        except ImportError:
            export_bundle = None
        if export_bundle is not None:
            try:
                paths = export_bundle(self.ws, out)
                if isinstance(paths, dict):
                    return {str(k): str(v) for k, v in paths.items()}
            except Exception as exc:  # noqa: BLE001
                self.ws.event("author", "error", f"export: {type(exc).__name__}: {exc}")
        return self._export_plain(out)

    def _export_plain(self, out: Path) -> dict[str, str]:
        from aquascope.studio.roles.author import to_markdown

        ws = self.ws
        paths: dict[str, str] = {}
        (out / "report.md").write_text(to_markdown(ws), encoding="utf-8")
        paths["report.md"] = str(out / "report.md")
        study = ws.study_obj()
        if study is not None:
            (out / "study.yaml").write_text(study.to_yaml(), encoding="utf-8")
            paths["study.yaml"] = str(out / "study.yaml")
        if ws.report:
            (out / "report.json").write_text(json.dumps(ws.report, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
            paths["report.json"] = str(out / "report.json")
        for a in ws.artifacts:
            if not a.data:
                continue
            target = out / a.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(a.data)
            paths[a.name] = str(target)
        (out / "workspace.json").write_text(ws.to_json(indent=None), encoding="utf-8")
        paths["workspace.json"] = str(out / "workspace.json")
        ws.event("coordinator", "export", f"{len(paths)} file(s) in {out}")
        return paths

    def to_dict(self, *, with_artifacts: bool = True) -> dict[str, Any]:
        return self.ws.to_dict(with_artifacts=with_artifacts)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | Workspace, **kwargs: Any) -> Studio:
        """Resume from a workspace dict; the model kwargs (provider, model, api_key, base_url, client),
        ``max_usd``, ``on_event``, ``on_artifact``, ``tools`` and ``data`` are the constructor's."""
        return cls(workspace=d, **kwargs)

    def add_artifact(self, artifact: Artifact) -> Artifact:
        return self.ws.add_artifact(artifact)
