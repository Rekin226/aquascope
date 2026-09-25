"""The Consultant: from the client's words to a brief the crew can plan from.

Keyless, the brief comes from the keyword rules that pick a playbook
(``team.choose_playbook``), the intake hints the text states outright
(``team.intake_hints``), the decision the text names, and the gaps that are
left asked as questions (at most three): a playbook field with no default,
the decision when none is stated (with the playbook's options), the return
period of a flood question, the period of a drought question, the column of
an upload with two or more numeric columns. "just go" proceeds on the
defaults and lists them as assumptions. With a model, one call reads the
text, the site, a catalog-only reconnaissance and the uploads' column names
and returns the structured brief with its questions. A brief a caller's own
model wrote (the Explorer's on-device model) arrives as ``proposed`` and is
merged with the same coercion a model reply gets. Answers arrive as later
calls of :func:`consult`.

After the report, :func:`classify_follow_up` says whether a follow-up is a
question (answered from the workspace) or a change (a new intake, a request
for the Methodologist).
"""

from __future__ import annotations

import csv
import io
import itertools
import re
from typing import Any

from aquascope.studio.model import Model, compact
from aquascope.studio.prompts import CONSULTANT, CONSULTANT_ANSWERS, CONSULTANT_FOLLOW_UP
from aquascope.studio.workspace import Message, Question, Workspace

MAX_QUESTIONS = 3
#: Checklist questions asked, one at a time, before the rest take their defaults as assumptions.
MAX_TURNS = 4
RADIUS_KM = 50.0

_PROCEED = re.compile(
    r"^\s*(just go|go|go ahead|proceed|defaults?|use (the )?defaults?|ok(ay)?|yes|y|fine|continue|run|"
    r"whatever you think|your call)\s*[.!]?\s*$", re.I,
)
_SPLIT = re.compile(r"\s*(?:[,;\n]|\band\b)\s*", re.I)
_CHANGE = re.compile(
    r"\b(redo|re-?run|instead|change|switch|add|also (compute|run|check|fit)|extend|repeat|with a|use the|"
    r"another|different|longer|shorter|what about)\b", re.I,
)
#: A period the text states for a drought question ("now", "this summer", "the last 12 months", "since 2018").
_PERIOD = re.compile(
    r"\b(now|currently|current|today|at present|this (month|season|summer|winter|spring|autumn|year)|"
    r"(the )?(last|past) (\d+ )?(months?|years?|decade|season|winter|summer)|since \d{4}|\d{4}\s*(to|-)\s*\d{4}|"
    r"the whole record|on record)\b", re.I,
)
#: Two intake fields that ask the same thing: one answered closes the other.
_ALTERNATES: dict[str, tuple[str, ...]] = {"demand_m3s": ("demand_ml_day",), "demand_ml_day": ("demand_m3s",)}
#: The intake field that carries the decision, per playbook (its options are the question's).
_DECISION_FIELDS: dict[str, str] = {
    "flood_risk": "decision", "irrigation_feasibility": "decision", "ungauged_flow": "purpose",
    "groundwater_decline": "concern", "water_quality": "use",
}
#: The decisions a client names for the playbooks that have no decision field, with the words that name them.
_DECISION_OPTIONS: dict[str, list[tuple[str, str]]] = {
    "supply_reliability": [("an abstraction licence", r"licen[cs]e|permit"),
                           ("the size of the scheme", r"\bsiz(e|ing)\b|capacity|how big"),
                           ("a screening of the source", r"screen")],
    "drought_status": [("drought restrictions", r"restrict|hosepipe|\bban\b|declar"),
                       ("irrigation planning", r"irrigat|plant|sow|harvest|crop"),
                       ("a situation report", r"\breport\b|briefing|update")],
}
#: The words that name an option of a decision field (the option's own words always count).
_OPTION_WORDS: dict[str, str] = {
    "design flow": r"design|culvert|bridge|crossing|spillway|levee|embankment|\bsiz(e|ing)\b",
    "risk screening": r"screen|how risky|at risk|\brisk",
    "insurance": r"insur",
    "inundation extent": r"inundat|flood (map|extent)|how deep|which (streets|houses|fields)",
    "seasonal demand": r"season|how much water|demand|requirement",
    "daily schedule": r"schedul|when to irrigate|each day",
}
_PERIOD_OPTIONS = ["now", "the last 3 months", "the last 12 months", "the whole record"]
#: Intake keys the Studio itself reads (the table loader's arguments), kept through the coercion.
_TABLE_INTAKE = ("value_column", "datetime_column")
_DECISION_TEXT = "What will be decided with the answer?"

__all__ = ["brief_context", "classify_follow_up", "consult", "known_playbooks"]


def known_playbooks() -> dict[str, Any]:
    """The playbooks by id (a broken file is skipped)."""
    from aquascope import playbooks as pbk

    out: dict[str, Any] = {}
    for row in pbk.list_playbooks():
        if "error" in row:
            continue
        try:
            out[row["id"]] = pbk.load(row["id"])
        except pbk.PlaybookError:
            continue
    return out


def _columns(ws: Workspace, tables: dict[str, Any] | None) -> dict[str, list[str]]:
    """The column names of every upload, without pandas when the header line is enough."""
    out: dict[str, list[str]] = {}
    for key, value in (tables or {}).items():
        cols = getattr(value, "columns", None)
        if cols is not None:
            out[key] = [str(c) for c in cols]
    for key, text in ws.tables.items():
        if key in out:
            continue
        head = io.StringIO(text).readline().strip()
        out[key] = [c.strip().strip('"') for c in head.split(",")] if head else []
    return out


def _is_number(text: str) -> bool:
    try:
        float(text.replace(",", ""))
    except ValueError:
        return False
    return True


def _numeric_columns(ws: Workspace, tables: dict[str, Any] | None) -> dict[str, list[str]]:
    """The numeric columns of every upload: from the frame's dtypes when a frame is given, else from the first
    fifty rows of the CSV text (no pandas)."""
    out: dict[str, list[str]] = {}
    for key, value in (tables or {}).items():
        if not hasattr(value, "select_dtypes"):
            continue
        try:
            out[key] = [str(c) for c in value.select_dtypes("number").columns]
        except Exception:  # noqa: BLE001 - an odd frame is not worth a question
            out[key] = []
    for key, text in ws.tables.items():
        if key in out:
            continue
        reader = csv.reader(io.StringIO(text))
        header = [h.strip() for h in (next(reader, None) or [])]
        rows = list(itertools.islice(reader, 50))
        nums: list[str] = []
        for i, name in enumerate(header):
            values = [r[i].strip() for r in rows if i < len(r) and r[i].strip()]
            if values and all(_is_number(v) for v in values):
                nums.append(name)
        out[key] = nums
    return out


def _quick_recon(ws: Workspace, problem: str | None) -> dict[str, Any] | None:
    """A catalog-only reconnaissance for the model's brief; None when it cannot be had."""
    site = ws.site or {}
    if site.get("lat") is None or site.get("lon") is None:
        return None
    try:
        from aquascope.ai_engine.team import _recon_summary
        from aquascope.explore import assess_site

        recon = assess_site(float(site["lat"]), float(site["lon"]), radius_km=RADIUS_KM, problem=problem)
        return _recon_summary(recon) if isinstance(recon, dict) else None
    except Exception as exc:  # noqa: BLE001 - the brief can be taken without it
        ws.event("consultant", "recon_skipped", f"{type(exc).__name__}: {exc}")
        return None


def _intake_schema(pb: Any) -> list[dict[str, Any]]:
    return [{"name": f.name, "type": f.type, "default": f.default, "options": f.options or None, "help": f.help}
            for f in pb.intake]


# ── the keyless questions ───────────────────────────────────────────────────


def _decision_options(pb: Any) -> tuple[str | None, list[str], Any, str]:
    """``(field, options, default, text)`` of the decision question for a playbook: its decision field's (the
    field's own label when it is a purpose, a concern or a use), or the table's for a playbook that has none."""
    field = _DECISION_FIELDS.get(pb.id)
    f = next((x for x in pb.intake if x.name == field), None) if field else None
    if f is not None:
        text = _DECISION_TEXT if field == "decision" else f"{f.label or f.name}?"
        return field, [str(o) for o in f.options], f.default, text
    return None, [o for o, _ in _DECISION_OPTIONS.get(pb.id, [])], None, _DECISION_TEXT


def _set_decision(ws: Workspace, pb: Any, decision: str | None) -> None:
    """The decision on the brief and, when the playbook has a field for it, in the intake."""
    if not decision:
        return
    ws.brief.decision = decision
    field = _DECISION_FIELDS.get(pb.id)
    if field and ws.brief.intake.get(field) is None:
        ws.brief.intake[field] = decision


def _decision_hint(text: str, pb: Any, intake: dict[str, Any]) -> str | None:
    """The decision the text names: the intake hint for the decision field, an option's own words, or the
    words the table gives an option."""
    field, options, _, _ = _decision_options(pb)
    if field and intake.get(field) is not None:
        return str(intake[field])
    words = dict(_DECISION_OPTIONS.get(pb.id, []))
    for option in options:
        pat = words.get(option) or _OPTION_WORDS.get(option)
        if re.search(re.escape(option), text, re.I) or (pat and re.search(pat, text, re.I)):
            return option
    return None


def _gap_questions(ws: Workspace, pb: Any, tables: dict[str, Any] | None) -> list[Question]:
    """What the brief and the site leave open, as questions, at most three: the playbook's fields with no
    default (one of a pair), the decision, the return period of a flood question, the period of a drought
    question, the value column of an upload with two or more numeric columns."""
    b = ws.brief
    out: list[Question] = []
    for f in pb.intake:
        if f.default is not None or b.intake.get(f.name) is not None:
            continue
        alternates = _ALTERNATES.get(f.name, ())
        if any(b.intake.get(a) is not None for a in alternates) or any(q.id in alternates for q in out):
            continue
        text = f.label or f.name
        if f.help:
            text += f" ({f.help})"
        out.append(Question(id=f.name, text=text, options=[str(o) for o in f.options] or None, default=f.default))
    if not b.decision:
        field, options, default, text = _decision_options(pb)
        out.append(Question(id=field or "decision", text=text, options=options or None, default=default))
    if pb.id == "flood_risk" and b.intake.get("return_period") is None:
        f = next((x for x in pb.intake if x.name == "return_period"), None)
        out.append(Question(id="return_period", text=(f.label if f else "Return period (years)"),
                            default=(f.default if f else 100)))
    if pb.id == "drought_status" and not b.period:
        out.append(Question(id="period", text="Which period is the drought question about?", options=_PERIOD_OPTIONS,
                            default="now"))
    if b.intake.get("value_column") is None:
        for key, cols in _numeric_columns(ws, tables).items():
            if len(cols) >= 2:
                out.append(Question(id="value_column", text=f"Which column of {key} holds the values to analyse?",
                                    options=cols, default=cols[0]))
                break
    return out[:MAX_QUESTIONS]


def _rules_questions(pb: Any | None, ws: Workspace, known: dict[str, Any],
                     tables: dict[str, Any] | None = None) -> list[Question]:
    """The gaps as questions; or which playbook, when none matched."""
    if pb is None:
        return [Question(id="playbook", text="Which kind of problem is this? One of: " + ", ".join(sorted(known))
                         + ". Say 'just go' to let the crew decide (a model is needed for that).",
                         options=sorted(known))]
    return _gap_questions(ws, pb, tables)


def _rules_quantities(playbook: str | None, intake: dict[str, Any]) -> list[str]:
    rp = intake.get("return_period")
    table = {
        "flood_risk": (["the trend in annual flood peaks: Sen's slope and the Mann-Kendall p-value"]
                       if intake.get("decision") == "flood trend" else
                       [f"the {rp or 100}-year return level with its interval and the spread between fits"]),
        "ungauged_flow": [f"the {intake.get('statistic') or 'flow'} statistics transferred from donor gauges, "
                          "with a band"],
        "groundwater_decline": ["the trend in groundwater level with Sen's slope and its significance"],
        "drought_status": ["SPI and SPEI at the requested timescales, the current status and the worst on record"],
        "supply_reliability": ["the fraction of days, years and volume the demand is met"],
        "irrigation_feasibility": ["the seasonal crop water demand in mm, m3 and m3/s"],
        "water_quality": ["the water-quality index and the guideline exceedances per parameter"],
    }
    return table.get(playbook or "", [])


# ── the checklist: what the study must know, asked one at a time ─────────────


def _checklist_item(pb: Any, field: str) -> Any:
    return next((i for i in getattr(pb, "checklist", None) or [] if i.field == field), None)


def _state(ws: Workspace, pb: Any, field: str, value: Any, *, answered: bool = False) -> None:
    """A checklist field the client stated: into the intake (and the decision), and counted as known. A
    decision a model already put in words ("size a culvert") stays, unless the client just answered it."""
    b = ws.brief
    b.intake[field] = value
    if field not in b.stated:
        b.stated.append(field)
    if field == _DECISION_FIELDS.get(pb.id) and value is not None and (answered or not b.decision
                                                                         or b.source == "rules"):
        b.decision = str(value)
    if field == _DECISION_FIELDS.get(pb.id) and b.source == "rules":
        b.quantities = _rules_quantities(b.playbook, b.intake)


def _checklist_read(ws: Workspace, pb: Any, text: str, given: dict[str, Any]) -> None:
    """What the client's words already answer, item by item in order (an earlier answer can open or close a
    later item): a field the intake hints or a model read (``given``), else an option whose words the text
    uses."""
    from aquascope import playbooks as pbk

    for item in pb.checklist:
        if item.field in ws.brief.stated or item not in pbk.checklist_open(pb, ws.brief.intake, ws.brief.stated):
            continue
        if given.get(item.field) is not None:
            understood, value = _checklist_value(item, pb, given[item.field])
            if understood:
                _state(ws, pb, item.field, value)
                continue
        opt = next((o for o in item.options if o.match and re.search(o.match, text or "", re.I)), None)
        if opt is not None:
            _state(ws, pb, item.field, opt.value)


def _checklist_value(item: Any, pb: Any, answer: Any) -> tuple[bool, Any]:
    """``(understood, value)`` for a reply to a checklist question: the option it picks, else a number for a
    number field or one of the field's own options by their words."""
    opt = item.pick(answer)
    if opt is not None:
        return True, opt.value
    field = next((f for f in pb.intake if f.name == item.field), None)
    if field is None:
        return False, None
    q = Question(id=item.field, text=item.ask, options=[str(o) for o in field.options] or None)
    value = _coerce_answer(str(answer), q, field, playbook=pb.id)
    if field.type in ("int", "float"):
        from aquascope import playbooks as pbk

        try:
            return True, pbk._coerce(value, field)   # within the field's own bounds, or asked again
        except (TypeError, ValueError, OverflowError):
            return False, None
    if field.options:
        return value in field.options, value
    return bool(str(value).strip()), value


def _next_question(ws: Workspace, pb: Any, tables: dict[str, Any] | None = None) -> None:
    """Leave exactly one question open: the next checklist item the client has not answered, else a gap the
    checklist does not cover (an upload's value column). Past :data:`MAX_TURNS` checklist questions, the rest
    take their defaults and say so."""
    from aquascope import playbooks as pbk

    b = ws.brief
    if b.open_questions:
        return
    fields = {i.field for i in pb.checklist}
    asked = sum(1 for q in b.questions if q.id in fields)
    for item in pbk.checklist_open(pb, b.intake, b.stated):
        default = next((f.default for f in pb.intake if f.name == item.field), None)
        if asked >= MAX_TURNS:
            _state(ws, pb, item.field, default)
            b.assumptions.append(f"{item.ask} {item.label_of(default) or default} (the default)")
            continue
        b.questions.append(Question(id=item.field, text=item.ask, why=item.why,
                                    options=[o.label or str(o.value) for o in item.options] or None,
                                    default=item.label_of(default) or default))
        return
    done = {q.id for q in b.questions}
    rest = [q for q in _gap_questions(ws, pb, tables) if q.id not in fields and q.id not in done]
    if rest:
        b.questions.append(rest[0])
        return
    if b.source == "rules":
        b.quantities = _rules_quantities(b.playbook, b.intake)


def _take_answer(ws: Workspace, q: Question, known: dict[str, Any]) -> None:
    """Where an answered question lands: the playbook, the decision, the period, or the intake."""
    b = ws.brief
    if q.answer is None or q.answer == "":
        return
    pb = known.get(b.playbook) if b.playbook else None
    field = _DECISION_FIELDS.get(pb.id) if pb is not None else None
    item = _checklist_item(pb, q.id) if pb is not None else None
    if item is not None:
        understood, value = _checklist_value(item, pb, q.answer)
        if understood:
            _state(ws, pb, q.id, value, answered=True)
        else:
            q.answer = None      # asked again: the reply named none of the options
        return
    if q.id == "playbook":
        picked = _match_option(str(q.answer), sorted(known))
        if picked in known:
            b.playbook, b.kind = picked, known[picked].problem
        else:
            q.answer = None
        return
    if q.id == "decision" and field != "decision":
        b.decision = str(q.answer)
        return
    if q.id == field:
        b.decision = str(q.answer)
    elif q.id == "period":
        b.period = str(q.answer)
        m = re.search(r"(\d+)\s*months?", str(q.answer))
        scales = b.intake.get("timescales")
        if m and isinstance(scales, list) and int(m.group(1)) not in [int(s) for s in scales if str(s).isdigit()]:
            b.intake["timescales"] = [*scales, int(m.group(1))]
        return
    b.intake[q.id] = q.answer


def _coerce_all(ws: Workspace, known: dict[str, Any], *, keep_unknown: bool) -> None:
    """The intake through the playbook's coercion; unknown fields kept (a question's answer) or dropped
    (a model's or a device's brief), the table loader's keys always kept."""
    from aquascope import playbooks as pbk

    b = ws.brief
    if b.playbook not in known:
        return
    coerced = pbk.coerce_intake(known[b.playbook], b.intake)
    out: dict[str, Any] = {}
    for k, v in b.intake.items():
        if k in coerced:
            continue
        if v is not None and (keep_unknown or k in _TABLE_INTAKE):
            out[k] = v
    for k, v in coerced.items():
        if v is not None:
            out[k] = v
    b.intake = out


# ── the model's brief, and a caller's ───────────────────────────────────────


def _apply_brief(ws: Workspace, obj: dict[str, Any], known: dict[str, Any], *, questions: bool = True) -> None:
    """Copy what a model (or a caller's model) wrote into the brief, kept within what the playbooks and the
    registry know. With ``questions`` False the object's questions are ignored (an answer round)."""
    b = ws.brief
    for key in ("decision", "period", "horizon"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            setattr(b, key, v.strip())
    for key in ("quantities", "constraints", "assumptions"):
        v = obj.get(key)
        if isinstance(v, list):
            items = [str(x) for x in v if isinstance(x, (str, int, float))]
            setattr(b, key, items if questions else list(dict.fromkeys([*getattr(b, key), *items])))
    if isinstance(obj.get("deliverables"), list) and obj["deliverables"]:
        b.deliverables = [str(x) for x in obj["deliverables"]]
    playbook = obj.get("playbook")
    by_problem = {pb.problem: pid for pid, pb in known.items()}
    if isinstance(playbook, str) and playbook in known:
        b.playbook = playbook
        b.kind = known[playbook].problem
    elif isinstance(obj.get("kind"), str):
        b.kind = obj["kind"]
        if b.playbook is None and obj["kind"] in by_problem:
            b.playbook = by_problem[obj["kind"]]
    intake = obj.get("intake")
    if isinstance(intake, dict):
        b.intake.update({k: v for k, v in intake.items() if v is not None})
        _coerce_all(ws, known, keep_unknown=b.playbook not in known)
    if not questions:
        return
    qs: list[Question] = []
    for q in obj.get("questions") or []:
        if not isinstance(q, dict) or not q.get("text"):
            continue
        qid = str(q.get("id") or f"q{len(qs) + 1}")
        if b.intake.get(qid) is not None:
            continue
        opts = q.get("options")
        qs.append(Question(id=qid, text=str(q["text"]),
                           options=[str(o) for o in opts] if isinstance(opts, list) and opts else None,
                           default=q.get("default")))
    b.questions = qs[:MAX_QUESTIONS]
    b.ready = not b.questions and bool(obj.get("ready", True))


def _proposed_brief(proposed: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(proposed, dict) or not isinstance(proposed.get("brief"), dict):
        return None, "device"
    return dict(proposed["brief"]), str(proposed.get("source") or "device")


_TABLE_WORDS = re.compile(r"\b(my|own|this|these|attached|uploaded|upload|csv|table|file|record|data)\b", re.I)


def _note_uploads(ws: Workspace, text: str) -> None:
    """An attached table is a constraint of the brief; when the text does not mention it, an assumption too."""
    for key in ws.tables:
        constraint = f"use the attached table {key}"
        if constraint not in ws.brief.constraints:
            ws.brief.constraints.append(constraint)
        if not _TABLE_WORDS.search(text or ""):
            note = f"The attached table {key} is used where the analysis can take it; the text did not say."
            if note not in ws.brief.assumptions:
                ws.brief.assumptions.append(note)


def brief_context(ws: Workspace, text: str, tables: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """The system prompt and the context the Consultant sends a model for ``text``: the brief's when no brief
    is open yet, the answers' when questions are open."""
    from aquascope.ai_engine.team import choose_playbook, intake_hints

    known = known_playbooks()
    b = ws.brief
    if b.problem and b.open_questions:
        return CONSULTANT_ANSWERS, {
            "reply": text, "problem": b.problem, "playbook": b.playbook, "intake": dict(b.intake),
            "questions": [q.to_dict() for q in b.open_questions],
            "fields": _intake_schema(known[b.playbook]) if b.playbook in known else [],
        }
    playbook, ambiguous = choose_playbook(text)
    playbook = playbook if playbook in known else None
    return CONSULTANT, {
        "problem": text, "site": ws.site,
        "recon": _quick_recon(ws, known[playbook].problem if playbook else None),
        "uploads": _columns(ws, tables),
        "intake_given": {**intake_hints(text, playbook), **{k: v for k, v in b.intake.items() if v is not None}},
        "rules_pick": {"playbook": playbook, "ambiguous": ambiguous},
        "playbooks": [{"id": pb.id, "problem": pb.problem, "title": pb.title, "intake": _intake_schema(pb),
                       **({"checklist": [{"field": i.field, "ask": i.ask,
                                          "values": [o.value for o in i.options]} for i in pb.checklist]}
                          if pb.checklist else {})}
                      for pb in known.values()],
    }


def _open(ws: Workspace, model: Model | None, text: str, tables: dict[str, Any] | None,
          proposed: dict[str, Any] | None = None) -> Message:
    from aquascope.ai_engine.team import choose_playbook, intake_hints

    known = known_playbooks()
    b = ws.brief
    preset = {k: v for k, v in b.intake.items() if v is not None}    # the caller's own intake (--intake)
    b.problem = text
    playbook, ambiguous = choose_playbook(text)
    playbook = playbook if playbook in known else None
    hints = intake_hints(text, playbook)
    for k, v in hints.items():
        b.intake.setdefault(k, v)
    b.playbook = playbook
    b.kind = known[playbook].problem if playbook else None
    from aquascope.playbooks import companions

    b.kinds = [k for k in [playbook, *[c for c, _ in companions(text, playbook)]] if k]
    b.source = "rules"
    ws.event("consultant", "keywords", f"rules pick {playbook or 'nothing'}" + (" (ambiguous)" if ambiguous else ""))

    obj, source = _proposed_brief(proposed)
    if obj is None and model:
        system, context = brief_context(ws, text, tables)
        obj, source = model.call_json("consultant", system, context), "model"
    if obj:
        _apply_brief(ws, obj, known)
        b.source = source
        pb = known.get(b.playbook) if b.playbook else None
        if source != "model":
            # A caller's model is read like the crew's, then the gaps it left are asked as the rules would.
            if pb is not None and not b.decision:
                _set_decision(ws, pb, _decision_hint(text, pb, b.intake))
            asked = {q.id for q in b.questions}
            b.questions = [*b.questions, *[q for q in _rules_questions(pb, ws, known, tables)
                                           if q.id not in asked]][:MAX_QUESTIONS]
            b.ready = not b.questions
        ws.event("consultant", "brief", f"{source}: playbook {b.playbook or 'none'}, {len(b.questions)} question(s)")
    else:
        pb = known.get(playbook) if playbook else None
        if pb is not None and not b.decision:
            _set_decision(ws, pb, _decision_hint(text, pb, b.intake))
        if pb is not None and not b.period and pb.id == "drought_status":
            m = _PERIOD.search(text)
            b.period = m.group(0).lower() if m else None
        b.questions = _rules_questions(pb, ws, known, tables)
        b.quantities = b.quantities or _rules_quantities(playbook, b.intake)
        if pb is not None:
            asked = {q.id for q in b.questions}
            for f in pb.intake:
                if f.default is not None and b.intake.get(f.name) is None and f.name not in asked:
                    b.assumptions.append(f"{f.label or f.name}: {f.default} (the playbook's default)")
        b.ready = not b.questions
        ws.event("consultant", "brief", f"rules: playbook {playbook or 'none'}, {len(b.questions)} question(s)")
    pb = known.get(b.playbook) if b.playbook else None
    if pb is not None and pb.checklist:
        # The checklist asks, one question at a time; the text (and a model's reading of it) answers first.
        labels = {(f.label or f.name) for f in pb.intake if _checklist_item(pb, f.name) is not None}
        b.assumptions = [a for a in b.assumptions if a.split(":")[0] not in labels]
        b.questions = []
        given = dict(intake_hints(text, b.playbook))
        if obj and isinstance(obj.get("intake"), dict):
            given.update({k: v for k, v in obj["intake"].items() if v is not None})
        given.update(preset)
        for k, v in preset.items():
            item = _checklist_item(pb, k)
            if item is not None and _checklist_value(item, pb, v)[0]:
                _state(ws, pb, k, _checklist_value(item, pb, v)[1])     # the caller said so: never asked
        _checklist_read(ws, pb, text, given)
        _next_question(ws, pb, tables)
        b.ready = not b.open_questions
        ws.event("consultant", "checklist", f"{len(b.stated)} known from the text, "
                 f"{len(b.open_questions)} to ask")
    _note_uploads(ws, text)
    return _message(ws)


def _proceed(ws: Workspace, known: dict[str, Any]) -> None:
    """"just go": every open question takes its default, each one listed as an assumption."""
    b = ws.brief
    for q in b.open_questions:
        q.answer = q.default if q.default is not None else ""
        if q.answer != "":
            b.assumptions.append(f"{q.text.split(' (')[0]}: {q.answer} (the default, the client asked to proceed)")
        _take_answer(ws, q, known)
    pb = known.get(b.playbook) if b.playbook else None
    if pb is not None and pb.checklist:
        from aquascope import playbooks as pbk

        for item in pbk.checklist_open(pb, b.intake, b.stated):
            default = next((f.default for f in pb.intake if f.name == item.field), None)
            _state(ws, pb, item.field, default)
            b.assumptions.append(f"{item.ask} {item.label_of(default) or default} (the default)")
        if b.source == "rules":
            b.quantities = _rules_quantities(b.playbook, b.intake)
    b.assumptions.append("The client asked to proceed on the defaults.")
    _coerce_all(ws, known, keep_unknown=True)
    b.ready = True


def _answer(ws: Workspace, model: Model | None, text: str, proposed: dict[str, Any] | None = None) -> Message:
    from aquascope.ai_engine.team import intake_hints

    known = known_playbooks()
    b = ws.brief
    open_qs = b.open_questions
    obj, source = _proposed_brief(proposed)
    if obj is not None:
        # A caller's model read the reply: what it wrote answers the questions it covers.
        _apply_brief(ws, obj, known, questions=False)
        b.source = source
        field = _DECISION_FIELDS.get(b.playbook or "")
        for q in open_qs:
            if b.intake.get(q.id) is not None:
                q.answer = b.intake[q.id]
            elif q.id in ("decision", field) and b.decision:
                q.answer = b.decision
            elif q.id == "period" and b.period:
                q.answer = b.period
        open_qs = b.open_questions
    if _PROCEED.match(text):
        _proceed(ws, known)
        ws.event("consultant", "answers", "proceed on defaults")
        return _message(ws)

    reply = None
    if model and open_qs:
        system, context = brief_context(ws, text)
        reply = model.call_json("consultant", system, context)
    if reply:
        answers = reply.get("answers") if isinstance(reply.get("answers"), dict) else {}
        for q in open_qs:
            if answers.get(q.id) is not None:
                q.answer = answers[q.id]
        if isinstance(reply.get("intake"), dict):
            b.intake.update({k: v for k, v in reply["intake"].items() if v is not None})
        if isinstance(reply.get("assumptions"), list):
            b.assumptions += [str(a) for a in reply["assumptions"]]
        if reply.get("ready") is True:
            for q in b.open_questions:
                q.answer = q.default if q.default is not None else ""
    elif open_qs and text.strip():
        fields = {f.name: f for f in known[b.playbook].intake} if b.playbook in known else {}
        hints = {k: v for k, v in intake_hints(text, b.playbook).items() if not fields or k in fields}
        for q in open_qs:
            if q.id in hints:
                q.answer = hints[q.id]
            elif any(a in hints for a in _ALTERNATES.get(q.id, ())):
                q.answer = ""      # the pair's other half was given
        parts = [p.strip() for p in _SPLIT.split(text) if p.strip()]
        if len(parts) == len(open_qs):
            # one part per question, in the order they were asked
            pairs = list(zip(open_qs, parts))
        else:
            rest = [q for q in open_qs if q.answer is None]
            pairs = list(zip(rest, parts if len(rest) > 1 else [text.strip()]))
        for q, part in pairs:
            if q.answer is None:
                q.answer = _coerce_answer(part, q, fields.get(q.id), playbook=b.playbook)
        for k, v in hints.items():
            b.intake.setdefault(k, v)
    elif not open_qs:
        for k, v in intake_hints(text, b.playbook).items():
            b.intake[k] = v      # no question open: the text changes the brief

    had_playbook = b.playbook
    for q in b.questions:
        _take_answer(ws, q, known)
    if b.playbook and b.playbook != had_playbook and b.playbook in known:
        # The playbook was just chosen: the problem text is read again with it, and its gaps asked once.
        pb = known[b.playbook]
        for k, v in intake_hints(b.problem, b.playbook).items():
            b.intake.setdefault(k, v)
        if not b.decision:
            _set_decision(ws, pb, _decision_hint(b.problem, pb, b.intake))
        if pb.id == "drought_status" and not b.period:
            m = _PERIOD.search(b.problem)
            b.period = m.group(0).lower() if m else None
        b.quantities = b.quantities or _rules_quantities(b.playbook, b.intake)
        if pb.checklist:
            _checklist_read(ws, pb, b.problem, intake_hints(b.problem, b.playbook))
        else:
            b.questions += _gap_questions(ws, pb, None)
    _coerce_all(ws, known, keep_unknown=True)
    pb = known.get(b.playbook) if b.playbook else None
    if pb is not None and pb.checklist:
        if reply and isinstance(reply.get("intake"), dict):
            for k, v in reply["intake"].items():
                if v is not None and k not in b.stated and _checklist_item(pb, k) is not None:
                    understood, value = _checklist_value(_checklist_item(pb, k), pb, v)
                    if understood:
                        _state(ws, pb, k, value)
        # an answer can open an item the first message already answered ("the 1 in 50 flood", then "insurance")
        _checklist_read(ws, pb, b.problem, intake_hints(b.problem, b.playbook))
        _next_question(ws, pb)
    b.ready = not b.open_questions
    ws.event("consultant", "answers", f"{len(open_qs) - len(b.open_questions)} answered, "
             f"{len(b.open_questions)} open")
    return _message(ws)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _coerce_answer(part: str, q: Question, field: Any, *, playbook: str | None = None) -> Any:
    """A free-text answer as the field wants it: the number in "200 years" for an int field, the option a word
    points at (a decision by the words that name it: "a licence", "screening"), else the text."""
    ftype = getattr(field, "type", None)
    if ftype in ("int", "float") or (field is None and q.id == "return_period"):
        m = _NUMBER.search(part)
        if m:
            value = float(m.group(0))
            return int(value) if (ftype == "int" or ftype is None) and value.is_integer() else value
        return part
    if q.options:
        picked = _match_option(part, q.options)
        if picked in q.options or q.id not in ("decision", *_DECISION_FIELDS.values()):
            return picked
        words = {**_OPTION_WORDS, **dict(_DECISION_OPTIONS.get(playbook or "", []))}
        for option in q.options:
            pat = words.get(option)
            if pat and re.search(pat, part, re.I):
                return option
        return picked
    return part.strip()


def _match_option(text: str, options: list[str] | None) -> Any:
    if not options:
        return text.strip()
    low = text.strip().lower()
    for o in options:
        if low == str(o).lower():
            return o
    for o in options:
        if str(o).lower() in low or low in str(o).lower():
            return o
    return text.strip()


def _message(ws: Workspace) -> Message:
    b = ws.brief
    site = ws.site or {}
    where = f"{site.get('lat')}, {site.get('lon')}" if site else "the site"
    if len(b.open_questions) == 1 and (b.open_questions[0].why or b.open_questions[0].options):
        q = b.open_questions[0]
        lines = [q.text]
        if q.why:
            lines.append(f"({q.why})")
        lines += [f"  {i}. {o}" for i, o in enumerate(q.options or [], 1)]
        dflt = f" ({q.default})" if q.default is not None else ""
        lines.append(f"Pick one or answer in your own words; 'just go' takes the default{dflt}.")
        return ws.say("consultant", "\n".join(lines), kind="questions",
                      payload={"questions": [q.to_dict()], "brief": b.to_dict()})
    if b.open_questions:
        lines = ["Before the crew plans, a few things the text does not say:"]
        for i, q in enumerate(b.open_questions, 1):
            opts = f" [{', '.join(q.options)}]" if q.options else ""
            dflt = f" (default {q.default})" if q.default is not None else ""
            lines.append(f"{i}. {q.text}{opts}{dflt}")
        lines.append("Answer in order, or say 'just go' to proceed on the defaults.")
        text = "\n".join(lines)
        return ws.say("consultant", text, kind="questions",
                      payload={"questions": [q.to_dict() for q in b.open_questions], "brief": b.to_dict()})
    bits = [f"Brief: {b.decision or b.problem} at {where}"]
    if b.playbook:
        bits.append(f"playbook {b.playbook}")
    if b.intake:
        bits.append("intake " + ", ".join(f"{k}={v}" for k, v in b.intake.items() if v is not None))
    if b.assumptions:
        bits.append("assumed: " + "; ".join(b.assumptions[-3:]))
    return ws.say("consultant", "; ".join(bits) + ".", kind="brief", payload={"brief": b.to_dict()})


def consult(ws: Workspace, model: Model | None, text: str, *, site: dict[str, float] | None = None,
            tables: dict[str, Any] | None = None, proposed: dict[str, Any] | None = None) -> Message:
    """Take the client's message: the first one opens the brief, later ones answer its questions.

    Returns the Consultant's message (``kind`` ``questions`` while questions are
    open, ``brief`` when the crew may proceed). ``site`` sets the workspace's
    site; ``tables`` are the uploads as DataFrames (their column names go into
    the model's context; the Scout reads the tables themselves). ``proposed``
    is ``{"brief": {...}, "source": "device"}``: a brief a caller's own model
    wrote from the text, merged with the coercion a model reply gets (the
    intake through ``coerce_intake``, unknown fields dropped) before the gaps
    are asked.
    """
    if text:
        ws.say("user", text)
    if site:
        ws.site = {"lat": float(site["lat"]), "lon": float(site["lon"])}
    if not ws.brief.problem:
        return _open(ws, model, text, tables, proposed)
    return _answer(ws, model, text, proposed)


def classify_follow_up(ws: Workspace, model: Model | None, text: str) -> dict[str, Any]:
    """A follow-up after the report: ``{"kind": "question", "answer"}`` or ``{"kind": "change", "intake",
    "request"}``. Keyless, a change is recognised by an intake the text states (a return period, a
    statistic, a crop), by change words, or by a phrase the Methodologist's keyless rules cover (another
    gauge, the donors, the flow duration curve, a trend, drought indices, baseflow); anything else is
    answered from the report."""
    from aquascope.ai_engine.team import intake_hints

    b = ws.brief
    report = ws.report or {}
    if model:
        results = [{"id": r.get("id"), "tool": r.get("tool"), "ok": r.get("ok"), "result": compact(r.get("result"))}
                   for r in (ws.run or {}).get("results") or []]
        obj = model.call_json("consultant", CONSULTANT_FOLLOW_UP, {
            "follow_up": text, "problem": b.problem, "playbook": b.playbook, "intake": dict(b.intake),
            "report": {"answer": report.get("answer"), "key_numbers": report.get("key_numbers"),
                       "not_established": report.get("not_established")},
            "results": results,
        })
        if obj and obj.get("kind") in ("question", "change"):
            if obj["kind"] == "question":
                return {"kind": "question", "answer": str(obj.get("answer") or report.get("answer") or "")}
            intake = obj.get("intake") if isinstance(obj.get("intake"), dict) else {}
            return {"kind": "change", "intake": {k: v for k, v in intake.items() if v is not None},
                    "request": str(obj.get("request") or text)}
    from aquascope.studio.roles.methodologist import FOLLOW_UP_RULES

    known = known_playbooks()
    fields = {f.name for f in known[b.playbook].intake} if b.playbook in known else None
    hints = intake_hints(text, b.playbook)
    changed = {k: v for k, v in hints.items() if b.intake.get(k) != v and (fields is None or k in fields)}
    if changed or _CHANGE.search(text) or any(re.search(pat, text, re.I) for pat, _ in FOLLOW_UP_RULES):
        return {"kind": "change", "intake": changed, "request": text}
    lines = [report.get("answer") or "The study produced no answer."]
    for kn in (report.get("key_numbers") or [])[:8]:
        lines.append(f"{kn.get('label')}: {kn.get('value')} {kn.get('unit') or ''} (step {kn.get('step')})".strip())
    return {"kind": "question", "answer": "\n".join(lines)}
