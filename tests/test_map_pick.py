"""The model picks, the code writes (#590, round 3): the shared schema check, and every recorded on-device reply
replayed through the new path of Explain, Scout, Review my model results and the What-if reading."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aquascope import map_explain as me
from aquascope import map_pick as mp
from aquascope import map_scout as ms
from aquascope import mesh_review as mr
from aquascope import whatif_map as wm

FIXTURE = Path(__file__).parent / "fixtures" / "ondevice_replies_590.json"
RECORDS = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]


# ── the schema check ────────────────────────────────────────────────────────


SCHEMA = {"type": "object", "required": ["pins"], "additionalProperties": False, "properties": {
    "pins": {"type": "array", "minItems": 1, "maxItems": 2, "items": {
        "type": "object", "required": ["id", "variant"], "additionalProperties": False,
        "properties": {"id": mp.choice(["r1", "r2"]), "variant": mp.choice(mp.VARIANTS)}}}}}


def test_a_reply_that_fits_passes_as_it_is():
    data, errors = mp.pick('```json\n{"pins": [{"id": "r2", "variant": "c"}]}\n```', SCHEMA)
    assert errors == [] and data == {"pins": [{"id": "r2", "variant": "c"}]}


@pytest.mark.parametrize("reply, why", [
    ("no JSON here", "the model did not reply with JSON"),
    ('{"pins": [{"id": "r3", "variant": "a"}]}', "reply.pins[0].id 'r3' is not one of r1, r2"),
    ('{"pins": [{"id": "r1", "variant": "a", "title": "Two cells"}]}', "has title, which the schema does not hold"),
    ('{"pins": [{"id": "r1"}]}', "reply.pins[0] has no variant"),
    ('{"pins": []}', "reply.pins has fewer than 1 items"),
    ('{"pins": [{"id": "r1", "variant": "a"}, {"id": "r2", "variant": "a"}, {"id": "r1", "variant": "b"}]}',
     "reply.pins has more than 2 items"),
    ('{"pins": "r1"}', "reply.pins is not array"),
    ('[]', "the model did not reply with JSON"),
    ('"pins"', "the model did not reply with JSON"),
])
def test_anything_outside_the_schema_is_refused(reply, why):
    data, errors = mp.pick(reply, SCHEMA)
    assert data is None and any(why in e for e in errors), errors


def test_the_rest_of_the_subset():
    assert mp.schema_errors(3, {"type": "integer", "minimum": 1, "maximum": 2}) == ["reply is over 2"]
    assert mp.schema_errors("abc", {"type": "string", "maxLength": 2}) == ["reply is longer than 2 characters"]
    assert mp.schema_errors(None, {"type": ["number", "null"]}) == []
    assert mp.schema_errors(True, {"type": "number"}) == ["reply is not number"]
    assert mp.schema_errors([1, 1], {"type": "array", "uniqueItems": True}) == ["reply holds an item twice"]
    assert mp.schema_errors("x", {"const": "y"}) == ["reply 'x' is not 'y'"]


# ── every recorded reply, replayed (check 3, 2026-10-26) ────────────────────


def _strings(reply):
    """The free text a reply holds: the words that must never reach the page (a short title such as "Deepest
    water" may be the package's own check name, so only sentences count)."""
    data = mp.reply_json(reply)
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("title", "text", "reason") and isinstance(v, str) and len(v.strip()) > 30:
                    out.append(v.strip())
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(data)
    return out


def _status_facts(month, shares, areas):
    """Status facts as aquascope.map_explain.view_facts writes them: ``shares`` (below, above) in %, ``areas``
    {class: (share %, size km2, lat, lon)} in the order the package writes them."""
    when = ms.month_words(month)
    facts = []
    for side, value in zip(("below", "above"), shares):
        facts.append(me._fact("status_share", "status", f"Mapped land in view {side} or much {side} normal", value,
                              "%", f"{round(value)}%", when=when, source="status", tone="#000",
                              extra={"side": side}))
    for k in ("much_above", "much_below", "above", "below"):
        if k not in areas:
            continue
        share, size, lat, lon = areas[k]
        label = me._CLASS_LABEL[k]
        facts.append(me._fact("status_class", "status", f"{label.capitalize()} means", me._CLASS_BOUND[k], "",
                              me._CLASS_BOUND[k], when=when, source="status", extra={"class": k}))
        facts.append(me._fact("status_area", "status", f"Largest area {label} in view", share, "% of the mapped land",
                              f"{round(share)}%", place={"lat": lat, "lon": lon}, when=when, source="status",
                              tone=me.CLASS_COLOURS[k][0],
                              extra={"class": k, "colour_words": me.CLASS_COLOURS[k][1], "cells": 40, "spots": [],
                                     "pinnable": True}))
        facts.append(me._fact("status_size", "status", "Its size", size, "km²", f"{size:,.0f} km²",
                              place={"lat": lat, "lon": lon}, when=when, source="status", extra={"class": k}))
    for i, f in enumerate(facts, 1):
        f["id"] = f"f{i}"
    return facts


# The views of check 3, rebuilt from what the page showed (the fact ids line up with the recorded replies).
VIEWS = {
    # South America: 58% below or much below; the largest area much below 14% (870,000 km2), below 15% (980,000).
    "explain-new-south-america": _status_facts("2026-09", (58, 3), {"much_below": (14, 870000, -5, -40),
                                                                    "below": (15, 980000, -30, -70)}),
    "explain-new-south-america-2": _status_facts("2026-09", (58, 3), {"much_below": (14, 870000, -5, -40),
                                                                      "below": (15, 980000, -30, -70)}),
    # US Midwest: the largest area much below 11% (76,000 km2), below 23% (160,000 km2): 23% is the larger.
    "explain-new-us-midwest": _status_facts("2026-09", (40, 5), {"much_below": (11, 76000, 45, -100),
                                                                 "below": (23, 160000, 30, -70)}),
    # Bangladesh: below or much below 27%, the largest area below normal 22%, its size 340,000 km2. The reply's
    # f10 and f11 are that area and its size.
    "explain-bangladesh-1": None,
}
BBOX = [-180.0, -85.0, 180.0, 85.0]


def _bangladesh():
    facts = _status_facts("2026-09", (27, 34), {"much_above": (9, 60000, 40, 120), "above": (25, 140000, 5, 60),
                                                "below": (22, 340000, 25, 90)})
    by_id = {f["id"]: f for f in facts}
    assert by_id["f10"]["kind"] == "status_area" and by_id["f11"]["kind"] == "status_size"
    return facts


def _facts_for(tag):
    if tag == "explain-bangladesh-1":
        return _bangladesh()
    return VIEWS.get(tag) or _bangladesh()


@pytest.mark.parametrize("rec", [r for r in RECORDS if r["role"] == "explain"], ids=lambda r: r["tag"])
def test_explain_replies_reach_the_page_only_as_the_packages_words(rec):
    facts = _facts_for(rec["tag"])
    for reply in rec["replies"]:
        out = me.check_model_notes(reply, facts, BBOX)
        # Each recorded reply is in the free-text shape: refused whole, said in one line, nothing of it shown.
        assert out["notes"] == [] and len(out["refused"]) == 1
        assert out["refused"][0].startswith("the model's reply was refused: ")
        texts = " ".join(n["title"] + " " + n["text"] for n in out["notes"])
        assert not any(s in texts for s in _strings(reply))


def _explain_intent(reply, facts):
    """A recorded reply's intent in the round-3 shape: the package's note that holds each note's first cited fact
    with a place (or, failing that, any it cites), in the reply's order, the rules' variant."""
    choices = me.note_choices(facts)
    picks = []
    for note in (mp.reply_json(reply) or {}).get("notes") or []:
        cited = [str(x) for x in note.get("facts") or []]
        hit = next((c["id"] for fid in cited for c in choices
                    if c["facts"][0]["id"] == fid), None) or next(
            (c["id"] for fid in cited for c in choices if any(f["id"] == fid for f in c["facts"])), None)
        if hit and hit not in [p["note"] for p in picks]:
            picks.append({"note": hit, "variant": "b"})
    return {"notes": picks}


@pytest.mark.parametrize("tag", ["explain-new-south-america", "explain-new-south-america-2", "explain-new-us-midwest",
                                 "explain-bangladesh-1"])
def test_explain_intents_read_right(tag):
    rec = next(r for r in RECORDS if r["tag"] == tag)
    facts = _facts_for(tag)
    intent = _explain_intent(rec["replies"][0], facts)
    assert intent["notes"]
    out = me.check_model_notes(intent, facts, BBOX)
    assert out["refused"] == [] and out["notes"]
    texts = [n["text"] for n in out["notes"] if "note" in n]
    for text in texts:
        # Check 3, escape 3: a largest area's share said as the share of all land. Escape 4: "A smaller area",
        # "Another ... area", "{size} of the mapped land".
        assert not re.search(r"\b(?:a smaller|another|a larger)\b", text, re.I), text
        assert "km² of the mapped land" not in text and "of the mapped land is" not in text
        for m in re.finditer(r"(\S+) of the mapped land in view is", text):
            assert m.group(1).endswith("%") and text[:m.start()].endswith("In all, "), text
    if tag.startswith("explain-new-south-america"):
        assert any("The largest area much below normal in view covers 14% of the mapped land in view (870,000 km²)."
                   in t for t in texts), texts
        assert any("In all, 58% of the mapped land in view is below or much below normal." in t for t in texts)
    if tag == "explain-new-us-midwest":
        # The comparison, when a variant makes one, is the code's: 23% is more than 11%.
        c = next(c for c in me.note_choices(facts) if c["facts"][0]["class"] == "below")
        assert "covers 23% of the mapped land in view, more than the largest area much below normal (11%" in \
            c["variants"]["c"][1]
    if tag == "explain-bangladesh-1":
        assert any("covers 22% of the mapped land in view (340,000 km²)" in t for t in texts), texts


def _scout_findings(reply):
    """Findings with the ids a recorded Scout reply named, as the package makes them (a reach, an area, a cell, a
    gauge with models that miss it), so the ids match."""
    data = mp.reply_json(reply) or {}
    ids = [str(x) for x in data.get("order") or []]
    out = []
    for i, fid in enumerate(ids):
        kind = fid.split(":")[0]
        if kind == "floods_ahead":
            f = ms.floods_ahead_findings([{"type": "Feature", "geometry": {"type": "Point", "coordinates": [131, -12]},
                                           "properties": {"river_id": i, "rp": 2, "peak": 6.86, "q2": 5.75,
                                                          "day": "2026-10-30", "share": 13 / 51}}],
                                         {"issue_date": "2026-10-25"})[0]
        elif kind == "models_disagree":
            f = ms.models_disagree_findings([{"source": "poland_imgw", "station_id": "149190160", "lat": 50, "lon": 20,
                                              "model": "glofas", "label": "GloFAS", "is_best": True, "kge": -0.6,
                                              "pbias": 20.0, "n_days": 4000, "mean_gauge": 30.0, "area_ratio": 1.0,
                                              "start": "2001-01-01", "end": "2024-12-31", "name": "Rudze"}])[0]
        elif kind == "floods_past":
            f = ms.floods_past_findings([[220, 330, 12, 0]], "2026-01")[0]
        else:
            import numpy as np

            classes = np.zeros((72, 144), dtype=np.uint8)
            classes[30:40, 60:70] = 1
            f = ms.status_findings(classes, "2026-09")[0]
        out.append({**f, "id": fid})
    return out


@pytest.mark.parametrize("rec", [r for r in RECORDS if r["role"] == "scout" and r["replies"]], ids=lambda r: r["tag"])
def test_scout_replies_reach_the_page_only_as_the_packages_words(rec):
    reply = rec["replies"][0]
    fs = _scout_findings(reply)
    res = ms.apply_wording(fs, reply, by="device")
    assert len(res["refused"]) == 1 and res["refused"][0].startswith("the model's reply was refused: ")
    assert all(f["by"] == "rules" for f in res["findings"])
    texts = " ".join(f["title"] + " " + f["reason"] for f in res["findings"])
    assert not any(s in texts for s in _strings(reply))
    # Check 3, escape 5 (Australia): no "severe" and no "substantial chance" for 13 of 51 members at the 2-year flow.
    assert "severe" not in texts.lower() and "substantial" not in texts.lower()
    # The intent in the round-3 shape: the reply's order, each in every variant; all pass, all the package's words.
    for v in mp.VARIANTS:
        intent = {"pins": [{"id": f["id"], "variant": v} for f in fs]}
        res = ms.apply_wording(fs, intent, by="device")
        assert res["refused"] == [] and [f["id"] for f in res["findings"]] == [f["id"] for f in fs]
        words = " ".join(f["title"] + " " + f["reason"] for f in res["findings"]).lower()
        assert "severe" not in words and "substantial" not in words and "{" not in words


def _review_findings():
    """The HEC-RAS Portsmouth review's findings as the package wrote them on 2026-10-26 (r1 to r8)."""
    def f(i, check, severity, title, reason, facts):
        return {"id": f"r{i}", "rank": i, "check": check, "label": mr.CHECKS[check], "severity": severity,
                "title": title, "reason": reason, "rules_title": title, "rules_reason": reason, "facts": facts,
                "by": "rules", "lat": 36.8, "lon": -76.3, "cells": [i], "value": None}
    deep = [17.1, 16.5, 16.4, 9.82, 8.97]
    out = [
        f(1, "flicker", "warning", "Wet and dry 5 times",
          "This cell goes from wet to dry or back 5 times in 109 steps; 2 cells flicker in all. Often a sign of a "
          "time step too long for the mesh or of a shallow, flat area at the edge of the flow.", []),
        f(2, "flicker", "warning", "Wet and dry 4 times", "Another cell that flickers: 4 changes in 109 steps.", []),
        f(3, "tiny", "warning", "Only a film of water: 0.0269 m",
          "This cell never gets deeper than 0.0269 m, under 0.03 m. 8 cells in all are like it, 0.13% of those that "
          "get wet: rain on the grid, a terrain or a tolerance problem.", []),
    ]
    for k, d in enumerate(deep, 4):
        out.append(f(k, "deepest", "info", f"Deepest water: {d} m",
                     f"The deepest water in this part of the mesh: {d} m at 12 Oct 03:00. Worth a look if it is not "
                     "a channel or a pit you expect.", []))
    return out


@pytest.mark.parametrize("rec", [r for r in RECORDS if r["role"] == "review"], ids=lambda r: r["tag"])
def test_review_replies_reach_the_page_only_as_the_packages_words(rec):
    reply = rec["replies"][0]
    fs = _review_findings()
    out = mr.apply_words(fs, reply, by="device")
    assert len(out["refused"]) == 1 and out["refused"][0].startswith("the model's reply was refused: ")
    assert [f["id"] for f in out["findings"]] == [f["id"] for f in fs]
    texts = " ".join(f["title"] + " " + f["reason"] for f in out["findings"])
    assert not any(s in texts for s in _strings(reply))
    # The intent (the reply's order), in the round-3 shape: every pick passes; "Two cells", "Check if this depth is
    # expected" and "The deepest water is X m" are gone, the rules' "not" and "in this part of the mesh" stay.
    order = (mp.reply_json(reply) or {}).get("order")
    if not order:
        return
    for v in mp.VARIANTS:
        res = mr.apply_words(fs, {"pins": [{"id": i, "variant": v} for i in order]}, by="device")
        assert res["refused"] == [] and res["left_out"] == [] and [f["id"] for f in res["findings"]] == order
        words = " ".join(f["title"] + " " + f["reason"] for f in res["findings"])
        assert "Two cells" not in words and "expected for the area" not in words
        assert "The deepest water is" not in words
        if v != "c":
            assert "if it is not a channel or a pit you expect" in words
        assert "2 cells flicker in all" in words and "in this part of the mesh" in words


@pytest.mark.parametrize("rec", [r for r in RECORDS if r["role"] == "whatif"], ids=lambda r: r["tag"])
def test_whatif_replies_are_refused_whole_and_never_run(rec):
    line = rec["request"]
    out = wm.parse_scenario_reply(rec["replies"][0], line)
    # The four-number shape is refused: no number of the model's runs (check 3, escape 2: "7.5 l/s" as 7.5 m³/s).
    assert out["scenario"] is None and out["errors"][0].startswith("the reply was refused: ")


def test_the_recorded_numbers_are_the_words_own_or_refused_by_the_schema():
    # For each recorded line, the numbers the model may name are exactly the words' own.
    want = {
        "whatif-twenty-litres": ["twenty", "20", "none"],
        "whatif-summers-three": ["three", "3", "none"],
        "whatif-much-wetter": ["none"],
        "whatif-new-winters-rain": ["two", "2", "15", "none"],
        "whatif-new-brewery": ["7.5", "one and a half", "1.5", "none"],
        "whatif-300-litres": ["300", "none"],
    }
    for rec in (r for r in RECORDS if r["role"] == "whatif"):
        enum = wm.scenario_schema(rec["request"])["properties"]["clauses"]["items"]["properties"]["number"]["enum"]
        assert enum == want[rec["tag"]], rec["tag"]


# ── replies quoted in the first two checks (2026-10-25 and 10-26) ───────────


@pytest.mark.parametrize("text", [
    "{f1} of the land is significantly below normal.",            # "96% ... significantly below normal" (Europe)
    "The river is mapped at {f4} above normal.",                  # a share of land as a flow (Bangladesh)
    "Here the flow is at least {f1} below normal.",               # (Africa)
    "{f4} of the mapped land is below normal. This area is pinned, unnamed.",
    "54% (54%) of the land.",
])
def test_quoted_explain_wordings_cannot_reach_a_note(text):
    facts = _bangladesh()
    out = me.check_model_notes({"notes": [{"facts": ["f4"], "title": "Dry", "text": text}]}, facts, BBOX)
    assert out["notes"] == [] and out["refused"][0].startswith("the model's reply was refused")


def test_quoted_scout_and_review_wordings_cannot_reach_a_pin():
    fs = _scout_findings('{"order": ["models_disagree:poland_imgw/149190160", "status:much_below:2026-09:1"]}')
    reply = {"order": [f["id"] for f in fs], "notes": [
        {"id": fs[0]["id"], "title": "Rudze", "reason": "poor agreement with a gauge's water flow, raising "
                                                         "questions about the gauge's measurement"},
        {"id": fs[1]["id"], "title": "{place} in January 2026", "reason": "Rivers in {area}."}]}
    res = ms.apply_wording(fs, reply, by="device")
    assert all(f["by"] == "rules" for f in res["findings"]) and len(res["refused"]) == 1
    rs = _review_findings()
    reply = {"order": ["r4"], "notes": [{"id": "r4", "title": "Deepest", "reason": "8 cells ... (< 0.0269 m); "
                                                                                    "investigate if it's expected"}]}
    out = mr.apply_words(rs, reply, by="device")
    assert all(f["by"] == "rules" for f in out["findings"]) and len(out["refused"]) == 1
