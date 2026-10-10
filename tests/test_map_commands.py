"""Talk to the map (#561): the keyless phrase grammar, the action checks, the gazetteer and the faces."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

from aquascope import cli
from aquascope import map_commands as mc

TODAY = "2026-10-10"
ROOT = Path(__file__).resolve().parents[1]


def acts(text: str) -> list[dict]:
    res = mc.parse_command(text, today=TODAY)
    assert res["matched"], f"{text!r} was not understood: {res.get('unknown')} {res['errors']}"
    return res["actions"]


# ── the grammar: a table of phrases and the actions they become ────────────

PHRASES = [
    ("show the Rhine", [
        {"type": "fly_to", "place": "Rhine", "river": True},
        {"type": "highlight_river", "place": "Rhine", "direction": "both", "optional": True}]),
    ("go to Bangladesh", [{"type": "fly_to", "place": "Bangladesh"}]),
    ("Fly to Kingston upon Thames", [{"type": "fly_to", "place": "Kingston upon Thames"}]),
    ("take me to Taipei please", [{"type": "fly_to", "place": "Taipei"}]),
    ("go to 23.7, 90.4", [{"type": "fly_to", "center": [23.7, 90.4], "zoom": 8.0}]),
    ("show the whole world", [{"type": "fly_to", "region": "world"}]),
    ("zoom in", [{"type": "fly_to", "zoom_by": 2.0}]),
    ("zoom out a lot", [{"type": "fly_to", "zoom_by": -3.5}]),
    ("September 2023", [{"type": "set_time", "date": "2023-09-15", "step": "month"}]),
    ("show 14 March 2019", [{"type": "set_time", "date": "2019-03-14", "step": "day"}]),
    ("2021-07-14", [{"type": "set_time", "date": "2021-07-14", "step": "day"}]),
    ("last July", [{"type": "set_time", "date": "2026-07-15", "step": "month"}]),
    ("yesterday", [{"type": "set_time", "date": "2026-10-09", "step": "day"}]),
    ("last month", [{"type": "set_time", "date": "2026-09-15", "step": "month"}]),
    ("play the last 5 years", [{"type": "set_time", "date": "2021-10-15", "step": "month",
                                "range": {"from": "2021-10-15", "to": "2026-10-10"}, "playing": True}]),
    ("play from 2010 to 2012", [{"type": "set_time", "date": "2010-01-01", "step": "month",
                                 "range": {"from": "2010-01-01", "to": "2012-12-31"}, "playing": True}]),
    ("in 2019", [{"type": "set_time", "date": "2019-12-15", "step": "month",
                  "range": {"from": "2019-01-15", "to": "2019-12-31"}}]),
    ("stop playing", [{"type": "set_time", "playing": False}]),
    ("turn on floods past", [{"type": "set_layer", "layer": "floods_past", "on": True}]),
    ("hide the flood forecast", [{"type": "set_layer", "layer": "floods_ahead", "on": False}]),
    ("show rain", [{"type": "set_layer", "layer": "precip", "on": True}]),
    ("soil moisture off", [{"type": "set_layer", "layer": "soil", "on": False}]),
    ("turn on rain and snow", [{"type": "set_layer", "layer": "precip", "on": True},
                               {"type": "set_layer", "layer": "snow", "on": True}]),
    ("turn on floods past and turn off floods ahead", [
        {"type": "set_layer", "layer": "floods_past", "on": True},
        {"type": "set_layer", "layer": "floods_ahead", "on": False}]),
    ("flat map", [{"type": "set_layer", "layer": "globe", "on": False}]),
    ("satellite view", [{"type": "set_basemap", "basemap": "satellite-recent"}]),
    ("switch to the dark map", [{"type": "set_basemap", "basemap": "dark"}]),
    ("trace the Nile to the sea", [
        {"type": "fly_to", "place": "Nile", "river": True},
        {"type": "highlight_river", "place": "Nile", "direction": "downstream"}]),
    ("what drains to the Danube", [
        {"type": "fly_to", "place": "Danube", "river": True},
        {"type": "highlight_river", "place": "Danube", "direction": "upstream"}]),
    ("where are rivers much above normal", [
        {"type": "set_layer", "layer": "status", "on": True},
        {"type": "focus_status", "classes": ["much_above"]}]),
    ("where are rivers below normal", [
        {"type": "set_layer", "layer": "status", "on": True},
        {"type": "focus_status", "classes": ["much_below", "below"]}]),
    ("show all status classes", [{"type": "focus_status", "classes": []}]),
    ("where are rivers much above normal in South Asia last July", [
        {"type": "set_layer", "layer": "status", "on": True},
        {"type": "focus_status", "classes": ["much_above"]},
        {"type": "set_time", "date": "2026-07-15", "step": "month"},
        {"type": "fly_to", "region": "south_asia", "bbox": mc.REGIONS["south_asia"]["bbox"]}]),
    ("draw an area around Bangladesh", [{"type": "draw_area", "place": "Bangladesh", "label": "Bangladesh"}]),
    ("draw a box from 20, 88 to 27, 93", [{"type": "draw_area", "bbox": [88.0, 20.0, 93.0, 27.0]}]),
    ("drop a pin here saying Dhaka: check the Buriganga gauges", [
        {"type": "add_pin", "at": "center", "title": "Dhaka", "text": "check the Buriganga gauges"}]),
    ("put a pin at 23.7, 90.4", [{"type": "add_pin", "lat": 23.7, "lon": 90.4, "title": "Note"}]),
    ("show high rivers in Europe", [
        {"type": "set_layer", "layer": "status", "on": True},
        {"type": "focus_status", "classes": ["above", "much_above"]},
        {"type": "fly_to", "region": "europe", "bbox": mc.REGIONS["europe"]["bbox"]}]),
    ("where is it dry", [
        {"type": "set_layer", "layer": "status", "on": True},
        {"type": "focus_status", "classes": ["much_below", "below"]}]),
    ("show me the way of the Nile to the sea", [
        {"type": "fly_to", "place": "Nile", "river": True},
        {"type": "highlight_river", "place": "Nile", "direction": "downstream"}]),
]


@pytest.mark.parametrize(("text", "expected"), PHRASES, ids=[p[0] for p in PHRASES])
def test_the_grammar_reads_each_phrase_into_its_actions(text, expected):
    assert acts(text) == expected


def test_at_least_twenty_kinds_of_request_work_without_a_model():
    assert len(PHRASES) >= 20
    assert {a["type"] for _, exp in PHRASES for a in exp} == set(mc.ACTION_TYPES)


def test_every_action_is_said_in_words_for_the_log():
    res = mc.parse_command("where are rivers much above normal in South Asia last July", today=TODAY)
    assert res["said"] == ["Turn on World river status", "River status: only much above normal", "Map date: July 2026",
                           "Fly to South Asia"]
    assert mc.parse_command("play the last 5 years", today=TODAY)["said"] == [
        "Play October 2021 to October 2026, by month"]
    assert mc.parse_command("trace the Nile to the sea", today=TODAY)["said"][1] == "Trace Nile to the sea"


@pytest.mark.parametrize("text", ["undo", "undo that", "Undo the last one"])
def test_undo_is_a_control_not_an_action(text):
    res = mc.parse_command(text, today=TODAY)
    assert res["matched"] and res["control"] == "undo" and res["actions"] == []


def test_undo_all():
    assert mc.parse_command("start over", today=TODAY)["control"] == "undo_all"
    assert mc.parse_command("undo everything", today=TODAY)["control"] == "undo_all"


@pytest.mark.parametrize("text", [
    "what is the hundred year flood here",   # a question for Ask, not the map
    "show me the Rhine flooding in a clever way",
    "show me the weather",                    # not a place, though a gazetteer has a Weatherby
    "",
])
def test_words_the_rules_do_not_understand_are_left_to_a_model(text):
    res = mc.parse_command(text, today=TODAY)
    assert not res["matched"] and res["actions"] == []


def test_a_place_name_with_and_in_it_is_one_place():
    assert acts("go to Bosnia and Herzegovina") == [{"type": "fly_to", "place": "Bosnia and Herzegovina"}]


def test_the_future_is_not_a_map_date():
    assert not mc.parse_command("September 2031", today=TODAY)["matched"]
    assert mc.parse_command("September 2031", today=TODAY)["errors"]


def test_a_month_is_never_after_today():
    assert acts("this month") == [{"type": "set_time", "date": "2026-10-10", "step": "month"}]


# ── the checks every action passes before it runs ──────────────────────────


@pytest.mark.parametrize(("action", "why"), [
    ({"type": "teleport"}, "unknown action type"),
    ({"type": "fly_to"}, "fly_to needs"),
    ({"type": "fly_to", "bbox": [10, 50, 5, 40]}, "fly_to needs"),
    ({"type": "set_time", "date": "2026-13-01"}, "YYYY-MM-DD"),
    ({"type": "set_time", "date": "2027-01-01"}, "after today"),
    ({"type": "set_time", "date": "1900-01-01"}, "before anything"),
    ({"type": "set_time"}, "needs a date"),
    ({"type": "set_layer", "layer": "lasers", "on": True}, "unknown layer"),
    ({"type": "set_layer", "layer": "status", "on": "yes"}, "true or false"),
    ({"type": "focus_status", "classes": ["purple"]}, "classes must be"),
    ({"type": "set_basemap", "basemap": "neon"}, "unknown basemap"),
    ({"type": "highlight_river", "direction": "sideways", "place": "Nile"}, "direction"),
    ({"type": "highlight_river"}, "needs a place"),
    ({"type": "draw_area"}, "needs a bbox"),
    ({"type": "add_pin", "lat": 95, "lon": 10, "title": "x"}, "needs a lat"),
    ({"type": "add_pin", "lat": 5, "lon": 10}, "needs a title"),
    ({"type": "add_pin", "lat": 5, "lon": 10, "title": "x", "facts": [{"label": "flow"}]}, "label and a value"),
    ("fly", "an action must be an object"),
])
def test_actions_that_cannot_run_are_refused_with_a_reason(action, why):
    clean, err = mc.validate_action(action, today=TODAY)
    assert clean is None and why in err


def test_a_clean_action_keeps_only_what_its_type_uses():
    clean, err = mc.validate_action({"type": "set_layer", "layer": "rivers", "on": False, "colour": "red",
                                     "run": "rm -rf /"}, today=TODAY)
    assert err is None and clean == {"type": "set_layer", "layer": "rivers", "on": False}
    clean, _ = mc.validate_action({"type": "add_pin", "lat": 23.712345678, "lon": 90.4, "title": "  A   pin ",
                                   "facts": [{"label": "flow", "value": 12.5, "unit": "m3/s"}],
                                   "source": "GEOGLOWS v2"}, today=TODAY)
    assert clean == {"type": "add_pin", "lat": 23.71235, "lon": 90.4, "title": "A pin",
                     "facts": [{"label": "flow", "value": 12.5, "unit": "m3/s"}], "source": "GEOGLOWS v2"}


def test_a_range_is_ordered_and_clipped_to_today():
    rng = {"from": "2026-12-01", "to": "2020-01-01"}
    clean, _ = mc.validate_action({"type": "set_time", "range": rng}, today=TODAY)
    assert clean["range"] == {"from": "2020-01-01", "to": TODAY}


def test_validate_actions_keeps_the_good_ones_and_says_why_for_the_rest():
    out = mc.validate_actions({"actions": [{"type": "set_layer", "layer": "rivers", "on": True}, {"type": "nope"}]},
                              today=TODAY)
    assert out["actions"] == [{"type": "set_layer", "layer": "rivers", "on": True}]
    assert out["errors"] == ["action 2: unknown action type 'nope'"]
    many = mc.validate_actions([{"type": "fly_to", "zoom_by": 1}] * 10, today=TODAY)
    assert len(many["actions"]) == 8 and "first 8 of 10" in many["errors"][0]


def test_the_schema_names_every_type_layer_and_basemap():
    item = mc.action_schema()["properties"]["actions"]["items"]["properties"]
    assert item["type"]["enum"] == list(mc.ACTION_TYPES)
    assert item["layer"]["enum"] == list(mc.LAYERS)
    assert item["basemap"]["enum"] == list(mc.BASEMAPS)


# ── the gazetteer ───────────────────────────────────────────────────────────


def photon(name, *, osm_key="place", osm_value="country", kind="country", extent=None, point=(0.0, 0.0), **props):
    p = {"name": name, "osm_key": osm_key, "osm_value": osm_value, "type": kind, **props}
    if extent:
        p["extent"] = extent
    return {"geometry": {"coordinates": list(point)}, "properties": p}


FEATURES = {
    "Nile": [photon("Nile", osm_key="boundary", osm_value="census", kind="other", country="United States",
                    point=(-120.9, 46.8)),
             photon("Nile River", osm_key="waterway", osm_value="river", kind="other", country="Egypt",
                    extent=[30.28, 31.52, 33.98, 15.64], point=(32.1, 22.7))],
    "Netherlands": [photon("Netherlands", extent=[-70.27, 53.75, 7.23, 11.78], point=(5.63, 52.24))],
    "United States": [photon("United States", extent=[-180.0, 71.6, 180.0, -14.8], point=(-100.4, 39.8))],
    "Taipei": [photon("Taipei", kind="city", osm_value="city", extent=[121.46, 25.21, 121.67, 24.96],
                      point=(121.56, 25.04), country="Taiwan")],
    "something clever": [photon("Something Fishy", kind="house", osm_value="restaurant", point=(-81.7, 41.4))],
    "in Europe": [photon("The Leuven Institute for Ireland In Europe", kind="house", osm_value="college",
                         point=(4.7, 50.9))],
    "Bangaldesh": [photon("Bangladesh", extent=[88.0, 26.6, 92.7, 20.4], point=(90.3, 23.8))],
    "weather": [photon("Weatherby", kind="city", osm_value="village", point=(-94.2, 39.9))],
    "Amazon": [photon("Amazon River", osm_key="waterway", osm_value="river", kind="other", country="Brazil",
                      extent=[-50.23, 0.69, -50.15, 0.68], point=(-50.19, 0.68))],
}


def fetch(query):
    return FEATURES.get(query, [])


def test_a_river_is_looked_up_as_a_river():
    hit = mc.resolve_place("Nile", want="river", fetch=fetch)
    assert hit["is_river"] and hit["name"] == "Nile River" and hit["bbox"] == [30.28, 15.64, 33.98, 31.52]
    assert hit["credit"] == mc.PLACE_CREDIT


def test_far_territories_do_not_drag_the_camera_across_the_ocean():
    nl = mc.resolve_place("Netherlands", fetch=fetch)["bbox"]
    assert nl[0] > 3 and nl[1] > 50, nl          # cut back to the near side, not the Caribbean
    us = mc.resolve_place("USA", fetch=fetch)  # an alias: "USA" alone finds a town in Japan
    assert us["name"] == "United States" and us["bbox"] is None and us["zoom"] == 4.0


def test_a_fuzzy_neighbour_is_not_the_place_asked_for():
    assert mc.resolve_place("something clever", fetch=fetch)["error"].startswith("no place called")


def test_a_long_name_holding_the_words_is_not_the_place():
    assert mc.resolve_place("in Europe", fetch=fetch)["error"].startswith("no place called")
    assert mc.resolve_place("weather", fetch=fetch)["error"].startswith("no place called")
    assert mc.resolve_place("Bangaldesh", fetch=fetch)["name"] == "Bangladesh"   # a spelling slip still finds it


def test_a_river_known_by_one_short_stretch_is_framed_from_its_point():
    hit = mc.resolve_place("Amazon", want="river", fetch=fetch)
    assert hit["is_river"] and hit["bbox"] is None and hit["zoom"] == 8.0


def test_regions_need_no_lookup():
    hit = mc.resolve_place("South Asia", fetch=lambda q: pytest.fail("no lookup for a region"))
    assert hit["bbox"] == mc.REGIONS["south_asia"]["bbox"]


def test_the_gazetteer_being_down_is_said_not_raised():
    def down(q):
        raise OSError("offline")

    assert "unavailable" in mc.resolve_place("Taipei", fetch=down)["error"]


def test_resolving_turns_names_into_boxes_and_points():
    out = mc.resolve_actions(acts("trace the Nile to the sea"), fetch=fetch)
    fly, river = out["actions"]
    assert fly == {"type": "fly_to", "bbox": [30.28, 15.64, 33.98, 31.52], "label": "Nile River",
                   "where": "Nile River, Egypt"}
    assert river["lat"] == 22.7 and river["lon"] == 32.1 and river["direction"] == "downstream"
    assert out["said"] == ["Fly to Nile River", "Trace Nile River to the sea"] and out["credit"] == mc.PLACE_CREDIT


def test_an_optional_river_light_is_dropped_for_a_place_that_is_not_a_river():
    out = mc.resolve_actions(acts("show the Netherlands"), fetch=fetch)
    assert [a["type"] for a in out["actions"]] == ["fly_to"] and out["notes"] == []


def test_a_place_that_cannot_be_found_drops_its_action_with_a_note():
    out = mc.resolve_actions([{"type": "fly_to", "place": "Atlantis"}], fetch=fetch)
    assert out["actions"] == [] and "Atlantis" in out["notes"][0] and out["credit"] is None


def test_a_river_at_a_place_is_found_where_the_place_is():
    out = mc.resolve_actions(acts("what drains to the Nile at Taipei"), fetch=fetch)
    fly, river = out["actions"]
    assert fly["center"] == [25.04, 121.56] and fly["zoom"] == 8.0 and fly["label"] == "Nile at Taipei"
    assert (river["lat"], river["lon"], river["direction"]) == (25.04, 121.56, "upstream")


def test_a_place_does_not_swallow_the_next_clause():
    assert acts("show the Mekong and play the last 2 years")[1] == {"type": "fly_to", "place": "Mekong",
                                                                     "river": True}


def test_an_area_needs_an_outline_not_a_point():
    out = mc.resolve_actions([{"type": "draw_area", "place": "something clever"}], fetch=fetch)
    assert out["actions"] == [] and out["notes"]
    out = mc.resolve_actions([{"type": "draw_area", "place": "Taipei", "label": "Taipei"}], fetch=fetch)
    assert out["actions"][0]["bbox"] == [121.46, 24.96, 121.67, 25.21]


# ── the model path ──────────────────────────────────────────────────────────


def test_the_prompt_carries_examples_computed_by_the_grammar():
    prompt = mc.model_prompt("a map centred on 23.7, 90.4", today=TODAY)
    for ex in mc.EXAMPLES:
        assert f"Request: {ex}" in prompt["system"]
    replies = re.findall(r"Reply: (\{.*\})", prompt["system"])
    assert len(replies) == len(mc.EXAMPLES)
    for reply in replies:
        assert mc.parse_model_reply(reply, today=TODAY)["errors"] == []
    assert "Today is 2026-10-10" in prompt["system"] and "23.7, 90.4" in prompt["system"]
    assert prompt["schema"] == mc.action_schema()


@pytest.mark.parametrize("reply", [
    '{"actions": [{"type": "set_layer", "layer": "snow", "on": true}]}',
    'Sure! ```json\n{"actions": [{"type": "set_layer", "layer": "snow", "on": true}]}\n```',
    '[{"type": "set_layer", "layer": "snow", "on": true}]',
    {"actions": [{"type": "set_layer", "layer": "snow", "on": True}]},
])
def test_a_models_reply_is_read_and_checked(reply):
    out = mc.parse_model_reply(reply, today=TODAY)
    assert out["actions"] == [{"type": "set_layer", "layer": "snow", "on": True}]
    assert out["said"] == ["Turn on Snow cover"]


def test_a_reply_that_is_not_json_runs_nothing():
    assert mc.parse_model_reply("I would fly to the Rhine.", today=TODAY)["actions"] == []
    out = mc.parse_model_reply('{"actions": [{"type": "set_time", "date": "2099-01-01"}]}', today=TODAY)
    assert out["actions"] == [] and "after today" in out["errors"][0]


class FakeClient:
    def __init__(self, content):
        self.content = content
        self.calls = []
        outer = self

        class Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                from types import SimpleNamespace as Ns

                return Ns(choices=[Ns(message=Ns(content=outer.content))])

        self.chat = type("Chat", (), {"completions": Completions()})()


def test_model_command_asks_the_readers_model_and_checks_its_answer():
    client = FakeClient('{"actions": [{"type": "fly_to", "place": "Mekong"}, {"type": "launch"}]}')
    out = mc.model_command("take me along the Mekong", client=client, model="m", provider="p", today=TODAY,
                           context="a map of Asia")
    assert out["actions"] == [{"type": "fly_to", "place": "Mekong"}] and out["errors"]
    assert out["model"] == "m via p"
    sent = client.calls[0]
    assert sent["temperature"] == 0 and sent["messages"][1]["content"] == "take me along the Mekong"
    assert "a map of Asia" in sent["messages"][0]["content"]


def test_model_command_never_runs_without_a_key(monkeypatch):
    for env in ("OPENAI_API_KEY", "GROQ_API_KEY", "NVIDIA_API_KEY", "HF_TOKEN", "AQUASCOPE_LLM_API_KEY",
                "ANTHROPIC_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
                "AQUASCOPE_LLM_BASE_URL"):
        monkeypatch.delenv(env, raising=False)
    with pytest.raises(RuntimeError):
        mc.model_command("show the Rhine", today=TODAY)


# ── the faces, and the Explorer's copies ────────────────────────────────────


def test_the_mcp_server_offers_the_grammar():
    pytest.importorskip("mcp")
    from aquascope import mcp_server as m

    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert "map_command" in names
    out = m.map_command("turn on floods past")
    assert out["actions"] == [{"type": "set_layer", "layer": "floods_past", "on": True}]
    assert out["schema"]["required"] == ["actions"]


def test_cli_prints_the_actions(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "map", "trace", "the", "Nile", "to", "the", "sea", "--today", TODAY])
    cli.main()
    out = capsys.readouterr().out
    assert "Understood by rules" in out and "Trace Nile to the sea" in out and '"direction": "downstream"' in out
    monkeypatch.setattr(sys, "argv", ["aquascope", "map", "September 2023", "--json", "--today", TODAY])
    cli.main()
    assert json.loads(capsys.readouterr().out)["actions"][0]["date"] == "2023-09-15"


def test_cli_exits_non_zero_when_nothing_was_understood(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["aquascope", "map", "what is the hundred year flood"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1 and "Not understood" in capsys.readouterr().out


def test_the_explorer_has_a_control_for_every_layer():
    js = (ROOT / "explorer/src/map-actions-core.js").read_text(encoding="utf-8")
    block = js[js.index("export const LAYER_CONTROLS"): js.index("};", js.index("export const LAYER_CONTROLS"))]
    keys = re.findall(r'^\s*"?([\w-]+)"?:\s*"', block, re.M)
    assert keys == list(mc.LAYERS)


def test_the_overlays_and_basemaps_are_the_explorers():
    layers_js = (ROOT / "explorer/src/layers.js").read_text(encoding="utf-8")
    overlays = layers_js[layers_js.index("export const OVERLAYS"): layers_js.index("export const OVERLAY_GROUPS")]
    basemaps = layers_js[layers_js.index("export const BASEMAPS"): layers_js.index("export const TERRAIN_DEM")]
    overlay_ids = re.findall(r'^\s{4}id: "([\w-]+)"', overlays, re.M)
    assert overlay_ids and set(overlay_ids) <= set(mc.LAYERS)
    assert re.findall(r'^\s{4}id: "([\w-]+)"', basemaps, re.M) == list(mc.BASEMAPS)


def test_the_status_classes_are_the_maps():
    from aquascope.map_layers import STATUS_CLASSES

    assert mc.STATUS_IDS == tuple(c["id"] for c in STATUS_CLASSES)
