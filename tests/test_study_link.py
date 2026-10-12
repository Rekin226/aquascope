"""Study links (aquascope.study_link): a plan in a URL, decoded and checked as untrusted input."""

from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

import pytest

from aquascope import study_link as sl

ROOT = Path(__file__).resolve().parents[1]
RECORDED = sorted((ROOT / "explorer" / "showcase" / "studies").glob("*/workspace.json"))


def _ws(name: str = "kingston-flood") -> dict:
    return json.loads((ROOT / "explorer" / "showcase" / "studies" / name / "workspace.json").read_text("utf-8"))


def _compact(**over) -> dict:
    base = {
        "v": 1, "q": "Design flow for a road crossing", "site": {"lat": 51.415, "lon": -0.308},
        "intake": {"return_period": 100},
        "steps": [
            {"id": "s1", "tool": "describe_catchment", "arguments": {"lat": 51.415, "lon": -0.308}},
            {"id": "s2", "tool": "flood_frequency", "arguments": {"source": "uk_ea", "station_id": "3400TH"},
             "expects": [{"check": "min_years", "value": 20, "path": "years"}], "depends_on": ["s1"]},
        ],
    }
    base.update(over)
    return base


def _token_of(obj: dict, prefix: str = "z1.") -> str:
    raw = json.dumps(obj).encode()
    if prefix == "z1.":
        c = zlib.compressobj(9, zlib.DEFLATED, -15)
        raw = c.compress(raw) + c.flush()
    return prefix + base64.urlsafe_b64encode(raw).decode().rstrip("=")


@pytest.mark.parametrize("path", RECORDED, ids=[p.parent.name for p in RECORDED])
def test_every_recorded_study_round_trips_through_a_link(path: Path) -> None:
    ws = json.loads(path.read_text("utf-8"))
    made = sl.link_from_workspace(ws)
    assert made["ok"], made["errors"]
    assert made["token"].startswith("z1.") and made["chars"] <= sl.MAX_LINK_CHARS
    opened = sl.open_link(made["token"])
    assert opened["ok"], opened["errors"]
    study = opened["study"]
    assert (study["lat"], study["lon"]) == (ws["site"]["lat"], ws["site"]["lon"])
    assert study["text"] == ws["brief"]["problem"]
    want = [(s["id"], s["tool"], s.get("arguments") or {}, s.get("method")) for s in ws["study"]["steps"]]
    got = [(s["id"], s["tool"], s.get("arguments") or {}, s.get("method")) for s in study["plan"]["steps"]]
    assert got == want
    assert "results" not in json.dumps(study), "a link never carries results"
    assert all("outputs" not in s for s in study["plan"]["steps"])


@pytest.mark.parametrize("path", RECORDED, ids=[p.parent.name for p in RECORDED])
def test_every_recorded_study_yaml_opens(path: Path) -> None:
    opened = sl.open_study_yaml((path.parent / "study.yaml").read_text("utf-8"))
    assert opened["ok"], opened["errors"]
    assert len(opened["study"]["plan"]["steps"]) == len(json.loads(path.read_text("utf-8"))["study"]["steps"])


def test_encode_decode_round_trip_both_forms() -> None:
    c = _compact()
    assert sl.decode(sl.encode(c)) == c
    assert sl.decode(_token_of(c, "j1.")) == c
    assert sl.validate(c) == []


def test_the_own_table_case_says_the_link_does_not_carry_the_table() -> None:
    opened = sl.open_link(sl.link_from_workspace(_ws("own-table-flood"))["token"])
    assert opened["ok"]
    assert opened["study"]["tables"] and "does not carry it" in opened["notes"][0]


def test_an_unregistered_function_is_refused() -> None:
    bad = _compact(steps=[{"id": "s1", "tool": "os_system", "arguments": {"cmd": "rm -rf /"}}])
    out = sl.open_link(sl.encode(bad))
    assert not out["ok"] and "unknown tool 'os_system'" in out["errors"][0]
    for step in ({"id": "s1", "tool": "run_python", "arguments": {"code": "1"}},        # the Analyst's own loop
                 {"id": "s1", "tool": "flood_frequency", "arguments": {"source": "uk_ea", "station_id": "x",
                                                                       "evil": 1}},
                 {"id": "s1", "tool": "describe_catchment", "arguments": {}, "method": "no_such_method"},
                 {"id": "s1", "tool": "describe_catchment", "arguments": {}, "expects": [{"check": "eval"}]}):
        assert sl.validate(_compact(steps=[step])), step


def test_references_may_only_point_backwards() -> None:
    steps = [{"id": "s1", "tool": "flood_frequency", "arguments": {"source": "uk_ea", "station_id": "x"},
              "depends_on": ["s2"]},
             {"id": "s2", "tool": "describe_catchment", "arguments": {}}]
    assert any("depends_on 's2'" in e for e in sl.validate(_compact(steps=steps)))


def test_the_caps() -> None:
    many = [{"id": f"s{i}", "tool": "describe_catchment", "arguments": {}} for i in range(1, sl.MAX_STEPS + 2)]
    assert "at most 12" in sl.validate(_compact(steps=many))[0]
    out = sl.open_link("z1." + "A" * (sl.MAX_LINK_CHARS + 1))
    assert not out["ok"] and "too long" in out["errors"][0]
    # a zip bomb: small token, huge JSON
    bomb = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = bomb.compress(b'{"q":"' + b"a" * (sl.MAX_STUDY_BYTES * 3) + b'"}') + bomb.flush()
    token = "z1." + base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert len(token) < sl.MAX_LINK_CHARS
    out = sl.open_link(token)
    assert not out["ok"] and "more than a study plan" in out["errors"][0]
    assert not sl.open_study_yaml("q: x\n" * sl.MAX_STUDY_BYTES)["ok"]
    with pytest.raises(sl.LinkError, match="too long for a link"):
        sl.encode(_compact(), max_chars=40)


def test_rationales_are_dropped_before_a_link_is_refused() -> None:
    c = _compact()
    for s in c["steps"]:
        s["rationale"] = "x" * 390
    full = len(sl.encode(c, max_chars=10_000))
    token = sl.encode(c, max_chars=full - 1)
    assert all("rationale" not in s for s in sl.decode(token)["steps"])


@pytest.mark.parametrize("token, words", [
    ("", "no study"),
    ("abc", "not a study link"),
    ("z1.!!!", "base64"),
    ("z1.AAAA", "damaged"),
    ("z1." + base64.urlsafe_b64encode(b"\x07garbage").decode(), "does not decompress"),
    (_token_of([1, 2]), "does not hold a study plan"),
    ("j1." + base64.urlsafe_b64encode(b"\xff\xfe").decode(), "damaged"),
])
def test_damaged_links_read_as_words(token: str, words: str) -> None:
    out = sl.open_link(token)
    assert not out["ok"] and words in out["errors"][0]


@pytest.mark.parametrize("over, words", [
    ({"v": 2}, "version"),
    ({"q": ""}, "no question"),
    ({"site": {"lat": 95, "lon": 0}}, "no valid site"),
    ({"site": {"lat": "nan", "lon": 0}}, "no valid site"),
    ({"site": "here"}, "no valid site"),
    ({"steps": []}, "no steps"),
    ({"steps": ["s1"]}, "not a mapping"),
    ({"intake": [1]}, "intake"),
])
def test_bad_shapes_are_refused(over: dict, words: str) -> None:
    errors = sl.validate(_compact(**over))
    assert any(words in e for e in errors), errors


def test_expand_passes_only_known_keys() -> None:
    c = _compact(plan={"objective": "o", "decline": True, "playbook": "flood_risk"})
    c["steps"][0]["results"] = {"x": 1}
    c["steps"][0]["sha256"] = "abc"
    c["extra"] = {"api_key": "sk-1"}
    out = sl.open_link(sl.encode(c))
    assert out["ok"]
    study = out["study"]
    assert set(study["plan"]) == {"steps", "objective"}, "a decline or any other key never reaches approve"
    assert set(study["plan"]["steps"][0]) <= {"id", "tool", "arguments", "method", "expects", "fallback",
                                              "depends_on", "rationale"}
    assert "sk-1" not in json.dumps(study)
    assert study["playbook"] == "flood_risk"


def test_a_workspace_without_a_plan_gives_no_link() -> None:
    out = sl.link_from_workspace({"site": {"lat": 1, "lon": 2}, "brief": {"problem": "p"}})
    assert not out["ok"] and "no plan" in out["errors"][0]


def test_the_worker_face_ops() -> None:
    made = sl.studio_op({"op": "link", "workspace": _ws()})
    assert made["ok"]
    assert sl.studio_op({"op": "open_link", "token": made["token"]})["ok"]
    yaml = (ROOT / "explorer" / "showcase" / "studies" / "kingston-flood" / "study.yaml").read_text("utf-8")
    assert sl.studio_op({"op": "open_link", "yaml": yaml})["ok"]
    assert not sl.studio_op({"op": "nope"})["ok"]
    assert not sl.studio_op({"op": "open_link", "yaml": "- just\n- a list\n"})["ok"]
