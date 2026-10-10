"""Scout (#563): aquascope.map_scout's rules, ranking, claim lock and faces, with no network."""

from __future__ import annotations

import builtins
import json
import re
from pathlib import Path

import numpy as np
import pytest

from aquascope import map_scout as ms

ROOT = Path(__file__).resolve().parents[1]


# ── formatting ───────────────────────────────────────────────────────────────


def test_numbers_are_formatted_by_code():
    assert ms.fmt_num(412345) == "412,000"
    assert ms.fmt_num(18.864) == "18.9"
    assert ms.fmt_num(0.10712) == "0.107"
    assert ms.fmt_num(2840.4) == "2,840"
    assert ms.fmt_num(None) == "unknown"
    assert ms.month_words("2026-09") == "September 2026"
    assert ms.day_words("2026-10-14") == "Wed 14 Oct"
    assert [ms.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]
    assert ms.short_name("La rivière Capot au Morne-Rouge [mackintosh] - Pont de Mackintosh") == \
        "La rivière Capot au Morne-Rouge"
    assert len(ms.short_name("x" * 90)) == 40


# ── views ────────────────────────────────────────────────────────────────────


def test_views_box_polygon_circle_and_world():
    assert ms.check_view(None) is None
    assert ms.check_view({"bbox": [-180, -90, 180, 90]}) is None
    box = ms.check_view({"bbox": [170, -10, -170, 10]})          # across the antimeridian
    assert ms.in_view(0, 175, box) and ms.in_view(0, -175, box) and not ms.in_view(0, 0, box)
    poly = ms.check_view({"polygon": [[170, -10], [190, -10], [190, 10], [170, 10]], "min_km": 120})
    assert poly["min_km"] == 120 and poly["bbox"] == [170, -10, -170, 10]
    assert ms.in_view(0, -175, poly) and ms.in_view(0, 175, poly) and not ms.in_view(0, 160, poly)
    circle = ms.check_view({"center": [0, 0], "radius_km": 1000})
    assert ms.in_view(0, 8, circle) and not ms.in_view(0, 10, circle)
    with pytest.raises(ValueError):
        ms.check_view({"bbox": [0, 10, 5, 5]})
    with pytest.raises(ValueError):
        ms.check_view({"polygon": [[0, 0], [1, 1]]})
    with pytest.raises(ValueError):
        ms.check_view({"center": [0, 0], "radius_km": 0})
    with pytest.raises(ValueError):
        ms.check_view({"bbox": [0, 0, 1, 1], "min_km": -3})
    # The mask over arrays agrees with the point test.
    lat, lon = np.array([0.0, 0.0, 0.0]), np.array([175.0, -175.0, 160.0])
    assert ms.view_mask(lat, lon, poly).tolist() == [True, True, False]


def test_ortho_distance_shrinks_towards_the_edge_of_the_globe():
    at_centre = ms.ortho_km([0, 0], 0, 0, 0, 5)
    at_edge = ms.ortho_km([0, 0], 0, 80, 0, 85)
    assert at_centre == pytest.approx(ms.km_between(0, 0, 0, 5), rel=0.01)
    assert at_edge < at_centre / 5


def test_labelling_without_scipy_matches(monkeypatch):
    rng = np.random.default_rng(3)
    mask = rng.random((40, 60)) > 0.6
    with_scipy, n1 = ms.label_components(mask)
    real_import = builtins.__import__

    def no_scipy(name, *a, **k):
        if name.startswith("scipy"):
            raise ImportError("no scipy here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    without, n2 = ms.label_components(mask)
    assert n1 == n2
    # Same partition: each scipy component maps onto exactly one fallback component.
    pairs = {(a, b) for a, b in zip(with_scipy.ravel(), without.ravel()) if a}
    assert len(pairs) == n1 and len({a for a, _ in pairs}) == n1 and len({b for _, b in pairs}) == n1


# ── world river status ───────────────────────────────────────────────────────


def _world(fill: int = 3) -> np.ndarray:
    """A 1 degree world of classes: land (normal) over 60 S to 70 N, sea (0) elsewhere."""
    g = np.zeros((180, 360), dtype=np.uint8)
    g[20:150, :] = fill
    return g


def _put(g: np.ndarray, cls: int, w: float, s: float, e: float, n: float) -> None:
    g[int(90 - n):int(90 - s), int(w + 180):int(e + 180)] = cls


def test_status_finds_the_largest_areas_on_each_side():
    g = _world()
    _put(g, 5, 15, -30, 35, -15)     # much above, southern Africa, 20 x 15 degrees
    _put(g, 5, 100, 20, 104, 24)     # much above, small
    _put(g, 1, -70, -10, -50, 0)     # much below, the Amazon
    out = ms.status_findings(g, "2026-09", min_area_km2=1000)
    above = [f for f in out if f["side"] == "much_above"]
    below = [f for f in out if f["side"] == "much_below"]
    assert len(above) == 2 and len(below) == 1
    big = above[0]
    assert big["facts"][0]["value"] > above[1]["facts"][0]["value"]
    assert 15 <= big["lon"] <= 35 and -30 <= big["lat"] <= -15          # the pin is on the area
    assert big["extent"] == [15.0, -30.0, 35.0, -15.0]
    assert big["slots"]["month"] == "September 2026" and "km²" in big["slots"]["area"]
    assert big["reason"].startswith(f"Rivers across {big['slots']['area']} were much above normal in September 2026")
    assert big["title"].endswith(": rivers much above normal")
    # A view leaves out what is not in it.
    out = ms.status_findings(g, "2026-09", view={"bbox": [-80, -20, -40, 10]}, min_area_km2=1000)
    assert [f["side"] for f in out] == ["much_below"]


def test_status_record_uses_the_fixed_named_region_not_the_area_itself():
    g = _world()
    _put(g, 5, 15, -30, 35, -15)
    blocks = ms.status_blocks(g, 1)
    out = ms.status_findings(g, "2026-09", blocks=blocks, min_area_km2=1000)
    history = {}
    for year in range(2015, 2026):     # every earlier September: a little of it much above
        h = _world()
        _put(h, 5, 15, -30, 18, -27)
        history[f"{year}-09"] = ms.status_blocks(h, 1)
    history["2020-08"] = ms.status_blocks(g, 1)   # another month: ignored
    ms.status_record(out, "2026-09", blocks, history)
    f = next(x for x in out if x["side"] == "much_above")
    assert f["record_rank"] == 1 and f["record_region"] == "southern Africa"
    assert "Across southern Africa as a whole, the share much above normal was the largest in September of 12 " \
           "years since 2015." in f["reason"]
    # A year with more of the region much above outranks this one.
    history["2016-09"] = ms.status_blocks(_world(5), 1)
    out = ms.status_findings(g, "2026-09", blocks=blocks, min_area_km2=1000)
    ms.status_record(out, "2026-09", blocks, history)
    f = next(x for x in out if x["side"] == "much_above")
    assert f["record_rank"] == 2 and "2nd largest" in f["reason"]


# ── floods ahead ─────────────────────────────────────────────────────────────


def _reach(rid, lat, lon, rp, peak, q2, share=1.0, day="2026-10-14", **q):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"river_id": rid, "rp": rp, "peak": peak, "q2": q2, "day": day, "share": share, **q}}


def test_floods_ahead_groups_a_river_and_never_claims_a_flow_it_lacks():
    feats = [
        _reach(1, -24.0, -50.7, 25, 2840, 1050, q25=2500.0),
        _reach(2, -24.2, -50.6, 25, 2780, 1030),                 # same river, 25 km away
        _reach(3, 60.8, 121.0, 100, 120, 24.7, share=36 / 51),   # a creek at its 100-year flow
        _reach(4, 39.1, -96.9, 2, 300, 290),
    ]
    out = ms.floods_ahead_findings(feats, {"issue_date": "2026-10-09"})
    assert [f["river_id"] for f in out] == [1, 3, 4]       # a big river at 25 years before a creek at 100
    big, creek = out[0], out[1]
    assert big["slots"]["nearby"] == " 1 more reach within 200 km does too."
    assert big["reason"] == ("GEOGLOWS forecast of Fri 9 Oct: 2,840 m³/s on Wed 14 Oct, at or above the 25-year "
                             "flow of 2,500 m³/s; the 2-year flow is 1,050 m³/s. All 51 ensemble members reach it. "
                             "1 more reach within 200 km does too.")
    assert creek["slots"]["rpflow"] == "" and "100-year flow; the 2-year flow is 24.7 m³/s" in creek["reason"]
    assert "36 of 51 ensemble members" in creek["reason"]
    assert not any(f["label"] == "100-year flow" for f in creek["facts"])


# ── floods past ──────────────────────────────────────────────────────────────


def test_floods_past_rows_start_at_the_south_pole_and_touching_cells_group():
    from aquascope.context.floods_past import cell_of

    r, c = cell_of(-6.2, 106.8)                  # Jakarta
    cells = [[r, c, 40, 0], [r, c + 1, 10, 0], [r + 1, c + 1, 3, 0],   # one group
             [r + 40, c, 5, 3000], [r + 80, c, 1, 0]]                  # a second; a cell under both floors
    out = ms.floods_past_findings(cells, "2026-01")
    assert len(out) == 2
    first = out[0]
    assert first["lat"] == pytest.approx(-6.25) and first["lon"] == pytest.approx(106.75)
    assert first["facts"][0]["value"] == 53 and first["slots"]["events"] == "53 flood events"
    assert first["reason"].startswith("53 flood events in the news started here in January 2026")
    second = out[1]
    assert "Sentinel-1 radar saw 3,000 flooded 20 m pixels" in second["reason"]
    hist = {f"{y}-01": [[r, c, 10, 0]] for y in range(2000, 2026)}
    hist["2010-01"] = [[r, c, 100, 0]]
    ranked = ms.floods_past_findings(cells, "2026-01", history=hist)[0]
    assert ranked["record_rank"] == 2 and "The 2nd most in these cells in January of 27 years since 2000." \
        in ranked["reason"]


def test_complete_month_skips_a_stub():
    months = [{"month": f"2025-{m:02d}", "news": 20000, "radar": 0} for m in range(1, 13)]
    months += [{"month": "2026-01", "news": 26000, "radar": 0}, {"month": "2026-02", "news": 12, "radar": 0}]
    assert ms.complete_month({"months": months}) == "2026-01"
    assert ms.complete_month({"months": months}, "2025-07") == "2025-07"
    assert ms.complete_month({"months": []}) is None


# ── gauges today and models ─────────────────────────────────────────────────


def _gauge(i, lat, lon, cls="much_below", pct=3.0, n=30):
    return {"source": "uk_ea", "station_id": f"g{i}", "name": f"Gauge {i}", "lat": lat, "lon": lon, "class": cls,
            "percentile": pct, "value": 0.115, "n_years": n, "value_date": "2026-10-08"}


def test_gauges_today_group_around_the_most_extreme():
    rows = [_gauge(1, 52.0, -3.0, pct=0.0, n=60), _gauge(2, 52.3, -2.5), _gauge(3, 51.8, -2.0),
            _gauge(4, 40.0, -3.7, cls="much_above", pct=100.0, n=12), _gauge(5, 30.0, 0, cls="normal")]
    out = ms.gauges_today_findings(rows)
    low = next(f for f in out if f["side"] == "much_below")
    assert low["gauge"] == "uk_ea/g1" and low["facts"][0]["value"] == 3
    assert low["reason"] == ("3 gauges within 150 km are much below normal for Thu 8 Oct. Gauge 1 is lower than "
                             "every value within a week of this date in its other 60 years.")
    high = next(f for f in out if f["side"] == "much_above")
    assert high["reason"].startswith("This gauge is much above normal") and "higher than every value" in high["reason"]


def _skill(sid, kge, mean, pbias=50.0, best=True, days=4000, lat=50.0, lon=5.0, model="geoglows", ratio=1.0):
    return {"source": "x", "station_id": sid, "lat": lat, "lon": lon, "model": model, "label": model.upper(),
            "is_best": best, "kge": kge, "pbias": pbias, "n_days": days, "mean_gauge": mean, "area_ratio": ratio,
            "start": "2001-01-01", "end": "2024-12-31", "computed_at": "2026-10-08T00:00:00", "name": f"River {sid}"}


def test_models_disagree_keeps_real_rivers_and_drops_units_errors():
    rows = [_skill("big", -0.6, 300.0, lat=45, lon=0), _skill("big", -0.9, 300.0, best=False, model="grrr",
                                                            lat=45, lon=0),
            _skill("small", -3.0, 5.0, lat=10, lon=10), _skill("units", -0.8, 99000.0, pbias=-99.9, lat=20, lon=20),
            _skill("short", -1.0, 50.0, days=300, lat=30, lon=30), _skill("fine", 0.6, 80.0, lat=35, lon=35),
            _skill("ratio", -1.0, 50.0, ratio=3.0, lat=36, lon=36)]
    out = ms.models_disagree_findings(rows)
    assert [f["gauge"] for f in out] == ["x/big", "x/small"]          # the largest river first
    f = out[0]
    assert f["slots"]["models"] == "2 models" and f["slots"]["kge"] == "-0.60"
    assert "Its mean flow is 50% above the gauge's." in f["reason"]
    # The page hands in the best rows only, with the count of models.
    out = ms.models_disagree_findings([{**rows[0], "n_models": 3}])
    assert out[0]["slots"]["models"] == "3 models"


# ── ranking ──────────────────────────────────────────────────────────────────


def _f(fid, kind, lat, lon, score):
    return {"id": fid, "kind": kind, "lat": lat, "lon": lon, "score": score, "slots": {}, "title": fid,
            "reason": fid, "template": "floods_ahead"}


def test_rank_interleaves_kinds_keeps_caps_and_spacing():
    fs = [_f(f"a{i}", "floods_ahead", 0, i * 20, 1 - i / 10) for i in range(2)]
    fs += [_f(f"s{i}", "status", 30, i * 20, 0.9 - i / 10) for i in range(3)]
    fs += [_f("g0", "gauges_today", -30, 0, 0.5), _f("p0", "floods_past", -30, 40, 0.99),
           _f("p1", "floods_past", -30, 80, 0.98), _f("p2", "floods_past", -30, 120, 0.97),
           _f("m0", "models_disagree", 60, 0, 0.2)]
    fs.append(_f("near", "status", 0, 0.5, 1.0))          # on top of a0: the earlier kind keeps the place
    out = ms.rank_findings(fs, max_pins=10)
    ids = [f["id"] for f in out]
    assert "near" not in ids and len(ids) == 9
    assert ids[:5] == ["a0", "s0", "g0", "p0", "m0"]          # the strongest of each kind first
    assert ids[5:] == ["a1", "s1", "p1", "s2"]
    assert "p2" not in ids                                     # the past holds two at most
    assert [f["rank"] for f in out] == list(range(1, 10))
    # Extras for a model come after the ten, in score order.
    many = [_f(f"a{i}", "floods_ahead", 0, i * 20 - 170, 1 - i / 20) for i in range(14)]
    more = ms.rank_findings(many, max_pins=10, extra=2)
    assert [f["id"] for f in more] == [f"a{i}" for i in range(12)]
    # On screen two pins closer than min_km cannot both stand.
    out = ms.rank_findings([_f("x", "status", 0, 0, 1), _f("y", "status", 0, 1, 0.9)], view={"min_km": 200,
                                                                                         "bbox": [-5, -5, 5, 5]})
    assert [f["id"] for f in out] == ["x"]


# ── places ───────────────────────────────────────────────────────────────────


def test_places_name_from_the_gazetteer_then_regions_then_coordinates():
    assert ms.place_words({"state": "Meghalaya", "country": "India"}) == "Meghalaya, India"
    assert ms.place_words({"country": "Chad"}) == "Chad"
    assert ms.place_words(None) is None
    feats = ms.floods_ahead_findings([_reach(1, 25.5, 90.5, 5, 900, 400), _reach(2, 20, 80, 5, 900, 400),
                                      _reach(3, -60, -150, 5, 900, 400)], {"issue_date": "2026-10-09"})
    answers = {1: {"state": "Meghalaya", "country": "India"}}

    def fetch(lat, lon):
        if lat == 20:
            raise OSError("down")
        return answers.get(1) if lat == 25.5 else None

    assert ms.name_places(feats, fetch=fetch) == 1
    places = {f["river_id"]: f["slots"]["place"] for f in feats}
    assert places[1] == "Meghalaya, India" and feats[0]["title"].startswith("Meghalaya, India: ")
    assert places[2] == "South Asia"                 # the named region
    assert places[3] == "60.0°S, 150.0°W"            # the coordinates
    assert ms.region_words(10.4, 19.9) == "Africa" and ms.region_words(47.0, 105.0) == "Asia"   # continents last


# ── the claim lock ───────────────────────────────────────────────────────────


def test_check_wording_holds_numbers_to_the_slots():
    slots = {"place": "Paraná, Brazil", "peak": "2,840 m³/s", "rp": "25-year"}
    ok, why = ms.check_wording("{place}: the river may reach {peak}", slots, 80)
    assert ok == "Paraná, Brazil: the river may reach 2,840 m³/s" and why is None
    for bad in ("{place}: about 3,000 m³/s", "{place}: twice its usual flow", "{place}: a record flood",
                "{place}: 50% of members", "{place}: the {nowhere}", ""):
        text, why = ms.check_wording(bad, slots, 80)
        assert text is None and why
    assert ms.check_wording("{place} " * 20, slots, 80) == (None, "too long")


def _three():
    fs = ms.floods_ahead_findings([_reach(1, -24, -50, 25, 2840, 1050), _reach(2, 60, 120, 100, 1400, 247),
                                   _reach(3, 39, -96, 2, 300, 290)], {"issue_date": "2026-10-09"})
    return ms.rank_findings(fs, max_pins=10)


def test_apply_wording_keeps_the_models_order_and_refuses_numbers():
    fs = _three()
    ids = [f["id"] for f in fs]
    reply = json.dumps({"order": [ids[2], ids[0], "nonsense"], "notes": [
        {"id": ids[2], "title": "{place}: a modest rise", "reason": "The forecast puts it at {peak} by {day}."},
        {"id": ids[0], "title": "{place}: 2,840 cubic metres", "reason": "Big."},
    ]})
    res = ms.apply_wording(fs, "```json\n" + reply + "\n```", by="key")
    out = res["findings"]
    assert [f["id"] for f in out] == [ids[2], ids[0]] and res["ordered"]
    assert [f["rank"] for f in out] == [1, 2]
    assert out[0]["by"] == "key" and out[0]["reason"] == f"The forecast puts it at {fs[2]['slots']['peak']} by " \
        f"{fs[2]['slots']['day']}."
    assert out[1]["by"] == "rules" and out[1]["title"] == fs[0]["title"]      # refused: the template stays
    assert len(res["refused"]) == 1 and ids[0] in res["refused"][0]
    # Not JSON: the rules' order and words.
    res = ms.apply_wording(fs, "I think the Paraná is worst.")
    assert [f["id"] for f in res["findings"]] == ids and not res["ordered"] and res["refused"]


def test_model_wording_uses_the_readers_client_and_the_lock():
    fs = _three()

    class Msg:
        content = json.dumps({"order": [fs[1]["id"]], "notes": [
            {"id": fs[1]["id"], "title": "{place}: high water ahead", "reason": "Forecast at {peak} on {day}."}]})

    class Client:
        class chat:  # noqa: N801 - the OpenAI client's shape
            class completions:  # noqa: N801
                calls: list = []

                @classmethod
                def create(cls, **kw):
                    cls.calls.append(kw)
                    return type("R", (), {"choices": [type("C", (), {"message": Msg})]})

    res = ms.model_wording(fs, client=Client, model="m", provider="p")
    assert res["model"] == "m via p" and res["findings"][0]["by"] == "key"
    sent = Client.chat.completions.calls[0]["messages"]
    assert "Never write a number" in sent[0]["content"] and fs[1]["id"] in sent[1]["content"]
    p = ms.wording_prompt(fs)
    assert p["schema"]["required"] == ["order", "notes"]


# ── one scan, the daily file, the faces ─────────────────────────────────────


@pytest.fixture
def offline(monkeypatch):
    """Every layer from memory: no network."""
    g = _world()
    _put(g, 5, 15, -30, 35, -15)
    _put(g, 1, -70, -10, -50, 0)
    monkeypatch.setattr(ms, "load_status", lambda month=None: ("2026-09", g))
    gj = {"type": "FeatureCollection", "features": [_reach(1, -24, -50, 25, 2840, 1050)]}

    def fake_json(url):
        if url.endswith("latest.geojson"):
            return gj
        if url.endswith("warnings/manifest.json"):
            return {"issue_date": "2026-10-09"}
        return None

    monkeypatch.setattr(ms, "_json", fake_json)
    from aquascope.context import floods_past

    monkeypatch.setattr(floods_past, "load_index", lambda *a, **k: {
        "deg": 0.5, "months": [{"month": f"2025-{m:02d}", "news": 100, "radar": 0} for m in range(1, 13)]})
    r, c = floods_past.cell_of(-6.2, 106.8)
    monkeypatch.setattr(floods_past, "load_month", lambda month, *a, **k: [[r, c, 30, 0]])
    monkeypatch.setattr(ms, "status_months", lambda *a, **k: ["2025-09", "2026-09"], raising=False)
    return g


def test_scan_reads_each_layer_and_ranks(offline):
    res = ms.scan(None, None, gauges=[_gauge(1, 52, -3, pct=0.0)], skill=[_skill("big", -0.6, 300.0, lon=60)],
                  places=False, extra=2)
    assert res["inputs"] == {"status_month": "2026-09", "floods_ahead_issue": "2026-10-09",
                             "floods_past_month": "2025-12", "gauges_date": "2026-10-08",
                             "skill_computed": "2026-10-08"}
    kinds = [f["kind"] for f in res["picks"]]
    assert kinds[:5] == ["floods_ahead", "status", "gauges_today", "floods_past", "models_disagree"]
    assert res["notes"] == [] and res["by"] == "rules"
    assert any("Groundsource" in s for s in res["sources"]) and res["sources"][-1].startswith("Places: Photon")
    json.dumps(res)                                   # what the worker posts back
    # Without rows and without pyarrow the gauges and the evidence are said to be missing, not invented.
    res = ms.scan({"bbox": [-80, -40, 40, 10]}, None, places=False, kinds=["status", "gauges_today"],
                  gauges=None, skill=None, base="file:///nowhere/")
    assert any("snapshot" in n for n in res["notes"]) and {f["kind"] for f in res["findings"]} == {"status"}


def test_daily_file_published_and_read_back(offline, tmp_path, monkeypatch):
    import aquascope.map_layers as ml

    monkeypatch.setattr(ml, "status_months", lambda *a, **k: ["2025-09", "2026-09"])
    today = ([_gauge(1, 52, -3, pct=0.0)], {"date": "2026-10-10"})
    monkeypatch.setattr(ms, "load_gauges_today", lambda base=None: today)
    monkeypatch.setattr(ms, "load_skill", lambda base=None: [_skill("big", -0.6, 300.0)])
    info = ms.daily(tmp_path, places=False)
    doc = json.loads((tmp_path / "scout" / "latest.json").read_text())
    assert info["picks"] == len(doc["picks"]) and (tmp_path / "scout" / f"{doc['date']}.json").exists()
    assert doc["inputs"]["status_month"] == "2026-09" and doc["counts"]["status"] == 2
    status = next(f for f in doc["findings"] if f["kind"] == "status" and f.get("side") == "much_above")
    assert status["record_rank"] == 1 and "southern Africa as a whole" in status["reason"]
    assert not any(k.startswith("_") for f in doc["findings"] for k in f)
    # Read back: the world takes the file's own picks; a view ranks its findings again.
    res = ms.from_published(doc, None)
    assert [f["id"] for f in res["picks"]] == doc["picks"]
    res = ms.from_published(doc, {"bbox": [-80, -40, -40, 10]})
    assert {f["kind"] for f in res["picks"]} <= {"floods_ahead", "status"}
    assert ms.use_published(doc, None, "2026-09", today=doc["date"])
    assert not ms.use_published(doc, None, "2024-07", today=doc["date"])          # another month: scan it
    assert not ms.use_published(doc, {"bbox": [0, 40, 10, 50]}, None, today=doc["date"])   # a small view
    assert not ms.use_published(doc, None, None, today="2099-01-01")               # stale
    seen = {}
    monkeypatch.setattr(ms, "name_places", lambda fs, **k: seen.setdefault("n", len(fs)))
    assert ms.scout_view(None, "2026-09", doc=doc, today=doc["date"])["mode"] == "daily"
    assert ms.scout_view({"bbox": [0, 40, 10, 50]}, None, doc=doc, places=False)["mode"] == "live"
    with pytest.raises(FileNotFoundError):
        ms.publish(tmp_path / "empty")


def test_cli_layers_scout(offline, monkeypatch, capsys):
    from aquascope.cli import main

    monkeypatch.setattr(ms, "load_gauges_today", lambda base=None: (None, None))
    monkeypatch.setattr(ms, "load_skill", lambda base=None: None)
    monkeypatch.setattr("sys.argv", ["aquascope", "layers", "scout", "--no-places", "--pins", "3"])
    main()
    out = capsys.readouterr().out
    assert "Scout (scanned now; status month 2026-09" in out and " 1. " in out and "Source: GEOGLOWS" in out
    monkeypatch.setattr("sys.argv", ["aquascope", "layers", "scout", "--no-places", "--json"])
    main()
    assert json.loads(capsys.readouterr().out)["mode"] == "live"


def test_mcp_tool(offline, monkeypatch):
    from aquascope import mcp_server as m

    monkeypatch.setattr(ms, "name_places", lambda fs, **k: 0)
    monkeypatch.setattr(ms, "load_gauges_today", lambda base=None: (None, None))
    monkeypatch.setattr(ms, "load_skill", lambda base=None: None)
    res = m.map_scout(-80, -40, 40, 10, pins=3)
    assert res["available"] and res["mode"] == "live" and len(res["picks"]) <= 3 and "findings" not in res
    monkeypatch.setattr(ms, "published", lambda *a, **k: None)
    assert m.map_scout(published=True)["available"] is False
    src = (ROOT / "aquascope" / "mcp_server.py").read_text()
    assert "server.tool()(map_scout)" in src


# ── the Explorer and the workflow agree with the package ────────────────────


def test_explorer_kind_labels_match():
    js = (ROOT / "explorer" / "src" / "scout-core.js").read_text()
    block = re.search(r"export const KIND_LABELS = \{(.*?)\};", js, re.S).group(1)
    labels = dict(re.findall(r'(\w+): "([^"]+)"', block))
    assert labels == {k: v["label"] for k, v in ms.KINDS.items()}


def test_worker_runs_the_scout_in_a_light_worker():
    worker = (ROOT / "explorer" / "worker.js").read_text()
    assert '"scout"]);' in worker and "_sc.scout_view(" in worker and "_sc.apply_wording(" in worker


def test_daily_workflow_writes_and_publishes_scout_only():
    import yaml

    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "flood-warnings.yml").read_text())
    steps = {s.get("name"): s for s in wf["jobs"]["warnings"]["steps"]}
    run = steps["Today's scout"]["run"]
    assert "python -m aquascope.map_scout daily --out build --warnings build" in run
    assert "inputs.smoke == ''" in steps["Today's scout"]["if"]
    assert steps["Publish scout/ to Hugging Face"]["run"].strip() == "python -m aquascope.map_scout publish --out build"
    src = (ROOT / "aquascope" / "map_scout.py").read_text()
    assert 'allow_patterns=[f"{FOLDER}/*"]' in src and 'FOLDER = "scout"' in src


def test_the_page_fetches_the_names_and_the_package_applies_them(offline):
    res = ms.scan(None, None, gauges=[], skill=[], places="page")
    cands = res["candidates"]
    assert cands and all(f["place_url"].startswith(ms.PHOTON_REVERSE + "?lat=") for f in cands)
    amazon = next(f for f in cands if f["kind"] == "status" and f["side"] == "much_below")
    assert amazon["slots"]["place"] == "The Amazon" and amazon["title"].startswith("The Amazon: ")   # meanwhile
    answers = [{"features": [{"properties": {"state": "Amazonas", "country": "Brazil"}}]}] + [None] * len(cands)
    assert ms.apply_places(cands, answers) == 1
    assert cands[0]["slots"]["place"] == "Amazonas, Brazil" and cands[0]["placed_by"] == "photon"
    assert not any("place_url" in f for f in cands)
