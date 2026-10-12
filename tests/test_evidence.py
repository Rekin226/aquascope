"""aquascope.evidence: the model-skill ladder on synthetic series whose skill is known, with no network (#518)."""

from __future__ import annotations

import asyncio
import json
import math
import sys

import numpy as np
import pandas as pd
import pytest

from aquascope import evidence

# ── synthetic records ────────────────────────────────────────────────────────


def _gauge(years: int = 30, seed: int = 1) -> pd.Series:
    """A daily flow with a seasonal cycle, noise and a few floods a year: positive, skewed, like a river."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-01", periods=int(years * 365.25), freq="D")
    doy = idx.dayofyear.to_numpy()
    base = 40 + 25 * np.sin(2 * np.pi * doy / 365.25)
    q = base * np.exp(rng.normal(0, 0.35, len(idx)))
    return pd.Series(q, index=idx)


# ── the grade ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("kge", "letter"), [(0.9, "A"), (0.75, "A"), (0.6, "B"), (0.5, "B"), (0.0, "C"),
                                             (-0.4, "C"), (-0.41, "D"), (-3.0, "D")])
def test_grade_bands_follow_the_documented_thresholds(kge, letter):
    assert evidence.grade(kge)["grade"] == letter


def test_a_flood_error_beyond_half_lowers_the_grade_one_letter():
    assert evidence.grade(0.8, 60.0) == {"grade": "B", "why": "KGE 0.80, lowered from A because the flood flow is "
                                                              "off by 60 %"}
    assert evidence.grade(0.8, -49.0)["grade"] == "A"
    assert evidence.grade(-1.0, 90.0)["grade"] == "D"
    assert evidence.grade(None)["grade"] is None


# ── scoring one model ───────────────────────────────────────────────────────


def test_a_perfect_model_scores_one_and_grade_a():
    obs = _gauge()
    row = evidence.score(obs, obs.copy(), model="geoglows")
    assert row["kge"] == pytest.approx(1.0) and row["nse"] == pytest.approx(1.0)
    assert row["r"] == pytest.approx(1.0) and row["alpha"] == pytest.approx(1.0) and row["beta"] == pytest.approx(1.0)
    assert row["pbias"] == pytest.approx(0.0)
    assert row["q100_error_pct"] == pytest.approx(0.0) and row["flood_years"] == 30
    assert row["grade"] == "A"
    assert row["sentence"] == "GEOGLOWS v2's 100-year flow is within 0 % of the gauge's."


def test_a_model_sixty_percent_high_has_the_kge_the_formula_gives_and_is_demoted():
    obs = _gauge()
    row = evidence.score(obs, obs * 1.6, model="glofas")
    # r = 1, alpha = beta = 1.6: KGE = 1 - sqrt(2 * 0.36)
    assert row["kge"] == pytest.approx(1 - math.sqrt(0.72), abs=1e-3)
    assert row["beta"] == pytest.approx(1.6) and row["alpha"] == pytest.approx(1.6)
    assert row["pbias"] == pytest.approx(60.0, abs=0.01)
    for t in (2, 10, 100):
        assert row[f"q{t}_error_pct"] == pytest.approx(60.0, abs=0.2)
    assert row["grade"] == "D" and "lowered from C" in row["why"]
    assert row["sentence"] == "GloFAS v4's 100-year flow is 60 % above the gauge's."


def test_noise_with_no_relation_to_the_gauge_does_no_better_than_the_mean():
    obs = _gauge(seed=1)
    other = _gauge(seed=99).sample(frac=1.0, random_state=3)
    other.index = obs.index
    row = evidence.score(obs, other, model="nwm")
    assert abs(row["r"]) < 0.1
    assert row["grade"] in ("C", "D")
    assert row["nse"] < 0


def test_scores_only_the_shared_days():
    obs = _gauge(years=10)
    sim = obs.iloc[: 365 * 5] * 0.5
    row = evidence.score(obs, sim, model="grrr")
    assert row["n_days"] == 365 * 5
    assert row["end"] == sim.index.max().strftime("%Y-%m-%d")
    assert row["flood_years"] < 10 and row["q100_error_pct"] is None and "flood_note" in row
    assert row["sentence"] == "Google GRRR's mean flow is 50 % below the gauge's."


def test_a_short_overlap_is_scored_but_not_graded():
    obs = _gauge(years=2)
    row = evidence.score(obs, obs, model="geoglows")
    assert row["kge"] == pytest.approx(1.0) and row["grade"] is None and "fewer than three years" in row["why"]


def test_a_model_site_on_another_river_is_not_graded():
    obs = _gauge()
    row = evidence.score(obs, obs, model="geoglows", site={"site_id": "1", "area_ratio": 3.2, "match": "area"})
    assert row["grade"] is None and "3.20 times" in row["why"]
    assert row["site_id"] == "1" and row["match"] == "area"


def test_summarize_names_the_best_and_the_largest_flood_disagreement():
    obs = _gauge()
    rows = [evidence.score(obs, obs * 1.38, model="geoglows"), evidence.score(obs, obs * 0.95, model="glofas")]
    out = evidence.summarize(rows)
    assert out["best"] == "glofas" and out["best_grade"] == "A"
    assert out["disagreement"] == "GEOGLOWS v2's 100-year flow is 38 % above the gauge's."
    assert out["sentence"].startswith("GloFAS v4 fits this gauge best (KGE 0.93, grade A).")
    assert evidence.summarize([])["sentence"] == "No model could be graded at this gauge."


# ── where the models are read ───────────────────────────────────────────────


def test_geoglows_site_picks_the_reach_whose_upstream_area_matches(monkeypatch):
    from aquascope import rivers

    near = [{"river_id": 110000001, "distance_m": 40.0, "lat": 1.0, "lon": 2.0},     # a tributary, nearest
            {"river_id": 110000002, "distance_m": 300.0, "lat": 1.0, "lon": 2.0}]    # the main stem
    areas = {110000001: 85.0, 110000002: 9800.0}
    monkeypatch.setattr(rivers, "reaches_near", lambda lat, lon, max_distance_m, limit: near)
    monkeypatch.setattr(rivers, "upstream_area", lambda rid, lat=None, lon=None: {"upstream_area_km2": areas[rid]})
    site = evidence.geoglows_site(1.0, 2.0, 10_000.0)
    assert site["site_id"] == "110000002" and site["match"] == "area" and site["area_ratio"] == pytest.approx(0.98)
    nearest = evidence.geoglows_site(1.0, 2.0, None)
    assert nearest["site_id"] == "110000001" and nearest["match"] == "nearest" and nearest["area_ratio"] is None
    monkeypatch.setattr(rivers, "reaches_near", lambda lat, lon, max_distance_m, limit: [])
    assert "no GEOGLOWS reach" in evidence.geoglows_site(1.0, 2.0, 100.0)["error"]


def _patch_models(monkeypatch, obs: pd.Series):
    monkeypatch.setattr(evidence, "geoglows_site", lambda lat, lon, area: {
        "site_id": "110000002", "distance_m": 300.0, "model_area_km2": 9800.0, "area_ratio": 0.98, "match": "area"})
    monkeypatch.setattr(evidence, "geoglows_series", lambda rid: obs * 1.38)
    monkeypatch.setattr(evidence, "glofas_site", lambda lat, lon, mean, area=None: {
        "site_id": "1.000,2.000", "site_lat": 1.0, "site_lon": 2.0, "distance_m": 0.0, "area_ratio": None,
        "match": "mean_flow"})
    monkeypatch.setattr(evidence, "glofas_series", lambda lat, lon, start, end: obs * 0.95)


def test_model_skill_scores_the_live_models_and_merges_the_published_ones(monkeypatch):
    obs = _gauge()
    _patch_models(monkeypatch, obs)
    published = [{"model": "nwm", "grade": "B", "kge": 0.62, "q100_error_pct": -12.0, "source": "usgs",
                  "station_id": "USGS-1"},
                 {"model": "geoglows", "grade": "D", "kge": 0.1}]          # a live model is not taken from the table
    res = evidence.model_skill(series=obs, lat=1.0, lon=2.0, area_km2=10_000.0, published=published,
                               include_series=True)
    by = {r["model"]: r for r in res["models"]}
    assert set(by) == {"geoglows", "glofas", "nwm"}
    assert by["geoglows"]["site_id"] == "110000002" and by["geoglows"]["grade"] in ("C", "D")
    assert by["glofas"]["grade"] == "A" and "flatters its bias" in by["glofas"]["note"]
    assert by["nwm"]["from_table"] is True and by["nwm"]["label"] == "NWM v3.0"
    assert by["nwm"]["sentence"] == "NWM v3.0's 100-year flow is 12 % below the gauge's."
    assert res["best"] == "glofas"
    assert "GEOGLOWS v2's 100-year flow is 38 % above the gauge's." in res["sentence"]
    assert set(res["series"]) == {"observed", "geoglows", "glofas"} and len(res["series"]["observed"]["t"]) <= 4000
    assert res["grading"]["bands"][0] == {"grade": "A", "min_kge": 0.75, "words": "tracks the gauge closely"}
    assert "CC BY 4.0" in res["attribution"]
    json.dumps(res)   # JSON-able, as the worker and the MCP server need


def test_model_skill_keeps_going_when_one_model_fails(monkeypatch):
    obs = _gauge()
    _patch_models(monkeypatch, obs)

    def down(*a, **k):
        raise RuntimeError("GEOGLOWS is down")

    monkeypatch.setattr(evidence, "geoglows_series", down)
    res = evidence.model_skill(series=obs, lat=1.0, lon=2.0, area_km2=10_000.0, published=[])
    by = {r["model"]: r for r in res["models"]}
    assert by["geoglows"]["grade"] is None and "GEOGLOWS is down" in by["geoglows"]["why"]
    assert by["glofas"]["grade"] == "A"
    assert any("monthly skill run" in n for n in res["notes"])


def test_model_skill_says_so_before_glofas_begins(monkeypatch):
    obs = _gauge()
    obs.index = obs.index - pd.DateOffset(years=40)   # 1950 to 1980
    _patch_models(monkeypatch, obs)
    res = evidence.model_skill(series=obs, lat=1.0, lon=2.0, models=["glofas"], published=[])
    assert res["models"][0]["why"] == "the gauge record ends before GloFAS begins (1984)"


def test_model_skill_needs_a_record_and_a_place():
    with pytest.raises(ValueError):
        evidence.model_skill()
    with pytest.raises(ValueError):
        evidence.model_skill(series=_gauge())
    empty = evidence.model_skill(series=pd.Series(dtype=float), lat=1.0, lon=2.0)
    assert empty["models"] == [] and "no daily discharge" in empty["error"]


def test_model_skill_does_not_score_a_record_in_another_unit(monkeypatch):
    from aquascope import explore

    monkeypatch.setattr(explore, "fetch_series", lambda source, station_id, variable=None: {
        "series": _gauge() * 35.3147, "variable": "discharge", "unit": "ft3/s"})
    res = evidence.model_skill("x", "1", lat=1.0, lon=2.0, area_km2=100.0, published=[])
    assert res["models"] == [] and "m3/s" in res["error"]
    assert any("ft3/s" in n for n in res["notes"])


def test_model_skill_scores_a_station_record_in_m3s(monkeypatch):
    from aquascope import explore

    obs = _gauge()
    _patch_models(monkeypatch, obs)
    monkeypatch.setattr(explore, "fetch_series", lambda source, station_id, variable=None: {
        "series": obs, "variable": "discharge", "unit": "m³/s"})
    res = evidence.model_skill("x", "1", lat=1.0, lon=2.0, area_km2=100.0, models=["geoglows"], published=[])
    assert res["models"][0]["kge"] is not None and not res.get("error")


def test_model_skill_keeps_the_last_n_years(monkeypatch):
    obs = _gauge(years=40)
    _patch_models(monkeypatch, obs)
    res = evidence.model_skill(series=obs, lat=1.0, lon=2.0, models=["geoglows"], published=[], years=20)
    assert res["obs_start"] >= "2009-01-01"
    assert res["models"][0]["n_days"] <= int(20 * 365.25) + 1


# ── which model to lean on near a site ──────────────────────────────────────


def _frame() -> pd.DataFrame:
    rows = []
    for i, (lat, lon) in enumerate([(1.0, 2.0), (1.1, 2.1), (1.2, 2.0), (9.0, 9.0)]):
        rows.append({"source": "x", "station_id": str(i), "lat": lat, "lon": lon, "model": "geoglows",
                     "grade": "C", "kge": 0.3})
        rows.append({"source": "x", "station_id": str(i), "lat": lat, "lon": lon, "model": "grrr", "grade": "B",
                     "kge": 0.6 + 0.01 * i})
    rows.append({"source": "x", "station_id": "9", "lat": 1.0, "lon": 2.0, "model": "nwm", "grade": None,
                 "kge": 0.99})
    return pd.DataFrame(rows)


def test_lean_on_names_the_model_with_the_best_median_kge_nearby():
    out = evidence.lean_on(1.0, 2.0, radius_km=100.0, frame=_frame())
    assert out["model"] == "grrr" and out["n_gauges"] == 3
    assert out["median_kge"] == pytest.approx(0.61)
    assert out["sentence"].startswith("Near this site, Google GRRR tracked the gauges best: median KGE 0.61 at 3")
    assert "GEOGLOWS v2 0.30 at 3" in out["sentence"] and "nwm" not in out["by_model"]


def test_lean_on_says_when_there_is_nothing_nearby():
    assert evidence.lean_on(-40.0, -40.0, frame=_frame())["model"] is None
    assert "No published model skill" in evidence.lean_on(0, 0, frame=pd.DataFrame())["sentence"]
    one = _frame()
    one = one[one["station_id"] == "0"]
    assert "Fewer than two graded gauges" in evidence.lean_on(1.0, 2.0, frame=one)["sentence"]


def test_published_rows_selects_one_gauge():
    rows = evidence.published_rows("x", "1", frame=_frame())
    assert {r["model"] for r in rows} == {"geoglows", "grrr"} and all(r["station_id"] == "1" for r in rows)
    assert evidence.published_rows("x", "1", frame=pd.DataFrame()) == []


# ── the faces: CLI and MCP ──────────────────────────────────────────────────


def test_cli_evidence_skill_on_a_csv(monkeypatch, capsys, tmp_path):
    from aquascope import cli

    obs = _gauge()
    _patch_models(monkeypatch, obs)
    path = tmp_path / "flow.csv"
    obs.rename("value").rename_axis("date").to_csv(path)
    monkeypatch.setattr(evidence, "published_rows", lambda s, i: [])
    monkeypatch.setattr(sys, "argv", ["aquascope", "evidence", "skill", "--csv", str(path), "--at", "1", "2",
                                      "--area", "10000", "--models", "geoglows", "glofas"])
    cli.main()
    out = capsys.readouterr().out
    assert "GloFAS v4 fits this gauge best" in out and "GEOGLOWS v2" in out and "Grades: A KGE >= 0.75" in out


def test_cli_evidence_near(monkeypatch, capsys):
    from aquascope import cli

    monkeypatch.setattr(evidence, "lean_on", lambda lat, lon, radius_km=150.0: {"model": "grrr", "sentence": "Lean."})
    monkeypatch.setattr(sys, "argv", ["aquascope", "evidence", "near", "1", "2"])
    cli.main()
    assert capsys.readouterr().out.strip() == "Lean."


def test_mcp_tools_are_registered_and_drop_the_series(monkeypatch):
    pytest.importorskip("mcp")
    from aquascope import mcp_server as m

    seen = {}

    def fake(source, station_id, **kw):
        seen.update(kw, source=source, station_id=station_id)
        return {"models": [], "series": {"observed": {}}, "sentence": "s"}

    monkeypatch.setattr(evidence, "model_skill", fake)
    out = m.model_skill("usgs", "USGS-1", years=20)
    assert "series" not in out and seen["source"] == "usgs" and seen["years"] == 20
    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert {"model_skill", "model_to_lean_on"} <= names
