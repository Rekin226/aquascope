"""aquascope.bulletin: the monthly state of the rivers (#523), on synthetic records, no network."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from aquascope import bulletin

# ── synthetic records ───────────────────────────────────────────────────────


def daily(start_year: int, end_year: int, month: int, level: dict[int, float] | float, *, days: int | None = None,
          last_days: int | None = None, end: int | None = None) -> pd.DataFrame:
    """A daily record covering ``month`` in each year, at ``level`` (per year or flat), as date/value rows."""
    rows = []
    for y in range(start_year, end_year + 1):
        n = pd.Period(f"{y}-{month:02d}").days_in_month
        take = n if days is None else days
        if y == end_year and last_days is not None:
            take = last_days
        v = level[y] if isinstance(level, dict) else level
        for d in range(1, take + 1):
            rows.append((pd.Timestamp(y, month, d), v))
    return pd.DataFrame(rows, columns=["date", "value"])


def frame(source: str, sid: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "station_id", sid)
    out.insert(0, "source", source)
    return out


# ── months, licences ─────────────────────────────────────────────────────────


def test_a_month_is_read_from_text_or_a_date_and_defaults_to_the_last_full_one():
    assert bulletin.parse_month("2026-09") == (2026, 9)
    assert bulletin.parse_month("2026-09-14") == (2026, 9)
    assert bulletin.parse_month(date(2025, 2, 3)) == (2025, 2)
    assert bulletin.parse_month(None, today=date(2026, 1, 3)) == (2025, 12)
    assert bulletin.parse_month(None, today="2026-10-03") == (2026, 9)
    for bad in ("2026", "2026-13", "Sept", "2026/xx"):
        with pytest.raises(ValueError, match="not a month"):
            bulletin.parse_month(bad)


def test_only_sources_whose_licence_asks_for_attribution_alone_are_used():
    assert bulletin.licence_ok("usgs")            # public domain
    assert bulletin.licence_ok("uk_ea")           # OGL v3
    assert bulletin.licence_ok("hubeau_hydrometrie")
    assert not bulletin.licence_ok("greece_openhi")   # CC BY-SA 4.0
    assert not bulletin.licence_ok("no_such_source")


# ── the numbers ─────────────────────────────────────────────────────────────


def test_monthly_means_keep_one_calendar_month_per_year_with_its_day_count():
    df = frame("usgs", "A", daily(2000, 2002, 9, {2000: 1.0, 2001: 2.0, 2002: 4.0}, last_days=10))
    other = frame("usgs", "A", daily(2000, 2000, 8, 99.0))
    m = bulletin.monthly_means(pd.concat([df, other]), 9)
    assert list(m.columns) == ["source", "station_id", "year", "mean", "n_days"]
    assert m["year"].tolist() == [2000, 2001, 2002]
    assert m["mean"].tolist() == [1.0, 2.0, 4.0]
    assert m["n_days"].tolist() == [30, 30, 10]
    assert bulletin.monthly_means(None, 9).empty


def test_a_station_is_classed_by_its_mid_rank_percentile_and_names_a_new_record():
    years = list(range(2000, 2016))
    means = [float(i) for i in range(1, 16)] + [20.0]   # 2015 is the highest of 16
    days = [30] * 16
    st = bulletin.classify_station(years, means, days, 2015)
    assert st["percentile"] == 100.0 and st["class"] == "much_above" and st["n_years"] == 15
    assert st["record"] == "high" and st["previous"] == 15.0 and st["previous_year"] == 2014
    assert st["rank"] == 1 and st["median"] == 8.0 and st["ratio"] == 2.5
    mid = bulletin.classify_station(years, means[:-1] + [8.0], days, 2015)
    assert mid["class"] == "normal" and mid["record"] is None
    assert mid["percentile"] == pytest.approx(100 * (7 + 0.5) / 15, abs=0.05)
    low = bulletin.classify_station(years, means[:-1] + [0.5], days, 2015)
    assert low["record"] == "low" and low["class"] == "much_below" and low["previous"] == 1.0


def test_a_station_that_cannot_be_classed_says_why():
    years = list(range(2000, 2016))
    days = [30] * 16
    assert bulletin.classify_station(years[:-1], [1.0] * 15, days[:-1], 2015)["reason"] == "no_days"
    assert bulletin.classify_station(years, [1.0] * 16, days[:-1] + [24], 2015)["reason"] == "few_days"
    short = bulletin.classify_station(years[-6:], [1.0] * 6, [30] * 6, 2015)
    assert short["reason"] == "short_record" and short["n_years"] == 5
    # a year with fewer than 25 days does not count among the ten
    thin = bulletin.classify_station(years, [1.0] * 16, [24] * 6 + [30] * 10, 2015)
    assert thin["reason"] == "short_record" and thin["n_years"] == 9


def test_a_negative_monthly_mean_is_classed_without_a_ratio():
    st = bulletin.classify_station(list(range(2000, 2012)), [1.0] * 11 + [-2.0], [30] * 12, 2011)
    assert st["class"] == "much_below" and st["ratio"] is None and st["record"] == "low"


def _means(rows: list[tuple[str, str, dict[int, float]]], days: int = 30) -> pd.DataFrame:
    out = []
    for src, sid, by_year in rows:
        for y, v in by_year.items():
            out.append({"source": src, "station_id": sid, "year": y, "mean": v, "n_days": days})
    return pd.DataFrame(out)


def _history(target: float, base: float = 10.0) -> dict[int, float]:
    h = {y: base + (y - 2000) for y in range(2000, 2015)}   # 10 .. 24
    h[2015] = target
    return h


CATALOG = [
    {"source": "usgs", "station_id": "U1", "name": "BIG RIVER AT TOWN", "river": None, "country": "USA",
     "latitude": 40.0, "longitude": -75.0},
    {"source": "usgs", "station_id": "U2", "name": "SMALL CREEK", "country": "USA", "latitude": 41.0,
     "longitude": -74.0},
    {"source": "usgs", "station_id": "U3", "name": "MID RUN", "country": "USA", "latitude": 41.5, "longitude": -74.5},
    {"source": "uk_ea", "station_id": "E1", "name": "Kingston", "river": "River Thames", "country": "GBR",
     "latitude": 51.4, "longitude": -0.3},
    {"source": "uk_ea", "station_id": "E2", "name": "Ewell", "river": "Hogsmill River", "country": "GBR",
     "latitude": 51.35, "longitude": -0.25},
]


def _sample() -> dict:
    means = _means([
        ("usgs", "U1", _history(100.0, base=50.0)),   # far above anything: a new September high
        ("usgs", "U2", _history(17.0)),               # mid: normal
        ("usgs", "U3", _history(25.0)),               # above the highest (24): a record high, much above
        ("uk_ea", "E1", _history(1.0, base=20.0)),    # a new low
        ("uk_ea", "E2", _history(10.5)),              # low but not a record: much below
        ("uk_ea", "E3", {2014: 1.0, 2015: 1.0}),      # too short a record
    ])
    basins = {("uk_ea", "E1"): 2120000010, ("uk_ea", "E2"): 2120000010, ("usgs", "U1"): 7120000020,
              ("usgs", "U2"): 7120000020, ("usgs", "U3"): 7120000020}
    return bulletin.compute_bulletin(
        means, "2015-09", catalog=CATALOG, basins=basins,
        considered=[("uk_ea", "E4"), ("usgs", "U1")], left_out_sources={"greece_openhi": 6},
        topped_up={"asked": 2, "added": 1, "failed": 1, "skipped_cap": 0, "skipped_time": 0},
        made="2015-10-03T07:00:00Z")


def test_the_bulletin_counts_rolls_up_and_names_the_notable_gauges():
    b = _sample()
    assert b["month"] == "2015-09" and b["label"] == "September 2015"
    cov = b["coverage"]
    assert cov["considered"] == 7 and cov["classed"] == 5      # E4 had no record at all
    assert cov["excluded"] == {"no_days": 1, "few_days": 0, "short_record": 1}
    assert cov["by_source"]["uk_ea"] == {"considered": 4, "classed": 2}
    assert cov["left_out_sources"] == {"greece_openhi": 6}
    assert sum(b["counts"].values()) == 5
    assert b["counts"]["much_above"] == 2 and b["counts"]["normal"] == 1 and b["counts"]["much_below"] == 2
    names = {c["country"]: c for c in b["countries"]}
    assert names["USA"]["name"] == "United States" and names["USA"]["n"] == 3
    assert names["GBR"]["counts"]["much_below"] == 2 and names["GBR"]["median_class"] == "much_below"
    basins = {x["basin_id"]: x for x in b["basins"]}
    assert set(basins) == {7120000020}        # the British basin has only two classed gauges
    assert basins[7120000020]["label"] == "BIG RIVER AT TOWN"    # the largest usual flow names it
    nt = b["notable"]
    assert nt["n_record_high"] == 2 and nt["n_record_low"] == 1
    assert nt["record_high"][0]["station_id"] == "U1"   # the largest share of its usual flow first
    assert nt["record_low"][0]["previous"] == 20.0 and nt["record_low"][0]["previous_year"] == 2000
    assert [r["station_id"] for r in nt["furthest_below"]][:1] == ["E1"]
    assert b["shares"]["much_above"] == 40
    sources = {c["source"] for c in b["credits"]}
    assert {"usgs", "uk_ea", "greece_openhi", "basinatlas"} <= sources


def test_the_summary_is_written_by_rules_from_the_numbers():
    b = _sample()
    s = b["summary"]
    assert s.startswith("In September 2015, 5 gauges in 2 countries had at least 25 days of flow and 10 or more")
    assert "Flow was normal at 20 % of them, below or much below normal at 40 % and above or much above normal at " \
           "40 %." in s
    assert "1 gauge set a new September low and 2 set a new September high." in s
    assert "driest" not in s        # no country has the 10 gauges a comparison needs
    assert b["headline"] == "September 2015: 40 % of 5 gauges below normal, 20 % normal, 40 % above."
    empty = bulletin.compute_bulletin(pd.DataFrame(columns=["source", "station_id", "year", "mean", "n_days"]),
                                      "2015-09")
    assert empty["coverage"]["classed"] == 0
    assert "classes none" in empty["summary"]


def test_the_driest_and_wettest_countries_are_named_with_enough_gauges():
    rows = [("usgs", f"U{i}", _history(30.0 + i)) for i in range(10)]
    rows += [("uk_ea", f"E{i}", _history(5.0)) for i in range(10)]
    cat = [{"source": s, "station_id": sid, "country": "USA" if s == "usgs" else "GBR"} for s, sid, _ in rows]
    b = bulletin.compute_bulletin(_means(rows), "2015-09", catalog=cat)
    assert "The United Kingdom was the driest, its median gauge at the 0th percentile (much below normal); the " \
           "United States the wettest, at the 100th (much above normal)." in b["summary"]
    assert "No gauge set" not in b["summary"]


# ── the map ─────────────────────────────────────────────────────────────────


def test_map_regions_split_far_apart_gauges_and_leave_strays_out():
    europe = [(0.0 + i * 0.5, 50.0 + i * 0.2) for i in range(20)]
    us = [(-75.0 + i * 0.3, 41.0) for i in range(8)]
    stray = [(-93.0, 31.0)]
    panels, left = bulletin.map_regions(europe + us + stray)
    assert len(panels) == 2 and left == 1
    assert len(panels[0]["idx"]) == 20 and len(panels[1]["idx"]) == 8
    w, s, e, n = panels[1]["bbox"]
    assert w == -75.0 and n == 41.0
    assert bulletin.map_regions([]) == ([], 0)


# ── the document ────────────────────────────────────────────────────────────


def test_the_document_renders_to_html_and_markdown_with_its_tables():
    b = _sample()
    html = bulletin.render_bulletin(b, "html")
    assert "<title>State of the rivers · September 2015</title>" in html
    assert "Monthly bulletin" in html and "By country" in html and "United States" in html
    assert "New September high" in html and "Data and licences" in html and "Natural Earth" in html
    assert "DRAFT" not in html          # a bulletin is not a study awaiting sign-off
    md = bulletin.render_bulletin(b, "md")
    assert md.startswith("# State of the rivers")
    assert "| Country | Gauges | Much below | Below | Normal | Above | Much above | Median percentile |" in md
    assert "Greece OpenHi.net" in md and "left out" in md
    assert "![Figure" not in md        # no map without a figure
    with_map = bulletin.render_bulletin(b, "md", figure=b"\x89PNG", outside=1)
    assert "![Figure 1](map.png)" in with_map and "1 gauge far from the others is not drawn." in with_map
    assert "\u2014" not in html and "\u2014" not in md


def test_write_bulletin_writes_the_month_folder(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("matplotlib")
    import pyarrow.parquet as pq

    from aquascope import river_path

    monkeypatch.setattr(river_path, "_countries", lambda: [])
    b = _sample()
    written = bulletin.write_bulletin(b, tmp_path)
    folder = tmp_path / "bulletins" / "2015-09"
    for name in ("bulletin.html", "bulletin.md", "bulletin.json", "status.parquet"):
        assert (folder / name).exists(), name
    assert "map" not in written        # five gauges in two far-apart groups: no group has the five a panel needs
    t = pq.read_table(folder / "status.parquet")
    assert t.column_names == bulletin.STATUS_COLUMNS
    assert t.num_rows == 5 and set(t.column("class").to_pylist()) == {"much_above", "normal", "much_below"}
    back = json.loads((folder / "bulletin.json").read_text(encoding="utf-8"))
    assert back["month"] == "2015-09" and len(back["gauges"]) == 5


def test_the_map_is_drawn_when_a_region_has_gauges(monkeypatch):
    pytest.importorskip("matplotlib")
    from aquascope import river_path

    monkeypatch.setattr(river_path, "_countries", lambda: [])
    b = _sample()
    b["gauges"] = [{**g, "latitude": 51.0 + i * 0.1, "longitude": -1.0 + i * 0.1} for i, g in enumerate(b["gauges"])]
    png, left = bulletin.map_png(b, dpi=60)
    assert png[:4] == b"\x89PNG" and left == 0


def test_the_index_keeps_every_month_newest_first():
    b = _sample()
    idx = bulletin.update_index({"months": [{"month": "2015-08"}, {"month": "2015-09", "old": True}]}, b)
    assert idx["latest"] == "2015-09"
    assert [m["month"] for m in idx["months"]] == ["2015-09", "2015-08"]
    assert idx["months"][0]["classed"] == 5 and "old" not in idx["months"][0]
    assert bulletin.update_index(None, b)["months"][0]["headline"].startswith("September 2015")


# ── building from an Archive copy ───────────────────────────────────────────


def _bundle(path, rows: pd.DataFrame) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    t = pa.table({"station_id": rows["station_id"].astype(str).tolist(),
                  "date": pa.array([d.date() for d in rows["date"]], type=pa.date32()),
                  "value": rows["value"].astype(float).tolist()})
    pq.write_table(t, path)


@pytest.fixture
def archive(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    hist = {y: 10.0 + (y - 2010) for y in range(2010, 2026)}   # 10 .. 25
    a = pd.concat([
        frame("usgs", "U1", daily(2010, 2026, 9, {**hist, 2026: 40.0})),          # a new high
        frame("usgs", "U2", daily(2010, 2026, 9, {**hist, 2026: 12.0}, last_days=10)),   # short month
        frame("usgs", "U3", daily(2020, 2026, 9, 5.0)),                             # short record
    ])
    _bundle(tmp_path / "obs" / "discharge" / "usgs.parquet", a.drop(columns=["source"]))
    g = frame("greece_openhi", "G1", daily(2010, 2026, 9, 3.0))
    _bundle(tmp_path / "obs" / "discharge" / "greece_openhi.parquet", g.drop(columns=["source"]))
    monkeypatch.setattr(bulletin, "_catalog_rows", lambda root: [
        {"source": "usgs", "station_id": "U1", "name": "ONE", "country": "USA", "latitude": 40.0, "longitude": -75.0},
    ])
    monkeypatch.setattr(bulletin, "_basin_ids", lambda root: ({}, {}))
    return tmp_path


def test_read_month_takes_one_calendar_month_and_each_station_last_day(archive):
    sub, last = bulletin.read_month(archive / "obs" / "discharge" / "usgs.parquet", "usgs", 9)
    assert set(sub["station_id"]) == {"U1", "U2", "U3"} and (sub["date"].dt.month == 9).all()
    lasts = dict(zip(last["station_id"], last["last"].dt.date))
    assert lasts["U2"] == date(2026, 9, 10) and lasts["U1"] == date(2026, 9, 30)


def test_build_bulletin_reads_the_bundles_and_leaves_share_alike_sources_out(archive):
    b = bulletin.build_bulletin("2026-09", archive=archive, today=date(2026, 10, 3))
    cov = b["coverage"]
    assert b["origin"] == "built" and not b["smoke"]
    assert cov["considered"] == 3 and cov["classed"] == 1
    assert cov["excluded"]["few_days"] == 1 and cov["excluded"]["short_record"] == 1
    assert cov["left_out_sources"] == {"greece_openhi": 1}
    assert b["gauges"][0]["record"] == "high" and b["gauges"][0]["name"] == "ONE"
    with pytest.raises(ValueError, match="not over yet"):
        bulletin.build_bulletin("2026-10", archive=archive, today=date(2026, 10, 3))
    with pytest.raises(ValueError, match="no mirrored discharge for nowhere"):
        bulletin.build_bulletin("2026-09", ["nowhere"], archive=archive, today=date(2026, 10, 3))


def test_the_agency_fills_the_days_the_mirror_does_not_have_yet(archive, monkeypatch):
    from aquascope import explore

    asked = []

    def fake_fetch(source, station_id, **kw):
        asked.append((source, station_id, kw.get("prefer_archive")))
        idx = pd.date_range("2026-09-01", "2026-09-30", freq="D")
        return {"series": pd.Series(np.full(len(idx), 12.0), index=idx), "variable": "discharge", "unit": "m3/s"}

    monkeypatch.setattr(explore, "fetch_series", fake_fetch)
    b = bulletin.build_bulletin("2026-09", archive=archive, top_up=5, workers=1, today=date(2026, 10, 3))
    assert asked == [("usgs", "U2", False)]       # only the gauge a full month would class
    top = b["coverage"]["top_up"]
    assert top["asked"] == 1 and top["added"] == 1
    assert b["coverage"]["classed"] == 2
    u2 = next(g for g in b["gauges"] if g["station_id"] == "U2")
    assert u2["n_days"] == 30 and u2["value"] == 12.0
    # a month too long ago is not topped up
    old = bulletin.build_bulletin("2024-09", archive=archive, top_up=5, workers=1, today=date(2026, 10, 3))
    assert old["coverage"]["top_up"]["asked"] == 0


def test_a_failed_top_up_keeps_the_mirror(archive, monkeypatch):
    from aquascope import explore

    def boom(*a, **k):
        raise RuntimeError("agency down")

    monkeypatch.setattr(explore, "fetch_series", boom)
    b = bulletin.build_bulletin("2026-09", archive=archive, top_up=5, workers=1, today=date(2026, 10, 3))
    assert b["coverage"]["top_up"]["failed"] == 1 and b["coverage"]["classed"] == 1


def test_a_smoke_build_is_marked_and_never_published(archive, tmp_path, monkeypatch):
    from aquascope import river_path
    from aquascope.archive import publish as pub

    monkeypatch.setattr(river_path, "_countries", lambda: [])
    monkeypatch.setattr(bulletin, "_today", lambda: date(2026, 10, 3))
    out = tmp_path / "out"
    assert bulletin.main(["build", "--archive", str(archive), "--out", str(out), "--month", "2026-09",
                          "--smoke"]) == 0
    assert json.loads((out / "bulletins" / "index.json").read_text())["latest"] == "2026-09"
    with pytest.raises(RuntimeError, match="smoke"):
        bulletin.publish(out)
    calls = []

    def fake_publish(folder, repo_id, **kw):
        calls.append(sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()))
        calls.append(kw)
        return "https://huggingface.co/commit/x"

    monkeypatch.setattr(pub, "publish_folder", fake_publish)
    assert bulletin.main(["build", "--archive", str(archive), "--out", str(out), "--month", "2026-09"]) == 0
    assert bulletin.publish(out, token="t") == "https://huggingface.co/commit/x"
    files, kw = calls
    assert all(f.startswith("bulletins/") for f in files)
    assert "bulletins/index.json" in files and "bulletins/2026-09/bulletin.html" in files
    assert "bulletins/2026-09/status.parquet" in files
    assert "*.html" in kw["allow_patterns"] and "*.md" in kw["allow_patterns"]


# ── published first ─────────────────────────────────────────────────────────


def test_status_bulletin_reads_the_published_one_first(monkeypatch):
    base = "https://huggingface.co/datasets/Rekin226/aquascope-gauges/resolve/main/bulletins/"
    files = {base + "index.json": {"latest": "2026-09"},
             base + "2026-09/bulletin.json": {"month": "2026-09", "summary": "s"}}
    monkeypatch.setattr(bulletin, "_fetch_json", lambda url: files.get(url))
    assert bulletin.latest_month() == "2026-09"
    got = bulletin.status_bulletin()
    assert got["origin"] == "published" and got["summary"] == "s"
    assert bulletin.read_published("2026-08") is None
    built = []
    monkeypatch.setattr(bulletin, "build_bulletin", lambda *a, **k: built.append((a, k)) or {"origin": "built"})
    assert bulletin.status_bulletin("2026-08")["origin"] == "built"       # nothing published for August
    assert bulletin.status_bulletin("2026-09", rebuild=True)["origin"] == "built"
    assert len(built) == 2


# ── the workflow ────────────────────────────────────────────────────────────


def test_the_monthly_workflow_runs_on_the_third_and_a_smoke_run_never_publishes():
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "bulletin.yml"
    wf = yaml.safe_load(path.read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert on["schedule"][0]["cron"].split()[2] == "3"
    assert {"month", "publish", "smoke"} <= set(on["workflow_dispatch"]["inputs"])
    steps = wf["jobs"]["bulletin"]["steps"]
    publish = next(s for s in steps if s.get("name", "").startswith("Publish"))
    assert "!inputs.smoke" in publish["if"] and "aquascope.bulletin publish" in publish["run"]
    build = next(s for s in steps if s.get("name") == "Build the bulletin")
    assert "${{" not in build["run"]            # dispatch inputs reach the shell as environment variables
    assert wf["jobs"]["bulletin"]["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"


def test_notable_lists_name_sizeable_gauges_and_keep_anomalies_to_their_class():
    # A trickle that dried up (0 against 0.001) is a record and counted, but a river's record is named first.
    trickle = {y: 0.001 + 0.0001 * (y - 2000) for y in range(2000, 2015)}
    trickle[2015] = 0.0
    river = {y: 20.0 + (y - 2000) for y in range(2000, 2015)}
    river[2015] = 15.0
    # A skewed record: the month is seven times its median yet in the normal band (a few huge years above it).
    skewed = {y: 1.0 for y in range(2000, 2008)}
    skewed.update({y: 50.0 + y - 2008 for y in range(2008, 2015)})
    skewed[2015] = 7.0
    means = _means([("uk_ea", "T", trickle), ("uk_ea", "R", river), ("usgs", "S", skewed)])
    b = bulletin.compute_bulletin(means, "2015-09", made="2015-10-03T07:00:00Z")
    nt = b["notable"]
    assert nt["n_record_low"] == 2
    assert [r["station_id"] for r in nt["record_low"]] == ["R", "T"]
    s = next(g for g in b["gauges"] if g["station_id"] == "S")
    assert s["class"] == "normal" and s["ratio"] > 1
    assert all(r["station_id"] != "S" for r in nt["furthest_above"])
    assert [r["station_id"] for r in nt["furthest_below"]] == ["R"]


def test_the_top_up_skips_a_gauge_whose_record_stopped_long_before_the_month(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    from aquascope import explore

    hist = {y: 10.0 + (y - 2010) for y in range(2010, 2024)}
    a = pd.concat([
        frame("usgs", "OLD", daily(2010, 2023, 9, hist)),                                  # closed in 2023
        frame("usgs", "NEW", daily(2010, 2026, 9, {**hist, 2024: 9.0, 2025: 9.5, 2026: 12.0}, last_days=10)),
    ])
    _bundle(tmp_path / "obs" / "discharge" / "usgs.parquet", a.drop(columns=["source"]))
    monkeypatch.setattr(bulletin, "_catalog_rows", lambda root: [])
    monkeypatch.setattr(bulletin, "_basin_ids", lambda root: ({}, {}))
    asked = []

    def fake_fetch(source, station_id, **kw):
        asked.append(station_id)
        idx = pd.date_range("2026-09-01", "2026-09-30", freq="D")
        return {"series": pd.Series(np.full(len(idx), 12.0), index=idx), "variable": "discharge"}

    monkeypatch.setattr(explore, "fetch_series", fake_fetch)
    bulletin.build_bulletin("2026-09", archive=tmp_path, top_up=5, workers=1, today=date(2026, 10, 3))
    assert asked == ["NEW"]
