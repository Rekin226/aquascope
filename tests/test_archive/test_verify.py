"""Issued forecasts scored against the gauge (#785): pairing, the scores, raw against corrected, the pair floor."""

from __future__ import annotations

import json
import math
import sys

import pandas as pd
import pytest

pytest.importorskip("pyarrow.parquet", reason="the archive extra")

from aquascope import cli  # noqa: E402
from aquascope.archive import verify as vf  # noqa: E402


def _issued_row(model: str, sid: str, issue: str, valid: str, lead: int, mean: float, *, corrected: float | None,
                q2: float | None = 100.0, spread: float = 1.0) -> dict:
    row = {"issue_date": issue, "source": "usgs", "station_id": sid, "model": model, "valid_date": valid,
           "lead_day": lead, "gauge_q2": q2, "mean": mean, "median": mean, "p25": mean / spread,
           "p75": mean * spread}
    for k in ("mean", "median", "p25", "p75"):
        row[f"{k}_c"] = None if corrected is None else row[k] * corrected / mean
    return row


def _status(snapshot: str, rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"source": "usgs", "station_id": sid, "value_date": day, "value": v, "percentile": 50.0,
                          "class": "normal", "n_years": 30} for sid, day, v in rows]).assign(snapshot=snapshot)


# GEOGLOWS raw ten times the gauge, corrected twice; GloFAS on the gauge but never corrected.
ISSUED = pd.DataFrame([
    _issued_row("geoglows", "A", "2026-10-01", "2026-10-02", 1, 100.0, corrected=20.0),
    _issued_row("geoglows", "B", "2026-10-01", "2026-10-02", 1, 50.0, corrected=10.0),
    _issued_row("geoglows", "A", "2026-10-01", "2026-10-03", 2, 100.0, corrected=20.0),
    _issued_row("glofas", "A", "2026-10-01", "2026-10-02", 1, 10.0, corrected=None),
    _issued_row("glofas", "B", "2026-10-01", "2026-10-02", 1, 5.0, corrected=None),
    _issued_row("glofas", "C", "2026-10-01", "2026-10-02", 1, 7.0, corrected=None),   # no observation
])
STATUS = pd.concat([
    _status("2026-10-03", [("A", "2026-10-02", 10.0), ("B", "2026-10-02", 5.0)]),
    _status("2026-10-04", [("A", "2026-10-03", 10.0), ("B", "2026-10-03", 0.0)]),
])


# ── CRPS ────────────────────────────────────────────────────────────────────


def test_crps_matches_the_hand_computed_value():
    # Quantiles 0, 1, 2 at 0.25, 0.5, 0.75 and y = 1: pinball losses 0.25, 0, 0.25; 2/3 of their sum is 1/3.
    got = vf.crps_from_quantiles([[0.0, 1.0, 2.0]], [0.25, 0.5, 0.75], [1.0])
    assert got[0] == pytest.approx(1 / 3)
    # y = 0 below all of 1, 2, 3: 0.75 * 1 + 0.5 * 2 + 0.25 * 3 = 2.5, times 2/3.
    got = vf.crps_from_quantiles([[1.0, 2.0, 3.0]], [0.25, 0.5, 0.75], [0.0])
    assert got[0] == pytest.approx(2.5 * 2 / 3)


def test_crps_of_a_point_forecast_is_the_absolute_error():
    got = vf.crps_from_quantiles([[5.0, 5.0, 5.0], [1.0, 1.0, 1.0]], [0.25, 0.5, 0.75], [2.0, 4.0])
    assert got == pytest.approx([3.0, 3.0])


def test_crossed_quantiles_are_sorted_first():
    a = vf.crps_from_quantiles([[2.0, 1.0, 0.0]], [0.25, 0.5, 0.75], [1.0])
    assert a[0] == pytest.approx(1 / 3)


def test_crps_rejects_a_level_count_mismatch():
    with pytest.raises(ValueError):
        vf.crps_from_quantiles([[1.0, 2.0]], [0.25, 0.5, 0.75], [1.0])


# ── pairing ─────────────────────────────────────────────────────────────────


def test_observations_keep_the_newest_snapshot_of_a_day():
    st = pd.concat([_status("2026-10-03", [("A", "2026-10-02", 9.0)]),
                    _status("2026-10-04", [("A", "2026-10-02", 10.0)])])
    got = vf.observations(st)
    assert got.to_dict("records") == [{"source": "usgs", "station_id": "A", "valid_date": "2026-10-02",
                                       "observed": 10.0}]


def test_pairs_join_on_station_and_the_value_date_with_positive_flows_only():
    p = vf.pair(ISSUED, vf.observations(STATUS))
    keys = set(zip(p["model"], p["station_id"], p["valid_date"]))
    assert ("glofas", "C", "2026-10-02") not in keys                       # no gauge value that day
    assert ("geoglows", "A", "2026-10-03") in keys                          # matched on value_date, not snapshot
    assert not any(sid == "B" and day == "2026-10-03" for _, sid, day in keys)  # observed zero is dropped
    assert len(p) == 5 and set(p["month"]) == {"2026-10"}


def test_nothing_to_pair_gives_an_empty_table():
    assert vf.pair(ISSUED, vf.observations(pd.DataFrame())).empty
    assert vf.score(pd.DataFrame()).empty


# ── the scores ──────────────────────────────────────────────────────────────


def _one(table: pd.DataFrame, model: str, variant: str, lead: int, month: str = vf.ALL_MONTHS) -> dict:
    t = table[(table["model"] == model) & (table["variant"] == variant) & (table["lead_day"] == lead)
              & (table["month"] == month)]
    assert len(t) == 1, (model, variant, lead, month)
    return t.iloc[0].to_dict()


def test_raw_and_corrected_are_scored_separately():
    table = vf.score(vf.pair(ISSUED, vf.observations(STATUS)), min_pairs=1)
    raw, cor = _one(table, "geoglows", "raw", 1), _one(table, "geoglows", "corrected", 1)
    assert raw["median_abs_log10_error"] == pytest.approx(1.0) and raw["factor"] == pytest.approx(10.0)
    assert cor["median_abs_log10_error"] == pytest.approx(math.log10(2), abs=1e-4)
    assert raw["median_log10_error"] > 0 and raw["n_pairs"] == cor["n_pairs"] == 2
    assert _one(table, "glofas", "raw", 1)["median_abs_log10_error"] == pytest.approx(0.0)
    assert table[(table["model"] == "glofas") & (table["variant"] == "corrected")].empty
    # The per-month row and the pooled row agree when there is one month.
    assert _one(table, "geoglows", "raw", 1, "2026-10")["n_pairs"] == 2


def test_each_score_carries_its_pairs_gauges_and_dates():
    table = vf.score(vf.pair(ISSUED, vf.observations(STATUS)), min_pairs=1)
    r = _one(table, "geoglows", "raw", 1)
    assert (r["n_gauges"], r["first_valid"], r["last_valid"]) == (2, "2026-10-02", "2026-10-02")


def test_a_corrected_value_missing_on_some_rows_counts_only_where_present():
    issued = ISSUED.copy()
    issued.loc[1, ["mean_c", "median_c", "p25_c", "p75_c"]] = None
    table = vf.score(vf.pair(issued, vf.observations(STATUS)), min_pairs=1)
    assert _one(table, "geoglows", "raw", 1)["n_pairs"] == 2
    assert _one(table, "geoglows", "corrected", 1)["n_pairs"] == 1


def test_below_the_floor_a_score_is_flagged_not_dropped():
    table = vf.score(vf.pair(ISSUED, vf.observations(STATUS)), min_pairs=3)
    r = _one(table, "geoglows", "raw", 1)
    assert r["n_pairs"] == 2 and not r["enough_pairs"]
    assert r["median_abs_log10_error"] == pytest.approx(1.0)
    assert vf.score(vf.pair(ISSUED, vf.observations(STATUS)), min_pairs=2)["enough_pairs"].any()


def test_crps_in_log_flow_from_the_stored_quantiles():
    # Quantiles 10/s, 10, 10*s with s = 10 around a gauge at 10: log10 quantiles 0, 1, 2 against 1, CRPS 1/3.
    issued = pd.DataFrame([_issued_row("glofas", "A", "2026-10-01", "2026-10-02", 1, 10.0, corrected=None,
                                       spread=10.0)])
    table = vf.score(vf.pair(issued, vf.observations(STATUS)), min_pairs=1)
    r = _one(table, "glofas", "raw", 1)
    assert r["crps_log10"] == pytest.approx(1 / 3, abs=1e-4) and r["n_crps"] == 1


def test_hits_misses_and_false_alarms_against_the_two_year_flow():
    obs = pd.DataFrame({"source": "usgs", "station_id": list("ABCD"), "valid_date": "2026-10-02",
                        "observed": [150.0, 150.0, 50.0, 50.0]})
    issued = pd.DataFrame([
        _issued_row("glofas", "A", "2026-10-01", "2026-10-02", 1, 120.0, corrected=None),   # hit
        _issued_row("glofas", "B", "2026-10-01", "2026-10-02", 1, 80.0, corrected=None),    # miss
        _issued_row("glofas", "C", "2026-10-01", "2026-10-02", 1, 130.0, corrected=None),   # false alarm
        _issued_row("glofas", "D", "2026-10-01", "2026-10-02", 1, 40.0, corrected=None),    # quiet
        _issued_row("glofas", "E", "2026-10-01", "2026-10-02", 1, 40.0, corrected=None, q2=None),
    ])
    obs = pd.concat([obs, pd.DataFrame([{"source": "usgs", "station_id": "E", "valid_date": "2026-10-02",
                                         "observed": 500.0}])])
    r = _one(vf.score(vf.pair(issued, obs), min_pairs=1), "glofas", "raw", 1)
    assert (r["hits"], r["misses"], r["false_alarms"], r["correct_negatives"]) == (1, 1, 1, 1)
    assert r["n_threshold"] == 4 and r["n_pairs"] == 5
    assert r["hit_rate"] == pytest.approx(0.5) and r["false_alarm_ratio"] == pytest.approx(0.5)


def test_no_events_leave_the_rates_empty():
    table = vf.score(vf.pair(ISSUED, vf.observations(STATUS)), min_pairs=1)
    r = _one(table, "geoglows", "corrected", 1)
    assert r["hits"] == r["misses"] == r["false_alarms"] == 0
    assert r["hit_rate"] is None or pd.isna(r["hit_rate"])


def test_summary_reports_both_variants_and_the_headline():
    table = vf.score(vf.pair(ISSUED, vf.observations(STATUS)), min_pairs=1)
    doc = vf.summary(table, min_pairs=1)
    assert doc["pairs"] == {"geoglows": {"raw": 3, "corrected": 3}, "glofas": {"raw": 2, "corrected": 0}}
    assert {(h["model"], h["variant"]) for h in doc["headline"]} == {("geoglows", "raw"), ("geoglows", "corrected"),
                                                                      ("glofas", "raw")}
    assert doc["by_lead"]["geoglows"]["raw"] == {"1": 1.0, "2": 1.0}
    json.dumps(doc)   # JSON-ready, no numpy scalars left


# ── files and the CLI ───────────────────────────────────────────────────────


def _local_archive(root):
    (root / "forecasts" / "issued").mkdir(parents=True)
    (root / "forecasts" / "status").mkdir(parents=True)
    ISSUED.to_parquet(root / "forecasts" / "issued" / "2026-10-01.parquet")
    later = ISSUED.assign(issue_date="2026-10-02", valid_date="2026-10-03", lead_day=1)
    later.to_parquet(root / "forecasts" / "issued" / "2026-10-02.parquet")
    for snap, g in STATUS.groupby("snapshot"):
        g.drop(columns="snapshot").to_parquet(root / "forecasts" / "status" / f"{snap}.parquet")
    STATUS.drop(columns="snapshot").to_parquet(root / "forecasts" / "status" / "latest.parquet")   # skipped
    return root


def test_read_archive_picks_issues_by_date(tmp_path):
    root = _local_archive(tmp_path / "arch")
    issued, status, meta = vf.read_archive(root, days=1)
    assert set(issued["issue_date"]) == {"2026-10-02"} and meta["issues"]["n"] == 1
    assert meta["snapshots"] == {"n": 2, "first": "2026-10-03", "last": "2026-10-04"}
    _, _, meta = vf.read_archive(root, since=pd.Timestamp("2026-10-01").date(), until=pd.Timestamp("2026-10-01").date())
    assert meta["issues"]["first"] == meta["issues"]["last"] == "2026-10-01"


def test_cli_writes_the_table_and_the_summary(tmp_path, monkeypatch, capsys):
    root = _local_archive(tmp_path / "arch")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["aquascope", "evidence", "verify", "--archive", str(root), "--out", str(out),
                                      "--min-pairs", "2"])
    with pytest.raises(SystemExit) as done:
        cli.main()
    assert done.value.code == 0
    table = pd.read_parquet(out / "verify" / "forecast_skill.parquet")
    doc = json.loads((out / "verify" / "summary.json").read_text())
    assert {"n_pairs", "enough_pairs", "median_abs_log10_error", "crps_log10", "hit_rate",
            "false_alarm_ratio"} <= set(table.columns)
    assert doc["min_pairs"] == 2 and doc["inputs"]["issues"]["n"] == 2
    assert doc["pairs"]["geoglows"]["corrected"] > 0 and doc["pairs"]["glofas"]["corrected"] == 0
    assert "headline" in json.loads(capsys.readouterr().out)
    gauges = pd.read_parquet(out / "verify" / "forecast_skill_gauges.parquet")
    assert list(gauges.columns) == vf.GAUGE_COLUMNS
    assert doc["gauges"]["file"] == "verify/forecast_skill_gauges.parquet"
    assert doc["gauges"]["window_days"] == vf.TRACK_DAYS and doc["gauges"]["leads"] == list(vf.TRACK_LEADS)


# ── each gauge's own track record ───────────────────────────────────────────


@pytest.mark.parametrize(("factor", "enough", "word"), [
    (1.0, True, "good"), (2.0, True, "good"), (2.01, True, "fair"), (4.0, True, "fair"), (4.2, True, "poor"),
    (1.5, False, "unknown"), (None, True, "unknown"), (float("nan"), True, "unknown"), ("x", True, "unknown"),
])
def test_trust_words_follow_the_bands(factor, enough, word):
    assert vf.trust(factor, enough) == word


def _gauge_pairs(days: int = 40) -> pd.DataFrame:
    """Gauge A, 40 days at leads 1 to 3: GEOGLOWS raw 3 times the gauge, corrected 1.5 times; GloFAS raw 10 times.
    Gauge B only the newest 5 days, GEOGLOWS raw on the gauge."""
    rows, obs = [], []
    end = pd.Timestamp("2026-11-17")
    for k in range(days):
        valid = (end - pd.Timedelta(days=k)).strftime("%Y-%m-%d")
        obs.append(("A", valid, 10.0))
        for lead in (1, 2, 3):
            issue = (end - pd.Timedelta(days=k + lead)).strftime("%Y-%m-%d")
            rows.append({**_issued_row("geoglows", "A", issue, valid, lead, 30.0, corrected=15.0), "river_id": 7})
            rows.append({**_issued_row("glofas", "A", issue, valid, lead, 100.0, corrected=None), "river_id": 7})
        if k < 5:
            obs.append(("B", valid, 4.0))
            rows.append({**_issued_row("geoglows", "B", valid, valid, 1, 4.0, corrected=None), "river_id": 9})
    return vf.pair(pd.DataFrame(rows), vf.observations(_status("2026-11-18", obs)))


def test_by_gauge_scores_each_gauge_over_the_newest_window():
    t = vf.by_gauge(_gauge_pairs())
    card = t[(t["lead_min"] == 1) & (t["lead_max"] == 3)].set_index(["station_id", "model", "variant"])
    a = card.loc[("A", "geoglows", "raw")]
    assert a["n_pairs"] == 90 and a["first_valid"] == "2026-10-19" and a["last_valid"] == "2026-11-17"
    assert a["factor"] == pytest.approx(3.0, rel=1e-3) and a["trust"] == "fair" and bool(a["enough_pairs"])
    assert card.loc[("A", "geoglows", "corrected")]["trust"] == "good"
    assert card.loc[("A", "glofas", "raw")]["trust"] == "poor"
    assert ("A", "glofas", "corrected") not in card.index   # never issued corrected: no row, not a zero
    b = card.loc[("B", "geoglows", "raw")]
    assert b["n_pairs"] == 5 and not bool(b["enough_pairs"]) and b["trust"] == "unknown"
    assert b["factor"] == pytest.approx(1.0)                 # the number stays; the word says it is too thin
    assert set(t["window_days"]) == {vf.TRACK_DAYS}
    assert int(card.loc[("A", "geoglows", "raw")]["river_id"]) == 7


def test_by_gauge_has_every_single_lead_too():
    t = vf.by_gauge(_gauge_pairs())
    single = t[(t["station_id"] == "A") & (t["model"] == "geoglows") & (t["variant"] == "raw")
               & (t["lead_min"] == t["lead_max"])]
    assert sorted(single["lead_min"]) == [1, 2, 3] and set(single["n_pairs"]) == {30}


def test_by_gauge_takes_positions_and_survives_nothing():
    pos = pd.DataFrame([{"source": "usgs", "station_id": "A", "lat": 45.5, "lon": -122.6}])
    t = vf.by_gauge(_gauge_pairs(), positions=pos)
    assert set(t.loc[t["station_id"] == "A", "lat"]) == {45.5}
    assert t.loc[t["station_id"] == "B", "lat"].isna().all()
    assert vf.by_gauge(pd.DataFrame()).empty and list(vf.by_gauge(None).columns) == vf.GAUGE_COLUMNS


def test_by_gauge_counts_threshold_events():
    p = _gauge_pairs()
    p["gauge_q2"] = 20.0                         # GEOGLOWS raw (30) always above, the gauge (10) never: false alarms
    t = vf.by_gauge(p)
    a = t[(t["station_id"] == "A") & (t["model"] == "geoglows") & (t["variant"] == "raw") & (t["lead_min"] == 1)
          & (t["lead_max"] == 3)].iloc[0]
    assert (a["hits"], a["misses"], a["false_alarms"], a["n_threshold"]) == (0, 0, 90, 90)


# ── the daily job: the Hub plus the files just written, a time budget ─────────


def _hub_and_build(tmp_path, monkeypatch, *, delay: float = 0.0):
    """The 2026-10-01 issue and the 2026-10-03 snapshot published; the 2026-10-02 issue and the 2026-10-04
    snapshot only in the build folder, as right after the forecast step."""
    import io
    import time as _time

    from aquascope.archive import forecasts as fa

    def parquet(df):
        buf = io.BytesIO()
        df.to_parquet(buf)
        return buf.getvalue()

    snaps = dict(tuple(STATUS.groupby("snapshot")))
    hub = {"forecasts/issued/2026-10-01.parquet": parquet(ISSUED),
           "forecasts/status/2026-10-03.parquet": parquet(snaps["2026-10-03"].drop(columns="snapshot"))}
    asked = []

    def fake_get(url):
        path = url.split("/resolve/main/", 1)[1]
        asked.append(path)
        if path.endswith("manifest.json"):
            raise AssertionError("the job passes its own manifest")
        _time.sleep(delay)
        return hub.get(path)

    monkeypatch.setattr(fa, "_get", fake_get)
    build = tmp_path / "build"
    (build / "forecasts" / "issued").mkdir(parents=True)
    (build / "forecasts" / "status").mkdir(parents=True)
    ISSUED.assign(issue_date="2026-10-02", valid_date="2026-10-03", lead_day=1).to_parquet(
        build / "forecasts" / "issued" / "2026-10-02.parquet")
    snaps["2026-10-04"].drop(columns="snapshot").to_parquet(build / "forecasts" / "status" / "2026-10-04.parquet")
    manifest = {"status": {"date": "2026-10-04"},
                "issues": [{"date": "2026-10-01", "file": "forecasts/issued/2026-10-01.parquet"},
                           {"date": "2026-10-02", "file": "forecasts/issued/2026-10-02.parquet"}]}
    return build, manifest, asked


def test_the_job_reads_the_hub_with_its_own_new_files_in_place(tmp_path, monkeypatch):
    build, manifest, asked = _hub_and_build(tmp_path, monkeypatch)
    issued, status, meta = vf.read_archive(None, manifest=manifest, overlay=build)
    assert set(issued["issue_date"]) == {"2026-10-01", "2026-10-02"}
    assert set(status["snapshot"]) == {"2026-10-03", "2026-10-04"}
    assert meta["unread"] == 0 and meta["snapshots"]["last"] == "2026-10-04"
    # Only what the build folder lacks is asked of the Hub (snapshots from the first issue on, missing ones too).
    assert sorted(asked) == ["forecasts/issued/2026-10-01.parquet", "forecasts/status/2026-10-01.parquet",
                             "forecasts/status/2026-10-02.parquet", "forecasts/status/2026-10-03.parquet"]


def test_run_writes_a_manifest_beside_the_scores(tmp_path, monkeypatch):
    build, manifest, _ = _hub_and_build(tmp_path, monkeypatch)
    vf.run(build, manifest=manifest, overlay=build, min_pairs=2)
    man = json.loads((build / "verify" / "manifest.json").read_text())
    assert man["files"] == [
        "verify/forecast_skill.parquet", "verify/forecast_skill_gauges.parquet", "verify/summary.json"]
    assert man["complete"] is True and man["unread"] == 0 and man["pairs"] > 0
    assert man["n_rows"] == len(pd.read_parquet(build / "verify" / "forecast_skill.parquet"))
    assert man["issues"]["n"] == 2 and man["pairs_by_model"]["geoglows"]["corrected"] > 0


@pytest.mark.parametrize("data", [None, b"not parquet"], ids=["missing", "corrupt"])
def test_an_unread_issued_file_keeps_the_scorecard_from_publishing(tmp_path, monkeypatch, data):
    from aquascope.archive import forecasts as fa

    build, manifest, _ = _hub_and_build(tmp_path, monkeypatch)
    get = fa._get
    path = manifest["issues"][0]["file"]
    monkeypatch.setattr(fa, "_get", lambda url: data if url.endswith(f"/{path}") else get(url))

    vf.run(build, manifest=manifest, overlay=build, min_pairs=2)

    folder = build / "verify"
    recorded = json.loads((folder / "manifest.json").read_text())
    assert recorded["pairs"] > 0
    assert recorded["unread"] == 1
    assert recorded["complete"] is False
    assert not fa._scorecard_ready(folder)


def test_files_past_the_time_budget_are_counted_not_waited_for(tmp_path, monkeypatch):
    build, manifest, _ = _hub_and_build(tmp_path, monkeypatch, delay=1.0)
    vf.run(build, manifest=manifest, overlay=build, budget_s=0.05)
    man = json.loads((build / "verify" / "manifest.json").read_text())
    # At least the four slow Hub files; a slow runner may not finish the two local reads in time either.
    assert man["unread"] >= 4 and man["complete"] is False and man["seconds"] < 1.0


def test_the_forecast_job_scores_after_the_issue(tmp_path, monkeypatch):
    from aquascope.archive import forecasts as fa

    build, manifest, _ = _hub_and_build(tmp_path, monkeypatch)
    got = fa.scorecard(build, manifest)
    assert got["complete"] is True and got["issues"] == 2 and got["pairs"] > 0 and got["rows"] > 0
    assert (build / "verify" / "summary.json").exists()


def test_a_failed_scorecard_never_costs_the_issue(tmp_path, monkeypatch):
    from aquascope.archive import forecasts as fa

    def boom(*a, **k):
        raise RuntimeError("hub down")

    monkeypatch.setattr(vf, "run", boom)
    assert fa.scorecard(tmp_path, {"issues": []}) == {"error": "RuntimeError: hub down"}
