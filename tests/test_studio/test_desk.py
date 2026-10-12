"""The Study Desk (aquascope.studio.desk): levers, revisions, sensitivity, review and sign-off on a finished study.

The study runs on the fake tools of conftest; the flood tool honours ``exclude_years`` the way the real one does
(the year's maximum leaves the fitted sample and the quantiles move), so a revision can be checked end to end.
"""

from __future__ import annotations

import json

import pytest

from aquascope.studio import desk
from aquascope.studio.steering import Control, _coerce
from tests.test_studio.conftest import FLOW, PROBLEM, fake_tools

YEARS = list(range(1986, 2026))
AMAX = [300.0 + (i * 53 % 220) for i in range(len(YEARS))]
AMAX[YEARS.index(2008)] = 900.0                                    # one outsized flood


def _flood(**kw):
    """FLOW with its annual maxima; excluding a year drops it and lowers every quantile by 10 %."""
    p = json.loads(json.dumps(FLOW))
    keep = [(y, v) for y, v in zip(YEARS, AMAX) if y not in (kw.get("exclude_years") or [])]
    p["annual_max"] = {"year": [y for y, _ in keep], "v": [v for _, v in keep]}
    if kw.get("exclude_years"):
        dropped = [(y, v) for y, v in zip(YEARS, AMAX) if y in kw["exclude_years"]]
        p["annual_max_excluded"] = {"year": [y for y, _ in dropped], "v": [v for _, v in dropped]}
        for fit in p["ffa"]["fits"].values():
            fit["q"] = [round(q * 0.9, 1) for q in fit["q"]]
    p["ffa"]["n_years"] = len(keep)
    return p


@pytest.fixture
def done(studio_factory):
    calls: list = []
    s, _ = studio_factory(tools=fake_tools(calls, flood_frequency=_flood, analyze_station=_flood))
    s.say(PROBLEM)
    s.approve()
    calls.clear()
    return s, calls


def test_a_finished_flood_study_offers_its_levers(done):
    s, _ = done
    ids = {lv["id"]: lv for lv in desk.levers(s.workspace)}
    assert set(ids) == {"return_period", "years", "exclude_years", "estimator"}
    assert ids["return_period"]["value"] == 100
    assert ids["exclude_years"]["candidates"][0] == {"year": 2008, "value": 900.0}
    assert set(ids["estimator"]["choices"]) >= {"gev_lmoments", "lp3"}


def test_leaving_a_year_out_reruns_the_flood_steps_and_records_a_revision(done):
    s, calls = done
    r = s.revise({"exclude_years": "2008"}, by="A. Reviewer")
    assert r.kind == "report" and r.payload["revision"]["rev"] == "B"
    flood_calls = [kw for name, kw in calls if name in ("flood_frequency", "analyze_station")]
    assert flood_calls and all(kw.get("exclude_years") == [2008] for kw in flood_calls)
    revs = s.workspace.desk["revisions"]
    assert [x["rev"] for x in revs] == ["A", "B"]
    assert revs[1]["description"] == "Left out the annual maxima of 2008" and revs[1]["by"] == "A. Reviewer"
    assert revs[1]["before"]["value"] == 520 and revs[1]["after"]["value"] == 468
    assert "exclude_years" in s.workspace.study_obj().to_yaml()       # study.yaml reproduces the revision


def test_the_quoted_distribution_changes_the_answer_without_a_rerun(done):
    s, calls = done
    r = s.revise({"estimator": "lp3"})
    assert not calls
    assert r.payload["decision"]["value"] == 548 and "Log-Pearson III" in r.payload["decision"]["answer"]
    assert s.workspace.desk["revisions"][-1]["description"] == "The answer now quotes Log-Pearson III"


def test_a_lever_that_does_not_exist_or_changes_nothing_is_refused(done):
    s, _ = done
    assert s.revise({"bogus": 1}).payload["errors"]
    assert "nothing would change" in s.revise({"return_period": 100}).text
    assert not (s.workspace.desk or {}).get("revisions")


def test_sign_off_moves_the_status_and_the_documents_follow(done):
    s, _ = done
    s.sign("checked_by", "B. Checker")
    style = s.workspace.house_style
    assert style["checked_by"] == "B. Checker" and style["status"] == "CHECKED" and style["checked_on"]
    s.sign("approved_by", "C. Approver")
    assert s.workspace.house_style["status"] == "ISSUED"
    assert [r["description"] for r in s.workspace.desk["revisions"]][-2:] == ["Checked by B. Checker",
                                                                               "Approved by C. Approver"]
    assert "not signed" in s.sign("reviewer", "X").text.lower()


def test_review_comments_round_trip_through_the_face_op(done):
    s, _ = done
    out = desk.studio_op(s, {"action": "comment", "text": "Was 2008 an ice jam?", "section": "Design flood"})
    assert out["desk"]["comments"][0]["status"] == "open"
    out = desk.studio_op(s, {"action": "resolve", "id": "c1", "response": "No: open-water flood."})
    assert out["desk"]["comments"][0]["status"] == "resolved"
    assert desk.studio_op(s, {"action": "resolve", "id": "c9"})["error"]
    view = desk.studio_op(s, {"action": "view"})["desk"]
    assert json.dumps(view) and view["status"] == "DRAFT"


def test_the_sensitivity_table_reads_the_stored_maxima(done):
    s, _ = done
    rows = desk.sensitivity(s.workspace)
    cases = [r["case"] for r in rows]
    assert cases[0].startswith("As reported")
    assert any(c.startswith("Without the largest flood (2008)") for c in cases)
    without = next(r for r in rows if r["case"].startswith("Without"))
    assert without["value"] < rows[0]["value"] * 1.5 and without["change_pct"] is not None


def test_the_report_carries_the_revisions_the_sensitivity_and_the_review(done):
    pytest.importorskip("matplotlib")
    from aquascope.studio.document import build_report, render_markdown

    s, _ = done
    s.revise({"exclude_years": [2008]}, by="A. Reviewer")
    desk.comment(s.workspace, "Check the 2008 rating.", author="A. Reviewer")
    md = render_markdown(build_report(s.workspace))
    assert "*Revision history.*" in md and "Left out the annual maxima of 2008" in md
    assert "Sensitivity of the design value" in md
    assert "Review comments and responses" in md and "Check the 2008 rating." in md


@pytest.mark.parametrize("raw, want", [("2008, 1936", [1936, 2008]), ([2008], [2008]), ("", None)])
def test_the_years_control_reads_a_list(raw, want):
    control = Control("exclude_years", "Leave out these years' floods", "years", optional=True)
    assert _coerce(control, raw) == want


def test_the_years_control_refuses_what_is_not_a_year():
    control = Control("exclude_years", "Leave out these years' floods", "years", optional=True)
    with pytest.raises(ValueError):
        _coerce(control, "next year")
    with pytest.raises(ValueError):
        _coerce(control, "1066")


# ── a revision refits the record the study holds ────────────────────────────


def _with_record(ws):
    """The study's flood step keeps its observations in a record table, as a real run does."""
    import numpy as np
    import pandas as pd

    from aquascope.studio.workspace import Artifact

    idx = pd.date_range("1986-01-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(7)
    values = rng.gamma(2.0, 20.0, len(idx))
    for y, v in zip(YEARS, AMAX):
        values[idx.get_loc(pd.Timestamp(f"{y}-04-15"))] = v
    csv = "datetime,value\n" + "\n".join(f"{t.isoformat()},{v:.3f}" for t, v in zip(idx, values))
    for r in ws.run["results"]:
        if r["tool"] == "flood_frequency":
            ws.add_artifact(Artifact(id=f"tab-{r['id']}-series", kind="table", name=f"tables/{r['id']}_series.csv",
                                     data=csv.encode(), media_type="text/csv", step=r["id"]))
    return ws


def test_the_desk_refits_the_stored_record_without_a_fetch(done, monkeypatch):
    import aquascope.explore

    s, _ = done
    _with_record(s.workspace)

    def no_fetch(*a, **k):
        raise AssertionError("the Desk must not fetch the record again")

    monkeypatch.setattr(aquascope.explore, "fetch_series", no_fetch)
    payload = next(r["result"] for r in s.workspace.run["results"] if r["tool"] == "flood_frequency")
    tools = desk.stored_record_tools(s.workspace)
    out = tools["flood_frequency"](payload["source"], payload["station_id"], exclude_years=[2008])
    assert out["ffa"]["n_years"] == len(YEARS) - 1
    assert out["annual_max_excluded"]["year"] == [2008]
    assert "nothing was fetched again" in out["fetch_note"]
    recent = tools["analyze_station"](payload["source"], payload["station_id"], years=20)
    assert recent["annual_max"]["year"][0] >= 2005


def test_a_station_the_study_holds_no_record_for_is_fetched_as_before(done):
    s, _ = done
    tools = desk.stored_record_tools(s.workspace)
    assert desk._stored_series(s.workspace, "usgs", "nowhere") is None
    assert callable(tools["flood_frequency"])
