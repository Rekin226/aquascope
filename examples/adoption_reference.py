"""Generate three recorded flood-screening examples from retained Explorer CSV exports.

No agency/model calls. First run the browser smoke study, then:
  python -m examples.adoption_reference --input-dir browser-smoke-output --out explorer/showcase/studies
Independent domain review is explicitly pending; this script is a maintainer reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from aquascope.explore import analyze_series, flood_ci
from aquascope.registry import SOURCES
from aquascope.studio import portable
from aquascope.studio.deliverables.figures import figures_for
from aquascope.studio.deliverables.report_md import report_html
from aquascope.studio.deliverables.tables import tables_for
from aquascope.studio.roles import author, interpreter
from aquascope.studio.showcase import Case, _write_case, write_index
from aquascope.studio.workspace import Workspace
from aquascope.study import Step, Study, run_study
from aquascope.trend_series import mark_reported_trend

CASES = [
    ("fish-river-us", "usgs", "USGS-01013500", "Fish River, Maine", 47.2375, -68.5827777777778),
    ("kingston-uk", "uk_ea", "8496ce69-482c-406a-a2f0-ac418ef8f099",
     "Thames at Kingston, England", 51.415482, -0.307629),
    ("seine-fr", "hubeau_hydrometrie", "F700000103", "Seine at Paris-Austerlitz, France", 48.84468962, 2.365510635),
]
# Positions checked against stations.parquet at this immutable archive revision.
# Observations came from the separately retained Explorer CSVs, not this catalog.
POSITION_CATALOG_REVISION = "f1f2fa19996aacb0abf82349b28ac5de16241fc7"


def generate(input_dir: Path, out: Path) -> list[dict]:
    generated = []
    for key, source, station, name, lat, lon in CASES:
        path = input_dir / f"{key}-cold.csv"
        raw = path.read_text(encoding="utf-8")
        frame = pd.read_csv(path, parse_dates=["date"])
        if list(frame.columns) != ["date", "discharge_m3_per_s"]:
            raise ValueError(f"Unexpected CSV schema for {key}; verify variable and units before analysis.")
        series = frame.set_index("date")["discharge_m3_per_s"].dropna().sort_index()
        payload = analyze_series(series, "discharge", "m3/s")
        payload.update(source=source, station_id=station, station_name=name,
                       software_revision=os.environ.get("AQUASCOPE_REVISION"),
                       fetch_note="Retained observed Explorer CSV; no observations refetched during reproduction.")
        if not payload.get("ffa"):
            raise ValueError(f"No eligible flood fit for {key}")
        payload["ffa"]["fits"]["gev_bootstrap"] = flood_ci(series)
        mark_reported_trend(payload, flood=True)
        question = f"Screen the 100-year daily-mean flood at {name}; identify estimator uncertainty and limitations."
        caveats = ["Maintainer reproduction; independent hydrologist review pending.",
                   "Daily mean maxima are not instantaneous annual peaks or a certified design flood.",
                   "No catchment regulation, channel topology or model/gauge comparability was established.",
                   "Re-run live uses current observations; use the retained CSV and this script for reproduction."]
        study = Study(question=question, title=f"Daily-mean flood screening: {name}", version=3, author="hand",
            problem={"kind": "flood_risk", "site": {"lat": lat, "lon": lon}, "params": {"return_period": 100}},
            plan={"playbook": "flood_risk", "objective": question, "caveats": caveats,
                  "methodology": ["Compare GEV L-moments, LP3 and GEV MLE quantiles on complete-year daily maxima."],
                  "assumptions": ["Independent stationary annual maxima."]},
            steps=[Step(id="s1", tool="flood_frequency", method="at_site_flood_frequency",
                arguments={"source": source, "station_id": station, "bootstrap_ci": True}, expects=[
                    {"check": "min_years", "path": "ffa.n_years", "value": 20},
                    {"check": "unit_present", "path": "unit"},
                    {"check": "max_return_period_factor", "path": "ffa.n_years", "value": 3, "return_period": 100},
                    {"check": "spread_within", "paths": ["ffa.fits.gev_lmoments.q", "ffa.fits.lp3.q"],
                     "value": .25, "return_period": 100},
                    {"check": "trend_on_series", "path": "ffa.amax_trend", "value": .05},
                ])])
        run = run_study(study, tools={"flood_frequency": lambda **kwargs: payload})
        ws = Workspace(status="done", site={"lat": lat, "lon": lon}, tables={"observations.csv": raw})
        ws.brief.problem = ws.brief.decision = question
        ws.brief.kind = ws.brief.playbook = "flood_risk"
        ws.brief.intake = {"return_period": 100}
        ws.set_study(study)
        ws.run = {"ok": run.ok, "results": run.results, "gates": run.gates, "failed_gates": run.failed_gates,
                  "failed_steps": run.failed_steps, "summary": run.summary, "started": run.started,
                  "finished": run.finished, "stop_reason": run.stop_reason}
        ws.artifacts = figures_for("s1", "flood_frequency", payload, unit="m3/s", site=ws.site,
                                   kinds=["frequency_curve", "annual_maxima", "trend"])
        ws.artifacts.extend(tables_for("s1", "flood_frequency", payload))
        interpreter.interpret(ws, None)
        author.author_report(ws, None)
        case = Case(id=f"reference-{key}", title=f"Daily-mean flood screening: {name}", lat=lat, lon=lon,
                    problem=question, kind="flood_risk", site=name,
                    shows="Observed snapshot with estimator-specific uncertainty. Independent domain review pending.")
        folder = out / case.id
        metadata = _write_case(ws, case, folder, seconds=0, price=None, error=None)
        metadata["seconds"] = None  # generation time was not measured; zero would imply a measured instant result
        (folder / "complete.aqstudy.json").write_text(portable.dumps(ws), encoding="utf-8")
        (folder / "report.html").write_text(report_html(ws), encoding="utf-8")
        (folder / "observations.csv").write_text(raw, encoding="utf-8")
        provenance = {"source": source, "station_id": station, "agency": SOURCES[source].agency,
                      "latitude": lat, "longitude": lon, "position_catalog_revision": POSITION_CATALOG_REVISION,
                      "license": SOURCES[source].license, "agency_homepage": SOURCES[source].homepage,
                      "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "analysis_snapshot": payload["data_snapshot"], "start": payload["start"], "end": payload["end"],
                      "n": payload["n"], "review_status": "independent domain review pending",
                      "software_revision": os.environ.get("AQUASCOPE_REVISION")}
        (folder / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        metadata["files"].extend(["complete.aqstudy.json", "report.html", "observations.csv", "provenance.json"])
        metadata["review_status"] = provenance["review_status"]
        metadata["input_sha256"] = provenance["csv_sha256"]
        metadata["software_revision"] = provenance["software_revision"]
        (folder / "meta.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
        generated.append({"id": case.id, "grade": ws.report["grade"], "input": provenance,
                          "decision": ws.report["decision"]})
    write_index(out)
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.input_dir, args.out), indent=2))
