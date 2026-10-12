"""aquascope.ingest: any export -> clean series + QA report, heuristics first, LLM optional."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aquascope import ingest as ing


def _nwis_like(tmp_path, n_years=12, sentinel=True):
    idx = pd.date_range("2005-01-01", periods=int(365.25 * n_years), freq="D")
    rng = np.random.default_rng(2)
    q = np.exp(rng.normal(3, 0.5, len(idx)))
    if sentinel:
        q[50:80] = -9999
    q[300] = 5e5
    lines = ["# USGS export", "# Units: cfs", "agency_cd\tsite_no\tdatetime\tflow_cfs\tqual_cd"]
    lines += [f"USGS\t01646500\t{d.strftime('%m/%d/%Y')}\t{v:.1f}\tA" for d, v in zip(idx, q)]
    lines.insert(10, lines[9])  # a duplicated row
    p = tmp_path / "nwis.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_read_table_handles_comments_and_tabs(tmp_path):
    p = _nwis_like(tmp_path, n_years=1)
    df = ing.read_table(p)
    assert list(df.columns) == ["agency_cd", "site_no", "datetime", "flow_cfs", "qual_cd"]
    assert len(df) > 300


def test_guess_mapping_picks_date_value_unit(tmp_path):
    df = ing.read_table(_nwis_like(tmp_path, n_years=1))
    m = ing.guess_mapping(df)
    assert m.datetime_column == "datetime" and m.value_column == "flow_cfs"
    assert m.variable == "discharge" and m.unit == "m3/s" and m.to_si_factor == pytest.approx(0.0283168, rel=1e-4)
    assert m.station_column == "site_no"


def test_ingest_end_to_end_qa(tmp_path):
    p = _nwis_like(tmp_path)
    r = ing.ingest(p)
    q = r["qa"]
    assert q["n_sentinels_dropped"] == 30 and q["n_duplicates_dropped"] == 1
    assert q["n_spikes_flagged"] >= 1 and q["coverage_pct"] > 98
    assert q["start"] == "2005-01-01"
    assert r["analysis"]["unit"] == "m3/s" and r["analysis"]["ffa"]["n_years"] >= 10
    # sentinels were caught before the cfs->m3/s scaling
    assert (r["series"] < 0).sum() == 0
    paths = ing.write_outputs(r, tmp_path / "out" / "potomac")
    assert (tmp_path / "out" / "potomac.csv").exists()
    md = (tmp_path / "out" / "potomac.qa.md").read_text(encoding="utf-8")
    assert "30 sentinel" in md and "Flood frequency" in md
    qa = json.loads((tmp_path / "out" / "potomac.qa.json").read_text(encoding="utf-8"))
    assert qa["mapping"]["value_column"] == "flow_cfs" and paths["qa_md"].endswith(".qa.md")


def test_ingest_overrides_and_semicolon_csv(tmp_path):
    p = tmp_path / "level.csv"
    rows = ["Datum;Pegel [cm];Bemerkung"] + [f"{d.strftime('%d.%m.%Y')};{100 + i % 7};ok" for i, d in
                                             enumerate(pd.date_range("2021-01-01", periods=400, freq="D"))]
    p.write_text("\n".join(rows), encoding="utf-8")
    r = ing.ingest(p, variable="water_level", date_column="Datum", value_column="Pegel [cm]", unit="cm")
    assert r["mapping"]["variable"] == "water_level" and r["mapping"]["unit"] == "m"
    assert r["series"].iloc[0] == pytest.approx(1.0)  # 100 cm -> 1 m
    assert r["qa"]["n_values"] == 400


def test_ingest_gaps_and_warnings(tmp_path):
    idx = pd.date_range("2010-01-01", periods=1000, freq="D")
    keep = [d for i, d in enumerate(idx) if not (200 <= i < 450)]  # a 250-day hole
    p = tmp_path / "gappy.csv"
    p.write_text(
        "date,discharge_m3s\n" + "\n".join(f"{d.date()},{5 + (i % 3)}" for i, d in enumerate(keep)),
        encoding="utf-8",
    )
    r = ing.ingest(p)
    assert r["qa"]["gaps"][0]["days"] == 250
    assert any("coverage" in w for w in r["qa"]["warnings"])


def test_ingest_without_a_date_column_is_explicit(tmp_path):
    p = tmp_path / "nodate.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="date/time column"):
        ing.ingest(p)


def test_llm_mapping_is_validated(tmp_path):
    df = ing.read_table(_nwis_like(tmp_path, n_years=1))

    def make_client(payload):
        msg = SimpleNamespace(content=json.dumps(payload))
        create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=msg)])  # noqa: E731
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    good = make_client({"datetime_column": "datetime", "value_column": "flow_cfs", "variable": "discharge",
                        "unit": "m3/s", "to_si_factor": 0.0283, "rationale": "flow column"})
    m = ing.llm_mapping(df, client=good, model="m")
    assert m and m.method == "llm:m" and m.value_column == "flow_cfs"
    bad = make_client({"datetime_column": "nope", "value_column": "flow_cfs"})
    assert ing.llm_mapping(df, client=bad, model="m") is None  # falls back to heuristics
    r = ing.ingest(_nwis_like(tmp_path, n_years=1), llm_client=bad, llm_model="m")
    assert r["mapping"]["method"] == "heuristic"
