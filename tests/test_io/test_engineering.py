"""Engineering export bundles (#519): one format test per tool, round trips where a reader exists."""

from __future__ import annotations

import base64
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from aquascope.io import engineering as eng


def _daily(years: int = 12, start: str = "2000-01-01", seed: int = 1) -> pd.Series:
    idx = pd.date_range(start, periods=int(365.25 * years), freq="D")
    rng = np.random.default_rng(seed)
    return pd.Series(np.exp(rng.normal(2, 0.5, len(idx))), index=idx)


# ── the record ───────────────────────────────────────────────────────────────


def test_make_record_cleans_the_series():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-01", "2020-01-02"]).tz_localize("UTC")
    rec = eng.make_record(pd.Series([2.0, 1.0, 3.0], index=idx), variable="discharge", location="a b/c")
    assert list(rec.series.values) == [1.0, 3.0] and rec.series.index.tz is None and rec.utc
    assert rec.kind == "flow" and rec.unit == "m3/s" and rec.location == "a_b_c"
    assert eng.make_record(_daily(1), variable="precipitation").unit == "mm"
    with pytest.raises(ValueError, match="no engineering export"):
        eng.make_record(_daily(1), variable="ph")
    with pytest.raises(ValueError, match="no values"):
        eng.make_record(pd.Series([np.nan], index=pd.DatetimeIndex(["2020-01-01"])))


def test_regular_keeps_the_step_and_marks_gaps():
    s = _daily(1)
    s = s.drop(s.index[10:13])
    reg, step, notes = eng.regular(eng.make_record(s))
    assert step == 86400 and reg.isna().sum() == 3 and len(reg) == len(s) + 3
    assert any("3 of" in n for n in notes)
    hourly = pd.Series(1.0, index=pd.date_range("2020-01-01", periods=48, freq="h"))
    assert eng.regular(eng.make_record(hourly))[1] == 3600
    odd = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:07",
                                                              "2020-01-03 05:00"]))
    reg, step, notes = eng.regular(eng.make_record(odd))
    assert step == 86400 and "averaged to daily" in notes[0]


def test_regular_keeps_a_reading_a_little_off_the_grid():
    idx = pd.date_range("2020-01-01", periods=96, freq="15min")
    s = pd.Series(np.arange(96, dtype=float), index=idx)
    s.index = s.index.where(s.index != idx[40], idx[40] + pd.Timedelta(seconds=20))  # one clock drifted 20 s
    reg, step, notes = eng.regular(eng.make_record(s))
    assert step == 900 and len(reg) == 96 and reg.isna().sum() == 0 and reg.iloc[40] == 40.0


# ── HEC-DSS ──────────────────────────────────────────────────────────────────


def test_dss_record_stamps_period_data_at_the_end_of_the_day():
    rec = eng.make_record(_daily(1), variable="discharge", location="X1", source="usgs")
    r, notes = eng.dss_record(rec)
    assert r.pathname == "/USGS/X1/FLOW/02JAN2000/1Day/AQUASCOPE/"
    assert r.data_type == "PER-AVER" and r.units == "CMS" and r.times[0] == pd.Timestamp("2000-01-02")
    text = eng.dss_csv_text(r)
    lines = text.splitlines()
    assert lines[0] == "Date/Time,Value,Units,Type,Path"
    assert lines[1].startswith("01Jan2000 2400,") and lines[1].endswith(",CMS,PER-AVER," + r.pathname)
    assert lines[2].count(",") == 1
    precip = eng.dss_record(eng.make_record(_daily(1), variable="precipitation"))[0]
    assert precip.data_type == "PER-CUM" and "/PRECIP-INC/" in precip.pathname
    hourly = pd.Series(1.0, index=pd.date_range("2020-01-01 01:00", periods=5, freq="h"))
    inst = eng.dss_record(eng.make_record(hourly, variable="water_level"))[0]
    assert inst.data_type == "INST-VAL" and "/STAGE/" in inst.pathname and "/1Hour/" in inst.pathname


def test_dss_csv_reads_back_with_hecdss():
    hecdss = pytest.importorskip("hecdss")
    s = _daily(1)
    s.iloc[5] = np.nan
    r, _ = eng.dss_record(eng.make_record(s, location="X1"))
    path = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "x.dss.csv"
    path.write_text(eng.dss_csv_text(r))
    ts = hecdss.RegularTimeSeries.read_csv(str(path))
    assert ts.id == r.pathname and ts.units == "CMS" and ts.data_type == "PER-AVER"
    assert len(ts.times) == len(r.times) and ts.times[0] == pd.Timestamp("2000-01-02").to_pydatetime()
    assert ts.values[5] is None
    assert float(ts.values[0]) == pytest.approx(r.values[0], rel=1e-6)


def test_real_dss_round_trip_where_hecdss_loads(tmp_path):
    if not eng.hecdss_available():
        pytest.skip("hecdss native library does not load on this platform")
    from hecdss import HecDss

    r, _ = eng.dss_record(eng.make_record(_daily(1), location="X1", source="usgs"))
    eng.write_dss(tmp_path / "x.dss", [r])
    with HecDss(str(tmp_path / "x.dss")) as dss:
        paths = [str(p) for p in dss.get_catalog()]
        got = dss.get(r.pathname.replace("/02JAN2000/", "//"))
    assert any("/FLOW/" in p for p in paths)
    assert len(got.values) >= len(r.values) - 1
    assert float(got.values[0]) == pytest.approx(r.values[0], rel=1e-5)


def test_dss_falls_back_to_the_csv(monkeypatch):
    monkeypatch.setattr(eng, "hecdss_available", lambda: False)
    files, notes = eng.export_files(eng.make_record(_daily(1), location="X1"), "dss")
    assert set(files) == {"dss/X1_flow.dss.csv", "dss/README.txt"}
    assert any("RegularTimeSeries.read_csv" in n for n in notes)
    files, _ = eng.export_files(eng.make_record(_daily(1), location="X1"), "dss", dss_binary=False)
    assert "dss/X1_flow.dss.csv" in files


# ── HEC-SSP ──────────────────────────────────────────────────────────────────


def test_annual_peaks_by_water_year_with_coverage():
    s = pd.Series(1.0, index=pd.date_range("1999-10-01", "2003-09-30", freq="D"))
    s[pd.Timestamp("2000-09-30")] = 50.0     # water year 2000, last day
    s[pd.Timestamp("2000-10-01")] = 60.0     # water year 2001, first day
    s[pd.Timestamp("2003-02-01"):pd.Timestamp("2003-08-31")] = np.nan  # water year 2003 too short
    peaks = eng.annual_peaks(eng.make_record(s))
    assert list(peaks["year"]) == [2000, 2001, 2002]
    assert list(peaks["peak"][:2]) == [50.0, 60.0]
    cal = eng.annual_peaks(eng.make_record(s), year_start_month=1)
    assert 2000 in list(cal["year"]) and cal.loc[cal["year"] == 2000, "peak"].iloc[0] == 60.0


def test_hec_ssp_files_and_the_b17c_comparison():
    files, notes = eng.export_files(eng.make_record(_daily(15), location="X1"), "hec-ssp", dss_binary=False,
                                    regional_skew=0.1, regional_skew_mse=0.3)
    text = files["hec-ssp/annual_peaks.csv"].decode()
    rows = text.splitlines()
    assert rows[0] == "Date/Time,Flow (m3/s)"
    assert all(re.fullmatch(r"\d\d[A-Z][a-z]{2}\d{4} 1200,[0-9.]+", r) for r in rows[1:])
    settings = files["hec-ssp/b17c_settings.txt"].decode()
    assert "Multiple Grubbs-Beck" in settings and "Weighted, regional skew 0.1, regional skew MSE 0.3" in settings
    assert re.search(r"^\s+0\.01\s+100\s+[0-9.]+", settings, re.M)
    assert "hec-ssp/X1_flow_annual_peak.dss.csv" in files
    assert "IR-Century" in files["hec-ssp/X1_flow_annual_peak.dss.csv"].decode()
    with pytest.raises(ValueError, match="complete years"):
        eng.export_files(eng.make_record(_daily(3)), "hec-ssp")


# ── HEC-RAS ──────────────────────────────────────────────────────────────────


def _ras_values(block: str) -> list[float]:
    vals = []
    for line in block.splitlines()[2:]:
        vals += [float(line[i:i + 8]) for i in range(0, len(line), 8)]
    return vals


def test_hec_ras_block_is_ten_eight_character_columns():
    s = _daily(1)
    s.iloc[3] = np.nan
    block, notes = eng.hec_ras_block(eng.make_record(s))
    lines = block.splitlines()
    assert lines[0] == "Interval=1DAY" and lines[1].startswith("Flow Hydrograph= 365")
    assert all(len(line) <= 80 and len(line) % 8 == 0 for line in lines[2:])
    vals = _ras_values(block)
    assert len(vals) == 365 and vals[0] == pytest.approx(round(s.iloc[0], 2))
    assert vals[3] == pytest.approx(round((s.iloc[2] + s.iloc[4]) / 2, 2), abs=0.011)
    assert any("interpolation" in n for n in notes)
    stage, _ = eng.hec_ras_block(eng.make_record(_daily(1), variable="water_level"))
    assert "Stage Hydrograph=" in stage
    big = pd.Series([123456.7, 2.0], index=pd.date_range("2020-01-01", periods=2, freq="D"))
    assert _ras_values(eng.hec_ras_block(eng.make_record(big))[0]) == [123456.7, 2.0]


# ── SWMM ─────────────────────────────────────────────────────────────────────


def test_swmm_files_follow_the_manual():
    s = _daily(1)
    files, notes = eng.swmm_files(eng.make_record(s, location="G1"))
    dat = files["G1.dat"].splitlines()
    assert dat[0].startswith(";") and dat[1].split() == ["01/01/2000", "00:00", eng._num(s.iloc[0])]
    inline = files["G1_timeseries.inp"].splitlines()
    assert inline[0] == "[TIMESERIES]" and inline[2].split()[0] == "G1" and len(inline) == 2 + len(s)
    wiring = files["G1_sections.inp"]
    assert 'G1 FILE "G1.dat"' in wiring and "[INFLOWS]" in wiring and "OUTLET1  FLOW  G1  FLOW  1.0  1.0" in wiring
    rain, notes = eng.swmm_files(eng.make_record(_daily(1), variable="precipitation", location="G1"))
    assert "RG_G1  VOLUME  24:00  1.0  TIMESERIES  G1" in rain["G1_sections.inp"]


def test_swmm_notes_name_the_real_unit_and_the_gaps():
    s = _daily(1)
    s = s.drop(s.index[5:8])
    files, notes = eng.swmm_files(eng.make_record(s, unit="ft3/s", location="G1"))
    assert any("ft3/s: set FLOW_UNITS CFS" in n for n in notes)
    assert not any("m3/s" in n for n in notes)
    assert any(n.startswith("3 missing steps are left out") for n in notes)
    assert len(files["G1.dat"].splitlines()) == 1 + len(s)


def test_readme_records_the_source_licence():
    files, _ = eng.export_files(eng.make_record(_daily(1), location="X", source="usgs"), "fews")
    readme = files["fews/README.txt"].decode()
    assert "Data licence: US-PD" in readme and "Attribution: U.S. Geological Survey" in readme
    plain, _ = eng.export_files(eng.make_record(_daily(1), location="X"), "fews")
    assert "Data licence" not in plain["fews/README.txt"].decode()


# ── MODFLOW 6 ────────────────────────────────────────────────────────────────


def test_modflow6_text_has_one_period_per_step():
    s = _daily(1).iloc[:30]
    files, notes = eng.modflow6_text(eng.make_record(s, variable="water_level", location="W1"), cellid=(1, 2, 3))
    riv, tdis = files["w1.riv"], files["w1.tdis"]
    assert riv.count("BEGIN PERIOD") == 30 and "MAXBOUND 1" in riv and "PLACEHOLDERS" in riv
    first = re.search(r"BEGIN PERIOD 1\n.*\n  (.+)\n", riv).group(1).split()
    assert first[:3] == ["1", "2", "3"] and float(first[3]) == pytest.approx(s.iloc[0], rel=1e-6)
    assert first[-1] == "W1" and float(first[5]) == pytest.approx(s.min() - 1.0, rel=1e-6)
    assert "NPER 30" in tdis and tdis.count("  1  1  1.0") == 30 and "TIME_UNITS DAYS" in tdis
    wel, notes = eng.modflow6_text(eng.make_record(s, location="W1"))
    q = re.search(r"BEGIN PERIOD 1\n.*\n  (.+)\n", wel["w1.wel"]).group(1).split()
    assert float(q[3]) == pytest.approx(s.iloc[0] * 86400, rel=1e-6) and any("m3/d" in n for n in notes)
    monthly, _ = eng.modflow6_text(eng.make_record(_daily(1), variable="water_level"), period="MS")
    assert "NPER 12" in monthly["gauge.tdis"] and "  31  1  1.0" in monthly["gauge.tdis"]


def test_modflow6_text_loads_in_flopy(tmp_path):
    pytest.importorskip("flopy")
    import flopy

    s = _daily(1).iloc[:20]
    for var, pkg in (("water_level", "riv"), ("discharge", "wel")):
        rec = eng.make_record(s, variable=var, location="W1")
        ws = tmp_path / pkg
        eng.modflow6_flopy(rec, ws)                      # FloPy writes the skeleton...
        files, _ = eng.modflow6_text(rec)
        for name, text in files.items():                 # ...and our text replaces its package and TDIS
            (ws / name).write_text(text)
        sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
        spd = sim.get_model().get_package(pkg).stress_period_data.get_data()
        assert sim.tdis.nper.get_data() == 20 and len(spd) == 20
        assert spd[0][0][0] == (0, 0, 0)
        expect = s.iloc[19] if pkg == "riv" else s.iloc[19] * 86400
        assert spd[19][0][1] == pytest.approx(expect, rel=1e-6)


def test_modflow6_flopy_engine_through_export_files():
    pytest.importorskip("flopy")
    files, notes = eng.export_files(eng.make_record(_daily(1).iloc[:10], variable="water_level", location="W1"),
                                    "modflow6", engine="flopy")
    assert "modflow6/w1.riv" in files and "modflow6/mfsim.nam" in files
    assert any("FloPy" in n for n in notes)


# ── Delft-FEWS ───────────────────────────────────────────────────────────────

PI = "{http://www.wldelft.nl/fews/PI}"
#: The header elements in the order pi_timeseries.xsd (HeaderComplexType) requires.
XSD_ORDER = ["type", "moduleInstanceId", "locationId", "parameterId", "qualifierId", "ensembleId", "timeStep",
             "startDate", "endDate", "forecastDate", "missVal", "longName", "stationName", "lat", "lon", "x", "y",
             "z", "units"]


def test_fews_pi_xml_header_order_and_events():
    s = _daily(1)
    s.iloc[2] = np.nan
    xml, notes = eng.fews_pi_xml(eng.make_record(s, location="G1", name="Gauge & co", lat=45.0, lon=6.5))
    root = ET.fromstring(xml)
    assert root.tag == f"{PI}TimeSeries" and root.attrib["version"] == "1.2"
    assert root.find(f"{PI}timeZone").text == "0.0"
    header = root.find(f"{PI}series/{PI}header")
    tags = [h.tag.replace(PI, "") for h in header]
    assert tags == [t for t in XSD_ORDER if t in tags]
    assert header.find(f"{PI}type").text == "mean" and header.find(f"{PI}stationName").text == "Gauge & co"
    assert header.find(f"{PI}timeStep").attrib == {"unit": "day", "multiplier": "1"}
    events = root.findall(f"{PI}series/{PI}event")
    assert len(events) == len(s) and events[2].attrib["value"] == "-999"
    assert events[0].attrib["date"] == "2000-01-01" and events[0].attrib["time"] == "00:00:00"
    assert any("time zone" in n for n in notes)


# ── Raven ────────────────────────────────────────────────────────────────────


def test_raven_observation_block():
    s = _daily(1)
    s.iloc[4] = np.nan
    rvt, _ = eng.raven_rvt(eng.make_record(s), subbasin_id=36)
    lines = rvt.splitlines()
    assert lines[1] == ":ObservationData HYDROGRAPH 36 m3/s"
    assert lines[2].split() == ["2000-01-01", "00:00:00.0", "1.0", str(len(s))]
    assert lines[-1] == ":EndObservationData" and len(lines) == 4 + len(s)
    assert float(lines[3 + 4]) == eng.RAVEN_MISSING
    rain, _ = eng.raven_rvt(eng.make_record(pd.Series(1.0, index=pd.date_range("2020-01-01", periods=48,
                                                                               freq="h")), variable="precipitation"))
    rl = rain.splitlines()
    assert rl[1] == ":Data PRECIP mm/d" and rl[2].split()[-1] == "2" and float(rl[3]) == 24.0


# ── the faces ────────────────────────────────────────────────────────────────


def test_export_series_returns_files_text_and_a_zip():
    out = eng.export_series(_daily(2), "raven", location="G1", as_zip=True)
    assert out["label"] == "Raven" and out["filename"] == "aquascope-G1-raven.zip"
    assert {f["path"] for f in out["files"]} == {"raven/G1.rvt", "raven/README.txt"}
    assert out["files"][0]["text"].startswith("#")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(out["zip_base64"]))) as zf:
        assert sorted(zf.namelist()) == ["raven/G1.rvt", "raven/README.txt"]
        assert "Format reference: https://raven.uwaterloo.ca" in zf.read("raven/README.txt").decode()
    assert eng.files_from(out)["raven/G1.rvt"].startswith(b"#")
    with pytest.raises(ValueError, match="takes flow"):
        eng.export_series(_daily(1), "hec-hms", variable="water_level")
    with pytest.raises(ValueError, match="unknown tool"):
        eng.export_series(_daily(1), "mike11")


def test_all_tools_bundle_skips_what_does_not_fit():
    out = eng.export_series(_daily(3), "all", location="G1", dss_binary=False, with_text=False)
    folders = {f["path"].split("/")[0] for f in out["files"]}
    assert folders == {"hec-hms", "hec-ras", "dss", "swmm", "modflow6", "fews", "raven"}
    assert any("HEC-SSP skipped" in n for n in out["notes"])
    assert "text" not in out["files"][0]


def test_menu_lists_the_tools_for_the_variable():
    assert [t["id"] for t in eng.menu("discharge")["tools"]] == eng.tools_for("flow")
    assert "hec-hms" not in [t["id"] for t in eng.menu("water_level")["tools"]]
    assert eng.menu("ph") == {"kind": None, "tools": []}


def test_export_station_fetches_then_exports():
    fetched = {"series": _daily(2), "variable": "discharge", "unit": "m3/s", "note": ""}
    with patch("aquascope.explore.fetch_series", return_value=fetched) as f:
        out = eng.export_station("usgs", "01646500", "fews")
    f.assert_called_once()
    assert out["location"] == "01646500" and "<locationId>01646500</locationId>" in out["files"][0]["text"]
    with patch("aquascope.explore.fetch_series", return_value={**fetched, "series": pd.Series(dtype=float)}):
        assert "error" in eng.export_station("usgs", "x", "fews")


def test_cli_export_from_a_csv(tmp_path, monkeypatch, capsys):
    from aquascope import cli

    assert list(cli.EXPORT_TOOLS) == list(eng.TOOLS)
    s = _daily(2)
    src = tmp_path / "q.csv"
    pd.DataFrame({"date": s.index, "flow": s.values}).to_csv(src, index=False)
    monkeypatch.setattr(sys, "argv", ["aquascope", "export", "--to", "swmm", "--file", str(src),
                                      "-o", str(tmp_path / "out")])
    cli.main()
    out = capsys.readouterr().out
    assert out.startswith("SWMM: 4 file(s)") and (tmp_path / "out" / "swmm" / "q.dat").exists()
    monkeypatch.setattr(sys, "argv", ["aquascope", "export", "--to", "raven", "--file", str(src), "--zip",
                                      "-o", str(tmp_path / "bundle"), "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["written"] == [str(tmp_path / "bundle.zip")]


def test_cli_export_from_a_station(tmp_path, monkeypatch, capsys):
    from aquascope import cli

    fetched = {"series": _daily(2), "variable": "water_level", "unit": "m", "note": ""}
    monkeypatch.setattr("aquascope.explore.fetch_series", lambda *a, **k: fetched)
    monkeypatch.setattr(sys, "argv", ["aquascope", "export", "--to", "modflow6", "--station", "uk_ea/ABC",
                                      "--cell", "2", "3", "4", "--cond", "50", "--rbot", "1.5",
                                      "-o", str(tmp_path / "o")])
    cli.main()
    riv = (tmp_path / "o" / "modflow6" / "abc.riv").read_text()
    assert "  2 3 4  " in riv and "  50  1.5  ABC" in riv and "PLACEHOLDERS" not in riv
    monkeypatch.setattr(sys, "argv", ["aquascope", "export", "--to", "raven", "--station", "nosuch"])
    with pytest.raises(SystemExit):
        cli.main()


def test_mcp_engineering_export_is_bounded_and_registered(tmp_path):
    pytest.importorskip("mcp")
    import asyncio

    from aquascope import mcp_server as m

    fetched = {"series": _daily(3), "variable": "discharge", "unit": "m3/s", "note": ""}
    with patch("aquascope.explore.fetch_series", return_value=fetched):
        out = m.engineering_export("usgs", "01646500", "swmm", max_chars=500, out_dir=str(tmp_path))
    dat = next(f for f in out["files"] if f["path"] == "swmm/01646500.dat")
    assert dat["truncated"] and len(dat["text"]) == 500 and out["license"]
    assert (tmp_path / "swmm" / "01646500.dat").stat().st_size > 500
    assert "error" in m.engineering_export("nowhere", "1", "swmm")
    assert "error" in m.engineering_export("usgs", "1", "mike11")
    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert "engineering_export" in names


def test_hec_module_no_longer_says_dss_is_proprietary():
    import aquascope.io.hec as hec

    assert "proprietary" not in (hec.__doc__ or "")
    assert "hecdss" in hec.__doc__


def test_a_long_gap_is_cut_not_invented():
    s = pd.Series(np.arange(100.0), index=pd.date_range("2000-01-01", periods=100, freq="D"))
    s.iloc[20:50] = np.nan
    s.iloc[70:73] = np.nan
    block, notes = eng.hec_ras_block(eng.make_record(s))
    assert "Flow Hydrograph= 50 " in block and _ras_values(block)[0] == 50.0
    assert any("latest stretch" in n for n in notes) and any("3 missing steps" in n for n in notes)
    files, notes = eng.modflow6_text(eng.make_record(s, variable="water_level"))
    assert "NPER 50" in files["gauge.tdis"] and files["gauge.riv"].count("BEGIN PERIOD") == 50
