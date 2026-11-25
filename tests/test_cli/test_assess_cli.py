"""`aquascope assess LAT LON`: the sufficiency table as a short readable table, or JSON."""

from __future__ import annotations

import json
import sys

from aquascope import cli

CANNED = {
    "point": {"lat": 51.415, "lon": -0.308},
    "stations": [{"source": "uk_ea", "station_id": "8496ce69", "name": "Kingston", "distance_km": 0.0,
                  "variables": ["discharge"], "period_start": "1883-01-01", "period_end": None, "years": 143.7,
                  "url": None}],
    "catchment": {"area_km2": 9948.0, "upstream_area_km2": 9948.0},
    "context": {"years_by_variable": {"discharge": 143.7}, "resolution_by_variable": {"discharge": "daily"},
                "area_km2": 9948.0, "return_period": 100.0, "donors": 10, "available": ["glofas"], "ungauged": False},
    "sufficiency": [
        {"method": "at_site_flood_frequency", "label": "At-site flood frequency", "status": "defensible",
         "reason": "the record supports it", "tool": "analyze_station",
         "station": {"source": "uk_ea", "station_id": "8496ce69"}},
        {"method": "similar_basins", "label": "Donor gauges by catchment similarity", "status": "marginal",
         "reason": "meant for an ungauged point; a gauge is available here", "tool": "similar_basins",
         "station": None},
        {"method": "spi", "label": "Standardized Precipitation Index", "status": "not_defensible",
         "reason": "no precipitation record at this site", "tool": "analyze_station", "station": None},
    ],
    "notes": ["Record resolution is not in the catalog; daily is assumed for every variable."],
}


def test_assess_prints_defensible_first_with_one_line_per_method(monkeypatch, capsys):
    seen = {}

    def fake(lat, lon, **kwargs):
        seen.update({"lat": lat, "lon": lon, **kwargs})
        return CANNED

    monkeypatch.setattr("aquascope.explore.assess_site", fake)
    monkeypatch.setattr(sys, "argv", ["aquascope", "assess", "51.415", "-0.308", "--problem", "flood_risk",
                                      "--return-period", "100", "--radius-km", "25"])
    cli.main()
    out = capsys.readouterr().out
    assert seen == {"lat": 51.415, "lon": -0.308, "radius_km": 25.0, "problem": "flood_risk", "return_period": 100.0}
    assert "1 gauge within 25 km" in out and "catchment 9,948 km²" in out and "10 donors" in out
    assert "discharge: 143.7 yr, Kingston (uk_ea/8496ce69)" in out
    assert out.index("defensible") < out.index("marginal") < out.index("not defensible")
    assert "At-site flood frequency" in out and "the record supports it" in out
    assert "Standardized Precipitation Index" in out and "no precipitation record at this site" in out
    assert "- Record resolution is not in the catalog" in out


def test_assess_json_prints_the_engine_result(monkeypatch, capsys):
    monkeypatch.setattr("aquascope.explore.assess_site", lambda *a, **k: CANNED)
    monkeypatch.setattr(sys, "argv", ["aquascope", "assess", "51.415", "-0.308", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == CANNED


def test_assess_ungauged_says_so(monkeypatch, capsys):
    bare = {**CANNED, "stations": [], "catchment": {"error": "ocean"},
            "context": {**CANNED["context"], "years_by_variable": {}, "area_km2": None, "donors": None,
                        "ungauged": True}}
    monkeypatch.setattr("aquascope.explore.assess_site", lambda *a, **k: bare)
    monkeypatch.setattr(sys, "argv", ["aquascope", "assess", "-20", "130"])
    cli.main()
    out = capsys.readouterr().out
    assert "0 gauges within 50 km" in out and "catchment unknown" in out and "ungauged" in out


def test_assess_shows_the_record_quality_and_passes_the_hub_opt_in(monkeypatch, capsys):
    """#375 follow-up: the quality half on the variable's line, and --archive-hub reaching the engine."""
    seen = {}
    read = {**CANNED, "context": {**CANNED["context"], "completeness_by_variable": {"discharge": 0.9952},
                                  "longest_gap_days": {"discharge": 174}, "flagged_fraction": {"discharge": 0.0021},
                                  "change_points": {"discharge": [1991]}}}

    def fake(lat, lon, **kwargs):
        seen.update(kwargs)
        return read

    monkeypatch.setattr("aquascope.explore.assess_site", fake)
    monkeypatch.setattr(sys, "argv", ["aquascope", "assess", "51.415", "-0.308", "--archive-hub"])
    cli.main()
    out = capsys.readouterr().out
    assert seen["archive_hub"] is True
    assert ("discharge: 143.7 yr, Kingston (uk_ea/8496ce69)  ·  99.5% complete, longest gap 174 days, 0.2% flagged, "
            "change point 1991") in out


def test_assess_without_a_quality_half_keeps_the_plain_line(monkeypatch, capsys):
    seen = {}

    def fake(lat, lon, **kwargs):
        seen.update(kwargs)
        return CANNED

    monkeypatch.setattr("aquascope.explore.assess_site", fake)
    monkeypatch.setattr(sys, "argv", ["aquascope", "assess", "51.415", "-0.308"])
    cli.main()
    out = capsys.readouterr().out
    assert "archive_hub" not in seen  # unset: AQUASCOPE_ARCHIVE_HUB decides, off by default
    assert "discharge: 143.7 yr, Kingston (uk_ea/8496ce69)\n" in out
