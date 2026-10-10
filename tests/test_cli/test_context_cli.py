"""`aquascope context LAT LON` (and `--bbox`): one line per layer and the attribution, or JSON (#520)."""

from __future__ import annotations

import json
import sys

import pytest

from aquascope import cli

POINT = {
    "lat": 51.415, "lon": -0.308,
    "layers": {
        "flood_history": {"summary": "3 flood events in the news within 25 km, latest 2024-01-05."},
        "dams": {"summary": "No dams in Global Dam Watch within 50 km."},
    },
    "summary": ["3 flood events in the news within 25 km, latest 2024-01-05.",
                "No dams in Global Dam Watch within 50 km."],
    "attribution": ["Global Dam Watch database v1.0 (Lehner et al. 2024), CC BY 4.0"],
}


def test_context_prints_one_line_per_layer(monkeypatch, capsys):
    seen = {}

    def fake(lat, lon, layers=None):
        seen.update(lat=lat, lon=lon, layers=layers)
        return POINT

    monkeypatch.setattr("aquascope.context.place_context", fake)
    monkeypatch.setattr(sys, "argv", ["aquascope", "context", "51.415", "-0.308", "--layers", "flood_history,dams"])
    cli.main()
    out = capsys.readouterr().out
    assert seen == {"lat": 51.415, "lon": -0.308, "layers": "flood_history,dams"}
    assert out.startswith("Context of 51.4150, -0.3080")
    assert "Flood history" in out and "3 flood events in the news" in out
    assert "Data: Global Dam Watch" in out


def test_context_json_and_a_box(monkeypatch, capsys):
    seen = {}

    def fake_area(w, s, e, n, layers=None):
        seen["bbox"] = (w, s, e, n)
        return {**POINT, "bbox": [w, s, e, n]}

    monkeypatch.setattr("aquascope.context.area_context", fake_area)
    monkeypatch.setattr(sys, "argv", ["aquascope", "context", "--bbox=-1,51,0,52", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["bbox"] == [-1.0, 51.0, 0.0, 52.0]
    assert seen["bbox"] == (-1.0, 51.0, 0.0, 52.0)


def test_context_needs_a_place(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["aquascope", "context"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


FLOODS = {
    "summary": "3 flood events in the news in this box, Jul 2021; Sentinel-1 radar made 150 flood detections (20 m "
               "pixels).",
    "available": True, "bbox": [5.5, 50.5, 6.0, 51.0],
    "news": {"events": [{"start": "2021-07-14", "end": "2021-07-16", "area_km2": 120.5}], "events_found": 3},
    "radar": {"by_month": {"2021-07": 150}},
    "attribution": "Groundsource (CC BY 4.0); Microsoft (MIT)",
}


def test_context_floods_past_lists_a_box(monkeypatch, capsys):
    seen = {}

    def fake(month=None, *, start=None, end=None, bbox=None, limit=20):
        seen.update(month=month, start=start, end=end, bbox=bbox, limit=limit)
        return FLOODS

    monkeypatch.setattr("aquascope.context.floods_past.flood_events_month", fake)
    monkeypatch.setattr(sys, "argv", ["aquascope", "context", "--floods-past", "--from", "2021-07", "--to", "2021-07",
                                      "--bbox=5.5,50.5,6,51"])
    cli.main()
    out = capsys.readouterr().out
    assert seen == {"month": None, "start": "2021-07", "end": "2021-07", "bbox": (5.5, 50.5, 6.0, 51.0), "limit": 20}
    assert out.startswith("3 flood events in the news in this box")
    assert "2021-07-14 to 2021-07-16  120 km2" in out and "and 2 more" in out
    assert "Radar detections by month: 2021-07 150" in out and "Data: Groundsource" in out


def test_context_month_alone_means_floods_past(monkeypatch, capsys):
    monkeypatch.setattr("aquascope.context.floods_past.flood_events_month",
                        lambda month=None, **kw: {**FLOODS, "bbox": None, "month": month})
    monkeypatch.setattr(sys, "argv", ["aquascope", "context", "--month", "2021-07", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["month"] == "2021-07"


def test_context_floods_past_rejects_a_long_window(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["aquascope", "context", "--floods-past", "--from", "2000-01", "--to", "2020-01"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
