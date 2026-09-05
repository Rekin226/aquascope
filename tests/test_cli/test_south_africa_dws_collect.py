"""CLI coverage for the South Africa DWS collector."""

from __future__ import annotations

import sys

import pytest

from aquascope.cli import main


def test_dws_options_are_forwarded(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCollector:
        def collect(self, **kwargs):
            captured.update(kwargs)
            return [object()]

    monkeypatch.setattr("aquascope.collectors.SouthAfricaDWSCollector", FakeCollector)
    monkeypatch.setattr("aquascope.utils.storage.save_records", lambda *args, **kwargs: "output.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aquascope",
            "collect",
            "--source",
            "south_africa_dws",
            "--station",
            "C1H001",
            "--variable",
            "water_level",
            "--start-date",
            "2026-01-20",
            "--end-date",
            "2026-01-21",
        ],
    )

    main()

    assert captured == {
        "station_id": "C1H001",
        "variable": "water_level",
        "start_date": "2026-01-20",
        "end_date": "2026-01-21",
    }


def test_dws_days_are_forwarded(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCollector:
        def collect(self, **kwargs):
            captured.update(kwargs)
            return []

    monkeypatch.setattr("aquascope.collectors.SouthAfricaDWSCollector", FakeCollector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aquascope",
            "collect",
            "--source",
            "south_africa_dws",
            "--station",
            "A2H076",
            "--days",
            "7",
        ],
    )

    main()

    assert captured == {"station_id": "A2H076", "variable": "discharge", "days": 7}


def test_dws_requires_station(monkeypatch, caplog):
    monkeypatch.setattr(
        sys,
        "argv",
        ["aquascope", "collect", "--source", "south_africa_dws"],
    )

    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    assert any("requires --station" in record.message for record in caplog.records)
