"""CLI coverage for USGS collector options."""

from __future__ import annotations

import sys

import pytest

from aquascope.cli import main


def test_usgs_filter_options_are_forwarded(monkeypatch):
    """The CLI exposes each filter supported by the keyless USGS path."""
    captured: dict[str, object] = {}

    class FakeUSGSCollector:
        def __init__(self, api_key: str):
            assert api_key == "DEMO_KEY"

        def collect(self, **kwargs):
            captured.update(kwargs)
            return [object()]

    monkeypatch.setattr("aquascope.collectors.USGSCollector", FakeUSGSCollector)
    monkeypatch.setattr("aquascope.utils.storage.save_records", lambda *args, **kwargs: "output.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aquascope", "collect", "--source", "usgs", "--days", "7",
            "--station-id", "01646500", "--parameter", "00060",
            "--bbox=-77.2,38.8,-77.0,39.0", "--state-code", "MD",
            "--county-code", "24033", "--huc", "02070010",
        ],
    )

    main()

    assert captured == {
        "days": 7,
        "station_id": "01646500",
        "parameter": "00060",
        "bbox": "-77.2,38.8,-77.0,39.0",
        "stateCd": "MD",
        "countyCd": "24033",
        "huc": "02070010",
    }


def test_usgs_denies_unrecognised_argument(monkeypatch):
    """CLI raises SystemExit when an unrecognised argument is passed to the USGS collector."""
    captured: dict[str, object] = {}

    class FakeUSGSCollector:
        def __init__(self, api_key: str):
            assert api_key == "DEMO_KEY"

        def collect(self, **kwargs):
            captured["collect_kwargs"] = kwargs
            return [object()]

    monkeypatch.setattr("aquascope.collectors.USGSCollector", FakeUSGSCollector)
    monkeypatch.setattr("aquascope.utils.storage.save_records", lambda *args, **kwargs: "output.json")
    monkeypatch.setattr(
        sys,
        "argv",
        ["aquascope", "collect", "--source", "usgs", "--unrecognised", "unrecognised"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2


def test_usgs_passes_explicit_api_key_to_collector(monkeypatch):
    """An explicit CLI API key is supplied to the USGS collector constructor."""
    captured: dict[str, object] = {}

    class FakeUSGSCollector:
        def __init__(self, api_key: str):
            captured["api_key"] = api_key

        def collect(self, **kwargs):
            return [object()]

    monkeypatch.setattr("aquascope.collectors.USGSCollector", FakeUSGSCollector)
    monkeypatch.setattr("aquascope.utils.storage.save_records", lambda *args, **kwargs: "output.json")
    monkeypatch.setattr(
        sys,
        "argv",
        ["aquascope", "collect", "--source", "usgs", "--api-key", "explicit-key"],
    )

    main()

    assert captured["api_key"] == "explicit-key"
