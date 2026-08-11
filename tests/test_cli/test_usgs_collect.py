"""CLI coverage for USGS collector options."""

from __future__ import annotations

import sys

import pytest

from aquascope.cli import main


@pytest.fixture
def run_usgs(monkeypatch):
    captured_args: dict[str, object] = {}

    class FakeUSGSCollector:
        def __init__(self, api_key: str | None):
            assert api_key == api_key

        def collect(self, **kwargs):
            captured_args.update(kwargs)
            return [object()]

    monkeypatch.setattr(
        "aquascope.collectors.USGSCollector",
        FakeUSGSCollector,
    )
    monkeypatch.setattr(
        "aquascope.utils.storage.save_records",
        lambda *args, **kwargs: "output.json",
    )

    def run(*args: str) -> dict[str, object]:
        monkeypatch.setattr(
            sys,
            "argv",
            ["aquascope", "collect", "--source", "usgs", *args],
        )
        main()
        return captured_args

    return run


def test_usgs_filter_options_are_forwarded(run_usgs):
    """Check that the CLI exposes each kwarg supported by the keyless USGS path, excluding api-key"""
    arguments = [
        "--days", "7", "--station-id", "01646500", "--parameter", "00060",
        "--bbox=-77.2,38.8,-77.0,39.0", "--state-code", "MD",
        "--county-code", "24033", "--huc", "02070010",
    ]

    captured_arguments = run_usgs(*arguments)
    assert captured_arguments == {
        "days": 7,
        "station_id": "01646500",
        "parameter": "00060",
        "bbox": "-77.2,38.8,-77.0,39.0",
        "stateCd": "MD",
        "countyCd": "24033",
        "huc": "02070010",
    }


def test_usgs_denies_unrecognised_argument(run_usgs):
    """Check that cli.py raises SystemExit when an unrecognised argument is passed to the USGS collector."""
    arguments = ["--unrecognised", "unrecognised"]

    with pytest.raises(SystemExit) as exc_info:
        run_usgs(*arguments)
    assert exc_info.value.code == 2


def test_usgs_passes_explicit_api_key_to_collector(run_usgs):
    """Check that explicit API keys are handled by the constructor and aren't considered as kwargs."""
    arguments = ["--api-key", "explicit-key"]
    captured_arguments = run_usgs(*arguments)

    assert captured_arguments == {}
