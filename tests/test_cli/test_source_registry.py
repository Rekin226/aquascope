"""Tests that the CLI's source registries stay in sync.

The ``--source`` choices, the collector map, and the ``list-sources`` info table
are three hand-maintained lists that have drifted apart before: a collector
could be registered in ``collectors/__init__.py`` yet be unreachable from the
CLI, and an info entry keyed by anything other than its ``DataSource`` value
silently rendered as a placeholder.
"""

from __future__ import annotations

import argparse
import sys

import pytest

from aquascope.cli import cmd_list_sources, main
from aquascope.schemas.water_data import DataSource

# Sources with a working collector reachable from `aquascope collect`.
# GRACE and USGS_GW are declared in DataSource but have no collector yet, and
# India WRIS needs required arguments the collect command does not pass.
CLI_SOURCES = (
    "grdc",
    "hubeau_hydrometrie",
    "japan_mlit",
    "korea_wamis",
    "pegelonline",
)


@pytest.mark.parametrize("source", CLI_SOURCES)
def test_source_is_a_valid_collect_choice(source, capsys, monkeypatch):
    """Every registered collector is reachable from `aquascope collect`."""
    monkeypatch.setattr(sys, "argv", ["aquascope", "collect", "--source", "__nonexistent__"])
    with pytest.raises(SystemExit):
        main()
    err = capsys.readouterr().err
    assert source in err, f"'{source}' is not an `aquascope collect --source` choice"


def test_list_sources_renders_metadata_for_recent_sources(capsys):
    """Recently added sources render real metadata, not placeholders."""
    cmd_list_sources(argparse.Namespace())
    out = capsys.readouterr().out

    assert "GRDC" in out
    assert "Hub'Eau" in out
    assert "PEGELONLINE" in out
    # The info table is keyed by DataSource.value; a mismatched key falls back
    # to printing the raw enum value alongside em-dash placeholders.
    assert DataSource.HUBEAU.value not in out
    assert DataSource.GRDC.value not in out
    assert "Germany" in out


def test_grdc_rejects_an_unknown_mode(monkeypatch):
    """GRDC's --mode maps to source_type and only accepts its two values."""
    monkeypatch.setattr(sys, "argv", ["aquascope", "collect", "--source", "grdc", "--mode", "bogus"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_pegelonline_requires_station_uuid(monkeypatch, caplog):
    """PEGELONLINE fails clearly before collection when no station is supplied."""
    monkeypatch.setattr(sys, "argv", ["aquascope", "collect", "--source", "pegelonline"])
    with caplog.at_level("ERROR"), pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert any("requires --station" in record.message for record in caplog.records)
