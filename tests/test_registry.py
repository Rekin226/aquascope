"""Drift guards for the shared source registry (#163, #187).

The registry is the one place that describes every collector. These tests
make sure it cannot quietly fall out of step with the code: every collector
class must be registered, every registered key must build, the
``supports_station_lookup`` flag must match whether ``stations()`` is really
overridden, and license metadata must be present before a source can be
marked redistributable.
"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import pytest

from aquascope import collectors as collectors_pkg
from aquascope.collectors.base import BaseCollector
from aquascope.registry import (
    SOURCES,
    StationCatalog,
    build_collector,
    find_stations,
    redistributable_sources,
    source_keys,
    station_catalogs,
    station_sources,
)
from aquascope.schemas.station import VARIABLES, Station, in_bbox


def _collector_classes() -> dict[str, type[BaseCollector]]:
    return {
        name: obj
        for name, obj in inspect.getmembers(collectors_pkg, inspect.isclass)
        if issubclass(obj, BaseCollector) and obj is not BaseCollector
    }


@pytest.fixture(scope="module")
def built() -> dict[str, BaseCollector]:
    """Instantiate every registered source once (no network: constructors only)."""
    return {key: build_collector(key) for key in source_keys()}


def test_every_registered_key_builds(built):
    assert set(built) == set(SOURCES)
    for key, collector in built.items():
        assert isinstance(collector, BaseCollector), key


def test_every_collector_class_is_registered(built):
    """A collector class that no registry key builds is unreachable from CLI, dashboard, MCP and harvest."""
    registered_types = {type(c) for c in built.values()}
    unregistered = sorted(name for name, cls in _collector_classes().items() if cls not in registered_types)
    assert not unregistered, f"Collector classes missing from aquascope.registry: {unregistered}"


def test_station_lookup_flag_matches_override(built):
    for key, collector in built.items():
        assert SOURCES[key].supports_station_lookup == type(collector).supports_stations(), (
            f"{key}: registry says supports_station_lookup={SOURCES[key].supports_station_lookup} "
            f"but {type(collector).__name__}.stations() override is {type(collector).supports_stations()}"
        )


def test_default_stations_raises_not_implemented(built):
    for key, collector in built.items():
        if not SOURCES[key].supports_station_lookup:
            with pytest.raises(NotImplementedError):
                collector.stations()


@pytest.mark.parametrize("key", sorted(SOURCES))
def test_metadata_is_complete(key):
    meta = SOURCES[key]
    assert meta.key == key
    assert meta.label and meta.region and meta.description
    assert meta.agency, f"{key}: agency missing"
    assert meta.country, f"{key}: country missing"
    assert meta.variables, f"{key}: variables missing"
    assert set(meta.variables) <= set(VARIABLES)
    assert meta.output_model, f"{key}: output_model missing"
    assert meta.license, f"{key}: license missing"
    assert meta.attribution, f"{key}: attribution missing"
    if meta.redistributable:
        assert meta.license != "unknown", f"{key}: redistributable=True needs a real license id"
    if meta.requires_api_key:
        assert meta.api_key_signup_url, f"{key}: requires_api_key needs a signup URL"


def test_helper_lists():
    assert set(station_sources()) == {k for k, m in SOURCES.items() if m.supports_station_lookup}
    assert set(station_sources("discharge")) <= set(station_sources())
    assert "usgs" in station_sources("discharge")
    assert "grdc" not in redistributable_sources()
    assert "usgs" in redistributable_sources()
    assert source_keys() == sorted(SOURCES)


def test_usgs_output_model():
    assert SOURCES["usgs"].output_model == "StreamflowReading | WaterLevelReading | WaterQualitySample"


def test_station_variables_validated():
    with pytest.raises(ValueError):
        Station(source="x", station_id="1", latitude=0, longitude=0, variables=("lava",))


def test_in_bbox():
    assert in_bbox(10, 20, None)
    assert in_bbox(10, 20, (19, 9, 21, 11))
    assert not in_bbox(10, 25, (19, 9, 21, 11))


class _Good(BaseCollector):
    name = "good"

    def __init__(self):
        pass

    def fetch_raw(self, **kwargs):
        return []

    def normalise(self, raw):
        return []

    def stations(self, *, bbox=None, variable=None, max_items=None):
        pts = [Station(source="good", station_id="a", latitude=1.0, longitude=2.0, variables=("discharge",))]
        return [p for p in pts if in_bbox(p.latitude, p.longitude, bbox)]


class _Bad(_Good):
    name = "bad"

    def stations(self, *, bbox=None, variable=None, max_items=None):
        raise RuntimeError("endpoint 404")


def test_station_catalogs_keeps_failures_visible():
    def fake_build(key, api_key=None, **kw):
        return _Good() if key == "usgs" else _Bad()

    with patch("aquascope.registry.build_collector", side_effect=fake_build):
        cats = station_catalogs(sources=["usgs", "uk_ea"])
    assert set(cats) == {"usgs", "uk_ea"}
    assert isinstance(cats["usgs"], StationCatalog)
    assert cats["usgs"].ok and len(cats["usgs"].stations) == 1
    assert not cats["uk_ea"].ok and "endpoint 404" in cats["uk_ea"].error
    with patch("aquascope.registry.build_collector", side_effect=fake_build):
        flat = find_stations(sources=["usgs", "uk_ea"])
        empty = find_stations(sources=["usgs"], bbox=(100, 40, 101, 41))
    assert [s.station_id for s in flat] == ["a"]
    assert empty == []


def test_a_wrapped_catalog_failure_keeps_its_cause():
    """BOM wraps every failed parameter type in one RuntimeError; the 503 under it must reach the health report
    (#498), or the harvest files the outage as an unclassified code fault."""

    class _Wrapped:
        def stations(self, **kw):
            try:
                raise RuntimeError("All 3 attempts failed for https://www.bom.gov.au/waterdata/services (status 503)")
            except RuntimeError as exc:
                raise RuntimeError("BOM getStationList failed for all 10 parameter type(s)") from exc

    with patch("aquascope.registry.build_collector", return_value=_Wrapped()):
        cat = station_catalogs(sources=["bom"])["bom"]
    assert cat.error.startswith("RuntimeError: BOM getStationList failed for all 10")
    assert "(from RuntimeError: All 3 attempts failed" in cat.error and "status 503" in cat.error


def test_station_catalogs_rejects_unknown_source():
    with pytest.raises(ValueError):
        station_catalogs(sources=["nope"])


def test_station_catalogs_filters_by_variable():
    """Sources that don't measure the variable are skipped without being called."""
    with patch("aquascope.registry.build_collector") as fake:
        cats = station_catalogs(sources=["ireland_opw"], variable="discharge")
    assert cats == {}
    fake.assert_not_called()


def _plain_http_hosts(collector: BaseCollector) -> list[str]:
    """Module-level and class-level ``http://`` URLs a collector calls (station page links inside functions are not)."""
    import sys

    module = sys.modules[type(collector).__module__]
    found: list[str] = []
    for name, value in list(vars(module).items()) + list(vars(type(collector)).items()):
        if name.startswith("__"):
            continue
        values = value.values() if isinstance(value, dict) else [value]
        found.extend(v for v in values if isinstance(v, str) and v.startswith("http://"))
    return found


def test_plain_http_catalog_sources_are_declared_browser_unreachable(built):
    """A plain-http API cannot be called from the https Explorer (mixed content), so its pins must say so (#408).

    Hydroscope shipped 704 clickable pins whose every click failed silently.
    This refuses the next one: a catalogue source whose collector talks to an
    ``http://`` host has to carry ``browser_reachable=False`` so the station
    card explains itself instead of reporting an empty record.
    """
    undeclared = {
        key for key in station_sources()
        if _plain_http_hosts(built[key]) and SOURCES[key].browser_reachable
    }
    assert not undeclared, f"http-only catalogue sources not declared browser_reachable=False: {sorted(undeclared)}"
    # The check is not vacuous: it sees the known case.
    assert _plain_http_hosts(built["greece_hydroscope"]) and not SOURCES["greece_hydroscope"].browser_reachable


def test_browser_reachable_defaults_on():
    unreachable = sorted(k for k, m in SOURCES.items() if not m.browser_reachable)
    assert unreachable == ["bom", "greece_hydroscope", "greece_openhi", "poland_imgw"], unreachable
