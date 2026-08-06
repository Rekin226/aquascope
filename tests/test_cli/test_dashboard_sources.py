"""Drift guard ensuring the dashboard Collect page stays in sync with all registered collectors."""

from __future__ import annotations

import pytest

from aquascope.dashboard.views.collect import SOURCES, _FACTORIES
from aquascope.schemas.water_data import DataSource

# DataSources that are enum placeholders or not yet standalone collector classes
UNIMPLEMENTED_SOURCES = {"usgs_groundwater", "grace"}

# Map DataSource enum values to their SOURCES dictionary key where they differ
ENUM_TO_SOURCE_KEY = {
    "france_hubeau": "hubeau_hydrometrie",
    "taiwan_wra": "taiwan_wra_level",
}


def test_dashboard_collect_page_covers_all_registered_sources():
    """The Collect page source map must cover every registered collector."""
    raw_sources = {ds.value for ds in DataSource} - UNIMPLEMENTED_SOURCES
    mapped_sources = {ENUM_TO_SOURCE_KEY.get(s, s) for s in raw_sources}
    missing = mapped_sources - set(SOURCES.keys())
    assert not missing, (
        f"The dashboard Collect page (aquascope/dashboard/views/collect.py) is missing "
        f"the following registered data sources: {sorted(missing)}. "
        f"Please wire them into `SOURCES`, `_source_form`, and `_run_collector`."
    )


def test_dashboard_run_collector_supports_all_sources():
    """Every key in SOURCES must have a factory entry in _FACTORIES."""
    assert set(_FACTORIES) == set(SOURCES), (
        f"Mismatch between SOURCES and _FACTORIES keys. "
        f"Missing in _FACTORIES: {set(SOURCES) - set(_FACTORIES)}. "
        f"Extra in _FACTORIES: {set(_FACTORIES) - set(SOURCES)}."
    )


@pytest.mark.parametrize("source_key", sorted(SOURCES.keys()))
def test_every_dashboard_source_has_metadata_tuple(source_key):
    """Every source entry has a (label, region, description) metadata tuple."""
    label, region, description = SOURCES[source_key]
    assert label and isinstance(label, str)
    assert region and isinstance(region, str)
    assert description and isinstance(description, str)
