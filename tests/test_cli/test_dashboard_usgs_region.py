"""USGS region-filter options depend on API key presence (issue #254)."""

from __future__ import annotations

from aquascope.dashboard.views.collect import _usgs_has_api_key, usgs_region_options

_NO_FILTER = "No filter (all US — slow)"


def test_no_filter_option_absent_without_key():
    options = usgs_region_options(has_api_key=False)
    # The filterless option is not offered, so the keyless USGS path can never be
    # reached with no filter and raise a bare ValueError.
    assert _NO_FILTER not in options
    # Every offered preset still carries a real filter (custom is the only None-ish
    # escape hatch and it prompts for a bbox).
    presets = {k: v for k, v in options.items() if k != "Custom bbox"}
    assert presets
    assert all(v for v in presets.values())
    assert options["Custom bbox"] == "__custom__"


def test_no_filter_option_present_with_key():
    options = usgs_region_options(has_api_key=True)
    assert _NO_FILTER in options
    assert options[_NO_FILTER] is None


def test_filtered_regions_are_unchanged_by_key_state():
    without = usgs_region_options(has_api_key=False)
    with_key = usgs_region_options(has_api_key=True)
    for label in ("Northeast US", "Southeast US", "Midwest US", "Pacific Northwest", "Southwest US"):
        assert without[label] == with_key[label]


def test_usgs_has_api_key_reads_env(monkeypatch):
    monkeypatch.delenv("USGS_API_KEY", raising=False)
    assert _usgs_has_api_key() is False
    monkeypatch.setenv("USGS_API_KEY", "DEMO_KEY")
    assert _usgs_has_api_key() is False
    monkeypatch.setenv("USGS_API_KEY", "real-key-123")
    assert _usgs_has_api_key() is True
