"""The Explorer's signature filter is wired end to end: rail container, boot call, map hook, Ask and WebMCP."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPLORER = ROOT / "explorer"


def _read(rel: str) -> str:
    return (EXPLORER / rel).read_text(encoding="utf-8")


def test_the_rail_has_a_hidden_filter_container():
    html = _read("index.html")
    assert '<details id="sigfilter" class="rail-group" open hidden>' in html, "hidden until the table is found"
    assert 'class="rail-list sigfilter-body"' in html


def test_boot_starts_the_filter_and_the_map_honours_it():
    assert "initSignatureFilter();" in _read("app.js")
    assert "state.sigMatch.has(stationKey(r))" in _read("src/catalog.js")
    assert "state.sigMatch" in _read("src/rail.js"), "the rail count agrees with the map"


def test_ask_and_webmcp_can_set_the_filter():
    assert "actions.applyAskFilter?.(res)" in _read("src/ask.js")
    webmcp = _read("src/webmcp.js")
    assert "aquascope_filter_gauges" in webmcp and "actions.setSignatureFilterFromQuestion" in webmcp
    module = _read("src/signature-filter.js")
    hooks = ("actions.setSignatureFilter =", "actions.setSignatureFilterFromQuestion =", "actions.applyAskFilter =")
    for name in hooks:
        assert name in module
    assert 'name: "filter_gauges"' in module and "spec_only: true" in module, "words are parsed by the Python rules"


def test_the_filter_reads_the_file_python_writes():
    from aquascope.archive.signatures import COLUMNS, SIGNATURES_FILE

    core = _read("src/signature-filter-core.js")
    assert SIGNATURES_FILE in core
    for col in ("data_years", "amax_trend", "bfi"):
        assert col in COLUMNS and col in core


def test_the_python_words_match_the_page():
    """describe_filter (Python) and describeFilter (JS) print the same text; the node suite checks the JS side."""
    from aquascope.archive.signatures import describe_filter

    assert describe_filter({"min_years": 50, "flood_trend": "rising"}) == "50+ years of data, rising flood trend"
    assert describe_filter({"bfi_min": 0.3, "bfi_max": 0.5}) == "BFI 0.3 to 0.5"
    assert describe_filter({"bfi_max": 0.3}) == "BFI up to 0.3"
    assert describe_filter({}) == "no filter"
