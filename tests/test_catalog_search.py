"""Catalog name search: several words match across name and river, accent-folded (#306 live run).

``find_stations(query="Kingston Thames")`` used to return nothing: the whole
string was one substring tested against the name alone.
"""

from __future__ import annotations

from aquascope.archive.catalog import fold_text, query_tokens, search_stations


def _row(source, sid, name, lat, lon, river=None):
    return {"source": source, "station_id": sid, "name": name, "latitude": lat, "longitude": lon,
            "variables": ["discharge"], "river": river}


ROWS = [
    _row("uk_ea", "8496ce69", "Kingston", 51.41, -0.31, river="River Thames"),
    _row("uk_ea", "hull-1", "Kingston upon Hull", 53.74, -0.33, river="River Hull"),
    _row("uk_ea", "reading", "Reading", 51.46, -0.97, river="River Thames"),
    _row("hubeau_hydrometrie", "V1", "Le Rhône à Anthon", 45.79, 5.16, river="Le Rhône"),
    _row("usgs", "USGS-01646500", "Potomac River near Wash, DC", 38.95, -77.13),
]


def test_fold_text_strips_accents_and_case():
    assert fold_text("Le Rhône à Anthon") == "le rhone a anthon"
    assert fold_text("MÜNCHEN") == "munchen"
    assert fold_text(None) == ""


def test_query_tokens_drop_connectives_only_when_words_remain():
    assert query_tokens("Thames at Kingston") == ["thames", "kingston"]
    assert query_tokens("River") == ["river"]
    assert query_tokens("  ") == []


def test_every_word_across_name_and_river_wins():
    hits = search_stations(ROWS, query="Kingston Thames")
    assert [r["station_id"] for r in hits][:1] == ["8496ce69"]
    # rows matching one word follow, name matches ahead of river-only ones
    assert [r["station_id"] for r in hits] == ["8496ce69", "hull-1", "reading"]
    assert [r["station_id"] for r in search_stations(ROWS, query="Thames at Kingston")][0] == "8496ce69"


def test_no_full_match_falls_back_to_the_best_partial_matches():
    hits = search_stations(ROWS, query="Kingston Bogus")
    assert [r["station_id"] for r in hits] == ["8496ce69", "hull-1"]
    assert search_stations(ROWS, query="Bogus Words") == []


def test_single_word_keeps_the_old_name_or_id_substring_behaviour():
    assert {r["station_id"] for r in search_stations(ROWS, query="kingston")} == {"8496ce69", "hull-1"}
    assert [r["station_id"] for r in search_stations(ROWS, query="01646500")] == ["USGS-01646500"]
    # one word does not search the river field, as before
    assert search_stations(ROWS, query="thames") == []


def test_accents_do_not_matter():
    assert [r["station_id"] for r in search_stations(ROWS, query="rhone")] == ["V1"]
    assert [r["station_id"] for r in search_stations(ROWS, query="Rhône Anthon")] == ["V1"]


def test_near_breaks_ties_among_equal_matches_and_limit_still_applies():
    hits = search_stations(ROWS, query="Kingston river", near=(53.7, -0.3))
    assert [r["station_id"] for r in hits][:2] == ["hull-1", "8496ce69"]
    assert len(search_stations(ROWS, query="Kingston Thames", limit=1)) == 1
