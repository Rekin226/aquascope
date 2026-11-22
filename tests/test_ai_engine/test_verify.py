"""The river check: a record has to be on the river the question names (#731).

``record_is_named`` only asks whether a record was named. The Kingston recording
answered a question about the Thames with the Hogsmill, a tributary whose
100-year flood is fifty times smaller, and passed every check it had.
"""

from __future__ import annotations

from typing import Any

from aquascope.ai_engine import verify as verify_mod

_THAMES = "8496ce69-482c-406a-a2f0-ac418ef8f099"
_HOGSMILL = "63b6ae06-60fc-4e92-8337-aa0e5fdd7e79"
_QUESTION = "What is the 100-year flood of the Thames at Kingston, and how sure can we be?"

_STATIONS = {
    "stations": [
        {"station_id": _HOGSMILL, "name": "Kingston Hogsmill", "river": "Hogsmill", "distance_km": 0.3},
        {"station_id": _THAMES, "name": "Kingston", "river": "River Thames", "distance_km": 0.7},
    ]
}


def _result(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "arguments": {}, "payload": payload, "ok": True}


def _ffa(station_id: str, value: float) -> dict[str, Any]:
    return {
        "station_id": station_id,
        "return_levels": [{"return_period": 100, "value": value, "lower": value * 0.9, "upper": value * 1.1}],
    }


def _river(checks) -> Any:
    return next((c for c in checks if c.name == "river_matches_the_question"), None)


def test_a_record_on_another_river_is_flagged() -> None:
    answer = "The 100-year flood at Kingston Hogsmill is 13.31 m3/s (11.68 to 15.17)."
    results = [_result("assess_site", _STATIONS), _result("flood_frequency", _ffa(_HOGSMILL, 13.31))]
    v = verify_mod.verify(answer, results, question=_QUESTION)
    check = _river(v.checks)
    assert check is not None and not check.passed
    assert "Hogsmill" in check.detail and "River Thames" in check.detail


def test_a_record_on_the_named_river_passes() -> None:
    answer = "The 100-year flood of the Thames at Kingston is 652.5 m3/s (573.2 to 679.4)."
    v = verify_mod.verify(answer, [_result("assess_site", _STATIONS), _result("flood_frequency", _ffa(_THAMES, 652.5))],
                          question=_QUESTION)
    check = _river(v.checks)
    assert check is not None and check.passed


def test_a_question_that_names_no_river_adds_no_check() -> None:
    answer = "The 100-year flood at this gauge is 13.31 m3/s (11.68 to 15.17)."
    results = [_result("assess_site", _STATIONS), _result("flood_frequency", _ffa(_HOGSMILL, 13.31))]
    v = verify_mod.verify(answer, results, question="What is the 100-year flood at this gauge?")
    assert _river(v.checks) is None


def test_results_without_a_river_skip_the_check() -> None:
    """An older payload has no river key; the check must not invent one."""
    answer = "The 100-year flood is 13.31 m3/s (11.68 to 15.17)."
    v = verify_mod.verify(answer, [_result("flood_frequency", _ffa(_HOGSMILL, 13.31))], question=_QUESTION)
    assert _river(v.checks) is None


def test_a_tributary_quoted_while_the_mainstem_was_queried_is_flagged() -> None:
    """#649 called flood_frequency on the Thames, then wrote the Hogsmill anyway."""
    answer = "The 100-year flood at Kingston Hogsmill is 13.31 m3/s (11.68 to 15.17)."
    results = [_result("assess_site", _STATIONS), _result("flood_frequency", _ffa(_THAMES, 652.5))]
    check = _river(verify_mod.verify(answer, results, question=_QUESTION).checks)
    assert check is not None and not check.passed
