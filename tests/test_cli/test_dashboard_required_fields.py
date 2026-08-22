"""The Collect page must not submit a form the collector will reject (#174).

The forms only write a fetch kwarg when the user typed something, so a blank
required field used to reach the collector as a missing argument and surface as
a raw ``ValueError`` -- for PEGELONLINE, ``station_id must not be empty.``
These tests pin the guard that stops that, and the guard's own coverage.
"""

from __future__ import annotations

import inspect

import pytest

from aquascope.dashboard.views.collect import (
    _REQUIRED_FETCH_FIELDS,
    _REQUIRED_ONE_OF_FETCH_FIELDS,
    SOURCES,
    missing_required_fields,
)


def test_pegelonline_without_a_station_is_caught_before_the_collector():
    """The reported case: Collect pressed with the UUID box empty."""
    assert missing_required_fields("pegelonline", {"days": 7}) == ["Station UUID"]


def test_pegelonline_with_a_station_passes():
    """Control. A guard that rejected everything would pass the test above."""
    assert missing_required_fields("pegelonline", {"station_id": "d3301a25-2401-44cd-9f79-aa66c61f22e0"}) == []


def test_a_blank_string_counts_as_missing():
    """The forms strip before writing, but the guard should not depend on that."""
    assert missing_required_fields("pegelonline", {"station_id": ""}) == ["Station UUID"]
    assert missing_required_fields("pegelonline", {"station_id": "   "}) == []  # pre-stripped by the form


def test_bom_without_a_station_is_caught():
    """BOMCollector.fetch_raw raises the same way; same form shape, same fix."""
    assert missing_required_fields("bom", {"parameter_type": "Water Course Discharge"}) == [
        "AWRC station number"
    ]


@pytest.mark.parametrize(
    "fetch, expected",
    [
        ({}, ["Station LID or Bounding box"]),
        ({"lid": "ANAW1"}, []),
        ({"bbox": (-80.0, 37.0, -66.0, 48.0)}, []),
        ({"bbox": ()}, ["Station LID or Bounding box"]),
    ],
)
def test_noaa_nwps_needs_one_of_lid_or_bbox(fetch, expected):
    """NOAA-NWPS raises "One of 'lid' or 'bbox' is required." -- an any-of rule,
    not an all-of one, so clearing the LID box with no bbox typed is invalid
    while either alone is fine."""
    assert missing_required_fields("noaa_nwps", fetch) == expected


def test_a_source_with_no_required_fields_never_blocks():
    """Most sources have no required field and must stay one click away."""
    assert missing_required_fields("ireland_opw", {"max_stations": 10}) == []
    assert missing_required_fields("not-a-real-source", {}) == []


def test_every_guarded_source_is_a_real_source():
    """A typo'd key would silently guard nothing."""
    guarded = set(_REQUIRED_FETCH_FIELDS) | set(_REQUIRED_ONE_OF_FETCH_FIELDS)
    assert guarded <= set(SOURCES), f"unknown source keys: {sorted(guarded - set(SOURCES))}"


def test_every_guarded_field_is_a_real_fetch_argument():
    """Drift guard. A field renamed in the collector must not leave the table
    guarding a kwarg that no longer exists -- that would pass silently and let
    the traceback back in."""
    from aquascope.registry import build_collector

    # Every guarded source is keyless, so building one is offline and cheap --
    # it is the same call the Collect button makes, minus the fetch.
    def parameters(source_key: str) -> set[str]:
        collector = build_collector(source_key)
        return set(inspect.signature(collector.fetch_raw).parameters)

    for source_key, fields in _REQUIRED_FETCH_FIELDS.items():
        available = parameters(source_key)
        for name in fields:
            assert name in available, (
                f"{source_key}: _REQUIRED_FETCH_FIELDS names {name!r}, "
                f"which is not a parameter of its fetch_raw ({sorted(available)})"
            )

    for source_key, groups in _REQUIRED_ONE_OF_FETCH_FIELDS.items():
        available = parameters(source_key)
        for group in groups:
            for name in group:
                assert name in available, (
                    f"{source_key}: _REQUIRED_ONE_OF_FETCH_FIELDS names {name!r}, "
                    f"which is not a parameter of its fetch_raw ({sorted(available)})"
                )
