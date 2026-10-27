"""Tests for the Theis well function and superposition (#660).

The textbook checks use two published worked examples:

* Fetter, C.W. (2001), *Applied Hydrogeology*, 4th edition, Prentice Hall, p. 180. Q = 2725 m3/day,
  T = 299 m2/day, S = 0.0051, r = 7.0 m, t = 1 day; the book prints u = 2.1e-4, W(u) = 7.94 and a
  drawdown of 5.7 m. The same inputs evaluated exactly give 5.7269 m, which is the three-significant-
  figure value pinned below.
* The worked "Example - Theis Equation" in the University of Houston GEOS 2J notes *Ground-Water
  Flow to Wells* (Fall 2019), slide 34: Q = 1500 m3/day, T = 600 m2/day, S = 4e-4, r = 1 km, t = 1
  year; the slide prints u = 4.6e-4, W(u) = 7.12 and a drawdown of 1.42 m.
"""

from __future__ import annotations

import json
import math

import pytest

from aquascope.sketch import (
    DEFAULT_RATE,
    DEFAULT_STORATIVITY,
    DEFAULT_TRANSMISSIVITY,
    superposition_drawdown,
    theis_drawdown,
    well_function,
)

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Textbook examples
# ---------------------------------------------------------------------------

def test_fetter_2001_theis_example():
    """Fetter (2001), p. 180: 5.7 m at 7 m after one day of pumping."""
    drawdown = theis_drawdown(
        radius=7.0, time=1.0, rate=2725.0, transmissivity=299.0, storativity=0.0051,
    )
    assert drawdown == pytest.approx(5.7, abs=0.05)  # the book prints 5.7 m

    u = 7.0**2 * 0.0051 / (4.0 * 299.0 * 1.0)
    assert u == pytest.approx(2.1e-4, rel=1e-2)  # the book prints u = 2.1e-4, rounded
    assert well_function(u) == pytest.approx(7.94, rel=6e-3)  # the book prints W(u) = 7.94


def test_fetter_2001_example_at_three_significant_figures():
    """The same Fetter inputs, evaluated without the book's table round-off: 5.73 m."""
    drawdown = theis_drawdown(
        radius=7.0, time=1.0, rate=2725.0, transmissivity=299.0, storativity=0.0051,
    )
    assert drawdown == pytest.approx(5.73, rel=1e-3)
    assert round(drawdown, 2) == 5.73


def test_worked_example_three_significant_figures():
    """GEOS 2J (2019), slide 34: 1.42 m at 1 km after one year."""
    drawdown = theis_drawdown(
        radius=1000.0, time=365.0, rate=1500.0, transmissivity=600.0, storativity=4e-4,
    )
    assert drawdown == pytest.approx(1.42, abs=0.005)  # the slide prints 1.42 m


def test_well_function_matches_the_published_table():
    """W(2e-4) = 7.94 in Fetter's appendix table of W(u), to three significant figures."""
    assert well_function(2.0e-4) == pytest.approx(7.94, rel=1e-4)


# ---------------------------------------------------------------------------
# One engine: the shared Theis in aquascope.groundwater.aquifer
# ---------------------------------------------------------------------------

def test_the_same_metres_as_the_aquifer_engine():
    """The sketch calls the one Theis engine, so its answers are that engine's answers."""
    from aquascope.groundwater.aquifer import theis_drawdown as aquifer_theis_drawdown

    cases = [
        (7.0, 1.0, 2725.0, 299.0, 0.0051),  # Fetter (2001), p. 180
        (1000.0, 365.0, 1500.0, 600.0, 4e-4),  # GEOS 2J (2019), slide 34
        (250.0, 0.5, 800.0, 750.0, 5e-5),
    ]
    for radius, time, rate, transmissivity, storativity in cases:
        sketch_s = theis_drawdown(radius, time, rate, transmissivity, storativity)
        # The engine's signature is (T, S, Q, r, t).
        engine_s = float(aquifer_theis_drawdown(transmissivity, storativity, rate, radius, time)[0])
        assert sketch_s == pytest.approx(engine_s, rel=1e-12)


def test_an_injection_well_is_the_pumping_answer_negated():
    """The engine refuses a rate that is not positive; the sketch negates its magnitude's answer."""
    assert theis_drawdown(100.0, 1.0, -1000.0) == -theis_drawdown(100.0, 1.0, 1000.0)
    assert theis_drawdown(100.0, 1.0, -1000.0) < 0.0


# ---------------------------------------------------------------------------
# Superposition
# ---------------------------------------------------------------------------

def test_swapping_two_equal_wells_changes_nothing():
    wells = [
        {"name": "a", "rate": 800.0, "radius": 100.0, "start_time": 0.0},
        {"name": "b", "rate": 800.0, "radius": 100.0, "start_time": 0.0},
    ]
    forward = superposition_drawdown(wells, 2.0, transmissivity=500.0, storativity=1e-4)
    backward = superposition_drawdown(list(reversed(wells)), 2.0, transmissivity=500.0, storativity=1e-4)
    assert forward["total_drawdown_m"] == backward["total_drawdown_m"]
    assert [w["drawdown_m"] for w in forward["wells"]] == [w["drawdown_m"] for w in backward["wells"]]


def test_superposition_is_order_independent_for_different_wells():
    a = {"rate": 1000.0, "radius": 50.0, "start_time": 0.0}
    b = {"rate": 500.0, "radius": 400.0, "start_time": 0.75}
    forward = superposition_drawdown([a, b], 3.0, transmissivity=800.0, storativity=2e-4)
    backward = superposition_drawdown([b, a], 3.0, transmissivity=800.0, storativity=2e-4)
    assert forward["total_drawdown_m"] == backward["total_drawdown_m"]
    assert forward["total_drawdown_m"] == pytest.approx(
        theis_drawdown(50.0, 3.0, 1000.0, 800.0, 2e-4) + theis_drawdown(400.0, 2.25, 500.0, 800.0, 2e-4),
        rel=1e-12,
    )


def test_a_zero_rate_well_adds_nothing():
    pump = {"rate": 1000.0, "radius": 100.0}
    with_zero = [
        pump,
        {"rate": 0.0, "radius": 100.0},
    ]
    one = superposition_drawdown([pump], 1.0, transmissivity=500.0, storativity=1e-4)
    two = superposition_drawdown(with_zero, 1.0, transmissivity=500.0, storativity=1e-4)
    assert two["total_drawdown_m"] == one["total_drawdown_m"]
    assert two["wells"][1]["drawdown_m"] == 0.0


def test_a_well_that_has_not_started_contributes_zero():
    result = superposition_drawdown(
        [{"rate": 1000.0, "radius": 100.0, "start_time": 5.0}],
        time=4.0,
        transmissivity=500.0,
        storativity=1e-4,
    )
    assert result["total_drawdown_m"] == 0.0
    assert result["wells"][0]["drawdown_m"] == 0.0
    # A well that has not started has no age yet: elapsed_days is clamped at 0, not -1.
    assert result["wells"][0]["elapsed_days"] == 0.0


def test_each_well_uses_its_own_elapsed_time():
    result = superposition_drawdown(
        [
            {"rate": 1000.0, "radius": 100.0, "start_time": 0.0},
            {"rate": 1000.0, "radius": 100.0, "start_time": 2.0},
        ],
        time=3.0,
        transmissivity=500.0,
        storativity=1e-4,
    )
    assert result["wells"][0]["elapsed_days"] == 3.0
    assert result["wells"][1]["elapsed_days"] == 1.0
    assert result["total_drawdown_m"] == pytest.approx(
        theis_drawdown(100.0, 3.0, 1000.0, 500.0, 1e-4) + theis_drawdown(100.0, 1.0, 1000.0, 500.0, 1e-4),
        rel=1e-12,
    )


def test_total_is_the_exact_sum_of_the_contributions():
    result = superposition_drawdown(
        [
            {"rate": 900.0, "radius": 30.0},
            {"rate": 400.0, "radius": 250.0, "start_time": 0.25},
            {"rate": 50.0, "radius": 1000.0, "start_time": 1.0},
        ],
        time=2.0,
        transmissivity=750.0,
        storativity=5e-5,
    )
    contributions = [well["drawdown_m"] for well in result["wells"]]
    assert result["total_drawdown_m"] == math.fsum(contributions)
    assert result["total_drawdown_m"] == pytest.approx(sum(contributions), rel=1e-15)


def test_result_is_json_safe():
    result = superposition_drawdown(
        [{"rate": 1000.0, "radius": 100.0, "start_time": 0.0, "name": "pump"}],
        time=1.0,
        transmissivity=500.0,
        storativity=1e-4,
    )
    assert json.loads(json.dumps(result, allow_nan=False))["total_drawdown_m"] == result["total_drawdown_m"]


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("time", [0.0, -1.0, -1e-9])
def test_no_drawdown_before_pumping_starts(time):
    assert theis_drawdown(100.0, time, 1000.0, 500.0, 1e-4) == 0.0


def test_radius_zero_is_refused_because_the_well_function_diverges():
    with pytest.raises(ValueError, match="radius"):
        theis_drawdown(0.0, 1.0, 1000.0, 500.0, 1e-4)


def test_well_function_at_zero_is_refused():
    with pytest.raises(ValueError, match="diverges"):
        well_function(0.0)


@pytest.mark.parametrize("kwargs", [
    {"transmissivity": 0.0},
    {"transmissivity": -1.0},
    {"storativity": 0.0},
    {"radius": -5.0},
    {"rate": float("nan")},
    {"time": float("inf")},
])
def test_invalid_parameters_are_refused(kwargs):
    params = {"radius": 10.0, "time": 1.0, "rate": 100.0, "transmissivity": 500.0, "storativity": 1e-4}
    params.update(kwargs)
    with pytest.raises(ValueError):
        theis_drawdown(**params)


def test_very_small_u_stays_finite_and_uses_the_asymptotic_form():
    # u = 1e-4 / (4 * 1 * 1e300) = 2.5e-305, far below the 1e-16 cut-off where exp1 underflows.
    drawdown = theis_drawdown(1.0, 1e300, 1.0, 1.0, 1e-4)
    expected_w = -EULER_GAMMA - math.log(2.5e-305)
    assert math.isfinite(drawdown)
    assert drawdown == pytest.approx(expected_w / (4.0 * math.pi * 1.0), rel=1e-12)


def test_time_so_long_that_four_t_times_t_overflows_still_finishes():
    # 4 * T * t = 4e314 overflows to inf, so u would be 0.0 and exp1(0) infinite; ln(u) is rebuilt
    # from logs instead.
    drawdown = theis_drawdown(1.0, 1e308, 1.0, 1e6, 1e-4)
    log_u = math.log(1e-4) - math.log(4.0) - math.log(1e6) - math.log(1e308)
    assert math.isfinite(drawdown)
    assert drawdown == pytest.approx((-EULER_GAMMA - log_u) / (4.0 * math.pi * 1e6), rel=1e-12)


def test_u_that_underflows_to_zero_still_gives_a_finite_drawdown():
    # r**2 and S both underflow, so the straightforward u is 0.0 and exp1(0) would be infinite.
    drawdown = theis_drawdown(1e-200, 1.0, 1.0, 1.0, 1e-200)
    assert math.isfinite(drawdown)
    log_u = 2.0 * math.log(1e-200) + math.log(1e-200) - math.log(4.0)
    assert drawdown == pytest.approx((-EULER_GAMMA - log_u) / (4.0 * math.pi), rel=1e-12)


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_defaults_are_the_stated_typical_values():
    assert DEFAULT_RATE == 1000.0
    assert DEFAULT_TRANSMISSIVITY == 500.0
    assert DEFAULT_STORATIVITY == 1e-4
    assert theis_drawdown(100.0, 1.0) > 0.0


def test_the_module_is_experimental():
    from aquascope import experimental

    assert experimental.is_experimental("aquascope.sketch")
    assert experimental.is_experimental("aquascope.sketch.well")


def test_public_surface():
    import aquascope.sketch as sketch

    assert set(sketch.__all__) == {
        "DEFAULT_RATE",
        "DEFAULT_STORATIVITY",
        "DEFAULT_TRANSMISSIVITY",
        "superposition_drawdown",
        "theis_drawdown",
        "well_function",
    }
