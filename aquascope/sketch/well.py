"""Theis drawdown for a pumping well, and the sum of several wells.

The Theis (1935) solution gives the drawdown around a fully penetrating well in a confined,
homogeneous, isotropic aquifer of infinite extent, pumped at a constant rate ``Q``::

    s(r, t) = Q / (4 * pi * T) * W(u),    u = r**2 * S / (4 * T * t)

where ``T`` is the transmissivity, ``S`` the storativity, ``r`` the distance from the well,
``t`` the time since pumping began and ``W(u) = E1(u)`` the well function (the exponential
integral). The drawdown is linear in ``Q``, so several wells add: the superposition principle
(:func:`superposition_drawdown`).

Units are fixed throughout the module, because the equation is only homogeneous when they agree:

======================  =========================================  ===========================
Quantity                Unit                                        Typical value or range
======================  =========================================  ===========================
``radius`` / drawdown   metre (m)                                   1 to 10,000 m
``time``                day (d)                                    0.01 to 1,000 d
``rate``                cubic metre per day (m3/day)                10 to 10,000 m3/day
``transmissivity``      square metre per day (m2/day)               1 to 10,000 m2/day
``storativity``         dimensionless (m3/m3)                       1e-5 to 1e-3 (confined)
======================  =========================================  ===========================

The defaults are the middle of those ranges, for a municipal well in a sandy confined aquifer:
:data:`DEFAULT_RATE`, :data:`DEFAULT_TRANSMISSIVITY` and :data:`DEFAULT_STORATIVITY`. Only the
confined aquifer is modelled; an unconfined aquifer needs the delayed yield of the water table
(Neuman, 1972), not Theis.

The equation assumes a fully penetrating well of infinitesimal radius, so the drawdown is
unbounded at ``r = 0`` (:func:`well_function` diverges there). This module refuses ``r = 0``
rather than returning an infinity that is not JSON-safe.

References
----------
Theis, C.V. (1935). The relation between the lowering of the piezometric surface and the rate and
duration of discharge of a well using ground-water storage. *Transactions, American Geophysical
Union*, 16, 519-524.

Fetter, C.W. (2001). *Applied Hydrogeology*, 4th edition. Prentice Hall, pp. 179-180.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from scipy.special import exp1

__all__ = [
    "DEFAULT_RATE",
    "DEFAULT_STORATIVITY",
    "DEFAULT_TRANSMISSIVITY",
    "superposition_drawdown",
    "theis_drawdown",
    "well_function",
]

#: Euler-Mascheroni constant, the leading term of ``W(u)`` as ``u`` goes to zero.
_EULER_GAMMA = 0.5772156649015329

#: Below this ``u``, ``-gamma - ln(u)`` is ``W(u)`` to better than one part in 1e15 and avoids
#: ``exp1`` evaluating a subnormal argument. Above it, ``scipy.special.exp1`` is used directly.
_U_ASYMPTOTIC = 1e-16

#: Default pumping rate, m3/day. The middle of the municipal-well range (10 to 10,000 m3/day).
DEFAULT_RATE = 1000.0

#: Default transmissivity, m2/day. The middle of the confined-sand range (1 to 10,000 m2/day).
DEFAULT_TRANSMISSIVITY = 500.0

#: Default storativity, dimensionless. The middle of the confined range (1e-5 to 1e-3).
DEFAULT_STORATIVITY = 1e-4

#: Keys a well mapping may have in :func:`superposition_drawdown`.
_WELL_KEYS = frozenset({"name", "rate", "radius", "start_time"})


def _finite(name: str, value: float) -> float:
    """Return ``value`` as a float, or raise if it is not a finite number."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _check_positive(name: str, value: float) -> float:
    """Return ``value`` as a finite float, or raise unless it is strictly positive."""
    number = _finite(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


def well_function(u: float) -> float:
    """The Theis well function ``W(u) = E1(u)``, the exponential integral of ``u``.

    Parameters
    ----------
    u:
        Dimensionless time-distance parameter, ``u = r**2 * S / (4 * T * t)``, greater than zero.
        ``u`` close to zero is the far field or a long time.

    Returns
    -------
    float
        ``W(u)``. For ``u < 1e-16`` the asymptotic form ``-gamma - ln(u)`` is used (relative
        error below ``1e-15``), so a subnormal ``u`` cannot lose the answer to underflow.

    Raises
    ------
    ValueError
        If ``u`` is negative, zero or not finite. ``W(0)`` diverges; the module returns no
        infinity, so that every result stays JSON-safe.
    """
    value = _finite("u", u)
    if value <= 0.0:
        raise ValueError(f"u must be > 0 (W(0) diverges), got {u!r}")
    if value < _U_ASYMPTOTIC:
        return -_EULER_GAMMA - math.log(value) + value
    return float(exp1(value))


def _drawdown(rate: float, transmissivity: float, storativity: float, radius: float, time: float) -> float:
    """Drawdown in metres for already validated, strictly positive parameters."""
    u = radius * radius * storativity / (4.0 * transmissivity * time)
    if u < _U_ASYMPTOTIC:
        if u > 0.0:
            w = -_EULER_GAMMA - math.log(u) + u
        else:
            # u underflowed to zero: rebuild ln(u) from the logs, so the answer stays finite.
            log_u = (2.0 * math.log(radius) + math.log(storativity)
                     - math.log(4.0) - math.log(transmissivity) - math.log(time))
            w = -_EULER_GAMMA - log_u
    else:
        w = float(exp1(u))
    drawdown = rate * w / (4.0 * math.pi * transmissivity)
    if not math.isfinite(drawdown):
        raise OverflowError(
            f"drawdown is not finite for rate={rate!r}, transmissivity={transmissivity!r}, "
            f"storativity={storativity!r}, radius={radius!r}, time={time!r}"
        )
    return drawdown


def theis_drawdown(
    radius: float,
    time: float,
    rate: float = DEFAULT_RATE,
    transmissivity: float = DEFAULT_TRANSMISSIVITY,
    storativity: float = DEFAULT_STORATIVITY,
) -> float:
    """Drawdown at a distance ``radius`` and time ``time`` from one pumping well (Theis, 1935).

    ``s = Q / (4 * pi * T) * W(u)`` with ``u = r**2 * S / (4 * T * t)``.

    Parameters
    ----------
    radius:
        Distance from the pumping well to the point, in metres, greater than zero.
    time:
        Time since pumping began, in days. Zero or negative means the well has not started and
        the drawdown is ``0.0``.
    rate:
        Constant pumping rate in m3/day; negative is injection. Defaults to
        :data:`DEFAULT_RATE` (1000 m3/day, the middle of the municipal-well range).
    transmissivity:
        Aquifer transmissivity in m2/day, greater than zero. Defaults to
        :data:`DEFAULT_TRANSMISSIVITY` (500 m2/day).
    storativity:
        Aquifer storativity, dimensionless, greater than zero. Defaults to
        :data:`DEFAULT_STORATIVITY` (1e-4).

    Returns
    -------
    float
        Drawdown in metres, positive for pumping. ``0.0`` before pumping starts.

    Raises
    ------
    ValueError
        If a parameter is not finite, or ``radius``, ``transmissivity`` or ``storativity`` is
        not strictly positive. ``radius = 0`` is refused because the Theis well function
        diverges at the well centre.
    OverflowError
        If the drawdown cannot be represented as a finite float (for example a vanishingly
        small radius or an astronomical rate).
    """
    t = _finite("time", time)
    q = _finite("rate", rate)
    r = _check_positive("radius", radius)
    aquifer_t = _check_positive("transmissivity", transmissivity)
    aquifer_s = _check_positive("storativity", storativity)
    if t <= 0.0:
        return 0.0
    if q == 0.0:
        return 0.0
    return _drawdown(q, aquifer_t, aquifer_s, r, t)


def superposition_drawdown(
    wells: Iterable[Mapping[str, Any]],
    time: float,
    *,
    transmissivity: float = DEFAULT_TRANSMISSIVITY,
    storativity: float = DEFAULT_STORATIVITY,
) -> dict[str, Any]:
    """Total drawdown of several pumping wells, added by superposition.

    Each well is a mapping with:

    ``radius``
        Distance from the well to the point, in metres (required, greater than zero).
    ``rate``
        Pumping rate in m3/day (default :data:`DEFAULT_RATE`; negative is injection).
    ``start_time``
        Time, on the same clock as ``time``, when the well started, in days (default ``0.0``).
        A well that has not started yet contributes exactly ``0.0``.
    ``name``
        Optional label carried into the result (default ``"well-1"``, ``"well-2"``, ...).

    Parameters
    ----------
    wells:
        The wells, each a mapping with the keys above (unknown keys are an error).
    time:
        The clock time at which the drawdown is wanted, in days. Usually the time since the
        first well started; each well's own elapsed time is ``time - start_time``.
    transmissivity, storativity:
        Aquifer properties, shared by every well, in m2/day and dimensionless.

    Returns
    -------
    dict
        JSON-safe, with lengths in metres and times in days::

            {"time_days": ..., "transmissivity_m2_per_day": ..., "storativity": ...,
             "total_drawdown_m": ..., "wells": [{"name": ..., "rate_m3_per_day": ...,
             "radius_m": ..., "start_time_days": ..., "elapsed_days": ...,
             "drawdown_m": ...}, ...]}

        ``wells`` follows the input order; ``total_drawdown_m`` is their exact sum
        (``math.fsum``), so swapping the input order cannot change it.

    Raises
    ------
    TypeError
        If a well is not a mapping.
    ValueError
        For an unknown well key, a missing ``radius``, a non-finite value, or the parameter
        errors of :func:`theis_drawdown` (raised for every well that has started).
    """
    t = _finite("time", time)
    aquifer_t = _check_positive("transmissivity", transmissivity)
    aquifer_s = _check_positive("storativity", storativity)

    rows: list[dict[str, Any]] = []
    for index, well in enumerate(wells, start=1):
        if not isinstance(well, Mapping):
            raise TypeError(f"well {index} must be a mapping, got {type(well).__name__}")
        unknown = set(well) - _WELL_KEYS
        if unknown:
            raise ValueError(f"well {index} has unknown keys: {sorted(unknown)}")
        if "radius" not in well:
            raise ValueError(f"well {index} is missing 'radius'")
        rate = _finite(f"well {index} rate", well.get("rate", DEFAULT_RATE))
        radius = _check_positive(f"well {index} radius", well["radius"])
        start = _finite(f"well {index} start_time", well.get("start_time", 0.0))
        elapsed = t - start
        if elapsed <= 0.0:
            drawdown = 0.0
        else:
            drawdown = _drawdown(rate, aquifer_t, aquifer_s, radius, elapsed)
        rows.append({
            "name": str(well.get("name", f"well-{index}")),
            "rate_m3_per_day": rate,
            "radius_m": radius,
            "start_time_days": start,
            "elapsed_days": elapsed,
            "drawdown_m": drawdown,
        })

    total = math.fsum(row["drawdown_m"] for row in rows)
    if not math.isfinite(total):
        raise OverflowError("total drawdown is not finite")
    return {
        "time_days": t,
        "transmissivity_m2_per_day": aquifer_t,
        "storativity": aquifer_s,
        "total_drawdown_m": total,
        "wells": rows,
    }
