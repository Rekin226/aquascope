"""The sketchpad's well (experimental): Theis drawdown for one well, or several added up.

This is the first computation behind the sketchpad's "drop a well and see the drawdown": the
steady question "what happens to the water table if I pump here?" becomes a transient one,
answered at a distance and a time. :func:`theis_drawdown` is the Theis (1935) solution for one
well in a confined aquifer; :func:`superposition_drawdown` adds several wells, each with its own
rate, distance and start time, by the superposition principle; :func:`well_function` exposes the
well function ``W(u) = E1(u)``. Lengths are metres, times days, rates m3/day and transmissivity
m2/day, and every result is a plain float or a JSON-safe dict.

Experimental: shipped and tested, but not covered by the 1.0 promise, so names and arguments may
change in a minor release (see ``docs/stability.md``). The Explorer's "Drop a well" UI is a
follow-up; this module is the package part of #660.
"""

from aquascope.sketch.well import (
    DEFAULT_RATE,
    DEFAULT_STORATIVITY,
    DEFAULT_TRANSMISSIVITY,
    superposition_drawdown,
    theis_drawdown,
    well_function,
)

__all__ = [
    "DEFAULT_RATE",
    "DEFAULT_STORATIVITY",
    "DEFAULT_TRANSMISSIVITY",
    "superposition_drawdown",
    "theis_drawdown",
    "well_function",
]
