"""The Budyko framework — catchment water-balance partitioning.

The framework is a model mapping out the entire
"space" between a water limit and an energy limit.  Given long-term mean
annual precipitation ``P`` and potential evapotranspiration ``PET``, it
assumes the evaporative ratio ``ET / P`` is a function only of the aridity
index ``phi = PET / P`` (Budyko 1974).  Each concrete curve (Schreiber,
Ol'dekop, Turc-Pike, Fu/Zhang) is one shape inside those limits; the result
carries both the prediction at a catchment's aridity and the whole curve
family, so a single :class:`BudykoResult` is both the answer and the
diagram.

The long-term water balance ``P = ET + Q`` (storage change neglected) means
a catchment can be positioned from either of its two observable fluxes: the
actual evapotranspiration ``ET``, or the runoff ``Q`` with
``ET = P - Q``.  ``observed_et`` and ``observed_runoff`` are mutually
exclusive; exactly one places the catchment relative to the curve.

See the theory guide (``docs/theory.md`` §12) for the derivations and
references.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

# Framework constants
BUDYKO_CURVES = ("schreiber", "oldekop", "turc_pike", "fu_zhang")
# Fixed canonical domain the curve family is evaluated over (0.02–4.0).
_DEFAULT_ARIDITY_GRID = np.linspace(0.02, 4.0, 300)


def _grid_covers(grid: np.ndarray, aridity: np.ndarray) -> bool:
    """Whether *grid* spans every element of *aridity*."""
    return bool(np.all(aridity >= grid.min()) and np.all(aridity <= grid.max()))


# Result dataclass
@dataclass
class BudykoResult:
    """Predicted evaporative ratios under the Budyko framework.

    Attributes
    ----------
    aridity_index : float | numpy.ndarray
        Aridity index (PET / P) at the query point(s): a float for scalar
        inputs, an array of the input shape otherwise.
    predicted : dict[str, float | numpy.ndarray]
        Predicted evaporative ratio (ET / P) per requested curve, evaluated
        at ``aridity_index``.
    observed_evaporative_ratio : float | numpy.ndarray, optional
        Observed evaporative ratio (ET / P) when *observed_et* or
        *observed_runoff* was given.
    observed_deviation : dict[str, float | numpy.ndarray], optional
        Observed ratio minus predicted ratio per curve; positive means the
        catchment sits above the curve (more evaporative than predicted).
    aridity_grid : numpy.ndarray
        The aridity domain the curves were evaluated over: the canonical
        ``[0.02, 4.0]`` domain, extended when an observed aridity index lies
        outside it so the family spans the plotted points.
    curves : dict[str, numpy.ndarray]
        Predicted evaporative ratio per requested curve over ``aridity_grid``,
        ready for plotting.
    """

    aridity_index: float | np.ndarray
    predicted: dict[str, float | np.ndarray]
    observed_evaporative_ratio: float | np.ndarray | None = None
    observed_deviation: dict[str, float | np.ndarray] | None = None
    aridity_grid: np.ndarray = field(default_factory=lambda: _DEFAULT_ARIDITY_GRID.copy())
    curves: dict[str, np.ndarray] = field(default_factory=dict)


# Curve helpers
def _to_scalar_if_zero_dim(value: np.ndarray) -> float | np.ndarray:
    """Return ``float`` for a zero-dimensional array, the array otherwise."""
    return float(value) if value.ndim == 0 else value


def _observed_input(name: str, value: float | np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cast an observed ET/runoff input to a broadcastable float array."""
    arr = np.asarray(value, dtype=float)
    try:
        arr = cast(np.ndarray, np.broadcast_arrays(arr, ref)[0])
    except ValueError as exc:
        raise ValueError(f"{name} must be broadcast-compatible with precipitation.") from exc
    return arr


def _budyko_curve(aridity: np.ndarray, curve_name: str, omega: float) -> np.ndarray:
    """Evaporative ratio (ET / P) for one Budyko curve over an aridity array.

    ``aridity`` is the aridity index ``phi = PET / P``. Every curve shares
    the same limits: ``(ET/P) / phi -> 1`` (equivalently ``ET/P -> phi`` and
    ``E -> PET``) as ``phi -> 0``, the energy limit; and ``ET/P -> 1``
    (``E -> P``) as ``phi -> inf``, the water limit.  The Turc-Pike and
    Fu/Zhang evaluations use numerically stable large-``phi`` forms so the
    curves stay 1 (not ``inf``/``nan``) even when ``phi**2`` or ``phi**omega``
    would overflow.
    """
    if curve_name == "schreiber":
        values: np.ndarray = 1.0 - np.exp(-aridity)
    elif curve_name == "oldekop":
        values = aridity * np.tanh(1.0 / aridity)
    elif curve_name == "turc_pike":
        values = aridity / np.hypot(1.0, aridity)
    elif curve_name == "fu_zhang":
        # For phi <= 1 the direct form is safe; for phi > 1 rewrite
        # (1 + phi**omega)**(1/omega) = phi * (1 + phi**-omega)**(1/omega)
        # so the power never overflows and the curve still melts into 1.
        small = np.minimum(aridity, 1.0)
        f_small = 1.0 + small - (1.0 + small**omega) ** (1.0 / omega)
        big = np.maximum(aridity, 1.0)
        correction = big * np.expm1(np.log1p(big ** (-omega)) / omega)
        f_big = 1.0 - correction
        values = np.where(aridity <= 1.0, f_small, f_big)
    else:
        raise ValueError(f"Unknown Budyko curve {curve_name!r}. Choose from {BUDYKO_CURVES}.")
    return values


def budyko(
    precipitation: float | np.ndarray,
    pet: float | np.ndarray,
    curves: Sequence[str] = BUDYKO_CURVES,
    fu_omega: float = 2.0,
    observed_et: float | np.ndarray | None = None,
    observed_runoff: float | np.ndarray | None = None,
) -> BudykoResult:
    """Partition long-term precipitation under the Budyko framework.

    Given long-term mean annual precipitation ``P`` and potential
    evapotranspiration ``PET``, the Budyko framework (Budyko 1974) assumes the
    evaporative ratio ``ET/P`` is a function only of the aridity index
    ``phi = PET / P``, bounded by an energy limit (``(ET/P) / phi -> 1`` as
    ``phi -> 0``, i.e. ``E -> PET``) and a water limit (``ET/P -> 1`` as
    ``phi -> inf``, i.e. ``E -> P``).
    Each curve is a concrete shape between those limits:

    - ``schreiber``  : ``1 - exp(-phi)``        (Schreiber 1904)
    - ``oldekop``    : ``phi * tanh(1 / phi)``  (Ol'dekop 1911)
    - ``turc_pike``  : ``phi / sqrt(1 + phi**2)``  (Turc 1954, Pike 1964)
    - ``fu_zhang``   : ``1 + phi - (1 + phi**omega) ** (1 / omega)``  (Fu 1981, Zhang et al. 2004)

    The result carries the prediction at the query point **and** the full
    curve family over the canonical ``aridity_grid`` (``[0.02, 4.0]``,
    extended to cover any observed aridity index outside it), so a single
    :class:`BudykoResult` can both report a catchment's position and
    render the diagram it lives in (:func:`aquascope.viz.plot_budyko`).

    To locate a real catchment, the long-term water balance
    ``P = ET + Q`` (storage change neglected) gives the observed evaporative
    ratio either directly from actual evapotranspiration,
    ``ET/P``, or from runoff, ``1 - Q/P``.  Pass *observed_et* **or**
    *observed_runoff* — not both — and the result reports the catchment's
    position relative to each curve.

    The framework assumes a closed long-term water balance at steady state.
    Where storage change is significant (multi-year snow or groundwater
    carry-over, glaciers, lakes or reservoirs) or fluxes cross the catchment
    boundary (irrigation imports, interbasin transfers, deep groundwater
    exchange), these assumptions are violated and deviations from a curve
    cannot be attributed to catchment properties alone.

    Parameters
    ----------
    precipitation : float or numpy.ndarray
        Long-term mean annual precipitation (mm/yr), scalar or broadcastable
        array with *pet*.
    pet : float or numpy.ndarray
        Long-term mean annual potential evapotranspiration (mm/yr).
    curves : sequence of str
        Budyko curves to evaluate.  Any subset of ``"schreiber"``,
        ``"oldekop"``, ``"turc_pike"``, ``"fu_zhang"``.  The result's
        ``predicted`` and ``curves`` members only hold the requested names.
    fu_omega : float
        The Fu/Zhang shape parameter ``omega``, ``>= 1``.  Larger values push
        the curve toward the perfect-limit ``min(phi, 1)`` shape.
    observed_et : float or numpy.ndarray, optional
        Observed long-term actual evapotranspiration (mm/yr), requiring
        ``0 <= observed_et <= precipitation``.  Mutually exclusive with
        *observed_runoff*.
    observed_runoff : float or numpy.ndarray, optional
        Observed long-term runoff (mm/yr), requiring
        ``0 <= observed_runoff <= precipitation``.  Converted to
        evapotranspiration via ``ET = P - Q``.  Mutually exclusive with
        *observed_et*.

    Returns
    -------
    BudykoResult
        Predicted evaporative ratios per curve, the observed position when
        *observed_et* or *observed_runoff* was given, and the curve family
        over ``aridity_grid``.

    Raises
    ------
    ValueError
        If *curves* is empty or names an unknown curve, *precipitation*,
        *pet*, *observed_et* or *observed_runoff* are non-finite,
        *precipitation* or *pet* are non-positive, ``fu_omega < 1``, the
        inputs are not broadcastable, an *observed_et* / *observed_runoff*
        outside ``[0, precipitation]`` is supplied, or both observed fluxes
        are given.
    """
    if isinstance(curves, str):
        raise ValueError("curves must be a sequence of curve names, not a single string.")
    if not curves:
        raise ValueError("Choose at least one Budyko curve.")
    unknown = [c for c in curves if c not in BUDYKO_CURVES]
    if unknown:
        raise ValueError(f"Unknown Budyko curve {unknown[0]!r}. Choose from {BUDYKO_CURVES}.")
    if fu_omega < 1.0:
        raise ValueError("fu_omega must be >= 1")
    if observed_et is not None and observed_runoff is not None:
        raise ValueError("Provide only one of 'observed_et' and 'observed_runoff'.")

    p = np.asarray(precipitation, dtype=float)
    e = np.asarray(pet, dtype=float)
    try:
        p, e = np.broadcast_arrays(p, e)
    except ValueError as exc:
        raise ValueError("precipitation and pet must be broadcast-compatible shapes.") from exc
    if not np.all(np.isfinite(p)):
        raise ValueError("precipitation must be finite")
    if not np.all(np.isfinite(e)):
        raise ValueError("pet must be finite")
    if np.any(p <= 0):
        raise ValueError("precipitation must be positive")
    if np.any(e <= 0):
        raise ValueError("pet must be positive")

    evapotranspiration = None
    if observed_et is not None:
        evapotranspiration = _observed_input("observed_et", observed_et, p)
        if not np.all(np.isfinite(evapotranspiration)):
            raise ValueError("observed_et must be finite")
        if np.any(evapotranspiration < 0) or np.any(evapotranspiration > p):
            raise ValueError("observed_et must lie within [0, precipitation]")
    elif observed_runoff is not None:
        runoff = _observed_input("observed_runoff", observed_runoff, p)
        if not np.all(np.isfinite(runoff)):
            raise ValueError("observed_runoff must be finite")
        if np.any(runoff < 0) or np.any(runoff > p):
            raise ValueError("observed_runoff must lie within [0, precipitation]")
        evapotranspiration = p - runoff

    if evapotranspiration is not None and np.any(evapotranspiration == 0.0):
        warnings.warn(
            "A zero long-term evapotranspiration is physically implausible; check "
            "the observed flux input.",
            stacklevel=2,
        )

    aridity = e / p
    predicted = {curve_name: _budyko_curve(aridity, curve_name, fu_omega) for curve_name in curves}
    aridity_grid = _DEFAULT_ARIDITY_GRID.copy()
    if evapotranspiration is not None and not _grid_covers(aridity_grid, aridity):
        lo = float(aridity.min())
        hi = float(aridity.max())
        aridity_grid = np.linspace(min(_DEFAULT_ARIDITY_GRID.min(), lo),
                                   max(_DEFAULT_ARIDITY_GRID.max(), hi), _DEFAULT_ARIDITY_GRID.size)
    curve_family = {curve_name: _budyko_curve(aridity_grid, curve_name, fu_omega) for curve_name in curves}

    return BudykoResult(
        aridity_index=_to_scalar_if_zero_dim(aridity),
        predicted={curve_name: _to_scalar_if_zero_dim(values) for curve_name, values in predicted.items()},
        observed_evaporative_ratio=None if evapotranspiration is None else _to_scalar_if_zero_dim(evapotranspiration / p),
        observed_deviation=None if evapotranspiration is None else {
            curve_name: _to_scalar_if_zero_dim(evapotranspiration / p - predicted[curve_name]) for curve_name in curves
        },
        aridity_grid=aridity_grid,
        curves=curve_family,
    )
