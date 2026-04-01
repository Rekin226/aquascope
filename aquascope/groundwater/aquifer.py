"""Aquifer hydraulics and pumping test analysis.

Implements the Theis equation, Cooper-Jacob approximation, recovery
test analysis, and transmissivity/storativity estimation from
pumping test data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

logger = logging.getLogger(__name__)


@dataclass
class AquiferParams:
    """Estimated aquifer hydraulic parameters.

    Attributes
    ----------
    transmissivity:
        Transmissivity T (m²/day).
    storativity:
        Storativity S (dimensionless).
    method:
        Estimation method used.
    """

    transmissivity: float
    storativity: float
    method: str


@dataclass
class SafeYieldResult:
    """Sustainable yield assessment.

    Attributes
    ----------
    safe_yield_mm:
        Estimated safe yield (mm/year).
    ratio:
        Extraction-to-recharge ratio (dimensionless).
    assessment:
        ``"sustainable"``, ``"at risk"``, or ``"unsustainable"``.
    """

    safe_yield_mm: float
    ratio: float
    assessment: str


# ---------------------------------------------------------------------------
# Theis well function helpers
# ---------------------------------------------------------------------------


def _well_function(u: np.ndarray) -> np.ndarray:
    """Evaluate the Theis well function W(u) = -Ei(-u) = E₁(u).

    Parameters
    ----------
    u:
        Argument of the well function (must be > 0).

    Returns
    -------
    np.ndarray
        W(u) values.
    """
    return special.exp1(u)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def theis_drawdown(
    T: float,  # noqa: N803
    S: float,  # noqa: N803
    Q: float,  # noqa: N803
    r: float,
    t: float | np.ndarray,
) -> np.ndarray:
    """Compute drawdown using the Theis equation.

    ``s = Q / (4πT) × W(u)`` where ``u = r²S / (4Tt)``.

    Parameters
    ----------
    T:
        Transmissivity (m²/day).
    S:
        Storativity (dimensionless).
    Q:
        Pumping rate (m³/day).
    r:
        Radial distance from pumping well (m).
    t:
        Time since pumping started (days). Scalar or array.

    Returns
    -------
    np.ndarray
        Drawdown (m) at distance *r* and time(s) *t*.

    Raises
    ------
    ValueError
        If any physical parameter is non-positive, or *t* contains
        non-positive values.
    """
    if T <= 0:
        raise ValueError(f"Transmissivity must be positive, got {T}.")
    if S <= 0:
        raise ValueError(f"Storativity must be positive, got {S}.")
    if Q <= 0:
        raise ValueError(f"Pumping rate must be positive, got {Q}.")
    if r <= 0:
        raise ValueError(f"Radial distance must be positive, got {r}.")

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t_arr <= 0):
        raise ValueError("Time values must be positive.")

    u = (r**2 * S) / (4.0 * T * t_arr)
    w_u = _well_function(u)
    drawdown = (Q / (4.0 * np.pi * T)) * w_u

    return np.asarray(drawdown)


def cooper_jacob(
    T: float,  # noqa: N803
    S: float,  # noqa: N803
    Q: float,  # noqa: N803
    r: float,
    t: float | np.ndarray,
) -> np.ndarray:
    """Compute drawdown using the Cooper-Jacob approximation.

    Valid when ``u = r²S/(4Tt) < 0.01``.

    ``s = Q / (4πT) × [−0.5772 − ln(u)]``
      ≈ ``Q / (4πT) × ln(2.25Tt / (r²S))``

    Parameters
    ----------
    T:
        Transmissivity (m²/day).
    S:
        Storativity (dimensionless).
    Q:
        Pumping rate (m³/day).
    r:
        Radial distance from pumping well (m).
    t:
        Time since pumping started (days).

    Returns
    -------
    np.ndarray
        Drawdown (m).

    Raises
    ------
    ValueError
        If any physical parameter is non-positive.
    """
    if T <= 0:
        raise ValueError(f"Transmissivity must be positive, got {T}.")
    if S <= 0:
        raise ValueError(f"Storativity must be positive, got {S}.")
    if Q <= 0:
        raise ValueError(f"Pumping rate must be positive, got {Q}.")
    if r <= 0:
        raise ValueError(f"Radial distance must be positive, got {r}.")

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t_arr <= 0):
        raise ValueError("Time values must be positive.")

    u = (r**2 * S) / (4.0 * T * t_arr)
    if np.any(u >= 0.01):
        logger.warning("Cooper-Jacob approximation may be inaccurate: max u=%.4f (should be < 0.01).", np.max(u))

    drawdown = (Q / (4.0 * np.pi * T)) * np.log(2.25 * T * t_arr / (r**2 * S))

    return np.asarray(drawdown)


def theis_recovery(
    T: float,  # noqa: N803
    Q: float,  # noqa: N803
    t: np.ndarray,
    tp: float,
) -> np.ndarray:
    """Compute residual drawdown during recovery.

    ``s' = Q / (4πT) × ln(t / t')``

    where ``t`` is time since pumping started and ``t' = t − tp``
    is time since pumping stopped.

    Parameters
    ----------
    T:
        Transmissivity (m²/day).
    Q:
        Pumping rate during the pumping phase (m³/day).
    t:
        Time since pumping started (days). Must be > *tp*.
    tp:
        Duration of the pumping phase (days).

    Returns
    -------
    np.ndarray
        Residual drawdown (m).

    Raises
    ------
    ValueError
        If parameters are invalid or any *t* ≤ *tp*.
    """
    if T <= 0:
        raise ValueError(f"Transmissivity must be positive, got {T}.")
    if Q <= 0:
        raise ValueError(f"Pumping rate must be positive, got {Q}.")
    if tp <= 0:
        raise ValueError(f"Pumping duration must be positive, got {tp}.")

    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    if np.any(t_arr <= tp):
        raise ValueError("All time values must be greater than the pumping duration tp.")

    t_prime = t_arr - tp
    residual = (Q / (4.0 * np.pi * T)) * np.log(t_arr / t_prime)

    return np.asarray(residual)


def estimate_transmissivity(
    time: np.ndarray,
    drawdown: np.ndarray,
    Q: float,  # noqa: N803
    r: float,
    method: str = "cooper_jacob",
) -> AquiferParams:
    """Estimate transmissivity and storativity from pumping test data.

    Parameters
    ----------
    time:
        Time values (days) during the pumping test.
    drawdown:
        Measured drawdown values (m) corresponding to *time*.
    Q:
        Pumping rate (m³/day).
    r:
        Radial distance from pumping well (m).
    method:
        Estimation method: ``"cooper_jacob"`` or ``"theis"``.

    Returns
    -------
    AquiferParams
        Estimated T and S.

    Raises
    ------
    ValueError
        If method is unknown or fitting fails.
    """
    time = np.asarray(time, dtype=float)
    drawdown = np.asarray(drawdown, dtype=float)

    if method == "cooper_jacob":
        # Cooper-Jacob: s = Q/(4πT) × ln(2.25Tt/(r²S))
        # Regress s against ln(t):  s = m × ln(t) + c
        # where m = Q/(4πT)  →  T = Q/(4πm)
        # and c = Q/(4πT) × ln(2.25T/(r²S))
        log_t = np.log(time)

        # Use numpy polyfit (degree 1)
        coeffs = np.polyfit(log_t, drawdown, 1)  # [slope, intercept]
        m = coeffs[0]  # ds/d(ln t)
        c = coeffs[1]

        if m <= 0:
            raise ValueError("Non-positive slope — data may not follow Cooper-Jacob model.")

        t_est = Q / (4.0 * np.pi * m)
        # From intercept: c = m × ln(2.25T/(r²S))  →  S = 2.25T / (r² × exp(c/m))
        s_est = 2.25 * t_est / (r**2 * np.exp(c / m))

        logger.info("Cooper-Jacob fit: T=%.2f m²/day, S=%.6f", t_est, s_est)
        return AquiferParams(transmissivity=float(t_est), storativity=float(s_est), method="cooper_jacob")

    elif method == "theis":
        # Non-linear least squares fit of the Theis equation

        def _residuals(params: np.ndarray) -> np.ndarray:
            t_trial, s_trial = params
            if t_trial <= 0 or s_trial <= 0:
                return np.full_like(drawdown, 1e10)
            u = (r**2 * s_trial) / (4.0 * t_trial * time)
            w_u = _well_function(u)
            predicted = (Q / (4.0 * np.pi * t_trial)) * w_u
            return predicted - drawdown

        # Initial guess from Cooper-Jacob
        try:
            cj = estimate_transmissivity(time, drawdown, Q, r, method="cooper_jacob")
            x0 = np.array([cj.transmissivity, cj.storativity])
        except ValueError:
            x0 = np.array([100.0, 0.001])

        result = optimize.least_squares(
            _residuals,
            x0,
            bounds=([1e-6, 1e-10], [1e6, 1.0]),
        )

        if not result.success:
            raise ValueError(f"Theis curve fitting failed: {result.message}")

        t_est = float(result.x[0])
        s_est = float(result.x[1])

        logger.info("Theis fit: T=%.2f m²/day, S=%.6f", t_est, s_est)
        return AquiferParams(transmissivity=t_est, storativity=s_est, method="theis")

    else:
        raise ValueError(f"Unknown method: {method!r}. Supported: 'cooper_jacob', 'theis'.")


def safe_yield(
    area_km2: float,
    recharge_mm: float,
    current_extraction_mm: float,
) -> SafeYieldResult:
    """Assess sustainable groundwater extraction.

    Parameters
    ----------
    area_km2:
        Aquifer area (km²).
    recharge_mm:
        Annual recharge rate (mm/year).
    current_extraction_mm:
        Current annual extraction rate (mm/year).

    Returns
    -------
    SafeYieldResult
        Safe yield, extraction ratio, and sustainability assessment.

    Raises
    ------
    ValueError
        If area or recharge is non-positive.
    """
    if area_km2 <= 0:
        raise ValueError("Area must be positive.")
    if recharge_mm <= 0:
        raise ValueError("Recharge must be positive.")
    if current_extraction_mm < 0:
        raise ValueError("Extraction must be non-negative.")

    # Safe yield = recharge (simplified — doesn't account for environmental flows)
    safe = recharge_mm
    ratio = current_extraction_mm / recharge_mm

    if ratio <= 0.7:
        assessment = "sustainable"
    elif ratio <= 1.0:
        assessment = "at risk"
    else:
        assessment = "unsustainable"

    logger.info("Safe yield: %.1f mm/year, ratio=%.2f → %s", safe, ratio, assessment)
    return SafeYieldResult(safe_yield_mm=safe, ratio=ratio, assessment=assessment)
