"""
Copula-based multivariate dependence modeling.

Copulas model dependence between random variables separately from their
marginal distributions — essential for joint flood-drought analysis,
multivariate water quality assessment, and correlated extreme-event modeling.

Supported families: Gaussian, Clayton, Gumbel, Frank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 10
_FAMILIES = ("gaussian", "clayton", "gumbel", "frank")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CopulaResult:
    """Result of copula fitting.

    Attributes:
        family: Copula family name — ``"gaussian"``, ``"clayton"``,
            ``"gumbel"``, or ``"frank"``.
        parameter: Copula parameter (theta).
        kendall_tau: Kendall rank-correlation coefficient.
        spearman_rho: Spearman rank-correlation coefficient.
        aic: Akaike Information Criterion.
        log_likelihood: Maximised log-likelihood.
        n_samples: Number of observation pairs used for fitting.
    """

    family: str
    parameter: float
    kendall_tau: float
    spearman_rho: float
    aic: float
    log_likelihood: float
    n_samples: int


@dataclass
class JointProbability:
    """Joint exceedance / non-exceedance probability result.

    Attributes:
        prob_both_exceed: P(X > x AND Y > y).
        prob_either_exceed: P(X > x OR Y > y).
        prob_x_exceed_given_y: P(X > x | Y > y).
        joint_return_period: 1 / P(both exceed).
    """

    prob_both_exceed: float
    prob_either_exceed: float
    prob_x_exceed_given_y: float
    joint_return_period: float


# ---------------------------------------------------------------------------
# Pseudo-observations
# ---------------------------------------------------------------------------


def to_pseudo_observations(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw data to pseudo-observations via rank transform.

    Uses the empirical CDF transform ``u_i = rank(x_i) / (n + 1)`` so
    that values are strictly inside (0, 1).

    Parameters:
        x: First variable (1-D array or Series).
        y: Second variable (1-D array or Series).

    Returns:
        Tuple ``(u, v)`` of pseudo-observations in (0, 1).

    Raises:
        ValueError: If *x* and *y* have different lengths or fewer than
            ``_MIN_SAMPLES`` observations.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length.")
    n = x_arr.shape[0]
    if n < _MIN_SAMPLES:
        raise ValueError(f"Need at least {_MIN_SAMPLES} observations, got {n}.")

    u = stats.rankdata(x_arr) / (n + 1)
    v = stats.rankdata(y_arr) / (n + 1)
    return u, v


# ---------------------------------------------------------------------------
# Copula C(u, v) evaluation
# ---------------------------------------------------------------------------


def copula_function(u: float, v: float, family: str, theta: float) -> float:
    """Evaluate the copula function C(u, v).

    Parameters:
        u: First pseudo-observation in (0, 1).
        v: Second pseudo-observation in (0, 1).
        family: One of ``"gaussian"``, ``"clayton"``, ``"gumbel"``,
            ``"frank"``.
        theta: Copula parameter.

    Returns:
        Copula value C(u, v) in [0, 1].

    Raises:
        ValueError: For an unknown *family*.
    """
    family = family.lower()
    if family == "gaussian":
        x1 = stats.norm.ppf(u)
        x2 = stats.norm.ppf(v)
        return stats.multivariate_normal.cdf(  # type: ignore[return-value]
            [x1, x2],
            mean=[0, 0],
            cov=[[1, theta], [theta, 1]],
        )
    if family == "clayton":
        if theta <= 0:
            raise ValueError("Clayton parameter must be > 0.")
        return float(max((u ** (-theta) + v ** (-theta) - 1), 0) ** (-1.0 / theta))
    if family == "gumbel":
        if theta < 1:
            raise ValueError("Gumbel parameter must be >= 1.")
        a = (-np.log(u)) ** theta + (-np.log(v)) ** theta
        return float(np.exp(-(a ** (1.0 / theta))))
    if family == "frank":
        if theta == 0:
            return u * v  # independence
        e_t = np.exp(-theta)
        num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        return float(-np.log(1 + num / (e_t - 1)) / theta)
    raise ValueError(f"Unknown copula family: {family!r}")


# ---------------------------------------------------------------------------
# Copula density c(u, v)
# ---------------------------------------------------------------------------


def copula_density(u: float, v: float, family: str, theta: float) -> float:
    """Evaluate copula density c(u, v) = ∂²C / ∂u ∂v.

    Parameters:
        u: First pseudo-observation in (0, 1).
        v: Second pseudo-observation in (0, 1).
        family: Copula family name.
        theta: Copula parameter.

    Returns:
        Copula density (positive for valid inputs).

    Raises:
        ValueError: For an unknown *family*.
    """
    family = family.lower()
    if family == "gaussian":
        rho = theta
        x = stats.norm.ppf(u)
        y = stats.norm.ppf(v)
        det = 1 - rho ** 2
        return float(
            (1.0 / np.sqrt(det))
            * np.exp(-(rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * det))
        )
    if family == "clayton":
        if theta <= 0:
            raise ValueError("Clayton parameter must be > 0.")
        c_uv = copula_function(u, v, "clayton", theta)
        return float(
            (1 + theta)
            * (u * v) ** (-(1 + theta))
            * c_uv ** (1 + 2 * theta)
        )
    if family == "gumbel":
        if theta < 1:
            raise ValueError("Gumbel parameter must be >= 1.")
        lu = -np.log(u)
        lv = -np.log(v)
        a = lu ** theta + lv ** theta
        a_inv = a ** (1.0 / theta)
        c_uv = np.exp(-a_inv)
        # Density formula for Gumbel copula
        t1 = c_uv / (u * v)
        t2 = (lu * lv) ** (theta - 1)
        t3 = a ** (2.0 / theta - 2)
        t4 = a_inv + theta - 1
        return float(t1 * t2 * t3 * t4)
    if family == "frank":
        if theta == 0:
            return 1.0  # independence
        e_tu = np.exp(-theta * u)
        e_tv = np.exp(-theta * v)
        e_t = np.exp(-theta)
        num = -theta * (e_t - 1) * np.exp(-theta * (u + v))
        den = ((e_t - 1) + (e_tu - 1) * (e_tv - 1)) ** 2
        return float(num / den)
    raise ValueError(f"Unknown copula family: {family!r}")


# ---------------------------------------------------------------------------
# Debye function helper (for Frank copula fitting)
# ---------------------------------------------------------------------------


def _debye1(theta: float) -> float:
    """First Debye function D₁(θ) = (1/θ) ∫₀^θ t/(eᵗ−1) dt."""
    if abs(theta) < 1e-10:
        return 1.0
    result, _ = integrate.quad(lambda t: t / (np.exp(t) - 1), 0, abs(theta))
    return result / abs(theta)


def _frank_tau_equation(theta: float, tau: float) -> float:
    """Equation to solve for Frank parameter: tau = 1 - 4/θ*(1 - D₁(θ))."""
    if abs(theta) < 1e-10:
        return -tau
    return 1.0 - 4.0 / theta * (1.0 - _debye1(theta)) - tau


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _fit_single_copula(
    u: np.ndarray,
    v: np.ndarray,
    family: str,
    tau: float,
    rho_s: float,
) -> CopulaResult:
    """Fit one copula family to pseudo-observations.

    Parameters:
        u: First pseudo-observations.
        v: Second pseudo-observations.
        family: Copula family name.
        tau: Pre-computed Kendall tau.
        rho_s: Pre-computed Spearman rho.

    Returns:
        A :class:`CopulaResult`.
    """
    n = len(u)
    family = family.lower()

    if family == "gaussian":
        theta = float(np.sin(np.pi * tau / 2))
        theta = np.clip(theta, -0.999, 0.999)
    elif family == "clayton":
        theta = 2 * tau / (1 - tau) if tau < 1 else 20.0
        theta = max(theta, 1e-4)
    elif family == "gumbel":
        theta = 1.0 / (1 - tau) if tau < 1 else 20.0
        theta = max(theta, 1.0)
    elif family == "frank":
        if abs(tau) < 1e-8:
            theta = 1e-4
        else:
            try:
                theta = optimize.brentq(_frank_tau_equation, -30, 30, args=(tau,))
            except ValueError:
                theta = 1e-4
    else:
        raise ValueError(f"Unknown copula family: {family!r}")

    # Log-likelihood
    ll = 0.0
    for i in range(n):
        d = copula_density(u[i], v[i], family, theta)
        ll += np.log(max(d, 1e-300))
    aic = -2.0 * ll + 2.0  # k = 1

    return CopulaResult(
        family=family,
        parameter=float(theta),
        kendall_tau=tau,
        spearman_rho=rho_s,
        aic=aic,
        log_likelihood=ll,
        n_samples=n,
    )


def fit_copula(
    u: np.ndarray,
    v: np.ndarray,
    family: str = "best",
) -> CopulaResult:
    """Fit a copula to pseudo-observations.

    Parameters:
        u: First pseudo-observations in (0, 1).
        v: Second pseudo-observations in (0, 1).
        family: ``"gaussian"``, ``"clayton"``, ``"gumbel"``, ``"frank"``,
            or ``"best"`` (try all families and pick the lowest AIC).

    Returns:
        A :class:`CopulaResult` for the selected family.

    Raises:
        ValueError: If fewer than ``_MIN_SAMPLES`` pairs are provided or
            an unknown *family* is given.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if len(u) < _MIN_SAMPLES:
        raise ValueError(f"Need at least {_MIN_SAMPLES} observations, got {len(u)}.")

    tau, _ = stats.kendalltau(u, v)
    rho_s, _ = stats.spearmanr(u, v)

    family_lower = family.lower()
    if family_lower == "best":
        results = compare_copulas(u, v)
        return results[0]

    return _fit_single_copula(u, v, family_lower, tau, rho_s)


# ---------------------------------------------------------------------------
# Compare all families
# ---------------------------------------------------------------------------


def compare_copulas(u: np.ndarray, v: np.ndarray) -> list[CopulaResult]:
    """Fit all copula families and return results sorted by AIC (best first).

    Parameters:
        u: First pseudo-observations in (0, 1).
        v: Second pseudo-observations in (0, 1).

    Returns:
        List of :class:`CopulaResult` ordered by ascending AIC.

    Raises:
        ValueError: If fewer than ``_MIN_SAMPLES`` pairs are provided.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if len(u) < _MIN_SAMPLES:
        raise ValueError(f"Need at least {_MIN_SAMPLES} observations, got {len(u)}.")

    tau, _ = stats.kendalltau(u, v)
    rho_s, _ = stats.spearmanr(u, v)

    # Clayton and Gumbel require positive dependence
    effective_tau = max(tau, 0.05)

    results: list[CopulaResult] = []
    for fam in _FAMILIES:
        try:
            t = effective_tau if fam in ("clayton", "gumbel") else tau
            res = _fit_single_copula(u, v, fam, t, rho_s)
            results.append(res)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to fit %s copula", fam)
    results.sort(key=lambda r: r.aic)
    return results


# ---------------------------------------------------------------------------
# Joint probabilities
# ---------------------------------------------------------------------------


def joint_exceedance_probability(
    copula: CopulaResult,
    u_threshold: float,
    v_threshold: float,
) -> JointProbability:
    """Compute joint exceedance probabilities from a fitted copula.

    Parameters:
        copula: A fitted :class:`CopulaResult`.
        u_threshold: Threshold on U scale (0, 1).
        v_threshold: Threshold on V scale (0, 1).

    Returns:
        A :class:`JointProbability` with exceedance statistics.
    """
    c_uv = copula_function(u_threshold, v_threshold, copula.family, copula.parameter)

    prob_both = 1.0 - u_threshold - v_threshold + c_uv
    prob_both = max(prob_both, 0.0)

    prob_either = 1.0 - c_uv

    p_v_exceed = 1.0 - v_threshold
    prob_conditional = prob_both / p_v_exceed if p_v_exceed > 1e-12 else 0.0

    return_period = 1.0 / prob_both if prob_both > 1e-12 else float("inf")

    return JointProbability(
        prob_both_exceed=prob_both,
        prob_either_exceed=prob_either,
        prob_x_exceed_given_y=prob_conditional,
        joint_return_period=return_period,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def generate_copula_samples(
    copula: CopulaResult,
    n: int = 1000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random samples from a fitted copula.

    Parameters:
        copula: A fitted :class:`CopulaResult`.
        n: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple ``(u, v)`` of pseudo-observations on (0, 1).

    Raises:
        ValueError: For an unsupported copula family.
    """
    rng = np.random.default_rng(seed)
    family = copula.family.lower()
    theta = copula.parameter

    if family == "gaussian":
        cov = np.array([[1.0, theta], [theta, 1.0]])
        z = rng.multivariate_normal([0, 0], cov, size=n)
        u = stats.norm.cdf(z[:, 0])
        v = stats.norm.cdf(z[:, 1])
        return u, v

    if family == "clayton":
        # Conditional inverse method
        u1 = rng.uniform(size=n)
        t = rng.uniform(size=n)
        # v2 = (u1^{-theta} * (t^{-theta/(1+theta)} - 1) + 1)^{-1/theta}
        v2 = (u1 ** (-theta) * (t ** (-theta / (1 + theta)) - 1) + 1) ** (-1.0 / theta)
        return u1, v2

    if family == "gumbel":
        # Marshall-Olkin algorithm: generate via stable distribution
        alpha = 1.0 / theta
        if abs(alpha - 1.0) < 1e-10:
            # theta ≈ 1 → independence: just return uniforms
            return rng.uniform(size=n), rng.uniform(size=n)
        # Positive stable(alpha, 1) via Chambers-Mallows-Stuck
        w = rng.exponential(size=n)
        unif = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
        num = np.sin(alpha * (unif + np.pi / 2))
        den = np.cos(unif) ** (1.0 / alpha)
        frac = np.cos(unif - alpha * (unif + np.pi / 2)) / w
        s = (num / den) * frac ** ((1 - alpha) / alpha)
        s = np.abs(s)  # ensure positive
        e1 = rng.exponential(size=n)
        e2 = rng.exponential(size=n)
        u = np.exp(-(e1 / s) ** alpha)
        v = np.exp(-(e2 / s) ** alpha)
        # Clip to strict (0, 1) — numerical edge cases
        u = np.clip(u, 1e-12, 1 - 1e-12)
        v = np.clip(v, 1e-12, 1 - 1e-12)
        return u, v

    if family == "frank":
        # Conditional inverse method
        u1 = rng.uniform(size=n)
        t = rng.uniform(size=n)
        if abs(theta) < 1e-10:
            return u1, t
        # Conditional CDF inverse for Frank
        v2 = -np.log(1 + t * (np.exp(-theta) - 1) / (t + (1 - t) * np.exp(-theta * u1))) / theta
        return u1, v2

    raise ValueError(f"Unsupported copula family: {family!r}")


def generate_synthetic_data(
    copula: CopulaResult,
    marginal_x: tuple[str, tuple],
    marginal_y: tuple[str, tuple],
    n: int = 1000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic bivariate data preserving the dependence structure.

    1. Generate copula samples (u, v).
    2. Apply the inverse CDF of each marginal to recover original scales.

    Parameters:
        copula: A fitted :class:`CopulaResult`.
        marginal_x: ``(dist_name, params)`` for :mod:`scipy.stats`, e.g.
            ``("norm", (0, 1))``.
        marginal_y: Same format as *marginal_x*.
        n: Number of samples.
        seed: Random seed.

    Returns:
        Tuple ``(x, y)`` of arrays in the marginal domains.
    """
    u, v = generate_copula_samples(copula, n=n, seed=seed)

    dist_x = getattr(stats, marginal_x[0])
    dist_y = getattr(stats, marginal_y[0])

    x = dist_x.ppf(u, *marginal_x[1])
    y = dist_y.ppf(v, *marginal_y[1])
    return x, y


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------


def tail_dependence(copula: CopulaResult) -> dict[str, float]:
    """Compute upper and lower tail dependence coefficients.

    Parameters:
        copula: A fitted :class:`CopulaResult`.

    Returns:
        Dict with keys ``"lower"`` and ``"upper"``.
    """
    family = copula.family.lower()
    theta = copula.parameter

    if family == "clayton":
        lam_l = 2.0 ** (-1.0 / theta) if theta > 0 else 0.0
        return {"lower": lam_l, "upper": 0.0}
    if family == "gumbel":
        lam_u = 2.0 - 2.0 ** (1.0 / theta) if theta >= 1 else 0.0
        return {"lower": 0.0, "upper": lam_u}
    if family == "frank":
        return {"lower": 0.0, "upper": 0.0}
    if family == "gaussian":
        return {"lower": 0.0, "upper": 0.0}
    raise ValueError(f"Unknown copula family: {family!r}")
