"""Shared hydrological model-evaluation metrics.

Standard goodness-of-fit statistics used to evaluate hydrological model
performance against observed data. Centralizes computations previously
duplicated across the models layer (transfer.py, base.py).

All functions share the signature ``metric(observed, simulated)`` and are
NaN-aware: any index where either array is NaN is dropped before computing.

References
----------
Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through
    conceptual models part I — A discussion of principles. Journal of
    Hydrology, 10(3), 282-290.
Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
    Decomposition of the mean squared error and NSE performance criteria:
    Implications for improving hydrological modelling. Journal of
    Hydrology, 377(1-2), 80-91.
"""

from __future__ import annotations

import numpy as np


def _paired_finite(observed: np.ndarray, simulated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop indices where either array is NaN, returning aligned arrays."""
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(simulated)
    return observed[mask], simulated[mask]


def nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)

    Ranges from -inf to 1; NSE = 1 is a perfect fit, NSE = 0 means the
    model is no better than the observed mean as a predictor.

    Reference: Nash & Sutcliffe (1970).
    """
    obs, sim = _paired_finite(observed, simulated)
    if len(obs) == 0:
        return float("nan")
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def log_nse(observed: np.ndarray, simulated: np.ndarray, epsilon: float = 1e-6) -> float:
    """Nash-Sutcliffe Efficiency computed on log-transformed flows.

    Emphasizes low-flow performance relative to standard NSE, which is
    dominated by peak-flow errors. A small ``epsilon`` is added before
    taking the log to safely handle zero flows.
    """
    obs, sim = _paired_finite(observed, simulated)
    if len(obs) == 0:
        return float("nan")
    log_obs = np.log(obs + epsilon)
    log_sim = np.log(sim + epsilon)
    return nse(log_obs, log_sim)


def kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Kling-Gupta Efficiency (2009 formulation).

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)

    where r is the Pearson correlation, alpha = std(sim)/std(obs) is the
    variability ratio, and beta = mean(sim)/mean(obs) is the bias ratio.
    Ranges from -inf to 1; KGE = 1 is a perfect fit.

    Reference: Gupta et al. (2009).
    """
    obs, sim = _paired_finite(observed, simulated)
    if len(obs) < 2:
        return float("nan")

    obs_std, sim_std = obs.std(), sim.std()
    obs_mean, sim_mean = obs.mean(), sim.mean()

    if obs_std == 0 or obs_mean == 0:
        return float("nan")

    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = float(sim_std / obs_std)
    beta = float(sim_mean / obs_mean)

    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Percent bias.

    PBIAS = 100 * sum(sim - obs) / sum(obs)

    Positive values indicate model overestimation bias; negative values
    indicate underestimation. PBIAS = 0 is a perfect fit (no bias).
    """
    obs, sim = _paired_finite(observed, simulated)
    if len(obs) == 0:
        return float("nan")
    obs_sum = np.sum(obs)
    if obs_sum == 0:
        return float("nan")
    return float(100 * np.sum(sim - obs) / obs_sum)


def rmse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Root Mean Squared Error.

    RMSE = sqrt(mean((obs - sim)^2))

    Always non-negative; RMSE = 0 is a perfect fit. Same units as the
    input data.
    """
    obs, sim = _paired_finite(observed, simulated)
    if len(obs) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def r2(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Coefficient of determination (R-squared).

    For this NaN-aware, NSE-style implementation, R2 is computed
    identically to NSE: 1 - sum((obs-sim)^2) / sum((obs-mean(obs))^2).
    Ranges from -inf to 1.
    """
    return nse(observed, simulated)
