"""Shared hydrological model-evaluation metrics.

Standard goodness-of-fit statistics used to evaluate hydrological model
performance against observed data. Centralizes computations previously
duplicated across the models layer (transfer.py, base.py).

The deterministic functions share the signature ``metric(observed, simulated)``
and are NaN-aware: any index where either array is NaN is dropped before
computing. The probabilistic functions (``pinball_loss``, ``picp``, ``mpiw``,
``crps_ensemble``) score interval and ensemble forecasts and are likewise
NaN-aware on their observed input.

References
----------
Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through
    conceptual models part I — A discussion of principles. Journal of
    Hydrology, 10(3), 282-290.
Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009).
    Decomposition of the mean squared error and NSE performance criteria:
    Implications for improving hydrological modelling. Journal of
    Hydrology, 377(1-2), 80-91.
Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
    prediction, and estimation. JASA, 102(477), 359-378. (CRPS)
Koenker, R., & Bassett, G. (1978). Regression quantiles. Econometrica,
    46(1), 33-50. (pinball / quantile loss)
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


# ── Probabilistic / uncertainty metrics ──────────────────────────────────


def pinball_loss(
    observed: np.ndarray, predicted: np.ndarray, quantile: float
) -> float:
    """Pinball (quantile) loss for a single quantile forecast.

    For quantile level ``q`` and error ``e = obs - pred``::

        loss = mean( max(q * e, (q - 1) * e) )

    Penalizes under-prediction by ``q`` and over-prediction by ``1 - q``,
    so it is minimized by the true conditional ``q``-quantile. Always
    non-negative; lower is better. Reference: Koenker & Bassett (1978).
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1); got {quantile}.")
    obs, pred = _paired_finite(observed, predicted)
    if len(obs) == 0:
        return float("nan")
    err = obs - pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def picp(
    observed: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Prediction Interval Coverage Probability.

    Fraction of observations that fall within the ``[lower, upper]``
    interval. For a well-calibrated central interval at nominal level
    ``1 - alpha`` (e.g. 0.90 for 5%/95% bounds), PICP should be close to
    that nominal value. Ranges 0 to 1.
    """
    observed = np.asarray(observed, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return float("nan")
    obs, lo, hi = observed[mask], lower[mask], upper[mask]
    inside = (obs >= lo) & (obs <= hi)
    return float(np.mean(inside))


def mpiw(
    lower: np.ndarray, upper: np.ndarray, observed: np.ndarray | None = None
) -> float:
    """Mean Prediction Interval Width.

    ``mean(upper - lower)``. Sharper (narrower) intervals are preferred,
    but only when coverage (:func:`picp`) is maintained — report both.
    If ``observed`` is given, the width is normalized by the observed
    range (max - min), yielding a unitless sharpness measure.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mask = np.isfinite(lower) & np.isfinite(upper)
    if not mask.any():
        return float("nan")
    width = float(np.mean(upper[mask] - lower[mask]))
    if observed is None:
        return width
    obs = np.asarray(observed, dtype=float)
    obs = obs[np.isfinite(obs)]
    if len(obs) == 0:
        return float("nan")
    obs_range = float(obs.max() - obs.min())
    if obs_range == 0:
        return float("nan")
    return width / obs_range


def crps_ensemble(observed: np.ndarray, ensemble: np.ndarray) -> float:
    """Continuous Ranked Probability Score for an ensemble forecast.

    Empirical estimator over an ensemble of shape ``(n_obs, n_members)``::

        CRPS_i = mean_j |x_ij - y_i| - (1 / (2 m^2)) sum_j sum_k |x_ij - x_ik|

    averaged over time steps ``i``. Generalizes MAE to probabilistic
    forecasts (a deterministic ensemble reduces exactly to MAE); lower is
    better, same units as the data. Ensemble members are assumed finite;
    time steps with a NaN observation are dropped. Reference: Gneiting &
    Raftery (2007).
    """
    observed = np.asarray(observed, dtype=float)
    ensemble = np.asarray(ensemble, dtype=float)
    if ensemble.ndim != 2 or ensemble.shape[0] != observed.shape[0]:
        raise ValueError(
            "ensemble must have shape (n_obs, n_members) matching observed."
        )
    row_mask = np.isfinite(observed)
    obs = observed[row_mask]
    ens = ensemble[row_mask]
    if len(obs) == 0:
        return float("nan")
    m = ens.shape[1]
    if m == 0:
        return float("nan")
    term1 = np.mean(np.abs(ens - obs[:, None]), axis=1)
    pairwise = np.abs(ens[:, :, None] - ens[:, None, :]).sum(axis=(1, 2))
    term2 = pairwise / (2.0 * m * m)
    return float(np.mean(term1 - term2))


def crps_from_quantiles(
    observed: np.ndarray, quantile_preds: dict[float, np.ndarray]
) -> float:
    """Approximate CRPS from a set of predictive quantiles.

    Uses the quantile decomposition ``CRPS = 2 * integral_0^1 pinball_q dq``,
    approximated as ``2 * mean_q pinball_loss(obs, pred_q, q)`` over the
    provided quantile levels. Accuracy improves with a denser, more uniform
    set of quantiles. Lets the residual quantile-band method report a CRPS
    without forming an explicit ensemble. Reference: Gneiting & Raftery (2007).
    """
    if not quantile_preds:
        return float("nan")
    losses = [
        pinball_loss(observed, pred, q) for q, pred in quantile_preds.items()
    ]
    losses = [v for v in losses if np.isfinite(v)]
    if not losses:
        return float("nan")
    return float(2.0 * np.mean(losses))
