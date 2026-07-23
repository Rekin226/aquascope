"""Bayesian uncertainty quantification for hydrological modelling.

Provides conjugate Bayesian linear/polynomial regression and a general-purpose
Metropolis-Hastings sampler, plus diagnostics (Gelman-Rubin, ESS, DIC, WAIC)
and model-comparison utilities.  Only depends on numpy, scipy, and pandas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PosteriorResult:
    """Result of Bayesian inference.

    Attributes:
        parameter_names: Human-readable names for each parameter.
        chains: Mapping of parameter name to its MCMC chain samples.
        posterior_mean: Point estimate (mean) per parameter.
        posterior_std: Standard deviation per parameter.
        credible_intervals: 95 % highest-density interval per parameter.
        predictions: Mean predictions on training data (if applicable).
        prediction_intervals: ``(lower, upper)`` 95 % prediction bands.
        r_hat: Gelman–Rubin convergence diagnostic per parameter.
        effective_sample_size: ESS per parameter.
        log_likelihood: Log-likelihood evaluated at the posterior mean.
        dic: Deviance Information Criterion.
        waic: Widely Applicable Information Criterion (``None`` when not computable).
    """

    parameter_names: list[str]
    chains: dict[str, np.ndarray]
    posterior_mean: dict[str, float]
    posterior_std: dict[str, float]
    credible_intervals: dict[str, tuple[float, float]]
    predictions: np.ndarray | None = None
    prediction_intervals: tuple[np.ndarray, np.ndarray] | None = None
    r_hat: dict[str, float] = field(default_factory=dict)
    effective_sample_size: dict[str, float] = field(default_factory=dict)
    log_likelihood: float = 0.0
    dic: float = 0.0
    waic: float | None = None


# ---------------------------------------------------------------------------
# Bayesian Linear Regression (conjugate normal-inverse-gamma)
# ---------------------------------------------------------------------------

class BayesianLinearRegression:
    """Bayesian linear regression with conjugate normal-inverse-gamma prior.

    Model::

        y = X @ beta + epsilon,   epsilon ~ N(0, sigma^2)

    Prior::

        beta ~ N(mu_0, sigma^2 * V_0)
        sigma^2 ~ InvGamma(a_0, b_0)

    Analytical posterior (conjugate)::

        beta | sigma^2, y ~ N(mu_n, sigma^2 * V_n)
        sigma^2 | y          ~ InvGamma(a_n, b_n)

    Parameters:
        prior_mean: Prior mean for beta (default: zeros).
        prior_precision: Prior precision (inverse variance scale) for beta.
        a0: Shape parameter of the InvGamma prior on sigma^2.
        b0: Scale parameter of the InvGamma prior on sigma^2.
    """

    def __init__(
        self,
        prior_mean: np.ndarray | None = None,
        prior_precision: float = 0.01,
        a0: float = 1.0,
        b0: float = 1.0,
    ) -> None:
        self._prior_mean = prior_mean
        self._prior_precision = prior_precision
        self._a0 = a0
        self._b0 = b0

        # Posterior quantities (set by fit)
        self._mu_n: np.ndarray | None = None
        self._V_n: np.ndarray | None = None
        self._a_n: float = 0.0
        self._b_n: float = 0.0
        self._result: PosteriorResult | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, n_samples: int = 5000) -> PosteriorResult:
        """Fit using the analytical conjugate posterior.

        Parameters:
            X: Design matrix of shape ``(n, p)``.
            y: Response vector of length ``n``.
            n_samples: Number of posterior draws to generate.

        Returns:
            A :class:`PosteriorResult` summarising the posterior.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape
        if n == 0:
            raise ValueError("Training data must not be empty.")

        # Prior setup
        mu_0 = np.zeros(p) if self._prior_mean is None else np.asarray(self._prior_mean, dtype=np.float64)
        V_0_inv = np.eye(p) * self._prior_precision

        # Posterior parameters
        XtX = X.T @ X
        Xty = X.T @ y
        V_n_inv = V_0_inv + XtX
        self._V_n = np.linalg.inv(V_n_inv)
        self._mu_n = self._V_n @ (V_0_inv @ mu_0 + Xty)
        self._a_n = self._a0 + n / 2.0
        self._b_n = (
            self._b0
            + 0.5 * (y @ y + mu_0 @ V_0_inv @ mu_0 - self._mu_n @ V_n_inv @ self._mu_n)
        )
        # Ensure b_n stays positive (numerical safety)
        self._b_n = max(self._b_n, 1e-12)

        # --- Draw posterior samples ---
        rng = np.random.default_rng(42)
        sigma2_samples = 1.0 / rng.gamma(shape=self._a_n, scale=1.0 / self._b_n, size=n_samples)
        beta_samples = np.zeros((n_samples, p))
        L_Vn = np.linalg.cholesky(self._V_n + np.eye(p) * 1e-10)
        for i in range(n_samples):
            z = rng.standard_normal(p)
            beta_samples[i] = self._mu_n + np.sqrt(sigma2_samples[i]) * (L_Vn @ z)

        # Build chains dict
        param_names = [f"beta_{j}" for j in range(p)]
        chains: dict[str, np.ndarray] = {name: beta_samples[:, j] for j, name in enumerate(param_names)}
        chains["sigma2"] = sigma2_samples
        all_names = param_names + ["sigma2"]

        # Summaries
        post_mean = {name: float(np.mean(chains[name])) for name in all_names}
        post_std = {name: float(np.std(chains[name])) for name in all_names}
        cred = {name: (float(np.percentile(chains[name], 2.5)), float(np.percentile(chains[name], 97.5))) for name in all_names}

        # Predictions on training set
        predictions = X @ self._mu_n

        # Log-likelihood at posterior mean
        residuals = y - predictions
        sigma2_mean = post_mean["sigma2"]
        ll = float(np.sum(stats.norm.logpdf(residuals, scale=np.sqrt(sigma2_mean))))

        # Pointwise log-likelihood matrix for DIC / WAIC
        ll_samples = np.zeros(n_samples)
        for i in range(n_samples):
            pred_i = X @ beta_samples[i]
            ll_samples[i] = float(np.sum(stats.norm.logpdf(y - pred_i, scale=np.sqrt(sigma2_samples[i]))))

        d_bar = float(-2.0 * np.mean(ll_samples))
        d_theta_bar = float(-2.0 * ll)
        p_d = d_bar - d_theta_bar
        dic_val = d_bar + p_d

        # WAIC (pointwise)
        waic_val = _compute_waic(X, y, beta_samples, sigma2_samples)

        # Convergence diagnostics (split-chain)
        r_hat_dict: dict[str, float] = {}
        ess_dict: dict[str, float] = {}
        for name in all_names:
            chain = chains[name]
            mid = len(chain) // 2
            r_hat_dict[name] = gelman_rubin([chain[:mid], chain[mid:]])
            ess_dict[name] = effective_sample_size(chain)

        self._result = PosteriorResult(
            parameter_names=all_names,
            chains=chains,
            posterior_mean=post_mean,
            posterior_std=post_std,
            credible_intervals=cred,
            predictions=predictions,
            prediction_intervals=None,
            r_hat=r_hat_dict,
            effective_sample_size=ess_dict,
            log_likelihood=ll,
            dic=dic_val,
            waic=waic_val,
        )
        logger.info("BayesianLinearRegression fit complete — DIC=%.2f, WAIC=%s", dic_val, waic_val)
        return self._result

    # ------------------------------------------------------------------
    def predict(
        self,
        X_new: np.ndarray | pd.DataFrame,
        n_samples: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior predictive distribution.

        Parameters:
            X_new: New design matrix of shape ``(m, p)``.
            n_samples: Number of posterior-predictive draws.

        Returns:
            ``(mean_prediction, lower_95, upper_95)``
        """
        if self._mu_n is None or self._V_n is None:
            raise RuntimeError("Model must be fitted before prediction.")

        X_new = np.asarray(X_new, dtype=np.float64)
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        rng = np.random.default_rng(123)
        sigma2_samples = 1.0 / rng.gamma(shape=self._a_n, scale=1.0 / self._b_n, size=n_samples)
        p = self._mu_n.shape[0]
        L_Vn = np.linalg.cholesky(self._V_n + np.eye(p) * 1e-10)

        preds = np.zeros((n_samples, X_new.shape[0]))
        for i in range(n_samples):
            z = rng.standard_normal(p)
            beta_i = self._mu_n + np.sqrt(sigma2_samples[i]) * (L_Vn @ z)
            mu_pred = X_new @ beta_i
            preds[i] = mu_pred + rng.normal(0, np.sqrt(sigma2_samples[i]), size=X_new.shape[0])

        mean_pred = np.mean(preds, axis=0)
        lower = np.percentile(preds, 2.5, axis=0)
        upper = np.percentile(preds, 97.5, axis=0)
        return mean_pred, lower, upper


# ---------------------------------------------------------------------------
# Bayesian Polynomial Regression
# ---------------------------------------------------------------------------

class BayesianPolynomialRegression:
    """Bayesian polynomial regression — wraps :class:`BayesianLinearRegression` with feature expansion.

    Parameters:
        degree: Polynomial degree (default 2).
        **kwargs: Forwarded to :class:`BayesianLinearRegression`.
    """

    def __init__(self, degree: int = 2, **kwargs) -> None:  # noqa: ANN003
        self._degree = degree
        self._blr = BayesianLinearRegression(**kwargs)

    @staticmethod
    def _expand(x: np.ndarray, degree: int) -> np.ndarray:
        """Build Vandermonde-style design matrix ``[1, x, x^2, …, x^degree]``."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.column_stack([x**d for d in range(degree + 1)])

    def fit(self, x: np.ndarray | pd.Series, y: np.ndarray | pd.Series, n_samples: int = 5000) -> PosteriorResult:
        """Fit polynomial by expanding features.

        Parameters:
            x: Predictor values of length ``n``.
            y: Response values of length ``n``.
            n_samples: Posterior draws to generate.

        Returns:
            A :class:`PosteriorResult`.
        """
        X = self._expand(np.asarray(x), self._degree)
        return self._blr.fit(X, y, n_samples=n_samples)

    def predict(
        self,
        x_new: np.ndarray | pd.Series,
        n_samples: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Posterior predictive for new *x* values.

        Returns:
            ``(mean_prediction, lower_95, upper_95)``
        """
        X_new = self._expand(np.asarray(x_new), self._degree)
        return self._blr.predict(X_new, n_samples=n_samples)


# ---------------------------------------------------------------------------
# Metropolis-Hastings MCMC sampler
# ---------------------------------------------------------------------------

class MetropolisHastings:
    """Metropolis-Hastings MCMC sampler for arbitrary log-posterior functions.

    Use when conjugate solutions are not available (e.g., non-linear models).

    Parameters:
        log_posterior_fn: Callable ``(theta) -> float`` returning the log-posterior
            (up to an additive constant).
        parameter_names: Human-readable name for each element of *theta*.
        proposal_scale: Standard deviation(s) for the Gaussian random-walk proposal.
    """

    def __init__(
        self,
        log_posterior_fn,  # noqa: ANN001
        parameter_names: list[str],
        proposal_scale: np.ndarray | float = 0.1,
    ) -> None:
        self._log_post = log_posterior_fn
        self._param_names = parameter_names
        self._proposal_scale = np.atleast_1d(np.asarray(proposal_scale, dtype=np.float64))

    def sample(
        self,
        initial: np.ndarray,
        n_samples: int = 5000,
        burn_in: int = 1000,
        thin: int = 1,
        seed: int | None = None,
    ) -> PosteriorResult:
        """Run MCMC sampling.

        Algorithm:
            1. Propose ``theta* = theta + N(0, proposal_scale)``.
            2. Accept with probability ``min(1, exp(log_post(theta*) - log_post(theta)))``.
            3. Repeat ``n_samples * thin + burn_in`` times, discard burn-in, thin.

        Parameters:
            initial: Starting parameter vector.
            n_samples: Desired number of posterior samples (after burn-in and thinning).
            burn_in: Iterations to discard at the start.
            thin: Keep every *thin*-th sample.
            seed: Random seed for reproducibility.

        Returns:
            A :class:`PosteriorResult`.
        """
        rng = np.random.default_rng(seed)
        theta = np.asarray(initial, dtype=np.float64).copy()
        d = len(theta)
        scale = np.broadcast_to(self._proposal_scale, d)

        total_iter = burn_in + n_samples * thin
        chain = np.empty((total_iter, d))
        log_p = self._log_post(theta)
        accepted = 0

        for t in range(total_iter):
            proposal = theta + rng.normal(0, scale, size=d)
            log_p_star = self._log_post(proposal)
            log_alpha = log_p_star - log_p
            if np.log(rng.uniform()) < log_alpha:
                theta = proposal
                log_p = log_p_star
                accepted += 1
            chain[t] = theta

        acceptance_rate = accepted / total_iter
        logger.info("Metropolis-Hastings acceptance rate: %.2f%%", 100 * acceptance_rate)

        # Discard burn-in, apply thinning
        samples = chain[burn_in::thin][:n_samples]

        chains_dict: dict[str, np.ndarray] = {name: samples[:, j] for j, name in enumerate(self._param_names)}
        post_mean = {name: float(np.mean(chains_dict[name])) for name in self._param_names}
        post_std = {name: float(np.std(chains_dict[name])) for name in self._param_names}
        cred = {
            name: (float(np.percentile(chains_dict[name], 2.5)), float(np.percentile(chains_dict[name], 97.5)))
            for name in self._param_names
        }

        # Convergence diagnostics (split-chain)
        r_hat_dict: dict[str, float] = {}
        ess_dict: dict[str, float] = {}
        for name in self._param_names:
            c = chains_dict[name]
            mid = len(c) // 2
            r_hat_dict[name] = gelman_rubin([c[:mid], c[mid:]])
            ess_dict[name] = effective_sample_size(c)

        return PosteriorResult(
            parameter_names=self._param_names,
            chains=chains_dict,
            posterior_mean=post_mean,
            posterior_std=post_std,
            credible_intervals=cred,
            predictions=None,
            prediction_intervals=None,
            r_hat=r_hat_dict,
            effective_sample_size=ess_dict,
            log_likelihood=float(log_p),
            dic=0.0,
            waic=None,
        )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def gelman_rubin(chains: list[np.ndarray]) -> float:
    """Gelman–Rubin *R-hat* convergence diagnostic.

    Parameters:
        chains: Two or more MCMC chains of equal length.

    Returns:
        R-hat value.  Values below 1.1 indicate approximate convergence.
    """
    m = len(chains)
    n = min(len(c) for c in chains)
    if n < 2 or m < 2:
        return float("nan")

    chain_means = np.array([np.mean(c[:n]) for c in chains])
    chain_vars = np.array([np.var(c[:n], ddof=1) for c in chains])

    W = float(np.mean(chain_vars))
    B = float(n * np.var(chain_means, ddof=1))

    if W < 1e-15:
        return 1.0

    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def effective_sample_size(chain: np.ndarray) -> float:
    """Effective sample size accounting for autocorrelation.

    Uses the initial-positive-sequence estimator: sum autocorrelation lags
    until the lag-*k* autocorrelation drops below 0.05.

    Parameters:
        chain: 1-D array of MCMC samples.

    Returns:
        Estimated ESS (always ≥ 1).
    """
    chain = np.asarray(chain, dtype=np.float64)
    n = len(chain)
    if n < 2:
        return 1.0

    var = np.var(chain)
    if var < 1e-15:
        return float(n)

    max_lag = min(n - 1, 500)
    rho_sum = 0.0
    for k in range(1, max_lag + 1):
        rho_k = np.corrcoef(chain[:-k], chain[k:])[0, 1]
        if np.isnan(rho_k) or abs(rho_k) < 0.05:
            break
        rho_sum += rho_k

    ess = n / (1.0 + 2.0 * rho_sum)
    return max(1.0, float(ess))


def dic(log_likelihood_fn, posterior_samples: dict[str, np.ndarray], data) -> float:  # noqa: ANN001
    """Deviance Information Criterion.

    Parameters:
        log_likelihood_fn: Callable ``(theta_dict, data) -> float``.
        posterior_samples: Parameter name → MCMC chain array.
        data: Opaque data object forwarded to *log_likelihood_fn*.

    Returns:
        DIC value (lower is better).
    """
    param_names = list(posterior_samples.keys())
    n_samples = len(next(iter(posterior_samples.values())))

    deviances = np.empty(n_samples)
    for i in range(n_samples):
        theta_i = {name: float(posterior_samples[name][i]) for name in param_names}
        deviances[i] = -2.0 * log_likelihood_fn(theta_i, data)

    d_bar = float(np.mean(deviances))

    theta_bar = {name: float(np.mean(posterior_samples[name])) for name in param_names}
    d_theta_bar = -2.0 * log_likelihood_fn(theta_bar, data)

    p_d = d_bar - d_theta_bar
    return float(d_bar + p_d)


def bayesian_model_comparison(results: list[tuple[str, PosteriorResult]]) -> pd.DataFrame:
    """Compare multiple Bayesian models by DIC and WAIC.

    Parameters:
        results: List of ``(model_name, PosteriorResult)`` pairs.

    Returns:
        :class:`pandas.DataFrame` with columns
        ``model_name``, ``DIC``, ``WAIC``, ``delta_DIC``, ``weight``.
    """
    rows: list[dict[str, object]] = []
    for name, res in results:
        rows.append({"model_name": name, "DIC": res.dic, "WAIC": res.waic})

    df = pd.DataFrame(rows)
    min_dic = df["DIC"].min()
    df["delta_DIC"] = df["DIC"] - min_dic

    # Akaike-style weights from DIC
    rel = np.exp(-0.5 * df["delta_DIC"].to_numpy(dtype=np.float64))
    df["weight"] = rel / rel.sum()

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_waic(
    X: np.ndarray,
    y: np.ndarray,
    beta_samples: np.ndarray,
    sigma2_samples: np.ndarray,
) -> float | None:
    """Pointwise WAIC from posterior samples.

    Returns:
        WAIC value, or ``None`` if computation fails.
    """
    n = len(y)
    n_samples = len(sigma2_samples)
    log_lik = np.zeros((n_samples, n))

    for i in range(n_samples):
        mu_i = X @ beta_samples[i]
        sd_i = np.sqrt(sigma2_samples[i])
        log_lik[i] = stats.norm.logpdf(y, loc=mu_i, scale=sd_i)

    # lppd
    max_ll = np.max(log_lik, axis=0)
    lppd = np.sum(np.log(np.mean(np.exp(log_lik - max_ll), axis=0)) + max_ll)

    # p_waic (effective number of params)
    p_waic = np.sum(np.var(log_lik, axis=0))

    waic_val = -2.0 * (lppd - p_waic)
    if np.isfinite(waic_val):
        return float(waic_val)
    return None
