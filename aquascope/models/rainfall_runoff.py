"""Conceptual rainfall-runoff modelling.

Implements the GR4J lumped conceptual model (Perrin, Michel & Andreassian,
2003) — a 4-parameter daily rainfall-runoff model widely used in
operational hydrology.

GR4J simulates streamflow from precipitation and potential
evapotranspiration (PET) alone, without requiring observed discharge as
an input (unlike the statistical/ML models in this package, which learn
from historical streamflow). Calibration — finding the 4 parameters that
best reproduce an observed discharge record — is a separate step handled
by :func:`calibrate`.

References
----------
Perrin, C., Michel, C., & Andreassian, V. (2003). Improvement of a
    parsimonious model for streamflow simulation. Journal of Hydrology,
    279(1-4), 275-289.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard GR4J parameter bounds from the literature (Perrin et al. 2003;
# airGR package defaults). X1/X3 in mm, X2 in mm/day, X4 in days.
GR4J_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "X1": (1.0, 1500.0),   # production store capacity (mm)
    "X2": (-10.0, 5.0),    # groundwater exchange coefficient (mm/day)
    "X3": (1.0, 500.0),    # routing store capacity (mm)
    "X4": (0.5, 10.0),     # unit hydrograph time base (days)
}


@dataclass
class GR4JResult:
    """Result of a GR4J simulation.

    Attributes
    ----------
    streamflow : pd.Series
        Simulated total streamflow (mm/day), aligned with the input index.
    production_store : pd.Series
        Production store level S (mm) at the end of each day.
    routing_store : pd.Series
        Routing store level R (mm) at the end of each day.
    params : dict
        The GR4J parameters (X1-X4) used for this simulation.
    """

    streamflow: pd.Series
    production_store: pd.Series
    routing_store: pd.Series
    params: dict[str, float]


class GR4J:
    """GR4J lumped conceptual rainfall-runoff model.

    A 4-parameter daily model with a production store, a unit-hydrograph
    routing scheme (90% via UH1/UH2 with a groundwater exchange term, 10%
    via UH2 alone), and a non-linear routing store.

    Parameters
    ----------
    x1 : float
        Production store capacity (mm). Typical range 1-1500.
    x2 : float
        Groundwater exchange coefficient (mm/day). Can be negative
        (water loss to deep aquifers) or positive (gain). Typical
        range -10 to 5.
    x3 : float
        Routing store capacity (mm). Typical range 1-500.
    x4 : float
        Unit hydrograph time base (days). Typical range 0.5-10.

    References
    ----------
    Perrin et al. (2003).
    """

    def __init__(self, x1: float = 350.0, x2: float = 0.0, x3: float = 90.0, x4: float = 1.7):
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.x3 = float(x3)
        self.x4 = float(x4)

    @property
    def params(self) -> dict[str, float]:
        """Current parameter set as a dict."""
        return {"X1": self.x1, "X2": self.x2, "X3": self.x3, "X4": self.x4}

    def _unit_hydrographs(self) -> tuple[np.ndarray, np.ndarray]:
        """Build the discrete UH1 and UH2 ordinates from X4.

        UH1 spreads 90% of effective rainfall over ceil(X4) days using
        an S-curve based on a power-law of exponent 2.5. UH2 spreads the
        remaining flow (and the full delayed branch) over ceil(2*X4)
        days using the same S-curve shape, doubled in length.
        """
        x4 = self.x4

        def s_curve_1(t: float) -> float:
            if t <= 0:
                return 0.0
            if t >= x4:
                return 1.0
            return (t / x4) ** 2.5

        def s_curve_2(t: float) -> float:
            if t <= 0:
                return 0.0
            if t >= 2 * x4:
                return 1.0
            if t <= x4:
                return 0.5 * (t / x4) ** 2.5
            return 1.0 - 0.5 * (2 - t / x4) ** 2.5

        n1 = int(np.ceil(x4))
        n2 = int(np.ceil(2 * x4))

        uh1 = np.array([s_curve_1(t) - s_curve_1(t - 1) for t in range(1, n1 + 1)])
        uh2 = np.array([s_curve_2(t) - s_curve_2(t - 1) for t in range(1, n2 + 1)])

        # Guard against tiny negative values from floating-point error at
        # the boundary, and renormalize so each UH sums to 1.
        uh1 = np.clip(uh1, 0, None)
        uh2 = np.clip(uh2, 0, None)
        if uh1.sum() > 0:
            uh1 = uh1 / uh1.sum()
        if uh2.sum() > 0:
            uh2 = uh2 / uh2.sum()
        return uh1, uh2

    def simulate(
        self,
        precip: pd.Series,
        pet: pd.Series,
        warmup_days: int = 365,
    ) -> GR4JResult:
        """Run the GR4J model over a precipitation/PET time series.

        Parameters
        ----------
        precip : pd.Series
            Daily precipitation (mm/day), with a DatetimeIndex.
        pet : pd.Series
            Daily potential evapotranspiration (mm/day), aligned with
            *precip*.
        warmup_days : int
            Number of initial days used only to spin up store states;
            included in the returned series but typically excluded from
            skill-metric evaluation by the caller.

        Returns
        -------
        GR4JResult
            Simulated streamflow and store trajectories.
        """
        p = precip.values.astype(float)
        e = pet.values.astype(float)
        n = len(p)
        idx = precip.index

        x1, x2, x3 = self.x1, self.x2, self.x3
        uh1, uh2 = self._unit_hydrographs()
        n1, n2 = len(uh1), len(uh2)

        # Initial store levels at 30% capacity (airGR convention).
        s = 0.3 * x1
        r = 0.3 * x3

        # UH convolution buffers.
        uh1_state = np.zeros(n1)
        uh2_state = np.zeros(n2)

        streamflow = np.zeros(n)
        s_trace = np.zeros(n)
        r_trace = np.zeros(n)

        for t in range(n):
            pn, en = self._net_rainfall_pet(p[t], e[t])

            # Production store update.
            if pn > 0:
                ps = self._production_store_input(pn, s, x1)
                s += ps
                perc_input = pn - ps
            else:
                es = self._production_store_evap(en, s, x1)
                s -= es
                perc_input = 0.0

            # Percolation from the production store.
            perc = s * (1 - (1 + (4.0 / 9.0 * s / x1) ** 4) ** -0.25)
            s -= perc

            pr = perc + perc_input

            # Split: 90% through UH1+UH2 with exchange, 10% through UH2 only.
            uh1_state = np.roll(uh1_state, -1)
            uh1_state[-1] = 0.0
            uh1_state += 0.9 * pr * uh1

            uh2_state = np.roll(uh2_state, -1)
            uh2_state[-1] = 0.0
            uh2_state += 0.1 * pr * uh2

            q9 = uh1_state[0]
            q1 = uh2_state[0]

            # Groundwater exchange term (applied to both branches' store).
            exch = x2 * (r / x3) ** 3.5

            r = max(0.0, r + q9 + exch)
            qr = r * (1 - (1 + (r / x3) ** 4) ** -0.25)
            r -= qr

            qd = max(0.0, q1 + exch)

            streamflow[t] = qr + qd
            s_trace[t] = s
            r_trace[t] = r

        return GR4JResult(
            streamflow=pd.Series(streamflow, index=idx, name="streamflow"),
            production_store=pd.Series(s_trace, index=idx, name="production_store"),
            routing_store=pd.Series(r_trace, index=idx, name="routing_store"),
            params=self.params,
        )

    @staticmethod
    def _net_rainfall_pet(p: float, e: float) -> tuple[float, float]:
        """Net rainfall (Pn) and net PET (En) for one day."""
        if p >= e:
            return p - e, 0.0
        return 0.0, e - p

    @staticmethod
    def _production_store_input(pn: float, s: float, x1: float) -> float:
        """Water added to the production store on a net-rainfall day."""
        ratio = s / x1
        tanh_term = np.tanh(pn / x1)
        return (x1 * (1 - ratio**2) * tanh_term) / (1 + ratio * tanh_term)

    @staticmethod
    def _production_store_evap(en: float, s: float, x1: float) -> float:
        """Water removed from the production store on a net-PET day."""
        ratio = s / x1
        tanh_term = np.tanh(en / x1)
        return (s * (2 - ratio) * tanh_term) / (1 + (1 - ratio) * tanh_term)


@dataclass
class CalibrationResult:
    """Result of calibrating a GR4J model against observed discharge.

    Attributes
    ----------
    params : dict
        Best-fit parameters (X1-X4).
    objective_value : float
        Final objective function value (the metric being optimized,
        negated internally for minimization — reported here in its
        natural sign, e.g. KGE = 0.85 means a good fit).
    objective_name : str
        Name of the objective used ("nse", "kge", or "log_nse").
    n_iterations : int
        Number of optimizer iterations/generations completed.
    simulated : pd.Series
        Streamflow simulated with the best-fit parameters over the
        full (warmup + evaluation) period.
    """

    params: dict[str, float]
    objective_value: float
    objective_name: str
    n_iterations: int
    simulated: pd.Series


def calibrate(
    precip: pd.Series,
    pet: pd.Series,
    observed: pd.Series,
    objective: str = "kge",
    warmup_days: int = 365,
    param_bounds: dict[str, tuple[float, float]] | None = None,
    seed: int | None = 42,
    maxiter: int = 100,
) -> CalibrationResult:
    """Calibrate GR4J parameters against observed discharge.

    Uses ``scipy.optimize.differential_evolution`` (a global optimizer,
    avoiding a SCE-UA dependency) to find the X1-X4 parameter set that
    maximizes the chosen objective function over the post-warmup period.

    Parameters
    ----------
    precip : pd.Series
        Daily precipitation (mm/day).
    pet : pd.Series
        Daily potential evapotranspiration (mm/day).
    observed : pd.Series
        Observed daily discharge (mm/day), aligned with *precip*/*pet*.
    objective : str
        One of ``"nse"``, ``"kge"``, ``"log_nse"``. Higher is better for
        all three; the optimizer internally minimizes the negative.
    warmup_days : int
        Initial days excluded from the objective calculation (but still
        simulated, to spin up store states).
    param_bounds : dict, optional
        Override bounds for any of X1-X4; merged with
        :data:`GR4J_PARAM_BOUNDS` defaults.
    seed : int, optional
        Random seed for reproducibility.
    maxiter : int
        Maximum optimizer generations.

    Returns
    -------
    CalibrationResult
    """
    from scipy.optimize import differential_evolution

    from aquascope.analysis import metrics as metrics_module

    objective_fns = {
        "nse": metrics_module.nse,
        "kge": metrics_module.kge,
        "log_nse": metrics_module.log_nse,
    }
    if objective not in objective_fns:
        raise ValueError(f"Unknown objective '{objective}'. Choose from {list(objective_fns)}.")
    objective_fn = objective_fns[objective]

    bounds = dict(GR4J_PARAM_BOUNDS)
    if param_bounds:
        bounds.update(param_bounds)
    order = ["X1", "X2", "X3", "X4"]
    bounds_list = [bounds[k] for k in order]

    obs_eval = observed.values[warmup_days:]

    def neg_objective(x: np.ndarray) -> float:
        model = GR4J(x1=x[0], x2=x[1], x3=x[2], x4=x[3])
        result = model.simulate(precip, pet, warmup_days=warmup_days)
        sim_eval = result.streamflow.values[warmup_days:]
        score = objective_fn(obs_eval, sim_eval)
        if np.isnan(score):
            return 1e6  # heavily penalize degenerate parameter sets
        return -score

    opt_result = differential_evolution(
        neg_objective,
        bounds=bounds_list,
        seed=seed,
        maxiter=maxiter,
        polish=True,
        tol=1e-6,
    )

    best_params = dict(zip(order, opt_result.x, strict=True))
    best_model = GR4J(**{k.lower(): v for k, v in best_params.items()})
    best_sim = best_model.simulate(precip, pet, warmup_days=warmup_days)

    return CalibrationResult(
        params=best_params,
        objective_value=float(-opt_result.fun),
        objective_name=objective,
        n_iterations=int(opt_result.nit),
        simulated=best_sim.streamflow,
    )


# ── Uncertainty quantification ────────────────────────────────────────────


@dataclass
class GR4JProbabilisticResult:
    """Probabilistic GR4J prediction.

    Attributes
    ----------
    median : pd.Series
        Central (q=0.5) streamflow estimate (mm/day).
    quantiles : dict[float, pd.Series]
        Predictive quantile bands keyed by quantile level (e.g. 0.05,
        0.95), each a streamflow series aligned with the input index.
        Bands are non-negative and monotonic in the quantile level.
    params : dict
        Calibrated GR4J parameters (X1-X4) underlying the prediction.
    method : str
        How the bands were produced: ``"residual"`` or ``"ensemble"``.
    """

    median: pd.Series
    quantiles: dict[float, pd.Series]
    params: dict[str, float]
    method: str


def residual_quantile_bands(
    simulated: pd.Series,
    observed: pd.Series,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    *,
    warmup_days: int = 0,
    heteroscedastic: bool = False,
    eps: float = 1e-6,
) -> dict[float, pd.Series]:
    """Predictive quantile bands from the empirical residual distribution.

    Fits residuals ``obs - sim`` over the post-warmup period, then offsets the
    simulated series by their quantiles. With ``heteroscedastic=True`` the
    residuals are taken relative to flow magnitude — ``(obs - sim) / (sim +
    eps)`` — so bands widen at high flows, which better matches streamflow
    error structure. Bands are clipped to be non-negative and sorted so that
    lower quantiles never exceed higher ones at any time step.
    """
    sim = simulated.astype(float)
    obs = observed.astype(float)
    sim_eval = sim.values[warmup_days:]
    obs_eval = obs.values[warmup_days:]
    mask = np.isfinite(sim_eval) & np.isfinite(obs_eval)
    se, oe = sim_eval[mask], obs_eval[mask]
    if len(se) == 0:
        raise ValueError("No finite paired (simulated, observed) values for residuals.")

    qs = sorted(quantiles)
    sim_all = sim.values
    if heteroscedastic:
        rel = (oe - se) / (se + eps)
        offsets = [np.quantile(rel, q) * (sim_all + eps) for q in qs]
    else:
        resid = oe - se
        offsets = [np.full_like(sim_all, np.quantile(resid, q)) for q in qs]

    stacked = np.clip(np.vstack([sim_all + off for off in offsets]), 0.0, None)
    # Enforce monotonicity across quantile levels per time step (clipping at 0
    # can otherwise tie or invert the lowest bands).
    stacked = np.sort(stacked, axis=0)
    return {
        q: pd.Series(stacked[i], index=sim.index, name=f"q{q:g}")
        for i, q in enumerate(qs)
    }


def _parameter_ensemble_bands(
    best_params: dict[str, float],
    precip: pd.Series,
    pet: pd.Series,
    quantiles: Sequence[float],
    n_members: int,
    perturb: float,
    seed: int | None,
    warmup_days: int,
) -> dict[float, pd.Series]:
    """Quantile bands from an ensemble of parameter-perturbed simulations.

    Each member adds Gaussian noise (sigma = ``perturb`` x the parameter's
    bound width) to the calibrated X1-X4, clipped to the GR4J bounds, then
    simulates. Bands are the empirical quantiles across members per time step.
    """
    rng = np.random.default_rng(seed)
    order = ["X1", "X2", "X3", "X4"]
    sims = []
    for _ in range(n_members):
        p = {}
        for k in order:
            lo, hi = GR4J_PARAM_BOUNDS[k]
            val = best_params[k] + rng.normal(0.0, perturb * (hi - lo))
            p[k] = float(np.clip(val, lo, hi))
        model = GR4J(x1=p["X1"], x2=p["X2"], x3=p["X3"], x4=p["X4"])
        sims.append(model.simulate(precip, pet, warmup_days=warmup_days).streamflow.values)
    ens = np.vstack(sims)
    idx = precip.index
    return {
        q: pd.Series(np.clip(np.quantile(ens, q, axis=0), 0.0, None), index=idx, name=f"q{q:g}")
        for q in sorted(quantiles)
    }


def predict_quantiles(
    precip: pd.Series,
    pet: pd.Series,
    observed: pd.Series,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    *,
    method: str = "residual",
    objective: str = "kge",
    warmup_days: int = 365,
    heteroscedastic: bool = True,
    n_members: int = 50,
    perturb: float = 0.05,
    seed: int | None = 42,
    maxiter: int = 100,
) -> GR4JProbabilisticResult:
    """Calibrate GR4J and produce calibrated predictive quantile bands.

    Two methods:

    - ``"residual"`` (default): the deterministic simulation offset by
      empirical residual quantiles (see :func:`residual_quantile_bands`).
      Fast, and the natural citable path for adding UQ to GR4J.
    - ``"ensemble"``: an ensemble of parameter-perturbed simulations
      (see :func:`_parameter_ensemble_bands`). Captures parameter
      uncertainty but is ``n_members`` times more expensive.

    The deterministic :meth:`GR4J.simulate` path is unchanged; UQ is opt-in.
    """
    if method not in ("residual", "ensemble"):
        raise ValueError(f"method must be 'residual' or 'ensemble'; got {method!r}.")

    cal = calibrate(
        precip,
        pet,
        observed,
        objective=objective,
        warmup_days=warmup_days,
        seed=seed,
        maxiter=maxiter,
    )
    best = GR4J(**{k.lower(): v for k, v in cal.params.items()})
    sim = best.simulate(precip, pet, warmup_days=warmup_days).streamflow

    qs = sorted(quantiles)
    if method == "residual":
        bands = residual_quantile_bands(
            sim, observed, qs, warmup_days=warmup_days, heteroscedastic=heteroscedastic
        )
    else:
        bands = _parameter_ensemble_bands(
            cal.params, precip, pet, qs, n_members, perturb, seed, warmup_days
        )

    median = bands.get(0.5, sim)
    return GR4JProbabilisticResult(
        median=median, quantiles=bands, params=cal.params, method=method
    )
