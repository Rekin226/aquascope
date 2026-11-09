"""Reference agents for AquaScope Bench and a small leaderboard helper.

Three baselines every new agent should beat, ordered by how much of the
simulator they use outside the environment:

* :func:`random_search`: uniform draws from the action space, one per step.
  Uses only ``env.step``.
* :func:`nelder_mead`: scipy's Nelder-Mead from the GR4J defaults, every
  function evaluation is one ``env.step`` (clipped to the bounds). Uses only
  ``env.step``.
* :func:`differential_evolution`: aquascope's own ``calibrate`` (scipy DE)
  runs on the calibration period with unlimited simulator calls, and each
  generation's best member is submitted as one step, so the env sees the
  learning curve. This is the "you have the simulator, use it" baseline.

:func:`run_leaderboard` plays each agent on each basin (fresh episode per
pair) and tabulates the best calibration reward, the validation metrics at
those parameters, steps used and simulator calls.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from aquascope.gym.env import PARAM_NAMES, CalibrationEnv

logger = logging.getLogger(__name__)

Agent = Callable[[CalibrationEnv, dict[str, Any]], dict[str, Any]]


def _finish(env: CalibrationEnv, name: str, t0: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    best = env.best or {}
    return {
        "agent": name, "basin": env.basin.id, "objective": env.objective, "steps": env.step_count,
        "best_reward": best.get("reward"), "best_params": best.get("params"),
        "validation": best.get("validation", {}), "calibration": best.get("calibration", {}),
        "seconds": round(time.perf_counter() - t0, 2), **(extra or {}),
    }


def random_search(env: CalibrationEnv, *, seed: int = 0, steps: int | None = None) -> dict[str, Any]:
    """Uniform random parameter vectors, one per step, until the budget is spent."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    lo = np.array([env.param_bounds[k][0] for k in PARAM_NAMES])
    hi = np.array([env.param_bounds[k][1] for k in PARAM_NAMES])
    n = steps if steps is not None else env.max_steps - env.step_count
    for _ in range(n):
        _, _, _term, trunc, _ = env.step(rng.uniform(lo, hi))
        if trunc:
            break
    return _finish(env, "random_search", t0)


def nelder_mead(env: CalibrationEnv, *, x0: dict[str, float] | None = None, steps: int | None = None) -> dict[str, Any]:
    """scipy Nelder-Mead in unit space; each objective evaluation is one env.step (bounded by clipping)."""
    from scipy.optimize import minimize

    t0 = time.perf_counter()
    budget = steps if steps is not None else env.max_steps - env.step_count
    start = env.to_unit(x0 or {"X1": 350.0, "X2": 0.0, "X3": 90.0, "X4": 1.7})
    calls = {"n": 0}

    class _BudgetError(Exception):
        pass

    def f(u: np.ndarray) -> float:
        if calls["n"] >= budget:
            raise _BudgetError
        calls["n"] += 1
        _, r, _term, trunc, _ = env.step(env.from_unit(u))
        if trunc:
            calls["n"] = budget
        return -r

    try:
        minimize(f, start, method="Nelder-Mead", options={"maxfev": budget, "xatol": 1e-3, "fatol": 1e-4,
                                                            "initial_simplex": _simplex(start)})
    except _BudgetError:
        pass
    return _finish(env, "nelder_mead", t0)


def _simplex(start: np.ndarray, size: float = 0.25) -> np.ndarray:
    pts = [start]
    for i in range(len(start)):
        p = start.copy()
        p[i] = p[i] + size if p[i] + size <= 1 else p[i] - size
        pts.append(p)
    return np.array(pts)


def differential_evolution(env: CalibrationEnv, *, seed: int = 42, maxiter: int | None = None,
                           popsize: int = 10) -> dict[str, Any]:
    """scipy differential evolution over the calibration period with free simulator calls; each generation's best
    is one env.step. Report ``simulator_calls`` next to ``steps`` when you compare it with step-only agents."""
    from scipy.optimize import differential_evolution as _de

    t0 = time.perf_counter()
    gens = maxiter if maxiter is not None else env.max_steps - env.step_count
    bounds = [env.param_bounds[k] for k in PARAM_NAMES]
    calls = {"n": 0}

    def neg(x: np.ndarray) -> float:
        calls["n"] += 1
        cal, _ = env.simulate(x)
        v = cal.get(env.objective)
        return 1e6 if v is None else -float(v)

    def cb(xk: np.ndarray, convergence: float | None = None) -> bool:
        _, _, _term, trunc, _ = env.step(xk)
        return trunc

    _de(neg, bounds=bounds, seed=seed, maxiter=max(gens, 1), popsize=popsize, polish=False, tol=1e-6, callback=cb)
    if env.step_count < env.max_steps and env.history:
        pass  # DE converged before the budget: fine, the best step is on record
    return _finish(env, "differential_evolution", t0, {"simulator_calls": calls["n"]})


BASELINES: dict[str, Agent] = {
    "random_search": lambda env, kw: random_search(env, **kw),
    "nelder_mead": lambda env, kw: nelder_mead(env, **kw),
    "differential_evolution": lambda env, kw: differential_evolution(env, **kw),
}


def run_leaderboard(
    basins: list[Any],
    agents: dict[str, Agent] | list[str] | None = None,
    *,
    objective: str = "nse",
    max_steps: int = 50,
    seeds: tuple[int, ...] = (0,),
    agent_kwargs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Play every agent on every basin (a fresh episode per pair and seed); one row per run.

    Columns: agent, basin, seed, steps, best_reward (calibration objective), val_nse, val_kge, val_pbias,
    simulator_calls (DE only), seconds, and the best X1..X4.
    """
    if agents is None:
        agents = BASELINES
    elif isinstance(agents, list):
        agents = {n: BASELINES[n] for n in agents}
    rows = []
    for basin in basins:
        for name, fn in agents.items():
            for seed in seeds:
                env = CalibrationEnv(basin, objective=objective, max_steps=max_steps)
                env.reset(seed=seed)
                kw = dict((agent_kwargs or {}).get(name, {}))
                if name in ("random_search", "differential_evolution") and "seed" not in kw:
                    kw["seed"] = seed
                res = fn(env, kw)
                val = res.get("validation") or {}
                p = res.get("best_params") or {}
                rows.append({"agent": name, "basin": basin.id, "seed": seed, "steps": res["steps"],
                             "best_reward": res["best_reward"], "val_nse": val.get("nse"), "val_kge": val.get("kge"),
                             "val_pbias": val.get("pbias"), "simulator_calls": res.get("simulator_calls", res["steps"]),
                             "seconds": res["seconds"], **{k: p.get(k) for k in PARAM_NAMES}})
                logger.info("%s on %s (seed %s): best %s %.3f, validation NSE %s", name, basin.id, seed, objective,
                            res["best_reward"] if res["best_reward"] is not None else float("nan"), val.get("nse"))
    return pd.DataFrame(rows)


__all__ = ["BASELINES", "Agent", "differential_evolution", "nelder_mead", "random_search", "run_leaderboard"]
