"""AquaScope Bench: a gym-style calibration environment over real (or synthetic) basins.

An episode is one basin. The agent proposes a GR4J parameter vector
(X1..X4), the environment runs the model over the basin's whole record and
returns the objective on the calibration period as the reward, with the
full metric set (NSE, KGE, log-NSE, PBIAS on calibration and validation) in
``info``. The observation is a fixed-length summary of the basin (climate,
runoff ratio, low/high-flow ratios, record length) plus the last action,
the last reward, the best reward so far and the fraction of the step budget
used; the raw daily frame is always reachable as ``env.basin.frame`` for
agents that want to look at the data (that is the point).

Follows the gymnasium API (``reset()`` -> ``(obs, info)``,
``step()`` -> ``(obs, reward, terminated, truncated, info)``) and registers
proper ``Box`` spaces when gymnasium is installed (``pip install
aquascope[gym]``); without it the same class works as a plain Python
environment with array bounds in place of spaces.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from aquascope.gym.basins import Basin

logger = logging.getLogger(__name__)

try:  # gymnasium is optional: the env works without it, the spaces are then simple bound arrays
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    _Base = _gym.Env
    HAS_GYMNASIUM = True
except Exception:  # noqa: BLE001 - not installed, or an incompatible numpy
    _gym = None
    _spaces = None
    _Base = object
    HAS_GYMNASIUM = False

PARAM_NAMES = ("X1", "X2", "X3", "X4")
OBJECTIVES = ("nse", "kge", "log_nse")
OBS_NAMES = (
    "mean_precip_mm_d", "mean_pet_mm_d", "mean_q_mm_d", "runoff_ratio", "aridity", "q95_over_mean", "q05_over_mean",
    "q_lag1_autocorr", "n_years", "last_x1_unit", "last_x2_unit", "last_x3_unit", "last_x4_unit", "last_reward",
    "best_reward", "steps_used_fraction",
)


class _BoundsBox:
    """A stand-in for gymnasium.spaces.Box when gymnasium is not installed (same low/high/shape/sample)."""

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low, self.high = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape
        self._rng = np.random.default_rng()

    def sample(self) -> np.ndarray:
        return self._rng.uniform(self.low, self.high).astype(np.float32)

    def contains(self, x: Any) -> bool:
        x = np.asarray(x, dtype=np.float32)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)


def _box(low: np.ndarray, high: np.ndarray):
    if _spaces is not None:
        return _spaces.Box(low=np.asarray(low, dtype=np.float32), high=np.asarray(high, dtype=np.float32),
                           dtype=np.float32)
    return _BoundsBox(low, high)


def _metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    from aquascope.analysis import metrics as m

    ok = ~np.isnan(obs) & ~np.isnan(sim)
    if ok.sum() < 30:
        return {"nse": float("nan"), "kge": float("nan"), "log_nse": float("nan"), "pbias": float("nan"),
                "n": int(ok.sum())}
    o, s = obs[ok], sim[ok]
    out = {"nse": float(m.nse(o, s)), "kge": float(m.kge(o, s)), "log_nse": float(m.log_nse(o, s)),
           "pbias": float(m.pbias(o, s)), "n": int(ok.sum())}
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in out.items()}


class CalibrationEnv(_Base):
    """Calibrate GR4J on a basin, one parameter vector per step; the reward is the objective on the calibration period.

    Parameters
    ----------
    basins : Basin or list of Basin
        One episode per basin. With several, ``reset()`` cycles through them (or takes
        ``options={"basin": i}`` / ``options={"basin": "usgs/USGS-01646500"}``).
    objective : {"nse", "kge", "log_nse"}
        The reward. All the metrics are in ``info`` whatever the choice.
    max_steps : int
        Step budget per episode; the episode is truncated when it runs out.
    warmup_days : int
        Days at the start of the record simulated but not scored.
    param_bounds : dict, optional
        Override any of the GR4J bounds (the action space).
    unit_actions : bool
        If True the action space is the unit cube [0, 1]^4 (what RL libraries like) and actions are
        mapped onto the bounds; otherwise actions are the parameters themselves (X1 in mm, ...).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        basins: Basin | list[Basin],
        *,
        objective: str = "nse",
        max_steps: int = 50,
        warmup_days: int = 365,
        param_bounds: dict[str, tuple[float, float]] | None = None,
        unit_actions: bool = False,
        render_mode: str | None = None,
    ):
        from aquascope.models.rainfall_runoff import GR4J_PARAM_BOUNDS

        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        self.basins: list[Basin] = [basins] if isinstance(basins, Basin) else list(basins)
        if not self.basins:
            raise ValueError("give at least one basin")
        self.objective = objective
        self.max_steps = int(max_steps)
        self.warmup_days = int(warmup_days)
        self.render_mode = render_mode
        bounds = dict(GR4J_PARAM_BOUNDS)
        if param_bounds:
            bounds.update(param_bounds)
        self.param_bounds = {k: (float(bounds[k][0]), float(bounds[k][1])) for k in PARAM_NAMES}
        self.unit_actions = bool(unit_actions)
        low = np.array([self.param_bounds[k][0] for k in PARAM_NAMES], dtype=np.float32)
        high = np.array([self.param_bounds[k][1] for k in PARAM_NAMES], dtype=np.float32)
        self.action_space = _box(np.zeros(4), np.ones(4)) if self.unit_actions else _box(low, high)
        obs_low = np.array([0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -10, -10, 0], dtype=np.float32)
        obs_high = np.array([100, 30, 100, 5, 50, 5, 20, 1, 200, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = _box(obs_low, obs_high)
        self._basin_cursor = -1
        self._np_random = np.random.default_rng()
        self.basin: Basin = self.basins[0]
        self.step_count = 0
        self.history: list[dict[str, Any]] = []
        self.best: dict[str, Any] | None = None
        self.n_simulations = 0  # simulate() calls made through this env (all episodes)
        self._summary_obs = np.zeros(9, dtype=np.float32)

    # ── gym API ─────────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        if HAS_GYMNASIUM:
            super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            try:
                self.action_space.seed(seed)
            except Exception:  # noqa: BLE001
                pass
        pick = (options or {}).get("basin")
        if pick is None:
            self._basin_cursor = (self._basin_cursor + 1) % len(self.basins)
        elif isinstance(pick, int):
            self._basin_cursor = pick % len(self.basins)
        else:
            ids = [b.id for b in self.basins]
            if pick not in ids:
                raise KeyError(f"basin {pick!r} not in this env: {ids}")
            self._basin_cursor = ids.index(pick)
        self.basin = self.basins[self._basin_cursor]
        self.step_count = 0
        self.history = []
        self.best = None
        self._summary_obs = self._basin_summary(self.basin)
        obs = self._observation(last=None, reward=None)
        info = {"basin": self.basin.summary(), "objective": self.objective, "max_steps": self.max_steps,
                "param_bounds": self.param_bounds, "param_names": list(PARAM_NAMES), "obs_names": list(OBS_NAMES),
                "note": "env.basin.frame holds the daily precip/pet/q_obs (mm/d); calibration period ends at "
                        f"{self.basin.split_date.date().isoformat()}"}
        return obs, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.step_count >= self.max_steps:
            raise RuntimeError("episode is over; call reset()")
        params = self._clip(self.from_unit(action) if self.unit_actions and not isinstance(action, dict) else action)
        cal, val = self.simulate(params)
        reward = cal.get(self.objective)
        reward_f = float(reward) if reward is not None else -10.0
        reward_f = max(reward_f, -10.0)  # a floor keeps degenerate parameter sets from wrecking learners
        self.step_count += 1
        rec = {"step": self.step_count, "params": params, "reward": reward_f, "calibration": cal, "validation": val}
        self.history.append(rec)
        if self.best is None or reward_f > self.best["reward"]:
            self.best = rec
        truncated = self.step_count >= self.max_steps
        info = {"params": params, "calibration": cal, "validation": val, "best": self.best,
                "steps_left": self.max_steps - self.step_count}
        return self._observation(last=params, reward=reward_f), reward_f, False, truncated, info

    def render(self) -> str | None:
        b = self.basin
        lines = [f"AquaScope Bench CalibrationEnv: {b.name} ({b.id}), {b.n_years} yr, objective {self.objective}, "
                 f"step {self.step_count}/{self.max_steps}"]
        if self.best:
            p = self.best["params"]
            lines.append(f"  best {self.objective} {self.best['reward']:.3f} at X1={p['X1']:.0f} X2={p['X2']:.2f} "
                         f"X3={p['X3']:.0f} X4={p['X4']:.2f}; validation NSE {self.best['validation'].get('nse')}")
        text = "\n".join(lines)
        if self.render_mode == "ansi" or self.render_mode is None:
            return text
        print(text)
        return None

    def close(self) -> None:
        return None

    # ── helpers agents may call ────────────────────────────────────────────
    def simulate(self, params: dict[str, float] | Any) -> tuple[dict[str, float], dict[str, float]]:
        """Run GR4J with ``params`` over the basin's record; metrics on (calibration, validation), post-warmup."""
        from aquascope.models.rainfall_runoff import GR4J

        p = self._clip(params)
        f = self.basin.frame
        model = GR4J(**{k.lower(): v for k, v in p.items()})
        sim = model.simulate(f["precip"], f["pet"], warmup_days=self.warmup_days).streamflow
        self.n_simulations += 1
        obs = f["q_obs"].to_numpy(dtype=float)
        s = sim.to_numpy(dtype=float)
        before = np.asarray(f.index < self.basin.split_date)
        mask_cal = before.copy()
        mask_cal[: self.warmup_days] = False
        mask_val = ~before
        return _metrics(obs[mask_cal], s[mask_cal]), _metrics(obs[mask_val], s[mask_val])

    def evaluate(self, params: dict[str, float] | Any) -> dict[str, Any]:
        """Score a parameter set on both periods without spending a step."""
        cal, val = self.simulate(params)
        return {"params": self._clip(params), "calibration": cal, "validation": val}

    def sample_action(self) -> np.ndarray:
        return self.action_space.sample()

    def to_unit(self, params: dict[str, float] | Any) -> np.ndarray:
        p = self._clip(params)
        return np.array([(p[k] - lo) / (hi - lo) for k, (lo, hi) in self.param_bounds.items()], dtype=np.float32)

    def from_unit(self, u: Any) -> dict[str, float]:
        u = np.clip(np.asarray(u, dtype=float).ravel(), 0.0, 1.0)
        return {k: float(lo + u[i] * (hi - lo)) for i, (k, (lo, hi)) in enumerate(self.param_bounds.items())}

    # ── internals ───────────────────────────────────────────────────────────
    def _clip(self, action: Any) -> dict[str, float]:
        if isinstance(action, dict):
            vec = [float(action[k]) for k in PARAM_NAMES]
        else:
            vec = [float(x) for x in np.asarray(action, dtype=float).ravel()]
        if len(vec) != 4:
            raise ValueError("action must be four numbers: X1, X2, X3, X4")
        return {k: float(np.clip(v, *self.param_bounds[k])) for k, v in zip(PARAM_NAMES, vec)}

    @staticmethod
    def _basin_summary(basin: Basin) -> np.ndarray:
        cal = basin.calibration
        p, e, q = cal["precip"], cal["pet"], cal["q_obs"]
        qq = q.dropna()
        pm = float(p.mean()) or 1e-6
        qm = float(qq.mean()) if len(qq) else float("nan")
        lag1 = float(qq.autocorr(1)) if len(qq) > 10 else 0.0
        vals = [pm, float(e.mean()), qm, qm / pm, float(e.mean()) / pm,
                float(np.percentile(qq, 5)) / qm if len(qq) and qm > 0 else 0.0,
                float(np.percentile(qq, 95)) / qm if len(qq) and qm > 0 else 0.0,
                lag1 if np.isfinite(lag1) else 0.0, basin.n_years]
        return np.nan_to_num(np.array(vals, dtype=np.float32), nan=0.0)

    def _observation(self, *, last: dict[str, float] | None, reward: float | None) -> np.ndarray:
        unit = self.to_unit(last) if last is not None else np.full(4, -1.0, dtype=np.float32)
        tail = np.array([reward if reward is not None else 0.0, self.best["reward"] if self.best else 0.0,
                         self.step_count / max(self.max_steps, 1)], dtype=np.float32)
        obs = np.concatenate([self._summary_obs, unit, tail]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)


def make(basins: Basin | list[Basin] | str | None = None, **kw: Any) -> CalibrationEnv:
    """Build a CalibrationEnv from Basin objects, ``"synthetic"`` (default), or ``"source/station_id"`` strings."""
    from aquascope.gym.basins import load_basin, synthetic_basin

    if basins is None or basins == "synthetic":
        return CalibrationEnv(synthetic_basin(0), **kw)
    if isinstance(basins, str):
        src, _, sid = basins.partition("/")
        return CalibrationEnv(load_basin(src, sid), **kw)
    return CalibrationEnv(basins, **kw)


def episode_table(env: CalibrationEnv) -> pd.DataFrame:
    """The current episode's history as a table (step, X1..X4, reward, validation NSE/KGE)."""
    rows = []
    for h in env.history:
        rows.append({"step": h["step"], **h["params"], "reward": h["reward"],
                     "cal_nse": h["calibration"].get("nse"), "cal_kge": h["calibration"].get("kge"),
                     "val_nse": h["validation"].get("nse"), "val_kge": h["validation"].get("kge")})
    return pd.DataFrame(rows)


__all__ = ["HAS_GYMNASIUM", "OBJECTIVES", "OBS_NAMES", "PARAM_NAMES", "CalibrationEnv", "episode_table", "make"]
