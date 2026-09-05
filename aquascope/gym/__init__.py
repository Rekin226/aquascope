"""HydroGym: a gym-style evaluation environment for hydrologic agents, on real basins from the Archive (#175).

Phase 0: :class:`CalibrationEnv` wraps GR4J calibration on one basin as an
episode (action = X1..X4, reward = NSE / KGE / log-NSE on the calibration
period, validation metrics in ``info``); :func:`synthetic_basin` for offline
work, :func:`load_basin` for any gauged basin in the Archive with a catchment
area (discharge bundle + Open-Meteo forcing at the gauge), and three
baselines (random search, Nelder-Mead through the env, differential
evolution with the simulator) with a leaderboard helper.

    >>> from aquascope.gym import CalibrationEnv, synthetic_basin
    >>> env = CalibrationEnv(synthetic_basin(0), objective="nse", max_steps=30)
    >>> obs, info = env.reset(seed=0)
    >>> obs, reward, terminated, truncated, info = env.step({"X1": 300, "X2": -1, "X3": 80, "X4": 2})

Phase 1: :mod:`aquascope.gym.tasks` generates a benchmark of problems at real
sites from the playbooks (the tree's branch, gates and declines as the key)
and :mod:`aquascope.gym.bench` plays the ``tree``, ``team`` and ``ask`` agents
on it and renders a leaderboard (``aquascope gym tasks|bench|leaderboard``).
"""

from aquascope.gym.baselines import (
    BASELINES,
    differential_evolution,
    nelder_mead,
    random_search,
    run_leaderboard,
)
from aquascope.gym.basins import Basin, load_basin, suggest_basins, synthetic_basin
from aquascope.gym.bench import Result, leaderboard, run_bench
from aquascope.gym.env import HAS_GYMNASIUM, OBJECTIVES, OBS_NAMES, PARAM_NAMES, CalibrationEnv, episode_table, make
from aquascope.gym.tasks import Task, read_tasks, suggest_sites, tasks_from_playbooks, write_tasks

__all__ = [
    "BASELINES", "HAS_GYMNASIUM", "OBJECTIVES", "OBS_NAMES", "PARAM_NAMES", "Basin", "CalibrationEnv", "Result", "Task",
    "differential_evolution", "episode_table", "leaderboard", "load_basin", "make", "nelder_mead", "random_search",
    "read_tasks", "run_bench", "run_leaderboard", "suggest_basins", "suggest_sites", "synthetic_basin",
    "tasks_from_playbooks", "write_tasks",
]
