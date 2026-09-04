# HydroGym: an evaluation environment for hydrologic agents

`aquascope.gym` is Phase 0 of [#175](https://github.com/Rekin226/aquascope/issues/175):
a gym-style environment where an agent gets a real (or synthetic) basin's
data, a simulator (GR4J) and a verifiable reward (NSE, KGE or log-NSE on a
calibration period, with the validation metrics reported next to it). The
recent agent papers each rebuilt this privately; here it is a `pip install`
away, and every gauged basin in the Archive with a catchment area is a task.

```bash
pip install "aquascope[gym]"          # gymnasium is optional; the env also works without it
aquascope gym run --synthetic --agent nelder_mead --steps 30
aquascope gym basins                  # gauged basins from the Archive that make good tasks
aquascope gym run --basin uk_ea/013054a3-670e-49ee-afda-e0865a449197 --objective kge --steps 20
aquascope gym leaderboard --synthetic --n-synthetic 3 --seeds 2 --out board.csv
```

## The environment

```python
from aquascope import gym as hg

basin = hg.synthetic_basin(0)                       # GR4J truth + noise, no network
env = hg.CalibrationEnv(basin, objective="nse", max_steps=40)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step({"X1": 300, "X2": -1, "X3": 80, "X4": 2})
```

* **Episode**: one basin. `reset()` returns a 16-number observation (mean
  precipitation, PET and flow in mm/d, runoff ratio, aridity, Q95 and Q05
  over the mean, lag-1 autocorrelation of flow, record length, the last
  action in unit space, the last reward, the best reward so far, the
  fraction of the budget used; names in `info["obs_names"]`) and the basin
  summary. The daily frame is always there as `env.basin.frame` (`precip`,
  `pet`, `q_obs` in mm/d): agents are meant to look at the data.
* **Action**: the four GR4J parameters within their bounds (`X1` 1 to
  1500 mm, `X2` -10 to 5 mm/d, `X3` 1 to 500 mm, `X4` 0.5 to 10 d), as a dict
  or a vector; out-of-range values are clipped. `unit_actions=True` makes the
  action space the unit cube for RL libraries.
* **Reward**: the objective on the calibration period after a one-year
  warm-up; `info` carries NSE, KGE, log-NSE and PBIAS on the calibration and
  the validation periods, the best step so far and the steps left. Episodes
  are truncated at `max_steps`. `env.evaluate(params)` scores a set without
  spending a step; `env.n_simulations` counts every model run.
* Passes `gymnasium.utils.env_checker.check_env` when gymnasium is installed;
  without it the same class works with array bounds in place of spaces.

Several basins in one env cycle on `reset()` (or `options={"basin": id}`).

## Basins

* `synthetic_basin(seed, years=12, params=None, noise=0.15)`: seasonal
  Markov rain, sinusoidal PET, GR4J with known parameters
  (`basin.meta["true_params"]`) and lognormal noise on the flow. Offline,
  reproducible.
* `load_basin(source, station_id)`: a real gauged basin from the Archive.
  Discharge from the source's bundle (daily means, m3/s to mm/d over the
  agency's catchment area, else BasinATLAS upstream area), precipitation and
  FAO-56 ET0 from Open-Meteo at the gauge point (the ERA5-Land/ERA5 blend the
  Caravan exporter uses), cached as Parquet under the archive cache. Any
  station with a catchment area and archived discharge works; the record is
  split 65/35 by default (`split=` a fraction or a date).
* `suggest_basins(n, sources=None, min_years=15, max_snow_pct=20)`: candidates
  from `basins/station_signatures.parquet` with long perennial records, an
  agency area, a plausible runoff ratio and little snow (GR4J here has no
  snow routine; a snowy basin is a fair way to watch an agent fail: the
  St. John River in Maine calibrates to KGE 0.2 whatever you do).

## Baselines and the leaderboard

Three agents every new one should beat, in order of how much of the
simulator they use outside the environment:

| agent | what it does | simulator use |
|---|---|---|
| `random_search` | uniform draws in the bounds, one per step | `env.step` only |
| `nelder_mead` | scipy Nelder-Mead from the GR4J defaults in unit space; each evaluation is one step | `env.step` only |
| `differential_evolution` | scipy DE on the calibration period (aquascope's `calibrate`); each generation's best is one step | free simulator calls (reported as `simulator_calls`) |

On the synthetic basin with 40 steps: random search reaches NSE 0.67,
Nelder-Mead 0.95, DE 0.96 against a truth ceiling of 0.96 (1,640 simulator
calls). `run_leaderboard(basins, agents, max_steps, seeds)` plays every
agent on every basin and returns one row per run (best reward, validation
NSE/KGE/PBIAS, steps, simulator calls, seconds, the best parameters);
`aquascope gym leaderboard` prints or saves it as CSV. Your own agent is any
callable `agent(env, kwargs) -> dict` (see `notebooks/08_hydrogym_phase0.ipynb`).

## What Phase 0 is not

One model (GR4J), one task type (calibration), CPU-bound (about 0.03 s per
simulation-year-decade: 12 years take 30 ms, 40 years 100 ms). Phase 1
(#175) is the benchmark of hydrology *agents* on real sites, generated from
the playbooks with unsolvable tasks, a held-out split, cost accounting and a
leaderboard: see [hydrogym.md](hydrogym.md) (`aquascope gym tasks | bench |
leaderboard`). Calibration tasks across regions and data-gap QA tasks remain
open.
