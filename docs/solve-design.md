# Solve: design and contracts

How aquascope goes from "a problem at a location" to a verified answer. This
page is the contract the pieces are built against; the user-facing guide is
[solve.md](solve.md).

## The shape

```
intake ──► recon ──► plan ──► review ──► execute ──► report
(form or   (assess_  (playbook  (the user  (aquascope   (answer + what it
 model)     site)     tree, a    sees the   run, gates   does not establish
                      model may  plan)      per step,    + study.yaml)
                      fill it)              replan once)
```

Every capability is a plain Python function in the package (one engine,
three faces): the CLI, the MCP server and the Explorer's worker call the
same functions. No agent framework: the loop must run inside Pyodide and
within free-tier token budgets, and the value is in the scaffold (playbooks)
and the gates, not in orchestration machinery. The same fact makes the roles
portable: `examples/langgraph_team.py` drives them from a LangGraph
`StateGraph` (one node per role, the review as an interrupt) without the
package gaining a dependency; see [solve.md](solve.md#using-the-team-from-langgraph-or-your-own-orchestrator).

## The team

The Analyst is a team of roles, each a prompt plus a bounded tool set,
sharing one study as the blackboard. Roles run as stateless subcalls: each
sees the study, its own step and its inputs, never the whole transcript.

| Role | Does | Model needed? |
| --- | --- | --- |
| Scout | reconnaissance: `assess_site`, the sufficiency table | no |
| Coordinator | intake → chooses the playbook branch → fills the plan; explains the choice | optional (the tree alone fills a plan; in the Explorer a model already on the reader's device may read the sentence into the intake, through `playbooks.coerce_intake`) |
| Specialist (one per problem kind: flood, ungauged flow, groundwater, drought, supply, climate, irrigation) | interprets a step's result, proposes the fallback when a gate fails | optional |
| Reviewer | evaluates gates, runs the deterministic checks, writes "what this does not establish" | no |
| Narrator | writes the prose of the report from the executed study | optional (a template otherwise) |

Keyless: Scout, tree-filled Coordinator, Reviewer and template Narrator
give a complete run with zero model calls. With a key, the Coordinator and
Specialists add adaptation and the Narrator adds prose. The team's timeline
(who did what, in order) is part of the result so a page can show it.

## Contracts

### Method preconditions (`aquascope/methods.py`)

`METHODS[id] -> MethodPrecondition` (variable, `min_years`, `marginal_years`,
resolution, `max_area_km2`, `max_return_period_factor`, `min_donors`,
`needs`, tool, problems, citation). `assess_method(id, SiteContext)` returns
`{"method", "status": defensible | marginal | not_defensible, "reason"}`;
`sufficiency_table(ctx, problem=...)` returns all rows, defensible first.

### Reconnaissance (`aquascope.explore.assess_site`)

```python
assess_site(lat, lon, *, radius_km=50.0, problem=None, return_period=None) -> dict
```

Returns:

```yaml
point: {lat, lon}
stations:            # nearest first, true catalog spans
  - {source, station_id, name, distance_km, variables: [...], period_start, period_end, years, url}
catchment:           # from describe_catchment, subset
  {area_km2, upstream_area_km2, dams, elevation_m, aridity, ...}
context:             # the SiteContext as a dict (years_by_variable, resolution_by_variable, area_km2, donors, available)
sufficiency:         # sufficiency_table(context, problem)
  - {method, label, status, reason, tool, station: {source, station_id} | null}
notes: ["…"]         # e.g. "catalogue lists this site from 1883; served record starts 1986"
```

Rules: use the catalog only (no agency fetch) for spans; `near` search by
position, never by name; `describe_catchment` may be skipped with a note
when BasinATLAS is unreachable.

### Study v2 (`aquascope/study.py`)

```yaml
version: 2
title: …
question: …
problem: {kind: flood_risk, site: {lat, lon}, params: {return_period: 100}}
plan: {playbook: flood_risk, branch: at_site, rationale: "…"}
steps:
  - id: s1
    tool: analyze_station          # any Analyst tool or workbench analysis
    arguments: {source: uk_ea, station_id: "…", bootstrap_ci: true}
    rationale: "39 years of daily flow: at-site fit is defensible"
    expects:                       # gates, evaluated by the runner after the step
      - {check: min_years, value: 20, path: years}
      - {check: ci_finite, path: ffa.fits.gev_bootstrap.ci}
      - {check: spread_within, value: 0.25, paths: [ffa.fits.gev_lmoments.q100, ffa.fits.lp3.q100]}
    fallback: {step: {tool: similar_basins, arguments: {...}}}   # or {branch: regional} or stop
    depends_on: [s0]
results:                           # written by the runner, one per step
  s1: {ok: true, gates: [{check, passed, detail}], summary: "…", fallback_used: false}
```

Gates live in `aquascope/gates.py`: `evaluate(expects, payload) -> list[dict]`.
Check vocabulary v1: `min_years`, `max_return_period_factor`, `ci_finite`,
`spread_within`, `nse_min` (validation split), `kge_min`, `not_empty`,
`unit_present`, `max_area_km2`, `min_donors`, `status_is` (for sufficiency
rows). Each check has a `path` (dotted, into the tool payload) and a
`value`. Version-1 studies (a list of steps, no `version`) keep running.

`run_study` executes steps in order, evaluates `expects`, and on a failed
gate runs the `fallback` step once (recording `fallback_used`) or stops the
study with the reason. `on_event` receives `{"role", "step", "event", …}`.

### Playbooks (`aquascope/playbooks/*.yaml`)

```yaml
id: flood_risk
title: Flood risk at a site
problem: flood_risk
intake:
  - {name: return_period, label: Return period (years), type: int, default: 100}
  - {name: decision, label: What is being decided, type: choice, options: [design flow, risk screening, insurance], default: design flow}
branches:                          # first match wins; conditions are over the recon dict
  - id: at_site
    when: [{path: context.years_by_variable.discharge, op: ">=", value: 20}]
    steps: [...]                   # study-v2 steps with {{ intake.return_period }} and {{ station.source }} placeholders
  - id: short_record
    when: [{path: context.years_by_variable.discharge, op: ">=", value: 8}]
    steps: [...]
  - id: regional
    when: [{path: context.ungauged, op: "==", value: true}]
    steps: [...]
declines:
  - {when: [...], say: "…"}        # what the playbook refuses and the sentence it prints
caveats: ["…"]                     # printed verbatim in every report
citations: ["…"]
```

`aquascope.playbooks.load(id)`, `list_playbooks()`, `plan(playbook, recon,
intake) -> Study` (the tree fills the study with no model), `validate()`.
Condition operators: `==`, `!=`, `>=`, `<=`, `>`, `<`, `in`, `exists`.

### The team loop (`aquascope/ai_engine/team.py`)

```python
solve(problem_text | dict, *, lat, lon, playbook=None, intake=None,
      provider=None, model=None, api_key=None, base_url=None,
      review=None, on_event=None, max_replans=1) -> SolveResult
```

`SolveResult`: `study` (the executed Study), `run` (StudyRun), `recon`,
`answer` (prose), `checks`, `timeline` (list of `{role, step, event, detail}`),
`declined` (bool + reason), `cost` (tokens per role). `review` is a callback
`(Study) -> Study | None` (None declines); the CLI prints the plan and asks,
the page shows a checklist. With `model=None` no role calls a model.

### Faces

- CLI: `aquascope assess LAT LON [--problem]`, `aquascope playbooks [list|show ID]`,
  `aquascope solve "PROBLEM" --lat --lon [--playbook] [--yes] [--out] [--study]`.
- MCP: `assess_site`, `list_playbooks`, `solve_plan` (returns the study to
  review), `solve_run` (executes a study).
- Explorer worker: `assess`, `solve_plan`, `solve_run` with progress events;
  `src/solve.js` renders intake, the recon card, the plan checklist, the
  timeline and the report.

### Benchmark (`aquascope/gym/tasks.py`)

`tasks_from_playbooks(sites, playbooks) -> list[Task]`: for each site
(catalog row or point) and playbook, the scoring key is the branch the tree
selects plus the gates it expects, and `unsolvable` when every branch
declines. `aquascope gym bench --tasks tasks.jsonl --agent team|ask|tree
--provider … --model …` runs an agent per task and records branch match,
gates respected, declined-when-unsolvable, tokens and seconds; results as
JSONL plus a Markdown leaderboard.

## Implementation notes and deviations

The pieces above are implemented in `aquascope/gates.py`, `aquascope/study.py`
(version 2), `aquascope/playbooks/`, `aquascope/ai_engine/team.py`, the CLI
(`playbooks`, `solve`), the MCP server and the Analyst's tool specs. Where
the implementation goes beyond or beside the contract:

- **Conditions see more than the recon dict.** `when` conditions and
  placeholders are evaluated over the recon extended with `intake`, `station`
  (the nearest station carrying the branch's variable), `site` and `derived`
  (`discharge_years`, `groundwater_years`, `donors`, `dams`,
  `return_period_cap` from the registry's factor, `return_period_beyond_cap`,
  `area_km2`). A decline such as "T beyond three times the record without
  regional information" is not expressible over the raw recon alone.
- **Steps may name their `method`** (a registry id) and be `optional`. At
  plan time the registry is asked whether the method is defensible at the
  site (record length, resolution, area ceiling); a required step that is not
  is refused with the reason (#273), an optional one is dropped with a note.
  What the reconnaissance did not find out (a donor count of `None`, an
  absent `available` set) is left to the run-time gate rather than held
  against the method.
- **Gate paths** take list indexes (`q[5]`, `q.5`) and a selector over lists
  of dicts (`sufficiency[method=gr4j_calibration].status`); `ci_finite` and
  `spread_within` take `return_period` and look the index up in the payload's
  own `return_periods`. `max_return_period_factor` carries `return_period` as
  well as `value` (the factor).
- **Caveats may be conditional** (`{say, when}`) next to plain strings, and
  the plan records the recon's notes (`plan.recon_notes`) apart from its own
  (`plan.notes`, the dropped steps).
- **A workbench step takes `from_step`**: the runner builds the table from the
  `points` or `series` of an earlier step's payload.
- **The runner reuses a prior run** (`run_study(study, prior=run)`): steps
  that succeeded and passed their gates are not fetched again on a replan.
- **A version-1 study stays version 1**: results are written back only into
  a plan (a study with `version: 2` or any v2 field).
- **`solve` is keyless unless asked**: a model is used only when `provider`,
  `model`, `api_key`, `base_url` or a `client` is given, even when a key is
  in the environment; `execute=False` returns the plan without running it
  (the MCP `solve_plan` face). `SolveResult.declined` is a bool with the
  reason in `declined_reason`; `cost` is `{role: {calls, prompt_tokens,
  completion_tokens}}`.
- **`aquascope solve` keeps its older meaning** without `--lat`/`--lon` (the
  challenge agent over a data file); with them it is the team.
- **`describe_playbook`** is exposed next to `list_playbooks` (issue #307).
- **Six playbooks.** `drought_status`, `supply_reliability` and
  `irrigation_feasibility` joined the three exemplars. They needed three
  small extensions of the contract: a `list` intake type (timescales); a
  per-step `station_variable`, so a step's `{{ station.* }}` can be the
  nearest well while the branch's is a rain gauge; and run-time placeholders
  `{{ result.<step>.<path> }}` that the plan leaves in the study and the
  runner resolves from an earlier step's payload (an irrigation demand
  feeding a supply check), with the validator requiring the referenced step
  to be earlier and in `depends_on`. `derived.has_temperature` reads the
  context's `available` set. The site-level tools these playbooks call
  (`aquascope/problems.py`: `drought_indices`, `drought_propagation`,
  `low_flow_context`, `supply_reliability`, `crop_water_demand`) take a point
  or a station and return JSON with their methods, and are MCP tools and
  Analyst tools as well as study steps. The registry gained
  `supply_reliability` and `spei_reanalysis` (ERA5 for the cell, so a
  drought answer exists where no rain gauge does).
- **`assess_site`** is called by the Scout through `aquascope.explore`; when
  it is unreachable the plan goes regional and says why in `recon_notes`.
- **Not done here**: the Explorer worker face, `src/solve.js`, and the gym
  benchmark (`tasks_from_playbooks`) are separate pieces of work.

