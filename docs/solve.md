# Solve: a problem at a place, planned first, checked at every step

`aquascope ask` answers a question. `aquascope solve` takes a *problem* at a
*location* ("design flow for a road crossing, 100-year return period", "what
flow can this ungauged stream give an irrigation scheme", "is the water table
under this well falling") and goes from there to a verified answer through a
plan you see before anything runs. The design contract is in
[solve-design.md](solve-design.md); this page is the user's guide.

## The flow

```
intake ──► recon ──► plan ──► review ──► execute ──► report
(the text,  (assess_  (a playbook  (you see   (aquascope   (answer + what it
 --intake)   site)     branch)      the plan)  run, a gate  does not establish
                                               per step,    + study.yaml)
                                               replan once)
```

1. **Intake.** The problem text and the coordinates. A playbook's intake
   fields (return period, what is being decided, ...) are read off the text
   where they are stated, or given with `--intake key=value`.
2. **Recon.** `assess_site(lat, lon)` says which records exist within reach,
   for how long, at what resolution, what the catchment looks like and how
   many donor gauges there are, and grades every method in the registry
   (`aquascope.methods`) as defensible, marginal or not defensible here.
3. **Plan.** The playbook's decision tree picks a branch for the data that
   exists and fills a **study** (version 2): steps with arguments, a
   rationale each, the gates each must pass, and a fallback for when a gate
   fails. No model is needed for this.
4. **Review.** The CLI prints the plan as a numbered checklist and asks
   `y/N`; the API gives it to a callback that may edit it or decline it.
5. **Execute.** `run_study` runs the steps in order and evaluates the gates
   after each one. A failed gate runs the step's fallback once, or stops the
   study with the reason. When a model is present, a Specialist may propose
   one more fallback step; the replan is bounded (`max_replans=1`).
6. **Report.** The answer, the plan and its rationale, every step with its
   gate outcomes, "what this answer does not establish", the playbook's
   caveats verbatim, Data and Methods assembled from the tool results, and
   the executed study, which `aquascope run study.yaml` reproduces with no
   model at all.

## A CLI example

```bash
aquascope playbooks                       # the playbooks and their branches
aquascope playbooks show flood_risk       # intake, branches, gates, declines, caveats

aquascope solve "Design flow for a road crossing, 100-year return period" \
    --lat 51.415 --lon -0.308 --playbook flood_risk \
    --out kingston.md --study kingston.yaml
```

The plan printed for the Thames at Kingston (39 years of daily discharge,
so the `at_site` branch):

```
Plan: playbook flood_risk, branch at_site, 3 step(s)
  39.5 years of daily discharge at Kingston (uk_ea 3400TH, 0.4 km from the point): an at-site
  frequency fit is defensible. ...
  1. describe_catchment(lat=51.415, lon=-0.308)
  2. analyze_station(source='uk_ea', station_id='3400TH')
     gate min_years 20 on years
     gate not_empty on trend
  3. flood_frequency(source='uk_ea', station_id='3400TH', bootstrap_ci=True)
     gate max_return_period_factor 3 on years
     gate ci_finite on ffa.fits.gev_bootstrap.ci
     gate spread_within 0.25 on ffa.fits.gev_lmoments.q, ffa.fits.lp3.q
     fallback: similar_basins
Run this plan? [y/N]
```

`--yes` skips the question (scripts, notebooks). `--intake return_period=200`
sets an intake field. `--quiet` hides the timeline. The report lands in
`--out`, the executed study in `--study`; `aquascope run kingston.yaml`
re-runs it and prints the same gate outcomes.

Without `--lat`/`--lon`, `aquascope solve` is the older challenge agent over a
data file, unchanged.

## The playbooks

A playbook is a YAML file under `aquascope/playbooks/`: data, not code. Each
has intake fields, branches over the reconnaissance (first match wins),
study-v2 steps with gates and fallbacks, the sentences it prints when it
declines, caveats printed verbatim in every report, and citations. The
thresholds (20 years, T at most three times the record, three donors, the
10,000 km2 ceiling for a lumped model) are the registry's, not the file's:
a step that names its `method` is checked against `aquascope.methods` when
the plan is filled, so a method the registry calls not defensible at this
site is refused before anything runs.

| Playbook | Branches | Declines |
| --- | --- | --- |
| `flood_risk` | `at_site` (20+ years of discharge: Mann-Kendall pre-test, GEV by L-moments and Log-Pearson III with a bootstrap band, the spread quoted; a stationary estimate with a climate caveat, never a nonstationary fit), `short_record` (8 to 20 years: marginal at-site numbers next to donors and regionalised signatures), `regional` (no gauge: donors, signatures, a GloFAS cross-check) | a return period beyond about three times the record with fewer than three donors; inundation extent (out of scope) |
| `ungauged_flow` | `at_gauge` (a gauge with 5+ years nearby: its flow-duration curve beside the regionalised signatures for the point), `regional` (donors, signatures with the band and the leave-one-out skill, GloFAS) | fewer than three donor gauges |
| `groundwater_decline` | `well` (10+ years of levels: Sen's slope with Mann-Kendall, the Standardised Groundwater Index, water-table-fluctuation recharge with a stated specific yield), `regional` (no well: the ERA5 water balance for the cell, labelled regional) | attributing the cause without pumping data |
| `drought_status` | `gauge_indices` (a rain gauge with 30+ years and ERA5 temperature reachable: SPI and SPEI at the intake timescales, default 1, 3 and 12 months, the divergence between them and the temperature trend behind it), `gauge_indices_marginal` (20 to 30 years, classes indicative), `gauge_spi_only` (no temperature: SPI with the caveat that SPEI is preferable under warming), `reanalysis` (no rain gauge: both indices from the ERA5 cell over 40 years); every branch adds, when the record exists, the low-flow context at the nearest discharge gauge and the SGI at the nearest well with the SPI-to-SGI propagation lag in months | a flash-drought question (no sub-monthly index; this playbook sees monthly droughts) |
| `supply_reliability` | `gauged` (10+ years of discharge: Q95, Q50, Q10, baseflow index and 7Q10, then the screening rule on the daily record, Q95 kept in the river and at most the abstraction share taken, as the fraction of days, of years without a shortfall and of the volume the demand is met), `gauged_short` (5 to 10 years: the same, marginal, with the regional estimate beside it), `regional` (no gauge: donors, Q95, median and Q05 transferred with band and skill over the upstream area, the reliability read off them as a band) | a scheme with a reservoir (a storage-yield analysis it does not do); a missing demand; fewer than three donors |
| `irrigation_feasibility` | `with_gauge` (ERA5 climate of the point, the crop's seasonal demand from FAO-56 single Kc on ERA5 ET0 averaged over the seasons in the window, then the supply screening at the gauge within reach on the peak-month demand over the season's months, the demand read from the previous step's result), `demand_only` (no gauge: the demand, with the note that supply was not checked) | a day-by-day schedule (needs soil parameters and the season's weather) |
| `water_quality` | `drinking` (a station with sampled parameters within reach: the last five years of samples, the WHO 2022 screen, the CCME WQI 1.0 against the WHO guidelines and the NSF WQI when its nine parameters are present), `irrigation` (the samples, the CCME index against FAO 29 thresholds, and SAR, sodium percentage, RSC with the FAO degree of restriction), `aquatic_life` (the samples and the CCME index against the CCME freshwater guidelines) | no sampled parameters within reach (the archive carries no water-quality variables until Phase 3, #188; samples are fetched from USGS and the Water Quality Portal in the US); a health verdict beyond the sampled parameters |

### Writing a playbook

Copy one of the three and keep to the block-style YAML subset the browser
worker reads (nested mappings and lists, quoted or bare scalars, `>-` block
text; no anchors). The schema:

```yaml
id: my_problem                 # the file name without .yaml
title: ...
problem: my_problem            # the problem kind the registry knows
variable: discharge            # picks the station for {{ station.* }}
intake:                        # types: int, float, str, bool, choice (with options), list
  - {name: return_period, label: Return period (years), type: int, default: 100}
  - {name: timescales, label: Timescales (months), type: list, default: [1, 3, 12]}
branches:                      # first match wins; conditions over the recon dict
  - id: at_site
    when: [{path: context.years_by_variable.discharge, op: ">=", value: 20}]
    station_variable: discharge
    rationale: >-              # placeholders: {{ intake.x }}, {{ station.source }},
      ...                      #   {{ station.station_id }}, {{ site.lat }}, {{ site.lon }}, {{ derived.x }}
    steps:
      - id: s1
        tool: analyze_station  # any Analyst tool, workbench analysis, or assess_site
        method: at_site_flood_frequency   # optional: checked against aquascope.methods at plan time
        optional: false        # true: dropped with a note when the registry says not defensible
        arguments: {source: "{{ station.source }}", station_id: "{{ station.station_id }}"}
        expects:               # gates, see aquascope.gates.CHECKS
          - {check: min_years, value: 20, path: years}
        fallback: {step: {tool: similar_basins, arguments: {...}}}   # or {branch: regional} or stop
        depends_on: []
      - id: s2
        tool: drought_propagation
        station_variable: groundwater_level   # this step's {{ station.* }} is the nearest well, not the branch's gauge
        optional: true
        arguments: {source: "{{ station.source }}", station_id: "{{ station.station_id }}", lat: "{{ site.lat }}", lon: "{{ site.lon }}"}
      - id: s3
        tool: supply_reliability
        depends_on: [s1]
        arguments: {source: "{{ station.source }}", station_id: "{{ station.station_id }}",
                    demand_m3s: "{{ result.s1.demand.peak_month_m3s }}"}   # a number an earlier step computed; the runner fills it
declines:
  - {when: [...], say: "the sentence printed verbatim"}
caveats: ["always", {say: "only when", when: [...]}]
citations: ["..."]
```

Conditions take `==`, `!=`, `>=`, `<=`, `>`, `<`, `in`, `exists`, and are
evaluated over the recon dict extended with `intake`, `station`, `site` and
`derived` (`discharge_years`, `groundwater_years`, `precipitation_years`,
`has_temperature`, `donors`, `dams`, `return_period_cap`,
`return_period_beyond_cap`, `area_km2`). A workbench step takes
`from_step: s2` to run on the series a previous `get_timeseries` step
fetched; a step that needs a number an earlier step computed writes
`{{ result.s2.demand.peak_month_m3s }}` (a dotted path into that step's
payload), which the plan leaves in place and the runner fills, and lists the
step in `depends_on`. The site-level tools a playbook can call with what a
plan knows (`drought_indices`, `drought_propagation`, `low_flow_context`,
`supply_reliability`, `crop_water_demand`) live in `aquascope/problems.py`.
`aquascope.playbooks.validate("my_problem")` lists every authoring mistake;
`tests/test_playbooks.py` shows the fixtures (gauged long, gauged short,
rain gauge with and without temperature, well, ungauged) a new playbook
should be exercised on.

## What keyless gives, what a key adds

With no key, no role calls a model: the Scout runs the reconnaissance, the
tree fills the plan, the Reviewer evaluates the gates and the deterministic
checks, and a template Narrator writes the answer from the results. That is
a complete, reproducible run, and it is what the MCP tools and the browser
do by default.

With `--provider` (or `--model`, `--api-key`, `--base-url`), three roles use
the model, each as a stateless subcall that sees only its own inputs, never a
transcript: the Coordinator writes a one-paragraph rationale for the plan
(and settles a problem the keyword rules cannot place), the Specialist
proposes one fallback step after a failed gate, and the Narrator writes the
prose under the analyst's rules (units, records named, no invented numbers).
The report's footer counts the calls and tokens per role. `solve` is keyless
unless you ask for a model, even when a key is in the environment.

## The honesty rules

- Every number in the report comes from a tool result; the gates checked it
  before it was quoted, and the deterministic checks of `aquascope ask`
  (numbers present in the results, units named, records named, intervals
  with return levels, significance wording matching the test) run on the
  prose as well. What fails is listed under "what this answer does not
  establish", never hidden.
- A playbook that declines prints its own sentence and stops. So does a plan
  whose method the registry calls not defensible at this site (the GR4J on a
  100,000 km2 catchment of #273 is refused before it runs).
- The caveats are the playbook's, verbatim, in every report: design-flood
  guidance under climate change is immature (Wasko et al. 2024), a transferred
  number without its band and skill is not an estimate, a trend says whether
  a level is changing and never why.
- The executed study is the answer's receipt and its plan at once:
  `aquascope run study.yaml` reproduces it with no model in the loop.

## Measuring an agent on it

The playbooks double as a benchmark: [hydrogym.md](hydrogym.md) generates
tasks (a problem at a real site, the tree's branch, gates or decline as the
key, unsolvable tasks included) and scores the plan-first team, the older
`ask` loop, or any agent against them (`aquascope gym tasks | bench |
leaderboard`).

## Using the team from LangGraph or your own orchestrator

The roles are plain functions with no framework between them: the Scout is
`aquascope.explore.assess_site`, the Coordinator and the tree are
`aquascope.ai_engine.team.solve(execute=False)` (or `aquascope.playbooks.plan`
when you already hold the recon), the review is whatever you put there, and
the runner with its gates, the Reviewer and the Narrator are
`aquascope.ai_engine.team.run_reviewed`. The browser runs these same functions
in a worker with no agent framework at all, which is why the package does not
depend on one; it also means they drop into the orchestrator your stack
already has.

`examples/langgraph_team.py` is that mapping for LangGraph, one node per role:

| Team role | Graph node | Calls |
| --- | --- | --- |
| Scout | `scout` | `assess_site(lat, lon, problem=...)` |
| Coordinator (tree, keyword rules) | `plan` | `solve(..., recon=recon, execute=False)` |
| You | `review` | LangGraph's `interrupt()` with the plan; resume with the study (edited or not) or `None` to decline |
| Runner, Reviewer, Narrator | `run` | `run_reviewed(study, recon=recon)` |
| Report | `report` | the Markdown report, or the decline in the playbook's words |

A declined plan (a decline rule, no branch, a refused method) skips the review
and the run and goes to the report. The graph needs a checkpointer for the
interrupt to pause; the example uses the in-memory one. Run it keyless:

```bash
pip install langgraph langchain-core     # not aquascope dependencies
python examples/langgraph_team.py --lat 51.415 --lon -0.308 --playbook flood_risk
```

The same mapping holds for CrewAI-style roles: a scout, a planner, a reviewer
and a runner agent whose tools are these four functions, in that order, with
the study as the artefact each hands to the next.

## The other faces

Over MCP (`aquascope mcp`): `list_playbooks()`, `describe_playbook(id)`,
`solve_plan(problem, lat, lon, playbook, intake)` returns the study to
review, `solve_run(study)` executes it. The Analyst (`aquascope ask`) has
the same four tools, so a question that turns out to be a problem at a place
can be handed to Solve. In Python:

```python
from aquascope.ai_engine.team import solve

result = solve("Design flow for a road crossing, 100-year return period",
               lat=51.415, lon=-0.308, playbook="flood_risk",
               review=lambda study: study)         # or edit it, or return None
print(result.answer)
print(result.to_markdown())
open("study.yaml", "w").write(result.study_yaml)   # aquascope run study.yaml
```
