# HydroGym Phase 1 tasks: a playbook-keyed benchmark of hydrology agents on real sites

`tasks.jsonl` holds 60 benchmark tasks for hydrology agents, generated
on 2026-09-03 by

```bash
aquascope gym tasks --n 60 --seed 2026 --out tasks.jsonl
```

with [aquascope](https://github.com/Rekin226/aquascope) 0.13 (branch
`feat/solve`, HydroGym Phase 1 of
[issue #175](https://github.com/Rekin226/aquascope/issues/175)). The
benchmark is documented at
<https://rekin226.github.io/aquascope/hydrogym/>; this file describes what is
in the deposit so it can be used without the package.

## What a task is

A task is a problem at a place: one of AquaScope's three method-selection
playbooks (`flood_risk`, `ungauged_flow`, `groundwater_decline`), a site (a
catalog gauge, a borehole, or a bare point), the playbook's intake fields, and
a snapshot of the data reconnaissance at that place taken when the task was
made. The scoring key is what the playbook's decision tree says on that
snapshot with no model in the loop: the branch it selects, the gates the
branch expects, the tools it would call, or the sentence it prints when it
declines.

The suite: 60 tasks on 15 sites (4 gauges with 20 years and more of discharge, 4 with 5 to 20 years, 3 boreholes with 10 years and more of levels, 4 bare points). 45
tasks are solvable and 15 are unsolvable: the right answer is to
refuse. All 15 of the unsolvable ones are *probes*, intakes that trigger a
decline rule of the playbook itself (flood risk asked as an inundation map, a
groundwater decline asked with its cause). The generator also produces
data-driven declines (a bare point with fewer than three donor gauges, a
method the registry refuses at this catchment size) when a site calls for
them; none arose at these 15 sites, see the site table below.

## Task format

One JSON object per line, UTF-8, keys:

| key | what |
| --- | --- |
| `id` | `<playbook>-<8 hex>`, a hash of the playbook, the site and the intake; stable across regenerations |
| `playbook` | `flood_risk`, `ungauged_flow` or `groundwater_decline` |
| `site` | `{lat, lon, source, station_id, name, kind, country, continent, years, variables}`; `source` and `station_id` are `null` for a bare point, whose `anchor` names the gauge it was offset from. `kind` is `gauged_long` (20 years and more of discharge), `gauged_short` (5 to 20), `groundwater` (10 years and more of levels) or `ungauged` |
| `intake` | the playbook's intake with defaults applied, e.g. `{"return_period": 100, "decision": "design flow"}` |
| `problem` | the problem in plain language; what an agent is given, with the coordinates |
| `recon` | the reconnaissance snapshot (`aquascope.explore.assess_site(lat, lon)` at generation time): `point`, `stations` (catalog gauges within 50 km, nearest first: source, id, name, distance, variables, catalog span, years, URL), `catchment` (BasinATLAS attributes of the sub-basin: area, upstream area, elevation, precipitation, aridity, dams), `context` (years and resolution per variable, area, donor count, what is assumed reachable), `sufficiency` (each method in the registry with `defensible`, `marginal` or `not_defensible` and the reason), `notes` |
| `expected` | the key: `branch` (the branch id), `gates` (`[{step, check, path}]`), `tools` (the tool names in order), `station` (the gauge the branch reads), `notes`; or `declined: true` with `decline_reason` (the sentence) and `decline_kind` (`declined` for a rule, `no_branch`, `refused` for a registry refusal) |
| `split` | `train` or `test` (see below) |
| `probe` | the first words of the decline rule this task probes, or `null` |
| `created` | ISO timestamp of generation |

Read it in Python with `aquascope.gym.read_tasks("tasks.jsonl")`, or with any
JSONL reader.

## Scoring

An agent is scored per task against `expected`:

* `branch_match`: the agent's branch is the key's, on the key's playbook. For
  an agent without a plan, the branch is inferred as the playbook branch
  whose tools its calls cover best.
* `gates_respected`: the fraction of the key's `(step, check)` gates the agent
  evaluated, pass or fail.
* `tools_matched`: the fraction of the key's tools the agent called.
* `declined_correctly`: on an unsolvable task, whether the agent refused.
* `correct`: `declined_correctly` on an unsolvable task; `branch_match` without
  a decline on a solvable one. An error or a timeout is wrong.

The key measures agreement with the playbooks, not hydrological truth: a
branch is "correct" because the decision tree chose it on the same snapshot.
The playbooks and their gates are in `aquascope/playbooks/*.yaml`; their
rationale is in the AquaScope documentation (`solve.md`).

## The split

`split` is `test` for one site in four by a SHA-1 hash of the site key (the
gauge as `source/station_id`, or the position rounded to four decimals),
`train` otherwise: `aquascope.gym.tasks.split_for`. Every task of a site has
the same split, so an agent tuned on `train` is checked on places it never
saw. In this suite 6 sites (24 tasks, 18
of them solvable) are held out.

## Sites and sources

Sites were sampled from the AquaScope station catalog (a published index of
the agencies' station lists, no agency record): round robin over kind,
continent and source, plus bare points offset 0.5 to 0.9 degrees from a
gauge in a sparse part of the catalog and confirmed on land by BasinATLAS.
Only sources whose catalog rows carry a record span can be sampled, which at
this date are the U.S. Geological Survey (`usgs`), the Environment Agency of
England (`uk_ea`) and Hub'Eau, the French hydrometry service
(`hubeau_hydrometrie`); the Australian Bureau of Meteorology, the German
Pegelonline and the Irish OPW lists carry no spans and are absent. The
suite therefore covers North America and Europe.

| split | kind | source | site | country | record (years) |
| --- | --- | --- | --- | --- | --- |
| train | gauged_long | uk_ea | Congleton Park (a49bd26a-9ea4-47ec-b7ab-3a5ceb369e70) | GBR | 46.8 |
| train | gauged_short | hubeau_hydrometrie | Le ruisseau de Predecelle à Saint-Maurice-Montcouronne (F462000401) | FRA | 7.8 |
| test | groundwater | uk_ea | Sprucely (877e6123-3f84-4a01-8eb9-0b47ae1a47f2) | GBR | 34.8 |
| train | gauged_long | usgs | Beaver Creek near Paulina, Oreg. (USGS-14078000) | USA | 33.0 |
| train | gauged_short | uk_ea | Nene Valley (2d37e684-7d68-4844-8353-05ada0054d29) | GBR | 10.0 |
| train | groundwater | uk_ea | Missenden Abbey (190a062c-f6fd-4543-9027-b0c6f4e1a773) | GBR | 34.9 |
| train | gauged_long | hubeau_hydrometrie | La Vègre à Asnières-sur-Vègre (M058302010) | FRA | 45.8 |
| train | gauged_short | usgs | Prickly Pear Creek at East Helena, MT (USGS-06062000) | USA | 5.2 |
| train | groundwater | uk_ea | Pallaflat Reservoir Trial (67522937-31fb-49a8-8de0-248b76e20ae6) | GBR | 42.5 |
| test | gauged_long | uk_ea | Easby (5b9e9503-32e2-46ea-a713-be0f283c23be) | GBR | 55.3 |
| test | gauged_short | hubeau_hydrometrie | La Gouaneyre à Arue [Téchené] (Q242431001) | FRA | 15.8 |
| test | ungauged | (bare point) | 51.4601, -2.1016 | GBR | none at the point |
| test | ungauged | (bare point) | 32.8324, -81.9755 | USA | none at the point |
| train | ungauged | (bare point) | 34.9322, -85.6230 | USA | none at the point |
| test | ungauged | (bare point) | 44.7579, -121.4263 | USA | none at the point |

Every bare point of this seed landed within 50 km of a long discharge gauge
(the catalog is denser than the 0.5 to 0.9 degree offset assumes), so the
flood risk and ungauged flow playbooks read a gauge there and no data-driven
decline (fewer than three donor gauges) arose: all 15 unsolvable tasks are
probes, 8 of the flood risk rule (inundation extent) and 7 of the
groundwater rule (attributing the cause). Keys by playbook and branch:
flood_risk at_site 11, short_record 2, regional 2, declined 8; ungauged_flow
at_gauge 15; groundwater_decline well 7, regional 8, declined 7.

## Licences of the underlying catalogs

The recon snapshots contain catalog-level metadata only (station
identifiers, names, positions, distances, record spans, URLs; aggregate
catchment attributes), no observation record. Every source whose metadata
appears in this deposit permits redistribution:

| source | licence | redistribution |
| --- | --- | --- |
| U.S. Geological Survey (`usgs`) | U.S. public domain (work of the U.S. Government) | yes, no attribution required; USGS asks to be credited |
| Environment Agency, England (`uk_ea`) | Open Government Licence v3.0 (OGL-UK-3.0) | yes, with attribution: "Contains Environment Agency information © Environment Agency and database right" |
| Hub'Eau / Eaufrance, SCHAPI (`hubeau_hydrometrie`) | Licence Ouverte / Open Licence 2.0 (Etalab) | yes, with attribution to the source and the date |
| BasinATLAS, HydroATLAS v1.0 (catchment attributes) | CC BY 4.0 | yes, with attribution: Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. Scientific Data 6: 283. <https://doi.org/10.1038/s41597-019-0300-6> |

The station lists in the recon snapshots name stations of these three sources only (uk_ea 176 rows, usgs 107, hubeau_hydrometrie 75 across the 15 snapshots), so no metadata from a source with unknown terms (the Bureau of Meteorology rows in the catalog are marked licence unknown) is in the deposit.

The tasks file itself (problem texts, intakes, keys, split) and this README
are released under CC BY 4.0. The scoring code is MIT (aquascope).

## The first results on this suite

Played on 2026-09-03 with a 240 s timeout per task (rows and answers in the
repository under `aquascope/gym/results/2026-09-03/`): the playbook tree and
the keyless plan-first team score 100 percent on the 45 solvable tasks and
decline all 15 probes, as they should by construction; the team on Claude
Sonnet 5 scores 100 percent (1.15 USD for 60 tasks) and on Claude Haiku 4.5
98 percent (one timeout; 0.41 USD); a plain tool loop on Claude Sonnet 5,
given only the question and the point, scores 68 percent on 25 solvable
tasks with 4 false declines and declines all 15 probes (3.02 USD for 40
tasks). The refusal of the tool loop is read off its wording; see
`docs/hydrogym.md` for what these numbers do and do not establish.

## Regenerating and running

```bash
pip install "aquascope[all]"
aquascope gym tasks --n 60 --seed 2026 --out tasks.jsonl      # a fresh snapshot: keys may move with the catalog
aquascope gym bench --tasks tasks.jsonl --agent tree           # the key's own baseline, offline
aquascope gym bench --tasks tasks.jsonl --agent team           # the keyless plan-first team
aquascope gym leaderboard results/*.jsonl
```

A regenerated suite is not identical to this one: the catalog is republished,
gauges close and open, and BasinATLAS coverage can change; the deposited
file is the snapshot the 2026-09-03 leaderboard was scored on.

## Citation

Ouédraogo, A. R. (2026). HydroGym Phase 1 tasks: a playbook-keyed benchmark
of hydrology agents on real sites (60 tasks, seed 2026) [Data set]. Zenodo.
The software: AquaScope, concept DOI 10.5281/zenodo.21903143.
