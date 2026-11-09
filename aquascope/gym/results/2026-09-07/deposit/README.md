# AquaScope Bench Phase 2 reference plans: expert plans for hydrology agents at real sites (25 cases)

`cases.jsonl` holds 25 reference plans for the plan-quality benchmark of
hydrology agents, written by hand on 2026-09-07 with
[aquascope](https://github.com/Rekin226/aquascope) 0.15 (branch `ws/bench`,
AquaScope Bench Phase 2 of [issue #367](https://github.com/Rekin226/aquascope/issues/367),
epic [#363](https://github.com/Rekin226/aquascope/issues/363)). The benchmark
is documented at <https://rekin226.github.io/aquascope/hydrogym/> (AquaScope Bench
was called HydroGym until October 2026, #440); this file describes what is in the
deposit so it can be used without the package.

## What a case is

A case is a brief at a real place, and the plan a hydrologist would write for
it given the data that exists there. Phase 1 of AquaScope Bench asked whether an
agent lands on the branch a method-selection playbook selects; Phase 2 asks
whether the plan itself, the ordered steps with their tools, methods and
gates, is the expert's. Each case carries the client's brief in plain words,
the playbook it maps to and the intake the brief fixes, a real site (a
catalog gauge or a bare point) with a snapshot of the data reconnaissance at
that place (`aquascope.explore.assess_site`, captured once on 2026-09-07:
the catalog stations within 50 km with their record spans, the BasinATLAS
catchment, the donor count, the registry's sufficiency verdict per method),
and the reference: the steps in order with the tool, the registry method,
the gates that must be present and whether the step is optional; the tools
and methods that are not defensible at that site; or a decline when the
data or the brief cannot carry the question. Twenty-five cases on seven
sites: 18 solvable (three of them off-tree briefs no single playbook branch
covers) and 7 that decline.

## Case format

One JSON object per line, UTF-8, keys:

| key | what |
| --- | --- |
| `id` | the case id, e.g. `flood_at_site_potomac` |
| `playbook` | `flood_risk`, `ungauged_flow`, `groundwater_decline`, `drought_status`, `supply_reliability`, `irrigation_feasibility` or `water_quality` |
| `title`, `brief` | what is decided, and the client's words |
| `intake` | the playbook's intake fields the brief fixes |
| `site` | `{name, lat, lon, source, station_id, years, recon}`; `source` and `station_id` are absent for a bare point |
| `expected_branch` | the playbook branch the tree selects on the saved reconnaissance (informational) |
| `steps` | the expert plan: `[{tool, method, gates: [{check, path}], optional, note}]` |
| `forbidden` | `{tools: [...], methods: [...]}`, not defensible at this site for this brief |
| `decline`, `decline_kind` | `true` with `declined` (a playbook rule), `refused`, `no_branch` or `intake` when the right plan is none |
| `rationale` | why this is the plan |
| `tags` | `off_tree` for the three briefs beyond any single branch |
| `recon` | the saved reconnaissance, narrowed to the playbook's problem kind as the Studio's Scout narrows it: `point`, `stations`, `catchment`, `context` (years and resolution per variable, area, donors, what is assumed reachable), `sufficiency`, `notes` |
| `recon_captured` | the capture date |

Read it in Python with `aquascope.gym.plans.Reference.from_dict(row)` on
each line, or with any JSONL reader; the YAML sources are in the repository
under `aquascope/gym/plans/`.

## Scoring

A candidate plan is scored against a case with
`aquascope.gym.plans.score_plan`: the fraction of the reference's required
tools the plan uses (weight 0.30), of its required registry methods it names
(0.25), of its required gates it carries on a step with that tool (0.20);
1 when no step uses a forbidden tool or method, else 0 (0.15); and one minus
the fraction of the plan's steps the reference does not name, framing steps
excepted (0.10). A part the reference cannot judge is left out and the
weights renormalised. A plan with no steps scores 0; a plan that declines a
solvable case scores 0; on a declining case a decline scores 1 and a plan 0.
The number of errors the Studio's plan validator raised on a model's first
attempt, and whether the plan that stands is the playbook tree's fallback,
are reported beside the score.

The key is expert judgement over the registry's preconditions and the
playbooks' branches, not hydrological truth: the reference says which
steps a defensible answer needs at that site, not what the numbers are.

## Sites and sources

| site | source | what is there | cases |
| --- | --- | --- | --- |
| Potomac River near Washington, DC, Little Falls pump station (USGS-01646500) | usgs | 96.5 years of discharge, sampled water quality; 30,090 km2, regulated | 6 |
| Malad River near Gooding, Idaho (USGS-13152500) | usgs | 110 years of discharge, sampled water quality; 8,363 km2 | 4 |
| Fish Creek near Battle Mountain, Nevada (USGS-10326800) | usgs | 8.5 years of discharge, 1977 to 1985; no longer gauge within 50 km | 1 |
| A bare point in the Big Smoky Valley, Nevada (38.0, -117.0) | none | no gauge within 50 km; 10 donor gauges | 4 |
| Tetbury, Gloucestershire (02bd4687-...) | uk_ea | a 35-year rain gauge; Tetbury Slads Farm discharge (48 years) 0.7 km; Brokenborough borehole (50 years) 3.9 km | 6 |
| A bare point on the Warm Springs reservation, Oregon (44.7579, -121.4263) | none | Shitike Creek gauge (22 years, to 1996) 9.7 km; Deschutes near Madras (103 years) 14.6 km | 1 |
| La Vègre à Asnières-sur-Vègre, Sarthe (M058302010) | hubeau_hydrometrie | 45.8 years of discharge; 411 km2 | 3 |

## Licences of the underlying catalogs

The recon snapshots contain catalog-level metadata only (station identifiers,
names, positions, distances, record spans, URLs; aggregate catchment
attributes), no observation record. Every source whose metadata appears in
this deposit permits redistribution:

| source | licence | redistribution |
| --- | --- | --- |
| U.S. Geological Survey (`usgs`) | U.S. public domain (work of the U.S. Government) | yes, no attribution required; USGS asks to be credited |
| Environment Agency, England (`uk_ea`) | Open Government Licence v3.0 (OGL-UK-3.0) | yes, with attribution: "Contains Environment Agency information (c) Environment Agency and database right" |
| Hub'Eau / Eaufrance, SCHAPI (`hubeau_hydrometrie`) | Licence Ouverte / Open Licence 2.0 (Etalab) | yes, with attribution to the source and the date |
| BasinATLAS, HydroATLAS v1.0 (catchment attributes) | CC BY 4.0 | yes, with attribution: Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. Scientific Data 6: 283. <https://doi.org/10.1038/s41597-019-0300-6> |

The cases file (briefs, intakes, reference plans, rationales) and this README
are released under CC BY 4.0. The scoring code is MIT (aquascope).

## The first results on this suite

Played on 2026-09-07 from the saved reconnaissance (rows and plans in the
repository under `aquascope/gym/results/2026-09-07/`): the playbook tree
scores 0.97 (1.00 on the 22 cases a branch covers, 0.65 to 0.78 on the three
off-tree briefs, all seven declines); the AquaScope Studio Methodologist on
Claude Sonnet 5 scores 0.64 overall, 0.89 on the solvable cases and 0.98 on
the off-tree briefs, on Claude Haiku 4.5 0.66, 0.92 and 0.96 (two repeats
each, spread of the means 0.03 and 0.01), both declining none of the seven
declining cases, the largest gap the suite finds and a Studio behaviour
rather than a model one. The two model runs cost 4.76 USD at list prices.
