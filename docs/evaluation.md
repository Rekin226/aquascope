# Evaluating studies: `aquascope eval`

Every study bundle (`aquascope studio --out DIR`, a recorded showcase study, an
export from the Explorer) carries its whole run in `workspace.json`: each role's
events with a timestamp, every step and gate, the Critic's checks, the report and
the model ledger. `aquascope eval` reads that file and nothing else. No model is
called and nothing is fetched, so it is free to run as often as you like.

| Command | Answers |
| :--- | :--- |
| `aquascope eval score STUDY` | How well did this study do? |
| `aquascope eval trace STUDY` | Where did its time go, and what did not pass? |
| `aquascope eval stats DIR...` | How do many studies compare, and what fails most often? |

`STUDY` is a bundle directory or its `workspace.json`. `DIR` is searched recursively
for bundles. Every command takes `--json`.

## A scorecard: `eval score`

```bash
aquascope studio "What is the 100-year flood for a culvert here?" --at USGS-01013500 --yes -q --out fish-river/
aquascope eval score fish-river/ --case flood_at_site_potomac
```

```text
  37f08805ca54 · flood_risk (at_site) · keyless · aquascope 0.22.0 · 2026-10-04
  question  What is the 100-year flood for a culvert here?
  outcome   done, grade indicative
            design flow: 100-year return level, GEV (L-moments) 463.9 m3/s (indicative)
  run       4 of 4 steps ok, 0 failed, 0 skipped · gates 11 passed, 0 failed, 1 skipped · 0 fallback(s), 0 replan(s)
  critic    10 of 10 checks passed · 0 issue(s) · 1 not established
  report    mean 1.00 · traceability 1.00, not established completeness 1.00, no filled holes 1.00, ...
  plan      vs flood_at_site_potomac: 1.00 (tools 100 %, methods 100 %, gates 100 %, extraneous 0 %, forbidden 0)
  cost      80 s (slowest phase running 45 s; slowest step s2 analyze_station 11 s) · keyless
```

- **outcome**: the status, the grade the Interpreter gave and the headline.
- **run**: steps and gates. A *skipped* gate is one that could not be checked (a
  model cell that could not be matched to the gauge, say); it is counted apart from
  a failure.
- **critic**: the Critic's checks, its open issues, and how many points the report
  lists as not established.
- **report**: the six report-quality axes of the [HydroGym benchmark](hydrogym.md),
  scored deterministically. When the package has a report reference for the study
  (by directory name, as for the recorded showcase), its score is added.
- **plan** (with `--case`): the plan scored against a HydroGym reference case
  (`aquascope gym plans list`), with what it missed or added.
- **cost**: wall time, the slowest phase and step, and model calls, tokens and USD.

## A trace: `eval trace`

```bash
aquascope eval trace explorer/showcase/studies/kingston-flood
```

```text
  e131698dc948 · 529 s · Design flow for a new road bridge over the Thames at Kingston: ...
  phases    scouting 20 s · planning 31 s · review 0 s (includes waiting for you) · running 63 s · critique 212 s · authoring 164 s
  steps
    s1   describe_catchment          10 s  ok · gates 0 passed, 0 failed, 0 skipped
    ...
    s4   anywhere                    21 s  ok · gates 1 passed, 1 failed, 0 skipped
           failed cross_check_ratio: 14.77 against the reference 652.5 at T = 100 years: ratio 0.02, ...
  model
    critic         2 call(s), 67,888 tokens, 0.266 USD
    ...
```

Phases run from one coordinator status to the next. `review` includes the time a
person took to approve the plan, so it is never named the slowest phase. `--events`
adds every event with its offset in seconds.

## Across studies: `eval stats`

```bash
aquascope eval stats explorer/showcase/studies studies/ --by model --csv stats.csv
```

```text
| model | n | grades | report | plan | gates passed / failed / skipped | critic | median time | tokens | USD | USD per study |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-sonnet-5 | 12 | established 3, indicative 4, not_established 3, screening 2 | 1.00 | - | 93 % / 7 % / 0 % | 99 % | 538 s | 1,525,578 | 10.19 | 0.85 |
| keyless | 4 | established 2, indicative 1, not_established 1 | 1.00 | - | 93 % / 4 % / 4 % | 100 % | 80 s | 0 | 0.00 | 0.00 |

  most often failed: not_empty (3 studies), min_years (2 studies), cross_check_ratio (1 study)
```

`--by` is `playbook` (the default), `model`, `date`, `grade` or `none`. `--case`
scores every plan against one reference case, for runs of the same question.
`--csv FILE` writes the rows, the grade mix flattened into `grade_<name>` columns.

## From Python

```python
from aquascope import evaluation as ev

card = ev.score_path("fish-river/")
cards = [ev.score_path(d) for d in ev.find_studies(["studies/"])]
table = ev.stats(cards, by="playbook")
```

`scorecard`, `trace` and `stats` take a `workspace.json` dict, so they work on a
study held in memory too.
