# Studio: a complete study at a place

`aquascope studio` puts a crew of roles over one shared workspace and runs a
hydrological study the way an engineer would: the brief is agreed with you,
the methodology is shown and approved before anything runs, every step
passes a gate, the report says what it does not establish, and the study
file re-runs with no model. [studio-design.md](studio-design.md) is the
contract; this page is the user's guide.

```
 you ──► Consultant ──► Scout ──► Methodologist ──► you ──► Analysts ──► Interpreter ──► Critic ──► Author ──► bundle
        (the brief)   (inventory)  (the plan)     (approve)  (run, gates)  (review)  (report)
```

## Start

```bash
pip install "aquascope[studio,basins]"
aquascope studio
```

In a terminal, `aquascope studio` with nothing else asks where (a gauge name or river words from the station
catalog, a station id such as `USGS-01013500`, or `lat, lon`) and what you want to know. With a model key in the
environment it offers to use it, with a $1 spend ceiling; it never uses one silently. For scripts, give everything
on the line: `aquascope studio "PROBLEM" --at USGS-01013500 --yes` (or `--lat`/`--lon`). Without the `studio`
extra the bundle holds only the Markdown and HTML report and the tables, and the CLI says so.

## The flow

1. **Brief.** You say the problem in plain language. The Consultant writes
   the brief: the decision, the quantities wanted, the playbook it maps to,
   the intake fields it can read off the text. At most three questions come
   back when something the analysis cannot proceed without is missing; say
   `just go` to proceed on the defaults, which are then listed as
   assumptions.
2. **Inventory.** The Scout lists what exists: the gauges within reach with
   their record spans, the catchment, the donor pool, the ERA5 cell (any
   point on land), and every table you attached, run through the ingest
   mapping and QA.
3. **Plan.** The Methodologist writes a version-3 study: objective,
   methodology in numbered sentences, steps with arguments, a registry
   method, gates and fallbacks, expected figures and tables. Keyless it is
   the playbook tree's plan; with a model it is composed from the catalogue
   and validated (the tool exists, the arguments are its own, the gates are
   known, a method the registry calls not defensible here is refused).
4. **Review.** The plan is a numbered checklist. Approve it, edit a step
   (`s3.return_period=200`) or change the brief ("make it a 200-year return
   period") and get a new plan.
5. **Run.** The Analysts run the steps in order with their gates; a failed
   gate runs the fallback once; a step that still fails is recorded as not established and the rest of
   the plan runs (its dependents are skipped with the reason); each failed step gets its replan or its
   Specialist fallback once (the playbook's
   branch, or a Specialist's proposal validated against the catalogue).
   Figures and tables are made per step as results land.
   **Waiting for data.** When a playbook's own rule would decline for data
   the user could bring (abstraction records to attribute a groundwater
   decline, a reservoir's capacity and rule, your own water-quality samples,
   a discharge record where no gauge reaches), the crew asks instead: what
   it needs, why, and what it changes. Drop the table in (the page, `--data`
   or a path at the prompt in the terminal, `tables` over MCP) and the plan
   is written again on it, or say "continue without" and the study goes on
   at the lower grade the request named (`--continue-without` in the
   terminal). A table can arrive at any point: at review it is inventoried
   and planned on, after the report it runs as a follow-up change.
6. **Interpret.** The Interpreter reads the results, the gates and the brief's
   decision and writes findings, not prose: claims that each point at the
   result they come from (`s3.ffa.fits.gev_lmoments.q.5`), whether the
   estimates agree, the decision block (the value, its band, its grade, the
   conditions, what would change it), the data the crew would ask for, and
   the assumptions. Keyless it is a rule table over the key numbers and the
   gates; with a model, one call the engine holds to the results (a basis
   that resolves to nothing drops the finding, a grade may go down, never up).
   Every answer carries a grade: `established` (at-site data, every gate
   passed), `indicative` (a fallback, a donor transfer, a marginal method),
   `screening` (regional or reanalysis data only), `not_established`.
7. **Report.** The Author writes the report from the findings; every sentence
   a model wrote passes the Critic's number check first (one whose numbers
   are in no result is dropped and counted). The Critic then runs its
   deterministic checks and, with a model, one independent pass; every
   failed check and every `fix` issue goes to the Author for one rewrite,
   the second critique keeps the model, and a report that still fails opens
   with one line naming the failed checks. What is not established is
   listed, never hidden.
8. **Bundle.** Markdown, `study.yaml`, `report.json`, `findings.json`, `workspace.json` and,
   when the deliverables package is installed, the figures, the Excel
   workbook, the Word report, the notebook and one zip. The raw record and
   the raw samples live in the workbook and the notebook; the documents say
   which sheet, and print the evidence tables only.
9. **Follow-up.** A question is answered from the workspace; a change (another
   return period, another statistic, another gauge, the donors) is planned,
   run and re-authored, reusing every step whose arguments and gates did not
   change.

## The keyless Consultant's questions

Without a model the Consultant still asks real questions, from the gaps the
text and the site leave, at most three in one round:

| Gap | The question |
| --- | --- |
| a playbook field with no default (the demand of a supply question) | the field, once for a pair that asks the same thing (`demand_m3s` or `demand_ml_day`) |
| no decision stated | what will be decided, with the playbook's options: design flow, risk screening, insurance, inundation extent; an abstraction licence, the size of the scheme, a screening; drought restrictions, irrigation planning, a situation report; the purpose, the concern or the use for the other playbooks |
| a flood question without a return period | the return period, default 100 years |
| a drought question without a period | now, the last 3 months, the last 12 months, the whole record; default now |
| an upload with two or more numeric columns | which column holds the values, default the first |
| no playbook matched | which kind of problem it is; the chosen playbook's gaps are asked once after that |

A decision the text names ("a culvert", "for a licence", "screening") is not
asked. `just go` takes every default and writes each one down as an
assumption; a decision with no default stays unstated rather than invented.

## Keyless follow-ups that add steps

After the report, a follow-up without a model can add steps as well as change
the intake. A rule table maps the words to catalogue steps, appended to the
plan with ids continuing the sequence, validated like any step, run with the
earlier results reused, and re-authored:

| You say | The step added |
| --- | --- |
| "add the nearest gauge", "also include the upstream gauge" | `analyze_station` on the next gauge of the same variable in the inventory |
| "compare with the donors" (regional, the neighbours) | `similar_basins` and `regionalize_signatures` at the site |
| "flow duration", "fdc", "Q95" | `low_flow_context` on the plan's gauge, or `flow_duration` on the attached table |
| "trend", "Mann-Kendall" | `analyze_station` with the `trend_mann_kendall` method |
| "SPI", "SPEI", "drought" | `drought_indices` for the ERA5 cell |
| "baseflow", "BFI" | `low_flow_context` with `baseflow_separation`, or `baseflow` on the attached table |
| "GloFAS", "ERA5", "cross-check" | `anywhere` at the site |
| "200-year" | the intake change, the tree planned again; the steps earlier follow-ups added come along |

A request no rule covers gets the honest answer: it cannot be added without a
model, and the list of what can.

## The tiers

| Tier | What runs | What you get |
| --- | --- | --- |
| Keyless (default) | keyword rules, the playbook tree, gates, template prose, the deterministic checks | a complete study with zero model calls |
| With a model | the same, plus one stateless call per role: the brief, a composed methodology, the Specialist's fallback, the Critic's issues, the Author's prose | the same bundle, with prose and a plan beyond the seven trees |

A model is used only when asked for (`--provider`, `--model`, `--api-key`,
`--base-url`, or a ready client). The ledger of calls and tokens per role is
in the report's footer, with the USD spent when the model is in the price table
(`aquascope.ai_engine.providers.PRICES`) and the sentences the checks dropped.
`--max-usd` (CLI), `max_usd` (`Studio`, MCP) is a spend ceiling: past it the roles
run keyless, the event and the footer say so, and the study still ends in a bundle.

## CLI

```bash
aquascope studio "Design flow for a road crossing, 100-year return period" --lat 51.415 --lon -0.308
```

Interactive in a terminal: the questions, `Run this plan? [y/N/e]` (`e` to
type overrides), the timeline as it happens, then a follow-up loop until
`done`. Non-interactive:

```bash
aquascope studio "Is this area in drought now?" --lat 24.15 --lon 120.68 --yes --out out/taichung
aquascope studio "..." --lat ... --lon ... --data flows.csv --intake return_period=200 --provider anthropic
aquascope studio --resume out/taichung/workspace.json --yes
```

`--yes` answers the defaults, approves and exports. `--data FILE` attaches a
table (CSV, Excel, JSON) as `upload:<name>`; a plan may load it with
`load_table` and analyse it with the workbench tools. `--resume` continues
a saved workspace from wherever it stopped.

## Python

```python
from aquascope.studio import Studio

s = Studio(lat=51.415, lon=-0.308, data={"flows.csv": df}, on_event=print)
r = s.say("Design flow for a road crossing, 100-year return period")   # -> questions or the plan
if r.kind == "questions":
    r = s.say("just go")
r = s.approve()                          # -> the report
r = s.follow_up("how sure can we be?")   # -> an answer from the workspace
r = s.follow_up("redo it with a 50-year return period")   # -> a new report
paths = s.export("out/")
ws = s.to_dict(); s2 = Studio.from_dict(ws)    # checkpoint and resume anywhere
```

Every method returns a `Reply` with `kind` (`questions`, `plan`, `report`,
`answer`, `declined`), `text` and `payload`. The roles are plain functions
under `aquascope.studio.roles` (`consult`, `scout`, `plan`, `run`,
`critique`, `author_report`), each over the workspace, for anyone who wants
to drive them from another orchestrator; `examples/langgraph_team.py
--studio` and `examples/crewai_studio.py` show two.

## Bring your own model

Any model you run yourself (Chrome's built-in model, a small model in the
tab, a client of your own) can do what the crew's model would, and the crew
treats what it wrote exactly like its own model's output: the same coercion,
the same validator, the same checks. Three entry points take the reply, three
give you the prompt and the context to send:

```python
s = Studio(lat=51.415, lon=-0.308)
ctx = s.consultant_context("A culvert on the Thames")        # {"system": <prompt>, "problem", "site", "recon", ...}
r = s.say("A culvert on the Thames", proposed={"brief": your_model(ctx), "source": "device"})
# the brief is merged (intake through coerce_intake, unknown fields dropped, brief.source = "device");
# the keyless questions cover what it left open; then the Scout and the plan
ctx = s.methodologist_context()                              # the plan's context, with the tree as the exemplar
r = s.approve(plan={**your_model(ctx), "source": "device"})
# the plan goes through validate_plan with repair (a wrong method replaced or dropped, a guessed gate path
# corrected), the invalid steps pruned, the tree when nothing valid remains;
# r.payload["plan_used"] is "proposed" or "tree", r.payload["plan_errors"] the validator's findings,
# study.plan["author"] the source
ctx = s.author_context()                                     # the report's context; author_context(issues=...) the fix round
r = s.narrate(your_model(ctx)["sections"], source="device")
# every sentence passes the Critic's check first: one whose numbers (or years) are in no tool result is
# dropped and counted (r.payload["dropped"]); ws.report["written_by"] and the footer say who wrote which
# section; the deliverables are rebuilt. Sections not given keep the template's text.
```

`narrate` takes the section ids of `ws.report["sections"]` plus `answer` and
`recommendations`; `references` and `appendix` are the crew's. The prompts
the roles use, with the JSON schemas of the three replies (the brief, the
plan, the sections), ship as `explorer/prompts.json`
(`python -m aquascope.studio.prompts`, `aquascope.studio.prompts.as_json()`),
so a page runs the same prompts on a device model.

## MCP

`studio_start(problem, lat, lon, intake=None, ...)`, `studio_say(workspace,
text, proposed=None)`, `studio_approve(workspace, edits=None, plan=None)`,
`studio_follow_up(workspace, text)`, `studio_narrate(workspace, sections,
source="device")`, `studio_context(workspace, role, text=None)` (role
`consultant`, `methodologist` or `author`; the prompt under `system`),
`studio_export(workspace, out_dir)`. The tools are stateless: each returns
the reply, a summary and the workspace dict to pass to the next.

## In the Explorer

**Study** is the second mode of the Explorer's drawer, next to Ask. Open it
with **Study this place** on a gauge or on a point (a bare point works: the
Scout finds what is in reach), or with the Study button at the top right.
Everything runs in the page: the crew is `aquascope.studio` in the Pyodide
worker, the same Coordinator the CLI and the MCP tools drive.

The drawer is a conversation and a board. The conversation is the workspace's
messages: what you said, what the Consultant asked, what the crew wrote. The
board above the input shows one thing at a time:

1. **Intake**: the place, the model line, and what to bring. Drop a CSV or an
   XLSX, or attach the table already open in My data. Type the problem and
   send. When the Consultant asks something, the options are chips (one click
   answers) and **Just go** proceeds on the defaults.
2. **Review**: the plan as a card: the objective, the numbered steps with
   their arguments in words, the gates as chips, the rationale on expand.
   **Approve** runs it; **Edit** opens the arguments inline and sends them as
   the Methodologist's edits, revalidated in the worker (a refused edit says
   why and keeps the plan); **Decline** keeps the input open for a change of
   brief, or start again.
3. **Running**: the timeline as it happens, one line per event, and the
   figures as they are drawn. **Stop** means stop: Python cannot be
   interrupted mid-call, so the worker is terminated and boots again (the
   progress bar as at first load, a few seconds when the runtime is cached).
   The page keeps its copy of the study, so the board returns to the plan
   with one line, "stopped; the figures made so far are gone, the plan is
   kept", and the next Approve rebuilds the study in the fresh worker from
   that copy. The table open in My data is handed to the new worker again.
4. **Done**: the answer, the key numbers, the figures, what the study does
   not establish when the Critic listed anything, **Download bundle** (the
   zip) and links for the Word, Excel, Markdown, notebook and `study.yaml`
   files. The input stays open: a question is answered from the workspace, a
   change ("redo it with a 200-year return period") is planned, run and
   re-authored, and the board refreshes. **New study** clears the board for
   another study at the same place.

The Done board opens with a grade badge (established, indicative, screening
or not established, each explained on hover) and the decision block above the
answer: what the value holds if, what would change it and what the crew would
ask for, then the findings, collapsed. When a playbook's own rule would
decline for data you could bring, the board shows a waiting card instead:
what the crew needs, why, and a **Continue without** button where the study
can still answer at the lower grade the request names. A CSV or XLSX can be
dropped at intake, waiting, review or done alike, not only before a study
starts: once one exists, a table is attached to it directly, and the crew
plans again on it at waiting or review, or runs it as a follow-up once the
report is in.

The tiers are Ask's. Keyless by default, which is a complete study: the
playbook tree plans, the gates check, templates write. When Ask holds a key,
one line offers it for the prose and the composed methodology.

**The device model on the crew.** When Chrome's built-in model is already on
the device, or Ask has loaded a small model in this tab, it joins the keyless
crew in three bounded places, and the card says who wrote what:

- the **brief**: it reads the first sentence (the decision, the quantities, a
  return period or drought timescales when stated) before the Consultant sees
  it, and a wrong reading costs one question, never a wrong number;
- the **plan**: at review, the page asks the worker for the Methodologist's
  context (the same compact JSON the crew's own model would read, with its
  system prompt) and the exported prompts (`explorer/prompts.json`, or the
  engine's own when the page has none), the model writes a plan in one call,
  and the engine's validator checks it before the card shows it. The card
  then says "planned on this device with Chrome's built-in model", or keeps
  the tree's plan and says "the playbook's plan (the device model's plan did
  not pass the validator: ...)" with the first error. Approve sends the
  device's plan, with any inline edits, and the engine validates it again
  before running it; the foot of the answer says which plan ran;
- the **prose**: after the run, the model writes the summary and the
  recommendations in one call and, while it is quick, one call per result
  step, four calls at most, and the engine's narrate keeps only what the
  Critic's checks allow (a sentence with a number the results do not carry
  is dropped). The line under the answer says "written on this device with
  Chrome's built-in model; N sentences dropped by the checks".

Every call has the 25 s limit Ask's on-device brief has; on a timeout or a
reply that is not a plan or a section, the keyless result stands and one
line says so. Nothing is downloaded for any of this: Study never starts a
model download, it only uses one that is already there. The device-model
path cannot run in a headless browser (no WebGPU, no Prompt API), so it is
tested against a fake model under node and by hand in Chrome.

**Saved studies.** After every reply the workspace (without its bytes) and
the PNG figures are saved in the browser's IndexedDB, the last five studies
kept, nothing sent anywhere. Opening Study at a place where a study was made
(or a `#study=1` link there) offers a **Resume the last study** chip that
reopens its board in its state, figures included; the documents are remade
by the next run. A `workspace.json` from a bundle (or from the CLI) dropped
on the board resumes that study the same way. Where the browser blocks
storage (a private window, a quota), nothing is saved and nothing is said.

**Tables.** A CSV travels as it is; an `.xlsx` is turned into CSV in the
worker (pandas, through the same `table` op) before the study starts, so
every table sits in the workspace the same way and round-trips through the
bundle. The Scout lists it with its QA next to the gauges, and keyless the
plan runs on it (`load_table`, then the workbench tools).

**Recorded studies.** The intake board offers the recorded studies as chips
("See a recorded study"): one opens as its finished board, the thread, the
answer, the key numbers and the figures as recorded, with one line saying
who recorded it and that the numbers were computed then. **Re-run live**
starts the same study at the recorded site in your browser and approves the
recorded plan, which runs keyless through the validator and the gates
(the template narrator writes, or the device model when it is there), and
`#study=<id>` opens a recording directly.

The Study modules (`studio.js`, `intake.js`, `studio-device.js`,
`studio-recorded.js`, `study-store.js`) load on first use of the Study button, the drawer's
radio, **Study this place** or a `#study=1` link, not on a first visit that
runs no study.

The bytes stay in the worker. The page holds the workspace without the
artifact data; a figure travels as a PNG when it is drawn, a document only
when you ask for it, the bundle when you download it. matplotlib is loaded
before the first run and the document libraries before the first bundle,
once per visit, and never on a visit that runs no study. Nothing is uploaded
anywhere: the tables you attach are read in your tab and travel inside the
workspace as CSV text.

## Recorded studies

The keyless tier plans from the playbook tree and writes template prose.
What the crew does with a model (a methodology composed from the
catalogue, a Critic with findings, an Author who writes) needs a model
once, not at every visit. So the maintainer records a dozen studies with a
model and commits the bundles under `explorer/showcase/studies/`: flood,
drought, supply, groundwater, ungauged flow, water quality, irrigation and
a table you bring; gauged sites and bare points; five continents. The Study
intake offers them as chips.

Opening one shows the recorded study as it was: the brief, the plan, the
run with its gates, the figures, the report. Two things are true of every
recording and the page says both:

- **The numbers re-run live.** The plan (`study.yaml`, the version-3 study
  in `workspace.json`) is handed back to the worker and every step runs
  again in your browser, keyless, at the same place with the same arguments
  and gates. Records grow, so a number can differ from the recording; a
  difference is new observations, not an error.
- **The prose is a recording.** The Author's text and the Critic's findings
  were written once by the model named in the label ("recorded on
  2026-09-07 with claude-sonnet-5, 0.42 USD") and are shown as recorded,
  never regenerated.

A recording is a directory: `workspace.json` (without the artifact bytes;
`aquascope studio --resume` picks it up), `report.md`, `study.yaml`
(`aquascope run study.yaml` replays it with no model), `figures/*.png` and
`meta.json` (the model, the date, the tokens and the estimated cost, the
seconds, the gates passed, the headline). `index.json` at the root lists
them. A declined study is kept as recorded: a decline with its reason is a
valid worked example.

Recording is a maintainer's command and needs a key:

```bash
aquascope studio-showcase record --out explorer/showcase/studies [--only kingston-flood] [--max-usd 15]
aquascope studio-showcase list --out explorer/showcase/studies
```

Cases fresher than 30 days are skipped (`--refresh-after`), so a run that
stops halfway tops up rather than starting again, and the run stops at
`--max-usd`. The cases are `aquascope.studio.showcase.CASES`.

## The honesty rules

- Every number in the report comes from a tool result and passes the gates
  and the checks before it is quoted; what fails is listed under "what this
  study does not establish".
- A method the registry calls not defensible at the site is refused at plan
  time, whoever wrote the plan.
- The playbook's caveats are printed verbatim when a playbook applies.
- The study replays with no model: `aquascope run study.yaml`.
- The ledger (calls and tokens per role) is in the report's footer, and the
  plan says who wrote it (`playbook`, `methodologist`, or the source of a plan
  you brought, `device` by default); the report says who wrote each section
  (`written_by`).
