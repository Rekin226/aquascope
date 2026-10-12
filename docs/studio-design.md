# Studio, the design contract

The contract behind `aquascope studio`, the Explorer's Study mode and the
`studio_*` MCP tools. [studio.md](studio.md) is the user's guide;
[solve-design.md](solve-design.md) is the contract of the layer underneath
(plans with gates, playbooks, the method registry), which Studio keeps.

## The one sentence

A crew of roles over one shared workspace does a complete study at a place,
with the data that exists there and the data the user brings, and reports it
the way an engineer would: the user agrees the brief and approves the
methodology, then the crew runs to the bundle.

```
 you ──► Consultant ──► Scout ──► Methodologist ──► you ──► Analysts ──► Interpreter ──► Critic ──► Author ──► bundle
        (the brief)   (inventory)  (the plan)     (approve)  (run, gates)  (review)  (report)
                                                                 ▲                              │
                                                                 └────────── follow-up ◄────────┘
```

## Why a crew and not a pipeline, and why no framework

The team behind `aquascope solve` is five names on a fixed pipeline: keyword
rules, a YAML tree, gates, templates, and at most three stateless model calls.
It proves that a plan can be shown first and checked at every step. It cannot
talk about the problem, cannot plan outside its seven trees, and produces
Markdown.

Studio keeps every property that made Solve honest (a registered function
behind every step, a gate on every step, a study the runner replays with no
model, the checks on the prose) and adds what Solve lacked: a conversation,
a methodology composed for the data at hand, figures and documents, and a
memory the study lives in.

No agent framework: the engine runs in the Explorer's Pyodide worker
(LangChain and LangGraph closed Pyodide support as won't-fix in June 2026,
langchain-ai/langchain#35881; CrewAI and AutoGen do not build for WASM),
free tiers cannot carry transcripts between roles, and a framework's tool
loop would sit between the numbers and the report. The roles are plain
functions; `examples/langgraph_team.py` and `examples/crewai_studio.py` map
them onto those frameworks for people who host the crew elsewhere.

## The workspace (`aquascope/studio/workspace.py`)

One serialisable object every role reads and writes:

| Field | Written by | What |
| --- | --- | --- |
| `brief` | Consultant | problem, decision, quantities, period, horizon, constraints, deliverables, kind, playbook, intake, assumptions, questions, ready, source |
| `inventory` | Scout | site, datasets (station, upload, reanalysis, catchment, donors, samples), the raw `assess_site` recon (the sufficiency table lives there), catchment, donors, notes |
| `study` | Methodologist | `Study.to_dict()`, version 3 |
| `run` | Analysts | ok, results per step with gates, `failed_steps`, `summary` (planned, ran, ok, failed, skipped), stopped_at and stop_reason for a hard stop only |
| `findings` | Interpreter | findings (claim, basis paths, grade), consistency, decision (answer, value, unit, band, grade, conditions, what_would_change_it), data_requests, assumptions, written_by, dropped |
| `critique` | Critic | ok, checks (with `findings_resolve` and `decision_in_answer`), failed, issues, not_established |
| `report` | Author | title, answer (opening with the notice when the critique is not ok), key_numbers, sections (id, title, text, figures, tables), not_established, critique, critique_ok, dropped, references, footer (ledger in tokens and USD, budget, dropped) |
| `artifacts` | Analysts, Author | figures, tables, documents, workbook, notebook, study, bundle: bytes with a name, a media type, a caption, a step |
| `messages` | everyone | the conversation, with a kind a face can render (text, questions, plan, report) |
| `events` | everyone | the timeline (`role, step, event, detail, at`) |
| `ledger` | Model | calls and tokens per role |
| `tables` | Coordinator | the user's uploads as CSV text, so they round-trip |
| `status` | Coordinator | intake, scouting, planning, waiting, review, running, critique, authoring, done, declined |
| `pending_request` | Methodologist | while `waiting`: what the crew asks for, why, the effect, `continue_without` (the intake to apply, or None when the decline stands), `grade_without` |

`to_dict` / `from_dict` (bytes as base64), `to_json` / `from_json`. The
browser holds the dict between worker calls, the CLI writes
`workspace.json`, MCP passes it in and out.

## The catalogue (`aquascope/studio/catalogue.py`)

Every step a plan may name, described once: id, what it is for, its kind
(site, station, frame, weather, recon, table, none), its argument schema
and required arguments, what its payload yields (series, annual_maxima, ffa,
fdc, trend, spi, spei, sgi, donors, signatures, climate, samples, wqi,
demand, reliability, catchment, ...), the registry methods it applies, the
gates that make sense on it, the figures and tables the makers draw from it,
a citation. Built from the Analyst's tool specs, `workbench.TOOLS`,
`assess_site` and the `load_table` step (a user's upload as a payload).

`compact(problem)` is the JSON the Methodologist reads. `callables(extra)`
is the dict the runner executes. `validate_step` and `validate_plan` are the
deterministic checks every model-written step passes: the tool exists, the
arguments are the tool's, required ones are present, references and
`depends_on` point backwards to known ids, every gate is in
`aquascope.gates.CHECKS`, a `method` is a registry id the tool applies and is
not marked not defensible at the site.

## Study version 3

The version-2 shape (steps with id, rationale, arguments, gates, fallback,
depends_on, method) plus, on the plan: `objective`, `decision`,
`methodology` (numbered sentences), `assumptions`, `alternatives` (considered
and why not), `limitations_expected`, `citations`, `author`
(`methodologist` or `playbook`); and on each step: `outputs`
(`[{kind: figure | table, id, caption}]`). `run_study` runs it as a
version-2 study; `aquascope run` replays it with no model.

## The roles

Each role is a module under `aquascope/studio/roles/` exposing one function
over the workspace. Each has a keyless behaviour; each uses the model, when
one is present, as stateless calls with compact JSON in and one JSON object
out (`aquascope.studio.model.Model.call_json`).

| Role | Keyless | With a model | Writes |
| --- | --- | --- | --- |
| Consultant | keyword rules (`team.choose_playbook`), intake hints, the decision the text names, the gaps as questions (a field with no default, the decision with the playbook's options, a flood question's return period, a drought question's period, an upload's value column; at most three); "just go" takes the defaults and lists them | the brief from the text, the site, the attached tables and a catalog-only recon; at most three questions in one round; "just go" proceeds on assumptions | `brief`, a `questions` message |
| Scout | `assess_site`, the ERA5 and GloFAS reach, uploads through `ingest` | the same (deterministic) | `inventory`, a `Dataset` per row |
| Methodologist | the playbook tree (`playbooks.plan`); after the report, a rule table from the follow-up's words to catalogue steps appended to the plan (`FOLLOW_UP_RULES`) | a version-3 study composed from the catalogue; `validate_plan`; one repair call with the errors; then the tree; then a decline with the errors | `study`, a `plan` message |
| Analysts | `run_study` with gates; a failed gate fails its step and the run goes on; the bounded replan per failed step of `team._execute`; figures per step as results land | the Specialist's fallback proposal after each failed gate, as in Solve | `run`, figure and table artifacts, events |
| Interpreter | one finding per key number anchored at the path its value sits at (`find_path`), consistency from the comparison gates, the decision from the brief and the headline number with its band, the grade from the run (`grade_for_study`), data requests from a rule table over the playbooks | one call over the compact results, the gates and the rules' draft; validated: a basis that resolves to nothing drops the finding, a claim number not at its basis drops it, a grade above the rule's is lowered, a decision value at no basis is replaced | `findings` |
| Critic | `verify.verify` on the draft, the gates, the plan's notes; every failed check is a fix for the Author (`fixes_for`), and the notice when the critique stays not ok | one independent pass over the draft sections and the compact results: issues with a section, a severity and a fix; the second pass after the fix round keeps the model | `critique` |
| Author | template prose (`team._template_answer` and the sentence makers); the key numbers harvested from every payload (both fits at every return period, every drought timescale with its class, the transferred signatures' bands, the reliability by year) | one call per report for the prose of every section, given the compact results, the critique and the caveats; the numbers come from the results | `report` (with `written_by`), the documents, the workbook, the notebook, the bundle |
| Coordinator | the state machine, checkpoints, follow-ups | the same; a follow-up is classified by the Consultant as a question (answered from the workspace) or a change (steps appended, run, re-authored) | `status`, `follow_ups` |

Reading between the roles: compact JSON of exactly what the role needs
(`aquascope.studio.model.compact`), never the whole workspace, never a
transcript.

## Data requests (`aquascope/studio/requests.py`)

A playbook decline whose reason is data the user could bring becomes a request (a small rule table keyed by
playbook, matched on the decline sentence and the intake): the study parks at `waiting` with
`ws.pending_request`, the reply kind is `data_request`, and `Studio.say("continue without")` applies the
rule's `continue_without` intake and plans again (or declines when the rule allows no continuation);
`Studio.add_table(name, frame_or_csv)` takes a table at any status: kept at intake, inventoried and planned
on at `waiting` or `review`, a follow-up change after the report. The Interpreter's post-run
`data_requests` are the other half: what the run itself could not establish and what would firm it up.

## The Coordinator's API (`aquascope.studio.Studio`)

```python
s = Studio(lat=..., lon=..., provider=..., model=..., api_key=..., base_url=..., client=...,
           data={"upload:flows.csv": df}, on_event=print)
r = s.say("Design flow for a road crossing, 100-year")   # -> the Consultant's reply (brief, questions or the plan)
r = s.say("200 years, and use the local gauge")            # answers; when the brief is ready the Scout and the
                                                            # Methodologist run and the plan comes back
r = s.approve(edits=None)                                   # runs Analysts, Critic, Author; returns the report
r = s.follow_up("redo it with a 500-year return period")   # a change: new steps, run, re-author
paths = s.export("out/")                                    # writes the bundle's files
ws = s.to_dict(); s2 = Studio.from_dict(ws)                 # checkpoint and resume anywhere
```

Every method returns a `Reply` (`kind`: questions, plan, report, answer,
declined; `text`; `payload`) and leaves the workspace consistent, so a face
can stop at any point and resume. The keyless path is the default: a model is
used only when one is asked for.

Bring your own model, the same crew: whatever model a face runs itself hands
its output to the crew, which treats it exactly like its own model's.

```python
ctx = s.consultant_context(text)          # the Consultant's prompt (under "system") and compact context
r = s.say(text, proposed={"brief": {...}, "source": "device"})
                                          # merged with the model-reply coercion (coerce_intake, unknown fields
                                          # dropped, brief.source set); the keyless questions cover the gaps
ctx = s.methodologist_context()           # the plan's context (methodologist_context(request) the change's)
r = s.approve(plan={...Methodologist reply shape..., "source": "device"})
                                          # validate_plan with repair, a wrong method dropped, the invalid steps
                                          # pruned, the tree when nothing valid remains; study.plan["author"] is
                                          # the source; payload plan_errors (list), plan_used ("proposed" | "tree")
ctx = s.author_context(issues=None)       # the report's context (with issues, the fix round's)
r = s.narrate({"summary": "...", "results-s3": "...", "answer": "..."}, source="device")
                                          # after "done": each sentence through verify against the Critic's pool,
                                          # dropped when its numbers are in no result; ws.report["written_by"]
                                          # and the footer say who wrote which section; deliverables rebuilt;
                                          # payload {"dropped": n, "written_by": {...}}
```

The prompts and the reply schemas ship as `explorer/prompts.json`
(`aquascope.studio.prompts.as_json()`, `python -m aquascope.studio.prompts`,
a test keeps the file in step); MCP mirrors the entry points with
`studio_say(workspace, text, proposed=None)`, `studio_approve(workspace,
edits=None, plan=None)`, `studio_narrate(workspace, sections, source)` and
`studio_context(workspace, role, text=None)`.

## The deliverables (`aquascope/studio/deliverables/`)

Pure Python so they run in the Pyodide worker; the plotting and document
libraries are imported inside the functions, never at module import.

| File | Makes | Needs |
| --- | --- | --- |
| `figures.py` | PNG and SVG per figure kind from the payload (series, annual_maxima, frequency_curve, fdc, trend, drought_strip, propagation, donors_map, signatures_band, monthly_climate, reliability_curve, demand_monthly, site_map, ...) with a caption | matplotlib |
| `workbook.py` | the Excel workbook: README, one sheet per dataset and result table, a figure index | openpyxl |
| `report_docx.py` | the Word report with figures and tables embedded | python-docx; skipped with a note when it cannot be imported |
| `report_md.py` | Markdown and HTML through `ReportBuilder` | nothing |
| `notebook.py` | the notebook that re-runs the study and redraws the figures, as plain JSON | nothing |
| `bundle.py` | the zip with everything, study.yaml, workspace.json, README.txt | nothing |

The report's sections, in this order: Summary (the answer, the key numbers
table), Problem and decision, Site and data (the inventory table, the site
map), Methodology (numbered, with the gates and the assumptions), Results
(one subsection per step with its figures and tables), Limitations and what
this study does not establish, Recommendations, References, Appendix
(reproducibility: the study, the commands, the ledger).

## The faces

- **CLI**: `aquascope studio "PROBLEM" --lat --lon [--data FILE ...] [--provider ...] [--out DIR] [--yes] [--resume workspace.json]`.
- **MCP**: `studio_start`, `studio_say`, `studio_approve`, `studio_follow_up`, `studio_narrate`, `studio_context`, `studio_export`, the workspace dict in and out.
- **Explorer**: Study mode in the drawer (`explorer/src/studio.js`), a thin face over one worker message, `studio`, with `op` start, say, approve, follow_up, file, export (and table, an XLSX to CSV). `start` builds the Studio at the page's site with the attached tables and the page's catchment row (`describe_catchment` from it) and calls `say`; the others rebuild it with `Studio.from_dict` and call the method. Every reply is `{reply, workspace, status}` with the workspace WITHOUT artifact bytes; the worker keeps its own copy WITH them by workspace id, so `file` returns one artifact by id as base64 and `export` the bundle zip. `studio_progress` posts every event as it happens; `studio_artifact` posts every artifact, PNG figures with their bytes, SVG and CSV without. matplotlib is loaded with `loadPackage` before the first run, openpyxl and python-docx with micropip before the first bundle, never on a visit that runs no study.

## The honesty rules, kept

Every number in the report comes from a tool result and passes the gates
and the checks before it is quoted; what fails is listed under "what this
study does not establish". A method the registry calls not defensible at the
site is refused at plan time. The playbooks' caveats are printed verbatim
when a playbook applies. The study replays with no model. The ledger is in
the report's footer.
