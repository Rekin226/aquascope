# The Analyst: `aquascope ask` and `aquascope ingest`

Two commands where a language model does the part hydrologists spend hours on
and aquascope does the part that has to be right.

## `aquascope ask`: a question in, a cited report out

```bash
pip install aquascope             # the openai and anthropic SDKs are optional (pip install "aquascope[llm]")
export ANTHROPIC_API_KEY=...       # or GROQ_API_KEY, OPENAI_API_KEY, NVIDIA_API_KEY, HF_TOKEN, MISTRAL_API_KEY, OPENROUTER_API_KEY, or AQUASCOPE_LLM_API_KEY + _BASE_URL + _MODEL
aquascope ask "What is the 100-year flood of the Seine at Paris, and how sure can we be?" -o seine.md
```

The model gets the same tools the [MCP server](mcp.md) exposes (`find_stations`,
`analyze_station`, `flood_frequency`, `get_timeseries`, `anywhere`,
`describe_catchment`, `similar_basins`, `regionalize_signatures`,
`list_sources`, `describe_methods`) and decides which to call; aquascope runs
them against real data. The Markdown report has three parts:

1. the model's answer (told to quote units, periods, stations and confidence
   intervals, and never to invent numbers);
2. **Data**: every station or point the tools touched, with period, licence and
   attribution, assembled from the tool results;
3. **Methods and citations**: the methods actually applied, with references,
   also assembled from the tool results (never from the model's memory).

A footer records the model, provider, date and the tools called. `--provider`
picks anthropic / openai / groq / nvidia / huggingface / mistral / openrouter /
ollama (defaults from the environment, scanned in that order), `--model`
overrides the default model, `--max-steps` bounds the tool loop. Works with any
OpenAI-compatible endpoint that supports tool calling, and with Anthropic's
Messages API: `--provider anthropic` defaults to `claude-opus-5`, and
`AQUASCOPE_LLM_EFFORT` (`low` to `max`) sets how hard Claude thinks per step.
An identity-linked key that can act in several workspaces also needs
`ANTHROPIC_WORKSPACE_ID` (the `wrkspc_...` id from the console), which is sent
as the `anthropic-workspace-id` header; a key created for one workspace does
not.

This is deliberately not an autonomous agent: no memory, no planning beyond
the tool loop, no writes. It is the "ask, get the work done, see the work"
surface from the direction review, and its numbers are exactly what
`aquascope analyze`/the Explorer would give you.

The same function runs inside the [Explorer](explorer.md) (the **Ask ✨**
button): the browser worker calls the provider directly with your key through
`aquascope.ai_engine.llm_transport.UrllibChatClient`, a dependency-free
OpenAI-compatible client that is also the fallback when the `openai` package
is not installed, so `pip install aquascope` alone is enough for `aquascope
ask`. Claude goes through `AnthropicChatClient` in the same module, which
speaks the Messages API behind the same surface (the `anthropic` SDK when it
is installed, plain HTTP in the browser) and sends the model's own content
blocks back on every tool turn so adaptive thinking carries across steps.
Providers: `anthropic`, `openai`, `groq`, `nvidia` (CLI only), `huggingface`,
`mistral`, `openrouter`, `ollama`, or `AQUASCOPE_LLM_BASE_URL` for anything
else that speaks the chat-completions protocol.

### Supported Providers

| Provider | ID | Environment Variable | Default Model | Free Tier / Trial | Browser (Explorer) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Anthropic** | `anthropic` | `ANTHROPIC_API_KEY` | `claude-opus-5` | Paid | Yes |
| **OpenAI** | `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` | Paid | Yes |
| **Groq** | `groq` | `GROQ_API_KEY` | `openai/gpt-oss-120b` | Free tier (~1,000 req/day) | Yes |
| **NVIDIA Build** | `nvidia` | `NVIDIA_API_KEY` | `openai/gpt-oss-120b` | 1,000 trial credits on signup | No (CORS restricted; CLI only) |
| **Hugging Face** | `huggingface` | `HF_TOKEN` | `Qwen/Qwen2.5-72B-Instruct` | Monthly free tier credit | Yes |
| **Mistral** | `mistral` | `MISTRAL_API_KEY` | `mistral-small-latest` | Paid | Yes |
| **OpenRouter** | `openrouter` | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` | `:free` models available | Yes |
| **Ollama** | `ollama` | Local (`None`) | `qwen2.5:7b` | Free (runs locally) | No (requires local daemon) |

## From a question to a problem: `aquascope solve`

`ask` improvises: the model decides the next tool call after seeing the last
result, and the checks run on the finished answer. `solve` plans first. The
problem and the coordinates go through a reconnaissance of the site, a
playbook's decision tree fills a study with a gate per step, you see the
plan before it runs, the engine executes it with the gates, and a failed
gate runs the fallback or replans once:

```
intake ──► recon ──► plan ──► [you review] ──► execute ──► report
                                                 │  ▲
                                                 ▼  │ gate fails
                                               verify ─► replan (once)
```

What the gates establish: that a number was computed from a record long
enough for it, that a return period is within the record's reach, that an
interval is finite, that two fits agree, that a transfer had donors. What
they do not: that the record is right, that the catchment is the one the
question means, or anything about the future. Everything the gates and the
checks did not establish is listed under the answer. See [solve.md](solve.md).

## `aquascope ingest`: any export in, a clean series and a QA report out

```bash
aquascope ingest nwis_export.txt --unit cfs
aquascope ingest pegel.csv --variable water_level --date-column Datum --value-column "Pegel [cm]" --unit cm
aquascope ingest agency.xlsx --sheet 2 --llm --describe "monthly discharge in l/s from the regional office"
```

What happens:

- the file is read with the usual agency quirks handled (comment lines,
  `;`/tab delimiters, Excel sheets, JSON);
- the mapping (date column, value column, variable, unit and SI factor,
  station column) is guessed by heuristics, or proposed by an LLM when `--llm`
  is set and validated by the heuristics; anything you pass on the command
  line wins;
- the mapping is applied deterministically: sentinel values (-9999 and friends)
  are dropped *before* unit conversion, duplicates are dropped, timestamps are
  normalised;
- the QA report counts what was dropped, flags negatives and spikes (robust
  sigma), lists gaps over 30 days, computes coverage per year, and warns when
  the record is too holey for statistics;
- the cleaned series gets `aquascope.explore.analyze_series` (flood frequency
  when there are 10+ complete years, FDC, trend);
- outputs: `<stem>.csv` (`date,value` in SI units), `<stem>.qa.json`
  (mapping + QA), `<stem>.qa.md` (the human-readable report).

Nothing in `ingest` needs a key or a network connection unless you ask for
`--llm`.

## Reconnaissance first: `aquascope assess`

Before any analysis, inventory what exists at the place and what that record
supports. `assess_site(lat, lon)` in `aquascope.explore` reads the published
station catalog (true record spans, no agency call), asks BasinATLAS for the
catchment and the similarity search for donors, and scores every method in
`aquascope.methods` as defensible, marginal or not defensible here, with the
reason. One engine, every face: `aquascope assess`, the `assess_site` MCP
tool, the Analyst's first tool call for any question about a place or a
station (it will not run a method the table marks not defensible, and says
why), and the "What can be answered here" card in the Explorer.

```
$ aquascope assess 51.4150 -0.3080 --problem flood_risk --return-period 100
  51.4150, -0.3080  ·  25 gauges within 50 km  ·  catchment 9,991 km²  ·  10 donors
  discharge: 142.9 yr, Kingston (uk_ea/8496ce69-482c-406a-a2f0-ac418ef8f099)
  water level: 142.9 yr, Kingston (uk_ea/8496ce69-482c-406a-a2f0-ac418ef8f099)
  groundwater level: 16.9 yr, Teddington (uk_ea/9eaa9d56-ef35-4029-972b-404da217bf90)
  precipitation: 37.7 yr, Hogsmill (uk_ea/a04aa8e8-45a2-4d8d-9983-7a55330693b0)

  defensible
    At-site flood frequency (GEV, LP3 / Bulletin 17C)  the record supports it
    GloFAS modelled discharge as an independent check  the record supports it
    Flow signatures transferred from donors            the record supports it
    Mann-Kendall trend with Sen's slope                the record supports it

  marginal
    Donor gauges by catchment similarity               meant for an ungauged point; a gauge is available here

  notes
    - Record resolution is not in the catalog; daily is assumed for every variable.
    - The catalog lists Kingston (uk_ea/8496ce69-482c-406a-a2f0-ac418ef8f099) from 1883-10-01 (142.9 yr); a default fetch serves the last 40 years, so a computed answer covers fewer years than this span.
    - 10 donor gauges from a pool of 37,053 gauged catchments.
    - ERA5 temperature and forcing and GloFAS discharge are assumed reachable for any point on land (Open-Meteo); not checked here.
    - CMIP6 change factors need model output you supply (aquascope.climate works on downloaded data); not counted.
```

The same point with a 12-year record would put the 100-year flood in
*marginal* ("T = 100 years is beyond about 36 years, 3 times the record"); an
ungauged point marks every at-site method not defensible and leaves the
regionalisation path (`similar_basins`, `regionalize_signatures`) and the
GloFAS cross-check. `--radius-km` sets how far a gauge may be to count
(default 50), `--problem` narrows the table to one kind of question
(`flood_risk`, `ungauged_flow`, `drought`, `groundwater_decline`,
`supply_reliability`, `climate_change`, `irrigation`, `water_quality`), and
`--json` prints the full result: `point`, `stations` (nearest first),
`catchment`, `context` (years per variable, area, donors, what else is
available), `sufficiency` (one row per method, with the station it would
use) and `notes`. The notes are the honest part: the catalog does not record
resolution, so daily is assumed and said; a fetch serves the last 40 years
whatever the span; a source that publishes only the last month is named.

## Level 3: code, checks and a study you can run again

Three things were added in #234, and they change what an answer *is*.

### `run_python`

The fixed tools cover the questions we anticipated. For the rest, the model can
write a short snippet and run it where the library and the data already are:
in your own browser tab, or in your own shell.

```
aquascope ask "Compare the wettest decade with the driest at USGS-01013500"
  · tool find_stations(query='01013500')
  · tool analyze_station(source='usgs', station_id='USGS-01013500')
  · tool run_python(code='decades = df.groupby(df.index.year // 10 * 10)...')
```

`aquascope`, `workbench`, `pandas` (`pd`) and `numpy` (`np`) are already in
scope, along with whatever data the caller passed (the Explorer passes the
record on screen as `df`). Whatever the snippet leaves in `result` comes back.

Imports are checked against an allow-list before anything runs, and `eval`,
`exec`, `open`, `__import__` and dunder attribute access are refused. That is
not a security boundary and is not sold as one: the boundary is the platform.
In the Explorer the snippet runs inside the reader's own browser (WASM, their
data, their machine); in the CLI it runs with the same rights as the aquascope
process you started. Run it on data and questions you would run a script on.

### Checks under the answer

The Data and Methods sections were always assembled from tool output. The prose
between them is still the model's, so it is now checked, deterministically, by
`aquascope.ai_engine.verify`:

* every number in the answer appears in a tool result (allowing for rounding),
* a return level is quoted with its confidence interval,
* a claim about significance agrees with the test's p-value,
* the units and the record are named,
* at least one tool call actually succeeded.

Unmet checks are printed under the answer, in the report and in the Explorer's
drawer, as "What this answer does not establish". No model grades another model.

### `study.yaml`

Every answer now also comes with the steps that produced it:

```bash
aquascope ask "What is the 100-year flood of the Thames at Kingston?" --study study.yaml
aquascope run study.yaml --out results/
```

`aquascope run` calls the same tools with the same arguments, with no model in
the loop, and writes `report.md`, `manifest.json` (aquascope version, times and
a hash per step, so drift is visible) and `results.json`. The model writes the
study; the engine runs the study. That is the reproducible unit, and it is what
the declarative runner of #54 was for.
