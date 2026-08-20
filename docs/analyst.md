# The Analyst: `aquascope ask` and `aquascope ingest`

Two commands where a language model does the part hydrologists spend hours on
and aquascope does the part that has to be right.

## `aquascope ask`: a question in, a cited report out

```bash
pip install aquascope             # the openai SDK is optional (pip install "aquascope[llm]")
export GROQ_API_KEY=...            # or OPENAI_API_KEY, HF_TOKEN, MISTRAL_API_KEY, OPENROUTER_API_KEY, or AQUASCOPE_LLM_API_KEY + _BASE_URL + _MODEL
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
picks openai / groq / huggingface / ollama (defaults from the environment),
`--model` overrides the default model, `--max-steps` bounds the tool loop.
Works with any OpenAI-compatible endpoint that supports tool calling.

This is deliberately not an autonomous agent: no memory, no planning beyond
the tool loop, no writes. It is the "ask, get the work done, see the work"
surface from the direction review, and its numbers are exactly what
`aquascope analyze`/the Explorer would give you.

The same function runs inside the [Explorer](explorer.md) (the **Ask ✨**
button): the browser worker calls the provider directly with your key through
`aquascope.ai_engine.llm_transport.UrllibChatClient`, a dependency-free
OpenAI-compatible client that is also the fallback when the `openai` package
is not installed, so `pip install aquascope` alone is enough for `aquascope
ask`. Providers: `openai`, `groq`, `huggingface`, `mistral`, `openrouter`,
`ollama`, or `AQUASCOPE_LLM_BASE_URL` for anything else that speaks the
protocol.

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
