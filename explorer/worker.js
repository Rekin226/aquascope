// AquaScope Explorer, worker thread: Pyodide + aquascope. Fetches the observed
// record through aquascope's own collectors and runs aquascope.explore (the
// same code the CLI and MCP server use). Sync XHR (pyodide-http) is allowed in
// workers, so the page stays responsive while Python is busy.

let pyodide = null;
let ready = null;

function post(type, extra = {}) { self.postMessage({ type, ...extra }); }

async function init({ pyodideIndexURL, wheelsJson }) {
  post("progress", { text: "Loading Python runtime (Pyodide)…" });
  importScripts(`${pyodideIndexURL}pyodide.js`);
  pyodide = await loadPyodide({ indexURL: pyodideIndexURL });

  post("progress", { text: "Loading numpy, scipy, pandas…" });
  await pyodide.loadPackage(["micropip", "numpy", "scipy", "pandas", "pydantic", "httpx"]);

  post("progress", { text: "Installing aquascope…" });
  const wheels = await (await fetch(wheelsJson, { cache: "no-store" })).json();
  const wheelUrl = new URL(wheels.wheel, wheelsJson).href;
  const micropip = pyodide.pyimport("micropip");
  // The wheel keeps its filename between deploys, so a browser that has been
  // here before will happily serve yesterday's Python against today's page
  // (seen in the wild: a cached wheel without a module the page had just
  // started calling). Fetch it ourselves with cache: "reload", hand the bytes
  // to Pyodide's filesystem, and install from there.
  let wheelSpec = wheelUrl;
  try {
    const resp = await fetch(wheelUrl, { cache: "reload" });
    if (!resp.ok) throw new Error(`wheel ${resp.status}`);
    pyodide.FS.writeFile(`/tmp/${wheels.wheel}`, new Uint8Array(await resp.arrayBuffer()));
    wheelSpec = `emfs:/tmp/${wheels.wheel}`;
  } catch (err) {
    console.warn("could not pre-fetch the wheel, falling back to the URL:", err);
  }
  await micropip.install(["pyodide-http", wheelSpec]);

  await pyodide.runPythonAsync(`
import json, logging
logging.basicConfig(level=logging.WARNING)
import pyodide_http
pyodide_http.patch_all()
import aquascope.explore as analysis
_STORE = {}
`);
  post("ready");
}

// The full record is requested unless the page passes a cap in years (#270).
// The catalog's first date for the station travels with the request so Python
// can ask from it, and say in the note when the agency served less than that.
async function analyze({ id, source, station_id, years, period_start }) {
  post("progress", { text: "Fetching the record from the agency…" });
  const cap = Number(years) > 0 ? `years=${Math.round(Number(years))}, ` : "";
  const since = period_start ? `period_start=${JSON.stringify(String(period_start).slice(0, 10))}, ` : "";
  const code = `
import json
_STORE.clear()
_res = analysis.analyze_station(${JSON.stringify(source)}, ${JSON.stringify(station_id)}, ${cap}${since}store=_STORE)
_STORE["result"] = _res
json.dumps(_res)
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}

async function anywhere({ id, lat, lon, years }) {
  post("progress", { text: "Asking Open-Meteo about this point (ERA5 climate, GloFAS discharge)…" });
  const code = `
import json
_STORE.clear()
_res = analysis.anywhere(${Number(lat)}, ${Number(lon)}, years=${Number(years) || 10})
_STORE["result"] = _res
json.dumps(_res)
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}

async function floodCi({ id }) {
  const code = `
import json
json.dumps(analysis.flood_ci(_STORE["series"]))
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}

async function csv({ id }) {
  const out = await pyodide.runPythonAsync(`
analysis.to_csv(_STORE["result"])
`);
  post("result", { id, result: out });
}

// "What can be answered here": aquascope.explore.assess_site over the catalog
// the page handed over (send it first with "catalog"). The page passes the
// catchment area and donor count it already holds, since BasinATLAS and the
// similarity table are read by DuckDB-WASM on the main thread, not here.
async function assess({ id, lat, lon, radius_km, problem, area_km2, donors }) {
  post("progress", { text: "Checking what the record here supports…" });
  self.__aqAssess = JSON.stringify({
    lat: Number(lat), lon: Number(lon), radius_km: Number(radius_km) || 50, problem: problem || null,
    area_km2: Number.isFinite(Number(area_km2)) && area_km2 !== null ? Number(area_km2) : null,
    donors: Number.isFinite(Number(donors)) && donors !== null ? Number(donors) : null,
  });
  const code = `
import json
from js import __aqAssess
_a = json.loads(__aqAssess)
json.dumps(analysis.assess_site(
    _a["lat"], _a["lon"], radius_km=_a["radius_km"], problem=_a.get("problem"),
    area_km2=_a.get("area_km2"), donors=_a.get("donors"),
))
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqAssess = null;
  }
}

// The main thread already holds the station catalog (DuckDB-WASM); hand it to
// Python once so find_stations() answers from memory instead of the Hub
// (httpx / pyarrow do not run here).
let catalogLoaded = false;
async function catalog({ id, rows }) {
  self.__aqCatalog = JSON.stringify(rows);
  await pyodide.runPythonAsync(`
import json
from js import __aqCatalog
from aquascope.archive import catalog as _catalog
_catalog.set_catalog(json.loads(__aqCatalog))
`);
  self.__aqCatalog = null;
  catalogLoaded = true;
  post("result", { id, result: { n: rows.length } });
}

// The Analyst (aquascope.ai_engine.analyst.ask) runs unchanged in the browser:
// the OpenAI-compatible call goes through urllib (sync XHR via pyodide-http),
// straight from this worker to the provider the user picked. The key never
// touches any server of ours (there is none).
async function ask({ id, question, provider, model, api_key, base_url, max_steps }) {
  self.__aqAskEvent = (text) => post("ask_progress", { id, text: String(text) });
  self.__aqAsk = JSON.stringify({ question, provider, model, api_key, base_url, max_steps: Number(max_steps) || 8 });
  const code = `
import json
from js import __aqAsk, __aqAskEvent
from aquascope.ai_engine import analyst as _analyst
_args = json.loads(__aqAsk)
_res = _analyst.ask(
    _args["question"],
    provider=_args.get("provider") or None,
    model=_args.get("model") or None,
    api_key=_args.get("api_key") or None,
    base_url=_args.get("base_url") or None,
    max_steps=int(_args.get("max_steps") or 8),
    on_event=lambda m: __aqAskEvent(m),
    # The record on screen, so run_python can work on it (#234).
    data={"df": _STORE["frame"]} if _STORE.get("frame") is not None else None,
)
json.dumps({
    "answer": _res.answer,
    "markdown": _res.to_markdown(),
    "model": _res.model,
    "provider": _res.provider,
    "steps": _res.steps,
    "tool_calls": [{"name": c.name, "arguments": c.arguments, "ok": c.ok} for c in _res.tool_calls],
    "data_used": _res.data_used,
    "methods": _res.methods,
    "checks": _res.checks,
    "verified": _res.verified,
    "study": _res.study,
})
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqAsk = null;
    self.__aqAskEvent = null;
  }
}

// ── Solve: a problem at a place, planned first ──────────────────────────────
// The two halves of aquascope.ai_engine.team, the same code the CLI and the MCP
// server run. The page has already run the reconnaissance (with the catchment
// area and donor count only it can read), so it travels in as `recon` and the
// Scout is not asked again. A model is used only when the page passes one.

function solveArgs() {
  return `provider=_a.get("provider") or None, model=_a.get("model") or None,
    api_key=_a.get("api_key") or None, base_url=_a.get("base_url") or None`;
}

// The plan half: the playbook the chips or the keyword rules pick, the branch
// the tree selects for the data that exists, the study it fills. Nothing runs.
async function solvePlan({ id, problem, lat, lon, playbook, intake, recon, provider, model, api_key, base_url }) {
  self.__aqSolve = JSON.stringify({
    problem: problem || "", lat: Number(lat), lon: Number(lon), playbook: playbook || null,
    intake: intake || null, recon: recon || null, provider, model, api_key, base_url,
  });
  const code = `
import json
from js import __aqSolve
from aquascope.ai_engine import team as _team
_a = json.loads(__aqSolve)
_res = _team.solve(
    _a["problem"], lat=_a["lat"], lon=_a["lon"], playbook=_a.get("playbook"), intake=_a.get("intake"),
    recon=_a.get("recon"), ${solveArgs()},
    execute=False,
)
json.dumps(_res.to_dict(), default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqSolve = null;
  }
}

// The intake a small model wrote on the reader's device, made safe by the
// package's own rules (aquascope.playbooks.coerce_intake): a field the playbook
// has not got is dropped, a value the field cannot take becomes its default.
// An unknown playbook comes back as null, and the page falls back to the
// keyword rules solve_plan applies anyway.
async function coerceIntake({ id, playbook, intake }) {
  self.__aqIntake = JSON.stringify({ playbook: playbook || null, intake: intake || null });
  const code = `
import json
from js import __aqIntake
from aquascope import playbooks as _pbk
_a = json.loads(__aqIntake)
try:
    _pb = _pbk.load(_a["playbook"] or "")
    _out = {"playbook": _pb.id, "intake": _pbk.coerce_intake(_pb, _a.get("intake"))}
except _pbk.PlaybookError as exc:
    _out = {"playbook": None, "intake": None, "error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqIntake = null;
  }
}

// The run half: the reviewed study (edited or not) with its gates, one bounded
// replan, the Reviewer's "not established" list and the Narrator. Every
// timeline event is posted as it happens, the way ask() streams its tool log.
// BasinATLAS cannot be read here (no pyogrio in Pyodide), so the sub-basin and
// attribute row the page found with DuckDB and FlatGeobuf travel in as
// `catchment`, and the package builds describe_catchment's payload from them.
async function solveRun({ id, study, recon, catchment, provider, model, api_key, base_url }) {
  self.__aqSolveEvent = (text) => post("solve_progress", { id, event: JSON.parse(text) });
  self.__aqSolve = JSON.stringify({ study, recon: recon || null, catchment: catchment || null, provider, model, api_key, base_url });
  const code = `
import json
from js import __aqSolve, __aqSolveEvent
from aquascope.ai_engine import team as _team
_a = json.loads(__aqSolve)
_tools = {}
_c = _a.get("catchment")
if _c and (_c.get("sub_basin") or {}).get("hybas_id") is not None:
    from aquascope.archive import basins as _basins
    _tools["describe_catchment"] = lambda lat=None, lon=None, **_kw: _basins.describe_catchment_from_row(
        lat, lon, _c["sub_basin"], _c.get("row"), n_upstream=_c.get("n_upstream"))
_res = _team.run_reviewed(
    _a["study"], recon=_a.get("recon"), ${solveArgs()},
    on_event=lambda e: __aqSolveEvent(json.dumps(e, default=str)),
    tools=_tools or None,
)
json.dumps(_res.to_dict(), default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqSolve = null;
    self.__aqSolveEvent = null;
  }
}


// ── the workbench: analyses of the user's own table ─────────────────────────
// aquascope.workbench holds what the dashboard pages used to hold, as plain
// functions returning JSON, so the browser runs exactly what the CLI runs.

async function ingestText({ id, text, filename, options }) {
  post("progress", { text: "Reading the file and working out its columns…" });
  self.__aqIngest = JSON.stringify({ text, filename: filename || "upload.csv", options: options || {} });
  const code = `
import json
from js import __aqIngest
from aquascope import ingest as _ingest
_args = json.loads(__aqIngest)
_res = _ingest.ingest_text(_args["text"], _args["filename"], **(_args.get("options") or {}))
_STORE["frame"] = _res["series"].rename("value").to_frame().reset_index().rename(columns={"index": "date"})
_STORE["result"] = _res["analysis"]
json.dumps({
    "mapping": _res["mapping"],
    "qa": _res["qa"],
    "analysis": _res["analysis"],
    "n": int(len(_res["series"])),
})
`;
  const out = await pyodide.runPythonAsync(code);
  self.__aqIngest = null;
  post("result", { id, result: JSON.parse(out) });
}

// A table the page already holds (CSV text), kept for the workbench analyses.
async function loadTable({ id, csv, label }) {
  self.__aqCsv = csv;
  const code = `
import json, io
import pandas as pd
from js import __aqCsv
_STORE["frame"] = pd.read_csv(io.StringIO(__aqCsv))
from aquascope import workbench as _wb
json.dumps({"n": int(len(_STORE["frame"])), "columns": [str(c) for c in _STORE["frame"].columns],
            "insights": _wb.insights(_STORE["frame"])})
`;
  const out = await pyodide.runPythonAsync(code);
  self.__aqCsv = null;
  post("result", { id, result: { ...JSON.parse(out), label: label || "table" } });
}

async function workbench({ id, analysis, params }) {
  post("progress", { text: `Running ${analysis}…` });
  self.__aqWb = JSON.stringify({ analysis, params: params || {} });
  const code = `
import json
from js import __aqWb
from aquascope import workbench as _wb
_a = json.loads(__aqWb)
_frame = _STORE.get("frame")
_res = _wb.run(_a["analysis"], _frame, **(_a.get("params") or {}))
_res.pop("frame", None)
json.dumps(_res)
`;
  const out = await pyodide.runPythonAsync(code);
  self.__aqWb = null;
  post("result", { id, result: JSON.parse(out) });
}

// The gauge record currently on screen, handed to the workbench.
async function frameFromStation({ id }) {
  const code = `
import json
import pandas as pd
_res = _STORE.get("result") or {}
_series = _res.get("series") or {}
_STORE["frame"] = pd.DataFrame({"date": pd.to_datetime(_series.get("t", [])), "discharge": _series.get("v", [])})
from aquascope import workbench as _wb
json.dumps({"n": int(len(_STORE["frame"])), "columns": ["date", "discharge"],
            "insights": _wb.insights(_STORE["frame"])})
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}


// Run one analyst tool by name, for the showcase's "run the tools again": the
// deterministic half of a recorded answer, live, with no model and no key.
async function runTool({ id, name, arguments: args }) {
  self.__aqTool = JSON.stringify({ name, args: args || {} });
  const code = `
import json
from js import __aqTool
from aquascope.ai_engine import analyst as _analyst
_a = json.loads(__aqTool)
_specs = {s.name: s for s in _analyst._tool_specs()}
_spec = _specs.get(_a["name"])
if _spec is None:
    _out = {"error": f"unknown tool {_a['name']}"}
else:
    try:
        _out = _spec.func(**(_a.get("args") or {}))
    except Exception as exc:
        _out = {"error": f"{type(exc).__name__}: {exc}"}
json.dumps(_out, default=str)
`;
  const out = await pyodide.runPythonAsync(code);
  self.__aqTool = null;
  post("result", { id, result: JSON.parse(out) });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") { ready = init(m); await ready; return; }
    await ready;
    if (m.type === "analyze") return await analyze(m);
    if (m.type === "anywhere") return await anywhere(m);
    if (m.type === "assess") return await assess(m);
    if (m.type === "flood_ci") return await floodCi(m);
    if (m.type === "csv") return await csv(m);
    if (m.type === "catalog") return await catalog(m);
    if (m.type === "ask") return await ask(m);
    if (m.type === "solve_plan") return await solvePlan(m);
    if (m.type === "coerce_intake") return await coerceIntake(m);
    if (m.type === "solve_run") return await solveRun(m);
    if (m.type === "ingest") return await ingestText(m);
    if (m.type === "load_table") return await loadTable(m);
    if (m.type === "workbench") return await workbench(m);
    if (m.type === "tool") return await runTool(m);
    if (m.type === "frame_from_station") return await frameFromStation(m);
  } catch (err) {
    // Pyodide raises PythonError with the full traceback in .message; keep the
    // exception line (last non-empty) and log the whole thing for debugging.
    const full = String(err && err.message ? err.message : err);
    console.error(full);
    const lines = full.split("\n").filter((l) => l.trim());
    post("error", { id: m.id, message: lines[lines.length - 1] || "unknown error" });
  }
};
