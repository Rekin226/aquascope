// AquaScope Explorer, worker thread: Pyodide + aquascope. Fetches the observed
// record through aquascope's own collectors and runs aquascope.explore (the
// same code the CLI and MCP server use). Sync XHR (pyodide-http) is allowed in
// workers, so the page stays responsive while Python is busy. Started as a module
// worker ({ type: "module" } in worker-client.js): Pyodide 314 no longer runs in
// classic workers, so the runtime comes in through import(), not importScripts.

let pyodide = null;
let ready = null;
// A light worker (worker-client.js's pool) loads no pandas or scipy and answers only the calls that read the
// network without them: a place's context layers, a click's river snap and the quick forecast. Several run at
// once, so those reads no longer queue behind the main worker's record, one sync request after another.
let lite = false;
const LITE_TYPES = new Set(["context", "river", "now", "map_command", "scout", "map_explain", "mesh_review",
  "whatif_words", "flood_exposure"]);

function post(type, extra = {}) { self.postMessage({ type, ...extra }); }

async function init({ pyodideIndexURL, wheelsJson, lite: light = false }) {
  lite = Boolean(light);
  post("progress", { text: "Loading Python runtime (Pyodide)…" });
  const { loadPyodide } = await import(`${pyodideIndexURL}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: pyodideIndexURL });

  post("progress", { text: lite ? "Loading numpy…" : "Loading numpy, scipy, pandas…" });
  await pyodide.loadPackage(lite ? ["micropip", "numpy", "pydantic", "httpx"]
    : ["micropip", "numpy", "scipy", "pandas", "pydantic", "httpx"]);

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
  if (lite) {
    // Without the wheel's dependencies: pandas and scipy are what make the main worker heavy.
    await micropip.install("pyodide-http");
    await micropip.install.callKwargs(wheelSpec, { deps: false });
  } else {
    await micropip.install(["pyodide-http", wheelSpec]);
  }
  pyodide.globals.set("_aq_build_revision", String(wheels.build || "unknown"));
  pyodide.globals.set("_aq_lite", lite);

  await pyodide.runPythonAsync(`
import json, logging, os
os.environ["AQUASCOPE_REVISION"] = _aq_build_revision
logging.basicConfig(level=logging.WARNING)
import pyodide_http
try:
    pyodide_http.patch_all()
except ImportError:  # no requests in a light worker
    pyodide_http.patch_urllib()
if not _aq_lite:
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

// Rivers as objects (#516): aquascope.rivers, the same functions as `aquascope river` and the MCP tools.
// snap reads a few byte ranges of the GEOGLOWS stream tiles; record asks the GEOGLOWS API for the reach's
// simulated daily flow since 1940; trace reads the processing unit's routing tables (a few MB, up to about
// 30 MB for the largest basins), the catalog the page sent with "catalog", the Archive's Global Dam Watch
// cells along the path and upstream, a few zoom-8 stream tiles, and Natural Earth's borders (750 kB, once).
// network (#545) reads the same routing tables and returns only ids, which the map lights up.
async function river({ id, op, args }) {
  // The arguments travel inside the code as a JSON string literal, not through a shared global: two river
  // calls can be in flight (a record still running when the next click snaps), and a global set by one
  // and cleared by the other is read as null.
  const payload = JSON.stringify(JSON.stringify({ op: String(op || ""), args: args || {} }));
  const code = `
import json
from aquascope import rivers as _rivers
_a = json.loads(${payload})
_op, _k = _a["op"], _a["args"]
if _op == "snap":
    _res = _rivers.snap_to_river(_k["lat"], _k["lon"], max_distance_m=_k.get("max_distance_m") or 1000.0,
                                 prefer=_k.get("prefer") or "main", area_km2=_k.get("area_km2"))
elif _op == "record":
    _res = _rivers.reach_record(_k.get("river_id"), lat=_k.get("lat"), lon=_k.get("lon"))
elif _op == "trace":
    _res = _rivers.trace_downstream(_k.get("river_id"), lat=_k.get("lat"), lon=_k.get("lon"),
                                    gauge_km=_k.get("gauge_km") or 2.0, max_points=3000)
elif _op == "area":
    _res = _rivers.upstream_area(_k["river_id"], lat=_k.get("lat"), lon=_k.get("lon"))
elif _op == "network":
    _res = {"upstream": _rivers.upstream_ids(_k["river_id"], max_n=_k.get("max_up") or 20000,
                                             lat=_k.get("lat"), lon=_k.get("lon")),
            "downstream": _rivers.downstream_ids(_k["river_id"], max_n=_k.get("max_down") or 5000,
                                                 lat=_k.get("lat"), lon=_k.get("lon"))}
else:
    raise ValueError(f"unknown river operation {_op!r}")
json.dumps(_res, default=str)
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}

// The evidence ladder (#518): aquascope.evidence.model_skill on the record the page just analysed (the same
// function as `aquascope evidence skill` and the MCP tool). GEOGLOWS and GloFAS are read here; the NWM and
// Google GRRR rows come from the published skill table, which the page reads with DuckDB and passes in.
async function evidence({ id, args }) {
  const payload = JSON.stringify(JSON.stringify(args || {}));
  const code = `
import json
from aquascope import evidence as _ev
_k = json.loads(${payload})
_same = (_STORE.get("source"), str(_STORE.get("station_id"))) == (_k.get("source"), str(_k.get("station_id")))
_res = _ev.model_skill(_k.get("source"), _k.get("station_id"), series=_STORE.get("series") if _same else None,
                       lat=_k.get("lat"), lon=_k.get("lon"), area_km2=_k.get("area_km2"),
                       published=_k.get("published") or [], include_series=True)
json.dumps(_res, default=str)
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}

// Now and next (#517): aquascope.nownext, the same functions as `aquascope now` and the MCP tools. op "status"
// places the stored gauge record's newest day (topped up from the agency) against the same days in other years;
// op "forecast" reads GEOGLOWS and GloFAS for the reach and, with use_gauge, corrects GEOGLOWS to the stored
// record. history false is the quick forecast (no 86-year simulated record, so no thresholds), which a light
// worker answers first; known_geoglows hands the full call that answer's GEOGLOWS part so it is not read twice.
// The gauge in the request must be the one in _STORE, so a reply for a gauge the reader has left is never
// computed from the next one's record.
async function nowNext({ id, op, args }) {
  const payload = JSON.stringify(JSON.stringify({ op: String(op || ""), args: args || {} }));
  const code = `
import json
from aquascope import nownext as _nn
_a = json.loads(${payload})
_op, _k = _a["op"], _a["args"]
_mine = (_STORE.get("source"), _STORE.get("station_id")) == (_k.get("source"), _k.get("station_id"))
_r = _STORE.get("result") or {}
if _op == "status":
    if not _mine or _STORE.get("series") is None:
        _res = {"error": "This gauge's record is not loaded any more."}
    else:
        _res = _nn.station_status(_k["source"], _k["station_id"], series=_STORE["series"],
                                  variable=_r.get("variable"), unit=_r.get("unit"))
elif _op == "forecast":
    _obs = _STORE.get("series") if (_k.get("use_gauge") and _mine) else None
    _res = _nn.forecast(_k.get("lat"), _k.get("lon"), river_id=_k.get("river_id"), obs=_obs,
                        match_mean_flow=_k.get("match_mean_flow"), snap=False,
                        history=_k.get("history", True), glofas=_k.get("glofas", True),
                        known_geoglows=_k.get("known_geoglows"))
elif _op == "plume":
    # The FEWS view of a reach (#556): the 51 members day by day, classed against the thresholds the page passes
    # (the Floods ahead layer's), with a gauge's record when the page has one. No pandas: a light worker answers.
    _res = _nn.plume(_k.get("river_id"), thresholds=_k.get("thresholds"), obs=_k.get("obs"),
                     days=_k.get("days", 15), run=_k.get("run"), look_up=False)
elif _op == "points":
    # The Archive's forecast gauges, classed against their own return-period flows, from one issued file's rows.
    _res = _nn.forecast_points(_k.get("rows") or [])
else:
    raise ValueError(f"unknown now operation {_op!r}")
json.dumps(_res, default=str)
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
analysis.to_csv(_STORE["result"], series=_STORE.get("series"))
`);
  post("result", { id, result: out });
}

// "Export for...": aquascope.io.engineering over the stored record. op "menu" lists the tools that take
// this variable; op "export" returns one tool's files as a base64 zip (text formats only: DSS goes as the
// CSV hecdss reads, since its native library cannot load here).
async function engineering({ id, op, tool, name, lat, lon }) {
  self.__aqEng = JSON.stringify({
    op: op === "menu" ? "menu" : "export", tool: String(tool || ""), name: name ? String(name) : null,
    lat: Number.isFinite(Number(lat)) && lat !== null ? Number(lat) : null,
    lon: Number.isFinite(Number(lon)) && lon !== null ? Number(lon) : null,
  });
  const code = `
import json
from js import __aqEng
from aquascope.io import engineering as _eng
_a = json.loads(__aqEng)
_r = _STORE.get("result") or {}
if _a["op"] == "menu":
    _out = _eng.menu(_r.get("variable"))
else:
    _out = _eng.export_series(_STORE["series"], _a["tool"], variable=_r.get("variable"), unit=_r.get("unit"),
                              location=_STORE.get("station_id"), name=_a.get("name"), source=_STORE.get("source"),
                              lat=_a.get("lat"), lon=_a.get("lon"), with_text=False, as_zip=True, dss_binary=False)
json.dumps(_out)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqEng = null;
  }
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

// My places, Compare: aquascope.compare over two to five gauges. The page
// passes each gauge's catchment area, which it reads from its own table.
async function compare({ id, stations, years }) {
  post("progress", { text: "Fetching the records to compare…" });
  self.__aqCompare = JSON.stringify({ stations: stations || [], years: Number(years) > 0 ? Math.round(Number(years)) : null });
  const code = `
import json
from js import __aqCompare
import aquascope.compare as _compare
_a = json.loads(__aqCompare)
json.dumps(_compare.compare_stations(_a["stations"], years=_a["years"]))
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqCompare = null;
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
// The Explorer's drawer moved from Solve to Study (below); these messages stay
// for the other faces that mirror them, and cost nothing while unused.

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



// ── Studio: a complete study at a place, by a crew of roles ─────────────────
// aquascope.studio.Studio, the same Coordinator the CLI and the MCP tools run.
// The page holds the workspace dict between calls; the worker keeps its own
// copy WITH the artifact bytes (_STUDIO, by workspace id) so a figure or a
// document never crosses to the page except on request: `file` returns one
// artifact by id, `export` the bundle zip. Events and PNG figures are posted
// as they happen. matplotlib is loaded before the first run, openpyxl and
// python-docx before the first bundle, never on a visit that runs no study.
//
// A model on the reader's device joins the crew through the same message:
// `prompts` hands the page the crew's prompts and schemas, `context` the
// compact context a role would send (methodologist, author, consultant) with
// its system prompt, `check_plan` the validator's verdict on a plan the page
// wrote, and `start`/`say` take a `proposed` brief, `approve` a `plan`, and
// `narrate` the sections, each validated by the engine and falling back to
// the tree and the templates. Every one of those is guarded: an engine
// without the method answers {error} and the page keeps the keyless path.
//
// The Python between the markers is plain functions over dicts, so the test
// suite can run it in CPython against the studio fixtures.

const STUDIO_PY = `
# --- studio face (explorer) ---
import base64 as _b64
import io as _io
import json as _json

from aquascope.studio import Studio as _Studio

_STUDIO = {}      # workspace id -> Studio, with the artifact bytes


def _studio_tools(catchment, donors=None):
    """describe_catchment from the sub-basin row the page found (BasinATLAS is read by DuckDB-WASM on the
    main thread; pyogrio does not run here), and the donor tools (similar_basins, regionalize_signatures)
    over the station catchments and signatures tables the page read the same way: nothing here opens a
    parquet file."""
    tools = {}
    c = catchment or {}
    if (c.get("sub_basin") or {}).get("hybas_id") is None:
        return tools
    from aquascope.archive import basins as _basins

    def describe(lat=None, lon=None, **_kw):
        return _basins.describe_catchment_from_row(lat, lon, c["sub_basin"], c.get("row"),
                                                   n_upstream=c.get("n_upstream"))

    tools["describe_catchment"] = describe
    d = donors or {}
    if not d.get("catchments"):
        return tools
    import pandas as _pd
    from aquascope.archive import regionalize as _rg
    from aquascope.archive import similar as _similar

    table = _pd.DataFrame(d["catchments"])
    sig = _pd.DataFrame(d["signatures"]) if d.get("signatures") else None
    skill = d.get("skill")

    def similar_basins(lat=None, lon=None, source=None, station_id=None, k=10, method="combined", **_kw):
        kk = max(1, min(int(k or 10), 50))
        if source and station_id:
            return _similar.similar_for_station(source, station_id, k=kk, method=method, table=table)
        return _similar.similar_for_point(float(lat), float(lon), k=kk, method=method, desc=describe(lat, lon),
                                          table=table)

    tools["similar_basins"] = similar_basins
    if sig is not None:
        def regionalize_signatures(lat=None, lon=None, k=10, method="similarity", **_kw):
            kk = max(1, min(int(k or 10), 50))
            return _rg.regionalize_point(float(lat), float(lon), k=kk, method=method, desc=describe(lat, lon),
                                         table=table, signatures=sig, skill=skill)

        tools["regionalize_signatures"] = regionalize_signatures
    return tools


def _with_recon_context(a, fn):
    """Run fn with the Scout's assess_site carrying the catchment area and donor count only the page can
    read (BasinATLAS and the donor table are DuckDB-WASM reads on the main thread); restored afterwards."""
    import aquascope.explore as _ex

    base = _ex.assess_site
    ctx = {"area_km2": a.get("area_km2"), "donors": a.get("donors")}

    def assess_site(lat, lon, **kw):
        for key, value in ctx.items():
            if kw.get(key) is None and value is not None:
                kw[key] = value
        return base(lat, lon, **kw)

    _ex.assess_site = assess_site
    try:
        return fn()
    finally:
        _ex.assess_site = base


def _studio_model(a):
    return {"provider": a.get("provider") or None, "model": a.get("model") or None,
            "api_key": a.get("api_key") or None, "base_url": a.get("base_url") or None}


def _studio_open(a, on_event, on_artifact):
    """The Studio for this call: from the worker's own copy (with the bytes) when it has one, else from the
    page's workspace dict. Rebuilt every call so the model and the callbacks are this call's."""
    ws = a.get("workspace") or {}
    kept = _STUDIO.get(ws.get("id"))
    d = kept.to_dict() if kept is not None else ws
    s = _Studio.from_dict(d, tools=_studio_tools(a.get("catchment"), a.get("donors_tables")), on_event=on_event, on_artifact=on_artifact,
                          **_studio_model(a))
    _STUDIO[s.ws.id] = s
    return s


def _studio_reply(s, r):
    return {"reply": r.to_dict(), "workspace": s.to_dict(with_artifacts=False), "status": s.ws.status,
            "plain": _studio_plain(s)}


def _studio_plain(s):
    """The plan in plain words and each step's controls (aquascope.studio.steering), for the page; None when this
    engine has no steering or the study cannot be read."""
    try:
        from aquascope.studio.steering import plain_plan as _plain_plan

        return _plain_plan(s.ws.study) or None
    except Exception:  # noqa: BLE001 - the page falls back to the raw gates
        return None


def _studio_file(art):
    return {"id": art.id, "name": art.name, "media_type": art.media_type, "size": art.size,
            "data": _b64.b64encode(art.data).decode("ascii")}


def _accepts(fn, name):
    """Whether an engine method takes the keyword this face wants to pass. The face is built against the
    bring-your-own-model contract (proposed, plan, narrate, the role contexts); an engine without it gets the
    call it knows, and the page is told what did not happen."""
    import inspect

    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _proposed(a):
    """What a model on the reader's device wrote for the Consultant: the page's proposed dict, or the older
    brief field wrapped the same way."""
    proposed = a.get("proposed")
    if isinstance(proposed, dict) and proposed.get("brief"):
        return {"brief": dict(proposed["brief"]), "source": str(proposed.get("source") or "device")}
    brief = a.get("brief")
    if isinstance(brief, dict) and brief:
        return {"brief": dict(brief), "source": "device"}
    return None


def _studio_say(s, text, proposed):
    if proposed and _accepts(s.say, "proposed"):
        return s.say(text, proposed=proposed)
    r = s.say(text)
    if proposed:
        # The engine has no proposed brief yet: the fields the device read go straight in, as before.
        brief = proposed["brief"]
        if isinstance(brief.get("decision"), str) and brief["decision"].strip():
            s.ws.brief.decision = brief["decision"].strip()
        if isinstance(brief.get("quantities"), list):
            s.ws.brief.quantities = [str(q) for q in brief["quantities"] if str(q).strip()]
        if s.ws.brief.source == "rules":
            s.ws.brief.source = proposed["source"]
    return r


def studio_prompts():
    """The crew's prompts and schemas as one JSON-able dict, for a model the page runs; None when this engine
    has no export (the page then reads only the brief on the device)."""
    try:
        from aquascope.studio import prompts as _prompts
    except ImportError:
        return None
    as_json = getattr(_prompts, "as_json", None)
    if as_json is None:
        return None
    out = as_json()
    return _json.loads(out) if isinstance(out, str) else out


def _studio_context(s, a):
    """The compact context a role would send to its model, with the system prompt under "system"."""
    role = str(a.get("role") or "")
    fn = getattr(s, f"{role}_context", None) if role in ("consultant", "methodologist", "author") else None
    if fn is None:
        return {"error": f"no {role or 'such'} context in this engine"}
    ctx = fn(str(a.get("text") or "")) if role == "consultant" else fn()
    if not isinstance(ctx, dict):
        return {"error": f"the {role} context is not a dict"}
    return ctx


def _studio_check_plan(s, a):
    """The validator's verdict on a plan written outside the crew, before the page shows it: ok, the errors, the
    normalised steps. ok is None when the engine lends no validator (the approval still checks)."""
    try:
        from aquascope.studio.roles import methodologist as _meth
    except ImportError:
        return {"ok": None, "errors": [], "steps": []}
    check = getattr(_meth, "_check", None)
    if check is None:
        return {"ok": None, "errors": [], "steps": []}
    steps, errors, notes = check(dict(a.get("plan") or {}), s.ws)
    return {"ok": not errors, "errors": list(errors), "notes": list(notes), "steps": list(steps)}


def studio_call(a, on_event=None, on_artifact=None, store=None):
    """One message from the page. op is start, say, approve, follow_up, steer, narrate, context, check_plan,
    prompts, file or export."""
    return _with_recon_context(a, lambda: _studio_dispatch(a, on_event, on_artifact, store))


def _studio_dispatch(a, on_event, on_artifact, store):
    op = a.get("op")
    if op == "import_portable":
        from aquascope.studio.portable import loads as _load_portable

        ws = _load_portable(str(a.get("text") or ""))
        _STUDIO.pop(ws.id, None)
        return {"workspace": ws.to_dict(with_artifacts=True)}
    if op == "prompts":
        return studio_prompts()
    if op == "start":
        tables = dict(a.get("tables") or {})
        frame = (store or {}).get("frame") if store is not None else None
        if a.get("use_frame") and frame is not None:
            tables[str(a.get("frame_label") or "my-data")] = frame.to_csv(index=False)
        s = _Studio(float(a["lat"]), float(a["lon"]), data=tables or None, intake=a.get("intake") or None,
                    tools=_studio_tools(a.get("catchment"), a.get("donors_tables")), on_event=on_event, on_artifact=on_artifact,
                    **_studio_model(a))
        _STUDIO[s.ws.id] = s
        return _studio_reply(s, _studio_say(s, str(a.get("text") or ""), _proposed(a)))
    if op in ("link", "open_link"):   # study links: encode a plan, or decode and check one (aquascope.study_link)
        from aquascope.study_link import studio_op as _link_op

        return _link_op(a)
    s = _studio_open(a, on_event, on_artifact)
    if op == "portable":
        from aquascope.studio.portable import dumps as _dump_portable

        data = _dump_portable(s.ws).encode("utf-8")
        return {"name": f"study-{s.ws.id}.aqstudy.json", "media_type": "application/json",
                "data": _b64.b64encode(data).decode("ascii"), "size": len(data)}
    if op == "add_table":
        return _studio_reply(s, s.add_table(str(a.get("name") or "table"), str(a.get("csv") or "")))
    if op == "say":
        for _name, _csv in (a.get("tables") or {}).items():
            s.add_table(str(_name), str(_csv))
        return _studio_reply(s, _studio_say(s, str(a.get("text") or ""), _proposed(a)))
    if op == "approve":
        edits = a.get("edits") or None
        plan = a.get("plan") if isinstance(a.get("plan"), dict) else None
        if plan and _accepts(s.approve, "plan"):
            return _studio_reply(s, s.approve(edits, plan=dict(plan, source=str(plan.get("source") or "device"))))
        r = s.approve(edits)
        if plan:
            r.payload.setdefault("plan_used", "tree")
            r.payload.setdefault("plan_errors", ["this engine does not take a proposed plan"])
        return _studio_reply(s, r)
    if op == "follow_up":
        return _studio_reply(s, s.follow_up(str(a.get("text") or "")))
    if op == "steer":
        steer = getattr(s, "steer", None)
        if steer is None:
            return {"error": "adjusting a step is not available in this engine"}
        return _studio_reply(s, steer(str(a.get("step_id") or ""), dict(a.get("changes") or {})))
    if op == "desk":   # the Study Desk: levers, sensitivity, revisions, review, sign-off (aquascope.studio.desk)
        try:
            from aquascope.studio.desk import studio_op as _desk_op
        except ImportError:
            return {"error": "the Study Desk is not available in this engine"}
        out = _desk_op(s, a)
        if "reply" in out:
            res = _studio_reply(s, out.pop("reply"))
            res["desk"] = out.get("desk")
            return res
        out["workspace"] = s.to_dict(with_artifacts=False)
        return out
    if op == "narrate":
        narrate = getattr(s, "narrate", None)
        if narrate is None:
            return {"error": "narrate is not available in this engine"}
        sections = {str(k): str(v) for k, v in (a.get("sections") or {}).items() if isinstance(v, str) and v.strip()}
        return _studio_reply(s, narrate(sections=sections, source=str(a.get("source") or "device")))
    if op == "context":
        return _studio_context(s, a)
    if op == "check_plan":
        return _studio_check_plan(s, a)
    if op == "file":
        art = s.ws.artifact(str(a.get("artifact_id") or ""))
        if art is None:
            return {"error": f"no artifact {a.get('artifact_id')!r} in this study"}
        if not art.data:
            # A study resumed from the page's copy (a reload, a dropped workspace.json, a stopped run): the
            # bytes were never here. The next run makes them again.
            return {"error": f"the bytes of {art.name} are not in this session; a follow-up remakes the files"}
        return _studio_file(art)
    if op == "export":
        from aquascope.studio.deliverables.bundle import bundle_bytes

        data = bundle_bytes(s.ws)
        return {"id": "bundle", "name": "bundle.zip", "media_type": "application/zip", "size": len(data),
                "data": _b64.b64encode(data).decode("ascii")}
    return {"error": f"unknown op {op!r}"}


def studio_table(name, data_b64):
    """An Excel upload as CSV text, so it travels in the workspace like any other table."""
    import pandas as pd

    df = pd.read_excel(_io.BytesIO(_b64.b64decode(data_b64)))
    return {"name": name, "csv": df.to_csv(index=False), "n": int(len(df)), "columns": [str(c) for c in df.columns]}
# --- end studio face ---
`;

let studioDefined = false;
let plottingLoaded = false;
let docsLoaded = false;

async function ensureStudioPython() {
  if (studioDefined) return;
  await pyodide.runPythonAsync(STUDIO_PY);
  studioDefined = true;
}

const studioNote = (id, detail) => post("studio_progress", { id, event: { role: "coordinator", step: null, event: "loading", detail } });

// The figures need matplotlib (in the Pyodide distribution), the workbook and
// the Word report need openpyxl and python-docx (from PyPI through micropip).
// Loaded once, before the first run; a failed document install is a note, and
// the bundle says the Word file was skipped.
async function ensurePlotting(id) {
  if (plottingLoaded) return;
  studioNote(id, "Loading the plotting library (once)");
  await pyodide.loadPackage("matplotlib");
  plottingLoaded = true;
}

async function ensureDocs(id) {
  if (docsLoaded) return;
  studioNote(id, "Loading the document libraries (once)");
  try {
    await pyodide.pyimport("micropip").install(["openpyxl", "python-docx"]);
  } catch (err) {
    console.warn("the document libraries did not install; the Word file will be skipped:", err);
    studioNote(id, "The document libraries did not load; the bundle will carry no Word file");
  }
  docsLoaded = true;
}

// One studio call at a time. The arguments travel through a global the Python
// reads at its start, and runPythonAsync yields before it runs (it scans the
// code for imports), so two calls in flight read each other's: seen when a
// follow-up was stopped on the page (abandoned here, still running) and the
// next message arrived behind it with JsNull for its arguments.
let studioChain = Promise.resolve();
function studioSerial(m) {
  const run = studioChain.then(() => studio(m));
  studioChain = run.catch(() => {});
  return run;
}

async function studio(m) {
  const { id, type: _type, ...args } = m;
  await ensureStudioPython();
  if (args.op === "table") {
    await ensureDocs(id);
    self.__aqStudio = JSON.stringify({ name: args.name || "table.xlsx", data: args.data || "" });
    const code = `
import json
from js import __aqStudio
_a = json.loads(__aqStudio)
json.dumps(studio_table(_a["name"], _a["data"]), default=str)
`;
    try {
      const out = await pyodide.runPythonAsync(code);
      post("result", { id, result: JSON.parse(out) });
    } finally {
      self.__aqStudio = null;
    }
    return;
  }
  // A run draws figures and, at the end, the documents; narrate rewrites the
  // report and remakes the documents with the new prose.
  if (args.op === "approve" || args.op === "follow_up" || args.op === "narrate") {
    await ensurePlotting(id);
    await ensureDocs(id);
  }
  self.__aqStudio = JSON.stringify(args);
  self.__aqStudioEvent = (text) => post("studio_progress", { id, event: JSON.parse(text) });
  self.__aqStudioArtifact = (text) => post("studio_artifact", { id, artifact: JSON.parse(text) });
  const code = `
import json
from js import __aqStudio, __aqStudioEvent, __aqStudioArtifact
_a = json.loads(__aqStudio)
_out = studio_call(
    _a,
    on_event=lambda e: __aqStudioEvent(json.dumps(e, default=str)),
    # PNG figures travel with their bytes so the page can show them as they land; SVG and CSV without.
    # The study map (study_map.geojson) travels with its bytes too, so the page draws it as the steps land.
    on_artifact=lambda art: __aqStudioArtifact(
        json.dumps(art.to_dict(with_data=art.media_type == "image/png" or art.id == "study-map"), default=str)),
    store=_STORE,
)
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqStudio = null;
    self.__aqStudioEvent = null;
    self.__aqStudioArtifact = null;
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
_STORE["series"] = _res["series"]
json.dumps({
    "mapping": _res["mapping"],
    "qa": _res["qa"],
    "analysis": _res["analysis"],
    "n": int(len(_res["series"])),
    "csv": analysis.to_csv(_res["analysis"], series=_res["series"]),
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
_series = _STORE.get("series")
if _series is None:
    raise ValueError("The full record is no longer in memory. Reopen the station before sending it to the workbench.")
_variable = _res.get("variable") or "value"
_STORE["frame"] = _series.dropna().rename(_variable).rename_axis("date").reset_index()
from aquascope import workbench as _wb
json.dumps({"n": int(len(_STORE["frame"])), "columns": ["date", _variable],
            "csv": _STORE["frame"].to_csv(index=False),
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

// ── On-device Ask (#590): aquascope.ai_engine.device_ask. The model picks, the package writes (round 3).
// op "plan" is a step's reply schema and message (step_schema: enums built from this question); op "tool" checks
// the model's call (check_call: an id, point or search phrase the words or results hold, no repeat) and runs it,
// returning only the compact result; op "render" writes the answer from the fact keys the model picked
// (render_answer). op "lock" holds free text to the results (lock_answer), kept as the second line of defence.
async function deviceAsk({ id, op, reply, results, words, step, budget, force, notes, why, answer }) {
  self.__aqDev = JSON.stringify({ op: String(op || ""), reply: reply === undefined ? null : reply,
    results: results || [], words: String(words || ""), step: Number(step) || 1,
    budget: Number(budget) || null, force: Boolean(force), notes: Array.isArray(notes) ? notes.map(String) : [],
    why: why ? String(why) : null, answer: String(answer || "") });
  const code = `
import json
from js import __aqDev
from aquascope.ai_engine import analyst as _analyst
from aquascope.ai_engine import device_ask as _dev
_a = json.loads(__aqDev)
if _a["op"] == "plan":
    _out = _dev.step_schema(_a["results"], _a["words"], step=_a.get("step") or 1, budget=_a.get("budget"),
                            force_answer=bool(_a.get("force")), notes=_a.get("notes") or [])
elif _a["op"] == "tool":
    _out = _dev.check_call(_a.get("reply"), _a["results"], _a["words"])
    if _out.get("ok"):
        _spec = {s.name: s for s in _analyst._tool_specs()}.get(_out["name"])
        try:
            _payload = _spec.func(**_out["arguments"]) if _spec else {"error": f"unknown tool {_out['name']}"}
        except Exception as exc:
            _payload = {"error": f"{type(exc).__name__}: {exc}"}
        _payload = json.loads(json.dumps(_payload, default=str))
        _out["ok"] = not (isinstance(_payload, dict) and _payload.get("error"))
        _out["compact"] = _dev.compact_result(_out["name"], _payload)
elif _a["op"] == "render":
    _out = _dev.render_answer(_a.get("reply"), _a["results"], _a["words"], why=_a.get("why"))
elif _a["op"] == "lock":
    _out = _dev.lock_answer(_a.get("answer") or "", _a["results"], _a["words"])
else:
    _out = {"error": "unknown op"}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqDev = null;
  }
}

// ── Study this area: aquascope.area_study over the gauges the page selected ──
// op "run" studies them (the Archive first, live fetches capped) and keeps the
// result; "areas" fills in catchment areas the page read afterwards; "csv" and
// "xlsx" are the downloads (openpyxl is installed on first use).
async function areaStudy({ id, op, stations, question, max_live, areas }) {
  if (op === "xlsx") await ensureDocs(id);
  self.__aqArea = JSON.stringify({ op, stations: stations || [], question: question || null,
    max_live: Number.isFinite(Number(max_live)) ? Number(max_live) : null, areas: areas || {} });
  self.__aqAreaEvent = (text) => post("area_progress", { id, event: JSON.parse(text) });
  const code = `
import json, base64
from js import __aqArea, __aqAreaEvent
from aquascope import area_study as _area_mod
_AREA_STORE = globals().setdefault("_AREA_STORE", {})
_a = json.loads(__aqArea)
if _a["op"] == "run":
    _kw = {"max_live": _a["max_live"]} if _a.get("max_live") is not None else {}
    _AREA_STORE["result"] = _area_mod.study_area(
        _a["stations"], question=_a.get("question"),
        on_progress=lambda e: __aqAreaEvent(json.dumps(e, default=str)), **_kw)
    _out = json.dumps(_AREA_STORE["result"], default=str)
elif _a["op"] == "areas":
    _out = json.dumps(_area_mod.apply_areas(_AREA_STORE["result"], _a["areas"]), default=str)
elif _a["op"] == "csv":
    _out = json.dumps(_area_mod.to_csv(_AREA_STORE["result"]))
elif _a["op"] == "xlsx":
    _out = json.dumps(base64.b64encode(_area_mod.to_xlsx(_AREA_STORE["result"])).decode("ascii"))
else:
    _out = json.dumps({"error": "unknown op"})
_out
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqArea = null;
    self.__aqAreaEvent = null;
  }
}

// ── Paint the globe (#657): aquascope.paint over the gauges in view ─────────
// The page plans the view and downloads the Archive's per-gauge files itself (several at once); this worker
// answers them batch by batch. op "start" keeps the plan (every gauge in view, with why the unread ones are
// unread), "gauges" answers a batch from the files' bytes, "signatures" from signatures.parquet rows, and
// each answer carries the field test over every gauge painted so far. "summary" is the whole painting
// (legend counts, licences, methods paragraph); "csv" and "methods" are the downloads.
async function paintOp({ id, op, question, params, bbox, rows, items, buffers }) {
  self.__aqPaint = JSON.stringify({ op, question: question || null, params: params || {}, bbox: bbox || null,
    rows: rows || [], items: items || [] });
  self.__aqPaintBufs = buffers || [];
  const code = `
import json
from js import __aqPaint, __aqPaintBufs
from aquascope import paint as _paint
_PAINT = globals().setdefault("_PAINT_STORE", {})
_a = json.loads(__aqPaint)
_op = _a["op"]
if _op == "questions":
    _out = _paint.questions()
elif _op == "start":
    _PAINT.clear()
    _PAINT.update(question=_a["question"], params=_a["params"], bbox=_a["bbox"],
                  order=[r["key"] for r in _a["rows"]], rows={r["key"]: r for r in _a["rows"]})
    _out = {"ok": True}
elif _op in ("gauges", "signatures"):
    _items = _a["items"]
    if _op == "gauges":
        for _i, _it in enumerate(_items):
            _it["csv_gz"] = (bytes(__aqPaintBufs[_i].to_py()) or None) if _i < len(__aqPaintBufs) else None
    _new = _paint.gauge_rows(_PAINT["question"], _items, _PAINT["params"])
    for _r in _new:
        _PAINT["rows"][_r["key"]] = _r
    _field = None
    if _paint.QUESTIONS[_PAINT["question"]].kind == "trend":
        _field = _paint.field_significance(list(_PAINT["rows"].values()),
                                           alpha_global=float(_PAINT["params"].get("alpha_global") or 0.05))
    _out = {"rows": _new, "field": _field}
elif _op == "summary":
    _rows = [_PAINT["rows"][k] for k in _PAINT["order"]]
    _PAINT["result"] = _paint.summarize(_PAINT["question"], _rows, params=_PAINT["params"], bbox=_PAINT["bbox"])
    _out = _PAINT["result"]
elif _op == "csv":
    _out = _paint.to_csv(_PAINT["result"])
elif _op == "methods":
    _out = _paint.methods_text(_PAINT["result"])
else:
    _out = {"error": "unknown paint op"}
json.dumps(_out, default=str)
`;
  try {
    post("result", { id, result: JSON.parse(await pyodide.runPythonAsync(code)) });
  } finally {
    self.__aqPaint = null;
    self.__aqPaintBufs = null;
  }
}

// ── Place context (#520): aquascope.context, one layer per message so the card
// fills line by line as each answers. op "point" reads a layer at (lat, lon),
// op "area" over bbox [west, south, east, north]. Every layer reads open data
// hosts that answer CORS (COG range reads, the Archive's context/ mirror).
// op "floods_month" is Floods past (#547): a clicked cell's news events and
// radar months from start to end (aquascope.context.floods_past).
async function placeContext({ id, op, name, lat, lon, bbox, start, end }) {
  self.__aqContext = JSON.stringify({
    op: op || "point", name: String(name || ""), lat: Number(lat), lon: Number(lon),
    bbox: Array.isArray(bbox) ? bbox.map(Number) : null, start: start || null, end: end || null,
  });
  const code = `
import json
from js import __aqContext
from aquascope import context as _ctx
_a = json.loads(__aqContext)
try:
    if _a["op"] == "point":
        _out = _ctx.layer(_a["name"], _a["lat"], _a["lon"])
    elif _a["op"] == "area":
        _out = _ctx.area_layer(_a["name"], *_a["bbox"])
    elif _a["op"] == "floods_month":
        from aquascope.context.floods_past import flood_events_month
        _out = flood_events_month(start=_a["start"], end=_a["end"], bbox=_a["bbox"], limit=12)
    else:
        _out = {"error": "unknown op"}
except ValueError as exc:
    _out = {"error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqContext = null;
  }
}

// ── Watch (#521): aquascope.watch, the same function as `aquascope watch` and the MCP tool. op "digest"
// checks the items it is given (the page sends one at a time, so the panel fills line by line) with the
// Archive's daily status snapshot and newest forecast issue the page read with DuckDB; the records, the
// top-up, a forecast the archive does not cover and the flood events are read here. op "summary" says the
// whole digest in one line.
async function watchDigest({ id, op, items, last_seen, snapshot, issued, today }) {
  self.__aqWatch = JSON.stringify({
    op: op || "digest", items: items || [], last_seen: last_seen || {}, snapshot: snapshot || [],
    issued: issued || [], today: today || null,
  });
  const code = `
import json
from js import __aqWatch
from aquascope import watch as _watch
_a = json.loads(__aqWatch)
try:
    if _a["op"] == "digest":
        _out = _watch.watch_digest(_a["items"], _a["last_seen"], today=_a["today"], snapshot=_a["snapshot"],
                                   issued=_a["issued"], archive=False)
    elif _a["op"] == "summary":
        _out = {"summary": _watch.digest_summary(_a["items"], today=_a["today"])}
    else:
        _out = {"error": "unknown op"}
except ValueError as exc:
    _out = {"error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqWatch = null;
  }
}

// ── Talk to the map (#561): aquascope.map_commands, the same functions as `aquascope map` and the MCP tool.
// op "parse" reads a request with the keyless grammar; "prompt" is the system prompt and schema for a model
// (with "text", the schema for that request: the places and layers its words name, and no free text);
// "reply" checks a model's reply; "validate" checks actions from anywhere (an agent over WebMCP); "resolve"
// looks place names up in the gazetteer (Photon); "llm" asks the reader's own model with their key (the main
// worker only: the provider client lives beside the Analyst).
async function mapCommand({ id, op, text, today, context, reply, actions, provider, model, api_key, base_url }) {
  self.__aqMap = JSON.stringify({
    op: String(op || "parse"), text: String(text || ""), today: today || null, context: String(context || ""),
    reply: reply === undefined ? null : reply, actions: actions || [],
    llm: { provider: provider || null, model: model || null, api_key: api_key || null, base_url: base_url || null },
  });
  const code = `
import json
from js import __aqMap
from aquascope import map_commands as _mc
_a = json.loads(__aqMap)
try:
    if _a["op"] == "parse":
        _out = _mc.parse_command(_a["text"], today=_a["today"])
    elif _a["op"] == "prompt":
        _out = _mc.model_prompt(_a["context"], today=_a["today"])
        if _a["text"]:
            # The device's schema for this request: places and layers its words name, no free text (#590).
            _out["schema"] = _mc.reply_schema(_a["text"])
    elif _a["op"] == "reply":
        _out = _mc.parse_model_reply(_a["reply"], today=_a["today"], text=_a["text"] or None)
    elif _a["op"] == "validate":
        _out = _mc.parse_model_reply({"actions": _a["actions"]}, today=_a["today"])
    elif _a["op"] == "resolve":
        _out = _mc.resolve_actions(_a["actions"])
    elif _a["op"] == "llm":
        _out = _mc.model_command(_a["text"], context=_a["context"], today=_a["today"], **_a["llm"])
    else:
        _out = {"error": "unknown op"}
except (ValueError, RuntimeError) as exc:
    _out = {"error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqMap = null;
  }
}

// ── Scout (#563): aquascope.map_scout, the same functions as `aquascope layers scout` and the MCP tool map_scout.
// op "view" scouts the view on screen (the daily file when it can answer, a live scan otherwise; the page hands
// in the gauge snapshot and the evidence rows it read with DuckDB); "places" names the findings from the
// gazetteer answers the page fetched in parallel (a worker's requests run one at a time); "prompt" is what a
// model is asked; "words" holds a model's reply to the claim lock; "llm" asks the reader's own model with their
// key (main worker only).
async function scoutOp({ id, op, view, month, gauges, skill, extra, places, findings, reply, answers, by, context,
                         max_pins, provider, model, api_key, base_url }) {
  self.__aqScout = JSON.stringify({
    op: String(op || "view"), view: view || null, month: month || null, gauges: gauges || null, skill: skill || null,
    extra: Number(extra) || 0, places: places === undefined ? true : places, findings: findings || [],
    reply: reply === undefined ? null : reply, answers: answers || [],
    by: by || "model", context: String(context || ""), max_pins: Number(max_pins) || 10,
    llm: { provider: provider || null, model: model || null, api_key: api_key || null, base_url: base_url || null },
  });
  const code = `
import json
from js import __aqScout
from aquascope import map_scout as _sc
_a = json.loads(__aqScout)
try:
    if _a["op"] == "view":
        _out = _sc.scout_view(_a["view"], _a["month"], doc=_sc.published(), gauges=_a["gauges"], skill=_a["skill"],
                              extra=_a["extra"], places=_a["places"], max_pins=_a["max_pins"])
    elif _a["op"] == "places":
        _sc.apply_places(_a["findings"], _a["answers"])
        _out = {"findings": _a["findings"]}
    elif _a["op"] == "prompt":
        _out = _sc.wording_prompt(_a["findings"], _a["max_pins"], _a["context"])
    elif _a["op"] == "words":
        _out = _sc.apply_wording(_a["findings"], _a["reply"], max_pins=_a["max_pins"], by=_a["by"])
    elif _a["op"] == "llm":
        _out = _sc.model_wording(_a["findings"], context=_a["context"], max_pins=_a["max_pins"], **_a["llm"])
    else:
        _out = {"error": "unknown op"}
except (ValueError, RuntimeError) as exc:
    _out = {"error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqScout = null;
  }
}

// ── What if, on the map (#565): aquascope.whatif_map, the same functions as `aquascope whatif` and the MCP tool
// whatif_map. Type "whatif" (the main worker: GR4J needs scipy and pandas): op "run" calibrates the basin's model
// and carries the scenario downstream; op "llm" reads the reader's line with their own key. Type "whatif_words"
// (a light worker can): op "parse" is the keyless rules, "prompt" the system prompt and schema for the on-device
// model, "read" both in one call, "reply" reads a model's parse against the reader's words (the code takes each
// number from the words and converts its unit, #590).
// Op "basin" (the main worker: the routing tables need pandas) says whether the reach's basin is within the size
// a run takes, before any calibration.
async function whatIf({ id, type, op, args, text, reply, context, provider, model, api_key, base_url }) {
  self.__aqWhatIf = JSON.stringify({
    op: String(op || (type === "whatif" ? "run" : "parse")), args: args || {}, text: String(text || ""),
    reply: reply === undefined ? null : reply, context: String(context || ""),
    llm: { provider: provider || null, model: model || null, api_key: api_key || null, base_url: base_url || null },
  });
  const code = `
import json
from js import __aqWhatIf
from aquascope import whatif_map as _wi
_a = json.loads(__aqWhatIf)
try:
    if _a["op"] == "run":
        _x = dict(_a["args"])
        _rid = _x.pop("river_id", None)
        _out = _wi.whatif(_rid, **_x)
    elif _a["op"] == "parse":
        _out = _wi.parse_scenario(_a["text"])
    elif _a["op"] == "prompt":
        _out = _wi.scenario_prompt(_a["context"], _a["text"])
    elif _a["op"] == "read":
        # The rules' reading and, when they do not read it all, the device's prompt (its number enum is the
        # line's own number tokens): one call, not two (#590).
        _out = _wi.parse_scenario(_a["text"])
        if not _out["matched"]:
            _out["prompt"] = _wi.scenario_prompt(_a["context"], _a["text"])
    elif _a["op"] == "basin":
        _x = _a["args"]
        _out = _wi._basin_check(_x.get("river_id"), _x.get("lat"), _x.get("lon"))
    elif _a["op"] == "reply":
        _out = _wi.parse_scenario_reply(_a["reply"], _a["text"])
    elif _a["op"] == "llm":
        _out = _wi.model_scenario(_a["text"], context=_a["context"], **_a["llm"])
    else:
        _out = {"error": "unknown op"}
except (ValueError, RuntimeError, LookupError, TypeError) as exc:
    _out = {"error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqWhatIf = null;
  }
}

// ── Review my model results (#564): what lies outside the dropped file, for aquascope.mesh_review (the checks
// themselves run in mesh-worker.js beside the file). op "floods" reads the Sentinel-1 radar and news flood events
// around the mesh from the Archive's context mirror (a light worker can); "gauges" reads each gauge's record over
// the run and "llm" asks the reader's own model with their key (the main worker only).
async function meshReviewOp({ id, op, bbox, gauges, start, end, findings, context, provider, model, api_key, base_url }) {
  self.__aqReview = JSON.stringify({
    op: String(op || ""), bbox: bbox || null, gauges: gauges || [], start: start || null, end: end || null,
    findings: findings || [], context: String(context || ""),
    llm: { provider: provider || null, model: model || null, api_key: api_key || null, base_url: base_url || null },
  });
  const code = `
import json
from js import __aqReview
from aquascope import mesh_review as _mr
_a = json.loads(__aqReview)
try:
    if _a["op"] == "floods":
        _out = _mr.gather_floods(_a["bbox"])
    elif _a["op"] == "gauges":
        _out = {"gauges": _mr.gauge_series(_a["gauges"], _a["start"], _a["end"])}
    elif _a["op"] == "llm":
        _out = _mr.model_words(_a["findings"], context=_a["context"], **_a["llm"])
    else:
        _out = {"error": "unknown op"}
except (ValueError, RuntimeError) as exc:
    _out = {"error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqReview = null;
  }
}

// ── What gets flooded (#622): aquascope.flood_exposure.read_sources reads the buildings and roads (Overture Maps
// GeoParquet, pyarrow loaded here on first use) and the population grid (GHS-POP COG tiles) around the model; the
// counting runs in mesh-worker.js, beside the file.
let pyarrowLoaded = false;
async function floodExposureOp({ id, bbox, what }) {
  if (!pyarrowLoaded) {
    post("progress", { text: "Loading pyarrow to read Overture's GeoParquet…" });
    await pyodide.loadPackage(["pyarrow"]);
    pyarrowLoaded = true;
  }
  self.__aqExposure = JSON.stringify({ bbox: bbox || null, what: what || ["buildings", "roads", "population"] });
  try {
    const out = await pyodide.runPythonAsync(`
import json
from js import __aqExposure
from aquascope import flood_exposure as _fx
_a = json.loads(__aqExposure)
json.dumps(_fx.read_sources(_a["bbox"], what=tuple(_a["what"])))
`);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqExposure = null;
  }
}

// ── Explain what is on screen (#562): aquascope.map_explain, the same functions as `aquascope layers explain`
// and the MCP tool explain_view. op "facts" reads the view into facts and template notes (the month's status
// file is fetched here and handed over as bytes); "check" holds a model's notes to the facts (the claim lock);
// "llm" asks the reader's own model with their key (the main worker only, beside the Analyst's client).
async function mapExplain({ id, op, view, status_url, status_month, warnings, floods_past, gauges, reply, facts,
  context, provider, model, api_key, base_url }) {
  let bytes = null;
  if (op === "facts" && status_url) {
    const res = await fetch(status_url);
    if (res.ok) bytes = new Uint8Array(await res.arrayBuffer());
  }
  self.__aqExplain = JSON.stringify({
    op: String(op || "facts"), view: view || {}, status_month: status_month || null, warnings: warnings || null,
    floods_past: floods_past || null, gauges: gauges || [], reply: reply === undefined ? null : reply,
    facts: facts || [], context: String(context || ""),
    llm: { provider: provider || null, model: model || null, api_key: api_key || null, base_url: base_url || null },
  });
  self.__aqExplainBytes = bytes;
  const code = `
import json
from js import __aqExplain, __aqExplainBytes
from aquascope import map_explain as _me
_a = json.loads(__aqExplain)
_bbox = (_a["view"] or {}).get("bbox")
try:
    if _a["op"] == "facts":
        _st = None
        if _a["status_month"] and __aqExplainBytes is not None:
            _st = {"month": _a["status_month"], "data": bytes(__aqExplainBytes.to_py())}
        _out = _me.explain(_a["view"], status=_st, warnings=_a["warnings"], floods_past=_a["floods_past"],
                           gauges=_a["gauges"], context=_a["context"])
    elif _a["op"] == "check":
        _out = _me.check_model_notes(_a["reply"], _a["facts"], _bbox)
        _me.name_notes(_out["notes"], (_a["view"] or {}).get("zoom") or 5)
    elif _a["op"] == "llm":
        _out = _me.model_notes(_a["facts"], context=_a["context"], bbox=_bbox, **_a["llm"])
        _me.name_notes(_out["notes"], (_a["view"] or {}).get("zoom") or 5)
    else:
        _out = {"error": "unknown op"}
except (ValueError, RuntimeError) as exc:
    _out = {"error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    self.__aqExplain = null;
    self.__aqExplainBytes = null;
  }
}

// ── Groundwater (#558): aquascope.groundwater.surface, the same functions as `aquascope groundwater surface`,
// `aquascope modflow` and the MCP tools. op "surface" interpolates the wells' readings the page read from the
// Archive with DuckDB. op "modflow" reads the head file (and its grid and sidecars) the reader dropped on the
// map: the bytes stay in this worker, and the cells and heads go back to the page as typed arrays.
let pyprojLoaded = false;
async function groundwater(m) {
  const { id, op } = m;
  if (op === "surface") {
    self.__aqGw = JSON.stringify({
      wells: m.wells || [], readings: m.readings || {}, month: m.month || null, mode: m.mode || "level",
      compare: m.compare || null, bbox: Array.isArray(m.bbox) ? m.bbox.map(Number) : null,
    });
    const code = `
import json
from js import __aqGw
from aquascope.groundwater.surface import surface_from_readings as _surface
_a = json.loads(__aqGw)
try:
    _out = _surface(_a["wells"], _a["readings"], month=_a["month"], mode=_a["mode"], compare=_a["compare"],
                    bbox=_a["bbox"])
except ValueError as exc:
    _out = {"kind": "groundwater_surface", "error": str(exc)}
json.dumps(_out)
`;
    try {
      const out = await pyodide.runPythonAsync(code);
      post("result", { id, result: JSON.parse(out) });
    } finally {
      self.__aqGw = null;
    }
    return;
  }
  if (op === "views") { await groundwaterViews(m); return; }
  if (op !== "modflow") throw new Error(`unknown groundwater op ${op}`);
  if (!pyprojLoaded) {
    // Placing a projected grid needs pyproj (3.8 MB, once, only when a model is dropped).
    await pyodide.loadPackage(["pyproj"]);
    pyprojLoaded = true;
  }
  pyodide.FS.mkdirTree("/tmp/gw");
  const write = (name, buf) => { pyodide.FS.writeFile(`/tmp/gw/${name}`, new Uint8Array(buf)); return `/tmp/gw/${name}`; };
  const hds = write("heads", m.hds);
  const grid = m.grid ? write(`grid-${String(m.gridName || "grid").replace(/[^A-Za-z0-9._-]/g, "_")}`, m.grid) : null;
  const sidecars = (m.sidecars || []).map((b, i) => write(`sidecar-${i}`, b));
  const wel = m.wel ? write("wells", m.wel) : null;
  self.__aqGw = JSON.stringify({ hds, grid, sidecars, wel, place: m.place || {} });
  const code = `
import json
import numpy as np
from js import __aqGw
from aquascope.groundwater.surface import read_modflow_heads as _read, modflow_summary as _summary
_a = json.loads(__aqGw)
_p = _a["place"]
_GW_RINGS = _GW_HEADS = None
_GW_RES = _GW_WELLS = None
try:
    _res = _read(_a["hds"], _a["grid"], sidecars=_a["sidecars"], origin=_p.get("origin"), rotation=_p.get("rotation"),
                 crs=_p.get("crs"), cell_size=_p.get("cell_size"), start_date=_p.get("start_date"),
                 approximate_at=_p.get("approximate_at"))
    if not _res.get("error"):
        _GW_RINGS = np.ascontiguousarray(_res["rings"], dtype="<f8").tobytes()
        _GW_OFFS = np.ascontiguousarray(_res["ring_offsets"], dtype="<i4").tobytes()
        _h = np.ascontiguousarray(_res["heads"], dtype="<f4")
        _GW_SHAPE = list(_h.shape)
        _GW_HEADS = _h.tobytes()
    _out = _summary(_res)
    _out.pop("frame", None)
    if not _res.get("error"):
        _GW_RES = _res  # kept for the views (#623, #624): contours, arrows, drawdown, pathlines, capture zones
        if _a.get("wel"):
            from aquascope.groundwater.views import read_wells as _read_wells
            try:
                _GW_WELLS = _read_wells(_a["wel"])
                _out["wells_file"] = {"format": _GW_WELLS["format"], "wells": len(_GW_WELLS["wells"])}
            except ValueError as exc:
                _out.setdefault("notes", []).append(f"The well file could not be read: {exc}")
    _out["times"], _out["dates"] = _res.get("times"), _res.get("dates")
    _out["kstpkper"] = _res.get("kstpkper")
except (ValueError, ImportError) as exc:
    _out = {"kind": "modflow_heads", "error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const meta = JSON.parse(await pyodide.runPythonAsync(code));
    if (meta.error) { post("result", { id, result: meta }); return; }
    const take = (name) => { const p = pyodide.globals.get(name); const u8 = p.toJs(); p.destroy(); return u8; };
    const rings = take("_GW_RINGS"), offs = take("_GW_OFFS"), heads = take("_GW_HEADS");
    const shape = pyodide.globals.get("_GW_SHAPE").toJs();
    pyodide.runPython("_GW_RINGS = _GW_HEADS = _GW_OFFS = _res = None");  // _GW_RES stays, for the views
    const result = {
      ...meta, shape,
      rings: new Float64Array(rings.buffer, rings.byteOffset, rings.byteLength / 8),
      offsets: new Int32Array(offs.buffer, offs.byteOffset, offs.byteLength / 4),
      heads: new Float32Array(heads.buffer, heads.byteOffset, heads.byteLength / 4),
    };
    self.postMessage({ type: "result", id, result }, [rings.buffer, offs.buffer, heads.buffer]);
  } finally {
    self.__aqGw = null;
    for (const f of [hds, grid, wel, ...sidecars]) { if (f) try { pyodide.FS.unlink(f); } catch (err) { /* already gone */ } }
  }
}

// What a hydrogeologist reads off the heads (#623, #624), for one step and layer of the model op "modflow" read:
// aquascope.groundwater.views.model_views, the same function as `aquascope modflow --views`. Geometry comes back in
// lon/lat, small enough for JSON.
async function groundwaterViews(m) {
  const { id } = m;
  self.__aqGv = JSON.stringify({ t: Number(m.t) || 0, layer: m.layer ?? null, reference: m.reference ?? 0,
    show: Array.isArray(m.show) ? m.show : null });
  const code = `
import json
from js import __aqGv
from aquascope.groundwater.views import model_views as _views, model_wells as _model_wells
_a = json.loads(__aqGv)
try:
    if globals().get("_GW_RES") is None:
        raise ValueError("no model is open")
    _key = (_a["layer"], id(_GW_RES), id(globals().get("_GW_WELLS")))
    if globals().get("_GW_WELLS_KEY") != _key:
        _GW_WELLS_FOR = _model_wells(_GW_RES, _a["layer"], globals().get("_GW_WELLS"))
        _GW_WELLS_KEY = _key
    _v = _views(_GW_RES, _a["t"], _a["layer"], reference=_a["reference"], wells=_GW_WELLS_FOR, show=_a["show"])
except ValueError as exc:
    _v = {"kind": "modflow_views", "error": str(exc)}
json.dumps(_v)
`;
  try {
    post("result", { id, result: JSON.parse(await pyodide.runPythonAsync(code)) });
  } finally {
    self.__aqGv = null;
  }
}

// ── Groundwater views (#625, #626): aquascope.groundwater.layers and .budget, the same functions as
// `aquascope modflow --layers --section --cell --budget`. op "open" reads the model again beside the map's copy and
// keeps it in this worker (with its layers and, when a .cbc came, its water budget); "section" cuts it along a line
// drawn on the map, "hydrograph" gives one cell's heads in every layer, "close" lets it go. Arrays come back as
// typed arrays.
async function gwLayers(m) {
  const { id, op } = m;
  const f32 = (name) => {
    const p = pyodide.globals.get(name);
    if (!p) return null;
    const u8 = p.toJs();
    p.destroy();
    return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4);
  };
  if (op === "close") {
    await pyodide.runPythonAsync("_GWV = None");
    post("result", { id, result: { ok: true } });
    return;
  }
  if (op === "open") {
    if (!pyprojLoaded) { await pyodide.loadPackage(["pyproj"]); pyprojLoaded = true; }
    pyodide.FS.mkdirTree("/tmp/gwv");
    const write = (name, buf) => { pyodide.FS.writeFile(`/tmp/gwv/${name}`, new Uint8Array(buf)); return `/tmp/gwv/${name}`; };
    const hds = write("heads", m.hds);
    const grid = m.grid ? write(`grid-${String(m.gridName || "grid").replace(/[^A-Za-z0-9._-]/g, "_")}`, m.grid) : null;
    const sidecars = (m.sidecars || []).map((b, i) => write(`sidecar-${i}`, b));
    const cbc = m.cbc ? write("budget", m.cbc) : null;
    self.__aqGwv = JSON.stringify({ hds, grid, sidecars, cbc, place: m.place || {} });
    const code = `
import json
import numpy as np
from js import __aqGwv
from aquascope.groundwater.surface import read_modflow_heads as _read
from aquascope.groundwater import layers as _views, budget as _budget
_a = json.loads(__aqGwv)
_p = _a["place"]
_out = {}
_GWV_TOP = _GWV_BOTM = _GWV_TOPS = None
# The map's read (op "modflow" of "groundwater", kept as _GW_RES for the #623/#624 views) is the same model:
# use it rather than holding a second copy of the heads; read again only when it is not there.
_res = globals().get("_GW_RES")
if _res is None:
    _res = _read(_a["hds"], _a["grid"], sidecars=_a["sidecars"], origin=_p.get("origin"),
                 rotation=_p.get("rotation"), crs=_p.get("crs"), cell_size=_p.get("cell_size"),
                 start_date=_p.get("start_date"), approximate_at=_p.get("approximate_at"))
_GWV = {"res": _res, "layers": None, "budget": None}
if _res.get("error"):
    _out["error"] = _res["error"]
else:
    if _a["grid"]:
        try:
            _l = _views.model_layers(_res, _a["grid"])
            _GWV["layers"] = _l
            _out["layers"] = _views.layers_summary(_l)
            _GWV_TOP = np.ascontiguousarray(_l["top"], dtype="<f4").tobytes()
            _GWV_BOTM = np.ascontiguousarray(_l["botm"], dtype="<f4").tobytes()
            _GWV_TOPS = np.ascontiguousarray(_l["tops"], dtype="<f4").tobytes()  # each layer's own top (LAYCBD)
        except ValueError as exc:
            _out["layers_error"] = str(exc)
    else:
        _out["layers_error"] = "Without the grid file the layers' tops and bottoms are not known."
    if _a["cbc"]:
        try:
            _GWV["budget"] = _budget.water_budget(_a["cbc"], heads=_res)
            _out["budget"] = _GWV["budget"]
        except ValueError as exc:
            _out["budget_error"] = str(exc)
json.dumps(_out, default=str)
`;
    try {
      const meta = JSON.parse(await pyodide.runPythonAsync(code));
      const top = f32("_GWV_TOP"), botm = f32("_GWV_BOTM"), tops = f32("_GWV_TOPS");
      pyodide.runPython("_GWV_TOP = _GWV_BOTM = _GWV_TOPS = _res = None");
      const result = { ...meta, top, botm, tops };
      self.postMessage({ type: "result", id, result }, [top, botm, tops].filter(Boolean).map((a) => a.buffer));
    } finally {
      self.__aqGwv = null;
      for (const f of [hds, grid, cbc, ...sidecars]) { if (f) try { pyodide.FS.unlink(f); } catch (err) { /* gone */ } }
    }
    return;
  }
  if (op === "section") {
    self.__aqGwv = JSON.stringify({ line: m.line || [] });
    const code = `
import json
import numpy as np
from js import __aqGwv
from aquascope.groundwater import layers as _views
_GWV_A = None
_GWV = globals().get("_GWV")
if _GWV is None or _GWV.get("layers") is None:
    _out = {"error": "The model's layers are not read yet."}
else:
    try:
        _s = _views.cross_section(_GWV["res"], _GWV["layers"], json.loads(__aqGwv)["line"])
    except ValueError as exc:
        _s = {"error": str(exc)}
    _out = {k: v for k, v in _s.items() if k not in ("top", "bottom", "heads", "water_table")}
    if not _s.get("error"):
        _out["shape"] = list(_s["heads"].shape)
        _GWV_A = [np.ascontiguousarray(_s[k], dtype="<f4").tobytes() for k in ("top", "bottom", "heads", "water_table")]
json.dumps(_out, default=str)
`;
    try {
      const meta = JSON.parse(await pyodide.runPythonAsync(code));
      if (meta.error) { post("result", { id, result: meta }); return; }
      const p = pyodide.globals.get("_GWV_A");
      const arrs = p.toJs().map((u8) => new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4));
      p.destroy();
      pyodide.runPython("_GWV_A = None");
      const [top, bottom, heads, waterTable] = arrs;
      self.postMessage({ type: "result", id, result: { ...meta, top, bottom, heads, water_table: waterTable } },
        arrs.map((a) => a.buffer));
    } finally {
      self.__aqGwv = null;
    }
    return;
  }
  if (op === "hydrograph") {
    self.__aqGwv = JSON.stringify({ cell: Number(m.cell) });
    const code = `
import json
from js import __aqGwv
from aquascope.groundwater import layers as _views
_GWV = globals().get("_GWV")
if _GWV is None:
    _out = {"error": "The model is not read yet."}
else:
    try:
        _out = _views.cell_hydrograph(_GWV["res"], json.loads(__aqGwv)["cell"], _GWV.get("layers"))
    except ValueError as exc:
        _out = {"error": str(exc)}
json.dumps(_out, default=str)
`;
    try {
      post("result", { id, result: JSON.parse(await pyodide.runPythonAsync(code)) });
    } finally {
      self.__aqGwv = null;
    }
    return;
  }
  throw new Error(`unknown groundwater views op ${op}`);
}

// ── Route it (#656): aquascope.routing, the same functions as `aquascope route` and the MCP tools route_flood and
// route_storm. Type "route" (the main worker: the routing tables need pandas and the routing scipy): op "flood"
// routes water entering at a reach down its path to the sea, with the gauges (the page's catalog) and the towns
// the page read off the basemap placed on the way; op "places" places more towns on a result's path (the page reads
// them once the path is framed); op "storm" routes a drawn storm's runoff.
// The T-year flows (floodsim.design_flow) read GEOGLOWS's annual maxima store, which needs numcodecs: loaded once,
// the first time a run asks for a return period.
let maximaReady = null;
async function maximaCodec(args) {
  if (!args || args.return_period == null) return;
  if (!maximaReady) maximaReady = pyodide.loadPackage(["numcodecs"]).catch(() => null);
  await maximaReady;
}

async function routeOp({ id, op, args }) {
  await maximaCodec(args);
  const payload = JSON.stringify(JSON.stringify({ op: String(op || "flood"), args: args || {} }));
  const code = `
import json
from aquascope import routing as _rt
_a = json.loads(${payload})
_k = dict(_a["args"])
try:
    if _a["op"] == "flood":
        _rid = _k.pop("river_id", None)
        _out = _rt.route(_rid, **_k)
    elif _a["op"] == "storm":
        _poly = _k.pop("polygon")
        _out = _rt.route_storm(_poly, **_k)
    elif _a["op"] == "places":
        _out = {"places": _rt.places_on_path(_k.get("result") or {}, _k.get("places") or [])}
    else:
        _out = {"ok": False, "error": "unknown op"}
except (ValueError, RuntimeError, LookupError, TypeError) as exc:
    _out = {"ok": False, "error": str(exc)}
json.dumps(_out, default=str)
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}

// ── Draw a reservoir (#662): aquascope.sketch.reservoir, the same functions as `aquascope route reservoir` and the
// MCP tool route_reservoir. Type "reservoir" (the main worker: the routing needs pandas and scipy): op "plan" finds
// the river and the dam under a drawn lake and the terrain window to read; op "run" takes that window's terrain (a
// Float32Array the page read from the Terrarium tiles) and routes a flood through the lake and on down the river.
async function reservoirOp({ id, op, args, z }) {
  await maximaCodec(args);
  pyodide.globals.set("_rv_args", JSON.stringify({ op: String(op || "plan"), args: args || {} }));
  pyodide.globals.set("_rv_z", z || null);
  const code = `
import json
import numpy as _np
from aquascope.sketch import reservoir as _rv
_a = json.loads(_rv_args)
_k = dict(_a["args"])
try:
    if _a["op"] == "plan":
        _out = _rv.plan(_k["polygon"])
    elif _a["op"] == "run":
        _poly = _k.pop("polygon")
        _zz = _np.frombuffer(_rv_z.to_py(), dtype="<f4").astype("f8")
        _out = _rv.route_reservoir(_poly, z=_zz, **_k)
    else:
        _out = {"ok": False, "error": "unknown op"}
except (ValueError, RuntimeError, LookupError, TypeError) as exc:
    _out = {"ok": False, "error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    pyodide.globals.set("_rv_z", null);
  }
}

// ── Flood it (#658): aquascope.floodsim, the same functions as `aquascope flood` and the MCP tool flood_reach.
// Type "floodsim" (the main worker: the network needs pandas, the valley scipy, the flows' store numcodecs):
// op "plan" finds the window, the reaches and their flows (kept here for the next ops); op "prepare" takes the
// window's terrain (a Float32Array the page read from the Terrarium tiles) and returns the grid the GPU or the
// fallback worker floods: the burnt bed and Manning's n as Float32Arrays, the inflows, the edges and the words;
// op "forecast" and op "routed" return a hydrograph at the clicked reach. Levees and breaches (#663): "prepare" takes
// the drawn lines as edits (aquascope.floodsim.edit_grid) and returns the edited terrain; op "profile" gives the
// ground and the flood level along a line and the crest the code suggests (levee_profile, z: the run's largest
// depth); op "exposure" counts what gets flooded before and after (compare_exposure, z: the two depths over the
// terrain, one after the other; read: what the light worker read).
let floodsimReady = null;
async function floodsimOp({ id, op, args, z }) {
  if (!floodsimReady) floodsimReady = pyodide.loadPackage(["numcodecs"]).catch(() => null);
  await floodsimReady;
  // Before and after (#663) reads the two runs back as a mesh, which places it with pyproj.
  if (op === "exposure" && !pyprojLoaded) { await pyodide.loadPackage(["pyproj"]); pyprojLoaded = true; }
  pyodide.globals.set("_fs_args", JSON.stringify({ op: String(op || "plan"), args: args || {} }));
  pyodide.globals.set("_fs_z", z || null);
  const code = `
import json
import numpy as _np
from aquascope import floodsim as _fs
_a = json.loads(_fs_args)
_k = dict(_a["args"])
_arrays = None
try:
    _FS = globals().setdefault("_FS_PLANS", {})
    if _a["op"] == "plan":
        _p = _fs.plan_window(_k.get("river_id"), _k.get("lat"), _k.get("lon"))
        _FS.clear()
        _FS[str(_p["river_id"])] = _p
        _out = {"ok": True, "river_id": _p["river_id"], "window": _p["window"], "stem": _p["stem"],
                "reaches": {r: {"order": v["order"], "curve": v["curve"]} for r, v in _p["reaches"].items()}}
    elif _a["op"] == "prepare":
        _p = _FS[str(_k["river_id"])]
        _zz = _np.frombuffer(_fs_z.to_py(), dtype="<f4").astype("f8")
        _g = _fs.prepare_grid(_p, _zz, _k.get("choice") or {"return_period": 100}, coarse=bool(_k.get("coarse")))
        _FS["grid"] = _g
        _edits = None
        if _k.get("edits"):
            _g = _fs.edit_grid(_g, _k["edits"], skip_outside=True)
            _edits = [{k: v for k, v in e.items() if k != "cells"} | {"cells": len(e["cells"])} for e in _g["edits"]]
        _arrays = {"z": _g["z_burnt"].astype("<f4").ravel(), "n": _g["n"].astype("<f4").ravel(),
                   "ground": _g["z"].astype("<f4").ravel()}
        _out = {"ok": True, "window": _g["window"], "nx": _g["nx"], "ny": _g["ny"], "dx": _g["dx"],
                "sources": [[s["cell"][0], s["cell"][1], s["flow"]] for s in _g["sources"]],
                "inflows": [{"river_id": s["river_id"], "into": s["into"], "enters": s["enters"], "order": s["order"],
                             "peak_cms": s["peak"], "q2_cms": s["curve"]["q"][0]} for s in _g["sources"]],
                "edges": {k: [None if _np.isnan(x) else float(x) for x in v] for k, v in _g["edges"].items()},
                "exits": _g["exits"], "slope": _g["slope"], "table_dt_s": _g["table_dt_s"], "hours": _g["hours"],
                "click": {k: v for k, v in _g["click"].items() if k != "curve"}, "label": _g["label"],
                "coarse": _g["coarse"], "channels": len(_g["channels"]), "reaches": len(_p["reaches"]),
                "flow_method": _fs.FLOW_METHOD, "licences": _fs.LICENCES, "edits": _edits}
    elif _a["op"] == "profile":
        _g = _fs.edit_grid(_FS["grid"], _k.get("edits") or [], skip_outside=True)
        _h = _np.frombuffer(_fs_z.to_py(), dtype="<f4").astype("f8") if _fs_z is not None else None
        _out = {"ok": True, **_fs.levee_profile(_g, _k["line"], _h,
                                                freeboard_m=float(_k.get("freeboard_m") or _fs.FREEBOARD_M))}
    elif _a["op"] == "exposure":
        _g = _fs.edit_grid(_FS["grid"], _k.get("edits") or [], skip_outside=True)
        _two = _np.frombuffer(_fs_z.to_py(), dtype="<f4").astype("f8").reshape(2, -1)
        _out = {"ok": True, **_fs.compare_exposure(_g, _two[0], _two[1], _k["read"])}
    elif _a["op"] == "forecast":
        _out = {"ok": True, **_fs.forecast_series(int(_k["river_id"]))}
    elif _a["op"] == "routed":
        _p = _FS[str(_k["river_id"])]
        _out = {"ok": True, **_fs.routed_series(_p, float(_k.get("return_period") or 100),
                                                duration_h=float(_k.get("duration_h") or 24))}
    else:
        _out = {"ok": False, "error": "unknown op"}
except (ValueError, RuntimeError, LookupError, TypeError, KeyError, ImportError) as exc:
    _out = {"ok": False, "error": str(exc)}
_fs_out = json.dumps(_out, default=str)
`;
  await pyodide.runPythonAsync(code);
  const result = JSON.parse(pyodide.globals.get("_fs_out"));
  const transfer = [];
  const arrays = pyodide.globals.get("_arrays");
  if (arrays) {
    for (const key of ["z", "n", "ground"]) {
      const arr = arrays.get(key);
      const buf = arr.getBuffer();
      try { result[key] = new Float32Array(buf.data); transfer.push(result[key].buffer); } finally { buf.release(); arr.destroy(); }
    }
    arrays.destroy();
  }
  pyodide.globals.set("_fs_z", null);
  self.postMessage({ type: "result", id, result }, transfer);
}

// ── Replay an event (#659): aquascope.replay, the same functions as `aquascope replay` and the MCP tools
// replay_candidates and replay_event. Type "replay" (the main worker: the gauges need pandas, the forecast archive
// numcodecs): op "candidates" lists Floods past's floods near a point in a month; op "record" assembles the event
// without the flood (the page runs Flood it on the GPU) and leaves the rain as a ClimateSERV request; op "rain"
// looks at that request once (the page asks again every few seconds); op "story" writes the story and the report,
// with the page's own flood and exposure numbers when it has them.
let replayReady = null;
async function replayOp({ id, op, args }) {
  if (!replayReady) replayReady = pyodide.loadPackage(["numcodecs"]).catch(() => null);
  await replayReady;
  pyodide.globals.set("_rp_args", JSON.stringify({ op: String(op || "record"), args: args || {} }));
  pyodide.globals.set("_rp_say", (text) => post("replay_progress", { id, text: String(text) }));
  const code = `
import json
from aquascope import replay as _rp
_a = json.loads(_rp_args)
_k = dict(_a["args"])
try:
    if _a["op"] == "candidates":
        _out = _rp.candidates(float(_k["lat"]), float(_k["lon"]), month=_k.get("month"), start=_k.get("start"),
                              end=_k.get("end"), radius_km=float(_k.get("radius_km") or 40))
    elif _a["op"] == "record":
        _out = _rp.event_record(_k.get("river_id"), lat=_k.get("lat"), lon=_k.get("lon"), start=_k.get("start"),
                                end=_k.get("end"), event=_k.get("event"), places=_k.get("places"),
                                with_flood=False, rain_wait=False, progress=_rp_say)
    elif _a["op"] == "rain":
        _out = _rp.rain_poll(_k["piece"])
    elif _a["op"] == "story":
        _st = _rp.story(_k["record"], flood_result=_k.get("flood"))
        _out = {"story": _st, "report": _rp.report(_k["record"], _st, flood_result=_k.get("flood"))}
    else:
        _out = {"ok": False, "error": "unknown op"}
except (ValueError, RuntimeError, LookupError, TypeError, KeyError, OSError) as exc:
    _out = {"ok": False, "error": str(exc)}
json.dumps(_out, default=str)
`;
  try {
    const out = await pyodide.runPythonAsync(code);
    post("result", { id, result: JSON.parse(out) });
  } finally {
    pyodide.globals.set("_rp_say", null);
  }
}

// ── Drop a well (#660): aquascope.sketch.drawdown, the same functions as `aquascope well` and the MCP tool
// drop_well. Type "well" (the main worker: the well function is scipy's exp1): op "map" draws the cone of the wells
// on the map (contours, the fill's grid, each well's drawdown at its face, and with frames the cone at a series of
// times for the time bar); op "point" gives the drawdown at a point with each well's share.
async function wellOp({ id, op, args }) {
  pyodide.globals.set("_wl_args", JSON.stringify({ op: String(op || "map"), args: args || {} }));
  const code = `
import json
from aquascope.sketch import drawdown as _wl
_a = json.loads(_wl_args)
_k = dict(_a["args"])
try:
    if _a["op"] == "map":
        _out = _wl.drawdown_map(_k.pop("wells"), **_k)
    elif _a["op"] == "point":
        _out = _wl.drawdown_at(_k.pop("lon"), _k.pop("lat"), _k.pop("wells"), **_k)
    elif _a["op"] == "check":
        _out = _wl.check()
    else:
        _out = {"ok": False, "error": "unknown op"}
except (ValueError, TypeError, KeyError) as exc:
    _out = {"ok": False, "error": str(exc)}
json.dumps(_out)
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}

// ── Change land use (#661): aquascope.sketch.landuse, the same function as `aquascope landuse` and the MCP tool
// change_landuse. Type "landuse" (the main worker: the soil group's COG and the network need numpy, the outlet the
// stream tiles): op "change" reads the cover, the soil and the slope under the polygon and computes TR-55 before
// and after, with the reach it drains to; op "check" recomputes TR-55's worked examples.
async function landuseOp({ id, op, args }) {
  pyodide.globals.set("_lu_args", JSON.stringify({ op: String(op || "change"), args: args || {} }));
  const code = `
import json
from aquascope.sketch import landuse as _lu
_a = json.loads(_lu_args)
_k = dict(_a["args"])
try:
    if _a["op"] == "change":
        _out = _lu.landuse_change(_k.pop("polygon"), _k.pop("use"), **_k)
    elif _a["op"] == "check":
        _out = _lu.check()
    else:
        _out = {"ok": False, "error": "unknown op"}
except (ValueError, TypeError, KeyError, RuntimeError, OSError) as exc:
    _out = {"ok": False, "error": str(exc)}
json.dumps(_out, default=str)
`;
  const out = await pyodide.runPythonAsync(code);
  post("result", { id, result: JSON.parse(out) });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") { ready = init(m); await ready; return; }
    await ready;
    if (lite && !LITE_TYPES.has(m.type)) throw new Error(`a light worker does not run ${m.type}`);
    if (m.type === "analyze") return await analyze(m);
    if (m.type === "anywhere") return await anywhere(m);
    if (m.type === "river") return await river(m);
    if (m.type === "evidence") return await evidence(m);
    if (m.type === "now") return await nowNext(m);
    if (m.type === "assess") return await assess(m);
    if (m.type === "compare") return await compare(m);
    if (m.type === "flood_ci") return await floodCi(m);
    if (m.type === "csv") return await csv(m);
    if (m.type === "engineering") return await engineering(m);
    if (m.type === "catalog") return await catalog(m);
    if (m.type === "ask") return await ask(m);
    if (m.type === "solve_plan") return await solvePlan(m);
    if (m.type === "coerce_intake") return await coerceIntake(m);
    if (m.type === "solve_run") return await solveRun(m);
    if (m.type === "studio") return await studioSerial(m);
    if (m.type === "ingest") return await ingestText(m);
    if (m.type === "load_table") return await loadTable(m);
    if (m.type === "workbench") return await workbench(m);
    if (m.type === "tool") return await runTool(m);
    if (m.type === "device_ask") return await deviceAsk(m);
    if (m.type === "frame_from_station") return await frameFromStation(m);
    if (m.type === "area_study") return await areaStudy(m);
    if (m.type === "paint") return await paintOp(m);
    if (m.type === "context") return await placeContext(m);
    if (m.type === "watch") return await watchDigest(m);
    if (m.type === "map_command") return await mapCommand(m);
    if (m.type === "scout") return await scoutOp(m);
    if (m.type === "map_explain") return await mapExplain(m);
    if (m.type === "mesh_review") return await meshReviewOp(m);
    if (m.type === "flood_exposure") return await floodExposureOp(m);
    if (m.type === "whatif" || m.type === "whatif_words") return await whatIf(m);
    if (m.type === "route") return await routeOp(m);
    if (m.type === "reservoir") return await reservoirOp(m);
    if (m.type === "floodsim") return await floodsimOp(m);
    if (m.type === "replay") return await replayOp(m);
    if (m.type === "well") return await wellOp(m);
    if (m.type === "landuse") return await landuseOp(m);
    if (m.type === "groundwater") return await groundwater(m);
    if (m.type === "gw_layers") return await gwLayers(m);
  } catch (err) {
    // Pyodide raises PythonError with the full traceback in .message; keep the
    // exception line (last non-empty) and log the whole thing for debugging.
    const full = String(err && err.message ? err.message : err);
    console.error(full);
    const lines = full.split("\n").filter((l) => l.trim());
    post("error", { id: m.id, message: lines[lines.length - 1] || "unknown error" });
  }
};
