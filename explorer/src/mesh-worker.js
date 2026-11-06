// Your model results on the globe (#555): the worker that reads a dropped file. A module worker with its own
// Python (Pyodide), so a large file never waits behind a station's record in the main worker.
//
// The file is not copied: it is mounted with WORKERFS, which hands Python a read-only file whose bytes come
// from the browser's File as they are read. aquascope.io.mesh (the same code as `aquascope mesh` and the MCP
// tool) reads the geometry and the maxima once, then one time step or one cell when asked. Nothing leaves
// the browser.
//
// Messages in: { type: "init", pyodideIndexURL, wheelsJson }, { type: "open", id, file, crs, geometry } (geometry
// the HEC-RAS .g01.hdf dropped with a stripped plan file, or null),
// { type: "frame", id, step }, { type: "series", id, cell }, { type: "close", id }, and for the review (#564,
// aquascope.mesh_review): { type: "inside", id, rows } (the gauges standing in a cell), { type: "review", id,
// context }, { type: "prompt", id, findings, context } and { type: "words", id, findings, reply, by } (a model's
// reply held to the claim lock), and for the flood views (#619, #620, aquascope.mesh_views): { type: "views", id }
// (the lenses and the smooth grid, built once per file) and { type: "field", id, what, step } (one smooth picture
// and its contour lines).
// Messages out: { type: "progress", text }, { type: "ready" }, { id, result } or { id, error }.

let pyodide = null;
let booting = null;
let mounted = 0;
let scipyLoaded = false;
let viewsLoaded = false;

const post = (msg, transfer = []) => self.postMessage(msg, transfer);

async function boot({ pyodideIndexURL, wheelsJson }) {
  post({ type: "progress", text: "Loading Python in your browser…" });
  const { loadPyodide } = await import(`${pyodideIndexURL}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: pyodideIndexURL });
  post({ type: "progress", text: "Loading the HDF5 and projection libraries…" });
  await pyodide.loadPackage(["micropip", "numpy", "h5py", "pyproj"]);
  post({ type: "progress", text: "Installing aquascope…" });
  const wheels = await (await fetch(wheelsJson, { cache: "no-store" })).json();
  const wheelUrl = new URL(wheels.wheel, wheelsJson).href;
  const resp = await fetch(wheelUrl, { cache: "reload" });
  if (!resp.ok) throw new Error(`could not fetch the aquascope wheel (${resp.status})`);
  pyodide.FS.writeFile(`/tmp/${wheels.wheel}`, new Uint8Array(await resp.arrayBuffer()));
  const micropip = pyodide.pyimport("micropip");
  // Without its dependencies: the mesh reader needs numpy, h5py and pyproj, already here.
  await micropip.install.callKwargs(`emfs:/tmp/${wheels.wheel}`, { deps: false });
  await pyodide.runPythonAsync(`
import json
from aquascope.io import mesh as _mesh
from aquascope import mesh_review as _review
_M = None
_L = _G = None
`);
  post({ type: "ready" });
}

// A dict of numpy arrays (a PyProxy) -> { name: TypedArray }, copied out of Python's memory.
function takeArrays(proxy) {
  const out = {};
  for (const key of proxy.keys()) {
    const arr = proxy.get(key);
    const buf = arr.getBuffer();
    try { out[key] = buf.data.slice(); } finally { buf.release(); arr.destroy(); }
  }
  proxy.destroy();
  return out;
}

const transferList = (arrays) => Object.values(arrays).map((a) => a.buffer);

function pyError(err) {
  const text = String((err && err.message) || err);
  const m = /MeshError: ([^\n]+)/.exec(text);
  if (m) return { message: m[1], mesh: true };
  const last = text.trim().split("\n").filter(Boolean).pop() || text;
  return { message: last.replace(/^\w+Error: /, ""), mesh: false };
}

async function openFile({ id, file, crs, geometry }) {
  // NetCDF-3 classic files are read with scipy, loaded only when one comes.
  const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  if (head[0] === 0x43 && head[1] === 0x44 && head[2] === 0x46 && !scipyLoaded) {
    post({ type: "progress", text: "Loading scipy for a NetCDF-3 file…" });
    await pyodide.loadPackage(["scipy"]);
    scipyLoaded = true;
  }
  // The new file is read beside the one on the map: only once it has read does it replace it, so a file
  // that fails (no coordinate system, not a mesh) leaves the model on the map working.
  mounted += 1;
  const dir = `/drop${mounted}`;
  pyodide.FS.mkdirTree(dir);
  pyodide.FS.mount(pyodide.FS.filesystems.WORKERFS, { files: geometry ? [file, geometry] : [file] }, dir);
  pyodide.globals.set("_path", `${dir}/${file.name}`);
  pyodide.globals.set("_geom", geometry ? `${dir}/${geometry.name}` : "");
  pyodide.globals.set("_crs", crs ? String(crs) : "");   // "" for none: a JS null arrives as jsnull, not None
  const t0 = performance.now();
  let res;
  try {
    res = pyodide.runPython(`
_N = _mesh.read_mesh(_path, crs=_crs, geometry=_geom or None)
_meta, _arr = _mesh.browser_payload(_N)
(json.dumps(_meta), _arr)
`);
  } catch (err) {
    pyodide.runPython("_N = None");
    try { pyodide.FS.unmount(dir); } catch { /* not mounted */ }
    throw err;
  }
  pyodide.runPython("_M.close() if _M is not None else None\n_M, _N = _N, None\n_L = _G = None");
  for (let k = 1; k < mounted; k++) {
    try { pyodide.FS.unmount(`/drop${k}`); } catch { /* not mounted */ }
  }
  const meta = JSON.parse(res.get(0));
  const arrays = takeArrays(res.get(1));
  res.destroy();
  meta.read_ms = Math.round(performance.now() - t0);
  meta.heap_mb = Math.round(pyodide._module.HEAPU8.length / 1048576);
  post({ id, result: { meta, arrays } }, transferList(arrays));
}

function frame({ id, step }) {
  pyodide.globals.set("_step", Number(step));
  const arrays = takeArrays(pyodide.runPython("_mesh.frame_payload(_M, _step)"));
  post({ id, result: { step: Number(step), arrays } }, transferList(arrays));
}

function series({ id, cell }) {
  pyodide.globals.set("_cell", Number(cell));
  const out = pyodide.runPython("json.dumps(_mesh.series_payload(_M, _cell))");
  post({ id, result: JSON.parse(out) });
}

// The section, the hydrograph and what gets flooded (#621, #622: aquascope.flood_section and
// aquascope.flood_exposure): { type: "section", id, line }, { type: "hydrograph", id, cell, threshold } and
// { type: "exposure", id, read, threshold } (read: the buildings, roads and population a light worker read;
// threshold: the depth that counts as flooded, in metres). Arguments in as one JSON string, the answer back as one.
const FLOOD_CALLS = new Set(["section", "hydrograph", "exposure"]);
function floodCall({ id, type, line, cell, threshold, read }) {
  pyodide.globals.set("_fq", JSON.stringify({ type, line: line || null, cell: cell === undefined ? null : cell,
    threshold: threshold === undefined ? null : threshold, read: read || null }));
  const out = pyodide.runPython(`
from aquascope import flood_section as _fs, flood_exposure as _fx
_q = json.loads(_fq)
if _M is None:
    raise _mesh.MeshError("No model result is open.")
if _q["type"] == "section":
    _o = _fs.section(_M, _q["line"])
elif _q["type"] == "hydrograph":
    _o = _fs.hydrograph(_M, _q["cell"], threshold=_q["threshold"])
else:
    _o = _fx.exposure_payload(_M, _q["read"], threshold_m=_q["threshold"])
json.dumps(_o)
`);
  post({ id, result: JSON.parse(out) });
}

// The review's calls: their arguments travel as one JSON string and come back as one.
function reviewCall({ id, type, rows, context, findings, reply, by, question }) {
  pyodide.globals.set("_rq", JSON.stringify({ type, rows: rows || [], context: context || null, findings: findings || [],
    reply: reply === undefined ? null : reply, by: by || "model", text: typeof context === "string" ? context : "",
    question: typeof question === "string" ? question : "" }));
  const out = pyodide.runPython(`
_q = json.loads(_rq)
if _M is None:
    raise _mesh.MeshError("No model result is open.")
if _q["type"] == "inside":
    _o = _review.gauges_inside(_M, _q["rows"])
elif _q["type"] == "review":
    _o = _review.review(_M, context=_q["context"])
elif _q["type"] == "prompt":
    _o = _review.review_prompt(_q["findings"], _q["text"], _q["question"])
else:
    _o = _review.apply_words(_q["findings"], _q["reply"], by=_q["by"], question=_q["question"])
json.dumps(_o, default=str)
`);
  post({ id, result: JSON.parse(out) });
}

// The flood views: the lenses of the whole run and the grid the smooth depth is drawn on, once per file (scipy
// for the triangulation and contourpy for the lines are loaded the first time).
async function views({ id }) {
  if (!viewsLoaded) {
    await pyodide.loadPackage(["scipy", "contourpy"]);
    scipyLoaded = true;
    viewsLoaded = true;
  }
  const res = pyodide.runPython(`
from aquascope import mesh_views as _views
if _M is None:
    raise _mesh.MeshError("No model result is open.")
if _L is None:
    _L = _views.mesh_lenses(_M)
    _G = _views.field_grid(_M)
_vmeta, _varr = _views.views_payload(_M, _L, _G)
(json.dumps(_vmeta), _varr)
`);
  const meta = JSON.parse(res.get(0));
  const arrays = takeArrays(res.get(1));
  res.destroy();
  post({ id, result: { meta, arrays } }, transferList(arrays));
}

function field({ id, what, step }) {
  pyodide.globals.set("_what", String(what || "depth"));
  pyodide.globals.set("_fstep", Number.isInteger(step) ? step : -1);
  const res = pyodide.runPython(`
if _G is None:
    raise _mesh.MeshError("The views are not ready.")
_f = _views.field_payload(_M, _G, _L, _what, None if _fstep < 0 else _fstep)
(_f["lines"], {"image": _f["image"]})
`);
  const lines = res.get(0);
  const arrays = takeArrays(res.get(1));
  res.destroy();
  post({ id, result: { what, step, lines: JSON.parse(lines), image: arrays.image } }, [arrays.image.buffer]);
}

self.onmessage = async (e) => {
  const m = e.data || {};
  try {
    if (m.type === "init") {
      booting = booting || boot(m);
      await booting;
      return;
    }
    await booting;
    if (FLOOD_CALLS.has(m.type)) { floodCall(m); return; }
    if (m.type === "open") await openFile(m);
    else if (m.type === "frame") frame(m);
    else if (m.type === "series") series(m);
    else if (m.type === "views") await views(m);
    else if (m.type === "field") field(m);
    else if (["inside", "review", "prompt", "words"].includes(m.type)) reviewCall(m);
    else if (m.type === "close") {
      pyodide.runPython("_M.close() if _M is not None else None\n_M = None");
      post({ id: m.id, result: true });
    }
  } catch (err) {
    const { message, mesh } = pyError(err);
    if (m.type === "init") post({ type: "failed", error: message });
    else post({ id: m.id, error: message, mesh });
  }
};
