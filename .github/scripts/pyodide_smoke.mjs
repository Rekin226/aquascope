#!/usr/bin/env node
// Pyodide smoke test: installs the aquascope wheel inside Pyodide (Node.js)
// and runs analyze_series on a synthetic 30-year record.  Fails the CI job if
// any core import breaks under Emscripten or a dependency is missing.

import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { resolve, basename } from "node:path";

const wheel = process.argv[2];
if (!wheel) {
  console.error("Usage: node pyodide_smoke.mjs <path/to/aquascope-*.whl>");
  process.exit(1);
}
const wheelPath = resolve(wheel);

console.log(`Loading Pyodide…`);
const pyodide = await loadPyodide({
  packageCacheDir: "/tmp/pyodide-cache",
  packages: ["micropip", "numpy", "scipy", "pandas", "pydantic", "httpx"],
});

const wheelName = basename(wheelPath);
console.log(`Installing ${wheelName} + pyodide-http…`);
// Write the wheel into Emscripten's MEMFS so micropip reads it via emfs:
// (emfs:/path — single slash, MEMFS-absolute; not an authority-style URL).
// httpx is not in Pyodide's built-in package set, so install it via micropip.
// The filename in MEMFS must retain standard PEP 427 wheel tags for micropip.
const wheelData = readFileSync(wheelPath);
pyodide.FS.writeFile(`/tmp/${wheelName}`, new Uint8Array(wheelData));
await pyodide.runPythonAsync(`
import micropip
await micropip.install(["pyodide-http", "emfs:/tmp/${wheelName}"])
`);

console.log("Applying pyodide_http.patch_all()…");
await pyodide.runPythonAsync(`
import pyodide_http
pyodide_http.patch_all()
`);

console.log("Verifying aquascope.explore import under Pyodide…");
await pyodide.runPythonAsync(`
import aquascope.explore as _explore
assert hasattr(_explore, "analyze_series"), "analyze_series missing"
`);

console.log("Running analyze_series on a synthetic 30-year series…");
const result = await pyodide.runPythonAsync(`
import json, numpy as np, pandas as pd
from aquascope.explore import analyze_series

idx = pd.date_range("1990-01-01", periods=int(365.25 * 30), freq="D")
rng = np.random.default_rng(7)
base = 50 + 30 * np.sin(np.arange(len(idx)) / 58.1)
series = pd.Series(np.exp(rng.normal(0, 0.5, len(idx))) * base, index=idx)

out = analyze_series(series, "discharge", "m3/s")

# Verify core outputs exist
for key in ("variable", "unit", "n", "ffa", "fdc", "trend", "annual_max"):
    assert key in out, f"Missing key: {key}"
assert out["variable"] == "discharge"
assert out["n"] > 10000
assert out["ffa"]["n_years"] >= 28
assert out["fdc"]["q95"] < out["fdc"]["q50"] < out["fdc"]["q10"]

json.dumps({"status": "ok", "n": out["n"], "years": out["years"]})
`);
console.log("Pyodide analyze_series passed:", result);

console.log("Verifying CachedHTTPClient Emscripten path…");
await pyodide.runPythonAsync(`
import sys
assert sys.platform == "emscripten", f"Expected emscripten, got {sys.platform}"
from aquascope.utils.http_client import CachedHTTPClient, IS_EMSCRIPTEN
assert IS_EMSCRIPTEN, "IS_EMSCRIPTEN should be True under Pyodide"
client = CachedHTTPClient(base_url="https://example.com")
assert hasattr(client, "_client")
client.close()
`);

console.log("Verifying the place-context package and the pure-Python COG reader (#520)…");
await pyodide.runPythonAsync(`
import struct, zlib
from aquascope import context
from aquascope.utils.cog import COG
assert "flood_history" in context.LAYERS
data = zlib.compress(bytes([1, 2, 3, 4]))
n = 11
extra = 8 + 2 + n * 12 + 4
scale_at, tie_at, data_at = extra, extra + 24, extra + 72
short = lambda tag, v: struct.pack("<HHIHxx", tag, 3, 1, v)
long_ = lambda tag, v: struct.pack("<HHII", tag, 4, 1, v)
ifd = struct.pack("<H", n) + b"".join([
    short(256, 2), short(257, 2), short(258, 8), short(259, 8), long_(273, data_at), short(277, 1),
    short(278, 2), long_(279, len(data)), struct.pack("<HHII", 33550, 12, 3, scale_at),
    struct.pack("<HHII", 33922, 12, 6, tie_at), short(339, 1)]) + struct.pack("<I", 0)
tif = (b"II" + struct.pack("<HI", 42, 8) + ifd + struct.pack("<3d", 0.5, 0.5, 0.0)
       + struct.pack("<6d", 0, 0, 0, 10.0, 50.0, 0.0) + data)
c = COG("mem://t.tif", fetch=lambda url, s, e: tif[s:e + 1])
assert c.value_at(10.75, 49.25) == 4 and c.value_at(10.25, 49.9) == 1, c.transform
`);

console.log("All Pyodide smoke tests passed.");
