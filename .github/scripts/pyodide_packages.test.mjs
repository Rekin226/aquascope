import assert from "node:assert/strict";
import test from "node:test";
import { loadPackagesWithRetry } from "./pyodide_packages.mjs";

function fixture(load) {
  const calls = [], delays = [], warnings = [];
  const pyodide = {
    loadedPackages: {},
    async loadPackage(names) {
      calls.push(names);
      await load(this.loadedPackages, names, calls.length);
    },
  };
  return {
    pyodide, calls, delays, warnings,
    options: { sleep: async (ms) => delays.push(ms), warn: (msg) => warnings.push(msg) },
  };
}

test("cold load succeeds without retry", async () => {
  const f = fixture((loaded, names) => names.forEach((name) => { loaded[name] = "default"; }));
  await loadPackagesWithRetry(f.pyodide, ["scipy"], f.options);
  assert.deepEqual(f.calls, [["scipy"]]);
  assert.deepEqual(f.delays, []);
});

test("already loaded packages need no request", async () => {
  const f = fixture(() => assert.fail("unexpected load"));
  f.pyodide.loadedPackages.scipy = "default";
  await loadPackagesWithRetry(f.pyodide, ["scipy"], f.options);
  assert.deepEqual(f.calls, []);
});

test("rejected downloads recover on a later attempt", async () => {
  const f = fixture((loaded, names, attempt) => {
    if (attempt < 3) throw new Error("fetch failed");
    loaded.scipy = "default";
  });
  await loadPackagesWithRetry(f.pyodide, ["scipy"], f.options);
  assert.equal(f.calls.length, 3);
  assert.deepEqual(f.delays, [1000, 2000]);
  assert.equal(f.warnings.length, 2);
});

test("resolved partial downloads retry only missing packages", async () => {
  const f = fixture((loaded, names, attempt) => {
    loaded.micropip = "default";
    if (attempt === 2) loaded.scipy = "default";
  });
  await loadPackagesWithRetry(f.pyodide, ["micropip", "scipy"], f.options);
  assert.deepEqual(f.calls, [["micropip", "scipy"], ["scipy"]]);
  assert.deepEqual(f.delays, [1000]);
});

test("persistent rejection fails after three attempts with original cause", async () => {
  const cause = new Error("fetch failed");
  const f = fixture(() => { throw cause; });
  await assert.rejects(loadPackagesWithRetry(f.pyodide, ["scipy"], f.options), (error) => {
    assert.match(error.message, /failed after 3 attempts/);
    assert.equal(error.cause, cause);
    return true;
  });
  assert.equal(f.calls.length, 3);
  assert.deepEqual(f.delays, [1000, 2000]);
});

test("persistent non-throwing failures report missing packages", async () => {
  const f = fixture(() => {});
  await assert.rejects(loadPackagesWithRetry(f.pyodide, ["scipy"], f.options), (error) => {
    assert.match(error.cause.message, /Packages not loaded: scipy/);
    return true;
  });
  assert.equal(f.calls.length, 3);
});
