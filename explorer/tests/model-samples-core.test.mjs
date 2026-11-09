// Try a sample model (#591): the samples, the Archive manifest and the button's words.
import test from "node:test";
import assert from "node:assert/strict";
import { ARCHIVE, FLOPY_COMMIT, SAMPLES, hex, manifestFiles, sampleButton, sampleById, sizeWords, totalBytes } from "../src/model-samples-core.js";
import { isModflowDrop, sortModelFiles } from "../src/groundwater-core.js";
import { pickModelFiles } from "../src/mesh-core.js";

test("the MODFLOW sample is FloPy's Freyberg, pinned to one commit, with its name file", () => {
  const s = sampleById("modflow-freyberg");
  assert.equal(s.kind, "modflow");
  assert.match(FLOPY_COMMIT, /^[0-9a-f]{40}$/);
  for (const f of s.files) {
    assert.ok(f.url.startsWith(`https://raw.githubusercontent.com/modflowpy/flopy/${FLOPY_COMMIT}/`), f.url);
    assert.match(f.sha256, /^[0-9a-f]{64}$/);
    assert.ok(f.bytes > 0);
  }
  const sorted = sortModelFiles(s.files);
  assert.equal(sorted.hds.name, "freyberg.hds");
  assert.equal(sorted.grid.name, "freyberg.dis");
  assert.deepEqual(sorted.sidecars.map((f) => f.name), ["freyberg.nam"]);   // xul, yul, rotation, proj4_str
  assert.equal(isModflowDrop(s.files), true);
  assert.match(s.credit, /public domain/);
  assert.match(s.link, new RegExp(FLOPY_COMMIT));
});

test("the HEC-RAS sample is read from the Archive's manifest, and only files under samples/ are taken", () => {
  const s = sampleById("hec-ras-portsmouth");
  assert.equal(s.kind, "mesh");
  assert.equal(s.manifest, `${ARCHIVE}samples/hec-ras/manifest.json`);
  assert.match(s.credit, /Zandsalimi.*GNN4CF.*HydroShare.*CC BY 4\.0/);
  const sha = "11c603af678d7f608ab75ef04c2cf8db97483702c8683c868e76b1405cd46fa5";
  const files = manifestFiles({ files: [
    { path: "samples/hec-ras/Flood_model.p01_BC12_S6_R6.hdf", name: "Flood_model.p01_BC12_S6_R6.hdf", bytes: 29906342, sha256: sha },
    { path: "obs/manifest.json", name: "manifest.json", bytes: 10, sha256: sha },
    { path: "samples/hec-ras/../x.hdf", name: "x.hdf", bytes: 10, sha256: sha },
    { path: "samples/hec-ras/a.hdf", name: "a.hdf", bytes: 10, sha256: "nope" },
  ] });
  assert.deepEqual(files, [{ name: "Flood_model.p01_BC12_S6_R6.hdf",
    url: `${ARCHIVE}samples/hec-ras/Flood_model.p01_BC12_S6_R6.hdf`, bytes: 29906342, sha256: sha }]);
  assert.deepEqual(manifestFiles(null), []);
  assert.deepEqual(manifestFiles({ files: "x" }), []);
  // the mesh viewer takes the plan file, as from a drop
  const got = pickModelFiles([{ name: files[0].name, size: files[0].bytes }]);
  assert.equal(got.file.name, "Flood_model.p01_BC12_S6_R6.hdf");
  assert.equal(isModflowDrop([{ name: files[0].name }]), false);
});

test("the button says what it can do: open, wait, or not published yet", () => {
  const frey = sampleById("modflow-freyberg");
  const ready = sampleButton(frey, "ready");
  assert.equal(ready.disabled, false);
  assert.match(ready.note, /11\.1 MB/);
  const hec = sampleById("hec-ras-portsmouth");
  assert.match(sampleButton(hec, "missing").note, /Not published yet/);
  assert.equal(sampleButton(hec, "missing").disabled, true);
  assert.equal(sampleButton(hec, "checking").disabled, true);
  assert.match(sampleButton(hec, "offline").note, /Could not reach/);
  assert.match(sampleButton(frey, "loading", { done: 5e6 }).note, /5\.0 MB of 11\.1 MB/);
});

test("sizes and hashes", () => {
  assert.equal(sizeWords(573), "573 bytes");
  assert.equal(sizeWords(97882), "98 kB");
  assert.equal(sizeWords(29906342), "29.9 MB");
  assert.equal(totalBytes(SAMPLES[0].files), 573 + 97882 + 10676004 + 327112);
  assert.equal(hex(new Uint8Array([0, 15, 255]).buffer), "000fff");
});
