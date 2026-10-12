import test from "node:test";
import assert from "node:assert/strict";
import { EXPORT_PROMPT, base64ToBytes, exportOptions, exportSummary } from "../src/export-menu.js";

test("the menu is a prompt plus one option per tool the worker offers", () => {
  const opts = exportOptions({ kind: "flow", tools: [
    { id: "hec-hms", label: "HEC-HMS", what: "Gage record" }, { id: "raven", label: "Raven", what: "rvt" },
  ] });
  assert.deepEqual(opts.map((o) => o.value), ["", "hec-hms", "raven"]);
  assert.equal(opts[0].label, EXPORT_PROMPT);
  assert.ok(opts[0].prompt && !opts[1].prompt);
  assert.equal(opts[1].title, "Gage record");
});

test("no tools means no menu at all", () => {
  assert.deepEqual(exportOptions({ kind: null, tools: [] }), []);
  assert.deepEqual(exportOptions(null), []);
  assert.deepEqual(exportOptions({ tools: [{ label: "no id" }] }).length, 1);
});

test("the zip travels as base64 and comes back byte for byte", () => {
  const bytes = base64ToBytes(Buffer.from([0x50, 0x4b, 0x03, 0x04, 0xff]).toString("base64"));
  assert.deepEqual(Array.from(bytes), [0x50, 0x4b, 0x03, 0x04, 0xff]);
});

test("the status line names the file and the note that matters", () => {
  const line = exportSummary({ label: "MODFLOW 6", filename: "aquascope-X-modflow6.zip", files: [1, 2, 3],
    notes: ["5 of 10 time steps have no value.", "cond and rbot are placeholders: set them."] });
  assert.match(line, /^MODFLOW 6: aquascope-X-modflow6\.zip \(3 files\)\./);
  assert.match(line, /placeholders/);
  assert.equal(exportSummary(null), "");
});
