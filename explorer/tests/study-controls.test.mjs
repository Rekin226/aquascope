// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import { readChanges, rememberPlain, steerHtml, stepChecksHtml } from "../src/study-controls.js";

const GATES = [{ check: "min_years", value: 20, path: "years" }, { check: "max_return_period_factor", value: 3, path: "years", return_period: 100 }];
const STEP = { id: "s3", tool: "flood_frequency", arguments: { source: "uk_ea" }, expects: GATES };
const PLAIN = {
  steps: [{
    id: "s3", tool: "flood_frequency", gates: GATES,
    checks: ["needs at least 20 years of record", "the 100-year estimate may not exceed 3x the record length"],
    controls: [
      { param: "return_period", label: "Design return period (years)", type: "choice", choices: [50, 100, 200], value: 100 },
      { param: "years", label: "Use only the last N years", type: "integer", min: 5, max: 200, optional: true, value: null },
      { param: "bootstrap_ci", label: "Bootstrap <band>", type: "boolean", value: true },
      { param: "locked", label: "Locked", type: "number", value: 1, locked: true },
    ],
  }],
  steering: [{ step: "s3" }],
};

test("the plan reads in plain words, with the raw gates behind details", () => {
  rememberPlain(PLAIN);
  const html = stepChecksHtml(STEP, "<span class=\"gate\">min_years 20</span>");
  assert.match(html, /<ul class="step-checks"><li>needs at least 20 years of record<\/li>/);
  assert.match(html, /<details class="step-raw"><summary>details<\/summary><div class="step-gates"><span class="gate">min_years 20/);
  assert.doesNotMatch(html, /[—–]/);
});

test("a step the worker did not describe keeps its raw gate chips", () => {
  rememberPlain(PLAIN);
  const other = { ...STEP, expects: [{ check: "min_years", value: 30 }] };
  assert.equal(stepChecksHtml(other, "chips"), '<div class="step-gates">chips</div>');
  rememberPlain(null);
  assert.equal(stepChecksHtml(STEP, "chips"), '<div class="step-gates">chips</div>');
  assert.equal(stepChecksHtml(STEP, ""), "");
});

test("the controls show one form per steerable step, current values selected, locked ones left out", () => {
  rememberPlain(PLAIN);
  const html = steerHtml({ steps: [STEP, { id: "s1", tool: "describe_catchment", expects: [] }] }, { label: (t) => `L:${t}` });
  assert.equal((html.match(/data-steer-step=/g) || []).length, 1);
  assert.match(html, /<legend>s3 · L:flood_frequency<\/legend>/);
  assert.match(html, /<option value="100" selected>100<\/option>/);
  assert.match(html, /type="checkbox"[^>]*checked/);
  assert.match(html, /placeholder="full record"/);
  assert.match(html, /Bootstrap &lt;band&gt;/, "labels are escaped");
  assert.doesNotMatch(html, /data-param="locked"/);
  assert.match(html, /Adjust a step \(1 change so far\)/);
  assert.match(steerHtml({ steps: [STEP] }, { busy: true }), /data-steer-run="s3" disabled/);
  rememberPlain(null);
  assert.equal(steerHtml({ steps: [STEP] }), "");
});

// A minimal stand-in for the form the board draws.
function field(param, type, current, props) {
  return { dataset: { param, type, current: JSON.stringify(current) }, value: "", checked: false, ...props };
}

test("only the fields that changed are read, each in its control's type", () => {
  const form = {
    querySelectorAll: () => [
      field("return_period", "choice", 100, { value: "200" }),
      field("distribution", "choice", "gev", { value: "gev" }),
      field("years", "integer", null, { value: "30" }),
      field("bootstrap_ci", "boolean", true, { checked: true }),
      field("threshold", "number", -1, { value: "-1.5" }),
    ],
  };
  assert.deepEqual(readChanges(form), { return_period: 200, years: 30, threshold: -1.5 });
  const cleared = { querySelectorAll: () => [field("years", "integer", 30, { value: "" })] };
  assert.deepEqual(readChanges(cleared), { years: null }, "an emptied optional field resets to the default");
});
