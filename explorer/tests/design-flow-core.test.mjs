// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { designFlowLine, designFlowWords, flowFor } from "../src/design-flow-core.js";

// The same curve as tests/test_floodsim.py::test_design_flow_twin_fixture: floodsim.flow_for and flow_words.
const FX = JSON.parse(readFileSync(new URL("../../tests/fixtures/design_flow_twin.json", import.meta.url), "utf8"));

test("a flow off the curve is the package's, held at 2 and 500 years", () => {
  FX.T.forEach((t, i) => assert.ok(Math.abs(flowFor(FX.curve, t) - FX.q[i]) < 1e-5, `T=${t}`));
  assert.equal(flowFor(null, 100), null);
});

test("every card words the method and the source as the package does", () => {
  assert.equal(designFlowWords(FX.curve), FX.words);
  assert.equal(designFlowWords({}), "");
  const d = { ...FX.curve, return_period: 100 };
  assert.equal(designFlowLine(d), `Flow: the 100-year flow, ${FX.words}.`);
  assert.equal(designFlowLine(null), "");
});
