import test from "node:test";
import assert from "node:assert/strict";
import { percentiles } from "../../.github/scripts/explorer_smoke.mjs";

test("small samples are not presented as performance percentiles", () => {
  assert.equal(percentiles([1, 2, 3, 4]).p50, undefined);
  assert.deepEqual(percentiles([50, 10, 30, 20, 40]), { n: 5, p50: 30, p75: 40 });
});
