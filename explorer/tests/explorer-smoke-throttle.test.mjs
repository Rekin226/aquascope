// #726: a cold load the live host throttled (429) is retried once rather than failing the daily check.
import assert from "node:assert/strict";
import test from "node:test";
import { isThrottled } from "../../.github/scripts/explorer_smoke.mjs";

test("a failed load that saw a 429 counts as throttled", () => {
  assert.equal(isThrottled({ status: "failed", network_errors: [{ url: "x.js", status: 429 }, { url: "x.js", error: "net::ERR_ABORTED" }] }), true);
});

test("other failures and passes are not throttled", () => {
  assert.equal(isThrottled({ status: "failed", network_errors: [{ url: "x.js", status: 404 }] }), false);
  assert.equal(isThrottled({ status: "failed", network_errors: [] }), false);
  assert.equal(isThrottled({ status: "failed" }), false);
  assert.equal(isThrottled({ status: "passed", network_errors: [{ url: "x.js", status: 429 }] }), false);
});
