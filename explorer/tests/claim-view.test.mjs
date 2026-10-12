import test from "node:test";
import assert from "node:assert/strict";
import { claimEvidenceHtml } from "../src/claim-view.js";

test("visible evidence keeps input identity and refuses another estimator's interval", () => {
  const evidence = { result_id: "s1.lmom.q.5", estimator: "gev_lmoments", dataset: {
    source: "<script>", station_id: "A", snapshot: "sha256:abc", start: "1903-07-29", end: "2026-09-21",
    unit: "m3/s", observations: 37534,
  }, interval: { result_id: "s1.mle.q.5", bounds: [403.8, 767.1], method: "bootstrap", level: .9 } };
  const html = claimEvidenceHtml(evidence);
  assert.match(html, /sha256:abc/);
  assert.match(html, /1903-07-29/);
  assert.match(html, /No interval recorded for this estimator/);
  assert.doesNotMatch(html, /403\.8|<script>/);
  assert.match(html, /&lt;script&gt;/);
  evidence.interval.result_id = evidence.result_id;
  assert.match(claimEvidenceHtml(evidence), /403\.8 to 767\.1 m3\/s/);
});
