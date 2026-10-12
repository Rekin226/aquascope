// Job health (#593): the legend's stale-data notes and Tools > Data status, from health/jobs/index.json.
import { test } from "node:test";
import assert from "node:assert/strict";

import { ageWords, jobState, legendNotes, statusHeadline, statusItems } from "../src/job-health-core.js";

const NOW = Date.parse("2026-10-12T12:00:00Z");
const ago = (h) => new Date(NOW - h * 3600e3).toISOString();

function entry(job, { okHours = 10, last = "success", limit = 48, layers = [], cadence = "daily" } = {}) {
  return {
    job, label: job, cadence, stale_after_hours: limit, layers, what: "something.",
    last_success: okHours === null ? null : { finished_at: ago(okHours), data_date: "2026-10-12" },
    last_run: last ? { status: last, finished_at: ago(1) } : null,
    last_failure: last === "failure" ? { finished_at: ago(1) } : null,
  };
}

test("a job is on time, late, failing, on demand or unknown, as health_jobs.assess reads it", () => {
  assert.deepEqual(jobState(entry("a"), NOW), { state: "ok", ageHours: 10 });
  assert.equal(jobState(entry("a", { okHours: 50 }), NOW).state, "stale");
  assert.equal(jobState(entry("a", { last: "failure" }), NOW).state, "failing");
  assert.equal(jobState(entry("a", { okHours: 60, last: "failure" }), NOW).state, "stale");
  assert.equal(jobState(entry("a", { okHours: null, last: "failure" }), NOW).state, "stale");
  assert.equal(jobState(entry("a", { okHours: 3000, limit: null }), NOW).state, "on demand");
  assert.equal(jobState(entry("a", { okHours: null, last: null }), NOW).state, "unknown");
  assert.equal(jobState(null, NOW).state, "unknown");
});

test("ages read in minutes, hours, then days", () => {
  assert.equal(ageWords(0.2), "12 min");
  assert.equal(ageWords(9.4), "9 h");
  assert.equal(ageWords(47), "47 h");
  assert.equal(ageWords(72), "3 d");
  assert.equal(ageWords(null), "");
});

test("only a layer whose job is late gets a legend note, with the expected cadence", () => {
  const index = { jobs: [
    entry("forecast-archive", { okHours: 80, layers: ["forecast-points"] }),
    entry("flood-warnings", { okHours: 5, layers: ["floods-ahead"] }),
    entry("water-balance", { okHours: 900, limit: 868, cadence: "monthly (the 9th)", layers: ["water-balance"] }),
    entry("mirror-context", { okHours: 5000, limit: null, layers: ["floods-past"] }),
    entry("bulletin", { okHours: 2000, limit: 868 }),
  ] };
  const notes = legendNotes(index, NOW);
  assert.deepEqual(Object.keys(notes), ["forecast-points", "water-balance"]);
  assert.deepEqual(notes["forecast-points"], { text: "3 d old",
    title: "Older than expected: the forecast-archive last updated it 3 d ago; it runs daily." });
  assert.equal(notes["water-balance"].text, "38 d old");
  assert.match(notes["water-balance"].title, /runs monthly \(the 9th\)/);
  assert.equal(legendNotes({ jobs: [entry("x", { okHours: null, last: "failure", layers: ["gauges"] })] }, NOW).gauges.text,
    "not updated");
  assert.deepEqual(legendNotes(null, NOW), {});
  assert.deepEqual(legendNotes({ jobs: "nope" }, NOW), {});
});

test("Data status lists every job and heads with how many are late or failing", () => {
  const index = { jobs: [entry("harvest"), entry("scout", { last: "failure" }), entry("bulletin", { okHours: 90 }),
    entry("mirror-context", { limit: null }), entry("repair", { okHours: null, last: null, limit: null })] };
  const items = statusItems(index, NOW);
  assert.deepEqual(items.map((i) => [i.job, i.state, i.text]), [
    ["harvest", "ok", "on time, 10 h ago"],
    ["scout", "failing", "last run failed, 10 h ago"],
    ["bulletin", "stale", "late, 4 d ago"],
    ["mirror-context", "on demand", "when dispatched, 10 h ago"],
    ["repair", "unknown", "no record yet"],
  ]);
  assert.match(items[1].title, /Last failure 2026-10-12/);
  assert.equal(statusHeadline(items), "Data status: 2 of 3 late or failing");
  assert.equal(statusHeadline(items.slice(0, 1)), "Data status: all 1 on time");
  assert.equal(statusHeadline([]), "Data status: no record yet");
});
