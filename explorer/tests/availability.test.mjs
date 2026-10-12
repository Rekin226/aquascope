import test from "node:test";
import assert from "node:assert/strict";
import { loadAvailability, catalogOnly, availabilityLabel, observationMetadata } from "../src/availability.js";

test("catalog coverage never becomes an observation or freshness claim", async () => {
  assert.equal(catalogOnly("unknown"), false, "unknown capabilities are not a negative retrieval claim");
  const calls = [];
  globalThis.fetch = async (url) => {
    calls.push(String(url));
    return { ok: true, json: async () => String(url).includes("source-capabilities")
      ? { record_sources: ["usgs"] }
      : { sources: { "usgs/discharge": { stations: { A: {
        n: 20, first: "2020-01-01", last: "2020-01-20", harvested_at: "2026-09-01",
        last_attempt_status: "failed",
      } } } } } };
  };
  await loadAvailability();
  assert.equal(catalogOnly("usgs"), false);
  assert.equal(catalogOnly("unsupported"), true);
  assert.match(availabilityLabel("unsupported"), /Catalog only/);
  assert.match(availabilityLabel("usgs"), /checked on opening/);
  const text = await observationMetadata("usgs", "A", ["discharge"]);
  assert.match(text, /2020-01-01/);
  assert.match(text, /last successful observation update 2026-09-01/);
  assert.match(text, /latest refresh failed, previous data retained/);
  assert.match(await observationMetadata("usgs", "absent", ["discharge"]), /No mirrored observations/);
  assert.equal(calls.filter(x => x.includes("manifest")).length, 1);
});
