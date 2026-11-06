import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { recordToExport } from "../../.github/scripts/explorer_smoke.mjs";

test("failed catalog load retains bounded diagnostics and removes listeners", async () => {
  class Page extends EventEmitter {
    async goto() {
      this.emit("console", { type: () => "error", text: () => "catalog failed" });
      this.emit("requestfailed", { url: () => "https://example.test/catalog?token=secret", failure: () => ({ errorText: "net::ERR_FAILED" }) });
      this.emit("response", { url: () => "https://example.test/data?key=secret", status: () => 503 });
    }
    async waitForFunction() { throw new Error("catalog timed out"); }
    locator(selector) { return { textContent: async () => selector === "#boot-error" ? "catalog unavailable" : "" }; }
    async evaluate() { return ["duckdb: import"]; }
  }
  const page = new Page();
  const result = await recordToExport(page, "https://example.test", { id: "a", source: "x", station: "y" });
  assert.equal(result.status, "failed");
  assert.equal(result.error, "catalog timed out");
  assert.equal(result.boot_error, "catalog unavailable");
  assert.deepEqual(result.trace, ["duckdb: import"]);
  assert.deepEqual(result.console_errors, ["catalog failed"]);
  assert.deepEqual(result.network_errors, [
    { url: "https://example.test/catalog", error: "net::ERR_FAILED" },
    { url: "https://example.test/data", status: 503 },
  ]);
  for (const event of ["pageerror", "console", "requestfailed", "response"]) assert.equal(page.listenerCount(event), 0);
});
