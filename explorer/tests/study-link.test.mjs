// node --test explorer/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
  MAX_FETCH_BYTES, fetchStudyYaml, isLinkToken, linkUrl, sharedBoardHtml, sharedPlanLine, studyUrlParam,
} from "../src/study-link.js";

const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const stepHtml = (s) => `<li>${esc(s.tool)}</li>`;

test("a token is a version prefix, a dot and base64url; a recorded id is not one", () => {
  assert.equal(isLinkToken("z1.abc-_09"), true);
  assert.equal(isLinkToken("j1.eyJ2IjoxfQ"), true);
  assert.equal(isLinkToken("kingston-flood"), false);
  assert.equal(isLinkToken("z1.a b"), false);
  assert.equal(isLinkToken("z2.abc"), false);
  assert.equal(isLinkToken(`z1.${"a".repeat(9000)}`), false);
});

test("study_url must be https", () => {
  assert.equal(studyUrlParam("?study_url=https%3A%2F%2Fexample.org%2Fstudy.yaml"), "https://example.org/study.yaml");
  assert.equal(studyUrlParam(""), null);
  assert.deepEqual(studyUrlParam("?study_url=http://example.org/s.yaml"), { error: "the study_url must be an https address" });
  assert.deepEqual(studyUrlParam("?study_url=javascript:alert(1)"), { error: "the study_url must be an https address" });
  assert.deepEqual(studyUrlParam("?study_url=not%20a%20url"), { error: "the study_url is not a URL" });
});

function fakeFetch({ status = 200, body = "version: 3\n", length = null } = {}) {
  const seen = [];
  const fn = async (url, opts) => {
    seen.push({ url, opts });
    return {
      ok: status >= 200 && status < 300, status,
      headers: { get: (k) => (k === "content-length" && length !== null ? String(length) : null) },
      text: async () => body,
    };
  };
  fn.seen = seen;
  return fn;
}

test("the study file is fetched without cookies and with a size cap", async () => {
  const f = fakeFetch();
  assert.equal(await fetchStudyYaml("https://example.org/s.yaml", { fetch: f }), "version: 3\n");
  assert.equal(f.seen[0].opts.credentials, "omit");
  await assert.rejects(fetchStudyYaml("http://example.org/s.yaml", { fetch: f }), /https/);
  await assert.rejects(fetchStudyYaml("https://example.org/s.yaml", { fetch: fakeFetch({ status: 404 }) }), /HTTP 404/);
  await assert.rejects(fetchStudyYaml("https://example.org/s.yaml", { fetch: fakeFetch({ length: MAX_FETCH_BYTES + 1 }) }), /too large/);
  await assert.rejects(fetchStudyYaml("https://example.org/s.yaml", { fetch: fakeFetch({ body: "x".repeat(MAX_FETCH_BYTES + 1) }) }), /too large/);
  const boom = async () => { throw new TypeError("Failed to fetch"); };
  await assert.rejects(fetchStudyYaml("https://example.org/s.yaml", { fetch: boom }), /may not allow it/);
});

test("the link keeps the page address and puts only the study in the hash", () => {
  assert.equal(linkUrl("https://huggingface.co/spaces/a/b#s=uk_ea/1&study=1", "z1.x"),
    "https://huggingface.co/spaces/a/b#study=z1.x");
});

test("whose plan ran", () => {
  assert.equal(sharedPlanLine({ used: "proposed" }), "the shared plan, re-run keyless");
  assert.match(sharedPlanLine({ used: "tree", errors: ["step s2: unknown tool"] }), /did not pass the validator here: step s2/);
  assert.equal(sharedPlanLine({}), "the playbook's plan");
});

test("the shared board: loading, refused with every reason, or the plan with Run, all escaped", () => {
  assert.match(sharedBoardHtml({ loading: true }, { escapeHtml: esc, stepHtml }), /Checking the shared study/);
  const refused = sharedBoardHtml({ ok: false, errors: ["step s1: unknown tool '<script>'"] }, { escapeHtml: esc, stepHtml });
  assert.match(refused, /cannot be opened/);
  assert.ok(refused.includes("&lt;script&gt;") && !refused.includes("<script>"));
  assert.ok(!refused.includes('data-act="run-shared"'), "nothing to run when the plan is refused");
  const ok = sharedBoardHtml({
    ok: true, notes: ["a note"], from: "https://example.org/s.yaml",
    study: { text: "<b>q</b>", lat: 51.415, lon: -0.308, name: "Kingston", plan: { objective: "obj", steps: [{ tool: "describe_catchment" }] } },
  }, { escapeHtml: esc, stepHtml });
  assert.match(ok, /shared study/);
  assert.match(ok, /data-act="run-shared"/);
  assert.match(ok, /Kingston \(51\.415, -0\.308\)/);
  assert.ok(ok.includes("&lt;b&gt;q&lt;/b&gt;") && ok.includes("<li>describe_catchment</li>"));
  assert.match(ok, /from https:\/\/example\.org\/s\.yaml/);
});
