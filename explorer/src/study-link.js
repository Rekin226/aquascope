// Study links: the page half of aquascope.study_link. A finished study's plan
// (never its results) travels as #study=z1.<token>; a study.yaml travels as
// ?study_url=<https URL>. The worker encodes, decodes and checks the plan
// against the method catalogue (studio ops "link" and "open_link"); this
// module only reads the URL, fetches the file with a size cap, builds the
// link, and renders the shared plan and the line that says whose plan ran.
// Pure: no DOM lookups and no imports, so node can test it; the escaper and
// the step renderer are handed in by studio.js.

export const STUDY_URL_PARAM = "study_url";
export const MAX_FETCH_BYTES = 200000;   // aquascope.study_link.MAX_STUDY_BYTES
export const MAX_LINK_CHARS = 8000;      // aquascope.study_link.MAX_LINK_CHARS

// A study token in the hash: a version prefix, a dot, base64url. Recorded study ids never hold a dot.
export function isLinkToken(v) {
  return typeof v === "string" && v.length <= MAX_LINK_CHARS && /^[zj]1\.[A-Za-z0-9_-]+$/.test(v);
}

// The https URL of a study.yaml from the query string, or null. Anything but https is refused.
export function studyUrlParam(search) {
  const raw = new URLSearchParams(String(search || "").replace(/^\?/, "")).get(STUDY_URL_PARAM);
  if (!raw) return null;
  let url;
  try { url = new URL(raw); } catch { return { error: "the study_url is not a URL" }; }
  if (url.protocol !== "https:") return { error: "the study_url must be an https address" };
  return url.href;
}

// The study.yaml at an https URL, as text: no cookies sent, at most MAX_FETCH_BYTES read.
export async function fetchStudyYaml(href, { fetch: fetchFn = globalThis.fetch } = {}) {
  let url;
  try { url = new URL(href); } catch { throw new Error("the study_url is not a URL"); }
  if (url.protocol !== "https:") throw new Error("the study_url must be an https address");
  let res;
  try {
    res = await fetchFn(url.href, { credentials: "omit", redirect: "follow", cache: "no-store" });
  } catch (err) {
    throw new Error(`could not fetch the study file (the host may not allow it): ${err && err.message}`);
  }
  if (!res.ok) throw new Error(`could not fetch the study file: HTTP ${res.status}`);
  const size = Number(res.headers && res.headers.get ? res.headers.get("content-length") : NaN);
  if (Number.isFinite(size) && size > MAX_FETCH_BYTES) throw new Error("the study file is too large");
  const text = await res.text();
  if (text.length > MAX_FETCH_BYTES) throw new Error("the study file is too large");
  return text;
}

// The link for a token: the page's shareable address with only the study in the hash.
export function linkUrl(canonical, token) {
  const base = String(canonical || "").split("#")[0];
  return `${base}#study=${token}`;
}

// Who planned, on the foot of a shared study's report.
export function sharedPlanLine({ used = null, errors = [] } = {}) {
  if (used === "proposed" || used === "device") return "the shared plan, re-run keyless";
  if (used === "tree" && errors && errors.length) {
    return `the playbook's plan (the shared plan did not pass the validator here: ${errors[0]})`;
  }
  return "the playbook's plan";
}

// The board for a shared study: loading, refused (every reason, in words), or the plan with Run.
// `shared` is the worker's open_link reply ({ ok, errors, notes, study }) or { loading: true }.
export function sharedBoardHtml(shared, { escapeHtml, stepHtml }) {
  const esc = escapeHtml;
  const label = `<p class="study-shared-label"><span class="study-grade grade-indicative">shared study</span></p>`;
  if (!shared || shared.loading) {
    return `<article class="study-shared" tabindex="-1" aria-label="Shared study">${label}` +
      `<p class="study-line muted">Checking the shared study…</p></article>`;
  }
  if (!shared.ok) {
    const errs = (shared.errors || []).slice(0, 12);
    return `<article class="study-shared" tabindex="-1" aria-label="Shared study">${label}` +
      `<p class="study-declined">This shared study cannot be opened.</p>` +
      (errs.length ? `<ul class="study-shared-errors">${errs.map((e) => `<li>${esc(String(e))}</li>`).join("")}</ul>` : "") +
      `<div class="row-actions"><button type="button" class="btn primary" data-act="again">Start a new study</button></div>` +
      `</article>`;
  }
  const s = shared.study || {};
  const plan = s.plan || {};
  const where = s.name ? `${s.name} (${Number(s.lat).toFixed(3)}, ${Number(s.lon).toFixed(3)})`
    : `${Number(s.lat).toFixed(3)}, ${Number(s.lon).toFixed(3)}`;
  const notes = shared.notes || [];
  const from = shared.from ? `<p class="study-line muted">from ${esc(String(shared.from))}</p>` : "";
  return `<article class="study-shared study-plan" tabindex="-1" aria-label="Shared study">${label}` +
    `<p class="study-where">${esc(where)}</p>` +
    `<p class="study-objective">${esc(s.text || "")}</p>` +
    (plan.objective && plan.objective !== s.text ? `<p class="study-line muted">${esc(plan.objective)}</p>` : "") +
    `<ol class="study-steps">${(plan.steps || []).map(stepHtml).join("")}</ol>` +
    notes.map((n) => `<p class="study-line muted">${esc(String(n))}</p>`).join("") +
    from +
    `<p class="study-line muted">Someone shared this plan. Run reruns every step here, in your browser, with no key; ` +
    `the plan is checked again at the site first.</p>` +
    `<div class="row-actions">` +
      `<button type="button" class="btn primary" data-act="run-shared">Run</button>` +
      `<button type="button" class="btn" data-act="again">Not now</button>` +
    `</div></article>`;
}
