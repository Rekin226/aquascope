// The recorded studies on the Study board: the chips the intake offers, the
// line that says a study is a recording, the links to its files, its figures
// as the board shows them, and the words for a plan that was re-run. Pure
// (no DOM), over studio-showcase.js's readers; node-importable.

import { chipText, recordedLabel } from "./studio-showcase.js?v=__BUILD__";

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export const CHIP_LIMIT = 6;

// The files a recording carries that are worth a link, in this order.
const LINKED = [["report.md", "Markdown"], ["report.html", "HTML report"], ["study.yaml", "study.yaml"],
  ["workspace.json", "workspace.json"], ["complete.aqstudy.json", "Complete study"],
  ["observations.csv", "Input observations"], ["provenance.json", "Provenance"]];

/**
 * One line of label and the chips: at most `limit` visible, a "more" chip
 * for the rest until `showAll`. Empty when there is nothing recorded.
 */
export function recordedChipsHtml(rows, { limit = CHIP_LIMIT, showAll = false } = {}) {
  const list = (Array.isArray(rows) ? rows : []).filter((r) => r && r.id && chipText(r));
  if (!list.length) return "";
  const shown = showAll ? list : list.slice(0, limit);
  const chips = shown.map((r) =>
    `<button type="button" class="chip" data-recorded="${esc(r.id)}" title="${esc(r.shows || (r.site && r.site.name) || "")}">${esc(chipText(r))}</button>`);
  if (!showAll && list.length > limit) {
    chips.push(`<button type="button" class="chip" data-act="more-recorded">${list.length - limit} more</button>`);
  }
  return `<div class="study-recorded"><p class="study-line muted">See a recorded study</p>` +
    `<div class="study-chips">${chips.join("")}</div></div>`;
}

/** The line above a recorded answer: who recorded it, and that the numbers are the recording's. */
export function recordedNoteHtml(meta) {
  return (meta?.review_status ? `<p class="study-by muted">${esc(meta.review_status)}</p>` : "") +
    `<p class="study-by muted">${esc(recordedLabel(meta))}; the numbers below were computed then; ` +
    `press Re-run live to compute them again in your browser, keyless.</p>`;
}

/** Links to the recorded files ({name: url}), and three words for the two that are not recorded. */
export function recordedFilesHtml(files) {
  const f = files || {};
  const links = LINKED.filter(([name]) => f[name])
    .map(([name, label]) => `<a href="${esc(f[name])}" download="${esc(name)}">${esc(label)}</a>`);
  links.push(`<span>Word, Excel: unrecorded</span>`);
  return `<p class="study-docs muted">${links.join(" · ")}</p>`;
}

/** A recording's figures as the board's figure entries: {id, src, caption, step}. */
export function recordedFigures(rec) {
  return ((rec && rec.figures) || []).filter((f) => f && f.url).map((f) => ({
    id: f.id || f.name, src: f.url, caption: f.caption || "", step: f.step || null, job: null,
  }));
}

/** Who planned a re-run: the recording's plan when the validator took it, else the tree and why. */
export function recordedPlanLine({ used = null, errors = [], model = null } = {}) {
  const by = model ? ` (${model})` : "";
  if (used === "proposed" || used === "device") return `the recorded plan${by}, re-run keyless`;
  if (used === "tree" && errors && errors.length) {
    return `the playbook's plan (the recorded plan did not pass the validator: ${errors[0]})`;
  }
  return "the playbook's plan";
}
