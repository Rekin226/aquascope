// Job health (#593): whether the scheduled jobs behind the map are on time.
//
// Reads health/jobs/index.json once (written by every scheduled workflow's last step, see
// aquascope/archive/health_jobs.py), then: a layer whose job is late says so in its legend row ("data 3 d old,
// expected daily"), and Tools > Data status lists every job. Before the first job has published its record, or
// when the file cannot be read, nothing is said: no record is not news about the data.
// The rules are in job-health-core.js.

import { CONFIG } from "../config.js?v=__BUILD__";
import { $, escapeHtml } from "./core.js?v=__BUILD__";
import { HEALTH_INDEX_URL, legendNotes, statusHeadline, statusItems } from "./job-health-core.js?v=__BUILD__";
import { setLegendNotes } from "./map-legend.js?v=__BUILD__";

let index = null;

function render() {
  setLegendNotes(legendNotes(index));
  const el = $("data-status");
  if (!el) return;
  const items = statusItems(index);
  el.hidden = !items.length;
  if (!items.length) return;
  const list = items.map((i) =>
    `<li class="ds-${i.state.replace(/\s+/g, "-")}" title="${escapeHtml(i.title)}"><i aria-hidden="true"></i>` +
    `<span class="ds-name">${escapeHtml(i.label)}</span><span class="ds-age">${escapeHtml(i.text)}</span></li>`).join("");
  const wasOpen = el.querySelector("details")?.open;
  el.innerHTML = `<details${wasOpen ? " open" : ""}><summary>${escapeHtml(statusHeadline(items))}</summary>` +
    `<ul class="ds-list">${list}</ul></details>`;
}

export async function initJobHealth() {
  try {
    const res = await fetch(CONFIG.healthIndex || HEALTH_INDEX_URL, { cache: "no-cache" });
    index = res.ok ? await res.json() : null;
  } catch {
    index = null;
  }
  render();
  // Ages move while the page stays open: read them again whenever the Tools menu opens.
  document.querySelector(".header-tools")?.addEventListener("toggle", (e) => { if (e.target.open) render(); });
}
