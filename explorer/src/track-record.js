// The forecast track record (#785): how close the issued forecasts came to what
// the gauge then recorded, read from the Archive's daily verification
// (verify/, aquascope.archive.verify) and said on the river card, on a Floods
// ahead reach and in the legend.
//
// It reads verify/summary.json once (every gauge pooled, the window, the trust
// bands) and, with DuckDB, the 1-to-3-day rows of every gauge from
// verify/forecast_skill_gauges.parquet (a couple of thousand rows). When verify/
// is not published, or a gauge has no row, nothing is shown: no line, no badge,
// no legend row. The words are track-record-core.js's.

import { CONFIG } from "../config.js?v=__BUILD__";
import { escapeHtml, trace } from "./core.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { refreshLegend, registerLegendRow } from "./map-legend.js?v=__BUILD__";
import {
  indexRows, legendSummary, networkLines, plainRow, reachTrust, trackLines, trustColor, trustLegend, verifyBase, windowSql,
} from "./track-record-core.js?v=__BUILD__";

const BASE = CONFIG.verifyBase || verifyBase(CONFIG.stationsParquet);

let summary = null;       // verify/summary.json, or null when it is not published
let index = null;         // { byGauge, byReach } of the card's window, or null
let loading = null;
let visible = true;       // the legend row's switch: the cards' lines and badges

/** Load verify/ once; resolves to true when there is a track record to show. */
export function loadTrackRecord() {
  if (loading) return loading;
  loading = (async () => {
    try {
      const res = await fetch(`${BASE}summary.json`, { cache: "no-cache", credentials: "omit" });
      if (!res.ok) return false;
      summary = await res.json();
    } catch (err) {
      trace(`forecast track record not available: ${err && err.message}`);
      return false;
    }
    try {
      const file = (summary.gauges && summary.gauges.file) || "verify/forecast_skill_gauges.parquet";
      const url = `${BASE}${String(file).replace(/^verify\//, "")}`;
      const { conn } = await duck();
      const t = await conn.query(windowSql(url));
      index = indexRows(t.toArray().map((r) => plainRow(r.toJSON())));
      trace(`forecast track record: ${index.byGauge.size} gauges`);
    } catch (err) {
      index = null;   // the pooled numbers can still show in the legend
      trace(`forecast track record per gauge not available: ${err && err.message}`);
    }
    refreshLegend("track-record");
    return true;
  })();
  return loading;
}

/** The card's block for a gauge ("source/station_id"), or null. Corrected to the gauge where it was issued so. */
export async function trackForGauge(key) {
  await loadTrackRecord();
  if (!visible || !index) return null;
  return trackLines(index.byGauge.get(key) || [], { variant: "corrected" });
}

/** The card's block for a river reach (raw, as the reach's forecast is), from the gauges on it; or null. */
export async function trackForReach(riverId) {
  await loadTrackRecord();
  if (!visible || !index) return null;
  return trackLines(index.byReach.get(Number(riverId)) || [], { variant: "raw" });
}

/** Floods ahead's badge for a reach ({trust, text}), or null when verify/ is not published. */
export async function trustForReach(riverId) {
  await loadTrackRecord();
  if (!visible || !index) return null;
  return reachTrust(index.byReach.get(Number(riverId)) || []);
}

/** The gauge's rows of the card's window, for the evidence ladder's ranking; [] when there are none. */
export async function verifiedRows(key) {
  await loadTrackRecord();
  return (index && index.byGauge.get(key)) || [];
}

/** A small badge: the colour and always the word. */
export function trustChip(trust) {
  return `<span class="trust-chip" style="--trust:${trustColor(trust)}">${escapeHtml(trust)}</span>`;
}

// ── the row in "On the map" ─────────────────────────────────────────────────

const mark = () => `<span class="fa-mini">${trustLegend().map((b) => `<i style="background:${b.color}"></i>`).join("")}</span>`;

function body() {
  const key = trustLegend().map((b) => `<li>${trustChip(b.trust)} ${escapeHtml(b.label.replace(/^\w+: /, ""))}</li>`).join("");
  const net = networkLines(summary);
  const n = summary && summary.gauges ? Number(summary.gauges.n_gauges || 0) : 0;
  return `<p class="ml-when">How close the issued forecasts came to the gauge, ${escapeHtml(legendSummary(summary))}, ` +
    `1 to 3 days ahead: half of them were within the factor shown on the card.</p>` +
    `<ul class="tr-key">${key}</ul>` +
    (net.length ? `<p class="ml-src">Every gauge pooled, 1 day ahead: ${net.map(escapeHtml).join("; ")}.</p>` : "") +
    `<p class="ml-src">${n ? `${n.toLocaleString("en-GB")} forecast gauges scored. ` : ""}` +
    `Scored daily from the Archive's issued forecasts and gauge values (verify/). Model forecasts, not official warnings.</p>`;
}

export function initTrackRecord() {
  registerLegendRow({
    id: "track-record", title: "Forecast track record",
    mark,
    summary: () => legendSummary(summary),
    on: () => visible,
    empty: () => !summary,
    shown: () => Boolean(summary),
    toggle: (on) => { visible = Boolean(on); },
    body,
  });
  // After the first paint: DuckDB is busy with the catalogue just then.
  setTimeout(() => { void loadTrackRecord(); }, 3000);
}
