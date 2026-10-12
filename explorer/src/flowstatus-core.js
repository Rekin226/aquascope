// Flow against normal (#559), the pure part: no DOM, no map, so node can test it.
//
// Every GEOGLOWS reach of Strahler order 5 and up, classed once a day by its
// forecast flow for the day against its own normal for the month (the 10th,
// 25th, 75th and 90th percentiles of daily flow, 1950 to 2019). The classes
// are computed in Python (aquascope/flowstatus.py, published daily under
// flowstatus/ by flood-warnings.yml); this module only reads the compact file,
// looks a reach up and says what the colours mean. The colours are the river
// status layer's five (now-core.js STATUS_CLASSES), so a river and the basin
// around it mean the same thing.

import { STATUS_CLASSES } from "./now-core.js?v=__BUILD__";
import { shortDay } from "./floods-ahead-core.js?v=__BUILD__";

export const FS_MIN_ORDER = 5;
export const FS_CREDIT = {
  label: "Flow vs normal",
  attribution: 'Flow vs normal: GEOGLOWS v2 forecast and retrospective flow duration curves ' +
    '(<a href="https://geoglows.org">GEOGloWS ECMWF Streamflow Service</a>), classed by AquaScope',
  licence: "CC BY 4.0",
};
// The lines the file has no class for (order 4 and under once you zoom in, a dry channel, a reach with no
// forecast): a quiet grey, thinner in effect through a lower opacity, so the classes carry the picture.
export const FS_UNCLASSED = { light: "#9aa5b1", dark: "#5c6670", imagery: "#c9d1d9" };
export const FS_UNCLASSED_OPACITY = 0.35;

/** Class code (1 much below .. 5 much above) to the river status class: id, label, colour. */
export const fsClass = (code) => STATUS_CLASSES[Number(code) - 1] || null;

// ── the file ────────────────────────────────────────────────────────────────

/**
 * The reaches and their classes from the file's bytes, once gunzipped: "AQFS", version 1, n (uint32 LE),
 * n LEB128 gaps between sorted river ids, n class bytes. Returns { ids: Uint32Array, classes: Uint8Array }.
 */
export function decodeFlowStatus(bytes) {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (b.length < 9 || b[0] !== 0x41 || b[1] !== 0x51 || b[2] !== 0x46 || b[3] !== 0x53 || b[4] !== 1) {
    throw new Error("not a flow status file (AQFS version 1)");
  }
  const n = (b[5] | (b[6] << 8) | (b[7] << 16)) + b[8] * 16777216;
  const ids = new Uint32Array(n);
  let pos = 9, acc = 0;
  for (let i = 0; i < n; i++) {
    let shift = 0, val = 0, byte;
    do {
      if (pos >= b.length) throw new Error("flow status file is cut short");
      byte = b[pos++];
      val += (byte & 0x7f) * 2 ** shift;   // not <<: a gap can pass 2^31
      shift += 7;
    } while (byte >= 0x80);
    acc += val;
    ids[i] = acc;
  }
  if (pos + n > b.length) throw new Error("flow status file is cut short");
  return { ids, classes: b.slice(pos, pos + n) };
}

/** The class code of reach `id` (0 when it has none): a binary search of the sorted ids. */
export function classOf(data, id) {
  if (!data || !data.ids) return 0;
  const { ids, classes } = data;
  const want = Number(id);
  let lo = 0, hi = ids.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const v = ids[mid];
    if (v === want) return classes[mid];
    if (v < want) lo = mid + 1; else hi = mid - 1;
  }
  return 0;
}

/** Whether the bytes are gzip (a server may already have undone it). */
export const isGzip = (b) => b && b.length > 1 && b[0] === 0x1f && b[1] === 0x8b;

// ── the style ───────────────────────────────────────────────────────────────

const FS = ["coalesce", ["feature-state", "fs"], 0];

/** The river line's colour: its class where it has one, the quiet grey elsewhere. */
export function flowColorExpr(basemapKind = "light") {
  const grey = FS_UNCLASSED[basemapKind] || FS_UNCLASSED.light;
  return ["match", FS, ...STATUS_CLASSES.flatMap((c, i) => [i + 1, c.color]), grey];
}

/** A reach with a class: the test the opacity uses. */
export const FS_CLASSED = [">", FS, 0];

// ── the words ───────────────────────────────────────────────────────────────

/** "11 Oct" style issue day, from the manifest. */
export const issueDay = (manifest) => shortDay(manifest && manifest.issue_date);

/** The legend row's few words: what the lines show and the day. */
export function rowSummary(manifest, { loading = false, error = false } = {}) {
  if (loading) return "flow vs normal, reading";
  if (error) return "flow vs normal, could not read";
  if (!manifest || manifest.missing) return "flow vs normal, not published yet";
  return `flow vs normal, ${issueDay(manifest)}${manifest.smoke ? " (sample)" : ""}`;
}

/** Of the classed reaches, the share above and below normal, from the manifest's counts (the claim lock: code). */
export function sharesLine(manifest) {
  const c = (manifest && manifest.counts) || {};
  const n = STATUS_CLASSES.reduce((s, k) => s + (Number(c[k.id]) || 0), 0);
  if (!n) return "";
  const pct = (k) => Math.round((100 * k) / n);
  const hi = (Number(c.above) || 0) + (Number(c.much_above) || 0);
  const lo = (Number(c.below) || 0) + (Number(c.much_below) || 0);
  return `Of ${n.toLocaleString("en-GB")} reaches, ${pct(hi)}% above normal and ${pct(lo)}% below.`;
}

const MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
  "October", "November", "December"];

/** The line under the key: the forecast day against which normal. */
export function issueLine(manifest) {
  if (!manifest || manifest.missing) return "Not published yet. It appears after the first daily run.";
  const month = MONTH_NAMES[Number(manifest.month) - 1] || "the month";
  const period = manifest.base_period || "1950 to 2019";
  return `GEOGLOWS forecast for ${issueDay(manifest)} against each reach's ${month} normal (${period})` +
    `${manifest.smoke ? ", a smoke sample only" : ""}.`;
}

/** The flag in a link: #fn=1 on, #fn=0 off, null when the link does not say. */
export function readFlowNormalParam(hash) {
  const q = new URLSearchParams(String(hash || "").replace(/^#/, ""));
  if (!q.has("fn")) return null;
  return q.get("fn") === "1";
}
