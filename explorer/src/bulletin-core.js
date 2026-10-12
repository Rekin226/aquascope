// The monthly bulletin (#523), the pure part: no DOM, no map, so node can test it.
// Every number in a bulletin comes from Python (aquascope.bulletin, run by the
// monthly workflow); this module only reads its index and says what the map
// legend and the reader show. The colours are now-core.js's (nowColor): the
// same five HydroSOS classes as "Today vs normal".

const MONTH = /^\d{4}-(0[1-9]|1[0-2])$/;

// The newest month in bulletins/index.json, or null when there is none (or the
// file is not what the workflow writes).
export function latestEntry(index) {
  if (!index || typeof index !== "object") return null;
  const months = Array.isArray(index.months) ? index.months.filter((m) => m && MONTH.test(m.month)) : [];
  const latest = MONTH.test(index.latest || "") ? index.latest : null;
  const hit = months.find((m) => m.month === latest) || months.slice().sort((a, b) => (a.month < b.month ? 1 : -1))[0];
  if (hit) return hit;
  return latest ? { month: latest } : null;
}

// Where a month's files live under the Archive's bulletins/ folder.
export function bulletinPaths(base, month) {
  const root = `${base}${month}/`;
  return { html: `${root}bulletin.html`, markdown: `${root}bulletin.md`, status: `${root}status.parquet`, folder: root };
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
  "November", "December"];

export function monthLabel(month) {
  const m = MONTH.exec(String(month || ""));
  return m ? `${MONTHS[Number(month.slice(5, 7)) - 1]} ${month.slice(0, 4)}` : "";
}

// The legend's sentence under the swatches.
export function bulletinLegendLine(meta) {
  if (!meta || meta.missing) {
    return "No bulletin yet: the first is written on the 3rd of next month. Until then the gauges keep their agency colours.";
  }
  const label = meta.label || monthLabel(meta.month);
  const n = Number(meta.classed) || 0;
  return `${label}: ${n.toLocaleString("en-US")} gauges classed against the same month in other years.`;
}

// What the Bulletin entry says when there is nothing to open yet.
export const NO_BULLETIN_TEXT = "No bulletin has been published yet. The monthly workflow writes the first one on the 3rd of next month, from every Archive gauge with a full month and ten years to compare with.";
