// Gauges coloured by today against normal (#517). The daily forecast job
// (forecast-archive.yml) writes forecasts/status/latest.parquet, one row per
// gauge with a fresh record, and latest.json saying when and from which
// sources. Read once, on first use, and only when the reader picks the style.

import { CONFIG } from "../config.js?v=__BUILD__";
import { escapeHtml, sourceStyle, state } from "./core.js?v=__BUILD__";
import { duck } from "./catalog.js?v=__BUILD__";
import { STATUS_CLASSES, snapshotLine } from "./now-core.js?v=__BUILD__";

let loading = null;

// Fills state.nowStatus (key -> { cls, pct, date, value, n_years }) and state.nowMeta. Before the
// first snapshot exists, nowMeta says so and nowStatus stays null, so the gauges
// keep their agency colours.
export function ensureNowStatus() {
  if (loading) return loading;
  loading = (async () => {
    const res = await fetch(`${CONFIG.forecastsBase}status/latest.json`, { cache: "no-cache" });
    if (!res.ok) {
      state.nowMeta = { missing: true };
      return;
    }
    const meta = await res.json();
    const { conn } = await duck();
    // value and n_years are for the map card's sentence (#548), in flow_status's own words.
    const table = await conn.query(`SELECT source, station_id, CAST(value_date AS VARCHAR) AS value_date, percentile,
      "class" AS cls, value, n_years FROM read_parquet('${CONFIG.forecastsBase}status/latest.parquet')`);
    const map = new Map();
    for (const row of table.toArray().map((r) => r.toJSON())) {
      map.set(`${row.source}/${row.station_id}`, { cls: row.cls, pct: Number(row.percentile), date: row.value_date,
        value: row.value === null || row.value === undefined ? null : Number(row.value),
        n_years: row.n_years === null || row.n_years === undefined ? null : Number(row.n_years) });
    }
    state.nowMeta = meta;
    state.nowStatus = map;
  })().catch((err) => {
    console.warn("status snapshot:", err && err.message);
    state.nowMeta = { missing: true };
    loading = null;  // a later pick tries again
  });
  return loading;
}

export function nowLegendHtml() {
  const swatch = (c, l) => `<span class="sw"><i style="background:${c}"></i>${escapeHtml(l)}</span>`;
  const meta = state.nowMeta;
  if (!meta) return `<p class="muted now-legend-note">Loading today's status…</p>`;
  const line = snapshotLine(meta, (s) => sourceStyle(s).label);
  if (meta.missing) return `<p class="muted now-legend-note">${escapeHtml(line)}</p>`;
  return STATUS_CLASSES.map((c) => swatch(c.color, c.label)).join("") +
    `<p class="muted now-legend-note">${escapeHtml(line)} Gauges without a fresh record keep their agency colour, drawn fainter.</p>`;
}
