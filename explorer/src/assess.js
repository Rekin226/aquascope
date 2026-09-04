// "What can be answered here": the sufficiency table from
// aquascope.explore.assess_site (the same function behind `aquascope assess`
// and the MCP tool), as one quiet card: three counts on a line, the rows
// behind a disclosure. The page passes the catchment area and donor count it
// already holds, since the worker cannot read BasinATLAS or the donor table.
// Solve reuses the card (target "solve", narrowed to its problem) and takes
// the reconnaissance back, so the plan is filled from the same dict.

import { $, VAR_LABEL, escapeHtml, fmt, state } from "./core.js?v=__BUILD__";
import { catchmentAreaAt, donorPoolSize, stationArea } from "./basins.js?v=__BUILD__";
import { setCard } from "./shell.js?v=__BUILD__";
import { call, ensureCatalogInWorker } from "./worker-client.js?v=__BUILD__";

const RADIUS_KM = 50;
const DONOR_K = 10;
const STATUS = {
  defensible: ["ok", "defensible"],
  marginal: ["warn", "marginal"],
  not_defensible: ["no", "not defensible"],
};
// One counter per card: Solve's request must not cancel the inspector's.
const runs = {};

export async function requestAssess({ lat, lon, target, key = null, problem = null }) {
  const my = (runs[target] = (runs[target] || 0) + 1);
  const el = $(`${target}-assess-card`);
  if (!el) return null;
  setCard(el, "loading", {
    message: state.workerReady ? "Checking what the record here supports…" : "Loading Python in your browser (about 15 MB, once)…",
  });
  try {
    const [area, pool] = await Promise.all([
      (key ? stationArea(key).then((a) => (a ? a.area : null)) : catchmentAreaAt(lat, lon)).catch(() => null),
      donorPoolSize().catch(() => 0),
    ]);
    if (my !== runs[target]) return null;
    await ensureCatalogInWorker();
    const res = await call("assess", {
      lat, lon, radius_km: RADIUS_KM, area_km2: area, donors: area ? Math.min(DONOR_K, pool) : null, problem,
    });
    if (my !== runs[target]) return null;
    render(el, res, key);
    return res;
  } catch (err) {
    if (my !== runs[target]) return null;
    setCard(el, "error", {
      message: `Could not assess this site: ${err.message}`,
      retry: () => requestAssess({ lat, lon, target, key, problem }),
    });
    return null;
  }
}

function stationLabel(st) {
  const row = state.byKey.get(`${st.source}/${st.station_id}`);
  return row ? (row.name || row.station_id) : st.station_id;
}

function render(el, res, key) {
  const rows = res.sufficiency || [];
  const n = { defensible: 0, marginal: 0, not_defensible: 0 };
  for (const r of rows) n[r.status] = (n[r.status] || 0) + 1;
  el.querySelector(".assess-counts").innerHTML = Object.entries(STATUS)
    .map(([k, [cls, word]]) => `<span class="${cls}">${n[k]} ${word}</span>`).join(" · ");

  const ctx = res.context || {};
  const years = Object.entries(ctx.years_by_variable || {});
  const bits = [years.length
    ? years.map(([v, y]) => `${VAR_LABEL[v] || v} ${fmt(y, 0)} yr`).join(", ")
    : `no gauge record within ${RADIUS_KM} km`];
  if (ctx.area_km2) bits.push(`catchment ${fmt(ctx.area_km2, 0)} km²`);
  if (ctx.donors) bits.push(`${ctx.donors} donors`);
  el.querySelector(".assess-foot").textContent = bits.join(" · ");

  el.querySelector(".assess-rows").innerHTML = rows.map((r) => {
    const [cls, word] = STATUS[r.status] || ["", r.status];
    const own = r.station && `${r.station.source}/${r.station.station_id}` === key;
    const via = r.station && !own ? ` · ${escapeHtml(stationLabel(r.station))}` : "";
    return `<li><span class="assess-label">${escapeHtml(r.label)}</span>` +
      `<span class="assess-status ${cls}">${word}</span>` +
      `<span class="assess-why muted">${escapeHtml(r.reason)}${via}</span></li>`;
  }).join("");
  el.querySelector(".assess-notes").innerHTML = (res.notes || []).map((t) => `<li>${escapeHtml(t)}</li>`).join("");
  el.querySelector("details").open = false;
  setCard(el, "ready");
}
