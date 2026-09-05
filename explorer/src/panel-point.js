// The "anywhere" inspector: click a spot that is not a gauge and get the
// climate of that point, the modelled river discharge, the catchment it sits
// in, the gauges nearest to it and what their regime suggests here.

import { $, actions, copyText, escapeHtml, fmt, haversineKm, sourceStyle, state, stationKey } from "./core.js?v=__BUILD__";
import { shapeSvg } from "./shapes.js?v=__BUILD__";
import { addTableDownload, plot } from "./charts.js?v=__BUILD__";
import { requestAssess } from "./assess.js?v=__BUILD__";
import { clearCatchment, requestBasin, requestCatchment } from "./basins.js?v=__BUILD__";
import { flyToPoint, setPointMarker, highlightStation } from "./map.js?v=__BUILD__";
import { addMethodOnce, methodsOnPage, openCite, renderMethodList } from "./methods.js?v=__BUILD__";
import { hideCard, selectTab, setCard, setTab, showSurface } from "./shell.js?v=__BUILD__";
import { call } from "./worker-client.js?v=__BUILD__";
import { canonicalUrl, writeUrl } from "./url.js?v=__BUILD__";

let pointRun = 0;
const root = () => $("panel-point");

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function nearestStations(lat, lon, n = 6) {
  const out = [];
  for (const r of state.stations) {
    if (state.hidden.has(r.source)) continue;
    const d = haversineKm(lat, lon, r.lat, r.lon);
    if (out.length < n) { out.push([d, r]); out.sort((a, b) => a[0] - b[0]); }
    else if (d < out[n - 1][0]) { out[n - 1] = [d, r]; out.sort((a, b) => a[0] - b[0]); }
  }
  return out;
}

export async function selectPoint(lat, lon, { tab = null, push = true, fly = false } = {}) {
  lat = Math.round(lat * 1e4) / 1e4;
  lon = Math.round(lon * 1e4) / 1e4;
  const my = ++pointRun;
  state.selected = null;
  state.result = null;
  state.point = { lat, lon };
  highlightStation(null);
  setPointMarker(lat, lon);
  if (fly) flyToPoint(lat, lon);

  showSurface("panel-point");
  $("pt-title").textContent = `${lat.toFixed(3)}°, ${lon.toFixed(3)}°`;
  $("pt-coords").textContent = `lat ${lat}, lon ${lon}`;
  for (const id of ["pt-climate-card", "pt-glofas-card", "pt-notes-card", "pt-assess-card"]) hideCard($(id));
  renderMethodList("pt-methods", []);
  $("pt-attribution").textContent = "";
  clearCatchment();
  for (const name of ["modelled", "catchment", "similar"]) {
    setTab(root(), name, { enabled: false, reason: "Looking this point up…", count: null });
  }
  setTab(root(), "overview", { enabled: true });
  setTab(root(), "methods", { enabled: true });
  // Push the selection before applying the tab (tab changes only replace).
  state.activeTab = tab && root().querySelector(`[role="tab"][data-tab="${tab}"]`) ? tab : "overview";
  writeUrl({ push });
  selectTab(root(), state.activeTab);

  // Nearest gauges are local, so they can be drawn immediately.
  const near = nearestStations(lat, lon);
  const ul = $("pt-nearest");
  ul.innerHTML = "";
  if (!near.length) {
    ul.innerHTML = `<li class="muted">no gauges in the catalog</li>`;
  } else {
    for (const [d, r] of near) {
      const li = document.createElement("li");
      li.tabIndex = 0;
      li.setAttribute("role", "button");
      li.dataset.key = stationKey(r);
      const st = sourceStyle(r.source);
      li.innerHTML = `${shapeSvg(st.shape, st.color)}${escapeHtml(r.name || r.station_id)} ` +
        `<span class="muted">${escapeHtml(st.label)}</span>` +
        `<span class="dist">${d < 10 ? d.toFixed(1) : Math.round(d)} km</span>`;
      const open = () => actions.selectStation(li.dataset.key, { fly: true });
      li.addEventListener("click", open);
      li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
      ul.appendChild(li);
    }
  }

  requestCatchment({ point: { lat, lon }, target: "pt" });
  requestBasin(lat, lon, "pt");
  requestAssess({ lat, lon, target: "pt" });

  setCard($("pt-climate-card"), "loading", {
    message: state.workerReady ? "Asking Open-Meteo about this point…" : "Loading Python in your browser (about 15 MB, once)…",
  });
  try {
    const res = await call("anywhere", { lat, lon, years: 10 });
    if (my !== pointRun || !state.point || state.point.lat !== lat || state.point.lon !== lon) return;
    renderPoint(res, { lat, lon });
  } catch (err) {
    if (my !== pointRun) return;
    setCard($("pt-climate-card"), "error", {
      message: `Could not describe this point: ${err.message}`,
      retry: () => selectPoint(lat, lon, { push: false }),
    });
  }
}

function renderPoint(res, { lat, lon }) {
  const c = res.climate;
  if (c) {
    $("pt-kpis").innerHTML = [
      ["rainfall", `${fmt(c.precipitation_mm_per_year, 0)} mm/yr`, "ERA5, 10-yr mean"],
      ["reference ET0", `${fmt(c.et0_mm_per_year, 0)} mm/yr`, "FAO-56"],
      ["aridity", `${fmt(c.aridity_index, 2)}`, c.aridity_class || "P / ET0"],
      ["temperature", `${fmt(c.temperature_mean_c, 1)} °C`, `wettest day ${fmt(c.wettest_day_mm, 0)} mm`],
    ].map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
    setCard($("pt-climate-card"), "ready");
    // Distinct month labels: Plotly merges categorical x values with the same
    // name, so one-letter months folded May/Jun/Jul/Aug into Mar/Jan/Apr.
    plot("plot-climate", [
      { x: MONTHS, y: c.monthly_precipitation_mm, type: "bar", name: "rain (mm)", marker: { color: "#1565c0" } },
      { x: MONTHS, y: c.monthly_et0_mm, type: "scatter", mode: "lines+markers", name: "ET0 (mm)", line: { color: "#ef6c00" } },
    ], {
      height: 220, yaxis: { title: { text: "mm / month" } },
      legend: { orientation: "h", y: 1.15 }, barmode: "group",
    }, `climate-${lat}-${lon}`);
  } else {
    setCard($("pt-climate-card"), "empty", { message: "No ERA5 climate for this point (it may be open sea)." });
  }

  const g = res.glofas;
  if (g && g.n) {
    setTab(root(), "modelled", { enabled: true });
    $("pt-glofas-kpis").innerHTML = [
      ["mean", `${fmt(g.stats.mean)} m³/s`, `${g.start} → ${g.end}`],
      ["max", `${fmt(g.stats.max)} m³/s`, "modelled daily"],
    ].map(([l, v, s]) => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
    const table = $("pt-ffa-table");
    if (g.ffa && g.ffa.fits) {
      const rps = g.ffa.return_periods, gv = g.ffa.fits.gev_lmoments || {}, lp = g.ffa.fits.lp3 || {};
      table.innerHTML = `<thead><tr><th>T (yr)</th><th>GEV L-moments</th><th>LP3 (90 % CI)</th></tr></thead><tbody>` +
        rps.map((rp, i) => `<tr><td>${rp}</td><td>${gv.q ? fmt(gv.q[i]) : "—"}</td>` +
          `<td>${lp.q ? `${fmt(lp.q[i])} <span class="ci">[${fmt(lp.ci[i][0])}, ${fmt(lp.ci[i][1])}]</span>` : "—"}</td></tr>`).join("") +
        `</tbody><tfoot><tr><td colspan="3" class="muted">Indicative only: GloFAS grid-cell discharge in m³/s, ${g.ffa.n_years} modelled years.</td></tr></tfoot>`;
      addTableDownload($("pt-glofas-actions"), table, `glofas-${lat}-${lon}-flood-frequency.csv`);
    } else {
      table.innerHTML = "";
    }
    setCard($("pt-glofas-card"), "ready");
  } else {
    setTab(root(), "modelled", { enabled: false, reason: "GloFAS has no modelled river discharge for this grid cell." });
  }

  const notes = res.notes || [];
  if (notes.length) {
    $("pt-notes").innerHTML = notes.map((n) => `<li>${escapeHtml(n)}</li>`).join("");
    setCard($("pt-notes-card"), "ready");
  } else hideCard($("pt-notes-card"));

  renderMethodList("pt-methods", res.methods || []);
  $("pt-attribution").textContent = res.attribution ? `Data: ${res.attribution}` : "";
}

export function initPointPanel() {
  const r = root();
  r.addEventListener("tabchange", (e) => {
    state.activeTab = e.detail.tab;
    writeUrl();
    for (const id of ["plot-climate"]) {
      const el = $(id);
      if (el && el.offsetParent !== null && el.data) Plotly.Plots.resize(el);
    }
  });
  $("btn-share-pt").addEventListener("click", (e) => copyText(canonicalUrl(), e.currentTarget, "Link copied"));
  $("btn-cite-pt").addEventListener("click", () => openCite(methodsOnPage("pt-methods")));
  actions.selectPoint = (lat, lon, opts) => selectPoint(lat, lon, opts);
}
