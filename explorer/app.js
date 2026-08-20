// AquaScope Explorer: catalog -> map -> click -> worker (Pyodide) -> panels.
// No build step. Everything static; the only servers are a CDN, the Archive on
// Hugging Face, and the agencies' own APIs.
//
// This file is the composition root: it boots the pieces in src/ and owns the
// single "apply this URL" path that Back, a pasted link and a deep link all
// go through.

import { $, actions, state, trace } from "./src/core.js?v=__BUILD__";
import { loadCatalog, toFeatureCollection } from "./src/catalog.js?v=__BUILD__";
import { addStationLayers, initMap, map, refreshMapData, setView, webglAvailable } from "./src/map.js?v=__BUILD__";
import { defaultDate } from "./src/layers.js?v=__BUILD__";
import { applyLayerState, initLayerUI, syncRailControls } from "./src/layer-ui.js?v=__BUILD__";
import { buildRail, syncRail, updateCount } from "./src/rail.js?v=__BUILD__";
import { setBasinsVisible } from "./src/basins.js?v=__BUILD__";
import { initSearch } from "./src/search.js?v=__BUILD__";
import { initShell, initTabs, selectTab, setStatusEl, showSurface } from "./src/shell.js?v=__BUILD__";
import { initStationPanel, selectStation } from "./src/panel-station.js?v=__BUILD__";
import { initPointPanel, selectPoint } from "./src/panel-point.js?v=__BUILD__";
import { initWorkbench, openWorkbench } from "./src/panel-workbench.js?v=__BUILD__";
import { initAsk } from "./src/ask.js?v=__BUILD__";
import { initUrl, readUrl, writeUrl } from "./src/url.js?v=__BUILD__";
import { ensureWorker } from "./src/worker-client.js?v=__BUILD__";
import { openCite } from "./src/methods.js?v=__BUILD__";
import { registerWebMcpTools } from "./src/webmcp.js?v=__BUILD__";

// Everything that can arrive from a URL: a station, a point, a tab, the map
// view and the source filter. Called at boot, on hashchange and on Back.
function applyUrl(url, { fromHistory = false } = {}) {
  if (url.hidden) {
    state.hidden = new Set(url.hidden);
    refreshMapData();
    syncRail();
  }
  if (fromHistory && readLayerState(url)) applyLayerState();
  if (url.basins !== undefined && url.basins !== state.basinsOn) setBasinsVisible(url.basins);
  if (url.view) { state.view = url.view; setView(url.view); }
  if (url.mode === "workbench") { openWorkbench(); return; }
  if (url.station) {
    const key = decodeURIComponent(url.station);
    if (!state.selected || `${state.selected.source}/${state.selected.station_id}` !== key) {
      selectStation(key, { fly: !url.view, tab: url.tab, push: false });
    } else if (url.tab) {
      selectTab($("panel-station"), url.tab);
    }
    return;
  }
  if (url.point) {
    const p = url.point;
    if (!state.point || state.point.lat !== p.lat || state.point.lon !== p.lon) {
      selectPoint(p.lat, p.lon, { tab: url.tab, push: false });
    } else if (url.tab) {
      selectTab($("panel-point"), url.tab);
    }
    return;
  }
  if (fromHistory) {           // back to the start: show the welcome surface again
    state.selected = null;
    state.point = null;
    showSurface("panel-empty");
  }
}

// Copy the layer part of a URL into state. Returns true when anything changed,
// so Back and a pasted link both restore the map as it was.
function readLayerState(url) {
  let changed = false;
  const set = (key, value) => {
    if (value === undefined || value === null) return;
    if (state[key] !== value) { state[key] = value; changed = true; }
  };
  set("basemap", url.basemap);
  set("date", url.date);
  set("terrain", url.terrain);
  set("hillshade", url.hillshade);
  set("globe", url.globe);
  set("gaugeStyle", url.gaugeStyle);
  set("heat", url.heat);
  if (url.overlays) {
    const next = new Set(url.overlays);
    if (next.size !== state.overlays.size || [...next].some((o) => !state.overlays.has(o))) {
      state.overlays = next;
      changed = true;
    }
  } else if (state.overlays.size) {
    state.overlays = new Set();
    changed = true;
  }
  return changed;
}

function goHome() {
  state.selected = null;
  state.point = null;
  state.activeTab = null;
  showSurface("panel-empty");
  writeUrl({ push: true });
}

(async function boot() {
  trace("boot");
  const url = readUrl();

  initShell();
  initTabs($("panel-station"));
  initTabs($("panel-point"));
  initTabs($("panel-workbench"));
  initStationPanel();
  initPointPanel();
  initWorkbench();
  initAsk();   // async: fills the provider list from providers.json
  initSearch();
  initUrl();
  actions.applyUrl = applyUrl;
  actions.refreshMapData = () => { refreshMapData(); syncRail(); };
  $("btn-home").addEventListener("click", goHome);
  $("btn-cite-top").addEventListener("click", () => openCite([]));
  for (const chip of document.querySelectorAll("[data-try]")) {
    chip.addEventListener("click", () => {
      if (chip.dataset.try === "s") selectStation(chip.dataset.key, { fly: true });
      else if (chip.dataset.try === "p") selectPoint(Number(chip.dataset.lat), Number(chip.dataset.lon));
      else actions.openAsk();
    });
  }

  if (url.hidden) state.hidden = new Set(url.hidden);
  state.date = url.date || defaultDate();
  readLayerState(url);

  const mapReady = initMap(url.view, { basemap: state.basemap, date: state.date });
  trace("map init called");
  const catalogReady = loadCatalog().then(() => true).catch((err) => {
    console.error(err);
    $("count").textContent = "catalog unavailable";
    setStatusEl($("boot-error"), `Could not load the station catalog: ${err.message}. The map and search need it; try reloading.`, "error");
    return false;
  });

  const [mapResult, catalogOk] = await Promise.all([mapReady, catalogReady]);
  const mapOk = Boolean(mapResult && mapResult.ok);
  state.mapOk = mapOk;
  trace(`ready: map=${mapOk} catalog=${catalogOk} stations=${state.stations.length}`);

  if (!mapOk) {
    // Say which of the two it actually is, instead of blaming WebGL for a slow
    // network (the old page always claimed "WebGL is off").
    const why = !webglAvailable()
      ? "This browser has WebGL turned off, so the map cannot draw. Search still works, and every gauge page below works."
      : "The map is taking longer than usual to load. Search still works; reload to try the map again.";
    setStatusEl($("map-fallback-text"), why, "warn");
    $("map-fallback").hidden = false;
  }
  if (!catalogOk) return;

  if (mapOk) {
    addStationLayers(toFeatureCollection(state.stations));
    // Default view: US west coast to Taiwan, which covers every source we
    // currently harvest.
    if (!url.view && !url.station && !url.point) {
      map.fitBounds([[-128, 12], [128, 62]], { padding: 12, animate: false });
    }
  }
  buildRail();
  updateCount();
  if (mapOk) {
    initLayerUI();
    applyLayerState();
    syncRailControls();
  }
  if (state.basinsOn || url.basins) setBasinsVisible(true);
  ensureWorker();  // warm Python in the background so the first click is quicker

  applyUrl(url);
  // Offer the page's tools to an in-browser agent, where the browser has WebMCP.
  registerWebMcpTools({ actions });
  state.booting = false;
})();
