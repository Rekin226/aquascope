// AquaScope Explorer: catalog -> map -> click -> worker (Pyodide) -> panels.
// No build step. Everything static; the only servers are a CDN, the Archive on
// Hugging Face, and the agencies' own APIs.
//
// This file is the composition root: it boots the pieces in src/ and owns the
// single "apply this URL" path that Back, a pasted link and a deep link all
// go through.

import { $, actions, state, trace } from "./src/core.js?v=__BUILD__";
import { loadCatalog, toFeatureCollection } from "./src/catalog.js?v=__BUILD__";
import {
  DEFAULT_CENTER, addStationLayers, fitWorldZoom, flyToStation, highlightStation, initMap, map,
  refreshMapData, setPointMarker, setView, syncMapPadding, watchPanelSizes, webglAvailable,
  whenMapLoadsLate,
} from "./src/map.js?v=__BUILD__";
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
import { initSolve } from "./src/solve.js?v=__BUILD__";
import { initUrl, readUrl, writeUrl } from "./src/url.js?v=__BUILD__";
import { ensureWorker } from "./src/worker-client.js?v=__BUILD__";
import { openCite } from "./src/methods.js?v=__BUILD__";
import { registerWebMcpTools } from "./src/webmcp.js?v=__BUILD__";

// The Solve drawer, after the selection it belongs to has been applied (a
// selection closes the drawer, so the order matters).
function openSolveIf(url) {
  if (url.solve) actions.openSolve(url.solve === "1" ? null : url.solve);
}

// Everything that can arrive from a URL: a station, a point, a tab, the map
// view, the source filter and the Solve drawer. Called at boot, on hashchange
// and on Back.
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
    openSolveIf(url);
    return;
  }
  if (url.point) {
    const p = url.point;
    if (!state.point || state.point.lat !== p.lat || state.point.lon !== p.lon) {
      selectPoint(p.lat, p.lon, { tab: url.tab, push: false, fly: !url.view });
    } else if (url.tab) {
      selectTab($("panel-point"), url.tab);
    }
    openSolveIf(url);
    return;
  }
  if (fromHistory) {           // back to the start: show the welcome surface again
    state.selected = null;
    state.point = null;
    showSurface("panel-empty");
  }
  openSolveIf(url);
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

// Everything that only makes sense once the map can draw. Called at boot when
// the map is ready, and again from whenMapLoadsLate if it arrives after the
// timeout, so a slow map ends up in the same state as a fast one instead of
// staying empty behind a warning until someone reloads.
function bringMapOnline(url) {
  state.mapOk = true;
  $("map-fallback").hidden = true;
  setStatusEl($("map-fallback-text"), "");
  addStationLayers(toFeatureCollection(state.stations));
  syncMapPadding();
  watchPanelSizes();
  // Default view: the whole world, framed to the map's actual size so the globe
  // fills it on a monitor and still fits on a phone.
  if (!url.view && !url.station && !url.point) {
    map.jumpTo({ center: DEFAULT_CENTER, zoom: fitWorldZoom($("map"), { globe: state.globe }) });
  } else if (url.view) {
    setView(url.view);
  }
  initLayerUI();
  applyLayerState();
  syncRailControls();
  if (state.basinsOn || url.basins) setBasinsVisible(true);
  // A selection made while the map was still dark has nothing on the map yet.
  if (state.selected) {
    highlightStation(`${state.selected.source}/${state.selected.station_id}`);
    if (!url.view) flyToStation(state.selected);
  } else if (state.point) {
    setPointMarker(state.point.lat, state.point.lon);
  }
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
  initSolve(); // async: fills the problem chips from playbooks.json
  initSearch();
  initUrl();
  actions.applyUrl = applyUrl;
  actions.refreshMapData = () => { refreshMapData(); syncRail(); };
  $("btn-home").addEventListener("click", goHome);
  $("btn-cite-top").addEventListener("click", () => openCite([]));
  for (const chip of document.querySelectorAll("[data-try]")) {
    chip.addEventListener("click", () => {
      if (chip.dataset.try === "s") selectStation(chip.dataset.key, { fly: true });
      else if (chip.dataset.try === "p") selectPoint(Number(chip.dataset.lat), Number(chip.dataset.lon), { fly: true });
      else actions.openAsk();
    });
  }

  if (url.hidden) state.hidden = new Set(url.hidden);
  state.date = url.date || defaultDate();
  readLayerState(url);
  // A white basemap inside a dark interface is a lamp in a dark room. With no
  // basemap in the URL, follow the reader's system theme; "Copy link" then
  // carries b=dark, so what they send is what they were looking at.
  if (url.basemap === undefined && globalThis.matchMedia
      && matchMedia("(prefers-color-scheme: dark)").matches) {
    state.basemap = "dark";
  }

  const mapReady = initMap(url.view, { basemap: state.basemap, date: state.date, globe: state.globe });
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

  buildRail();
  updateCount();
  if (mapOk) bringMapOnline(url);
  else if (mapResult && mapResult.reason === "slow") whenMapLoadsLate(() => bringMapOnline(url));
  ensureWorker();  // warm Python in the background so the first click is quicker

  applyUrl(url);
  // Offer the page's tools to an in-browser agent, where the browser has WebMCP.
  registerWebMcpTools({ actions });
  state.booting = false;
})();
