// AquaScope Explorer: catalog -> map -> click -> worker (Pyodide) -> panels.
// No build step. Everything static; the only servers are a CDN, the Archive on
// Hugging Face, and the agencies' own APIs.
//
// This file is the composition root: it boots the pieces in src/ and owns the
// single "apply this URL" path that Back, a pasted link and a deep link all
// go through.

import { $, actions, setTime, state, trace } from "./src/core.js?v=__BUILD__";
import { loadCatalog, toFeatureCollection } from "./src/catalog.js?v=__BUILD__";
import {
  DEFAULT_CENTER, addStationLayers, fitWorldZoom, flyToStation, highlightStation, initMap, map,
  refreshMapData, setPointMarker, setView, syncMapPadding, watchPanelSizes, webglAvailable,
  whenMapLoadsLate,
} from "./src/map.js?v=__BUILD__";
import { defaultDate } from "./src/layers.js?v=__BUILD__";
import { applyLayerState, initLayerUI, renderCredits, syncRailControls } from "./src/layer-ui.js?v=__BUILD__";
import { initTimeBar } from "./src/time-ui.js?v=__BUILD__";
import { initFloodsPast } from "./src/floods-past.js?v=__BUILD__";  // Floods past (#547)
import { initStatusLayer } from "./src/status-layer.js?v=__BUILD__";
import { buildRail, syncRail, updateCount } from "./src/rail.js?v=__BUILD__";
import { setBasinsVisible } from "./src/basins.js?v=__BUILD__";
import { setRiversVisible } from "./src/river-map.js?v=__BUILD__";
import { clearRiver, initRiver } from "./src/river.js?v=__BUILD__";
import { initNow } from "./src/now.js?v=__BUILD__";
import { initBulletin } from "./src/bulletin.js?v=__BUILD__";
import { initFloodsAhead } from "./src/floods-ahead.js?v=__BUILD__";  // Floods ahead (#546), on by default
import { initFloodDepth } from "./src/flood-depth.js?v=__BUILD__";  // flood depth where floods are forecast (#554)
import { initSearch } from "./src/search.js?v=__BUILD__";
import { initShell, initTabs, selectTab, setStatusEl, showSurface } from "./src/shell.js?v=__BUILD__";
import { initStationPanel, reanalyze, selectStation, setPeriod } from "./src/panel-station.js?v=__BUILD__";
import { initPointPanel, selectPoint } from "./src/panel-point.js?v=__BUILD__";
import { initWorkbench, openSampleTable, openWorkbench } from "./src/panel-workbench.js?v=__BUILD__";
import { initAsk } from "./src/ask.js?v=__BUILD__";
import { initUrl, readUrl, writeUrl } from "./src/url.js?v=__BUILD__";
import { ensureWorker } from "./src/worker-client.js?v=__BUILD__";
import { openCite } from "./src/methods.js?v=__BUILD__";
import { registerWebMcpTools } from "./src/webmcp.js?v=__BUILD__";
import { studyUrlParam } from "./src/study-link.js?v=__BUILD__";
import { initSignatureFilter } from "./src/signature-filter.js?v=__BUILD__";
import { initPlaces } from "./src/places.js?v=__BUILD__";  // My places + Compare
import { greetOnLoad, initWatch } from "./src/watch.js?v=__BUILD__";  // Watch: since you were here (#521)
import { loadAvailability } from "./src/availability.js?v=__BUILD__";
import { initMapCard } from "./src/map-card.js?v=__BUILD__";  // map first (#548): a click answers on the map
import { initMapActions } from "./src/map-actions.js?v=__BUILD__";  // the AI's action log with undo (#561)
import { initMapCommand } from "./src/map-command.js?v=__BUILD__";  // Ask the map (#561)

import { initMetrics } from "./src/metrics-ui.js?v=__BUILD__";

// Study is loaded when it is first used (the Study button, the drawer's radio,
// "Study this place", a #study=1 link): its modules are the larger part of the
// drawer's code and most visits run no study. The button is wired here,
// synchronously, before anything is awaited (#271), and the click loads the
// module and toggles the drawer.
let studyLoading = null;
function loadStudy() {
  if (!studyLoading) {
    studyLoading = import("./src/studio.js?v=__BUILD__").then((m) => { m.initStudy(); return m; });
    studyLoading.catch((err) => {
      console.error(err);
      studyLoading = null;
      setStatusEl($("study-status"), `Study could not load: ${err.message}. Reload to try again.`, "error");
    });
  }
  return studyLoading;
}

function initStudyLoader() {
  actions.openStudy = (opts) => loadStudy().then((m) => m.openStudy(opts)).catch(() => {});
  actions.openSharedStudy = (opts) => loadStudy().then((m) => m.openSharedStudy(opts)).catch(() => {});
  $("btn-study").addEventListener("click", () => loadStudy().then((m) => m.toggleStudy()).catch(() => {}));
  $("drawer").addEventListener("drawermode", (e) => { if (e.detail.mode === "study") loadStudy(); });
}

// The Study drawer, after the selection it belongs to has been applied (a
// selection closes the drawer, so the order matters).
function openStudyIf(url) {
  if (url.study && url.studyLink) { actions.openSharedStudy({ link: url.studyLink }); return; }   // study links
  if (url.study) actions.openStudy(url.studyId ? { recorded: url.studyId } : {});
}

// Everything that can arrive from a URL: a station, a point, a tab, the map
// view, the source filter, the analysis period and the Study drawer. Called at boot, on hashchange
// and on Back.
function applyUrl(url, { fromHistory = false } = {}) {
  if (url.hidden) {
    state.hidden = new Set(url.hidden);
    refreshMapData();
    syncRail();
  }
  if (fromHistory && readLayerState(url)) applyLayerState();
  if (fromHistory) {   // the map date, step, range and compare (#522), through the one setter
    setTime({ date: url.date || state.date, step: url.step || "day", range: url.range || null,
      compare: url.compare || null }, { source: "url" });
  }
  if (url.basins !== undefined && url.basins !== state.basinsOn) setBasinsVisible(url.basins);
  if (url.rivers !== undefined && url.rivers !== state.riversOn) { setRiversVisible(url.rivers); renderCredits(); }
  if (url.view) { state.view = url.view; setView(url.view); }
  if (url.mode === "workbench") { openWorkbench(); return; }
  if (url.station) {
    const key = decodeURIComponent(url.station);
    const periodChanged = setPeriod(url.period);   // &yr= (#270); no yr is the page default
    if (!state.selected || `${state.selected.source}/${state.selected.station_id}` !== key) {
      selectStation(key, { fly: !url.view, tab: url.tab, push: false });
    } else {
      if (periodChanged) reanalyze();
      if (url.tab) selectTab($("panel-station"), url.tab);
    }
    openStudyIf(url);
    return;
  }
  if (url.point) {
    const p = url.point;
    if (!state.point || state.point.lat !== p.lat || state.point.lon !== p.lon) {
      selectPoint(p.lat, p.lon, { tab: url.tab, push: false, fly: !url.view });
    } else if (url.tab) {
      selectTab($("panel-point"), url.tab);
    }
    openStudyIf(url);
    return;
  }
  if (fromHistory) {           // back to the start: the welcome surface again, the panel left as it was (#548)
    state.selected = null;
    state.point = null;
    state.activeTab = null;
    showSurface("panel-empty", { reveal: false });
  }
  openStudyIf(url);
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
  set("terrain", url.terrain);
  set("hillshade", url.hillshade);
  set("globe", url.globe);
  set("gaugeStyle", url.gaugeStyle);
  set("heat", url.heat);
  set("status", url.status);
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
  initTimeBar();
  initStatusLayer(url);   // the world river status (#544), before the layers are applied
  applyLayerState();
  syncRailControls();
  if (state.basinsOn || url.basins) setBasinsVisible(true);
  if (url.rivers === false) state.riversOn = false;   // on by default (#545); a link can say rivers=0
  if (state.riversOn) { setRiversVisible(true); renderCredits(); }
  initFloodsAhead();
  initFloodDepth();
  initFloodsPast();
  initMapActions();   // after every layer it can switch (#561)
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
  clearRiver();  // the trace to the sea belongs to the panel being closed
  showSurface("panel-empty");
  writeUrl({ push: true });
}

(async function boot() {
  trace("boot");
  const url = readUrl();

  initShell();
  initMetrics();
  initTabs($("panel-station"));
  initTabs($("panel-point"));
  initTabs($("panel-workbench"));
  initStationPanel();
  initPointPanel();
  initRiver();
  initNow();
  initBulletin();
  initWorkbench();
  initPlaces();  // My places + Compare
  initWatch();   // Watch (#521)
  initMapCard(); // after Watch: the card mirrors its buttons
  initAsk();   // async: fills the provider list from providers.json
  initStudyLoader();
  initSearch();
  initMapCommand();   // Ask the map (#561): the box and the / key
  void loadAvailability();
  initUrl();
  actions.applyUrl = applyUrl;
  actions.refreshMapData = () => { refreshMapData(); syncRail(); };
  $("btn-home").addEventListener("click", goHome);
  $("btn-cite-top").addEventListener("click", () => openCite([]));
  $("btn-open-study").addEventListener("click", () => $("open-study-file").click());
  $("open-study-file").addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (file) await loadStudy().then((module) => module.importCompletedStudy(file));
    event.target.value = "";
  });
  for (const chip of document.querySelectorAll("[data-try]")) {
    chip.addEventListener("click", () => {
      if (chip.dataset.try === "s") selectStation(chip.dataset.key, { fly: true });
      else if (chip.dataset.try === "p") selectPoint(Number(chip.dataset.lat), Number(chip.dataset.lon), { fly: true });
      else if (chip.dataset.try === "study") actions.openStudy({ recorded: "reference-fish-river-us" });
      else if (chip.dataset.try === "table") openSampleTable();
      else actions.openAsk();
    });
  }
  document.querySelector(".header-tools-menu").addEventListener("click", (event) => {
    if (event.target.closest("button")) document.querySelector(".header-tools").open = false;
  });

  if (url.hidden) state.hidden = new Set(url.hidden);
  // The map date and how it moves (#522). Set directly here, before anything
  // subscribes; every later change goes through setTime().
  state.date = url.date || defaultDate();
  state.timeStep = url.step || "day";
  state.timeRange = url.range || null;
  state.compare = url.compare || null;
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
  initSignatureFilter();  // async: shows the rail's signature filter when signatures.parquet exists
  if (mapOk) bringMapOnline(url);
  else if (mapResult && mapResult.reason === "slow") whenMapLoadsLate(() => bringMapOnline(url));
  ensureWorker();  // warm Python in the background so the first click is quicker

  applyUrl(url);
  greetOnLoad({ ...url, study: url.study || Boolean(studyUrlParam(location.search)) });  // Watch (#521): "Since you were here", when something is watched and the link opens nothing else
  const studyUrl = studyUrlParam(location.search);   // ?study_url=<https study.yaml> (study-link.js)
  if (studyUrl && !url.studyLink) actions.openSharedStudy({ studyUrl });
  // Offer the page's tools to an in-browser agent, where the browser has WebMCP.
  registerWebMcpTools({ actions });
  state.booting = false;
})();
