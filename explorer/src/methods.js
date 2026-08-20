// Methods, citations and how to cite AquaScope itself. The method blocks that
// travel with every result are collected here; the page also finally tells
// people how to cite the tool they just used (the repo has had a CITATION.cff
// and a DOI all along, and the Explorer never showed either).

import { $, copyText, escapeHtml } from "./core.js?v=__BUILD__";
import { openModal } from "./shell.js?v=__BUILD__";

export const NLDI_METHOD = {
  name: "Upstream catchment (USGS NLDI)",
  text: "Drainage basin upstream of the gauge (or of the NHDPlus V2 flowline nearest to a clicked point), traced by the USGS Network Linked Data Index over NHDPlus V2 catchments; simplified geometry, area computed on the sphere. US only.",
  citation: "U.S. Geological Survey, Network Linked Data Index (NLDI) API, https://api.water.usgs.gov/nldi/; NHDPlus Version 2 (U.S. EPA and USGS). Public domain.",
};

export const BASIN_METHOD = {
  name: "Catchment attributes from BasinATLAS (HydroATLAS v1.0)",
  text: "Level-12 HydroBASINS sub-basin containing the point, its upstream sub-basins traced through NEXT_DOWN, and BasinATLAS's upstream-aggregated attributes (climate, land cover, soils, population, regulation) read for the outlet sub-basin.",
  citation: "Linke, S., Lehner, B., Ouellet Dallaire, C., et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution. Scientific Data 6: 283. https://doi.org/10.1038/s41597-019-0300-6. CC BY 4.0.",
};

export const SIMILAR_METHOD = {
  name: "Similar gauged basins (physical similarity / spatial proximity)",
  text: "Gauged stations ranked by weighted Euclidean distance in standardised BasinATLAS catchment attribute space (area, relief, climate, land cover, soils, human pressure) combined with great-circle distance (in units of 500 km); the donor-selection step of regionalisation.",
  citation: "Blöschl, G., Sivapalan, M., Wagener, T., Viglione, A., Savenije, H. (eds.) (2013). Runoff Prediction in Ungauged Basins. Cambridge University Press; Oudin, L. et al. (2008). Spatial proximity, physical similarity, regression and ungaged catchments. Water Resour. Res. 44, W03413.",
};

export const REGIME_METHOD = {
  name: "Regionalisation of flow signatures from similar gauged basins",
  text: "Flow signatures of the gauged donors (mean, median, Q95 and Q05 daily flow in mm/d, mean annual maximum, baseflow index, seasonality, flashiness) transferred as an inverse-distance-weighted average over the 10 most similar catchments in standardised BasinATLAS attribute space (geometric mean for magnitudes); the band is one weighted standard deviation of the donors; the skill is leave-one-out over all donors.",
  citation: "Blöschl, G. et al. (eds.) (2013). Runoff Prediction in Ungauged Basins. Cambridge University Press; Oudin, L. et al. (2008). Water Resour. Res. 44, W03413; Addor, N. et al. (2018). A ranking of hydrological signatures based on their predictability in space. Water Resour. Res. 54, 8792-8812.",
};

export const GR4J_METHODS = [
  {
    name: "GR4J rainfall-runoff model, calibrated by differential evolution",
    text: "Four-parameter daily lumped model (production store X1, groundwater exchange X2, routing store X3, unit-hydrograph time base X4) run on ERA5-Land/ERA5 rainfall and FAO-56 ET0 at the gauge; parameters found by differential evolution (best/1/bin, population 20, 40 generations) maximising KGE on the first 65 % of the record after a one-year warm-up; the last 35 % is validation only.",
    citation: "Perrin, C., Michel, C., Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. J. Hydrol. 279, 275-289; Storn, R., Price, K. (1997). Differential evolution. J. Global Optim. 11, 341-359; Gupta, H. V. et al. (2009). Decomposition of the mean squared error and NSE performance criteria. J. Hydrol. 377, 80-91.",
  },
  {
    name: "Forcing at the gauge point",
    text: "Daily precipitation and FAO-56 Penman-Monteith reference ET0 from Open-Meteo's historical weather API (ERA5-Land where available, ERA5 elsewhere), at the gauge coordinates, not catchment-averaged.",
    citation: "Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth Syst. Sci. Data 13, 4349-4383; Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc. 146, 1999-2049; Open-Meteo (CC BY 4.0).",
  },
];

export function methodItemHtml(m) {
  return `<strong>${escapeHtml(m.name)}.</strong> ${escapeHtml(m.text)}<br><span class="cite">${escapeHtml(m.citation)}</span>`;
}

export function renderMethodList(listId, methods) {
  const ol = $(listId);
  if (!ol) return;
  ol.innerHTML = (methods || []).map((m) => `<li>${methodItemHtml(m)}</li>`).join("");
}

export function addMethodOnce(listId, m) {
  const ol = $(listId);
  if (!ol || [...ol.querySelectorAll("li strong")].some((el) => el.textContent.startsWith(m.name))) return;
  const li = document.createElement("li");
  li.innerHTML = methodItemHtml(m);
  ol.appendChild(li);
}

// The methods currently listed, as plain sentences. The citation lives in its
// own span, so pull it out first: textContent alone runs the two together
// ("read from the curve.Vogel, R. M., ...").
export function methodsOnPage(listId) {
  const ol = $(listId);
  if (!ol) return [];
  return [...ol.querySelectorAll("li")].map((li) => {
    const clone = li.cloneNode(true);
    const cite = clone.querySelector(".cite");
    const citation = cite ? cite.textContent.trim() : "";
    if (cite) cite.remove();
    const body = clone.textContent.replace(/\s+/g, " ").trim();
    return citation ? `${body} ${citation}` : body;
  });
}

// ── how to cite AquaScope ───────────────────────────────────────────────────

export const AQUASCOPE_DOI = "10.5281/zenodo.21903143";     // concept DOI, all versions
export const ARCHIVE_URL = "https://huggingface.co/datasets/Rekin226/aquascope-gauges";

export const BIBTEX = `@software{aquascope,
  author  = {Ouedraogo, Rachid and the AquaScope contributors},
  title   = {AquaScope: the open record of the world's public water gauges,
             and the tools to analyse them},
  url     = {https://github.com/Rekin226/aquascope},
  doi     = {${AQUASCOPE_DOI}},
  note    = {Concept DOI, resolves to the latest release}
}`;

export function openCite(extraMethods = []) {
  const list = extraMethods.length
    ? `<h3>Methods used on this page</h3><ol class="cite-methods">${extraMethods.map((t) => `<li>${escapeHtml(t)}</li>`).join("")}</ol>`
    : "";
  openModal("Cite this", `
    <p>Cite the software with its concept DOI (it resolves to the version you used), and the agency whose
    observations you looked at. The data licence is on every result, in Methods and citations.</p>
    <h3>Software</h3>
    <pre id="cite-bibtex">${escapeHtml(BIBTEX)}</pre>
    <p><button class="btn" id="cite-copy">Copy BibTeX</button>
       <a class="btn" href="https://doi.org/${AQUASCOPE_DOI}" target="_blank" rel="noopener">DOI ↗</a>
       <a class="btn" href="https://github.com/Rekin226/aquascope/blob/main/CITATION.cff" target="_blank" rel="noopener">CITATION.cff ↗</a></p>
    <h3>The gauge archive</h3>
    <p>The catalog and the daily observations behind this page are published as GeoParquet at
      <a href="${ARCHIVE_URL}" target="_blank" rel="noopener">Rekin226/aquascope-gauges</a>, rebuilt weekly.
      Each source keeps its own licence and attribution.</p>
    ${list}
  `);
  const b = $("cite-copy");
  if (b) b.addEventListener("click", () => copyText(BIBTEX, b, "BibTeX copied"));
}
