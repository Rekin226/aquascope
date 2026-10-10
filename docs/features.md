# Features

Complete capability reference for AquaScope. For installation and a quick example, see the [Getting started](getting_started.md) guide. For data-source details, see [Data sources](data_sources.md).

---

## Data Collection (37 sources)

- **Taiwan** — MOENV water quality, WRA levels / reservoirs / groundwater (annual and daily) / FHY / IoT, CWA climate, Civil IoT sensors, data.gov.tw
- **Americas** — USGS streamflow, NOAA NWPS, Colorado DWR/CDSS, Water Quality Portal (400+ agencies), CAMELS-CL, CAMELS-BR, ANA Hidroweb
- **Europe** — EU Water Framework Directive, France Hub'Eau, Germany PEGELONLINE, England's Environment Agency, Ireland OPW, Greece Hydroscope
- **Asia-Pacific** — Japan MLIT, Korea WAMIS, India WRIS, Australia BOM
- **Global** — GEMStat (170 countries), GRDC river discharge, UN SDG 6, OpenMeteo weather, Copernicus climate
- **FAO** — AQUASTAT country-level water use, WaPOR satellite evapotranspiration

See [docs/data_sources.md](data_sources.md) for the full list with endpoints and API-key requirements.

---

## Hydrological Analysis

- **Flood frequency** — GEV, LP3 (Bulletin 17C compliant), Gumbel, GPD/POT, L-moments, non-stationary GEV, regional frequency analysis, EMA for censored data
- **Baseflow separation** — Lyne-Hollick & Eckhardt digital filters
- **Flow duration curves** — Weibull plotting, FDC slope
- **22 hydrological signatures** — magnitude, variability, timing, recession, flashiness
- **Rating curves** — power-law fitting, segmented curves, shift detection, HEC-RAS export
- **Q-Q/P-P diagnostics** — distribution fit validation with 4-panel diagnostic plots
- **Cross-validation** — leave-one-out CV and coverage probability for flood frequency

---

## Agricultural Water Management

- **FAO-56 Penman-Monteith ET₀** — reference evapotranspiration with all intermediate steps
- **Hargreaves ET₀** — temperature-only alternative
- **Crop water requirements** — 26 crops with FAO-56 Kc coefficients and growth stages; single (Kc) and dual (Kcb + Ke) coefficient modes; olive, grape, citrus and winter wheat resolve to 32 FAO-56 Rev.1 sub-classes by variety, planting density and ground cover
- **Irrigation scheduling** — effective rainfall, net/gross demand, efficiency
- **Soil water balance** — daily tracking, depletion, auto-irrigation triggers
- **WaPOR productivity workflows** — biomass water productivity and AETI-to-RET performance metrics

---

## Statistical & ML Methods

- **Copula analysis** — Gaussian, Clayton, Gumbel, Frank with AIC selection
- **Change-point detection** — PELT, CUSUM, Pettitt test, binary segmentation
- **Bayesian UQ** — conjugate linear regression, Metropolis-Hastings MCMC, Gelman-Rubin R̂
- **Model ensembles** — weighted, stacking, adaptive strategies
- **Transfer learning** — donor selection via signature similarity for ungauged basins
- **Predictive models** — Prophet, ARIMA, SPI, Random Forest, XGBoost, Isolation Forest, LSTM

---

## Spatial & I/O

- **Spatial hydrology** — DEM processing, D8 flow direction, watershed delineation, Strahler ordering
- **Scientific I/O** — WaterML 2.0, HEC-DSS/RAS, EPA SWMM, NetCDF, HDF5, GeoJSON
- **Place context** (`aquascope.context`, `aquascope context LAT LON`, MCP `place_context`): flood events in the news (Groundsource) and Sentinel-1 radar floods 2014-2024, JRC Global Surface Water since 1984, JRC GloFAS flood depth at return periods, Global Dam Watch dams, SoilGrids texture and available water, FAO WaPOR actual ET and the nearest NOAA GHCN-Daily rain gauge, at a point or over a box, each with its licence
- **Cloud-Optimized GeoTIFF point reads** (`aquascope.utils.cog`): pure Python (TIFF and BigTIFF, Deflate, LZW, predictors, overviews) over HTTP range requests, so it runs in the browser too
- **Engineering exports**: any record as inputs for HEC-HMS, HEC-RAS, HEC-SSP (with a Bulletin 17C check against the published examples), HEC-DSS, SWMM, MODFLOW 6, Delft-FEWS and Raven ([details](engineering_exports.md))

---

## AI Engine & Workflows

- **27 research methodologies** — scored and ranked against dataset profiles
- **26 auto-executable pipelines** — including trend analysis, WQI, PCA and ARIMA
- **Challenge workflows** — flood risk (GEV), drought severity (SPI), water quality (WHO)
- **Natural-language agent** — describe your goal, get recommendations + execution

### Built-in Research Methodologies (27)

| Category | Methodologies | Pipelines |
| :--- | :--- | :--- |
| Statistical | Mann-Kendall Trend, WQI/RPI, PCA + Clustering, Correlation, Bayesian Inference, Copula Dependence | 4 |
| Machine Learning | LSTM, Random Forest, XGBoost, Transformer, Autoencoder Anomaly Detection | 2 |
| Time-Series | ARIMA/SARIMA Forecasting | 1 |
| Process Engineering | MBBR Pilot, MBR Fouling, A2O Nutrient Removal, SWMM, QUAL2K | — |
| Spatial Analysis | Satellite Eutrophication, GIS Watershed, Kriging Interpolation | — |
| Hydrological | SWAT Modelling, Isotope Hydrology, Paired Watershed Design, Budyko Framework | — |
| Policy | SDG 6 Benchmarking, IWRM Assessment | — |

For when-to-use-which guidance, see the [methodology matrix](methodology_matrix.md).

---

## Visualization & Reporting

- **17 plot functions** — time-series, box plots, heatmaps, spatial maps (Folium), FDC, hydrographs
- **Diagnostic plots** — Q-Q, P-P, return level, 4-panel diagnostic panel
- **Automated reports** — Markdown & HTML with embedded plots, metrics, TOC
- **Alerts** — WHO, US EPA, EU WFD threshold checking

---

## Infrastructure

- **Regression tests and numerical benchmarks** — see the [observed/synthetic validation scope](validation_scope.md)
- **Interactive dashboard** — 10-page Streamlit app
- **Rivers as objects**: `aquascope.rivers` snap any point to its GEOGLOWS v2 river reach (or say no stream is near), the reach's simulated daily flow since 1940 with return periods and the flow-duration curve (modelled, CC BY 4.0), its upstream area, the trace to the sea with the gauges and Global Dam Watch dams on the way and the countries it crosses, and the dams upstream of a reach with the degree of regulation (`aquascope.river_path`). The Explorer's River tab, `aquascope river`, MCP and the Studio share it.
- **The evidence ladder**: `aquascope.evidence` scores GEOGLOWS v2, GloFAS, NWM v3 (US) and Google's GRRR reanalysis against a gauge's own record (KGE and its parts, NSE, bias, the error at the 2-, 10- and 100-year flows), grades each A to D and says where they disagree; a monthly CI table in the Archive, the Explorer's Evidence tab and map colouring, `aquascope evidence` and MCP share it ([evidence.md](evidence.md)).
- **42 CLI commands**: `collect`, `recommend`, `eda`, `quality`, `run`, `completion`, `list-methods`, `list-sources`, `stations`, `harvest`, `ask`, `ingest`, `basins`, `river`, `evidence`, `now`, `bulletin`, `warnings`, `watch`, `assess`, `context`, `gym`, `caravan`, `mcp`, `playbooks`, `solve`, `studio`, `desk`, `studio-showcase`, `eval`, `forecast`, `plot`, `dashboard`, `agri`, `alerts`, `groundwater`, `climate`, `hydro`, `area-study`, `layers`, `export`, `update`
- **[Theory guide](theory.md)** — mathematical equations, DOI citations, decision trees
