# Features

Complete capability reference for AquaScope. For installation and a quick example, see the [Getting started](getting_started.md) guide. For data-source details, see [Data sources](data_sources.md).

---

## Data Collection (15 sources)

- **Taiwan** — MOENV water quality, WRA levels/reservoirs, Civil IoT sensors
- **USA** — USGS streamflow, Water Quality Portal (400+ agencies)
- **Global** — GEMStat (170 countries), UN SDG 6, OpenMeteo weather, Copernicus climate
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
- **Crop water requirements** — 20 crops with FAO-56 Kc coefficients and growth stages
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

---

## AI Engine & Workflows

- **26 research methodologies** — scored and ranked against dataset profiles
- **7 auto-executable pipelines** — trend analysis, WQI, PCA, RF, XGBoost, ARIMA, correlation
- **Challenge workflows** — flood risk (GEV), drought severity (SPI), water quality (WHO)
- **Natural-language agent** — describe your goal, get recommendations + execution

### Built-in Research Methodologies (26)

| Category | Methodologies | Pipelines |
| :--- | :--- | :--- |
| Statistical | Mann-Kendall Trend, WQI/RPI, PCA + Clustering, Correlation, Bayesian Inference, Copula Dependence | 4 |
| Machine Learning | LSTM, Random Forest, XGBoost, Transformer, Autoencoder Anomaly Detection | 2 |
| Time-Series | ARIMA/SARIMA Forecasting | 1 |
| Process Engineering | MBBR Pilot, MBR Fouling, A2O Nutrient Removal, SWMM, QUAL2K | — |
| Spatial Analysis | Satellite Eutrophication, GIS Watershed, Kriging Interpolation | — |
| Hydrological | SWAT Modelling, Isotope Hydrology, Paired Watershed Design | — |
| Policy | SDG 6 Benchmarking, IWRM Assessment | — |

For when-to-use-which guidance, see the [methodology matrix](methodology_matrix.md).

---

## Visualization & Reporting

- **16 plot functions** — time-series, box plots, heatmaps, spatial maps (Folium), FDC, hydrographs
- **Diagnostic plots** — Q-Q, P-P, return level, 4-panel diagnostic panel
- **Automated reports** — Markdown & HTML with embedded plots, metrics, TOC
- **Alerts** — WHO, US EPA, EU WFD threshold checking

---

## Infrastructure

- **534 tests** with CAMELS benchmark validation
- **Interactive dashboard** — 7-page Streamlit app
- **14 CLI commands** — `collect`, `recommend`, `eda`, `quality`, `run`, `solve`, `forecast`, `plot`, `hydro`, `alerts`, `dashboard`, `agri`, `list-methods`, `list-sources`
- **[Theory guide](theory.md)** — mathematical equations, DOI citations, decision trees
