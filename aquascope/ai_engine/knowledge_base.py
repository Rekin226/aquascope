"""
Built-in knowledge base of water research methodologies.

This serves as the foundation that the AI engine draws on to recommend
methodologies.  Researchers can extend this by contributing new entries
via pull request.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResearchMethodology:
    """Describes a single research methodology applicable to water studies."""

    id: str
    name: str
    category: str
    description: str
    applicable_parameters: list[str] = field(default_factory=list)
    data_requirements: list[str] = field(default_factory=list)
    typical_scale: str = ""           # lab / pilot / field / regional / global
    complexity: str = ""              # low / medium / high
    references: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ── Pre-populated methodology catalogue ──────────────────────────────

METHODOLOGIES: list[ResearchMethodology] = [
    # ── Statistical / data-driven ────────────────────────────────────
    ResearchMethodology(
        id="trend_analysis",
        name="Mann-Kendall Trend Analysis",
        category="statistical",
        description=(
            "Non-parametric test for detecting monotonic trends in time-series "
            "water quality data.  Often paired with Sen's slope estimator."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "pH", "Temperature", "SS"],
        data_requirements=["time-series ≥ 10 years", "monthly or quarterly sampling"],
        typical_scale="regional",
        complexity="low",
        references=[
            "Mann, H.B. (1945). Non-parametric tests against trend. Econometrica 13:245-259.",
            "Kendall, M.G. (1975). Rank Correlation Methods. Griffin, London.",
        ],
        tags=["trend", "time-series", "long-term", "non-parametric"],
    ),
    ResearchMethodology(
        id="wqi_calculation",
        name="Water Quality Index (WQI) Computation",
        category="statistical",
        description=(
            "Aggregate multiple parameters into a single dimensionless score. "
            "Common variants: CCME WQI, NSF WQI, Taiwan RPI."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "SS", "NH3-N", "pH", "Temperature", "TP"],
        data_requirements=["simultaneous multi-parameter measurements"],
        typical_scale="field",
        complexity="low",
        references=[
            "CCME (2001). Canadian Water Quality Guidelines. Canadian Council of Ministers of the Environment.",
        ],
        tags=["index", "aggregation", "monitoring", "assessment"],
    ),
    ResearchMethodology(
        id="pca_clustering",
        name="PCA + Cluster Analysis for Source Apportionment",
        category="statistical",
        description=(
            "Principal Component Analysis combined with hierarchical or k-means clustering "
            "to identify pollution sources and group similar monitoring stations."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "TP", "Conductivity", "SS", "pH"],
        data_requirements=["multi-site", "multi-parameter", "≥ 12 sampling events"],
        typical_scale="regional",
        complexity="medium",
        references=[
            "Singh, K.P. et al. (2004). Multivariate statistical techniques for evaluation of spatial and temporal variations. Water Research 38(18):3980-3992.",
        ],
        tags=["multivariate", "source apportionment", "spatial"],
    ),

    # ── Machine Learning ─────────────────────────────────────────────
    ResearchMethodology(
        id="lstm_forecasting",
        name="LSTM Neural Network for Water Quality Forecasting",
        category="machine_learning",
        description=(
            "Long Short-Term Memory recurrent neural network for predicting future "
            "water quality values from historical time-series data."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "pH", "Turbidity", "Flow"],
        data_requirements=["high-frequency time-series", "≥ 2 years", "sensor or daily data"],
        typical_scale="field",
        complexity="high",
        references=[
            "Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. Neural Computation 9(8):1735-1780.",
        ],
        tags=["deep-learning", "forecasting", "time-series", "prediction"],
    ),
    ResearchMethodology(
        id="random_forest_classification",
        name="Random Forest for Water Quality Classification",
        category="machine_learning",
        description=(
            "Ensemble learning method to classify water bodies into quality categories "
            "(e.g., RPI classes, WQI categories) or predict potability."
        ),
        applicable_parameters=["pH", "DO", "BOD5", "COD", "SS", "NH3-N", "Conductivity", "Turbidity"],
        data_requirements=["labelled dataset", "multi-parameter", "≥ 200 samples"],
        typical_scale="regional",
        complexity="medium",
        references=[
            "Breiman, L. (2001). Random Forests. Machine Learning 45:5-32.",
        ],
        tags=["classification", "ensemble", "potability", "supervised"],
    ),
    ResearchMethodology(
        id="xgboost_regression",
        name="XGBoost Regression for Parameter Prediction",
        category="machine_learning",
        description=(
            "Gradient-boosted decision trees for predicting a target water quality "
            "parameter from other measured variables and environmental covariates."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "Chlorophyll-a", "Turbidity"],
        data_requirements=["tabular multi-parameter dataset", "≥ 300 samples"],
        typical_scale="regional",
        complexity="medium",
        references=[
            "Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.",
        ],
        tags=["regression", "gradient-boosting", "prediction"],
    ),

    # ── Process-based / engineering ──────────────────────────────────
    ResearchMethodology(
        id="mbbr_pilot_study",
        name="MBBR Pilot Plant Performance Evaluation",
        category="process_engineering",
        description=(
            "Experimental methodology for evaluating Moving Bed Biofilm Reactor "
            "performance at pilot scale.  Assess BOD/COD/NH3-N removal under varying "
            "hydraulic and organic loading rates."
        ),
        applicable_parameters=["BOD5", "COD", "NH3-N", "SS", "DO", "Temperature"],
        data_requirements=["pilot influent/effluent sampling", "≥ 3 months steady-state"],
        typical_scale="pilot",
        complexity="high",
        references=[
            "Ødegaard, H. (2006). Innovations in wastewater treatment: the moving bed biofilm process. Water Science and Technology 53(9):17-33.",
        ],
        tags=["MBBR", "biofilm", "wastewater", "pilot", "treatment"],
    ),
    ResearchMethodology(
        id="mbr_optimisation",
        name="MBR Fouling and Optimisation Study",
        category="process_engineering",
        description=(
            "Methodology for investigating membrane fouling mechanisms in Membrane "
            "Bioreactors and optimising operating parameters (flux, SRT, MLSS)."
        ),
        applicable_parameters=["COD", "SS", "MLSS", "TMP", "Flux", "DO"],
        data_requirements=["pilot or full-scale MBR monitoring", "TMP and flux logging"],
        typical_scale="pilot",
        complexity="high",
        references=[
            "Judd, S. (2010). The MBR Book: Principles and Applications of MBR. Butterworth-Heinemann.",
        ],
        tags=["MBR", "membrane", "fouling", "wastewater", "optimisation"],
    ),
    ResearchMethodology(
        id="a2o_nutrient_removal",
        name="A2O Process Nutrient Removal Evaluation",
        category="process_engineering",
        description=(
            "Assess nitrogen and phosphorus removal performance in Anaerobic-Anoxic-Oxic "
            "(A2O) configurations.  Track N and P species through each zone."
        ),
        applicable_parameters=["NH3-N", "NO3-N", "TN", "TP", "PO4-P", "COD", "DO"],
        data_requirements=["zone-specific sampling", "influent/effluent", "SRT and HRT data"],
        typical_scale="pilot",
        complexity="high",
        references=[
            "Metcalf & Eddy (2014). Wastewater Engineering: Treatment and Resource Recovery. 5th ed.",
        ],
        tags=["A2O", "nutrient removal", "BNR", "wastewater"],
    ),

    # ── Remote sensing / GIS ─────────────────────────────────────────
    ResearchMethodology(
        id="satellite_eutrophication",
        name="Satellite-Based Eutrophication Assessment",
        category="remote_sensing",
        description=(
            "Use Sentinel-2 / Landsat imagery to estimate chlorophyll-a concentration "
            "and trophic state of lakes and reservoirs."
        ),
        applicable_parameters=["Chlorophyll-a", "Turbidity", "CDOM"],
        data_requirements=["satellite imagery", "in-situ validation points"],
        typical_scale="regional",
        complexity="medium",
        references=[
            "Pahlevan, N. et al. (2020). Seamless retrievals of chlorophyll-a from Sentinel-2 (MSI) and Sentinel-3 (OLCI). Remote Sensing of Environment 240:111604.",
        ],
        tags=["remote-sensing", "satellite", "eutrophication", "chlorophyll"],
    ),
    ResearchMethodology(
        id="gis_watershed_analysis",
        name="GIS-Based Watershed Land-Use Impact Analysis",
        category="remote_sensing",
        description=(
            "Relate land-use/land-cover changes to water quality deterioration using "
            "GIS overlay analysis, buffer delineation, and regression."
        ),
        applicable_parameters=["COD", "BOD5", "NH3-N", "TP", "SS"],
        data_requirements=["land-use raster", "water quality point data", "watershed boundaries"],
        typical_scale="regional",
        complexity="medium",
        references=[
            "Allan, J.D. (2004). Landscapes and Riverscapes. Annual Review of Ecology, Evolution, and Systematics 35:257-284.",
        ],
        tags=["GIS", "watershed", "land-use", "spatial"],
    ),

    # ── Hydrological modelling ───────────────────────────────────────
    ResearchMethodology(
        id="swat_modelling",
        name="SWAT Hydrological Modelling",
        category="hydrological_modelling",
        description=(
            "Soil and Water Assessment Tool for simulating watershed-scale hydrology, "
            "sediment transport, and nutrient loading under different scenarios."
        ),
        applicable_parameters=["Flow", "SS", "TN", "TP"],
        data_requirements=["DEM", "land-use", "soil", "climate data", "observed flow"],
        typical_scale="regional",
        complexity="high",
        references=[
            "Arnold, J.G. et al. (1998). Large area hydrologic modelling and assessment part I. JAWRA 34(1):73-89.",
        ],
        tags=["SWAT", "watershed", "hydrology", "modelling", "simulation"],
    ),

    # ── SDG / policy ─────────────────────────────────────────────────
    ResearchMethodology(
        id="sdg6_benchmarking",
        name="SDG 6 Cross-Country Benchmarking",
        category="policy_analysis",
        description=(
            "Compare SDG 6 indicators across countries to identify best practices, "
            "gaps, and policy recommendations for water and sanitation."
        ),
        applicable_parameters=["SDG 6.1.1", "SDG 6.2.1", "SDG 6.3.1", "SDG 6.4.2"],
        data_requirements=["SDG 6 indicator data", "≥ 10 countries", "multiple years"],
        typical_scale="global",
        complexity="low",
        references=[
            "UN-Water (2021). Summary Progress Update 2021: SDG 6 — Water and Sanitation for All.",
        ],
        tags=["SDG", "policy", "benchmarking", "global", "indicators"],
    ),

    # ── v0.2.0 additions ─────────────────────────────────────────────

    ResearchMethodology(
        id="arima_forecast",
        name="ARIMA / SARIMA Time-Series Forecasting",
        category="statistical",
        description=(
            "Auto-Regressive Integrated Moving Average model for forecasting "
            "seasonal and non-seasonal water quality time-series.  Suitable for "
            "monthly or quarterly data with clear temporal autocorrelation."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "pH", "Flow", "Water level"],
        data_requirements=["univariate time-series", "≥ 24 monthly observations", "stationary or differentiable"],
        typical_scale="field",
        complexity="medium",
        references=[
            "Box, G.E.P. & Jenkins, G.M. (1976). Time Series Analysis: Forecasting and Control. Holden-Day.",
        ],
        tags=["ARIMA", "SARIMA", "forecasting", "time-series", "statistical"],
    ),
    ResearchMethodology(
        id="transformer_forecast",
        name="Transformer-Based Time-Series Forecasting",
        category="machine_learning",
        description=(
            "Self-attention neural network architecture for capturing long-range "
            "dependencies in water quality time-series.  State-of-the-art performance "
            "on many forecasting benchmarks."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "pH", "Flow", "Turbidity"],
        data_requirements=["high-frequency time-series", "≥ 3 years", "multi-variate preferred"],
        typical_scale="regional",
        complexity="high",
        references=[
            "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.",
        ],
        tags=["transformer", "attention", "deep-learning", "forecasting", "time-series"],
    ),
    ResearchMethodology(
        id="bayesian_network",
        name="Bayesian Network for Causal Inference",
        category="machine_learning",
        description=(
            "Probabilistic graphical model to discover causal relationships "
            "between water quality parameters, land use, and pollution sources."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "TP", "Conductivity", "pH"],
        data_requirements=["multi-parameter dataset", "≥ 100 samples", "domain knowledge for priors"],
        typical_scale="regional",
        complexity="high",
        references=[
            "Pearl, J. (2009). Causality: Models, Reasoning, and Inference. 2nd ed. Cambridge University Press.",
        ],
        tags=["Bayesian", "causal", "probabilistic", "network", "inference"],
    ),
    ResearchMethodology(
        id="svr_prediction",
        name="Support Vector Regression (SVR)",
        category="machine_learning",
        description=(
            "Kernel-based regression for nonlinear water quality prediction.  "
            "Effective with small-to-medium datasets and high-dimensional inputs."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "Turbidity", "Chlorophyll-a"],
        data_requirements=["tabular multi-parameter dataset", "≥ 100 samples"],
        typical_scale="field",
        complexity="medium",
        references=[
            "Drucker, H. et al. (1997). Support Vector Regression Machines. NeurIPS 1997.",
        ],
        tags=["SVR", "SVM", "regression", "kernel", "nonlinear"],
    ),
    ResearchMethodology(
        id="kriging_interpolation",
        name="Kriging / Geostatistical Interpolation",
        category="remote_sensing",
        description=(
            "Spatial interpolation using variogram modelling to estimate water "
            "quality at unsampled locations.  Produces prediction maps with "
            "uncertainty estimates."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "Conductivity", "pH", "Hardness"],
        data_requirements=["georeferenced point data", "≥ 30 stations", "spatial coordinates"],
        typical_scale="regional",
        complexity="medium",
        references=[
            "Cressie, N. (1993). Statistics for Spatial Data. Wiley.",
        ],
        tags=["kriging", "geostatistics", "spatial", "interpolation", "mapping"],
    ),
    ResearchMethodology(
        id="hec_ras_modelling",
        name="HEC-RAS Hydraulic Modelling",
        category="hydrological_modelling",
        description=(
            "1D/2D hydraulic modelling for flood inundation mapping, river "
            "profile analysis, and dam-break simulation using the US Army "
            "Corps of Engineers HEC-RAS software."
        ),
        applicable_parameters=["Flow", "Water level", "Channel geometry"],
        data_requirements=["river cross-sections", "flow data", "DEM", "boundary conditions"],
        typical_scale="regional",
        complexity="high",
        references=[
            "Brunner, G.W. (2020). HEC-RAS River Analysis System: Hydraulic Reference Manual. US Army Corps of Engineers.",
        ],
        tags=["HEC-RAS", "hydraulic", "flood", "modelling", "1D", "2D"],
    ),
    ResearchMethodology(
        id="qual2k_modelling",
        name="QUAL2K River Water Quality Modelling",
        category="process_engineering",
        description=(
            "Steady-state, one-dimensional river water quality model for simulating "
            "dissolved oxygen, BOD, nutrients, and temperature along a river reach."
        ),
        applicable_parameters=["DO", "BOD5", "NH3-N", "NO3-N", "TP", "Temperature"],
        data_requirements=["river geometry", "flow data", "point source inputs", "upstream boundary conditions"],
        typical_scale="field",
        complexity="high",
        references=[
            "Chapra, S.C. et al. (2008). QUAL2K: A Modeling Framework for Simulating River and Stream Water Quality. EPA.",
        ],
        tags=["QUAL2K", "river", "water-quality", "modelling", "DO", "steady-state"],
    ),
    ResearchMethodology(
        id="monte_carlo_uncertainty",
        name="Monte Carlo Uncertainty Analysis",
        category="statistical",
        description=(
            "Propagate parameter uncertainty through water quality models using "
            "random sampling.  Quantifies confidence intervals on predictions "
            "and identifies most sensitive parameters."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "Flow", "Temperature"],
        data_requirements=["parameter distributions", "model equations", "≥ 1000 simulations"],
        typical_scale="field",
        complexity="medium",
        references=[
            "Metropolis, N. & Ulam, S. (1949). The Monte Carlo Method. JASA 44:335-341.",
        ],
        tags=["Monte-Carlo", "uncertainty", "sensitivity", "simulation", "stochastic"],
    ),
    ResearchMethodology(
        id="wavelet_analysis",
        name="Wavelet Transform Analysis",
        category="statistical",
        description=(
            "Multi-resolution time-frequency analysis to detect periodic patterns, "
            "regime shifts, and scale-dependent relationships in hydrological "
            "and water quality time-series."
        ),
        applicable_parameters=["DO", "Flow", "Water level", "Precipitation", "Temperature"],
        data_requirements=["long time-series", "≥ 5 years", "regular interval"],
        typical_scale="regional",
        complexity="medium",
        references=[
            "Torrence, C. & Compo, G.P. (1998). A Practical Guide to Wavelet Analysis. BAMS 79(1):61-78.",
        ],
        tags=["wavelet", "time-frequency", "periodicity", "multi-scale", "spectral"],
    ),
    ResearchMethodology(
        id="copula_analysis",
        name="Copula-Based Dependence Modelling",
        category="statistical",
        description=(
            "Model multivariate dependence structure between water quality "
            "parameters or hydrological variables using copula functions, "
            "capturing tail dependencies that correlation misses."
        ),
        applicable_parameters=["Flow", "Water level", "Precipitation", "DO", "COD"],
        data_requirements=["paired observations", "≥ 50 samples", "marginal distributions"],
        typical_scale="regional",
        complexity="high",
        references=[
            "Nelsen, R.B. (2006). An Introduction to Copulas. 2nd ed. Springer.",
        ],
        tags=["copula", "dependence", "multivariate", "joint-distribution", "extremes"],
    ),
    ResearchMethodology(
        id="transfer_learning_wq",
        name="Transfer Learning for Water Quality",
        category="machine_learning",
        description=(
            "Fine-tune pre-trained deep learning models on local water quality data, "
            "leveraging knowledge from data-rich regions to improve predictions in "
            "data-scarce environments."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "pH", "Turbidity"],
        data_requirements=["source domain dataset (large)", "target domain dataset (small, ≥ 50)", "similar parameters"],
        typical_scale="regional",
        complexity="high",
        references=[
            "Pan, S.J. & Yang, Q. (2010). A Survey on Transfer Learning. IEEE TKDE 22(10):1345-1359.",
        ],
        tags=["transfer-learning", "deep-learning", "data-scarce", "fine-tuning"],
    ),
    ResearchMethodology(
        id="constructed_wetland_design",
        name="Constructed Wetland Design Optimisation",
        category="process_engineering",
        description=(
            "Design and optimise nature-based treatment systems using constructed "
            "wetlands for decentralised wastewater treatment.  Evaluate hydraulic "
            "loading, vegetation selection, and pollutant removal efficiency."
        ),
        applicable_parameters=["BOD5", "COD", "NH3-N", "TN", "TP", "SS", "E.coli"],
        data_requirements=["influent/effluent quality", "hydraulic loading rates", "climate data"],
        typical_scale="pilot",
        complexity="medium",
        references=[
            "Kadlec, R.H. & Wallace, S. (2009). Treatment Wetlands. 2nd ed. CRC Press.",
        ],
        tags=["wetland", "nature-based", "wastewater", "treatment", "NBS"],
    ),
    ResearchMethodology(
        id="correlation_analysis",
        name="Pearson / Spearman Correlation Analysis",
        category="statistical",
        description=(
            "Quantify linear (Pearson) or monotonic (Spearman) relationships between "
            "water quality parameters.  Foundation for identifying co-varying pollutants "
            "and potential cause-effect links."
        ),
        applicable_parameters=["DO", "BOD5", "COD", "NH3-N", "SS", "pH", "Conductivity", "Temperature", "TP"],
        data_requirements=["paired measurements", "≥ 30 samples", "multi-parameter"],
        typical_scale="field",
        complexity="low",
        references=[
            "Helsel, D.R. & Hirsch, R.M. (2002). Statistical Methods in Water Resources. USGS TWRI Book 4.",
        ],
        tags=["correlation", "Pearson", "Spearman", "bivariate", "association"],
    ),
]


def get_all_methodologies() -> list[ResearchMethodology]:
    """Return the full methodology catalogue."""
    return list(METHODOLOGIES)


def get_methodology(method_id: str) -> ResearchMethodology | None:
    """Look up a methodology by ID."""
    return next((m for m in METHODOLOGIES if m.id == method_id), None)


def search_methodologies(
    parameters: list[str] | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    scale: str | None = None,
) -> list[ResearchMethodology]:
    """Filter the catalogue by parameters, category, tags, or scale."""
    results = METHODOLOGIES
    if category:
        results = [m for m in results if m.category == category]
    if scale:
        results = [m for m in results if m.typical_scale == scale]
    if parameters:
        param_set = {p.lower() for p in parameters}
        results = [
            m
            for m in results
            if param_set & {p.lower() for p in m.applicable_parameters}
        ]
    if tags:
        tag_set = {t.lower() for t in tags}
        results = [
            m for m in results if tag_set & {t.lower() for t in m.tags}
        ]
    return results
