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
]


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
