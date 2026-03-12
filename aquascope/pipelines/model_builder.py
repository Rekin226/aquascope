"""
Model implementation pipeline.

After the AI recommender suggests methodologies and the user approves one,
this module auto-builds and runs the corresponding analysis, producing
results, metrics, and visualisation figures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_DIR = Path("data/figures")


@dataclass
class PipelineResult:
    """Structured output from a methodology pipeline."""

    method_id: str
    method_name: str
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: list[Path] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _ensure_figure_dir() -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_DIR


def _pivot_params(df: pd.DataFrame, dt_col: str = "sample_datetime") -> pd.DataFrame:
    """Pivot long-form data to wide: datetime index × parameter columns."""
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    pivot = df.pivot_table(index=dt_col, columns="parameter", values="value", aggfunc="mean")
    return pivot.sort_index().dropna(how="all")


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline implementations
# ═══════════════════════════════════════════════════════════════════════


def run_mann_kendall(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Mann-Kendall trend test for each parameter at each station."""
    try:
        import pymannkendall as mk
    except ImportError:
        raise ImportError("pip install pymannkendall  (required for trend analysis)")

    config = config or {}
    alpha = config.get("alpha", 0.05)
    target_params = config.get("parameters", df["parameter"].unique().tolist())

    results_rows: list[dict] = []
    for station in df["station_id"].unique():
        for param in target_params:
            subset = df[(df["station_id"] == station) & (df["parameter"] == param)]
            vals = pd.to_numeric(subset["value"], errors="coerce").dropna()
            if len(vals) < 10:
                continue
            try:
                res = mk.original_test(vals.values, alpha=alpha)
                results_rows.append({
                    "station": station,
                    "parameter": param,
                    "trend": res.trend,
                    "p_value": round(res.p, 6),
                    "z_score": round(res.z, 4),
                    "tau": round(res.Tau, 4),
                    "slope": round(res.slope, 6),
                    "significant": res.p < alpha,
                })
            except Exception as e:
                logger.debug("MK test failed for %s/%s: %s", station, param, e)

    summary_df = pd.DataFrame(results_rows)
    n_sig = summary_df["significant"].sum() if len(summary_df) > 0 else 0

    return PipelineResult(
        method_id="trend_analysis",
        method_name="Mann-Kendall Trend Analysis",
        summary=f"Tested {len(results_rows)} station-parameter combinations. {n_sig} show significant trends (α={alpha}).",
        metrics={"n_tests": len(results_rows), "n_significant": int(n_sig), "alpha": alpha},
        details={"results": summary_df.to_dict("records") if len(summary_df) > 0 else []},
    )


def run_wqi(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """
    Compute Taiwan River Pollution Index (RPI) for each station/date.

    RPI uses: DO, BOD5, SS, NH3-N with point assignments 1/3/6/10.
    """
    config = config or {}

    rpi_thresholds = {
        "DO": [(6.5, 1), (4.6, 3), (2.0, 6), (0, 10)],
        "BOD5": [(3.0, 1), (4.9, 3), (15.0, 6), (999, 10)],
        "SS": [(20.0, 1), (49.9, 3), (100.0, 6), (9999, 10)],
        "NH3-N": [(0.50, 1), (0.99, 3), (3.00, 6), (999, 10)],
    }

    def _score_param(param: str, value: float) -> int | None:
        thresholds = rpi_thresholds.get(param)
        if thresholds is None:
            return None
        if param == "DO":
            for threshold, score in thresholds:
                if value >= threshold:
                    return score
            return 10
        else:
            for threshold, score in thresholds:
                if value <= threshold:
                    return score
            return 10

    results_rows: list[dict] = []
    required = {"DO", "BOD5", "SS", "NH3-N"}

    for station in df["station_id"].unique():
        station_data = df[df["station_id"] == station]
        dt_col = "sample_datetime" if "sample_datetime" in df.columns else "reading_datetime"
        for date, date_group in station_data.groupby(pd.to_datetime(station_data[dt_col]).dt.date):
            param_vals = {}
            for _, row in date_group.iterrows():
                p = row.get("parameter", "")
                v = row.get("value")
                if p in required and v is not None:
                    try:
                        param_vals[p] = float(v)
                    except (ValueError, TypeError):
                        pass

            if len(param_vals) < 3:  # need at least 3 of 4
                continue

            scores = {}
            for p, v in param_vals.items():
                s = _score_param(p, v)
                if s is not None:
                    scores[p] = s

            if scores:
                rpi = sum(scores.values()) / len(scores)
                if rpi <= 2.0:
                    category = "Non-polluted"
                elif rpi <= 3.0:
                    category = "Lightly polluted"
                elif rpi <= 6.0:
                    category = "Moderately polluted"
                else:
                    category = "Severely polluted"

                results_rows.append({
                    "station": station,
                    "date": str(date),
                    "rpi": round(rpi, 2),
                    "category": category,
                    **{f"score_{k}": v for k, v in scores.items()},
                })

    summary_df = pd.DataFrame(results_rows)
    category_counts = summary_df["category"].value_counts().to_dict() if len(summary_df) > 0 else {}

    return PipelineResult(
        method_id="wqi_calculation",
        method_name="Taiwan River Pollution Index (RPI)",
        summary=f"Computed RPI for {len(results_rows)} station-date records. Distribution: {category_counts}",
        metrics={"n_computed": len(results_rows), "category_distribution": category_counts},
        details={"results": summary_df.to_dict("records") if len(summary_df) > 0 else []},
    )


def run_pca_clustering(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """PCA + K-Means clustering for pollution source apportionment."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    config = config or {}
    n_clusters = config.get("n_clusters", 3)
    n_components = config.get("n_components", 2)

    pivot = _pivot_params(df)
    pivot = pivot.dropna()

    if len(pivot) < 10 or pivot.shape[1] < 2:
        return PipelineResult(
            method_id="pca_clustering",
            method_name="PCA + Cluster Analysis",
            summary="Insufficient data for PCA (need ≥ 10 samples with ≥ 2 parameters).",
            metrics={},
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pivot)

    n_comp = min(n_components, scaled.shape[1])
    pca = PCA(n_components=n_comp)
    pca_result = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=min(n_clusters, len(pivot)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_result)

    return PipelineResult(
        method_id="pca_clustering",
        method_name="PCA + Cluster Analysis",
        summary=(
            f"PCA with {n_comp} components explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance. "
            f"K-Means found {len(set(labels))} clusters across {len(pivot)} samples."
        ),
        metrics={
            "explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
            "n_clusters": len(set(labels)),
            "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
            "loadings": {
                col: [round(float(pca.components_[i, j]), 4) for i in range(n_comp)]
                for j, col in enumerate(pivot.columns)
            },
        },
        details={"labels": labels.tolist(), "parameters": pivot.columns.tolist()},
    )


def run_random_forest(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Random Forest classification of water quality categories."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    config = config or {}
    target = config.get("target", "category")

    pivot = _pivot_params(df).dropna()
    if len(pivot) < 20:
        return PipelineResult(
            method_id="random_forest_classification",
            method_name="Random Forest Classification",
            summary="Insufficient data (need ≥ 20 complete samples).",
            metrics={},
        )

    # Create simple quality labels based on first parameter quartiles if no target provided
    if target not in pivot.columns:
        ref_col = pivot.columns[0]
        pivot["category"] = pd.qcut(pivot[ref_col], q=3, labels=["Good", "Fair", "Poor"])
        target = "category"

    X = pivot.drop(columns=[target])
    y = pivot[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    importances = {col: round(float(imp), 4) for col, imp in zip(X.columns, rf.feature_importances_)}

    return PipelineResult(
        method_id="random_forest_classification",
        method_name="Random Forest Classification",
        summary=f"Random Forest accuracy: {accuracy*100:.1f}% on test set ({len(X_test)} samples). Top feature: {max(importances, key=importances.get)}",
        metrics={"accuracy": accuracy, "feature_importances": importances, "classification_report": report},
    )


def run_xgboost_regression(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """XGBoost regression to predict a target parameter."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("pip install xgboost  (required for XGBoost regression)")
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    config = config or {}
    target = config.get("target_parameter")

    pivot = _pivot_params(df).dropna()
    if len(pivot) < 20 or pivot.shape[1] < 2:
        return PipelineResult(
            method_id="xgboost_regression",
            method_name="XGBoost Regression",
            summary="Insufficient data (need ≥ 20 samples with ≥ 2 parameters).",
            metrics={},
        )

    if target is None or target not in pivot.columns:
        target = pivot.columns[0]

    X = pivot.drop(columns=[target])
    y = pivot[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = round(float(r2_score(y_test, y_pred)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
    importances = {col: round(float(imp), 4) for col, imp in zip(X.columns, model.feature_importances_)}

    return PipelineResult(
        method_id="xgboost_regression",
        method_name="XGBoost Regression",
        summary=f"Predicting {target}: R² = {r2}, RMSE = {rmse} on {len(X_test)} test samples.",
        metrics={"R2": r2, "RMSE": rmse, "target": target, "feature_importances": importances},
    )


def run_arima_forecast(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """ARIMA/SARIMA time-series forecasting."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError("pip install statsmodels  (required for ARIMA)")

    config = config or {}
    target = config.get("target_parameter")
    order = config.get("order", (1, 1, 1))
    forecast_steps = config.get("forecast_steps", 12)

    if target is None:
        target = df["parameter"].value_counts().index[0]

    ts = df[df["parameter"] == target].copy()
    dt_col = "sample_datetime" if "sample_datetime" in ts.columns else "reading_datetime"
    ts[dt_col] = pd.to_datetime(ts[dt_col], errors="coerce")
    ts["value"] = pd.to_numeric(ts["value"], errors="coerce")
    ts = ts.dropna(subset=[dt_col, "value"]).set_index(dt_col).sort_index()
    series = ts["value"].resample("MS").mean().dropna()

    if len(series) < 24:
        return PipelineResult(
            method_id="arima_forecast",
            method_name="ARIMA Forecast",
            summary=f"Insufficient data for ARIMA (got {len(series)} months, need ≥ 24).",
            metrics={},
        )

    train = series[:-forecast_steps] if len(series) > forecast_steps else series
    test = series[-forecast_steps:] if len(series) > forecast_steps else None

    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_steps)

        metrics: dict[str, Any] = {
            "aic": round(float(fitted.aic), 2),
            "bic": round(float(fitted.bic), 2),
            "order": list(order),
            "forecast_steps": forecast_steps,
            "target": target,
        }

        if test is not None and len(test) > 0:
            common_idx = forecast.index.intersection(test.index)
            if len(common_idx) > 0:
                rmse = round(float(np.sqrt(((forecast[common_idx] - test[common_idx]) ** 2).mean())), 4)
                metrics["RMSE"] = rmse

        return PipelineResult(
            method_id="arima_forecast",
            method_name="ARIMA Forecast",
            summary=f"ARIMA{order} for {target}: AIC={metrics['aic']}. Forecasted {forecast_steps} periods ahead.",
            metrics=metrics,
            details={
                "forecast_values": forecast.to_dict(),
                "train_size": len(train),
            },
        )
    except Exception as e:
        return PipelineResult(
            method_id="arima_forecast",
            method_name="ARIMA Forecast",
            summary=f"ARIMA fitting failed: {e}",
            metrics={},
        )


def run_correlation_analysis(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Correlation analysis between water quality parameters."""
    from scipy import stats as sp_stats

    pivot = _pivot_params(df).dropna()
    if pivot.shape[1] < 2:
        return PipelineResult(
            method_id="correlation_analysis",
            method_name="Correlation Analysis",
            summary="Need at least 2 parameters with overlapping data.",
            metrics={},
        )

    corr = pivot.corr(method="pearson").round(4)

    # Compute p-values
    pvals: dict[str, dict[str, float]] = {}
    sig_pairs: list[dict] = []
    for i, c1 in enumerate(pivot.columns):
        pvals[c1] = {}
        for j, c2 in enumerate(pivot.columns):
            if i < j:
                r, p = sp_stats.pearsonr(pivot[c1], pivot[c2])
                pvals[c1][c2] = round(float(p), 6)
                if p < 0.05 and abs(r) > 0.3:
                    sig_pairs.append({"param1": c1, "param2": c2, "r": round(float(r), 4), "p": round(float(p), 6)})

    sig_pairs.sort(key=lambda x: abs(x["r"]), reverse=True)

    return PipelineResult(
        method_id="correlation_analysis",
        method_name="Pearson Correlation Analysis",
        summary=f"Analysed {pivot.shape[1]} parameters. {len(sig_pairs)} significant pairs found (p<0.05, |r|>0.3).",
        metrics={
            "n_parameters": pivot.shape[1],
            "n_significant_pairs": len(sig_pairs),
            "top_correlations": sig_pairs[:10],
        },
        details={"correlation_matrix": corr.to_dict()},
    )


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline dispatcher
# ═══════════════════════════════════════════════════════════════════════

PIPELINE_REGISTRY: dict[str, callable] = {
    "trend_analysis": run_mann_kendall,
    "wqi_calculation": run_wqi,
    "pca_clustering": run_pca_clustering,
    "random_forest_classification": run_random_forest,
    "xgboost_regression": run_xgboost_regression,
    "arima_forecast": run_arima_forecast,
    "correlation_analysis": run_correlation_analysis,
}


def run_pipeline(method_id: str, df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """
    Dispatch to the correct pipeline based on methodology ID.

    Parameters
    ----------
    method_id : str
        ID from the knowledge base (e.g. ``"trend_analysis"``).
    df : pd.DataFrame
        Water data in long format.
    config : dict | None
        Pipeline-specific configuration.

    Returns
    -------
    PipelineResult with summary, metrics, and details.

    Raises
    ------
    ValueError if method_id is not found in the registry.
    """
    func = PIPELINE_REGISTRY.get(method_id)
    if func is None:
        available = ", ".join(sorted(PIPELINE_REGISTRY.keys()))
        raise ValueError(f"No pipeline for '{method_id}'. Available: {available}")

    logger.info("Running pipeline: %s", method_id)
    result = func(df, config)
    logger.info("Pipeline %s complete: %s", method_id, result.summary)
    return result


def list_available_pipelines() -> list[str]:
    """Return IDs of all implemented pipelines."""
    return sorted(PIPELINE_REGISTRY.keys())
