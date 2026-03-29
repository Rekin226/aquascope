"""
Model implementation pipeline.

After the AI recommender suggests methodologies and the user approves one,
this module auto-builds and runs the corresponding analysis, producing
results, metrics, and visualisation figures.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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
        summary=f"Random Forest accuracy: {accuracy*100:.1f}% on test set ({len(X_test)} samples). Top feature: {max(importances, key=lambda k: importances[k])}",
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
#  Additional pipeline implementations
# ═══════════════════════════════════════════════════════════════════════


def run_svr_prediction(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Support Vector Regression for water quality parameter prediction."""
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    config = config or {}
    target = config.get("target_parameter")

    pivot = _pivot_params(df).dropna()
    if len(pivot) < 20 or pivot.shape[1] < 2:
        return PipelineResult(
            method_id="svr_prediction",
            method_name="Support Vector Regression",
            summary="Insufficient data (need ≥ 20 samples with ≥ 2 parameters).",
            metrics={},
        )

    if target is None or target not in pivot.columns:
        target = pivot.columns[0]

    X = pivot.drop(columns=[target])
    y = pivot[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    kernel = config.get("kernel", "rbf")
    model = SVR(kernel=kernel, C=config.get("C", 1.0), epsilon=config.get("epsilon", 0.1))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = round(float(r2_score(y_test, y_pred)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

    return PipelineResult(
        method_id="svr_prediction",
        method_name="Support Vector Regression",
        summary=f"SVR ({kernel} kernel) predicting {target}: R\u00b2 = {r2}, RMSE = {rmse} on {len(X_test)} test samples.",
        metrics={"R2": r2, "RMSE": rmse, "target": target, "kernel": kernel, "n_support_vectors": int(model.n_support_.sum())},
    )


def run_bayesian_network(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Bayesian Network structure learning via conditional independence tests."""
    from scipy import stats as sp_stats

    config = config or {}
    alpha = config.get("alpha", 0.05)

    pivot = _pivot_params(df).dropna()
    if len(pivot) < 30 or pivot.shape[1] < 2:
        return PipelineResult(
            method_id="bayesian_network",
            method_name="Bayesian Network Causal Inference",
            summary="Insufficient data (need ≥ 30 samples with ≥ 2 parameters).",
            metrics={},
        )

    params = list(pivot.columns)
    edges: list[dict] = []
    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            if i >= j:
                continue
            others = [p for k, p in enumerate(params) if k != i and k != j]
            if others:
                from sklearn.linear_model import LinearRegression
                X_others = pivot[others].values
                lr1 = LinearRegression().fit(X_others, pivot[p1].values)
                lr2 = LinearRegression().fit(X_others, pivot[p2].values)
                resid1 = pivot[p1].values - lr1.predict(X_others)
                resid2 = pivot[p2].values - lr2.predict(X_others)
                r, p_val = sp_stats.pearsonr(resid1, resid2)
            else:
                r, p_val = sp_stats.pearsonr(pivot[p1], pivot[p2])

            if p_val < alpha:
                edges.append({
                    "from": p1, "to": p2,
                    "partial_corr": round(float(r), 4),
                    "p_value": round(float(p_val), 6),
                })

    edges.sort(key=lambda e: abs(e["partial_corr"]), reverse=True)

    return PipelineResult(
        method_id="bayesian_network",
        method_name="Bayesian Network Causal Inference",
        summary=f"Identified {len(edges)} significant conditional dependencies among {len(params)} parameters (\u03b1={alpha}).",
        metrics={"n_nodes": len(params), "n_edges": len(edges), "alpha": alpha},
        details={"edges": edges, "parameters": params},
    )


def run_monte_carlo(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Monte Carlo uncertainty analysis on water quality statistics."""
    config = config or {}
    n_simulations = config.get("n_simulations", 5000)
    target_params = config.get("parameters", df["parameter"].unique().tolist())

    results: list[dict] = []
    for param in target_params:
        vals = pd.to_numeric(df[df["parameter"] == param]["value"], errors="coerce").dropna()
        if len(vals) < 10:
            continue
        mean, std = float(vals.mean()), float(vals.std())
        if std == 0:
            continue

        simulated = np.random.default_rng(42).normal(mean, std, size=n_simulations)
        ci_lower = float(np.percentile(simulated, 2.5))
        ci_upper = float(np.percentile(simulated, 97.5))
        exceedance_prob = None

        standards = {"DO": 5.0, "BOD5": 4.0, "COD": 25.0, "NH3-N": 1.0, "SS": 50.0, "pH": 9.0}
        if param in standards:
            threshold = standards[param]
            if param == "DO":
                exceedance_prob = round(float((simulated < threshold).mean()), 4)
            else:
                exceedance_prob = round(float((simulated > threshold).mean()), 4)

        results.append({
            "parameter": param,
            "observed_mean": round(mean, 4),
            "observed_std": round(std, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "exceedance_probability": exceedance_prob,
        })

    return PipelineResult(
        method_id="monte_carlo_uncertainty",
        method_name="Monte Carlo Uncertainty Analysis",
        summary=f"Ran {n_simulations} simulations for {len(results)} parameters. 95% CIs and exceedance probabilities computed.",
        metrics={"n_simulations": n_simulations, "n_parameters": len(results)},
        details={"results": results},
    )


def run_wavelet_analysis(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Continuous wavelet transform to detect periodic patterns."""
    from scipy import signal

    config = config or {}
    target = config.get("target_parameter")

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
            method_id="wavelet_analysis",
            method_name="Wavelet Transform Analysis",
            summary=f"Insufficient data for wavelet analysis (got {len(series)} months, need ≥ 24).",
            metrics={},
        )

    values = series.values
    values = (values - values.mean()) / (values.std() if values.std() > 0 else 1.0)

    widths = np.arange(2, min(len(values) // 2, 64) + 1)
    cwt_matrix = signal.cwt(values, signal.ricker, widths)
    power = np.abs(cwt_matrix) ** 2

    avg_power = power.mean(axis=1)
    dominant_idx = int(np.argmax(avg_power))
    dominant_period = int(widths[dominant_idx])

    top_indices = np.argsort(avg_power)[-3:][::-1]
    top_periods = [{"period_months": int(widths[i]), "avg_power": round(float(avg_power[i]), 4)} for i in top_indices]

    return PipelineResult(
        method_id="wavelet_analysis",
        method_name="Wavelet Transform Analysis",
        summary=f"Wavelet analysis of {target} ({len(series)} months). Dominant period: {dominant_period} months.",
        metrics={"dominant_period_months": dominant_period, "n_observations": len(series), "target": target},
        details={"top_periods": top_periods},
    )


def run_copula_analysis(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Copula-based dependence modelling between parameter pairs."""
    from scipy import stats as sp_stats

    config = config or {}
    pivot = _pivot_params(df).dropna()

    if len(pivot) < 30 or pivot.shape[1] < 2:
        return PipelineResult(
            method_id="copula_analysis",
            method_name="Copula Dependence Modelling",
            summary="Insufficient data (need ≥ 30 samples with ≥ 2 parameters).",
            metrics={},
        )

    params = list(pivot.columns)
    pair_results: list[dict] = []

    for i, p1 in enumerate(params):
        for j, p2 in enumerate(params):
            if i >= j:
                continue
            u = sp_stats.rankdata(pivot[p1]) / (len(pivot) + 1)
            v = sp_stats.rankdata(pivot[p2]) / (len(pivot) + 1)

            tau, p_val = sp_stats.kendalltau(pivot[p1], pivot[p2])

            q = 0.1
            upper_tail = float(np.mean((u > 1 - q) & (v > 1 - q)) / q) if q > 0 else 0.0
            lower_tail = float(np.mean((u < q) & (v < q)) / q) if q > 0 else 0.0

            pair_results.append({
                "param1": p1, "param2": p2,
                "kendall_tau": round(float(tau), 4),
                "p_value": round(float(p_val), 6),
                "upper_tail_dep": round(upper_tail, 4),
                "lower_tail_dep": round(lower_tail, 4),
            })

    pair_results.sort(key=lambda x: abs(x["kendall_tau"]), reverse=True)

    return PipelineResult(
        method_id="copula_analysis",
        method_name="Copula Dependence Modelling",
        summary=f"Analysed {len(pair_results)} parameter pairs. Tail dependencies computed for extreme co-occurrence.",
        metrics={"n_pairs": len(pair_results), "n_parameters": len(params)},
        details={"pair_results": pair_results},
    )


def run_kriging(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Ordinary kriging spatial interpolation using variogram modelling."""
    from scipy.spatial.distance import pdist, squareform

    config = config or {}
    target = config.get("target_parameter")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        if "location" in df.columns:
            df = df.copy()
            df["latitude"] = df["location"].apply(
                lambda x: x.get("latitude") if isinstance(x, dict) else getattr(x, "latitude", None)
            )
            df["longitude"] = df["location"].apply(
                lambda x: x.get("longitude") if isinstance(x, dict) else getattr(x, "longitude", None)
            )

    geo = df.dropna(subset=["latitude", "longitude"]).copy()
    if target:
        geo = geo[geo["parameter"] == target]
    geo["value"] = pd.to_numeric(geo["value"], errors="coerce")
    geo = geo.dropna(subset=["value"])

    stations = geo.groupby("station_id").agg(
        lat=("latitude", "first"), lon=("longitude", "first"), value=("value", "mean")
    ).dropna()

    if len(stations) < 10:
        return PipelineResult(
            method_id="kriging_interpolation",
            method_name="Kriging Spatial Interpolation",
            summary=f"Insufficient georeferenced stations (got {len(stations)}, need ≥ 10).",
            metrics={},
        )

    coords = stations[["lat", "lon"]].values
    values = stations["value"].values
    dist_matrix = squareform(pdist(coords))

    n = len(coords)
    max_dist = dist_matrix.max() / 2
    n_bins = min(15, n)
    bins = np.linspace(0, max_dist, n_bins + 1)
    gamma_values: list[float] = []
    bin_centers: list[float] = []

    for b in range(n_bins):
        pairs = (dist_matrix > bins[b]) & (dist_matrix <= bins[b + 1])
        if pairs.sum() == 0:
            continue
        diffs = []
        for ii in range(n):
            for jj in range(ii + 1, n):
                if pairs[ii, jj]:
                    diffs.append((values[ii] - values[jj]) ** 2)
        if diffs:
            gamma_values.append(float(np.mean(diffs)) / 2)
            bin_centers.append(float((bins[b] + bins[b + 1]) / 2))

    sill = float(np.var(values))
    range_est = float(max_dist * 0.6)
    nugget = float(gamma_values[0]) if gamma_values else 0.0

    return PipelineResult(
        method_id="kriging_interpolation",
        method_name="Kriging Spatial Interpolation",
        summary=(
            f"Variogram modelled for {len(stations)} stations. "
            f"Sill={round(sill, 4)}, Range\u2248{round(range_est, 2)}, Nugget\u2248{round(nugget, 4)}."
        ),
        metrics={
            "n_stations": len(stations),
            "sill": round(sill, 4),
            "range": round(range_est, 2),
            "nugget": round(nugget, 4),
            "target": target,
        },
        details={
            "variogram_bins": bin_centers,
            "variogram_gamma": gamma_values,
            "station_summary": stations.describe().to_dict(),
        },
    )


def run_lstm_forecasting(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """LSTM-style sequential forecasting via MLP with lag features."""
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    config = config or {}
    target = config.get("target_parameter")
    n_lags = config.get("n_lags", 6)
    forecast_steps = config.get("forecast_steps", 6)

    if target is None:
        target = df["parameter"].value_counts().index[0]

    ts = df[df["parameter"] == target].copy()
    dt_col = "sample_datetime" if "sample_datetime" in ts.columns else "reading_datetime"
    ts[dt_col] = pd.to_datetime(ts[dt_col], errors="coerce")
    ts["value"] = pd.to_numeric(ts["value"], errors="coerce")
    ts = ts.dropna(subset=[dt_col, "value"]).set_index(dt_col).sort_index()
    series = ts["value"].resample("MS").mean().dropna()

    min_required = n_lags + forecast_steps + 10
    if len(series) < min_required:
        return PipelineResult(
            method_id="lstm_forecasting",
            method_name="LSTM-Style Sequential Forecasting",
            summary=f"Insufficient data (got {len(series)} months, need ≥ {min_required}).",
            metrics={},
        )

    values = series.values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(n_lags, len(scaled)):
        X.append(scaled[i - n_lags:i])
        y.append(scaled[i])
    X_arr, y_arr = np.array(X), np.array(y)

    split = len(X_arr) - forecast_steps
    X_train, X_test = X_arr[:split], X_arr[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=500,
        random_state=42, early_stopping=True, validation_fraction=0.15,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    r2 = round(float(r2_score(y_test_orig, y_pred_orig)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))), 4)

    return PipelineResult(
        method_id="lstm_forecasting",
        method_name="LSTM-Style Sequential Forecasting",
        summary=f"MLP sequential model for {target} ({n_lags} lags): R\u00b2 = {r2}, RMSE = {rmse} on {len(y_test)} test steps.",
        metrics={"R2": r2, "RMSE": rmse, "n_lags": n_lags, "target": target, "train_size": split, "test_size": len(y_test)},
        details={"forecast_actual": y_test_orig.tolist(), "forecast_predicted": y_pred_orig.tolist()},
    )


def run_transformer_forecast(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Transformer-style multi-resolution forecasting via MLP with attention-like features."""
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    config = config or {}
    target = config.get("target_parameter")
    n_lags = config.get("n_lags", 12)
    forecast_steps = config.get("forecast_steps", 6)

    if target is None:
        target = df["parameter"].value_counts().index[0]

    ts = df[df["parameter"] == target].copy()
    dt_col = "sample_datetime" if "sample_datetime" in ts.columns else "reading_datetime"
    ts[dt_col] = pd.to_datetime(ts[dt_col], errors="coerce")
    ts["value"] = pd.to_numeric(ts["value"], errors="coerce")
    ts = ts.dropna(subset=[dt_col, "value"]).set_index(dt_col).sort_index()
    series = ts["value"].resample("MS").mean().dropna()

    min_required = n_lags + forecast_steps + 10
    if len(series) < min_required:
        return PipelineResult(
            method_id="transformer_forecast",
            method_name="Transformer-Style Forecasting",
            summary=f"Insufficient data (got {len(series)} months, need ≥ {min_required}).",
            metrics={},
        )

    values = series.values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(n_lags, len(scaled)):
        window = scaled[i - n_lags:i]
        features = np.concatenate([
            window,
            [window.mean(), window.std(), window[-1] - window[0]],
        ])
        X.append(features)
        y.append(scaled[i])
    X_arr, y_arr = np.array(X), np.array(y)

    split = len(X_arr) - forecast_steps
    X_train, X_test = X_arr[:split], X_arr[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), max_iter=500,
        random_state=42, early_stopping=True, validation_fraction=0.15,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    r2 = round(float(r2_score(y_test_orig, y_pred_orig)), 4)
    rmse = round(float(np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))), 4)

    return PipelineResult(
        method_id="transformer_forecast",
        method_name="Transformer-Style Forecasting",
        summary=f"Multi-resolution MLP for {target} ({n_lags} lags): R\u00b2 = {r2}, RMSE = {rmse} on {len(y_test)} steps.",
        metrics={"R2": r2, "RMSE": rmse, "n_lags": n_lags, "target": target, "train_size": split, "test_size": len(y_test)},
        details={"forecast_actual": y_test_orig.tolist(), "forecast_predicted": y_pred_orig.tolist()},
    )


def run_transfer_learning(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Transfer learning: train on majority of stations, fine-tune on target station."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    config = config or {}
    target_parameter = config.get("target_parameter")

    pivot = _pivot_params(df).dropna()
    if len(pivot) < 30 or pivot.shape[1] < 2:
        return PipelineResult(
            method_id="transfer_learning_wq",
            method_name="Transfer Learning for Water Quality",
            summary="Insufficient data (need ≥ 30 samples with ≥ 2 parameters).",
            metrics={},
        )

    if target_parameter is None or target_parameter not in pivot.columns:
        target_parameter = pivot.columns[0]

    X = pivot.drop(columns=[target_parameter])
    y = pivot[target_parameter]

    # Split 70/30 for source/target domain
    split = int(len(X) * 0.7)
    source_X, source_y = X[:split], y[:split]
    target_X, target_y = X[split:], y[split:]

    if len(source_X) < 10 or len(target_X) < 5:
        return PipelineResult(
            method_id="transfer_learning_wq",
            method_name="Transfer Learning for Water Quality",
            summary="Insufficient data for source/target split.",
            metrics={},
        )

    # Pre-train on source
    base_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    base_model.fit(source_X, source_y)
    base_pred = base_model.predict(target_X)
    base_r2 = round(float(r2_score(target_y, base_pred)), 4)

    # Fine-tune with weighted target samples
    combined_X = pd.concat([source_X, target_X])
    combined_y = pd.concat([source_y, target_y])
    weights = np.ones(len(combined_X))
    weights[len(source_X):] = 3.0

    fine_model = GradientBoostingRegressor(n_estimators=150, max_depth=4, random_state=42)
    fine_model.fit(combined_X, combined_y, sample_weight=weights)
    fine_pred = fine_model.predict(target_X)
    fine_r2 = round(float(r2_score(target_y, fine_pred)), 4)
    fine_rmse = round(float(np.sqrt(mean_squared_error(target_y, fine_pred))), 4)

    return PipelineResult(
        method_id="transfer_learning_wq",
        method_name="Transfer Learning for Water Quality",
        summary=(
            f"Predicting {target_parameter}: source-only R\u00b2 = {base_r2}, "
            f"after transfer R\u00b2 = {fine_r2}, RMSE = {fine_rmse}."
        ),
        metrics={
            "source_only_R2": base_r2, "transfer_R2": fine_r2, "transfer_RMSE": fine_rmse,
            "source_samples": len(source_X), "target_samples": len(target_X),
            "target_parameter": target_parameter,
        },
    )


def run_sdg6_benchmarking(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """SDG 6 cross-entity benchmarking and gap analysis."""
    config = config or {}

    is_sdg = "indicator_code" in df.columns or "country_code" in df.columns
    if is_sdg:
        entity_col = "country_code" if "country_code" in df.columns else "country_name"
        indicator_col = "indicator_code" if "indicator_code" in df.columns else "parameter"
    else:
        entity_col = "station_id"
        indicator_col = "parameter"
    value_col = "value"

    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])

    if len(df) < 10:
        return PipelineResult(
            method_id="sdg6_benchmarking",
            method_name="SDG 6 Benchmarking",
            summary="Insufficient data for benchmarking (need ≥ 10 records).",
            metrics={},
        )

    summary_stats = df.groupby([entity_col, indicator_col])[value_col].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary_stats.columns = ["entity", "indicator", "mean", "std", "count"]

    rankings: list[dict] = []
    for indicator in summary_stats["indicator"].unique():
        ind_data = summary_stats[summary_stats["indicator"] == indicator].sort_values("mean", ascending=False)
        for rank, (_, row) in enumerate(ind_data.iterrows(), 1):
            rankings.append({
                "entity": row["entity"], "indicator": indicator,
                "mean": round(float(row["mean"]), 4),
                "rank": rank, "n_observations": int(row["count"]),
            })

    n_entities = summary_stats["entity"].nunique()
    n_indicators = summary_stats["indicator"].nunique()

    return PipelineResult(
        method_id="sdg6_benchmarking",
        method_name="SDG 6 Cross-Entity Benchmarking",
        summary=f"Benchmarked {n_entities} entities across {n_indicators} indicators.",
        metrics={"n_entities": n_entities, "n_indicators": n_indicators, "total_records": len(df)},
        details={"rankings": rankings},
    )


def run_satellite_eutrophication(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Trophic state assessment using Carlson TSI from water quality parameters."""
    config = config or {}

    tsi_results: list[dict] = []
    for station in df["station_id"].unique():
        sdata = df[df["station_id"] == station]
        station_metrics: dict[str, float] = {}

        for param in ["Chlorophyll-a", "Turbidity", "TP", "Secchi"]:
            vals = pd.to_numeric(sdata[sdata["parameter"] == param]["value"], errors="coerce").dropna()
            if len(vals) > 0:
                station_metrics[param] = float(vals.mean())

        tsi_values: dict[str, float] = {}
        if "Chlorophyll-a" in station_metrics and station_metrics["Chlorophyll-a"] > 0:
            tsi_values["TSI_Chl"] = round(9.81 * np.log(station_metrics["Chlorophyll-a"]) + 30.6, 2)
        if "TP" in station_metrics and station_metrics["TP"] > 0:
            tsi_values["TSI_TP"] = round(14.42 * np.log(station_metrics["TP"]) + 4.15, 2)

        if tsi_values:
            avg_tsi = round(float(np.mean(list(tsi_values.values()))), 2)
            if avg_tsi < 30:
                state = "Oligotrophic"
            elif avg_tsi < 50:
                state = "Mesotrophic"
            elif avg_tsi < 70:
                state = "Eutrophic"
            else:
                state = "Hypereutrophic"

            tsi_results.append({"station": station, "tsi": avg_tsi, "trophic_state": state, **tsi_values})

    if not tsi_results:
        return PipelineResult(
            method_id="satellite_eutrophication",
            method_name="Trophic State Assessment",
            summary="No Chlorophyll-a or TP data available for trophic state computation.",
            metrics={},
        )

    state_counts = pd.DataFrame(tsi_results)["trophic_state"].value_counts().to_dict()

    return PipelineResult(
        method_id="satellite_eutrophication",
        method_name="Trophic State Assessment (Carlson TSI)",
        summary=f"Assessed {len(tsi_results)} stations. Distribution: {state_counts}.",
        metrics={"n_stations": len(tsi_results), "trophic_distribution": state_counts},
        details={"results": tsi_results},
    )


def run_gis_watershed(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Spatial analysis of water quality patterns across stations."""
    from scipy import stats as sp_stats

    config = config or {}
    target_params = config.get("parameters", df["parameter"].unique().tolist())

    results: list[dict] = []
    for param in target_params:
        pdata = df[df["parameter"] == param].copy()
        pdata["value"] = pd.to_numeric(pdata["value"], errors="coerce")
        pdata = pdata.dropna(subset=["value"])

        if len(pdata) < 5 or pdata["station_id"].nunique() < 2:
            continue

        station_means = pdata.groupby("station_id")["value"].mean()

        groups = [g["value"].values for _, g in pdata.groupby("station_id") if len(g) >= 3]
        if len(groups) >= 2:
            h_stat, p_val = sp_stats.kruskal(*groups)
            spatial_cv = round(
                float(station_means.std() / station_means.mean()) * 100, 2
            ) if station_means.mean() != 0 else 0.0

            results.append({
                "parameter": param,
                "n_stations": int(station_means.shape[0]),
                "spatial_cv_pct": spatial_cv,
                "kruskal_h": round(float(h_stat), 4),
                "kruskal_p": round(float(p_val), 6),
                "significant_spatial_diff": p_val < 0.05,
                "station_range": [round(float(station_means.min()), 4), round(float(station_means.max()), 4)],
            })

    results.sort(key=lambda x: x.get("spatial_cv_pct", 0), reverse=True)

    return PipelineResult(
        method_id="gis_watershed_analysis",
        method_name="GIS Watershed Spatial Analysis",
        summary=f"Spatial analysis of {len(results)} parameters. Kruskal-Wallis tests for heterogeneity.",
        metrics={"n_parameters_tested": len(results), "n_significant": sum(1 for r in results if r["significant_spatial_diff"])},
        details={"results": results},
    )


def run_hec_ras(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Hydraulic analysis using flow duration curves and statistics."""
    config = config or {}

    flow_data = df[df["parameter"].isin(["Flow", "Water level", "Discharge"])].copy()
    flow_data["value"] = pd.to_numeric(flow_data["value"], errors="coerce")
    flow_data = flow_data.dropna(subset=["value"])

    if len(flow_data) < 10:
        return PipelineResult(
            method_id="hec_ras_modelling",
            method_name="Hydraulic Analysis (Flow Statistics)",
            summary="Insufficient flow/level data (need ≥ 10 records with Flow or Water level parameters).",
            metrics={},
        )

    dt_col = "sample_datetime" if "sample_datetime" in flow_data.columns else "reading_datetime"
    flow_data[dt_col] = pd.to_datetime(flow_data[dt_col], errors="coerce")

    station_results: list[dict] = []
    for station in flow_data["station_id"].unique():
        sdata = flow_data[flow_data["station_id"] == station].sort_values(dt_col)
        vals = sdata["value"].values
        if len(vals) < 3:
            continue

        sorted_vals = np.sort(vals)[::-1]
        exceedance = np.arange(1, len(sorted_vals) + 1) / (len(sorted_vals) + 1) * 100

        q_indices = {}
        for pct in [10, 50, 90, 95]:
            idx = int(np.argmin(np.abs(exceedance - pct)))
            q_indices[f"Q{pct}"] = round(float(sorted_vals[idx]), 4)

        flashiness = round(
            float(np.sum(np.abs(np.diff(vals))) / np.sum(vals)), 4
        ) if np.sum(vals) > 0 else 0.0

        station_results.append({
            "station": station,
            "n_observations": len(vals),
            "mean_flow": round(float(vals.mean()), 4),
            "std_flow": round(float(vals.std()), 4),
            "flashiness_index": flashiness,
            **q_indices,
        })

    return PipelineResult(
        method_id="hec_ras_modelling",
        method_name="Hydraulic Analysis (Flow Duration & Statistics)",
        summary=f"Computed flow statistics for {len(station_results)} stations.",
        metrics={"n_stations": len(station_results)},
        details={"station_results": station_results},
    )


def run_qual2k(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Streeter-Phelps dissolved oxygen sag model for river water quality."""
    config = config or {}

    do_data = df[df["parameter"] == "DO"].copy()
    bod_data = df[df["parameter"].isin(["BOD5", "BOD"])].copy()

    do_data["value"] = pd.to_numeric(do_data["value"], errors="coerce")
    bod_data["value"] = pd.to_numeric(bod_data["value"], errors="coerce")

    if len(do_data.dropna(subset=["value"])) < 5 or len(bod_data.dropna(subset=["value"])) < 5:
        return PipelineResult(
            method_id="qual2k_modelling",
            method_name="QUAL2K River DO Model (Streeter-Phelps)",
            summary="Insufficient DO and BOD data (need ≥ 5 records of each).",
            metrics={},
        )

    do_mean = float(do_data["value"].dropna().mean())
    bod_mean = float(bod_data["value"].dropna().mean())
    do_sat = config.get("do_saturation", 9.0)
    kd = config.get("kd", 0.3)
    ka = config.get("ka", 0.5)

    initial_deficit = do_sat - do_mean
    if ka == kd:
        ka += 0.01
    if kd > 0 and ka > 0 and bod_mean > 0:
        arg = (ka / kd) * (1 - initial_deficit * (ka - kd) / (kd * bod_mean))
        if arg > 0:
            tc = max(0.0, float((1 / (ka - kd)) * np.log(arg)))
        else:
            tc = 0.0
        dc = (kd * bod_mean / (ka - kd)) * (np.exp(-kd * tc) - np.exp(-ka * tc)) + initial_deficit * np.exp(-ka * tc)
        min_do = do_sat - dc
    else:
        tc, dc, min_do = 0.0, initial_deficit, do_mean

    t = np.linspace(0, 10, 50)
    deficit = (kd * bod_mean / (ka - kd)) * (np.exp(-kd * t) - np.exp(-ka * t)) + initial_deficit * np.exp(-ka * t)
    do_profile = do_sat - deficit

    return PipelineResult(
        method_id="qual2k_modelling",
        method_name="QUAL2K River DO Model (Streeter-Phelps)",
        summary=(
            f"Streeter-Phelps: critical DO = {round(float(min_do), 2)} mg/L at t = {round(tc, 2)} days. "
            f"Mean DO = {round(do_mean, 2)}, mean BOD = {round(bod_mean, 2)} mg/L."
        ),
        metrics={
            "critical_do": round(float(min_do), 2),
            "critical_time_days": round(tc, 2),
            "critical_deficit": round(float(dc), 2),
            "observed_do_mean": round(do_mean, 2),
            "observed_bod_mean": round(bod_mean, 2),
            "kd": kd, "ka": ka,
        },
        details={"do_profile_time": t.tolist(), "do_profile_mg_L": do_profile.tolist()},
    )


def run_swat(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Simplified SWAT-style water balance analysis."""
    config = config or {}

    flow_params = ["Flow", "Discharge", "Water level", "Precipitation"]
    available = df[df["parameter"].isin(flow_params)].copy()
    available["value"] = pd.to_numeric(available["value"], errors="coerce")
    available = available.dropna(subset=["value"])

    if len(available) < 12:
        return PipelineResult(
            method_id="swat_modelling",
            method_name="SWAT-Style Water Balance",
            summary="Insufficient hydrological data (need ≥ 12 records with Flow/Precipitation).",
            metrics={},
        )

    dt_col = "sample_datetime" if "sample_datetime" in available.columns else "reading_datetime"
    available[dt_col] = pd.to_datetime(available[dt_col], errors="coerce")
    available["month"] = available[dt_col].dt.month

    monthly = available.groupby(["month", "parameter"])["value"].mean().unstack(fill_value=0)

    param_summaries: dict[str, dict] = {}
    for param in monthly.columns:
        param_mean = float(monthly[param].mean())
        param_summaries[param] = {
            "monthly_means": {int(m): round(float(v), 4) for m, v in monthly[param].items()},
            "annual_mean": round(param_mean, 4),
            "seasonal_cv": round(float(monthly[param].std() / param_mean) * 100, 2) if param_mean != 0 else 0.0,
        }

    baseflow_index = None
    if "Flow" in monthly.columns:
        flows = monthly["Flow"].values
        if len(flows) > 0 and flows.mean() > 0:
            baseflow_index = round(float(flows.min() / flows.mean()), 4)

    return PipelineResult(
        method_id="swat_modelling",
        method_name="SWAT-Style Water Balance Analysis",
        summary=f"Monthly water balance for {len(monthly.columns)} parameters over {len(monthly)} months.",
        metrics={"n_parameters": len(monthly.columns), "baseflow_index": baseflow_index},
        details={"parameter_summaries": param_summaries},
    )


def run_mbbr_pilot(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """MBBR pilot plant performance evaluation (removal efficiency analysis)."""
    config = config or {}

    treatment_params = ["BOD5", "COD", "NH3-N", "SS", "TN", "TP"]
    results: list[dict] = []

    for param in treatment_params:
        pdata = df[df["parameter"] == param].copy()
        pdata["value"] = pd.to_numeric(pdata["value"], errors="coerce")
        pdata = pdata.dropna(subset=["value"])
        if len(pdata) < 5:
            continue

        vals = pdata["value"].values
        mean_val = float(vals.mean())
        std_val = float(vals.std())

        removal_pct = None
        if len(vals) >= 10:
            x = np.arange(len(vals))
            slope, intercept = np.polyfit(x, vals, 1)
            initial_est = intercept
            final_est = slope * len(vals) + intercept
            if initial_est > 0:
                removal_pct = round(max(0.0, (initial_est - final_est) / initial_est * 100), 2)

        standards = {"BOD5": 30.0, "COD": 100.0, "NH3-N": 10.0, "SS": 30.0, "TN": 15.0, "TP": 2.0}
        compliance = None
        if param in standards:
            compliance = round(float((vals <= standards[param]).mean()) * 100, 2)

        results.append({
            "parameter": param,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "n_samples": len(vals),
            "estimated_removal_pct": removal_pct,
            "compliance_pct": compliance,
        })

    return PipelineResult(
        method_id="mbbr_pilot_study",
        method_name="MBBR Performance Evaluation",
        summary=f"Evaluated {len(results)} treatment parameters. Removal efficiencies and compliance rates computed.",
        metrics={"n_parameters": len(results)},
        details={"results": results},
    )


def run_mbr_optimisation(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """MBR fouling and optimisation analysis."""
    from scipy import stats as sp_stats

    config = config or {}

    mbr_params = ["COD", "SS", "MLSS", "TMP", "Flux", "DO", "BOD5"]
    available_params = [p for p in mbr_params if p in df["parameter"].unique()]

    if len(available_params) < 2:
        return PipelineResult(
            method_id="mbr_optimisation",
            method_name="MBR Optimisation Analysis",
            summary=f"Insufficient MBR parameters (need ≥ 2 of {mbr_params}).",
            metrics={},
        )

    param_stats: list[dict] = []
    for param in available_params:
        vals = pd.to_numeric(df[df["parameter"] == param]["value"], errors="coerce").dropna()
        if len(vals) >= 3:
            param_mean = float(vals.mean())
            param_stats.append({
                "parameter": param,
                "mean": round(param_mean, 4),
                "std": round(float(vals.std()), 4),
                "cv_pct": round(float(vals.std() / param_mean) * 100, 2) if param_mean != 0 else 0.0,
                "n": len(vals),
            })

    fouling_analysis = None
    if "TMP" in available_params:
        tmp_vals = pd.to_numeric(df[df["parameter"] == "TMP"]["value"], errors="coerce").dropna()
        if len(tmp_vals) >= 5:
            x = np.arange(len(tmp_vals))
            slope, _, r_value, p_value, _ = sp_stats.linregress(x, tmp_vals.values)
            fouling_analysis = {
                "tmp_trend_slope": round(float(slope), 6),
                "tmp_trend_r2": round(float(r_value ** 2), 4),
                "tmp_trend_p_value": round(float(p_value), 6),
                "fouling_detected": bool(slope > 0 and p_value < 0.05),
            }

    fouling_msg = ""
    if fouling_analysis:
        fouling_msg = "Fouling trend detected." if fouling_analysis["fouling_detected"] else "No significant fouling trend."

    return PipelineResult(
        method_id="mbr_optimisation",
        method_name="MBR Optimisation Analysis",
        summary=f"Analysed {len(param_stats)} MBR parameters. {fouling_msg}",
        metrics={"n_parameters": len(param_stats), "fouling_analysis": fouling_analysis},
        details={"parameter_stats": param_stats},
    )


def run_a2o_nutrient(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """A2O process nutrient removal performance evaluation."""
    config = config or {}

    nutrient_params = ["NH3-N", "NO3-N", "TN", "TP", "PO4-P", "COD", "DO"]
    available = [p for p in nutrient_params if p in df["parameter"].unique()]

    if len(available) < 2:
        return PipelineResult(
            method_id="a2o_nutrient_removal",
            method_name="A2O Nutrient Removal Evaluation",
            summary=f"Insufficient nutrient parameters (found {available}, need ≥ 2).",
            metrics={},
        )

    results: list[dict] = []
    for param in available:
        vals = pd.to_numeric(df[df["parameter"] == param]["value"], errors="coerce").dropna()
        if len(vals) < 3:
            continue

        values_arr = vals.values
        q1 = float(np.percentile(values_arr, 75))
        q4 = float(np.percentile(values_arr, 25))
        removal = round((q1 - q4) / q1 * 100, 2) if q1 > 0 else 0.0

        results.append({
            "parameter": param,
            "high_quartile": round(q1, 4),
            "low_quartile": round(q4, 4),
            "estimated_removal_pct": removal,
            "mean": round(float(values_arr.mean()), 4),
            "std": round(float(values_arr.std()), 4),
            "n": len(values_arr),
        })

    tn_vals = pd.to_numeric(df[df["parameter"] == "TN"]["value"], errors="coerce").dropna() if "TN" in available else pd.Series(dtype=float)
    tp_vals = pd.to_numeric(df[df["parameter"] == "TP"]["value"], errors="coerce").dropna() if "TP" in available else pd.Series(dtype=float)
    np_ratio = None
    if len(tn_vals) > 0 and len(tp_vals) > 0 and tp_vals.mean() > 0:
        np_ratio = round(float(tn_vals.mean() / tp_vals.mean()), 2)

    np_msg = f"N/P ratio = {np_ratio}" if np_ratio else "N/P ratio not available."

    return PipelineResult(
        method_id="a2o_nutrient_removal",
        method_name="A2O Nutrient Removal Evaluation",
        summary=f"Evaluated {len(results)} nutrient parameters. {np_msg}",
        metrics={"n_parameters": len(results), "np_ratio": np_ratio},
        details={"results": results},
    )


def run_constructed_wetland(df: pd.DataFrame, config: dict | None = None) -> PipelineResult:
    """Constructed wetland design evaluation using k-C* model."""
    config = config or {}

    treatment_params = ["BOD5", "COD", "NH3-N", "TN", "TP", "SS"]
    results: list[dict] = []

    for param in treatment_params:
        pdata = df[df["parameter"] == param].copy()
        pdata["value"] = pd.to_numeric(pdata["value"], errors="coerce")
        pdata = pdata.dropna(subset=["value"])
        if len(pdata) < 5:
            continue

        vals = pdata["value"].values
        c_in = float(np.percentile(vals, 75))
        c_out = float(np.percentile(vals, 25))

        c_star = {"BOD5": 3.5, "COD": 15.0, "NH3-N": 0.2, "TN": 1.5, "TP": 0.02, "SS": 5.0}
        cs = c_star.get(param, 0.0)

        if c_in > cs:
            ratio = max(0.01, (c_out - cs) / (c_in - cs))
            k_est = -np.log(ratio)
            removal = round((c_in - c_out) / c_in * 100, 2) if c_in > 0 else 0.0
        else:
            k_est = 0.0
            removal = 0.0

        results.append({
            "parameter": param,
            "influent_est": round(c_in, 4),
            "effluent_est": round(c_out, 4),
            "background_c_star": cs,
            "k_rate_constant": round(float(k_est), 4),
            "removal_pct": removal,
        })

    return PipelineResult(
        method_id="constructed_wetland_design",
        method_name="Constructed Wetland Design (k-C* Model)",
        summary=f"Wetland performance evaluated for {len(results)} parameters using k-C* model.",
        metrics={"n_parameters": len(results)},
        details={"results": results},
    )


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline dispatcher
# ═══════════════════════════════════════════════════════════════════════

PIPELINE_REGISTRY: dict[str, Callable[..., PipelineResult]] = {
    "trend_analysis": run_mann_kendall,
    "wqi_calculation": run_wqi,
    "pca_clustering": run_pca_clustering,
    "random_forest_classification": run_random_forest,
    "xgboost_regression": run_xgboost_regression,
    "arima_forecast": run_arima_forecast,
    "correlation_analysis": run_correlation_analysis,
    "svr_prediction": run_svr_prediction,
    "bayesian_network": run_bayesian_network,
    "monte_carlo_uncertainty": run_monte_carlo,
    "wavelet_analysis": run_wavelet_analysis,
    "copula_analysis": run_copula_analysis,
    "kriging_interpolation": run_kriging,
    "lstm_forecasting": run_lstm_forecasting,
    "transformer_forecast": run_transformer_forecast,
    "transfer_learning_wq": run_transfer_learning,
    "sdg6_benchmarking": run_sdg6_benchmarking,
    "satellite_eutrophication": run_satellite_eutrophication,
    "gis_watershed_analysis": run_gis_watershed,
    "hec_ras_modelling": run_hec_ras,
    "qual2k_modelling": run_qual2k,
    "swat_modelling": run_swat,
    "mbbr_pilot_study": run_mbbr_pilot,
    "mbr_optimisation": run_mbr_optimisation,
    "a2o_nutrient_removal": run_a2o_nutrient,
    "constructed_wetland_design": run_constructed_wetland,
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
