"""Tests for the pipeline / model builder module."""

import numpy as np
import pandas as pd
import pytest

from aquascope.pipelines.model_builder import (
    PipelineResult,
    list_available_pipelines,
    run_a2o_nutrient,
    run_bayesian_network,
    run_constructed_wetland,
    run_copula_analysis,
    run_correlation_analysis,
    run_gis_watershed,
    run_hec_ras,
    run_kriging,
    run_lstm_forecasting,
    run_mbbr_pilot,
    run_mbr_optimisation,
    run_monte_carlo,
    run_pca_clustering,
    run_pipeline,
    run_qual2k,
    run_random_forest,
    run_satellite_eutrophication,
    run_sdg6_benchmarking,
    run_svr_prediction,
    run_swat,
    run_transfer_learning,
    run_transformer_forecast,
    run_wavelet_analysis,
    run_wqi,
)


def _make_multivariate_df(n_per_param: int = 60) -> pd.DataFrame:
    """Create wide-ish water data with correlated parameters."""
    rng = np.random.default_rng(42)
    params = ["DO", "BOD5", "COD", "NH3-N", "SS"]
    rows = []
    for i in range(n_per_param):
        base = rng.normal(5.0, 1.0)
        for p in params:
            rows.append({
                "source": "taiwan_moenv",
                "station_id": f"ST{i % 5:03d}",
                "parameter": p,
                "value": round(base + rng.normal(0, 0.5), 2),
                "unit": "mg/L",
                "sample_datetime": f"2022-{(i % 12) + 1:02d}-15T10:00:00",
            })
    return pd.DataFrame(rows)


def _make_wqi_df() -> pd.DataFrame:
    """Create data suitable for WQI / RPI computation."""
    rows = []
    for d in range(30):
        for param, val in [("DO", 6.5), ("BOD5", 2.0), ("SS", 15.0), ("NH3-N", 0.3)]:
            rows.append({
                "station_id": "ST001",
                "parameter": param,
                "value": val + (d * 0.01),
                "sample_datetime": f"2024-01-{d + 1:02d}T10:00:00",
            })
    return pd.DataFrame(rows)


class TestPipelineRegistry:
    def test_list_pipelines(self):
        pipelines = list_available_pipelines()
        assert "trend_analysis" in pipelines
        assert "wqi_calculation" in pipelines
        assert "pca_clustering" in pipelines
        assert "correlation_analysis" in pipelines
        assert len(pipelines) >= 26

    def test_run_unknown_pipeline(self):
        with pytest.raises(ValueError, match="No pipeline"):
            run_pipeline("nonexistent_pipeline", pd.DataFrame())


class TestCorrelationAnalysis:
    def test_returns_result(self):
        df = _make_multivariate_df()
        result = run_correlation_analysis(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "correlation_analysis"

    def test_finds_parameters(self):
        df = _make_multivariate_df()
        result = run_correlation_analysis(df)
        assert result.metrics.get("n_parameters", 0) >= 2


class TestPCAClustering:
    def test_returns_result(self):
        df = _make_multivariate_df()
        result = run_pca_clustering(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "pca_clustering"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_pca_clustering(df)
        assert "Insufficient" in result.summary

    def test_explained_variance(self):
        df = _make_multivariate_df()
        result = run_pca_clustering(df)
        if result.metrics:
            ev = result.metrics.get("explained_variance", [])
            assert sum(ev) > 0


class TestWQI:
    def test_computes_rpi(self):
        df = _make_wqi_df()
        result = run_wqi(df)
        assert isinstance(result, PipelineResult)
        assert result.metrics.get("n_computed", 0) > 0

    def test_rpi_categories(self):
        df = _make_wqi_df()
        result = run_wqi(df)
        dist = result.metrics.get("category_distribution", {})
        assert len(dist) > 0


class TestRandomForest:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_random_forest(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "random_forest_classification"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO", "BOD5"],
            "value": [5.0, 2.0],
            "sample_datetime": ["2024-01-01T00:00:00", "2024-01-01T00:00:00"],
        })
        result = run_random_forest(df)
        assert "Insufficient" in result.summary


class TestDispatcher:
    def test_dispatch_correlation(self):
        df = _make_multivariate_df()
        result = run_pipeline("correlation_analysis", df)
        assert result.method_id == "correlation_analysis"

    def test_dispatch_wqi(self):
        df = _make_wqi_df()
        result = run_pipeline("wqi_calculation", df)
        assert result.method_id == "wqi_calculation"


class TestSVRPrediction:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_svr_prediction(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "svr_prediction"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_svr_prediction(df)
        assert isinstance(result, PipelineResult)


class TestBayesianNetwork:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_bayesian_network(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "bayesian_network"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_bayesian_network(df)
        assert isinstance(result, PipelineResult)


class TestMonteCarlo:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_monte_carlo(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "monte_carlo_uncertainty"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_monte_carlo(df)
        assert isinstance(result, PipelineResult)


class TestWaveletAnalysis:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_wavelet_analysis(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "wavelet_analysis"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_wavelet_analysis(df)
        assert isinstance(result, PipelineResult)


class TestCopulaAnalysis:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_copula_analysis(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "copula_analysis"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_copula_analysis(df)
        assert isinstance(result, PipelineResult)


class TestKriging:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        rng = np.random.default_rng(99)
        df["latitude"] = rng.uniform(24.0, 25.5, len(df))
        df["longitude"] = rng.uniform(120.0, 121.5, len(df))
        result = run_kriging(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "kriging_interpolation"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
            "latitude": [24.5],
            "longitude": [120.5],
        })
        result = run_kriging(df)
        assert isinstance(result, PipelineResult)


class TestLSTMForecasting:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_lstm_forecasting(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "lstm_forecasting"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_lstm_forecasting(df)
        assert isinstance(result, PipelineResult)


class TestTransformerForecast:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_transformer_forecast(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "transformer_forecast"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_transformer_forecast(df)
        assert isinstance(result, PipelineResult)


class TestTransferLearning:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_transfer_learning(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "transfer_learning_wq"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_transfer_learning(df)
        assert isinstance(result, PipelineResult)


class TestSDG6Benchmarking:
    def test_returns_result(self):
        df = _make_multivariate_df()
        result = run_sdg6_benchmarking(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "sdg6_benchmarking"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_sdg6_benchmarking(df)
        assert isinstance(result, PipelineResult)


class TestSatelliteEutrophication:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_satellite_eutrophication(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "satellite_eutrophication"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_satellite_eutrophication(df)
        assert isinstance(result, PipelineResult)


class TestGISWatershed:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_gis_watershed(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "gis_watershed_analysis"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_gis_watershed(df)
        assert isinstance(result, PipelineResult)


class TestHECRAS:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_hec_ras(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "hec_ras_modelling"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_hec_ras(df)
        assert isinstance(result, PipelineResult)


class TestQUAL2K:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_qual2k(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "qual2k_modelling"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_qual2k(df)
        assert isinstance(result, PipelineResult)


class TestSWAT:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_swat(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "swat_modelling"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_swat(df)
        assert isinstance(result, PipelineResult)


class TestMBBRPilot:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_mbbr_pilot(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "mbbr_pilot_study"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_mbbr_pilot(df)
        assert isinstance(result, PipelineResult)


class TestMBROptimisation:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_mbr_optimisation(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "mbr_optimisation"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_mbr_optimisation(df)
        assert isinstance(result, PipelineResult)


class TestA2ONutrient:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_a2o_nutrient(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "a2o_nutrient_removal"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_a2o_nutrient(df)
        assert isinstance(result, PipelineResult)


class TestConstructedWetland:
    def test_returns_result(self):
        df = _make_multivariate_df(n_per_param=100)
        result = run_constructed_wetland(df)
        assert isinstance(result, PipelineResult)
        assert result.method_id == "constructed_wetland_design"

    def test_insufficient_data(self):
        df = pd.DataFrame({
            "parameter": ["DO"],
            "value": [5.0],
            "station_id": ["ST001"],
            "sample_datetime": ["2024-01-01T00:00:00"],
        })
        result = run_constructed_wetland(df)
        assert isinstance(result, PipelineResult)
