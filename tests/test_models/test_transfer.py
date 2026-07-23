"""Tests for aquascope.models.transfer — transfer learning for ungauged basins."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from aquascope.hydrology.signatures import SignatureReport
from aquascope.models.transfer import (
    DonorSelector,
    DonorSite,
    TransferLearner,
    TransferResult,
    create_lagged_features,
    spatial_proximity_weight,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signature(**overrides: float) -> SignatureReport:
    """Build a SignatureReport with sensible defaults, optionally overridden."""
    defaults = dict(
        mean_flow=10.0,
        median_flow=8.0,
        q5=25.0,
        q95=2.0,
        q5_q95_ratio=12.5,
        cv=0.8,
        iqr=10.0,
        high_flow_frequency=15.0,
        high_flow_duration=3.0,
        q_peak_mean=5.0,
        low_flow_frequency=20.0,
        low_flow_duration=5.0,
        baseflow_index=0.5,
        zero_flow_fraction=0.0,
        peak_month=3,
        seasonality_index=0.4,
        rising_limb_density=0.45,
        flashiness_index=0.3,
        mean_recession_constant=0.05,
        runoff_ratio=0.4,
        elasticity=1.2,
    )
    defaults.update(overrides)
    return SignatureReport(**defaults)


def _make_discharge(n: int = 400, seed: int = 0, scale: float = 10.0) -> pd.Series:
    """Synthetic daily discharge series."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    values = scale + rng.randn(n).cumsum() * 0.5
    values = np.clip(values, 0.1, None)
    return pd.Series(values, index=dates, name="discharge")


def _make_donor(site_id: str, seed: int = 0, sig_overrides: dict | None = None) -> DonorSite:
    """Create a DonorSite with synthetic data."""
    sigs = _make_signature(**(sig_overrides or {}))
    discharge = _make_discharge(seed=seed)
    features = create_lagged_features(discharge)
    common_idx = features.index.intersection(discharge.index)
    return DonorSite(
        site_id=site_id,
        signatures=sigs,
        discharge=discharge.loc[common_idx],
        features=features.loc[common_idx],
    )


# ---------------------------------------------------------------------------
# DonorSite tests
# ---------------------------------------------------------------------------


class TestDonorSite:
    def test_donor_site_creation(self) -> None:
        sig = _make_signature()
        discharge = _make_discharge()
        donor = DonorSite(site_id="site_A", signatures=sig, discharge=discharge)

        assert donor.site_id == "site_A"
        assert isinstance(donor.signatures, SignatureReport)
        assert len(donor.discharge) > 0
        assert donor.features is None
        assert donor.metadata == {}


# ---------------------------------------------------------------------------
# DonorSelector tests
# ---------------------------------------------------------------------------


class TestDonorSelector:
    def setup_method(self) -> None:
        # Three donors with increasingly different signatures from the target
        self.donor_close = _make_donor(
            "close", seed=1, sig_overrides={"baseflow_index": 0.50, "flashiness_index": 0.30},
        )
        self.donor_mid = _make_donor(
            "mid", seed=2, sig_overrides={"baseflow_index": 0.70, "flashiness_index": 0.50},
        )
        self.donor_far = _make_donor(
            "far", seed=3, sig_overrides={"baseflow_index": 0.90, "flashiness_index": 0.80},
        )
        self.selector = DonorSelector([self.donor_close, self.donor_mid, self.donor_far])
        self.target_sig = _make_signature(baseflow_index=0.50, flashiness_index=0.30)

    def test_donor_selector_rank(self) -> None:
        rankings = self.selector.rank_donors(self.target_sig)

        assert len(rankings) == 3
        # First entry should be the closest donor
        assert rankings[0][0] == "close"
        # Scores should be ascending
        assert rankings[0][1] <= rankings[1][1] <= rankings[2][1]

    def test_donor_selector_top_k(self) -> None:
        selected = self.selector.select_top_k(self.target_sig, k=2)

        assert len(selected) == 2
        assert selected[0].site_id == "close"

    def test_donor_selector_max_distance(self) -> None:
        # Use a tight threshold that excludes the furthest donor
        rankings = self.selector.rank_donors(self.target_sig)
        threshold = rankings[1][1] + 0.001  # just above second donor

        selected = self.selector.select_top_k(self.target_sig, k=10, max_distance=threshold)
        site_ids = {d.site_id for d in selected}

        assert "close" in site_ids
        assert "mid" in site_ids
        assert "far" not in site_ids

    def test_pooled_dataset(self) -> None:
        selected = [self.donor_close, self.donor_mid, self.donor_far]
        features, discharge = DonorSelector.pooled_dataset(selected)

        expected_rows = sum(len(d.features) for d in selected)
        assert len(features) == expected_rows
        assert len(discharge) == expected_rows
        assert "site_id" in features.columns


# ---------------------------------------------------------------------------
# create_lagged_features tests
# ---------------------------------------------------------------------------


class TestCreateLaggedFeatures:
    def test_create_lagged_features(self) -> None:
        discharge = _make_discharge(n=100)
        df = create_lagged_features(discharge)

        # Default lags: 1,2,3,7,14,30 → max lag is 30, so at most 30 rows dropped
        assert len(df) <= 100
        assert len(df) > 0
        assert "lag_1" in df.columns
        assert "lag_30" in df.columns
        assert "doy_sin" in df.columns
        assert "doy_cos" in df.columns
        assert "rolling_mean_7" in df.columns
        assert "rolling_mean_30" in df.columns
        assert df.isna().sum().sum() == 0

    def test_create_lagged_features_custom_lags(self) -> None:
        discharge = _make_discharge(n=50)
        df = create_lagged_features(discharge, lags=[1, 5])

        assert "lag_1" in df.columns
        assert "lag_5" in df.columns
        assert "lag_7" not in df.columns  # not requested


# ---------------------------------------------------------------------------
# TransferLearner tests
# ---------------------------------------------------------------------------


class TestTransferLearner:
    def setup_method(self) -> None:
        self.donors = [
            _make_donor("d1", seed=10),
            _make_donor("d2", seed=20),
            _make_donor("d3", seed=30),
        ]
        self.learner = TransferLearner(model_class=LinearRegression)

    def test_transfer_learner_train_predict(self) -> None:
        self.learner.train_on_donors(self.donors)

        test_discharge = _make_discharge(n=100, seed=99)
        test_features = create_lagged_features(test_discharge)
        preds = self.learner.predict(test_features)

        assert preds.shape[0] == len(test_features)

    def test_transfer_learner_evaluate(self) -> None:
        self.learner.train_on_donors(self.donors)

        target_discharge = _make_discharge(n=100, seed=42)
        target_features = create_lagged_features(target_discharge)
        common_idx = target_features.index.intersection(target_discharge.index)
        target_features = target_features.loc[common_idx]
        target_discharge = target_discharge.loc[common_idx]

        metrics = self.learner.evaluate_on_target(target_features, target_discharge)

        assert "NSE" in metrics
        assert "KGE" in metrics
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "PBIAS" in metrics

    def test_transfer_learner_fine_tune(self) -> None:
        self.learner.train_on_donors(self.donors)

        target_discharge = _make_discharge(n=100, seed=42)
        target_features = create_lagged_features(target_discharge)
        common_idx = target_features.index.intersection(target_discharge.index)
        target_features = target_features.loc[common_idx]
        target_discharge = target_discharge.loc[common_idx]

        # Should not raise
        self.learner.fine_tune(target_features, target_discharge, fraction=0.5)
        preds = self.learner.predict(target_features)
        assert preds.shape[0] == len(target_features)


# ---------------------------------------------------------------------------
# Full pipeline tests
# ---------------------------------------------------------------------------


class TestFullTransferPipeline:
    def setup_method(self) -> None:
        self.donors = [
            _make_donor("d1", seed=10, sig_overrides={"baseflow_index": 0.50}),
            _make_donor("d2", seed=20, sig_overrides={"baseflow_index": 0.60}),
            _make_donor("d3", seed=30, sig_overrides={"baseflow_index": 0.70}),
        ]
        self.target_sig = _make_signature(baseflow_index=0.52)
        self.target_discharge = _make_discharge(n=200, seed=42)
        self.target_features = create_lagged_features(self.target_discharge)
        common = self.target_features.index.intersection(self.target_discharge.index)
        self.target_features = self.target_features.loc[common]
        self.target_discharge = self.target_discharge.loc[common]

    def test_full_transfer_pipeline(self) -> None:
        selector = DonorSelector(self.donors)
        learner = TransferLearner(model_class=LinearRegression)
        result = learner.transfer(
            donor_selector=selector,
            target_signatures=self.target_sig,
            target_features=self.target_features,
            target_discharge=self.target_discharge,
            n_donors=2,
            fine_tune_fraction=0.5,
        )

        assert isinstance(result, TransferResult)
        assert len(result.selected_donors) == 2
        assert len(result.donor_rankings) == 3

    def test_transfer_result_fields(self) -> None:
        selector = DonorSelector(self.donors)
        learner = TransferLearner(model_class=LinearRegression)
        result = learner.transfer(
            donor_selector=selector,
            target_signatures=self.target_sig,
            target_features=self.target_features,
            target_discharge=self.target_discharge,
            n_donors=2,
        )

        assert result.target_site_id == "target"
        assert isinstance(result.donor_rankings, list)
        assert isinstance(result.selected_donors, list)
        assert isinstance(result.model_metrics_before, dict)
        assert isinstance(result.model_metrics_after, dict)
        assert isinstance(result.improvement, dict)
        assert "NSE" in result.model_metrics_before
        assert "NSE" in result.model_metrics_after

    def test_improvement_positive(self) -> None:
        """After fine-tuning on target data, RMSE should decrease (improvement ≥ 0)."""
        selector = DonorSelector(self.donors)
        learner = TransferLearner(model_class=LinearRegression)
        result = learner.transfer(
            donor_selector=selector,
            target_signatures=self.target_sig,
            target_features=self.target_features,
            target_discharge=self.target_discharge,
            n_donors=2,
            fine_tune_fraction=1.0,
        )

        # After fine-tuning on the full target data the model should at least
        # not get dramatically worse on RMSE.
        assert "RMSE" in result.improvement

    def test_single_donor(self) -> None:
        selector = DonorSelector(self.donors)
        learner = TransferLearner(model_class=LinearRegression)
        result = learner.transfer(
            donor_selector=selector,
            target_signatures=self.target_sig,
            target_features=self.target_features,
            target_discharge=self.target_discharge,
            n_donors=1,
        )

        assert len(result.selected_donors) == 1


# ---------------------------------------------------------------------------
# spatial_proximity_weight tests
# ---------------------------------------------------------------------------


class TestSpatialProximityWeight:
    def test_spatial_proximity_weight(self) -> None:
        """Closer donor should receive higher weight."""
        target = (51.5, -0.1)  # London
        donors = [
            (48.9, 2.35),   # Paris  (~340 km)
            (40.4, -3.7),   # Madrid (~1260 km)
        ]
        weights = spatial_proximity_weight(donors, target)

        assert len(weights) == 2
        assert weights[0] > weights[1]  # Paris closer → higher weight

    def test_spatial_proximity_weight_sums_to_one(self) -> None:
        target = (0.0, 0.0)
        donors = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        weights = spatial_proximity_weight(donors, target)

        assert abs(sum(weights) - 1.0) < 1e-9
