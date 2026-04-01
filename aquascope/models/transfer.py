"""Transfer learning for ungauged or data-poor basins.

This module implements a transfer-learning workflow that leverages hydrological
similarity to predict streamflow at sites with little or no observed data.

Workflow
--------
1. Characterise donor catchments with hydrological signatures.
2. Rank donors by similarity to the target site.
3. Train a base model on pooled donor data.
4. (Optional) Fine-tune on limited target-site observations.
5. Report improvement metrics.

The approach is model-agnostic: any estimator with ``fit()`` and ``predict()``
methods (scikit-learn compatible or :class:`~aquascope.models.base.BaseHydroModel`
sub-classes) can be used.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aquascope.hydrology.signatures import SignatureReport, similarity_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DonorSite:
    """A potential donor site for transfer learning.

    Attributes:
        site_id: Unique identifier for the donor catchment.
        signatures: :class:`~aquascope.hydrology.signatures.SignatureReport`
            characterising the donor's flow regime.
        discharge: Daily discharge series with a :class:`~pandas.DatetimeIndex`.
        features: Optional predictor features (e.g. precipitation, temperature)
            aligned with *discharge*.
        metadata: Arbitrary extra information (area, coordinates, …).
    """

    site_id: str
    signatures: Any  # SignatureReport
    discharge: pd.Series
    features: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransferResult:
    """Result of a transfer-learning experiment.

    Attributes:
        target_site_id: Identifier of the target (ungauged) site.
        donor_rankings: ``(site_id, similarity_score)`` pairs sorted by
            ascending score (best first).
        selected_donors: IDs of the donors actually used for training.
        model_metrics_before: Performance of the donor-trained model on
            target data *before* fine-tuning.
        model_metrics_after: Performance *after* fine-tuning on target data.
        improvement: Per-metric improvement (after − before for NSE/KGE,
            before − after for error metrics).
    """

    target_site_id: str
    donor_rankings: list[tuple[str, float]]
    selected_donors: list[str]
    model_metrics_before: dict[str, float]
    model_metrics_after: dict[str, float]
    improvement: dict[str, float]


# ---------------------------------------------------------------------------
# Donor selection
# ---------------------------------------------------------------------------


class DonorSelector:
    """Select donor catchments for transfer learning based on hydrological similarity.

    Parameters:
        donors: List of potential donor sites with pre-computed signatures.
        weights: Optional weights for signature comparison passed to
            :func:`~aquascope.hydrology.signatures.similarity_score`.
            Default emphasises BFI, flashiness, and seasonality.
    """

    def __init__(
        self,
        donors: list[DonorSite],
        weights: dict[str, float] | None = None,
    ) -> None:
        self.donors = donors
        self.weights = weights

    def rank_donors(self, target_signatures: SignatureReport) -> list[tuple[str, float]]:
        """Rank all donor sites by similarity to the target.

        Uses :func:`~aquascope.hydrology.signatures.similarity_score` with
        the configured weights.

        Parameters:
            target_signatures: Signature report of the target catchment.

        Returns:
            List of ``(site_id, score)`` sorted by ascending score (0 = best).
        """
        rankings: list[tuple[str, float]] = []
        for donor in self.donors:
            score = similarity_score(donor.signatures, target_signatures, weights=self.weights)
            rankings.append((donor.site_id, score))
        rankings.sort(key=lambda x: x[1])
        return rankings

    def select_top_k(
        self,
        target_signatures: SignatureReport,
        k: int = 3,
        max_distance: float | None = None,
    ) -> list[DonorSite]:
        """Select the *k* most similar donors.

        Parameters:
            target_signatures: Signature report of the target catchment.
            k: Maximum number of donors to return.
            max_distance: If given, exclude donors with similarity score
                exceeding this threshold.

        Returns:
            List of up to *k* :class:`DonorSite` objects, best first.
        """
        rankings = self.rank_donors(target_signatures)
        donor_map = {d.site_id: d for d in self.donors}
        selected: list[DonorSite] = []
        for site_id, score in rankings:
            if max_distance is not None and score > max_distance:
                continue
            selected.append(donor_map[site_id])
            if len(selected) >= k:
                break
        return selected

    @staticmethod
    def pooled_dataset(selected_donors: list[DonorSite]) -> tuple[pd.DataFrame, pd.Series]:
        """Pool training data from multiple donors into a single dataset.

        For each donor the method uses pre-computed ``features`` when
        available, otherwise it creates lagged-discharge features via
        :func:`create_lagged_features`.

        A ``site_id`` column is appended for provenance tracking.

        Parameters:
            selected_donors: Donors to pool.

        Returns:
            Tuple of ``(features_df, discharge_series)`` with aligned indices.
        """
        all_features: list[pd.DataFrame] = []
        all_discharge: list[pd.Series] = []

        for donor in selected_donors:
            if donor.features is not None:
                feats = donor.features.copy()
            else:
                feats = create_lagged_features(donor.discharge)

            # Align features and discharge to common index
            common_idx = feats.index.intersection(donor.discharge.index)
            feats = feats.loc[common_idx]
            discharge = donor.discharge.loc[common_idx]

            feats = feats.copy()
            feats["site_id"] = donor.site_id
            all_features.append(feats)
            all_discharge.append(discharge)

        features_df = pd.concat(all_features, axis=0)
        discharge_series = pd.concat(all_discharge, axis=0)
        return features_df, discharge_series


# ---------------------------------------------------------------------------
# Transfer learner
# ---------------------------------------------------------------------------


class TransferLearner:
    """Transfer learning workflow for ungauged or data-poor basins.

    Workflow:
        1. Select similar donor sites using hydrological signatures.
        2. Train a base model on donor-site data.
        3. Evaluate on target site (if some target data is available).
        4. Fine-tune on limited target data (if available).
        5. Report improvement metrics.

    Parameters:
        model_class: A model class with ``fit()`` and ``predict()`` methods.
            Can be any scikit-learn-compatible estimator or a
            :class:`~aquascope.models.base.BaseHydroModel` sub-class.
        model_kwargs: Keyword arguments forwarded to *model_class* on
            instantiation.
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self._model: Any = None
        self._donor_features: pd.DataFrame | None = None
        self._donor_discharge: pd.Series | None = None

    # -- training / prediction ------------------------------------------

    def train_on_donors(self, donors: list[DonorSite]) -> None:
        """Train a fresh model on pooled donor data.

        Creates feature matrices from each donor (using pre-computed features
        when available, otherwise lagged-discharge features) and fits the
        model.

        Parameters:
            donors: Donor sites to train on.
        """
        features, discharge = DonorSelector.pooled_dataset(donors)

        # Drop the categorical site_id column before fitting
        numeric_features = features.drop(columns=["site_id"], errors="ignore")

        self._model = self.model_class(**self.model_kwargs)
        self._model.fit(numeric_features.values, discharge.values)
        self._donor_features = numeric_features
        self._donor_discharge = discharge
        logger.info("Trained model on %d donor samples from %d sites.", len(discharge), len(donors))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict using the donor-trained model.

        Parameters:
            features: Predictor features with the same columns used during
                training.

        Returns:
            Array of predicted values.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call train_on_donors() first.")
        return np.asarray(self._model.predict(features.values))

    def evaluate_on_target(
        self,
        target_features: pd.DataFrame,
        target_discharge: pd.Series,
    ) -> dict[str, float]:
        """Evaluate the current model on the target site.

        Parameters:
            target_features: Predictor features for the target site.
            target_discharge: Observed discharge at the target site.

        Returns:
            Dictionary of metrics: ``NSE``, ``KGE``, ``RMSE``, ``MAE``,
            ``PBIAS``.
        """
        y_pred = self.predict(target_features)
        y_true = target_discharge.values
        return _compute_transfer_metrics(y_true, y_pred)

    def fine_tune(
        self,
        target_features: pd.DataFrame,
        target_discharge: pd.Series,
        fraction: float = 1.0,
    ) -> None:
        """Fine-tune the model on (limited) target-site data.

        Parameters:
            target_features: Predictor features for the target site.
            target_discharge: Observed discharge at the target site.
            fraction: Fraction of target data to use (0–1).  Values < 1
                simulate a limited-data scenario by using only the first
                *fraction* of the data.

        Strategy:
            * Combine donor training data with target data (target samples
              receive 2× weight via repetition) so that target-site patterns
              are emphasised while donor information is preserved.
        """
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call train_on_donors() first.")

        n_use = max(1, int(len(target_features) * fraction))
        tf = target_features.iloc[:n_use]
        td = target_discharge.iloc[:n_use]

        # Combine donor and target data, doubling target rows for emphasis
        if self._donor_features is not None and self._donor_discharge is not None:
            combined_features = pd.concat([self._donor_features, tf, tf], axis=0, ignore_index=True)
            combined_discharge = pd.concat([self._donor_discharge, td, td], axis=0, ignore_index=True)
        else:
            combined_features = pd.concat([tf, tf], axis=0, ignore_index=True)
            combined_discharge = pd.concat([td, td], axis=0, ignore_index=True)

        self._model = self.model_class(**self.model_kwargs)
        self._model.fit(combined_features.values, combined_discharge.values)
        logger.info(
            "Fine-tuned model on %d target samples (fraction=%.2f) + donor data.",
            n_use,
            fraction,
        )

    # -- full pipeline --------------------------------------------------

    def transfer(
        self,
        donor_selector: DonorSelector,
        target_signatures: SignatureReport,
        target_features: pd.DataFrame | None = None,
        target_discharge: pd.Series | None = None,
        n_donors: int = 3,
        fine_tune_fraction: float = 0.5,
    ) -> TransferResult:
        """Execute the full transfer-learning pipeline.

        Steps:
            1. Select top *n_donors* using *donor_selector*.
            2. Train on pooled donor data.
            3. If target data is available, evaluate before fine-tuning.
            4. Fine-tune on target data (using *fine_tune_fraction*).
            5. Evaluate after fine-tuning.
            6. Return :class:`TransferResult` with all metrics.

        Parameters:
            donor_selector: Pre-configured :class:`DonorSelector`.
            target_signatures: Signature report for the target catchment.
            target_features: Optional predictor features for the target.
            target_discharge: Optional observed discharge at the target.
            n_donors: Number of donor sites to use.
            fine_tune_fraction: Fraction of target data for fine-tuning.

        Returns:
            :class:`TransferResult` summarising the experiment.
        """
        # Step 1 — select donors
        rankings = donor_selector.rank_donors(target_signatures)
        selected = donor_selector.select_top_k(target_signatures, k=n_donors)
        selected_ids = [d.site_id for d in selected]

        # Step 2 — train on donors
        self.train_on_donors(selected)

        metrics_before: dict[str, float] = {}
        metrics_after: dict[str, float] = {}

        # Prepare target features if discharge is given but features are not
        if target_discharge is not None and target_features is None:
            target_features = create_lagged_features(target_discharge)
            common_idx = target_features.index.intersection(target_discharge.index)
            target_features = target_features.loc[common_idx]
            target_discharge = target_discharge.loc[common_idx]

        # Steps 3–5 — evaluate → fine-tune → evaluate
        if target_features is not None and target_discharge is not None:
            metrics_before = self.evaluate_on_target(target_features, target_discharge)
            self.fine_tune(target_features, target_discharge, fraction=fine_tune_fraction)
            metrics_after = self.evaluate_on_target(target_features, target_discharge)

        improvement = _compute_improvement(metrics_before, metrics_after)

        return TransferResult(
            target_site_id="target",
            donor_rankings=rankings,
            selected_donors=selected_ids,
            model_metrics_before=metrics_before,
            model_metrics_after=metrics_after,
            improvement=improvement,
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_lagged_features(
    discharge: pd.Series,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create lagged discharge features for autoregressive modelling.

    Parameters:
        discharge: Daily discharge series with a :class:`~pandas.DatetimeIndex`.
        lags: Lag steps (in days) to include.  Defaults to
            ``[1, 2, 3, 7, 14, 30]``.

    Returns:
        DataFrame with columns for each lag, sin/cos day-of-year encoding,
        and 7-day / 30-day rolling means.  Rows with NaN from lagging are
        dropped.
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14, 30]

    df = pd.DataFrame(index=discharge.index)
    for lag in lags:
        df[f"lag_{lag}"] = discharge.shift(lag)

    # Cyclic day-of-year encoding
    doy = discharge.index.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Rolling statistics
    df["rolling_mean_7"] = discharge.rolling(window=7, min_periods=1).mean()
    df["rolling_mean_30"] = discharge.rolling(window=30, min_periods=1).mean()

    df = df.dropna()
    return df


def spatial_proximity_weight(
    donor_coords: list[tuple[float, float]],
    target_coord: tuple[float, float],
) -> list[float]:
    """Compute inverse-distance weights based on geographic proximity.

    Uses the Haversine formula to compute great-circle distances between
    each donor and the target site.

    Parameters:
        donor_coords: ``(latitude, longitude)`` pairs for each donor.
        target_coord: ``(latitude, longitude)`` of the target site.

    Returns:
        Normalised weights (sum = 1) where closer donors receive higher
        weight.
    """
    distances: list[float] = []
    for lat, lon in donor_coords:
        d = _haversine(lat, lon, target_coord[0], target_coord[1])
        distances.append(d)

    # Inverse distance (add small epsilon to avoid division by zero)
    eps = 1e-6
    inv = [1.0 / (d + eps) for d in distances]
    total = sum(inv)
    return [w / total for w in inv]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points (Haversine)."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _compute_transfer_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard hydrological transfer-learning metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {}

    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    # Kling-Gupta Efficiency
    if len(y_true) > 1 and y_true.std() > 0 and y_true.mean() != 0:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
        alpha = float(y_pred.std() / y_true.std())
        beta = float(y_pred.mean() / y_true.mean())
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    else:
        kge = float("nan")

    # Percent bias
    pbias = float(np.sum(y_pred - y_true) / np.sum(y_true) * 100) if np.sum(y_true) != 0 else float("nan")

    return {
        "NSE": round(float(nse), 4),
        "KGE": round(float(kge), 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "PBIAS": round(pbias, 4),
    }


def _compute_improvement(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    """Compute metric improvements (positive = better).

    For NSE and KGE higher is better so improvement = after − before.
    For RMSE, MAE, PBIAS lower magnitude is better so improvement = |before| − |after|.
    """
    improvement: dict[str, float] = {}
    higher_is_better = {"NSE", "KGE"}
    for key in before:
        if key not in after:
            continue
        if key in higher_is_better:
            improvement[key] = round(after[key] - before[key], 4)
        else:
            improvement[key] = round(abs(before[key]) - abs(after[key]), 4)
    return improvement
