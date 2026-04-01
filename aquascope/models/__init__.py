"""Predictive models for hydrological forecasting and anomaly detection."""

from aquascope.models.base import BaseHydroModel
from aquascope.models.bayesian import (
    BayesianLinearRegression,
    BayesianPolynomialRegression,
    MetropolisHastings,
    PosteriorResult,
    bayesian_model_comparison,
    dic,
    effective_sample_size,
    gelman_rubin,
)
from aquascope.models.ensemble import (
    AdaptiveEnsemble,
    EnsembleResult,
    StackingEnsemble,
    WeightedEnsemble,
    ensemble_cross_validate,
)
from aquascope.models.transfer import (
    DonorSelector,
    DonorSite,
    TransferLearner,
    TransferResult,
    create_lagged_features,
    spatial_proximity_weight,
)

MODEL_MAP: dict[str, type[BaseHydroModel]] = {}


def _lazy_load() -> None:
    """Populate MODEL_MAP on first access (avoids hard dependency on optional packages)."""
    if MODEL_MAP:
        return
    from aquascope.models.lstm import LSTMModel
    from aquascope.models.ml import IsolationForestModel, RandomForestModel, XGBoostModel
    from aquascope.models.statistical import ARIMAModel, ProphetModel, SPIModel

    MODEL_MAP.update({
        "prophet": ProphetModel,
        "arima": ARIMAModel,
        "spi_drought_index": SPIModel,
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "isolation_forest": IsolationForestModel,
        "lstm": LSTMModel,
    })


def get_model_map() -> dict[str, type[BaseHydroModel]]:
    """Return the mapping of model ID → class (lazy-loaded)."""
    _lazy_load()
    return MODEL_MAP


__all__ = [
    "AdaptiveEnsemble",
    "BaseHydroModel",
    "BayesianLinearRegression",
    "BayesianPolynomialRegression",
    "DonorSelector",
    "DonorSite",
    "EnsembleResult",
    "MetropolisHastings",
    "PosteriorResult",
    "StackingEnsemble",
    "TransferLearner",
    "TransferResult",
    "WeightedEnsemble",
    "bayesian_model_comparison",
    "create_lagged_features",
    "dic",
    "effective_sample_size",
    "ensemble_cross_validate",
    "gelman_rubin",
    "get_model_map",
    "MODEL_MAP",
    "spatial_proximity_weight",
]
