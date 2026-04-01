"""
Model recommender — expert-curated decision matrix mapping challenge types to models.

Works alongside the existing research-methodology recommender by providing
model-specific recommendations for predictive / forecasting tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CHALLENGE_MODEL_MATRIX: dict[str, dict[str, list[str]]] = {
    "flood": {
        "forecast": ["prophet", "lstm", "random_forest"],
        "anomaly": ["isolation_forest", "xgboost"],
        "analysis": ["arima", "random_forest"],
    },
    "drought": {
        "forecast": ["prophet", "arima", "lstm"],
        "index": ["spi_drought_index"],
        "analysis": ["random_forest", "xgboost"],
    },
    "water_quality": {
        "forecast": ["prophet", "random_forest", "xgboost"],
        "anomaly": ["isolation_forest"],
        "analysis": ["xgboost", "random_forest", "arima"],
    },
}


@dataclass
class ModelRecommendation:
    """A single model recommendation."""

    model_id: str
    rank: int
    challenge_type: str
    task_type: str
    rationale: str


class ModelRecommender:
    """Recommend predictive models for a given challenge type and task.

    Example
    -------
    >>> rec = ModelRecommender()
    >>> picks = rec.recommend("flood", "forecast")
    >>> [p.model_id for p in picks]
    ['prophet', 'lstm', 'random_forest']
    """

    def recommend(
        self,
        challenge_type: str,
        task_type: str = "forecast",
        top_k: int = 3,
    ) -> list[ModelRecommendation]:
        """Return ranked model recommendations.

        Parameters
        ----------
        challenge_type : str
            One of ``flood``, ``drought``, ``water_quality``.
        task_type : str
            Task within the challenge: ``forecast``, ``anomaly``, ``analysis``, ``index``.
        top_k : int
            Maximum number of recommendations to return.

        Returns
        -------
        list[ModelRecommendation]
        """
        tasks = CHALLENGE_MODEL_MATRIX.get(challenge_type, {})
        model_ids = tasks.get(task_type, [])

        if not model_ids:
            # Fallback: combine all tasks for this challenge type
            model_ids = []
            for models in tasks.values():
                for m in models:
                    if m not in model_ids:
                        model_ids.append(m)

        results: list[ModelRecommendation] = []
        for i, model_id in enumerate(model_ids[:top_k]):
            results.append(ModelRecommendation(
                model_id=model_id,
                rank=i + 1,
                challenge_type=challenge_type,
                task_type=task_type,
                rationale=_rationale(model_id, challenge_type, task_type),
            ))
        return results


def _rationale(model_id: str, challenge: str, task: str) -> str:
    """Generate a brief rationale for the recommendation."""
    rationales: dict[str, str] = {
        "prophet": "Captures seasonality and trends with uncertainty intervals; handles missing data well.",
        "arima": "Classical time-series model; fast and interpretable for stationary data.",
        "lstm": "Deep-learning sequence model; captures complex non-linear temporal patterns.",
        "random_forest": "Robust ensemble method; handles mixed features and provides feature importance.",
        "xgboost": "Gradient-boosted trees; top performance on tabular/structured data.",
        "isolation_forest": "Unsupervised anomaly detector; effective for finding contamination events.",
        "spi_drought_index": "WMO-standard drought index; direct precipitation-based drought monitoring.",
    }
    base = rationales.get(model_id, "Suitable model for this task.")
    return f"[{challenge}/{task}] {base}"
