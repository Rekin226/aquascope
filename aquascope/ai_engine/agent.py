"""
HydroAgent — end-to-end orchestrator that connects NL parsing, data loading,
model recommendation, and challenge execution into a single workflow.

Usage
-----
>>> from aquascope.ai_engine.agent import HydroAgent
>>> agent = HydroAgent()
>>> result = agent.solve("Forecast flooding at lat 13.5, lon 2.1 for 14 days")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from aquascope.ai_engine.model_recommender import ModelRecommender
from aquascope.ai_engine.planner import ChallengePlanner, ChallengeSpec

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of an end-to-end agent run."""

    challenge_spec: ChallengeSpec
    model_used: str = ""
    forecast: pd.DataFrame | None = None
    risk_assessment: dict | None = None
    status: dict | None = None
    anomalies: pd.DataFrame | None = None
    explanation: str = ""
    steps: list[str] = field(default_factory=list)


class HydroAgent:
    """Orchestrator that parses a natural-language query and executes the corresponding challenge.

    Parameters
    ----------
    default_model : str | None
        Override the auto-recommended model.  If ``None``, the
        ``ModelRecommender`` picks the top model for the challenge type.

    Example
    -------
    >>> agent = HydroAgent()
    >>> result = agent.solve("Drought monitoring near Sahel at lat 15, lon 0")
    >>> print(result.explanation)
    """

    def __init__(self, default_model: str | None = None):
        self.planner = ChallengePlanner()
        self.recommender = ModelRecommender()
        self.default_model = default_model

    def solve(
        self,
        query: str,
        data: pd.DataFrame | None = None,
        extra_data: dict[str, pd.DataFrame] | None = None,
    ) -> AgentResult:
        """Parse *query*, load data, pick model, and run the challenge.

        Parameters
        ----------
        query : str
            Natural-language challenge description.
        data : pd.DataFrame | None
            Primary data (discharge / precipitation). If ``None`` and
            coordinates are detected, the agent tries to fetch data from
            Open-Meteo.
        extra_data : dict[str, pd.DataFrame] | None
            Additional named DataFrames (e.g. ``{"et": et_df}``).

        Returns
        -------
        AgentResult
        """
        spec = self.planner.parse(query)
        result = AgentResult(challenge_spec=spec)
        result.steps.append(f"Parsed query → challenge_type={spec.challenge_type}")

        # Pick model
        model_id = self.default_model
        if not model_id:
            recs = self.recommender.recommend(spec.challenge_type, "forecast")
            model_id = recs[0].model_id if recs else "prophet"
        result.model_used = model_id
        result.steps.append(f"Selected model: {model_id}")

        # Load data if not provided
        if data is None and spec.latitude is not None and spec.longitude is not None:
            data = self._fetch_data(spec)
            result.steps.append(
                f"Fetched data from Open-Meteo ({len(data)} rows)" if data is not None else "Data fetch failed"
            )

        if data is None or data.empty:
            result.explanation = "No data available. Provide a DataFrame or coordinates."
            return result

        # Dispatch to challenge handler
        try:
            if spec.challenge_type == "flood":
                self._run_flood(result, spec, data, model_id)
            elif spec.challenge_type == "drought":
                self._run_drought(result, spec, data, extra_data)
            elif spec.challenge_type == "water_quality":
                self._run_quality(result, spec, data, extra_data)
            else:
                result.explanation = f"Unknown challenge type: {spec.challenge_type}"
        except Exception as e:
            logger.exception("Agent execution error")
            result.explanation = f"Error: {e}"

        return result

    def explain(self, result: AgentResult) -> str:
        """Produce a human-readable summary of an ``AgentResult``."""
        lines = [f"## AquaScope Agent Report: {result.challenge_spec.challenge_type.title()} Challenge\n"]
        lines.append(f"**Query:** {result.challenge_spec.raw_query}")
        lines.append(f"**Model:** {result.model_used}")
        lines.append(f"**Steps:** {' → '.join(result.steps)}\n")

        if result.risk_assessment:
            lines.append("### Risk Assessment")
            for k, v in result.risk_assessment.items():
                lines.append(f"  - **{k}**: {v}")

        if result.status:
            lines.append("\n### Status")
            for k, v in result.status.items():
                lines.append(f"  - **{k}**: {v}")

        if result.forecast is not None:
            lines.append(f"\n### Forecast ({len(result.forecast)} days)")
            lines.append(result.forecast.head(7).to_string())

        if result.anomalies is not None and not result.anomalies.empty:
            lines.append(f"\n### Anomalies ({len(result.anomalies)} found)")
            lines.append(result.anomalies.head(10).to_string())

        if result.explanation:
            lines.append(f"\n**Notes:** {result.explanation}")

        return "\n".join(lines)

    # ── Private helpers ──────────────────────────────────────────────

    def _fetch_data(self, spec: ChallengeSpec) -> pd.DataFrame | None:
        """Try to fetch data from Open-Meteo for the given coordinates."""
        try:
            from aquascope.collectors.openmeteo import OpenMeteoCollector

            if spec.challenge_type == "flood":
                collector = OpenMeteoCollector(mode="flood")
                records = collector.collect(
                    latitude=spec.latitude,
                    longitude=spec.longitude,
                    daily=["river_discharge"],
                )
            else:
                collector = OpenMeteoCollector(mode="weather")
                records = collector.collect(
                    latitude=spec.latitude,
                    longitude=spec.longitude,
                    start_date="2020-01-01",
                    end_date="2024-12-31",
                    daily=["precipitation_sum"],
                )

            if not records:
                return None

            rows = [{"datetime": r.sample_datetime, "value": r.value} for r in records]
            df = pd.DataFrame(rows).set_index("datetime").sort_index()
            df.index = pd.DatetimeIndex(df.index)
            return df
        except Exception as e:
            logger.warning("Open-Meteo fetch failed: %s", e)
            return None

    def _run_flood(self, result: AgentResult, spec: ChallengeSpec, data: pd.DataFrame, model_id: str) -> None:
        from aquascope.challenges.flood import FloodChallenge

        challenge = FloodChallenge(
            lat=spec.latitude, lon=spec.longitude, name=spec.location_name,
        )
        challenge.load_dataframe(data)
        challenge.fit(model=model_id)
        result.forecast = challenge.forecast(days=spec.forecast_days)
        result.risk_assessment = challenge.assess_risk(result.forecast)
        result.steps.append("Ran flood forecast + risk assessment")
        result.explanation = result.risk_assessment.get("description", "")

    def _run_drought(
        self, result: AgentResult, spec: ChallengeSpec, data: pd.DataFrame,
        extra: dict[str, pd.DataFrame] | None,
    ) -> None:
        from aquascope.challenges.drought import DroughtChallenge

        challenge = DroughtChallenge(
            lat=spec.latitude or 0, lon=spec.longitude or 0, name=spec.location_name,
        )
        et_df = extra.get("et") if extra else None
        challenge.load_dataframe(data, et_df=et_df)
        result.status = challenge.current_status()
        result.forecast = challenge.forecast_precipitation(days=spec.forecast_days)
        result.steps.append("Computed SPI + drought status")
        result.explanation = result.status.get("overall", "")

    def _run_quality(
        self, result: AgentResult, spec: ChallengeSpec, data: pd.DataFrame,
        extra: dict[str, pd.DataFrame] | None,
    ) -> None:
        from aquascope.challenges.quality import WaterQualityChallenge

        wq = WaterQualityChallenge(site_id=spec.location_name or "unknown")

        var_data: dict[str, pd.DataFrame] = {}
        if extra:
            var_data.update(extra)
        if "value" in data.columns:
            var_name = spec.variables[0] if spec.variables else "unknown"
            var_data[var_name] = data

        wq.load_dataframes(var_data)
        result.anomalies = wq.detect_anomalies()
        guideline_df = wq.check_who_guidelines()
        result.status = {"who_guidelines": guideline_df.to_dict(orient="records")} if not guideline_df.empty else {}
        result.steps.append("Ran anomaly detection + WHO guideline check")
