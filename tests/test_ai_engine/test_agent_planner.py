"""Tests for aquascope.ai_engine — planner, model_recommender, agent."""

from __future__ import annotations


class TestChallengePlanner:
    def test_parse_flood(self):
        from aquascope.ai_engine.planner import ChallengePlanner

        planner = ChallengePlanner()
        spec = planner.parse("Forecast flooding on the Niger River at lat 13.5, lon 2.1 for 14 days")
        assert spec.challenge_type == "flood"
        assert spec.latitude == 13.5
        assert spec.longitude == 2.1
        assert spec.forecast_days == 14
        assert "Niger River" in spec.location_name

    def test_parse_drought(self):
        from aquascope.ai_engine.planner import ChallengePlanner

        planner = ChallengePlanner()
        spec = planner.parse("Drought monitoring near Sahel using precipitation data")
        assert spec.challenge_type == "drought"
        assert "precipitation" in spec.variables

    def test_parse_water_quality(self):
        from aquascope.ai_engine.planner import ChallengePlanner

        planner = ChallengePlanner()
        spec = planner.parse("Detect nitrate contamination and check WHO guidelines at station XYZ")
        assert spec.challenge_type == "water_quality"
        assert "nitrate" in spec.variables

    def test_parse_unknown(self):
        from aquascope.ai_engine.planner import ChallengePlanner

        planner = ChallengePlanner()
        spec = planner.parse("What is the meaning of life?")
        assert spec.challenge_type == "unknown"


class TestModelRecommender:
    def test_recommend_flood_forecast(self):
        from aquascope.ai_engine.model_recommender import ModelRecommender

        rec = ModelRecommender()
        picks = rec.recommend("flood", "forecast")
        assert len(picks) > 0
        assert picks[0].model_id == "prophet"

    def test_recommend_drought_index(self):
        from aquascope.ai_engine.model_recommender import ModelRecommender

        rec = ModelRecommender()
        picks = rec.recommend("drought", "index")
        assert any(p.model_id == "spi_drought_index" for p in picks)

    def test_recommend_unknown_falls_back(self):
        from aquascope.ai_engine.model_recommender import ModelRecommender

        rec = ModelRecommender()
        picks = rec.recommend("flood", "nonexistent_task")
        # Should fallback to all models for flood
        assert len(picks) > 0


class TestOpenMeteoCollector:
    def test_normalise(self):
        from aquascope.collectors.openmeteo import OpenMeteoCollector

        collector = OpenMeteoCollector(mode="weather")
        raw = {
            "latitude": 25.03,
            "longitude": 121.57,
            "daily": {
                "time": ["2023-01-01", "2023-01-02"],
                "precipitation_sum": [5.0, 0.0],
            },
            "daily_units": {"precipitation_sum": "mm"},
        }
        records = collector.normalise(raw)
        assert len(records) == 2
        assert records[0].parameter == "precipitation_sum"
        assert records[0].value == 5.0
