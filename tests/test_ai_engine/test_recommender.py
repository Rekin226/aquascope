"""Tests for the AI methodology recommender."""


from aquascope.ai_engine.recommender import DatasetProfile, Recommendation, recommend


class TestRecommender:
    def test_recommend_returns_results(self):
        profile = DatasetProfile(
            parameters=["DO", "BOD5", "COD", "NH3-N", "SS"],
            n_records=500,
            n_stations=12,
            time_span_years=5.0,
            geographic_scope="Taiwan",
            research_goal="trend analysis water quality",
            keywords=["trend", "monitoring"],
        )
        recs = recommend(profile, top_k=5)
        assert len(recs) > 0
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_recommend_sorted_by_score(self):
        profile = DatasetProfile(
            parameters=["DO", "BOD5", "pH"],
            n_records=1000,
            time_span_years=10.0,
            geographic_scope="regional",
        )
        recs = recommend(profile, top_k=10)
        scores = [r.score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_trend_analysis_ranked_high_for_long_timeseries(self):
        profile = DatasetProfile(
            parameters=["DO", "BOD5", "COD", "SS", "NH3-N"],
            n_records=2000,
            n_stations=20,
            time_span_years=15.0,
            geographic_scope="Taiwan",
            research_goal="long-term trend analysis",
            keywords=["trend", "time-series", "long-term"],
        )
        recs = recommend(profile, top_k=5)
        method_ids = [r.methodology.id for r in recs]
        assert "trend_analysis" in method_ids

    def test_recommend_mbbr_for_wastewater_keywords(self):
        profile = DatasetProfile(
            parameters=["BOD5", "COD", "NH3-N", "SS", "DO"],
            n_records=100,
            time_span_years=0.5,
            geographic_scope="pilot",
            research_goal="MBBR pilot performance evaluation",
            keywords=["MBBR", "biofilm", "wastewater", "pilot"],
        )
        recs = recommend(profile, top_k=5)
        method_ids = [r.methodology.id for r in recs]
        assert "mbbr_pilot_study" in method_ids

    def test_recommend_respects_min_score(self):
        profile = DatasetProfile(parameters=["Chlorophyll-a"])
        recs = recommend(profile, top_k=20, min_score=80.0)
        for r in recs:
            assert r.score >= 80.0

    def test_recommend_includes_rationale(self):
        profile = DatasetProfile(
            parameters=["DO", "BOD5"],
            n_records=300,
        )
        recs = recommend(profile, top_k=3)
        for r in recs:
            assert r.rationale  # non-empty string

    def test_empty_profile_still_returns_results(self):
        profile = DatasetProfile()
        recs = recommend(profile, top_k=3, min_score=0)
        assert len(recs) > 0
