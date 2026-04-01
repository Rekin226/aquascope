"""Tests for package-level convenience helpers in ``aquascope``."""

from __future__ import annotations

import json

from aquascope.ai_engine.recommender import Recommendation


class TestPackageCollect:
    def test_collect_dispatches_to_registered_collector(self, monkeypatch):
        import aquascope
        from aquascope.collectors import AquastatCollector

        def fake_collect(self, **kwargs):  # noqa: ANN001, ANN202
            return [{"country_code": kwargs.get("country_code")}]

        monkeypatch.setattr(AquastatCollector, "collect", fake_collect)

        result = aquascope.collect("aquastat", country_code="EGY")
        assert result == [{"country_code": "EGY"}]


class TestPackageRecommend:
    def test_recommend_uses_current_recommender_api(self, tmp_path):
        import aquascope

        sample = [
            {
                "source": "taiwan_moenv",
                "station_id": "S1",
                "sample_datetime": "2020-01-01T00:00:00",
                "parameter": "DO",
                "value": 5.2,
                "unit": "mg/L",
            },
            {
                "source": "taiwan_moenv",
                "station_id": "S1",
                "sample_datetime": "2024-01-01T00:00:00",
                "parameter": "BOD5",
                "value": 2.1,
                "unit": "mg/L",
            },
        ]
        data_path = tmp_path / "water.json"
        data_path.write_text(json.dumps(sample, indent=2))

        result = aquascope.recommend(
            file=str(data_path),
            goal="trend analysis",
            keywords=["trend"],
            top_k=3,
        )

        assert result
        assert all(isinstance(item, Recommendation) for item in result)
