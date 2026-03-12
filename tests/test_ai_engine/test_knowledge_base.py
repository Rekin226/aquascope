"""Tests for the AI engine knowledge base."""

from aquascope.ai_engine.knowledge_base import (
    ResearchMethodology,
    get_all_methodologies,
    get_methodology,
    search_methodologies,
)


class TestKnowledgeBase:
    def test_get_all_returns_list(self):
        methods = get_all_methodologies()
        assert isinstance(methods, list)
        assert len(methods) >= 20  # We have 26 methodologies

    def test_all_have_required_fields(self):
        for m in get_all_methodologies():
            assert isinstance(m, ResearchMethodology)
            assert m.id
            assert m.name
            assert m.category
            assert m.description

    def test_get_methodology_by_id(self):
        m = get_methodology("trend_analysis")
        assert m is not None
        assert m.name == "Mann-Kendall Trend Analysis"

    def test_get_methodology_missing(self):
        m = get_methodology("nonexistent_id")
        assert m is None

    def test_search_by_category(self):
        stats = search_methodologies(category="statistical")
        assert len(stats) >= 2
        for m in stats:
            assert m.category == "statistical"

    def test_search_by_parameters(self):
        results = search_methodologies(parameters=["DO", "BOD5"])
        assert len(results) > 0

    def test_search_by_tags(self):
        results = search_methodologies(tags=["trend"])
        assert len(results) >= 1

    def test_search_by_scale(self):
        results = search_methodologies(scale="regional")
        assert len(results) >= 1
        for m in results:
            assert m.typical_scale == "regional"

    def test_unique_ids(self):
        methods = get_all_methodologies()
        ids = [m.id for m in methods]
        assert len(ids) == len(set(ids)), "Duplicate methodology IDs found"

    def test_categories_present(self):
        methods = get_all_methodologies()
        categories = {m.category for m in methods}
        assert "statistical" in categories
        assert "machine_learning" in categories

    def test_methodology_has_references(self):
        methods = get_all_methodologies()
        methods_with_refs = [m for m in methods if m.references]
        assert len(methods_with_refs) > len(methods) * 0.5  # At least half have references
