"""
Tests for OWASP LLM Top 10 coverage mapping.
"""

from auto_art.core.evaluation.owasp_mapping import (
    OWASP_LLM_TOP_10_2025,
    get_coverage_report,
    get_coverage_markdown,
)


class TestOwaspMapping:

    def test_has_10_categories(self):
        assert len(OWASP_LLM_TOP_10_2025) == 10

    def test_all_categories_have_ids(self):
        ids = [c.id for c in OWASP_LLM_TOP_10_2025]
        assert ids == [f"LLM{i:02d}" for i in range(1, 11)]

    def test_all_categories_have_attacks(self):
        for cat in OWASP_LLM_TOP_10_2025:
            assert len(cat.attacks) > 0, f"{cat.id} has no attacks"

    def test_all_categories_have_defences(self):
        for cat in OWASP_LLM_TOP_10_2025:
            assert len(cat.defences) > 0, f"{cat.id} has no defences"

    def test_coverage_report_structure(self):
        report = get_coverage_report()
        assert report["standard"] == "OWASP LLM Top 10 (2025)"
        assert report["total_categories"] == 10
        assert "full_coverage" in report
        assert "partial_coverage" in report
        assert "coverage_percentage" in report
        assert len(report["categories"]) == 10

    def test_coverage_percentage_reasonable(self):
        report = get_coverage_report()
        assert 50 <= report["coverage_percentage"] <= 100

    def test_coverage_markdown_format(self):
        md = get_coverage_markdown()
        assert "# OWASP LLM Top 10" in md
        assert "LLM01" in md
        assert "LLM10" in md
        assert "Prompt Injection" in md
        assert "Detailed Mapping" in md

    def test_no_empty_names(self):
        for cat in OWASP_LLM_TOP_10_2025:
            assert cat.name, f"{cat.id} has empty name"
            assert cat.description, f"{cat.id} has empty description"
