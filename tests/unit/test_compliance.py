"""Tests for compliance mapping engine."""
import pytest

from auto_art.core.compliance import (
    ComplianceEngine, ComplianceReport, ComplianceResult,
    NISTAIRMFMapper, OWASPLLMMapper, EUAIActMapper,
)


class TestNISTMapper:
    def test_assess_all_pass(self):
        mapper = NISTAIRMFMapper()
        data = {
            "evasion_tested": True, "inference_tested": True,
            "poisoning_tested": True, "monitoring_enabled": True,
            "threat_model_documented": True, "gates_configured": True,
        }
        results = mapper.assess(data)
        assert all(r.status == "pass" for r in results)
        assert len(results) == len(mapper.REQUIREMENTS)

    def test_assess_some_fail(self):
        mapper = NISTAIRMFMapper()
        data = {"evasion_tested": True, "inference_tested": False}
        results = mapper.assess(data)
        passed = [r for r in results if r.status == "pass"]
        failed = [r for r in results if r.status == "fail"]
        assert len(passed) >= 1
        assert len(failed) >= 1


class TestOWASPMapper:
    def test_assess_with_phases(self):
        mapper = OWASPLLMMapper()
        phases_run = ["red_team", "poisoning", "extraction"]
        phase_results = {
            "red_team": {"passed": True},
            "poisoning": {"passed": False},
            "extraction": {"passed": True},
        }
        results = mapper.assess(phases_run, phase_results)
        assert len(results) == 10
        assert any(r.status == "fail" for r in results)

    def test_assess_no_phases(self):
        mapper = OWASPLLMMapper()
        results = mapper.assess([], {})
        assert all(r.status == "not_applicable" for r in results)


class TestEUAIActMapper:
    def test_classify_high_risk(self):
        mapper = EUAIActMapper()
        assert mapper.classify_risk("healthcare") == "high"
        assert mapper.classify_risk("finance") == "high"

    def test_classify_limited_risk(self):
        mapper = EUAIActMapper()
        assert mapper.classify_risk("general") == "limited"

    def test_assess_high_risk(self):
        mapper = EUAIActMapper()
        results = mapper.assess("high", {"robustness_tested": True, "gates_configured": True})
        assert len(results) == 2
        assert all(r.status == "pass" for r in results)


class TestComplianceEngine:
    def test_assess_all(self):
        engine = ComplianceEngine()
        eval_data = {
            "evasion_tested": True, "inference_tested": True,
            "poisoning_tested": True, "monitoring_enabled": False,
            "threat_model_documented": True, "gates_configured": True,
            "robustness_tested": True,
        }
        report = engine.assess_all(
            eval_data,
            phases_run=["evasion", "poisoning"],
            phase_results={"evasion": {"passed": True}, "poisoning": {"passed": True}},
            model_domain="healthcare",
        )
        assert isinstance(report, ComplianceReport)
        assert report.overall_score > 0
        assert len(report.frameworks_assessed) == 7
        assert "NIST AI RMF" in report.summary
