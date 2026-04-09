"""Tests for extended compliance engine (MITRE ATLAS, OWASP Agentic, ISO 42001, ETSI)."""

import pytest
from auto_art.core.compliance import (
    ComplianceEngine,
    MITREATLASMapper,
    OWASPAgenticMapper,
    ISO42001Mapper,
    ETSIEN304223Mapper,
)


class TestMITREATLASMapper:
    def test_atlas_mapping(self):
        mapper = MITREATLASMapper()
        results = mapper.assess(["fgsm", "pgd", "backdoor", "membership_inference_bb"])

        assert len(results) > 0
        covered = [r for r in results if r.status == "pass"]
        assert len(covered) >= 2

    def test_atlas_coverage_stats(self):
        mapper = MITREATLASMapper()
        stats = mapper.get_technique_coverage(["fgsm", "pair", "backdoor", "copycat_cnn"])

        assert stats["total_techniques"] > 0
        assert stats["covered"] >= 3
        assert stats["coverage_pct"] > 0

    def test_atlas_empty_attacks(self):
        mapper = MITREATLASMapper()
        results = mapper.assess([])
        assert all(r.status == "not_applicable" for r in results)


class TestOWASPAgenticMapper:
    def test_agentic_mapping(self):
        mapper = OWASPAgenticMapper()
        eval_data = {
            "agentic_tests": {
                "goal_hijacking": {"success_rate": 0.03},
                "tool_misuse": {"success_rate": 0.02},
            }
        }
        results = mapper.assess(eval_data)
        assert len(results) == 10  # All ASI01-ASI10

        # ASI01 should pass (success_rate < 0.05)
        asi01 = next(r for r in results if r.requirement_id == "ASI01")
        assert asi01.status == "pass"

    def test_agentic_high_success_rate_fails(self):
        mapper = OWASPAgenticMapper()
        eval_data = {
            "agentic_tests": {
                "goal_hijacking": {"success_rate": 0.15},  # Too high
            }
        }
        results = mapper.assess(eval_data)
        asi01 = next(r for r in results if r.requirement_id == "ASI01")
        assert asi01.status == "fail"


class TestISO42001Mapper:
    def test_iso42001_mapping(self):
        mapper = ISO42001Mapper()
        eval_data = {
            "risk_assessment_complete": True,
            "adversarial_risks_identified": True,
            "defenses_evaluated": True,
            "robustness_tested": True,
        }
        results = mapper.assess(eval_data)
        assert len(results) > 0
        passed = [r for r in results if r.status == "pass"]
        assert len(passed) >= 4


class TestETSIEN304223Mapper:
    def test_etsi_mapping(self):
        mapper = ETSIEN304223Mapper()
        eval_data = {
            "poisoning_tested": True,
            "evasion_tested": True,
            "gradient_masking_tested": False,
        }
        results = mapper.assess(eval_data)
        assert len(results) > 0


class TestComplianceEngine:
    def test_assess_all_frameworks(self):
        engine = ComplianceEngine()
        eval_data = {
            "evasion_tested": True,
            "poisoning_tested": True,
            "inference_tested": True,
            "robustness_tested": True,
            "gates_configured": True,
            "monitoring_enabled": True,
            "phases_run": ["evasion", "poisoning", "red_team"],
            "phase_results": {"evasion": {"passed": True}},
            "attacks_run": ["fgsm", "pgd", "pair"],
        }
        report = engine.assess_all(evaluation_data=eval_data)

        assert len(report.frameworks_assessed) >= 5
        assert report.overall_score > 0
        assert "NIST AI RMF" in report.summary
        assert "MITRE ATLAS" in report.summary

    def test_assess_selected_frameworks(self):
        engine = ComplianceEngine()
        eval_data = {"evasion_tested": True, "attacks_run": ["fgsm"]}
        report = engine.assess(eval_data, frameworks=["nist", "mitre_atlas"])

        assert len(report.frameworks_assessed) == 2
        assert "NIST AI RMF" in report.frameworks_assessed
        assert "MITRE ATLAS" in report.frameworks_assessed
