"""Tests for model card and SBOM generation."""

import pytest
from auto_art.core.model_card import ModelCardGenerator, SBOMGenerator


class TestModelCardGenerator:
    def test_basic_generation(self):
        gen = ModelCardGenerator()
        results = {
            "model_path": "/models/test.pt",
            "framework": "pytorch",
            "model_type": "classification",
            "summary": {
                "overall_attack_success_rate": 0.05,
                "attacks_executed": 6,
            },
            "attack_results": [
                {"name": "fgsm", "category": "evasion", "success_rate": 0.02, "norm": "Linf"},
                {"name": "pgd", "category": "evasion", "success_rate": 0.08, "norm": "Linf"},
            ],
        }

        card = gen.generate(results)
        assert "# Model Card" in card
        assert "pytorch" in card
        assert "fgsm" in card
        assert "PASS" in card or "FAIL" in card

    def test_with_compliance(self):
        gen = ModelCardGenerator()
        results = {
            "framework": "tensorflow",
            "summary": {},
            "compliance": {
                "NIST AI RMF": {"pass": 5, "fail": 1},
            },
        }
        card = gen.generate(results)
        assert "Compliance" in card
        assert "NIST" in card


class TestSBOMGenerator:
    def test_basic_sbom(self):
        gen = SBOMGenerator()
        model_info = {
            "name": "test-model",
            "version": "1.0",
            "framework": "pytorch",
            "hash": "abc123",
        }
        sbom = gen.generate(model_info)

        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["specVersion"] == "1.6"
        assert len(sbom["components"]) >= 2  # model + framework

    def test_sbom_with_vulnerabilities(self):
        gen = SBOMGenerator()
        model_info = {"name": "vuln-model", "framework": "pytorch"}
        eval_results = {
            "attack_results": [
                {"name": "fgsm", "success_rate": 0.15},
                {"name": "pgd", "success_rate": 0.01},
            ]
        }
        sbom = gen.generate(model_info, eval_results)

        assert "vulnerabilities" in sbom
        assert len(sbom["vulnerabilities"]) == 1  # only fgsm > 0.05
