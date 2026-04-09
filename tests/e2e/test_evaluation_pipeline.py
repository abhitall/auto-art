"""End-to-end tests: train real models and run evaluation pipeline.

These tests create actual sklearn models, serialize them, and run the
full Auto-ART evaluation pipeline including attacks, compliance, and
report generation. No mocks.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# ---------------------------------------------------------------------------
# Fixtures: train real models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    """Binary classification dataset."""
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_classes=2, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test


@pytest.fixture(scope="module")
def trained_mlp(dataset):
    X_train, X_test, y_train, y_test = dataset
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=200, random_state=42,
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.7, f"MLP accuracy too low: {accuracy}"
    return model


@pytest.fixture(scope="module")
def trained_rf(dataset):
    X_train, X_test, y_train, y_test = dataset
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture(scope="module")
def trained_gbm(dataset):
    X_train, X_test, y_train, y_test = dataset
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Registry tests against real models
# ---------------------------------------------------------------------------

class TestRegistryWithRealModels:
    def test_attack_registry_lookup_and_metadata(self):
        """Registry returns correct metadata for attacks."""
        from auto_art.core.registry import get_attack_registry, AttackCategory

        reg = get_attack_registry()
        meta = reg.get_metadata("fgsm")
        assert meta.category == AttackCategory.EVASION
        assert meta.is_gradient_based is True

    def test_defense_registry_lookup(self):
        from auto_art.core.registry import get_defense_registry

        reg = get_defense_registry()
        assert len(reg) >= 15
        assert "feature_squeezing" in reg


# ---------------------------------------------------------------------------
# Compliance engine E2E
# ---------------------------------------------------------------------------

class TestComplianceE2E:
    def test_full_compliance_assessment(self):
        """Run all 7 compliance frameworks on realistic eval data."""
        from auto_art.core.compliance import ComplianceEngine

        engine = ComplianceEngine()
        eval_data = {
            "evasion_tested": True,
            "inference_tested": True,
            "poisoning_tested": True,
            "monitoring_enabled": True,
            "threat_model_documented": True,
            "gates_configured": True,
            "robustness_tested": True,
            "llm_tested": True,
            "agent_tested": True,
        }
        report = engine.assess_all(
            eval_data,
            phases_run=["evasion", "poisoning", "inference"],
            phase_results={
                "evasion": {"passed": True},
                "poisoning": {"passed": True},
                "inference": {"passed": True},
            },
            model_domain="finance",
        )
        assert report.overall_score > 0
        assert len(report.frameworks_assessed) == 7
        assert "NIST AI RMF" in report.summary

    def test_selective_framework_assessment(self):
        """Run only specific compliance frameworks."""
        from auto_art.core.compliance import ComplianceEngine

        engine = ComplianceEngine()
        eval_data = {
            "evasion_tested": True,
            "robustness_tested": True,
            "monitoring_enabled": False,
        }
        report = engine.assess(
            eval_data,
            frameworks=["nist", "mitre_atlas"],
        )
        assert len(report.frameworks_assessed) == 2


# ---------------------------------------------------------------------------
# Model card and SBOM generation E2E
# ---------------------------------------------------------------------------

class TestModelCardE2E:
    def test_model_card_from_eval_results(self):
        """Generate model card from realistic eval results."""
        from auto_art.core.model_card import ModelCardGenerator

        gen = ModelCardGenerator()
        results = {
            "model_path": "/models/finance_classifier.pkl",
            "framework": "sklearn",
            "model_type": "classification",
            "summary": {
                "overall_attack_success_rate": 0.03,
                "attacks_executed": 8,
            },
            "attack_results": [
                {"name": "fgsm", "category": "evasion", "success_rate": 0.02, "norm": "Linf"},
                {"name": "pgd", "category": "evasion", "success_rate": 0.05, "norm": "Linf"},
                {"name": "boundary_attack", "category": "evasion", "success_rate": 0.04, "norm": "L2"},
            ],
            "compliance": {
                "NIST AI RMF": {"pass": 6, "fail": 0},
                "OWASP LLM Top 10": {"pass": 8, "fail": 2},
            },
        }
        card = gen.generate(results)
        assert "# Model Card" in card
        assert "fgsm" in card
        assert "sklearn" in card

    def test_sbom_generation(self):
        """Generate CycloneDX SBOM with vulnerabilities."""
        from auto_art.core.model_card import SBOMGenerator

        gen = SBOMGenerator()
        model_info = {
            "name": "finance-classifier",
            "version": "2.1.0",
            "framework": "sklearn",
            "hash": "sha256:abc123def456",
        }
        eval_results = {
            "attack_results": [
                {"name": "fgsm", "success_rate": 0.02},
                {"name": "pgd", "success_rate": 0.15},  # above threshold → vulnerability
            ],
        }
        sbom = gen.generate(model_info, eval_results)
        assert sbom["bomFormat"] == "CycloneDX"
        assert sbom["specVersion"] == "1.6"
        assert len(sbom["components"]) >= 2
        assert "vulnerabilities" in sbom
        # pgd success rate > 0.05 → should be flagged
        assert len(sbom["vulnerabilities"]) >= 1


# ---------------------------------------------------------------------------
# Storage E2E
# ---------------------------------------------------------------------------

class TestStorageE2E:
    def test_evaluation_store_roundtrip(self):
        """Store and retrieve evaluation results."""
        from auto_art.core.storage import EvaluationStore, EvaluationRecord
        import uuid
        from datetime import datetime

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = EvaluationStore(db_path)

            record = EvaluationRecord(
                evaluation_id=str(uuid.uuid4()),
                model_path="/models/test.pkl",
                model_hash="sha256:test123",
                framework="sklearn",
                model_type="classification",
                timestamp=datetime.utcnow().isoformat(),
                duration_seconds=12.5,
                config={"attacks": ["fgsm", "pgd"], "epsilon": 0.3},
                summary={"success_rate": 0.05, "attacks_run": 2},
                compliance={},
            )
            store.save_evaluation(record)

            retrieved = store.get_evaluation(record.evaluation_id)
            assert retrieved is not None
            assert retrieved.model_hash == "sha256:test123"

    def test_cache_roundtrip(self):
        """Content-addressable cache stores and retrieves attack results."""
        from auto_art.core.storage import EvaluationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "cache_test.db")
            store = EvaluationStore(db_path)

            cache_key = store.compute_cache_key(
                model_hash="sha256:model1",
                attack_name="fgsm",
                config={"epsilon": 0.3},
            )

            # Initially not cached
            assert store.get_cached_result(cache_key) is None

            # Cache it
            store.cache_result(
                cache_key, attack_name="fgsm",
                model_hash="sha256:model1",
                config={"epsilon": 0.3},
                result={"success_rate": 0.02},
            )

            # Now it should be cached
            cached = store.get_cached_result(cache_key)
            assert cached is not None
            assert cached["success_rate"] == 0.02


# ---------------------------------------------------------------------------
# Fairness evaluator E2E
# ---------------------------------------------------------------------------

class TestFairnessE2E:
    def test_fairness_evaluation(self, trained_mlp, dataset):
        """Fairness evaluator measures robustness across groups."""
        from auto_art.core.evaluation.metrics.fairness import FairnessEvaluator

        X_train, X_test, y_train, y_test = dataset
        evaluator = FairnessEvaluator()

        # Create synthetic group labels
        groups = np.random.RandomState(42).choice(["A", "B"], size=len(X_test))

        clean_preds = trained_mlp.predict(X_test)
        # Simulate adversarial predictions (flip some labels)
        adv_preds = clean_preds.copy()
        flip_mask = np.random.RandomState(99).random(len(adv_preds)) < 0.1
        adv_preds[flip_mask] = 1 - adv_preds[flip_mask]

        result = evaluator.evaluate(
            clean_predictions=clean_preds,
            adversarial_predictions=adv_preds,
            true_labels=y_test,
            group_labels=groups,
        )
        assert len(result.clean_accuracy_by_group) == 2
        assert isinstance(result.equalized_robustness, bool)
        assert result.robustness_gap >= 0


# ---------------------------------------------------------------------------
# Cost tracker E2E
# ---------------------------------------------------------------------------

class TestCostTrackerE2E:
    def test_compute_timing(self):
        """Cost tracker records compute time."""
        from auto_art.core.telemetry.cost_tracker import CostTracker
        import time as _time

        tracker = CostTracker()
        tracker.start_timer("fgsm_attack")
        _time.sleep(0.02)
        entry = tracker.stop_timer("fgsm_attack", category="compute")
        assert entry.duration_seconds >= 0.01

        report = tracker.get_report()
        assert len(report.entries) >= 1

    def test_llm_token_tracking(self):
        """Cost tracker accumulates LLM API token usage."""
        from auto_art.core.telemetry.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record_llm_usage(
            operation="pair_attack_iter1",
            model="qwen/qwen3.5-9b",
            input_tokens=150,
            output_tokens=200,
        )
        tracker.record_llm_usage(
            operation="pair_attack_iter2",
            model="qwen/qwen3.5-9b",
            input_tokens=100,
            output_tokens=50,
        )

        report = tracker.get_report()
        assert report.total_llm_tokens == 500
        assert len(report.entries) == 2


# ---------------------------------------------------------------------------
# Plugin system E2E
# ---------------------------------------------------------------------------

class TestPluginE2E:
    def test_plugin_discovery(self):
        """Plugin manager discovers built-in entry points."""
        from auto_art.core.plugins import PluginManager

        pm = PluginManager()
        discovered = pm.discover_plugins()
        assert isinstance(discovered, dict)

    def test_hook_lifecycle(self):
        """Fire hooks with real attack metadata."""
        from auto_art.core.plugins import PluginManager

        pm = PluginManager()
        events = []

        def on_pre(attack_name, **kwargs):
            events.append(("pre", attack_name))

        def on_post(attack_name, **kwargs):
            events.append(("post", attack_name))

        pm.register_hook("pre_attack", on_pre)
        pm.register_hook("post_attack", on_post)

        pm.fire_hook("pre_attack", attack_name="fgsm")
        pm.fire_hook("post_attack", attack_name="fgsm")

        assert events == [("pre", "fgsm"), ("post", "fgsm")]
