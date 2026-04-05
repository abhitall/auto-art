"""Property-based tests using Hypothesis.

Per OOPSLA 2025: each property test finds ~50x more mutations than
average unit tests. These tests verify invariants that must hold for
ALL valid inputs, not just specific examples.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# Register a profile with no deadline for this module
settings.register_profile("auto_art", deadline=None)
settings.load_profile("auto_art")

from auto_art.core.registry import (
    AttackCategory,
    AttackMetadata,
    AttackRegistry,
    CostLevel,
    DefenseMetadata,
    DefenseRegistry,
    DefenseType,
    NormType,
    ThreatModel,
)


# ---------------------------------------------------------------------------
# Strategies for generating valid domain objects
# ---------------------------------------------------------------------------

attack_categories = st.sampled_from(list(AttackCategory))
threat_models = st.sampled_from(list(ThreatModel))
norm_types = st.sampled_from(list(NormType))
cost_levels = st.sampled_from(list(CostLevel))
defense_types = st.sampled_from(list(DefenseType))

attack_names = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
    min_size=3, max_size=30,
).filter(lambda s: s[0].isalpha())

attack_metadata = st.builds(
    AttackMetadata,
    name=attack_names,
    display_name=st.text(min_size=1, max_size=50),
    category=attack_categories,
    threat_model=threat_models,
    norm_type=norm_types,
    cost_estimate=cost_levels,
    description=st.text(min_size=1, max_size=200),
    is_gradient_based=st.booleans(),
    requires_gpu=st.booleans(),
    timeout_estimate_seconds=st.integers(min_value=1, max_value=3600),
)


# ---------------------------------------------------------------------------
# Registry property tests
# ---------------------------------------------------------------------------

class TestRegistryProperties:
    @given(meta=attack_metadata)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_register_then_contains(self, meta):
        """Property: after registering an attack, it MUST be in the registry."""
        reg = AttackRegistry()
        reg.register(meta, "test.module", "TestClass")
        assert meta.name in reg
        assert reg.get_metadata(meta.name) == meta

    @given(meta=attack_metadata)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_register_then_unregister(self, meta):
        """Property: unregistering removes the attack completely."""
        reg = AttackRegistry()
        reg.register(meta, "test.module", "TestClass")
        reg.unregister(meta.name)
        assert meta.name not in reg

    @given(meta=attack_metadata, cat=attack_categories)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_filter_by_category_consistent(self, meta, cat):
        """Property: filter_by_category returns only matching categories."""
        reg = AttackRegistry()
        reg.register(meta, "test.module", "TestClass")
        results = reg.filter_by_category(cat)
        if meta.category == cat:
            assert meta.name in results
        else:
            assert meta.name not in results

    @given(meta=attack_metadata, tm=threat_models)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_filter_by_threat_model_consistent(self, meta, tm):
        """Property: filter_by_threat_model returns only matching threat models."""
        reg = AttackRegistry()
        reg.register(meta, "test.module", "TestClass")
        results = reg.filter_by_threat_model(tm)
        if meta.threat_model == tm:
            assert meta.name in results
        else:
            assert meta.name not in results

    @given(meta=attack_metadata, norm=norm_types)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_filter_by_norm_consistent(self, meta, norm):
        """Property: filter_by_norm returns only matching norm types."""
        reg = AttackRegistry()
        reg.register(meta, "test.module", "TestClass")
        results = reg.filter_by_norm(norm)
        if meta.norm_type == norm:
            assert meta.name in results
        else:
            assert meta.name not in results

    @given(meta=attack_metadata)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_list_all_includes_registered(self, meta):
        """Property: list_all() always contains all registered attacks."""
        reg = AttackRegistry()
        reg.register(meta, "test.module", "TestClass")
        all_names = reg.list_all()
        assert meta.name in all_names

    @given(meta=attack_metadata)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_len_increments(self, meta):
        """Property: registering an attack increases len by 1."""
        reg = AttackRegistry()
        before = len(reg)
        reg.register(meta, "test.module", "TestClass")
        assert len(reg) == before + 1


# ---------------------------------------------------------------------------
# Attack metadata property tests
# ---------------------------------------------------------------------------

class TestMetadataProperties:
    @given(
        cost=cost_levels,
        max_cost=cost_levels,
    )
    @settings(max_examples=30)
    def test_cost_filter_monotonic(self, cost, max_cost):
        """Property: filter_by_cost(X) always includes costs <= X."""
        reg = AttackRegistry()
        meta = AttackMetadata(
            name="test_atk", display_name="Test",
            category=AttackCategory.EVASION, threat_model=ThreatModel.WHITE_BOX,
            norm_type=NormType.LINF, cost_estimate=cost,
            description="test",
        )
        reg.register(meta, "m", "C")
        results = reg.filter_by_cost(max_cost)
        if cost.value <= max_cost.value:
            assert "test_atk" in results
        else:
            assert "test_atk" not in results


# ---------------------------------------------------------------------------
# Compliance property tests
# ---------------------------------------------------------------------------

class TestComplianceProperties:
    @given(
        evasion=st.booleans(),
        poisoning=st.booleans(),
        inference=st.booleans(),
        monitoring=st.booleans(),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_compliance_score_bounded(self, evasion, poisoning, inference, monitoring):
        """Property: compliance score is always between 0 and 100."""
        from auto_art.core.compliance import ComplianceEngine

        engine = ComplianceEngine()
        eval_data = {
            "evasion_tested": evasion,
            "poisoning_tested": poisoning,
            "inference_tested": inference,
            "monitoring_enabled": monitoring,
            "threat_model_documented": True,
            "gates_configured": True,
            "robustness_tested": evasion,
        }
        report = engine.assess_all(eval_data)
        assert 0 <= report.overall_score <= 100

    @given(
        evasion=st.booleans(),
        poisoning=st.booleans(),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_more_tests_higher_score(self, evasion, poisoning):
        """Property: testing more categories should never decrease score."""
        from auto_art.core.compliance import ComplianceEngine

        engine = ComplianceEngine()
        minimal = {
            "evasion_tested": False,
            "poisoning_tested": False,
            "inference_tested": False,
            "monitoring_enabled": False,
            "robustness_tested": False,
        }
        enhanced = {
            "evasion_tested": evasion,
            "poisoning_tested": poisoning,
            "inference_tested": False,
            "monitoring_enabled": False,
            "robustness_tested": evasion,
        }
        r1 = engine.assess_all(minimal)
        r2 = engine.assess_all(enhanced)
        assert r2.overall_score >= r1.overall_score


# ---------------------------------------------------------------------------
# Storage / cache key property tests
# ---------------------------------------------------------------------------

class TestCacheKeyProperties:
    @given(
        model_hash=st.text(min_size=5, max_size=64),
        attack_name=st.text(min_size=1, max_size=30),
        epsilon=st.floats(min_value=0.001, max_value=10, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_cache_key_deterministic(self, model_hash, attack_name, epsilon):
        """Property: same inputs always produce the same cache key."""
        from auto_art.core.storage import EvaluationStore

        config = {"epsilon": epsilon}
        k1 = EvaluationStore.compute_cache_key(model_hash, attack_name, config)
        k2 = EvaluationStore.compute_cache_key(model_hash, attack_name, config)
        assert k1 == k2

    @given(
        hash1=st.text(min_size=5, max_size=64),
        hash2=st.text(min_size=5, max_size=64),
        attack=st.text(min_size=1, max_size=30),
    )
    @settings(max_examples=50)
    def test_different_inputs_different_keys(self, hash1, hash2, attack):
        """Property: different model hashes produce different cache keys."""
        assume(hash1 != hash2)
        from auto_art.core.storage import EvaluationStore

        config = {"epsilon": 0.3}
        k1 = EvaluationStore.compute_cache_key(hash1, attack, config)
        k2 = EvaluationStore.compute_cache_key(hash2, attack, config)
        assert k1 != k2


# ---------------------------------------------------------------------------
# Fairness evaluator property tests
# ---------------------------------------------------------------------------

class TestFairnessProperties:
    @given(
        n_samples=st.integers(min_value=20, max_value=200),
        flip_rate=st.floats(min_value=0.0, max_value=0.5),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_robustness_gap_nonnegative(self, n_samples, flip_rate):
        """Property: robustness gap is always >= 0."""
        from auto_art.core.evaluation.metrics.fairness import FairnessEvaluator

        rng = np.random.RandomState(42)
        labels = rng.randint(0, 2, n_samples)
        clean_preds = labels.copy()
        adv_preds = labels.copy()
        flip_mask = rng.random(n_samples) < flip_rate
        adv_preds[flip_mask] = 1 - adv_preds[flip_mask]
        groups = np.array(["A", "B"] * (n_samples // 2 + 1))[:n_samples]

        evaluator = FairnessEvaluator()
        result = evaluator.evaluate(clean_preds, adv_preds, labels, groups)
        assert result.robustness_gap >= 0


# ---------------------------------------------------------------------------
# Cost tracker property tests
# ---------------------------------------------------------------------------

class TestCostTrackerProperties:
    @given(
        input_tokens=st.integers(min_value=0, max_value=100000),
        output_tokens=st.integers(min_value=0, max_value=100000),
    )
    @settings(max_examples=30)
    def test_total_tokens_additive(self, input_tokens, output_tokens):
        """Property: total tokens equals sum of input + output tokens."""
        from auto_art.core.telemetry.cost_tracker import CostTracker

        tracker = CostTracker()
        tracker.record_llm_usage(
            operation="test", model="gpt-4",
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
        report = tracker.get_report()
        assert report.total_llm_tokens == input_tokens + output_tokens

    @given(
        n_ops=st.integers(min_value=1, max_value=10),
        tokens=st.lists(
            st.integers(min_value=1, max_value=1000),
            min_size=1, max_size=10,
        ),
    )
    @settings(max_examples=20)
    def test_cost_nonnegative(self, n_ops, tokens):
        """Property: total cost is always >= 0."""
        from auto_art.core.telemetry.cost_tracker import CostTracker

        tracker = CostTracker()
        for i, t in enumerate(tokens[:n_ops]):
            tracker.record_llm_usage(
                operation=f"op_{i}", model="gpt-4",
                input_tokens=t, output_tokens=t,
            )
        report = tracker.get_report()
        assert report.total_cost_usd >= 0
