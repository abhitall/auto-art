"""Tests for evaluation store and content-addressable caching."""

import os
import tempfile
import pytest
from auto_art.core.storage import EvaluationStore, EvaluationRecord, AttackResultRecord


@pytest.fixture
def store():
    """Create a temporary evaluation store."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield EvaluationStore(db_path=db_path)
    os.unlink(db_path)


class TestEvaluationStore:
    def test_save_and_get_evaluation(self, store):
        record = EvaluationRecord(
            evaluation_id="eval_001",
            model_path="/models/test.pt",
            model_hash="abc123",
            framework="pytorch",
            model_type="classification",
            timestamp="2026-04-05T00:00:00Z",
            duration_seconds=120.5,
            config={"attacks": ["fgsm", "pgd"]},
            summary={"overall_attack_success_rate": 0.15},
            compliance={"nist": {"pass": 5, "fail": 1}},
        )
        store.save_evaluation(record)

        retrieved = store.get_evaluation("eval_001")
        assert retrieved is not None
        assert retrieved.model_path == "/models/test.pt"
        assert retrieved.framework == "pytorch"
        assert retrieved.config["attacks"] == ["fgsm", "pgd"]

    def test_list_evaluations(self, store):
        for i in range(3):
            store.save_evaluation(EvaluationRecord(
                evaluation_id=f"eval_{i:03d}",
                model_path="/models/test.pt",
                model_hash=f"hash_{i}",
                framework="pytorch",
                model_type="classification",
                timestamp=f"2026-04-0{i+1}T00:00:00Z",
                duration_seconds=100.0,
                config={}, summary={}, compliance={},
            ))

        results = store.list_evaluations()
        assert len(results) == 3

    def test_list_evaluations_with_filter(self, store):
        store.save_evaluation(EvaluationRecord(
            evaluation_id="eval_pt", model_path="/m.pt", model_hash="h1",
            framework="pytorch", model_type="classification",
            timestamp="2026-04-01T00:00:00Z", duration_seconds=10,
            config={}, summary={}, compliance={},
        ))
        store.save_evaluation(EvaluationRecord(
            evaluation_id="eval_tf", model_path="/m.h5", model_hash="h2",
            framework="tensorflow", model_type="classification",
            timestamp="2026-04-01T00:00:00Z", duration_seconds=10,
            config={}, summary={}, compliance={},
        ))

        pt_results = store.list_evaluations(framework="pytorch")
        assert len(pt_results) == 1
        assert pt_results[0].framework == "pytorch"


class TestAttackResults:
    def test_save_and_query(self, store):
        store.save_evaluation(EvaluationRecord(
            evaluation_id="eval_001", model_path="/m.pt", model_hash="abc",
            framework="pytorch", model_type="classification",
            timestamp="2026-04-05T00:00:00Z", duration_seconds=10,
            config={}, summary={}, compliance={},
        ))
        store.save_attack_result(AttackResultRecord(
            evaluation_id="eval_001",
            attack_name="fgsm",
            attack_category="evasion",
            success_rate=0.25,
            perturbation_norm=0.03,
            num_samples=100,
            duration_seconds=5.0,
            config={"epsilon": 0.3},
            result={"adversarial_examples": 25},
        ))

        history = store.get_attack_history(attack_name="fgsm")
        assert len(history) == 1
        assert history[0]["success_rate"] == 0.25


class TestContentAddressableCache:
    def test_cache_key_computation(self, store):
        key1 = store.compute_cache_key("model_a", "fgsm", {"eps": 0.3})
        key2 = store.compute_cache_key("model_a", "fgsm", {"eps": 0.3})
        key3 = store.compute_cache_key("model_b", "fgsm", {"eps": 0.3})

        assert key1 == key2  # same inputs = same key
        assert key1 != key3  # different model = different key

    def test_cache_store_and_retrieve(self, store):
        cache_key = "test_cache_key_001"
        result = {"success_rate": 0.15, "perturbation": 0.03}

        store.cache_result(cache_key, "fgsm", "model_hash", {"eps": 0.3}, result)
        cached = store.get_cached_result(cache_key)

        assert cached is not None
        assert cached["success_rate"] == 0.15

    def test_cache_miss(self, store):
        assert store.get_cached_result("nonexistent_key") is None

    def test_cache_stats(self, store):
        stats = store.get_cache_stats()
        assert "cache_entries" in stats
        assert "total_evaluations" in stats

    def test_cache_eviction(self, store):
        for i in range(5):
            store.cache_result(
                f"key_{i}", "fgsm", "model", {"i": i},
                {"result": i},
            )

        evicted = store.evict_cache(max_entries=3)
        stats = store.get_cache_stats()
        assert stats["cache_entries"] <= 3
