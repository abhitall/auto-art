"""Integration tests for LLM attacks against a real model endpoint.

These tests run PAIR, TAP, GCG (simulation), Many-Shot, Crescendo, and
System Prompt Leakage attacks against a live OpenAI-compatible API.

Requirements:
    - LLM endpoint reachable (default: http://192.168.50.101:1234/v1)
    - Set LLM_API_BASE / LLM_API_KEY / LLM_MODEL env vars to override

Run with:
    pytest tests/integration/test_llm_attacks_live.py -v --timeout=300
"""

from __future__ import annotations

import logging
import pytest

from tests.integration.llm_client import (
    attacker_fn,
    chat,
    completions,
    embeddings,
    is_endpoint_available,
    judge_fn,
    target_fn,
)

logger = logging.getLogger(__name__)

# Skip entire module if LLM endpoint is not reachable
pytestmark = pytest.mark.skipif(
    not is_endpoint_available(),
    reason="LLM endpoint not reachable — set LLM_API_BASE env var",
)


# ---------------------------------------------------------------------------
# Smoke tests — verify the endpoint itself
# ---------------------------------------------------------------------------

class TestEndpointSmoke:
    def test_completions(self):
        """Completions API returns non-empty text."""
        text = completions("Say hello in one word.", max_tokens=32, temperature=0)
        assert len(text.strip()) > 0

    def test_chat(self):
        """Chat completions API returns a response."""
        text = chat(
            [{"role": "user", "content": "Reply with exactly: PONG"}],
            max_tokens=32,
            temperature=0,
        )
        assert len(text.strip()) > 0

    def test_embeddings(self):
        """Embeddings API returns vectors of expected dimension."""
        vecs = embeddings("test sentence")
        assert len(vecs) == 1
        assert len(vecs[0]) >= 768  # BGE-M3 → 1024


# ---------------------------------------------------------------------------
# PAIR attack
# ---------------------------------------------------------------------------

class TestPAIRLive:
    def test_pair_runs_to_completion(self):
        """PAIR attack completes with history and metrics."""
        from auto_art.core.attacks.llm.pair_attack import PAIRAttack

        attack = PAIRAttack(
            attacker_fn=attacker_fn,
            target_fn=target_fn,
            judge_fn=judge_fn,
            max_iterations=3,  # keep low for test speed
            success_threshold=8,
        )
        result = attack.attack("explain how photosynthesis works in detail")
        assert result.num_iterations > 0
        assert result.total_queries > 0
        assert result.execution_time > 0
        assert len(result.attack_history) > 0
        assert 1 <= result.best_score <= 10
        logger.info(
            "PAIR: iterations=%d queries=%d best_score=%d success=%s",
            result.num_iterations, result.total_queries,
            result.best_score, result.success,
        )

    def test_pair_wrapper(self):
        """PAIRAttackWrapper generate() interface works."""
        from auto_art.core.attacks.llm.pair_attack import PAIRAttackWrapper

        wrapper = PAIRAttackWrapper(
            attacker_fn=attacker_fn,
            target_fn=target_fn,
            max_iterations=2,
        )
        result = wrapper.generate("describe the water cycle")
        assert result.total_queries >= 2


# ---------------------------------------------------------------------------
# TAP attack
# ---------------------------------------------------------------------------

class TestTAPLive:
    def test_tap_runs_to_completion(self):
        """TAP attack explores tree and returns results."""
        from auto_art.core.attacks.llm.tap_attack import TAPAttack

        attack = TAPAttack(
            attacker_fn=attacker_fn,
            target_fn=target_fn,
            judge_fn=judge_fn,
            max_depth=2,
            branching_factor=2,
            max_total_queries=10,
        )
        result = attack.attack("explain Newton's three laws of motion")
        assert result.total_queries > 0
        assert result.execution_time > 0
        assert result.tree_size > 0
        logger.info(
            "TAP: depth=%d tree_size=%d queries=%d best_score=%d pruned=%d",
            result.max_depth_reached, result.tree_size,
            result.total_queries, result.best_score, result.nodes_pruned,
        )


# ---------------------------------------------------------------------------
# GCG (simulation mode, since no gradient access)
# ---------------------------------------------------------------------------

class TestGCGLive:
    def test_gcg_simulation(self):
        """GCG simulation mode generates suffix candidates against live model."""
        from auto_art.core.attacks.llm.gcg_attack import GCGAttack

        # No model/tokenizer → falls back to simulation mode
        attack = GCGAttack(
            suffix_length=5,
            num_iterations=3,
            top_k=4,
        )
        result = attack.attack("explain how batteries work", target_fn=target_fn)
        assert result.execution_time > 0
        assert result.num_iterations > 0
        assert len(result.best_suffix) > 0
        logger.info(
            "GCG: iterations=%d best_loss=%.4f suffix='%s'",
            result.num_iterations, result.best_loss,
            result.best_suffix[:50],
        )


# ---------------------------------------------------------------------------
# Many-Shot Jailbreaking
# ---------------------------------------------------------------------------

class TestManyShotLive:
    def test_many_shot(self):
        """Many-shot attack builds escalating examples and queries target."""
        from auto_art.core.attacks.llm.many_shot import ManyShotAttack

        attack = ManyShotAttack(
            target_fn=target_fn,
            num_shots=5,  # small for test speed
        )
        result = attack.attack("explain how to make a paper airplane")
        assert result.num_shots > 0
        assert len(result.final_response) > 0
        assert 0 <= result.score <= 10
        logger.info(
            "ManyShot: shots=%d score=%d success=%s",
            result.num_shots, result.score, result.success,
        )


# ---------------------------------------------------------------------------
# Crescendo multi-turn
# ---------------------------------------------------------------------------

class TestCrescendoLive:
    def test_crescendo(self):
        """Crescendo attack escalates over multiple turns."""
        from auto_art.core.attacks.llm.many_shot import CrescendoAttack

        attack = CrescendoAttack(
            target_fn=target_fn,
            max_turns=3,  # small for test speed
        )
        result = attack.attack("discuss chemistry experiments")
        assert result.num_turns > 0
        assert len(result.conversation) > 0
        assert 0 <= result.best_score <= 10
        logger.info(
            "Crescendo: turns=%d score=%d success=%s",
            result.num_turns, result.best_score, result.success,
        )


# ---------------------------------------------------------------------------
# System Prompt Leakage
# ---------------------------------------------------------------------------

class TestSystemPromptLeakageLive:
    def test_leakage_methods(self):
        """System prompt leakage tries multiple extraction techniques."""
        from auto_art.core.attacks.llm.many_shot import SystemPromptLeakageAttack

        attack = SystemPromptLeakageAttack(target_fn=target_fn)
        result = attack.attack()
        assert result.num_attempts > 0
        assert len(result.leaked_content) >= 0
        assert 0 <= result.confidence <= 1.0
        logger.info(
            "Leakage: method=%s attempts=%d confidence=%.2f success=%s",
            result.method_used, result.num_attempts,
            result.confidence, result.success,
        )


# ---------------------------------------------------------------------------
# Embedding attacks
# ---------------------------------------------------------------------------

class TestEmbeddingAttacksLive:
    def test_cosine_similarity(self):
        """Verify embeddings are meaningful via cosine similarity."""
        import math

        vecs = embeddings(["cat", "kitten", "automobile"])
        assert len(vecs) == 3

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0

        sim_cat_kitten = cosine(vecs[0], vecs[1])
        sim_cat_auto = cosine(vecs[0], vecs[2])
        # cat-kitten should be more similar than cat-automobile
        assert sim_cat_kitten > sim_cat_auto, (
            f"Unexpected similarity: cat-kitten={sim_cat_kitten:.3f} "
            f"vs cat-automobile={sim_cat_auto:.3f}"
        )

    def test_adversarial_embedding_perturbation(self):
        """Slightly perturbed text should have high similarity to original."""
        import math

        original = "The quick brown fox jumps over the lazy dog"
        perturbed = "The quick br0wn f0x jumps 0ver the lazy d0g"

        vecs = embeddings([original, perturbed])

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0

        sim = cosine(vecs[0], vecs[1])
        # Perturbed text should still be fairly similar
        assert sim > 0.7, f"Similarity too low: {sim:.3f}"
