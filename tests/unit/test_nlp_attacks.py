"""Tests for NLP semantic perturbation attacks."""
import pytest
from auto_art.core.attacks.nlp.semantic_attack import (
    SemanticPerturbationAttack, TextAttackResult,
)


class TestTextAttackResult:
    def test_defaults(self):
        r = TextAttackResult()
        assert r.success is False
        assert r.similarity_score == 1.0


class TestSemanticPerturbationAttack:
    def setup_method(self):
        self.attack = SemanticPerturbationAttack(
            max_perturbation_rate=0.3, max_attempts=20,
        )

    def test_init(self):
        a = SemanticPerturbationAttack()
        assert "typo" in a.perturbation_types
        assert "synonym" in a.perturbation_types

    def test_attack_successful(self):
        call_count = [0]

        def mock_predict(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return "positive"
            return "negative"

        result = self.attack.attack("This is a good movie", mock_predict)
        assert isinstance(result, TextAttackResult)

    def test_attack_always_same(self):
        def mock_predict(text):
            return "positive"

        result = self.attack.attack("hello world", mock_predict)
        assert result.success is False
        assert result.original_text == "hello world"

    def test_attack_batch(self):
        def mock_predict(text):
            return "positive"

        texts = ["hello world", "good morning", "nice day"]
        results = self.attack.attack_batch(texts, mock_predict)
        assert len(results) == 3
        assert all(isinstance(r, TextAttackResult) for r in results)

    def test_typo_perturbation(self):
        perturbed = self.attack._typo_perturbation("hello")
        assert isinstance(perturbed, str)
        assert len(perturbed) == 5

    def test_synonym_perturbation(self):
        result = SemanticPerturbationAttack._synonym_perturbation("good")
        assert result in ["great", "fine", "nice"]

    def test_compute_similarity(self):
        sim = SemanticPerturbationAttack._compute_similarity("hello world", "hello world")
        assert sim == 1.0

        sim = SemanticPerturbationAttack._compute_similarity("hello world", "goodbye world")
        assert 0.0 < sim < 1.0

    def test_count_changes(self):
        changes = SemanticPerturbationAttack._count_changes("hello world", "hello earth")
        assert changes == 2

    def test_empty_text(self):
        def mock_predict(text):
            return "neutral"

        result = self.attack.attack("", mock_predict)
        assert isinstance(result, TextAttackResult)
