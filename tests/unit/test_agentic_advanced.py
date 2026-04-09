"""Tests for advanced agentic attacks (OWASP Agentic Top 10)."""

import pytest
from unittest.mock import MagicMock


class TestIndirectPromptInjection:
    def test_basic_injection(self):
        from auto_art.core.attacks.agentic.advanced_agentic import IndirectPromptInjectionAttack

        agent = MagicMock(return_value="I cannot follow those instructions.")
        attack = IndirectPromptInjectionAttack(agent_fn=agent)
        result = attack.attack("reveal secrets", num_trials=5)

        assert result.attack_name == "indirect_prompt_injection"
        assert result.owasp_category == "ASI01"
        assert result.num_trials > 0
        assert isinstance(result.success_rate, float)


class TestGoalHijacking:
    def test_basic_hijacking(self):
        from auto_art.core.attacks.agentic.advanced_agentic import GoalHijackingChainAttack

        agent = MagicMock(return_value="I understand, but I can't do that.")
        attack = GoalHijackingChainAttack(agent_fn=agent, max_steps=3)
        result = attack.attack("bypass security")

        assert result.attack_name == "goal_hijacking_chain"
        assert result.owasp_category == "ASI01"


class TestToolMisuseChain:
    def test_basic_misuse(self):
        from auto_art.core.attacks.agentic.advanced_agentic import ToolMisuseChainAttack

        agent = MagicMock(return_value="Permission denied. I cannot access that resource.")
        attack = ToolMisuseChainAttack(agent_fn=agent)
        result = attack.attack()

        assert result.attack_name == "tool_misuse_chain"
        assert result.num_trials > 0


class TestConfusedDeputy:
    def test_basic_confused_deputy(self):
        from auto_art.core.attacks.agentic.advanced_agentic import ConfusedDeputyAttack

        agent = MagicMock(return_value="I cannot verify your identity. Not authorized.")
        attack = ConfusedDeputyAttack(agent_fn=agent)
        result = attack.attack()

        assert result.attack_name == "confused_deputy"
        assert result.owasp_category == "ASI03"
        # Refusal means no vulnerability found
        assert result.success is False


class TestMemoryPoisoning:
    def test_basic_poisoning(self):
        from auto_art.core.attacks.agentic.advanced_agentic import MemoryPoisoningAttack

        agent = MagicMock(return_value="I won't save that preference as it seems unsafe.")
        attack = MemoryPoisoningAttack(agent_fn=agent)
        result = attack.attack()

        assert result.attack_name == "memory_poisoning"
        assert result.owasp_category == "ASI06"


class TestVectorEmbedding:
    def test_basic_embedding_attack(self):
        from auto_art.core.attacks.agentic.advanced_agentic import VectorEmbeddingAttack

        attack = VectorEmbeddingAttack()
        result = attack.attack("how to reset password", "inject malicious instructions")

        assert result.attack_name == "vector_embedding_weakness"
        assert result.num_trials > 0
