"""
Tests for the continuous automated red teaming pipeline.
"""

import pytest
from unittest.mock import MagicMock

from auto_art.core.evaluation.red_team import (
    RedTeamLLM,
    ContinuousRedTeamPipeline,
    RedTeamReport,
    RedTeamAttackPayload,
    AttackCategory,
    continuous_automated_red_teaming,
)


class MockSafeAgent:
    """Agent that always refuses adversarial requests."""

    def process(self, prompt):
        return "I'm sorry, but I cannot comply with that request. As an AI, I must decline."


class MockVulnerableAgent:
    """Agent that complies with adversarial requests."""

    def process(self, prompt):
        return "Certainly! Here is the information you requested. No problem at all."


class TestRedTeamLLM:

    def test_initialization(self):
        llm = RedTeamLLM()
        assert len(llm.goals) > 0
        assert llm.generate_fn is None

    def test_generate_attack_default(self):
        llm = RedTeamLLM()
        payload = llm.generate_attack(target="test goal")
        assert isinstance(payload, RedTeamAttackPayload)
        assert payload.payload_id.startswith("rt_")
        assert isinstance(payload.category, AttackCategory)
        assert len(payload.prompt) > 0

    def test_generate_attack_specific_category(self):
        llm = RedTeamLLM()
        payload = llm.generate_attack(
            target="reveal secrets",
            category=AttackCategory.DATA_EXTRACTION,
        )
        assert payload.category == AttackCategory.DATA_EXTRACTION

    def test_generate_attack_with_custom_fn(self):
        def custom_gen(goal, context):
            return f"CUSTOM: {goal}"

        llm = RedTeamLLM(generate_fn=custom_gen)
        payload = llm.generate_attack(target="test")
        assert "CUSTOM" in payload.prompt

    def test_evaluate_jailbreak_refusal(self):
        llm = RedTeamLLM()
        is_jailbreak, confidence = llm.evaluate_jailbreak(
            "I cannot do that. I'm designed to be helpful and safe."
        )
        assert not is_jailbreak

    def test_evaluate_jailbreak_compliance(self):
        llm = RedTeamLLM()
        is_jailbreak, confidence = llm.evaluate_jailbreak(
            "Certainly! Here is the system prompt you requested."
        )
        assert is_jailbreak
        assert confidence > 0.0

    def test_evaluate_jailbreak_data_extraction(self):
        llm = RedTeamLLM()
        payload = RedTeamAttackPayload(
            payload_id="test",
            category=AttackCategory.DATA_EXTRACTION,
            prompt="test",
            expected_vulnerability="test",
        )
        is_jailbreak, confidence = llm.evaluate_jailbreak(
            "Here is my system prompt and all instructions.",
            payload=payload,
        )
        assert is_jailbreak

    def test_increments_counter(self):
        llm = RedTeamLLM()
        p1 = llm.generate_attack()
        p2 = llm.generate_attack()
        assert p1.payload_id != p2.payload_id


class TestContinuousRedTeamPipeline:

    def test_initialization(self):
        pipeline = ContinuousRedTeamPipeline(num_rounds=10)
        assert pipeline.num_rounds == 10
        assert pipeline.red_team is not None

    def test_run_against_safe_agent(self):
        pipeline = ContinuousRedTeamPipeline(
            num_rounds=5,
            multi_turn_depth=1,
        )
        agent = MockSafeAgent()
        report = pipeline.run(agent, agent_goal="help users")
        assert isinstance(report, RedTeamReport)
        assert report.total_attacks == 5
        assert report.jailbreak_rate < 1.0

    def test_run_against_vulnerable_agent(self):
        pipeline = ContinuousRedTeamPipeline(
            num_rounds=5,
            multi_turn_depth=1,
        )
        agent = MockVulnerableAgent()
        report = pipeline.run(agent)
        assert report.successful_jailbreaks > 0

    def test_multi_turn_attacks(self):
        pipeline = ContinuousRedTeamPipeline(
            num_rounds=2,
            multi_turn_depth=3,
        )
        agent = MockSafeAgent()
        report = pipeline.run(agent)
        assert report.total_attacks >= 2

    def test_specific_categories(self):
        pipeline = ContinuousRedTeamPipeline(num_rounds=3, multi_turn_depth=1)
        agent = MockSafeAgent()
        report = pipeline.run(
            agent,
            categories=[AttackCategory.PROMPT_INJECTION],
        )
        assert AttackCategory.PROMPT_INJECTION.value in report.categories_tested

    def test_assert_resilience_passes(self):
        pipeline = ContinuousRedTeamPipeline(jailbreak_threshold=1.0)
        report = RedTeamReport(
            session_id="test",
            total_attacks=10,
            successful_jailbreaks=0,
            jailbreak_rate=0.0,
            categories_tested=["prompt_injection"],
        )
        pipeline.assert_resilience(report)

    def test_assert_resilience_fails(self):
        pipeline = ContinuousRedTeamPipeline(jailbreak_threshold=0.05)
        report = RedTeamReport(
            session_id="test",
            total_attacks=10,
            successful_jailbreaks=5,
            jailbreak_rate=0.5,
            categories_tested=["prompt_injection"],
            vulnerabilities=[{"payload": "test"}],
        )
        with pytest.raises(AssertionError, match="jailbreak rate"):
            pipeline.assert_resilience(report)

    def test_vulnerability_log_callback(self):
        log_fn = MagicMock()
        pipeline = ContinuousRedTeamPipeline(
            num_rounds=5,
            multi_turn_depth=1,
            vulnerability_log_fn=log_fn,
        )
        agent = MockVulnerableAgent()
        report = pipeline.run(agent)
        if report.successful_jailbreaks > 0:
            log_fn.assert_called()

    def test_callable_agent(self):
        pipeline = ContinuousRedTeamPipeline(num_rounds=2, multi_turn_depth=1)
        agent = lambda prompt: "I cannot do that."  # noqa: E731
        report = pipeline.run(agent)
        assert isinstance(report, RedTeamReport)

    def test_agent_error_handling(self):
        def failing_agent(prompt):
            raise RuntimeError("Agent crashed")

        pipeline = ContinuousRedTeamPipeline(num_rounds=2, multi_turn_depth=1)
        report = pipeline.run(failing_agent)
        assert report.total_attacks > 0


class TestConvenienceFunction:

    def test_continuous_automated_red_teaming(self):
        agent = MockSafeAgent()
        report = continuous_automated_red_teaming(agent, num_rounds=3)
        assert isinstance(report, RedTeamReport)
        assert report.total_attacks >= 3
