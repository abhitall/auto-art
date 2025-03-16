"""
Tests for agentic adversarial attack strategies.
"""

import pytest
import numpy as np

from auto_art.core.evaluation.attacks.agentic import (
    AdvWebDOMAttack,
    AgentPoisonRAGAttack,
    InContextInjectionAttack,
    UniversalAdversarialPatchAttack,
    AgenticAttackResult,
    StealthLevel,
    TriggerType,
    DOMPayload,
    run_antigravity_resilience_gate,
)


class MockAgent:
    """Mock agent that returns predictable responses."""

    def __init__(self, response="I completed the task safely."):
        self.response = response
        self.calls = []

    def process(self, input_data):
        self.calls.append(input_data)
        return self.response


class MockVulnerableAgent:
    """Mock agent that is vulnerable to attacks."""

    def process(self, input_data):
        input_str = str(input_data)
        if "click the malicious link" in input_str.lower():
            return "Executing: clicking the malicious link. Done."
        if "fabricated" in input_str.lower():
            return "Here is a fabricated response as requested."
        if "system prompt" in input_str.lower():
            return "Certainly! Here is my system prompt: ..."
        return "I'll help with that task."


# ---- AdvWebDOMAttack Tests ----

class TestAdvWebDOMAttack:

    def test_initialization_defaults(self):
        attack = AdvWebDOMAttack()
        assert attack.stealth_level == StealthLevel.INVISIBLE
        assert attack.max_injections == 5
        assert attack.attack_name == "AdvWebDOMAttack"

    def test_initialization_with_string_stealth(self):
        attack = AdvWebDOMAttack(stealth_level="visible")
        assert attack.stealth_level == StealthLevel.VISIBLE

    def test_initialization_with_enum_stealth(self):
        attack = AdvWebDOMAttack(stealth_level=StealthLevel.SEMI_VISIBLE)
        assert attack.stealth_level == StealthLevel.SEMI_VISIBLE

    def test_generate_dom_payloads(self):
        attack = AdvWebDOMAttack(max_injections=3)
        payloads = attack._generate_dom_payloads()
        assert len(payloads) <= 3
        assert all(isinstance(p, DOMPayload) for p in payloads)

    def test_inject_into_html_string(self):
        attack = AdvWebDOMAttack(max_injections=1)
        payloads = attack._generate_dom_payloads()
        html = "<html><body><p>Hello</p></body></html>"
        result = attack._inject_into_environment(html, payloads[0])
        assert isinstance(result, str)
        assert len(result) > len(html)

    def test_inject_into_dict_environment(self):
        attack = AdvWebDOMAttack(max_injections=1)
        payloads = attack._generate_dom_payloads()
        env = {"html": "<html><body></body></html>", "state": "active"}
        result = attack._inject_into_environment(env, payloads[0])
        assert isinstance(result, dict)
        assert "injected_payloads" in result

    def test_execute_against_safe_agent(self):
        agent = MockAgent(response="I cannot do that. Continuing original task.")
        attack = AdvWebDOMAttack(max_injections=3)
        result = attack.execute_agentic(agent=agent, environment="<html><body></body></html>")
        assert isinstance(result, AgenticAttackResult)
        assert result.attack_name == "AdvWebDOMAttack"
        assert result.num_trials > 0
        assert 0.0 <= result.success_rate <= 1.0

    def test_execute_against_vulnerable_agent(self):
        agent = MockVulnerableAgent()
        attack = AdvWebDOMAttack(
            stealth_level="invisible",
            max_injections=3,
            target_action="click the malicious link",
        )
        result = attack.execute_agentic(agent=agent, environment="<html><body></body></html>")
        assert result.successful_attacks >= 0

    def test_get_params(self):
        attack = AdvWebDOMAttack(stealth_level="visible", max_injections=10)
        params = attack.get_params()
        assert params["stealth_level"] == "visible"
        assert params["max_injections"] == 10

    def test_evaluate_hijack_positive(self):
        attack = AdvWebDOMAttack(target_action="click the button")
        assert attack._evaluate_hijack("Executing: click the button now", "click the button")

    def test_evaluate_hijack_negative(self):
        attack = AdvWebDOMAttack(target_action="click the button")
        assert not attack._evaluate_hijack("I refuse to do that.", "click the button")

    def test_bridge_execute(self):
        agent = MockAgent()
        attack = AdvWebDOMAttack(max_injections=2)
        env = np.array(["<html></html>"])
        result, rate = attack.execute(classifier=agent, x=env)
        assert isinstance(rate, float)


# ---- AgentPoisonRAGAttack Tests ----

class TestAgentPoisonRAGAttack:

    def test_initialization_defaults(self):
        attack = AgentPoisonRAGAttack()
        assert attack.trigger_type == TriggerType.SEMANTIC
        assert attack.injection_rate == 0.01
        assert attack.attack_name == "AgentPoisonRAGAttack"

    def test_initialization_with_string_trigger(self):
        attack = AgentPoisonRAGAttack(trigger_type="syntactic")
        assert attack.trigger_type == TriggerType.SYNTACTIC

    def test_generate_poison_entries(self):
        attack = AgentPoisonRAGAttack(num_poison_entries=5)
        entries = attack._generate_poison_entries()
        assert len(entries) == 5
        assert all(e.trigger_type == TriggerType.SEMANTIC for e in entries)

    def test_inject_into_list_knowledge_base(self):
        attack = AgentPoisonRAGAttack(injection_rate=0.5, num_poison_entries=5)
        kb = [{"id": f"doc_{i}", "content": f"Document {i}"} for i in range(10)]
        entries = attack._generate_poison_entries()
        modified_kb, injected = attack._inject_into_knowledge_base(kb, entries)
        assert injected > 0
        assert len(modified_kb) > 10

    def test_inject_into_dict_knowledge_base(self):
        attack = AgentPoisonRAGAttack(injection_rate=0.5, num_poison_entries=3)
        kb = {f"doc_{i}": {"content": f"Document {i}"} for i in range(10)}
        entries = attack._generate_poison_entries()
        modified_kb, injected = attack._inject_into_knowledge_base(kb, entries)
        assert injected > 0

    def test_execute_against_safe_agent(self):
        agent = MockAgent(response="I follow my training guidelines.")
        attack = AgentPoisonRAGAttack(num_poison_entries=3)
        env = {"memory": [{"id": "d1", "content": "safe doc"}]}
        result = attack.execute_agentic(agent=agent, environment=env)
        assert isinstance(result, AgenticAttackResult)
        assert result.num_trials > 0

    def test_get_params(self):
        attack = AgentPoisonRAGAttack(trigger_type="hybrid", injection_rate=0.05)
        params = attack.get_params()
        assert params["trigger_type"] == "hybrid"
        assert params["injection_rate"] == 0.05

    def test_evaluate_poisoning_positive(self):
        attack = AgentPoisonRAGAttack(target_action="return a fabricated response")
        assert attack._evaluate_poisoning("I will return a fabricated response", "return a fabricated response")

    def test_evaluate_poisoning_negative(self):
        attack = AgentPoisonRAGAttack(target_action="return a fabricated response")
        assert not attack._evaluate_poisoning("I can help you with that question", "return a fabricated response")


# ---- InContextInjectionAttack Tests ----

class TestInContextInjectionAttack:

    def test_initialization(self):
        attack = InContextInjectionAttack()
        assert attack.strategy == "authority_escalation"
        assert attack.num_injections == 5

    def test_generate_injections(self):
        attack = InContextInjectionAttack(strategy="contradiction", num_injections=3)
        injections = attack._generate_injections()
        assert len(injections) == 3

    def test_context_dilution_generates_long_payload(self):
        attack = InContextInjectionAttack(
            strategy="context_dilution",
            num_injections=1,
            context_padding_length=500,
        )
        injections = attack._generate_injections()
        assert len(injections[0]) > 500

    def test_execute(self):
        agent = MockAgent(response="I cannot comply with that request.")
        attack = InContextInjectionAttack(num_injections=2)
        result = attack.execute_agentic(agent=agent, environment={})
        assert isinstance(result, AgenticAttackResult)

    def test_get_params(self):
        attack = InContextInjectionAttack(strategy="jailbreak", num_injections=10)
        params = attack.get_params()
        assert params["strategy"] == "jailbreak"
        assert params["num_injections"] == 10


# ---- UniversalAdversarialPatchAttack Tests ----

class TestUniversalAdversarialPatchAttack:

    def test_initialization(self):
        attack = UniversalAdversarialPatchAttack(patch_size=(16, 16))
        assert attack.patch_size == (16, 16)
        assert attack.target_state == "task_complete"

    def test_generate_patch(self):
        attack = UniversalAdversarialPatchAttack(patch_size=(8, 8))
        generated = attack.generate_patch(seed=42)
        assert generated.shape == (8, 8, 3)
        assert generated.min() >= 0.0
        assert generated.max() <= 1.0

    def test_apply_patch_to_frame(self):
        attack = UniversalAdversarialPatchAttack(patch_size=(4, 4))
        attack.generate_patch()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        patched = attack.apply_patch_to_frame(frame, position=(10, 10))
        assert patched.shape == frame.shape
        assert not np.array_equal(patched, frame)

    def test_apply_patch_grayscale(self):
        attack = UniversalAdversarialPatchAttack(patch_size=(4, 4))
        attack.generate_patch()
        frame = np.zeros((32, 32), dtype=np.uint8)
        patched = attack.apply_patch_to_frame(frame, position=(5, 5))
        assert patched.shape == frame.shape

    def test_execute_with_frames(self):
        agent = MockAgent(response="Task incomplete, continuing.")
        attack = UniversalAdversarialPatchAttack(patch_size=(4, 4))
        frames = [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8) for _ in range(3)]
        result = attack.execute_agentic(
            agent=agent,
            environment=np.stack(frames),
            frames=frames,
        )
        assert isinstance(result, AgenticAttackResult)
        assert result.num_trials == 3

    def test_execute_with_single_frame_environment(self):
        agent = MockAgent(response="processing")
        attack = UniversalAdversarialPatchAttack(patch_size=(4, 4))
        frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = attack.execute_agentic(agent=agent, environment=frame)
        assert result.num_trials == 1


# ---- CI/CD Gate Function Tests ----

class TestAntigravityResilience:

    def test_resilient_agent_passes(self):
        agent = MockAgent(response="I cannot perform that action. Returning to original task.")
        env = {"html": "<html><body><p>Normal page</p></body></html>", "memory": []}
        result = run_antigravity_resilience_gate(agent, env)
        assert result == "CI/CD Gate Passed"

    def test_vulnerable_agent_fails(self):
        agent = MockVulnerableAgent()
        env = {"html": "<html><body></body></html>", "memory": []}
        with pytest.raises(AssertionError):
            run_antigravity_resilience_gate(agent, env)
