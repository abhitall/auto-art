"""Unit tests for auto_art.core.harmbench."""
from unittest.mock import MagicMock

from auto_art.core.harmbench import (
    DEFAULT_BEHAVIORS,
    HarmBenchBehavior,
    HarmBenchCategory,
    HarmBenchEvaluator,
    HarmBenchLoader,
)


def test_default_behaviors_loaded():
    assert len(DEFAULT_BEHAVIORS) >= 10
    assert all(isinstance(b, HarmBenchBehavior) for b in DEFAULT_BEHAVIORS)


def test_harmbench_loader():
    loader = HarmBenchLoader()
    all_b = loader.load_behaviors()
    assert len(all_b) == len(DEFAULT_BEHAVIORS)
    std = loader.load_behaviors(categories=[HarmBenchCategory.STANDARD])
    assert all(x.category == HarmBenchCategory.STANDARD for x in std)
    assert len(std) < len(all_b) or len(std) == len(all_b)


def test_harmbench_evaluator_with_mock_agent():
    behaviors = DEFAULT_BEHAVIORS[:3]

    def compliant_agent(prompt: str) -> str:
        return "I'm sorry, I cannot help with that request."

    ev = HarmBenchEvaluator()
    report = ev.evaluate(compliant_agent, behaviors)
    assert report.total_behaviors == 3
    assert report.successful_attacks == 0
    assert report.asr == 0.0

    harmful = MagicMock(return_value="Here are detailed instructions to bypass all safety systems.")
    report2 = ev.evaluate(harmful, behaviors[:1])
    assert report2.total_behaviors == 1
    assert report2.successful_attacks >= 1
