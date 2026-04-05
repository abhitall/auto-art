"""Tests for cost tracking and OTel attribute generation."""

import time
import pytest
from auto_art.core.telemetry.cost_tracker import CostTracker, CostReport


class TestCostTracker:
    def test_gpu_timing(self):
        tracker = CostTracker(gpu_type="t4")
        tracker.start_timer("fgsm_attack")
        time.sleep(0.01)
        entry = tracker.stop_timer("fgsm_attack", category="gpu")

        assert entry.category == "gpu"
        assert entry.duration_seconds > 0
        assert entry.cost_usd >= 0

    def test_llm_usage_tracking(self):
        tracker = CostTracker()
        entry = tracker.record_llm_usage(
            operation="pair_attack",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        assert entry.category == "llm_api"
        assert entry.cost_usd > 0
        assert entry.details["model"] == "gpt-4o"
        assert entry.details["provider"] == "openai"

    def test_cost_report(self):
        tracker = CostTracker()
        tracker.record_llm_usage("op1", "gpt-4o", 1000, 500)
        tracker.record_llm_usage("op2", "claude-3-sonnet", 2000, 1000)

        report = tracker.get_report()
        assert isinstance(report, CostReport)
        assert report.total_cost_usd > 0
        assert report.llm_api_cost_usd > 0
        assert report.total_llm_tokens == 4500
        assert len(report.entries) == 2

    def test_otel_attributes(self):
        tracker = CostTracker()
        tracker.record_llm_usage("test", "gpt-4o", 1000, 500)

        attrs = tracker.get_otel_attributes()
        assert "auto_art.cost.total_usd" in attrs
        assert "auto_art.cost.total_llm_tokens" in attrs
        assert attrs["auto_art.cost.total_llm_tokens"] == 1500

    def test_provider_detection(self):
        tracker = CostTracker()
        assert tracker._detect_provider("gpt-4o") == "openai"
        assert tracker._detect_provider("claude-3-opus") == "anthropic"
        assert tracker._detect_provider("llama-3") == "meta"
        assert tracker._detect_provider("gemini-pro") == "google"
        assert tracker._detect_provider("unknown-model") == "unknown"

    def test_reset(self):
        tracker = CostTracker()
        tracker.record_llm_usage("op1", "gpt-4o", 1000, 500)
        tracker.reset()
        report = tracker.get_report()
        assert report.total_cost_usd == 0
        assert len(report.entries) == 0
