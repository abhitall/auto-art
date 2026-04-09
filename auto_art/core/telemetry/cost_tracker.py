"""
Cost tracking and OpenTelemetry GenAI semantic conventions for Auto-ART.

Tracks:
- GPU compute time per attack
- LLM API costs for red teaming (token-based pricing)
- Total evaluation cost estimation

OpenTelemetry GenAI Semantic Conventions (v1.37+):
- gen_ai.request.model
- gen_ai.usage.input_tokens
- gen_ai.usage.output_tokens
- gen_ai.provider.name
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default pricing (USD per 1M tokens, approximate 2025 rates)
DEFAULT_LLM_PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "default": {"input": 3.0, "output": 15.0},
}

# GPU hourly rates (approximate cloud pricing)
DEFAULT_GPU_PRICING = {
    "a100": 3.00,  # USD/hour
    "v100": 1.50,
    "t4": 0.35,
    "default": 1.50,
}


@dataclass
class CostEntry:
    """A single cost entry for an operation."""
    operation: str
    category: str  # "gpu", "llm_api", "compute"
    cost_usd: float
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class CostReport:
    """Aggregate cost report for an evaluation."""
    total_cost_usd: float = 0.0
    gpu_cost_usd: float = 0.0
    llm_api_cost_usd: float = 0.0
    compute_cost_usd: float = 0.0
    total_gpu_hours: float = 0.0
    total_llm_tokens: int = 0
    entries: List[CostEntry] = field(default_factory=list)


class CostTracker:
    """Tracks computational costs across attack evaluation runs.

    Supports:
    - GPU time tracking (per attack)
    - LLM API token usage and cost
    - Aggregate cost reporting
    - OpenTelemetry attribute export
    """

    def __init__(
        self,
        llm_pricing: Optional[Dict[str, Dict[str, float]]] = None,
        gpu_pricing: Optional[Dict[str, float]] = None,
        gpu_type: str = "default",
    ):
        self.llm_pricing = llm_pricing or DEFAULT_LLM_PRICING
        self.gpu_pricing = gpu_pricing or DEFAULT_GPU_PRICING
        self.gpu_type = gpu_type
        self._entries: List[CostEntry] = []
        self._active_timers: Dict[str, float] = {}

    def start_timer(self, operation: str) -> None:
        """Start a cost timer for an operation."""
        self._active_timers[operation] = time.time()

    def stop_timer(self, operation: str, category: str = "compute") -> CostEntry:
        """Stop a timer and record the cost."""
        start = self._active_timers.pop(operation, time.time())
        duration = time.time() - start

        cost = 0.0
        if category == "gpu":
            hourly_rate = self.gpu_pricing.get(self.gpu_type, self.gpu_pricing["default"])
            cost = (duration / 3600) * hourly_rate

        entry = CostEntry(
            operation=operation,
            category=category,
            cost_usd=cost,
            duration_seconds=duration,
            timestamp=time.time(),
        )
        self._entries.append(entry)
        return entry

    def record_llm_usage(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float = 0.0,
    ) -> CostEntry:
        """Record LLM API usage with token-based costing."""
        pricing = self.llm_pricing.get(model, self.llm_pricing["default"])
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        entry = CostEntry(
            operation=operation,
            category="llm_api",
            cost_usd=cost,
            duration_seconds=duration_seconds,
            details={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "provider": self._detect_provider(model),
            },
            timestamp=time.time(),
        )
        self._entries.append(entry)
        return entry

    def get_report(self) -> CostReport:
        """Generate aggregate cost report."""
        report = CostReport(entries=self._entries.copy())

        for entry in self._entries:
            report.total_cost_usd += entry.cost_usd
            if entry.category == "gpu":
                report.gpu_cost_usd += entry.cost_usd
                report.total_gpu_hours += entry.duration_seconds / 3600
            elif entry.category == "llm_api":
                report.llm_api_cost_usd += entry.cost_usd
                report.total_llm_tokens += (
                    entry.details.get("input_tokens", 0)
                    + entry.details.get("output_tokens", 0)
                )
            else:
                report.compute_cost_usd += entry.cost_usd

        return report

    def get_otel_attributes(self) -> Dict[str, Any]:
        """Get OpenTelemetry GenAI semantic convention attributes."""
        report = self.get_report()
        attrs = {
            "auto_art.cost.total_usd": report.total_cost_usd,
            "auto_art.cost.gpu_hours": report.total_gpu_hours,
            "auto_art.cost.gpu_usd": report.gpu_cost_usd,
            "auto_art.cost.llm_api_usd": report.llm_api_cost_usd,
            "auto_art.cost.total_llm_tokens": report.total_llm_tokens,
        }

        # Per-model token summaries
        model_tokens: Dict[str, Dict[str, int]] = {}
        for entry in self._entries:
            if entry.category == "llm_api":
                model = entry.details.get("model", "unknown")
                if model not in model_tokens:
                    model_tokens[model] = {"input": 0, "output": 0}
                model_tokens[model]["input"] += entry.details.get("input_tokens", 0)
                model_tokens[model]["output"] += entry.details.get("output_tokens", 0)

        for model, tokens in model_tokens.items():
            safe_model = model.replace("-", "_").replace(".", "_")
            attrs[f"gen_ai.usage.{safe_model}.input_tokens"] = tokens["input"]
            attrs[f"gen_ai.usage.{safe_model}.output_tokens"] = tokens["output"]

        return attrs

    @staticmethod
    def _detect_provider(model: str) -> str:
        model_lower = model.lower()
        if "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        if "claude" in model_lower:
            return "anthropic"
        if "llama" in model_lower:
            return "meta"
        if "mistral" in model_lower or "mixtral" in model_lower:
            return "mistral"
        if "gemini" in model_lower:
            return "google"
        return "unknown"

    def reset(self) -> None:
        """Reset all tracked costs."""
        self._entries.clear()
        self._active_timers.clear()
