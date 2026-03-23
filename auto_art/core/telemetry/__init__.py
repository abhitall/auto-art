"""
Telemetry and observability for Auto-ART.

Two layers:
- **TelemetryProvider**: Production-grade OpenTelemetry integration with
  traces, metrics, and structured JSON logs. Install with ``pip install auto-art[telemetry]``.
- **AgentTracer**: Lightweight agent state-machine tracer for autonomous
  agent evaluation (no external dependencies).
"""

from .agent_tracer import (
    AgentTracer,
    AgentSpan,
    StateTransition,
    SpanKind,
    AgentState,
    MultiAgentReliabilityTracker,
)
from .provider import (
    TelemetryProvider,
    TelemetryConfig,
    ExportTarget,
)

__all__ = [
    "TelemetryProvider",
    "TelemetryConfig",
    "ExportTarget",
    "AgentTracer",
    "AgentSpan",
    "StateTransition",
    "SpanKind",
    "AgentState",
    "MultiAgentReliabilityTracker",
]
