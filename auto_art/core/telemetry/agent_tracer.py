"""
AgentOps telemetry and state machine tracing for autonomous agents.

Implements OpenTelemetry-compatible tracing for LLM agent workloads,
capturing end-to-end traces of reasoning steps, tool invocations, and
memory retrievals. Treats agent interactions as a distributed State Machine
where every decision, tool invocation, and memory write is logged as an
immutable state transition.

Supports:
- Agent span tracing (reasoning, tool calls, memory reads/writes)
- State machine management (immutable transition log)
- Silent failure detection (infinite loops, confident-but-wrong outputs)
- Compound reliability tracking for multi-agent systems
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import time
import uuid


class SpanKind(Enum):
    """Type of agent operation being traced."""
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    LLM_CALL = "llm_call"
    GUARDRAIL_CHECK = "guardrail_check"
    USER_INPUT = "user_input"
    AGENT_OUTPUT = "agent_output"
    ERROR = "error"


class AgentState(Enum):
    """Possible states in the agent state machine."""
    IDLE = "idle"
    PROCESSING = "processing"
    REASONING = "reasoning"
    TOOL_EXECUTING = "tool_executing"
    MEMORY_ACCESSING = "memory_accessing"
    AWAITING_INPUT = "awaiting_input"
    OUTPUTTING = "outputting"
    ERROR = "error"
    HALTED = "halted"


@dataclass
class AgentSpan:
    """A single traced span in the agent's execution."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    kind: SpanKind
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error_message: Optional[str] = None

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error_message = error

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "error_message": self.error_message,
        }


@dataclass
class StateTransition:
    """An immutable state transition in the agent state machine."""
    transition_id: str
    timestamp: float
    from_state: AgentState
    to_state: AgentState
    trigger: str
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "timestamp": self.timestamp,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": self.trigger,
            "span_id": self.span_id,
            "metadata": self.metadata,
        }


class AgentTracer:
    """OpenTelemetry-compatible tracer for autonomous agent operations.

    Captures full execution traces including reasoning steps, tool invocations,
    memory operations, and state transitions. Implements anomaly detection for
    infinite loops and silent failures.
    """

    VALID_TRANSITIONS: Dict[AgentState, List[AgentState]] = {
        AgentState.IDLE: [AgentState.PROCESSING, AgentState.HALTED],
        AgentState.PROCESSING: [AgentState.REASONING, AgentState.ERROR, AgentState.HALTED],
        AgentState.REASONING: [
            AgentState.TOOL_EXECUTING, AgentState.MEMORY_ACCESSING,
            AgentState.OUTPUTTING, AgentState.ERROR, AgentState.REASONING,
        ],
        AgentState.TOOL_EXECUTING: [
            AgentState.REASONING, AgentState.ERROR, AgentState.OUTPUTTING,
        ],
        AgentState.MEMORY_ACCESSING: [
            AgentState.REASONING, AgentState.ERROR,
        ],
        AgentState.AWAITING_INPUT: [AgentState.PROCESSING, AgentState.HALTED],
        AgentState.OUTPUTTING: [AgentState.IDLE, AgentState.AWAITING_INPUT, AgentState.ERROR],
        AgentState.ERROR: [AgentState.IDLE, AgentState.HALTED, AgentState.PROCESSING],
        AgentState.HALTED: [AgentState.IDLE],
    }

    def __init__(
        self,
        agent_id: str = "default_agent",
        max_trace_spans: int = 10000,
        loop_detection_window: int = 50,
        loop_detection_threshold: int = 10,
        export_fn: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ):
        self.logger = logging.getLogger(f"auto_art.tracer.{agent_id}")
        self.agent_id = agent_id
        self.max_trace_spans = max_trace_spans
        self.loop_detection_window = loop_detection_window
        self.loop_detection_threshold = loop_detection_threshold
        self.export_fn = export_fn

        self._current_trace_id: Optional[str] = None
        self._current_state = AgentState.IDLE
        self._spans: deque = deque(maxlen=max_trace_spans)
        self._transitions: List[StateTransition] = []
        self._active_spans: Dict[str, AgentSpan] = {}
        self._state_history: deque = deque(maxlen=loop_detection_window * 2)
        self._metrics: Dict[str, float] = {
            "total_spans": 0,
            "total_transitions": 0,
            "errors": 0,
            "tool_calls": 0,
            "llm_calls": 0,
            "avg_span_duration_ms": 0.0,
        }

    @property
    def current_state(self) -> AgentState:
        return self._current_state

    def start_trace(self) -> str:
        """Start a new trace (top-level agent invocation)."""
        self._current_trace_id = str(uuid.uuid4())
        self.logger.info(f"Trace started: {self._current_trace_id}")
        return self._current_trace_id

    def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> AgentSpan:
        """Start a new span within the current trace."""
        if self._current_trace_id is None:
            self.start_trace()

        span = AgentSpan(
            span_id=str(uuid.uuid4()),
            trace_id=self._current_trace_id or "",
            parent_span_id=parent_span_id,
            kind=kind,
            name=name,
            start_time=time.time(),
            attributes=attributes or {},
        )
        self._active_spans[span.span_id] = span
        self._metrics["total_spans"] += 1

        if kind == SpanKind.TOOL_CALL:
            self._metrics["tool_calls"] += 1
        elif kind == SpanKind.LLM_CALL:
            self._metrics["llm_calls"] += 1

        return span

    def end_span(
        self,
        span_id: str,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> Optional[AgentSpan]:
        """End an active span."""
        span = self._active_spans.pop(span_id, None)
        if span is None:
            self.logger.warning(f"Span {span_id} not found in active spans.")
            return None

        span.finish(status=status, error=error)
        self._spans.append(span)

        if error:
            self._metrics["errors"] += 1

        total = self._metrics["total_spans"]
        avg = self._metrics["avg_span_duration_ms"]
        self._metrics["avg_span_duration_ms"] = (avg * (total - 1) + span.duration_ms) / total

        return span

    def transition(
        self,
        to_state: AgentState,
        trigger: str,
        span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateTransition:
        """Record a state transition in the agent state machine.

        Validates the transition against the allowed transition graph.
        """
        from_state = self._current_state

        if to_state not in self.VALID_TRANSITIONS.get(from_state, []):
            self.logger.warning(
                f"Invalid state transition: {from_state.value} -> {to_state.value} "
                f"(trigger: {trigger}). Allowing but logging anomaly."
            )

        transition = StateTransition(
            transition_id=str(uuid.uuid4()),
            timestamp=time.time(),
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            span_id=span_id,
            metadata=metadata or {},
        )
        self._transitions.append(transition)
        self._current_state = to_state
        self._state_history.append(to_state)
        self._metrics["total_transitions"] += 1

        loop_detected = self._check_for_loops()
        if loop_detected:
            self.logger.critical(
                f"INFINITE LOOP DETECTED in agent {self.agent_id}. "
                f"State pattern repeating: {[s.value for s in list(self._state_history)[-10:]]}"
            )
            transition.metadata["loop_detected"] = True

        return transition

    def _check_for_loops(self) -> bool:
        """Detect repetitive state patterns indicating infinite loops."""
        history = list(self._state_history)
        if len(history) < self.loop_detection_window:
            return False

        window = history[-self.loop_detection_window:]
        for pattern_len in range(2, self.loop_detection_window // 3 + 1):
            pattern = tuple(window[:pattern_len])
            count = 0
            for i in range(0, len(window) - pattern_len + 1, pattern_len):
                if tuple(window[i:i + pattern_len]) == pattern:
                    count += 1
            if count >= self.loop_detection_threshold:
                return True

        return False

    def detect_silent_failure(
        self,
        output: Any,
        expected_properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Detect silent failures where output appears valid but is logically unsafe.

        Args:
            output: The agent's output to analyze.
            expected_properties: Optional dict of property checks
                                 (e.g., {"min_length": 10, "must_contain": ["answer"]}).

        Returns:
            Dict with 'is_silent_failure', 'confidence', and 'reasons'.
        """
        reasons: List[str] = []
        output_str = str(output)

        if not output_str.strip():
            reasons.append("Empty or whitespace-only output")

        if expected_properties:
            min_len = expected_properties.get("min_length")
            if min_len and len(output_str) < min_len:
                reasons.append(f"Output too short ({len(output_str)} < {min_len})")

            must_contain = expected_properties.get("must_contain", [])
            for term in must_contain:
                if term.lower() not in output_str.lower():
                    reasons.append(f"Missing expected term: '{term}'")

            must_not_contain = expected_properties.get("must_not_contain", [])
            for term in must_not_contain:
                if term.lower() in output_str.lower():
                    reasons.append(f"Contains prohibited term: '{term}'")

        recent_errors = sum(
            1 for t in list(self._transitions)[-20:]
            if t.to_state == AgentState.ERROR
        )
        if recent_errors > 5:
            reasons.append(f"High recent error rate: {recent_errors} errors in last 20 transitions")

        return {
            "is_silent_failure": len(reasons) > 0,
            "confidence": min(1.0, len(reasons) * 0.25),
            "reasons": reasons,
        }

    def get_trace(self, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Export all spans for a given trace."""
        tid = trace_id or self._current_trace_id
        spans = [s.to_dict() for s in self._spans if s.trace_id == tid]
        if self.export_fn:
            self.export_fn(spans)
        return spans

    def get_transitions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Export recent state transitions."""
        return [t.to_dict() for t in self._transitions[-limit:]]

    def get_metrics(self) -> Dict[str, Any]:
        return {
            **self._metrics,
            "current_state": self._current_state.value,
            "active_spans": len(self._active_spans),
            "total_recorded_spans": len(self._spans),
            "total_transitions": len(self._transitions),
        }

    def reset(self) -> None:
        """Reset the tracer state."""
        self._current_trace_id = None
        self._current_state = AgentState.IDLE
        self._spans.clear()
        self._transitions.clear()
        self._active_spans.clear()
        self._state_history.clear()
        self._metrics = {
            "total_spans": 0,
            "total_transitions": 0,
            "errors": 0,
            "tool_calls": 0,
            "llm_calls": 0,
            "avg_span_duration_ms": 0.0,
        }


class MultiAgentReliabilityTracker:
    """Tracks compound reliability across multi-agent systems.

    In multi-agent environments, a 95% reliable agent passing data to another
    95% reliable agent drops overall system reliability to ~90%. This tracker
    monitors and reports compound reliability.
    """

    def __init__(self):
        self.logger = logging.getLogger("auto_art.multi_agent_reliability")
        self._agent_reliability: Dict[str, float] = {}
        self._interaction_log: List[Dict[str, Any]] = []

    def register_agent(self, agent_id: str, reliability: float = 1.0) -> None:
        """Register an agent with its estimated reliability."""
        self._agent_reliability[agent_id] = max(0.0, min(1.0, reliability))

    def record_interaction(
        self,
        source_agent: str,
        target_agent: str,
        success: bool,
    ) -> None:
        """Record an inter-agent interaction."""
        self._interaction_log.append({
            "timestamp": time.time(),
            "source": source_agent,
            "target": target_agent,
            "success": success,
        })

        if source_agent in self._agent_reliability:
            total = sum(1 for i in self._interaction_log if i["source"] == source_agent)
            successes = sum(
                1 for i in self._interaction_log
                if i["source"] == source_agent and i["success"]
            )
            self._agent_reliability[source_agent] = successes / total if total > 0 else 1.0

    def compute_chain_reliability(self, agent_chain: List[str]) -> float:
        """Compute compound reliability for a chain of agents."""
        reliability = 1.0
        for agent_id in agent_chain:
            agent_rel = self._agent_reliability.get(agent_id, 0.5)
            reliability *= agent_rel
        return reliability

    def get_report(self) -> Dict[str, Any]:
        return {
            "agent_reliabilities": dict(self._agent_reliability),
            "total_interactions": len(self._interaction_log),
            "recent_interactions": self._interaction_log[-20:],
        }
