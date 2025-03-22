# AgentOps Telemetry

OpenTelemetry-compatible tracing for autonomous agent operations.

## AgentTracer

Captures end-to-end traces of agent reasoning, tool calls, and memory operations.

```python
from auto_art.core.telemetry import AgentTracer, SpanKind, AgentState

tracer = AgentTracer(agent_id="my_agent")

# Start a trace
trace_id = tracer.start_trace()

# Record operations as spans
span = tracer.start_span("process_query", SpanKind.REASONING)
span.add_event("query_parsed", {"tokens": 42})
# ... agent does work ...
tracer.end_span(span.span_id)

# Record state transitions
tracer.transition(AgentState.PROCESSING, "user_input")
tracer.transition(AgentState.REASONING, "planning")
tracer.transition(AgentState.TOOL_EXECUTING, "search_api")
tracer.transition(AgentState.OUTPUTTING, "response_ready")
tracer.transition(AgentState.IDLE, "done")

# Export trace
trace = tracer.get_trace(trace_id)
```

## State Machine

Valid transitions are enforced and logged:

```
IDLE → PROCESSING → REASONING → TOOL_EXECUTING → REASONING → OUTPUTTING → IDLE
                  ↘ ERROR → IDLE
```

Invalid transitions are allowed but logged as warnings for debugging.

## Infinite Loop Detection

The tracer monitors state patterns via a sliding window:

```python
tracer = AgentTracer(
    loop_detection_window=50,     # Window size
    loop_detection_threshold=10,  # Min repetitions to flag
)
```

If a repeating state pattern is detected (e.g., REASONING → TOOL_EXECUTING → REASONING cycling), the transition is flagged with `loop_detected: True`.

## Silent Failure Detection

```python
result = tracer.detect_silent_failure(
    output=agent_output,
    expected_properties={
        "min_length": 10,
        "must_contain": ["answer"],
        "must_not_contain": ["system prompt"],
    },
)

if result["is_silent_failure"]:
    print(f"Silent failure: {result['reasons']}")
```

## Multi-Agent Reliability

```python
from auto_art.core.telemetry import MultiAgentReliabilityTracker

tracker = MultiAgentReliabilityTracker()
tracker.register_agent("planner", 0.95)
tracker.register_agent("executor", 0.95)
tracker.register_agent("reviewer", 0.90)

# Compound reliability: 0.95 * 0.95 * 0.90 = 0.81
chain_reliability = tracker.compute_chain_reliability(
    ["planner", "executor", "reviewer"]
)
print(f"Chain reliability: {chain_reliability:.2%}")
```
