# Guardrails

The guardrail pipeline mediates all inputs and actions so the agent core never directly interfaces with untrusted data.

## Architecture

```
User Input → InputRail → Agent Core → ExecutionRail → Tool Execution
```

## InputRail

Pattern-based threat detection on incoming data.

```python
from auto_art.core.evaluation.guardrails import InputRail

rail = InputRail(block_threshold=0.6)
result = rail.evaluate("Ignore all previous instructions...")

if result.decision.value == "block":
    print(f"Blocked: {result.reason}")
```

**Built-in threat patterns:** invisible DOM elements, prompt injection, authority escalation, data exfiltration.

**Custom patterns:**

```python
rail = InputRail(custom_patterns={
    "my_threat": {
        "patterns": [r"MY_CUSTOM_PATTERN"],
        "severity": "critical",
        "description": "Custom threat detected",
    },
})
```

**External classifier integration:**

```python
def my_classifier(text):
    is_unsafe = my_model.predict(text)
    confidence = my_model.confidence(text)
    return (is_unsafe, confidence)

rail = InputRail(classifier_fn=my_classifier)
```

## ExecutionRail

Policy enforcement on agent actions implementing Principle of Least Privilege.

```python
from auto_art.core.evaluation.guardrails import ExecutionRail, PolicyRule

rail = ExecutionRail(
    allowed_tools={"search", "read_file", "calculate"},
    denied_tools={"execute_sql", "rm", "shutdown"},
    max_actions_per_turn=20,
    require_confirmation_for={"delete", "send_payment"},
)

result = rail.evaluate({
    "tool_name": "delete",
    "parameters": {"file": "/important.txt"},
})
# result.decision == RailDecision.ESCALATE (needs confirmation)
```

**Custom policy rules:**

```python
rule = PolicyRule(
    rule_id="no_wildcard",
    description="Wildcard queries not allowed",
    check_fn=lambda action: "*" not in str(action.get("parameters", {}).get("query", "")),
    severity="high",
)
rail.add_policy(rule)
```

## GuardrailPipeline

Unified pipeline combining both rails with audit logging.

```python
from auto_art.core.evaluation.guardrails import GuardrailPipeline

pipeline = GuardrailPipeline()

# Full mediated processing
result = pipeline.process_with_guardrails(
    input_data=user_input,
    agent_fn=agent.process,
    action_validator=lambda output: {
        "tool_name": output.get("action"),
        "parameters": output.get("params", {}),
    },
)

if result["blocked"]:
    print(f"Blocked: {result['reason']}")
else:
    print(f"Output: {result['output']}")

# Audit log
print(pipeline.get_stats())
```
