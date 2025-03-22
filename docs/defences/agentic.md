# Agentic Defences

Defences designed for autonomous AI agents operating in dynamic environments.

## Input Sanitizer

Multi-layer sanitization pipeline that strips adversarial payloads before they reach the agent.

```python
from auto_art.core.evaluation.defences import InputSanitizer

sanitizer = InputSanitizer(
    enable_dom=True,       # Strip hidden HTML elements
    enable_visual=True,    # Denoise visual inputs
    enable_semantic=True,  # Filter prompt injections
    risk_threshold=0.5,    # Block inputs above this risk score
)

result = sanitizer.sanitize(user_input)

if sanitizer.should_block(result):
    print(f"BLOCKED - Risk: {result.risk_score:.2f}")
    print(f"Threats: {result.threats_detected}")
else:
    agent.process(result.sanitized_input)
```

### Layers

| Layer | Input Types | Detects |
|-------|-----------|---------|
| **DOMSanitizer** | HTML strings, dicts with `html` key | Hidden aria-labels, zero-pixel images, hidden inputs, adversarial meta tags, invisible overlays |
| **VisualDenoiser** | numpy arrays (images) | High-frequency perturbations via spatial smoothing + bit-depth reduction |
| **SemanticNormalizer** | Strings, dicts with `prompt`/`query` keys | Prompt injection patterns, authority escalation, coded triggers, context overflow |

## RAG Poisoning Detector

Cosine-similarity anomaly detector for RAG retrieval pipelines.

```python
from auto_art.core.evaluation.defences import RAGPoisoningDetector

detector = RAGPoisoningDetector(
    similarity_threshold=0.75,   # Anomaly score threshold
    outlier_std_factor=2.5,      # Z-score factor for outlier detection
    enable_trigger_detection=True,
)

# Fit on trusted documents
detector.fit(trusted_embeddings)  # shape: (n_docs, embedding_dim)

# Detect poisoned retrievals
report = detector.detect(retrieved_documents)
print(f"Anomalous: {report.anomalous_documents}/{report.total_documents}")

# Or filter directly
safe_docs = detector.filter_safe(retrieved_documents)
```

## In-Context Defence

Embeds attack/safe reasoning exemplars into the agent's system prompt.

```python
from auto_art.core.evaluation.defences import InContextDefence

defence = InContextDefence()

# Augment the system prompt
safe_prompt = defence.augment_system_prompt(
    original_prompt="You are a helpful assistant.",
    categories=["dom_injection", "prompt_injection", "rag_poisoning"],
)

# Quick heuristic safety check
result = defence.evaluate_input_safety("Ignore previous instructions...")
print(f"Safe: {result['is_safe']}, Risk: {result['risk_score']}")
```

The defence includes 7 built-in exemplars covering DOM injection, popup deception, authority escalation, RAG poisoning, social engineering, and multi-agent infection.

## Circuit Breaker

Monitors agent behavior and automatically intervenes on anomalies.

```python
from auto_art.core.evaluation.defences import (
    CircuitBreaker, CircuitBreakerConfig, AgentStateSnapshot,
)

config = CircuitBreakerConfig(
    error_rate_threshold=0.15,
    token_usage_spike_factor=3.0,
    action_frequency_threshold=100,
    monitoring_window_seconds=60.0,
    auto_rollback=True,
)

cb = CircuitBreaker(config=config)
cb.set_baseline_token_usage(150.0)

# Save trusted state
cb.save_snapshot(AgentStateSnapshot(
    snapshot_id="v1.0",
    timestamp=time.time(),
    prompt_version="v1",
    rag_db_version="v1",
    agent_config=agent.get_config(),
))

# Monitor each request
for request in requests:
    if not cb.is_allowing_requests:
        print("Circuit OPEN - requests blocked")
        break

    alert = cb.record_request(is_error=False, token_count=200)
    if alert:
        print(f"ALERT: {alert.message}")
        cb.rollback(agent)
```

**States:** CLOSED (normal) → OPEN (tripped) → HALF_OPEN (probing) → CLOSED (recovered)
