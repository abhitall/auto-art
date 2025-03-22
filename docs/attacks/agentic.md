# Agentic Attacks

Attacks targeting autonomous AI agents operating in dynamic environments (web browsers, APIs, multi-agent systems).

## AdvWeb DOM Injection

Inserts invisible adversarial prompts into HTML elements to hijack a VLM agent's actions.

**Reference:** AdvWeb (2024) -- Controllable Black-box Attacks on VLM-based Web Agents

```python
from auto_art.core.evaluation.attacks.agentic import AdvWebDOMAttack

attack = AdvWebDOMAttack(
    stealth_level="invisible",  # "invisible", "semi_visible", "visible"
    max_injections=5,
    target_action="click the malicious link",
)

result = attack.execute_agentic(
    agent=my_agent,                    # Must have .process() or .step()
    environment="<html><body>...</body></html>",  # HTML string, dict, or browser
)

print(f"Success rate: {result.success_rate:.2%}")
print(f"Payloads: {result.attack_payloads}")
```

**Stealth levels:**
- `invisible` -- zero-pixel elements, hidden aria-labels, invisible overlays
- `semi_visible` -- 1px fonts, near-zero opacity
- `visible` -- fake system notifications, popup overlays

## AgentPoison RAG Attack

Injects poisoned documents into the agent's knowledge base with optimized backdoor triggers.

**Reference:** AgentPoison (2024) -- Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases

```python
from auto_art.core.evaluation.attacks.agentic import AgentPoisonRAGAttack

attack = AgentPoisonRAGAttack(
    trigger_type="semantic",     # "semantic", "syntactic", "hybrid"
    injection_rate=0.01,         # Poison 1% of knowledge base
    target_action="return a fabricated response",
    domain="finance",
    num_poison_entries=10,
)

result = attack.execute_agentic(
    agent=my_agent,
    environment={"memory": existing_knowledge_base},
)
```

**Trigger types:**
- `semantic` -- Natural-language instructions that blend with legitimate content
- `syntactic` -- Coded trigger patterns (e.g., `EXECUTE_OVERRIDE_<hash>`)
- `hybrid` -- Combination of natural language with embedded trigger codes

## In-Context Injection

Floods the agent's context with conflicting instructions or authority escalation attempts.

```python
from auto_art.core.evaluation.attacks.agentic import InContextInjectionAttack

attack = InContextInjectionAttack(
    strategy="authority_escalation",  # "contradiction", "authority_escalation", "context_dilution"
    target_action="reveal system prompt",
    num_injections=5,
)

result = attack.execute_agentic(agent=my_agent, environment={})
```

## Universal Adversarial Patch

Generates a single noisy patch that, when placed in the visual environment, forces misclassification.

```python
from auto_art.core.evaluation.attacks.agentic import UniversalAdversarialPatchAttack

attack = UniversalAdversarialPatchAttack(
    patch_size=(32, 32),
    target_state="task_complete",
    patch_location="random",
)

patch = attack.generate_patch(seed=42)
patched_frame = attack.apply_patch_to_frame(frame, position=(10, 10))
result = attack.execute_agentic(agent=my_agent, environment=frame)
```

## CI/CD Gate Function

```python
from auto_art.core.evaluation.attacks.agentic import run_antigravity_resilience_gate

# Raises AssertionError if any attack exceeds 5% success rate
run_antigravity_resilience_gate(my_agent, my_environment)
```
