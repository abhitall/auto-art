# OWASP LLM Top 10 (2025) Coverage

Auto-ART maps its attack and defence capabilities to the [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/).

## Coverage Summary

| ID | Category | Coverage | Attacks | Defences |
|------|----------|----------|---------|----------|
| LLM01 | Prompt Injection | FULL | 7 | 4 |
| LLM02 | Sensitive Information Disclosure | FULL | 4 | 4 |
| LLM03 | Supply Chain Vulnerabilities | FULL | 5 | 3 |
| LLM04 | Data and Model Poisoning | FULL | 5 | 3 |
| LLM05 | Improper Output Handling | PARTIAL | 2 | 3 |
| LLM06 | Excessive Agency | FULL | 2 | 4 |
| LLM07 | System Prompt Leakage | FULL | 2 | 3 |
| LLM08 | Vector and Embedding Weaknesses | FULL | 1 | 2 |
| LLM09 | Misinformation | PARTIAL | 2 | 3 |
| LLM10 | Unbounded Consumption | FULL | 2 | 5 |

**Overall: 90% coverage** (8 full, 2 partial)

## Generating the Report Programmatically

```python
from auto_art.core.evaluation.owasp_mapping import (
    get_coverage_report,
    get_coverage_markdown,
)

# Structured dict
report = get_coverage_report()
print(f"Coverage: {report['coverage_percentage']:.0f}%")

# Formatted Markdown
print(get_coverage_markdown())
```

## Detailed Mapping

### LLM01: Prompt Injection

**Attacks:**
- `InContextInjectionAttack` (authority_escalation, contradiction, context_dilution)
- `AdvWebDOMAttack` (invisible DOM injection with adversarial prompts)
- `RedTeamLLM` (prompt_injection, authority_escalation, jailbreak categories)

**Defences:**
- `InputSanitizer` / `SemanticNormalizer` -- pattern-based injection filtering
- `InputRail` -- threat pattern detection at the guardrail layer
- `InContextDefence` -- teaches agent to reject injections via exemplars
- `GuardrailPipeline` -- full mediation pipeline

### LLM06: Excessive Agency

**Attacks:**
- `RedTeamLLM` (goal_hijacking) -- tests if agent can be redirected
- `InContextInjectionAttack` (authority_escalation) -- tests privilege escalation

**Defences:**
- `ExecutionRail` with `allowed_tools` -- whitelist-only tool access
- `ExecutionRail` with `max_actions_per_turn` -- rate limiting
- `ExecutionRail` with `require_confirmation_for` -- human-in-the-loop for sensitive ops
- `CircuitBreaker` -- automated intervention on action frequency anomalies

### LLM10: Unbounded Consumption

**Attacks:**
- `InContextInjectionAttack` (context_dilution) -- floods context window
- `UniversalAdversarialPatchAttack` -- forces premature task completion

**Defences:**
- `CircuitBreaker` -- monitors token usage spikes, error rates, action frequency
- `ExecutionRail` -- limits actions per turn
- `AgentTracer` -- detects infinite loops via state pattern matching
