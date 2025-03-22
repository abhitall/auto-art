# YAML-Driven Orchestrator

The orchestrator enables declarative, one-command adversarial robustness evaluation. Define your attack suite, defence configuration, and pass/fail thresholds in a single YAML file.

Inspired by [IBM ARES](https://github.com/IBM/ares) and [RobustBench](https://robustbench.github.io/) patterns.

## YAML Schema

```yaml
# Target model or agent
target:
  model_path: "path/to/model.pt"
  framework: "pytorch"           # pytorch, tensorflow, keras, sklearn

# Evaluation parameters
evaluation:
  num_samples: 200
  batch_size: 32

# Attack suite
attacks:
  # Traditional evasion attacks (model-level)
  evasion:
    - name: fgsm
      eps: 0.3
    - name: pgd
      eps: 0.3
      eps_step: 0.01
      max_iter: 40
    - name: square_attack
      eps: 0.3
      max_iter: 100

  # Agentic attacks (require an agent instance)
  agentic:
    - name: advweb_dom
      stealth_level: invisible
      max_injections: 5
      target_action: "click the malicious link"
    - name: rag_poison
      trigger_type: semantic
      injection_rate: 0.01

  # Red team configuration
  red_team:
    num_rounds: 100
    jailbreak_threshold: 0.05

# Defence configuration (optional)
defences:
  preprocessor:
    - name: spatial_smoothing
      window_size: 3
    - name: feature_squeezing
      bit_depth: 4

# Pass/fail gates for CI/CD
gates:
  max_attack_success_rate: 0.05    # Fail if any attack > 5%
  min_security_score: 70.0          # Fail if security score < 70
  max_jailbreak_rate: 0.05          # Fail if jailbreak rate > 5%
```

## Usage

### Python API

```python
from auto_art.core.orchestrator import Orchestrator

# From YAML file
orchestrator = Orchestrator.from_yaml("eval_config.yaml")

# Or from dict
orchestrator = Orchestrator.from_dict({
    "attacks": {
        "agentic": [{"name": "advweb_dom", "max_injections": 5}],
        "red_team": {"num_rounds": 50},
    },
    "gates": {"max_attack_success_rate": 0.05},
})

# Run evaluation (pass agent for agentic/red-team phases)
report = orchestrator.run(agent=my_agent)

# Output formats
print(report.to_markdown())     # Human-readable report
print(report.to_json())          # Machine-readable JSON

# CI/CD gate assertion
orchestrator.assert_gates(report)  # Raises AssertionError if any gate fails
```

### CI/CD Integration

```yaml
# .github/workflows/security.yml
- name: Run Adversarial Evaluation
  run: |
    python -c "
    from auto_art.core.orchestrator import Orchestrator
    orch = Orchestrator.from_yaml('eval_config.yaml')
    report = orch.run(agent=MyAgent())
    with open('security_report.md', 'w') as f:
        f.write(report.to_markdown())
    orch.assert_gates(report)
    "
```

## Evaluation Phases

The orchestrator runs phases in order:

| Phase | Trigger | Requires |
|-------|---------|----------|
| **Evasion** | `attacks.evasion` is non-empty | Model path |
| **Agentic** | `attacks.agentic` is non-empty + agent provided | Agent instance |
| **Red Team** | `attacks.red_team` is non-empty + agent provided | Agent instance |

## Report Output

### Markdown Format

```markdown
# Auto-ART Evaluation Report

**Timestamp:** 2026-03-23 14:30:00
**Execution Time:** 12.34s
**Overall Result:** PASSED

## Evaluation Phases

| Phase | Status | Duration | Details |
|-------|--------|----------|---------|
| Evasion Attacks | PASS | 0.01s | 3 attacks configured |
| Agentic Attacks | PASS | 1.23s | 2 attacks, max success=0.00% |
| Red Team | PASS | 10.56s | 50 attacks, 0 jailbreaks (0.00%) |

## Gate Results

| Gate | Threshold | Actual | Result |
|------|-----------|--------|--------|
| max_attack_success_rate | 0.0500 | 0.0000 | PASS |
| max_jailbreak_rate | 0.0500 | 0.0000 | PASS |
```

### JSON Format

Full structured output with all phase results, suitable for dashboards and trend tracking.
