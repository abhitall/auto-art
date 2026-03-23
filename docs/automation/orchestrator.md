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

# Evasion execution strategy (optional)
# sequential (default) | adaptive | parallel — see docs/automation/adaptive.md
# and docs/automation/parallel.md. Adaptive and parallel are mutually exclusive.
execution:
  mode: sequential
  max_workers: 4
  gpu_workers: 1
  timeout_per_attack: 300.0
  budget: 3600.0
  success_threshold: 0.5
  escalate_if_below: 0.1

# Report serialization targets for CI dashboards (orchestrator always exposes
# to_json / to_markdown / to_sarif on the report object; this documents intent)
output:
  formats: [json, markdown, sarif]

# Attack suite
attacks:
  # Evasion (model-level; requires target.model_path)
  evasion:
    - name: fgsm
      eps: 0.3
    - name: bim
      eps: 0.3
      eps_step: 0.02
      max_iter: 10
    - name: pgd
      eps: 0.3
      eps_step: 0.01
      max_iter: 40
    - name: auto_pgd
      eps: 0.3
      max_iter: 20
    - name: autoattack
      eps: 0.3
    - name: carlini_wagner_l2
      eps: 0.3
    - name: adversarial_patch
      eps: 0.3
    - name: elastic_net
      eps: 0.3
    - name: jsma
      theta: 1.0
    - name: deepfool
      max_iter: 50
    - name: boundary_attack
      eps: 0.3
    - name: square_attack
      eps: 0.3
      max_iter: 100
    - name: hopskipjump
      max_iter: 50
    - name: simba
      eps: 0.3
    - name: zoo
      max_iter: 40
    - name: shadow_attack
      eps: 0.3
    - name: wasserstein
      eps: 0.3
    - name: brendel_bethge
      eps: 0.3
    - name: composite
      eps: 0.3
    - name: geoda
      max_iter: 50
    # Also supported: newtonfool, universal_perturbation, decision_tree_attack,
    # pixel_attack, threshold_attack, spatial_transformation, feature_adversaries,
    # overload, and other wrappers registered in AttackGenerator.

  # Poisoning (YAML scheduling; full runs need model + training pipeline)
  poisoning:
    - name: backdoor
      poison_rate: 0.05
    - name: clean_label
      poison_rate: 0.02
    - name: feature_collision
      poison_rate: 0.03
    - name: gradient_matching
      poison_rate: 0.03

  # Model extraction
  extraction:
    - name: copycat_cnn
      epochs: 10
    - name: knockoff_nets
      budget: 1000
    - name: functionally_equivalent_extraction
      queries: 5000

  # Privacy / inference attacks
  inference:
    - name: membership_inference_bb
    - name: attribute_inference_bb
    - name: model_inversion_miface

  # Agentic attacks (require an agent instance at run())
  agentic:
    - name: advweb_dom
      stealth_level: invisible
      max_injections: 5
      target_action: "click the malicious link"
    - name: rag_poison
      trigger_type: semantic
      injection_rate: 0.01

  # Red team configuration (nested under attacks in YAML; requires agent at run())
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
sarif_json = report.to_sarif()   # SARIF 2.1.0 for code scanning platforms

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
| **Evasion** | `attacks.evasion` non-empty | Model path |
| **Poisoning** | `attacks.poisoning` non-empty | Model path + training data |
| **Extraction** | `attacks.extraction` non-empty | Model path |
| **Inference** | `attacks.inference` non-empty | Model path |
| **Defence Evaluation** | `defences` non-empty | Model path |
| **Agentic** | `attacks.agentic` non-empty + agent | Agent instance |
| **Red Team** | `attacks.red_team` non-empty + agent | Agent instance |

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
| Evasion Attacks | PASS | 8.42s | 5 attacks, max_success=2.40%, avg=1.10% |
| Poisoning Attacks | PASS | 0.02s | 2 poisoning attacks configured |
| Extraction Attacks | PASS | 0.01s | 1 extraction attack configured |
| Inference Attacks | PASS | 0.01s | 1 inference attack configured |
| Defence Evaluation | PASS | 0.15s | 4 defences evaluated |
| Agentic Attacks | PASS | 1.23s | 2 attacks, max success=0.00% |
| Red Team | PASS | 10.56s | 50 attacks, 0 jailbreaks (0.00%) |

## Gate Results

| Gate | Threshold | Actual | Result |
|------|-----------|--------|--------|
| evasion_max_attack_success_rate | 0.0500 | 0.0240 | PASS |
| agentic_max_attack_success_rate | 0.0500 | 0.0000 | PASS |
| max_jailbreak_rate | 0.0500 | 0.0000 | PASS |
| min_security_score | 70.0000 | 75.0000 | PASS |
```

### JSON Format

Full structured output with all phase results, suitable for dashboards and trend tracking.

## SARIF Output

`OrchestratorReport.to_sarif()` emits **SARIF 2.1.0** JSON that third-party security workflows can ingest. It is compatible with [GitHub Code Scanning](https://docs.github.com/en/code-security/code-scanning/integrating-with-code-scanning/uploading-a-sarif-file-to-github), GitLab SAST, and Azure DevOps SARIF-based pipelines. Each attack (or defence) result from the report phases becomes a SARIF rule plus a result row, including success rates and status when present.

Example:

```python
from pathlib import Path

sarif_json = report.to_sarif()
Path("results.sarif").write_text(sarif_json, encoding="utf-8")
```

## Adaptive Attack Selection

Auto-ART includes an **AdaptiveAttackSelector** (`auto_art.core.adaptive`) inspired by AutoRedTeamer-style **memory-guided** prioritization: cheap attacks run first, and the engine **escalates across tiers** (fast gradient → iterative PGD/AutoAttack → black-box Square/HopSkipJump/ZOO → advanced elastic-net/JSMA/Brendel & Bethge, etc.) while respecting a time **budget**. Historical outcomes are stored in **persistent attack memory** (default `~/.auto_art/attack_memory.json`) so later runs favor attacks that worked well on similar model families.

```python
from auto_art.core.adaptive import AdaptiveAttackSelector

selector = AdaptiveAttackSelector(budget_seconds=3600.0)
plan = selector.select_attacks(model_arch="resnet18", max_attacks=8)
# plan is an ordered list of {name, tier, priority, estimated_cost} dicts
if selector.should_escalate(current_results):
    # widen the suite or deepen expensive attacks
    pass
selector.update_memory("pgd", success_rate=0.08, cost_seconds=42.0, model_arch="resnet18")
```
