# Getting Started

## Prerequisites

- Python 3.8+
- pip

## Installation

```bash
git clone https://github.com/yourusername/auto-art.git
cd auto-art
pip install .
```

### Optional Dependencies

```bash
pip install ".[pytorch]"       # PyTorch models
pip install ".[tensorflow]"    # TensorFlow models
pip install ".[agentic]"       # Agentic security (beautifulsoup4, lxml)
pip install ".[dev]"           # Development (pytest, black, mypy, flake8)
pip install ".[security]"      # Security scanning (bandit, safety)
```

For YAML-driven orchestration, also install:
```bash
pip install pyyaml
```

## Quick Start: Evaluate a Model

The simplest way to evaluate a model's robustness:

```python
from auto_art.core.evaluation.art_evaluator import ARTEvaluator
from auto_art.core.evaluation.config.evaluation_config import (
    EvaluationConfig, ModelType, Framework,
)

config = EvaluationConfig(
    model_type=ModelType.CLASSIFICATION,
    framework=Framework.PYTORCH,
)

evaluator = ARTEvaluator(model_obj=None, config=config)
results = evaluator.evaluate_robustness_from_path(
    model_path="path/to/model.pt",
    framework="pytorch",
    num_samples=100,
)
print(evaluator.generate_report(results))
```

This automatically:
1. Loads the model from the given path
2. Analyzes its architecture (input/output shapes, layer info)
3. Generates synthetic test data matching the model's input shape
4. Runs all supported attacks for the model type (FGSM, PGD, DeepFool, etc.)
5. Computes robustness metrics (empirical robustness, CLEVER scores, security score)
6. Returns a structured results dictionary

## Quick Start: Test Agent Resilience

```python
from auto_art.core.evaluation.attacks.agentic import (
    AdvWebDOMAttack, AgentPoisonRAGAttack,
)

# Test DOM injection resilience
dom_attack = AdvWebDOMAttack(stealth_level="invisible", max_injections=5)
result = dom_attack.execute_agentic(agent=my_agent, environment=my_env)
print(f"DOM attack success rate: {result.success_rate:.2%}")

# Test RAG poisoning resilience
rag_attack = AgentPoisonRAGAttack(trigger_type="semantic", injection_rate=0.01)
result = rag_attack.execute_agentic(agent=my_agent, environment=my_env)
print(f"RAG poison success rate: {result.success_rate:.2%}")
```

## Quick Start: YAML-Driven Evaluation

Create `eval_config.yaml`:

```yaml
target:
  model_path: "path/to/model.pt"
  framework: "pytorch"

attacks:
  evasion:
    - name: fgsm
      eps: 0.3
    - name: pgd
      eps: 0.3
      max_iter: 40
  agentic:
    - name: advweb_dom
      stealth_level: invisible
      max_injections: 5
    - name: rag_poison
      trigger_type: semantic
  red_team:
    num_rounds: 50

gates:
  max_attack_success_rate: 0.05
  max_jailbreak_rate: 0.05
```

Run it:

```python
from auto_art.core.orchestrator import Orchestrator

orchestrator = Orchestrator.from_yaml("eval_config.yaml")
report = orchestrator.run(agent=my_agent)
print(report.to_markdown())
orchestrator.assert_gates(report)  # Fails if any gate breached
```

## Next Steps

- [Architecture Overview](architecture.md) -- Understand the system design
- [Attack Overview](attacks/README.md) -- Explore all available attacks
- [Orchestrator Guide](automation/orchestrator.md) -- Deep dive into YAML-driven automation
- [CI/CD Integration](automation/ci-cd.md) -- Set up your pipeline
