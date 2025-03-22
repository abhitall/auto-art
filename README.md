# Auto-ART: Automated Adversarial Robustness Testing

A Python framework for adversarial robustness evaluation of ML models and autonomous AI agents. Built on [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox), aligned with [OWASP LLM Top 10 (2025)](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) and [NIST AI 100-2](https://csrc.nist.gov/pubs/ai/100/2/e2025/final).

## What It Does

- **30+ attacks** across evasion, poisoning, extraction, inference, LLM, and agentic categories
- **20+ defences** including ART wrappers, input sanitization, guardrails, and circuit breakers
- **YAML-driven orchestrator** for one-command evaluation with CI/CD gates
- **Continuous red teaming** with multi-turn adaptive attacks across 7 OWASP-mapped categories
- **Robustness certification** via Randomized Smoothing and GREAT Score (ART 1.20+)
- **AgentOps telemetry** with state machine tracing and infinite loop detection

## Install

```bash
pip install .                   # Core
pip install ".[pytorch]"        # + PyTorch
pip install ".[agentic]"        # + Agentic security
pip install ".[dev]"            # + Dev tools
```

## Quick Start

**Evaluate a model:**
```python
from auto_art.core.evaluation.art_evaluator import ARTEvaluator
from auto_art.core.evaluation.config.evaluation_config import EvaluationConfig, ModelType, Framework

evaluator = ARTEvaluator(model_obj=None, config=EvaluationConfig(
    model_type=ModelType.CLASSIFICATION, framework=Framework.PYTORCH,
))
results = evaluator.evaluate_robustness_from_path("model.pt", "pytorch", num_samples=100)
```

**Test agent resilience:**
```python
from auto_art.core.evaluation.attacks.agentic import AdvWebDOMAttack
result = AdvWebDOMAttack(max_injections=5).execute_agentic(agent=my_agent, environment=my_env)
```

**One-command YAML evaluation:**
```python
from auto_art.core.orchestrator import Orchestrator
report = Orchestrator.from_yaml("eval_config.yaml").run(agent=my_agent)
report.to_markdown()  # Human-readable report
```

## OWASP Coverage

90% coverage of OWASP LLM Top 10 (2025) -- 8 categories full, 2 partial. See [detailed mapping](docs/automation/owasp-mapping.md).

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

| Section | Contents |
|---------|----------|
| [Getting Started](docs/getting-started.md) | Installation, quick start, first evaluation |
| [Architecture](docs/architecture.md) | System design, layers, data flow |
| [Attacks](docs/attacks/README.md) | All 30+ attacks with usage examples |
| [Defences](docs/defences/README.md) | All 20+ defences with configuration |
| [Orchestrator](docs/automation/orchestrator.md) | YAML schema, CI/CD integration |
| [Red Teaming](docs/automation/red-teaming.md) | Continuous adversarial probing |
| [OWASP Mapping](docs/automation/owasp-mapping.md) | LLM Top 10 coverage tracking |
| [CI/CD](docs/automation/ci-cd.md) | Pipeline phases, deployment gates |
| [Telemetry](docs/telemetry.md) | Agent tracing, loop detection |

## Tests

```bash
pip install ".[dev]"
pytest tests/ -v              # 223+ tests
```

## License

MIT

## Acknowledgments

- [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) by Trusted-AI / Linux Foundation AI & Data
- [IBM ARES](https://github.com/IBM/ares) for YAML orchestration patterns
- [RobustBench](https://robustbench.github.io/) for AutoAttack evaluation methodology
- [OWASP LLM Top 10](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) and [NIST AI 100-2](https://csrc.nist.gov/pubs/ai/100/2/e2025/final) for threat taxonomy
