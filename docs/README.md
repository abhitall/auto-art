# Auto-ART Documentation

Comprehensive documentation for the Automated Adversarial Robustness Testing Framework.

## Table of Contents

### Getting Started
- [Installation & Quick Start](getting-started.md) -- Setup, dependencies, first evaluation run
- [Architecture Overview](architecture.md) -- System design, component relationships, data flow

### Attacks
- [Attack Overview](attacks/README.md) -- All attack categories at a glance
- [Evasion Attacks](attacks/evasion.md) -- White-box (FGSM, PGD, C&W, AutoAttack) and black-box (Square, HopSkipJump, SimBA, Boundary)
- [Poisoning Attacks](attacks/poisoning.md) -- Backdoor, Clean Label, Feature Collision, Gradient Matching
- [Extraction Attacks](attacks/extraction.md) -- Copycat CNN, Knockoff Nets, Functionally Equivalent
- [Inference Attacks](attacks/inference.md) -- Membership, Attribute, Model Inversion
- [Agentic Attacks](attacks/agentic.md) -- DOM injection, RAG poisoning, context injection, adversarial patches
- [LLM Attacks](attacks/llm.md) -- HotFlip token replacement

### Defences
- [Defence Overview](defences/README.md) -- All defence categories at a glance
- [ART Defence Wrappers](defences/art-wrappers.md) -- Preprocessor, Postprocessor, Trainer, Detector
- [Agentic Defences](defences/agentic.md) -- Input Sanitizer, RAG Detector, In-Context Defence, Circuit Breaker
- [Guardrails](defences/guardrails.md) -- InputRail, ExecutionRail, GuardrailPipeline
- [Certification](defences/certification.md) -- Randomized Smoothing, GREAT Score

### Automation
- [Orchestrator](automation/orchestrator.md) -- YAML-driven one-command evaluation
- [Red Teaming](automation/red-teaming.md) -- Continuous automated adversarial probing
- [CI/CD Integration](automation/ci-cd.md) -- GitHub Actions pipeline, deployment gates
- [OWASP Mapping](automation/owasp-mapping.md) -- LLM Top 10 (2025) coverage tracking

### Telemetry
- [AgentOps Telemetry](telemetry.md) -- State machine tracing, loop detection, multi-agent reliability

### Reference
- [API Reference](api-reference.md) -- Module and class reference
- [Configuration](configuration.md) -- EvaluationConfig, AttackConfig, YAML schema
