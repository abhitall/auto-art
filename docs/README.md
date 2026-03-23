# Auto-ART Documentation

Comprehensive documentation for the Automated Adversarial Robustness Testing Framework v0.3.0.

## Table of Contents

### Getting Started
- [Installation & Quick Start](getting-started.md) -- Setup, dependencies, first evaluation run
- [Architecture Overview](architecture.md) -- System design, component relationships, data flow
- [Configuration](configuration.md) -- EvaluationConfig, AttackConfig, YAML schema

### Attacks
- [Attack Overview](attacks/README.md) -- All attack categories at a glance
- [Evasion Attacks](attacks/evasion.md) -- 37+ attacks: FGSM, PGD, AutoAttack, C&W, DPatch, GRAPHITE, LowProFool, etc.
- [Poisoning Attacks](attacks/poisoning.md) -- Backdoor, Clean Label, Sleeper Agent, Hidden Trigger, Bullseye Polytope, BadDet, DGM
- [Extraction Attacks](attacks/extraction.md) -- Copycat CNN, Knockoff Nets, Functionally Equivalent
- [Inference Attacks](attacks/inference.md) -- Membership, Attribute, Model Inversion, Label-Only, DB Reconstruction
- [Agentic Attacks](attacks/agentic.md) -- DOM injection, RAG poisoning, context injection, adversarial patches
- [LLM Attacks](attacks/llm.md) -- HotFlip token replacement
- [NLP Attacks](attacks/nlp.md) -- Semantic perturbation (typo, synonym, contextual)

### Defences
- [Defence Overview](defences/README.md) -- All defence categories at a glance
- [ART Defence Wrappers](defences/art-wrappers.md) -- Preprocessor, Postprocessor, Trainer, Detector
- [Adversarial Training](defences/training.md) -- PGD, TRADES, AWP, OAAT, FBF, Certified AT
- [Advanced Preprocessors](defences/preprocessors.md) -- Label Smoothing, Thermometer Encoding, TVM, Video/MP3 Compression
- [Agentic Defences](defences/agentic.md) -- Input Sanitizer, RAG Detector, In-Context Defence, Circuit Breaker
- [Guardrails](defences/guardrails.md) -- InputRail, ExecutionRail, GuardrailPipeline
- [Certification](defences/certification.md) -- Randomized Smoothing, GREAT Score, IBP, DeepZ

### Metrics & Evaluation
- [RDI Metric](metrics/rdi.md) -- Attack-independent robustness evaluation (30x faster)
- [Multi-Norm Evaluation](metrics/multi-norm.md) -- L1/L2/Linf simultaneous evaluation, MultiRobustBench-style CR/SC
- [Privacy Metrics](metrics/privacy.md) -- PDTP, SHAPr membership risk
- [Gradient Masking Detection](metrics/gradient-masking.md) -- FOSC-based detection (ICLR 2025)

### Automation
- [Orchestrator](automation/orchestrator.md) -- YAML-driven one-command evaluation
- [Adaptive Attack Selection](automation/adaptive.md) -- Memory-guided tiered escalation
- [Parallel Execution](automation/parallel.md) -- GPU/CPU worker pools
- [Warm-Start Evaluation](automation/warmstart.md) -- Cache-based incremental evaluation
- [Red Teaming](automation/red-teaming.md) -- Continuous automated adversarial probing
- [CI/CD Integration](automation/ci-cd.md) -- GitHub Actions pipeline, deployment gates

### Production & Compliance
- [Robustness Monitoring](production/monitoring.md) -- Drift detection via RDI baseline comparison
- [Supply Chain Security](production/supply-chain.md) -- Model file scanning, SafeTensors validation
- [NIST AI RMF Compliance](compliance/nist.md) -- AI 600-1, AI 800-1, IR 8596 mapping
- [OWASP LLM Top 10](compliance/owasp.md) -- 2025 risk coverage tracking
- [EU AI Act](compliance/eu-ai-act.md) -- Risk classification and conformity assessment

### Telemetry
- [Agent Tracing](telemetry.md) -- State machine tracing, loop detection, multi-agent reliability
- [OpenTelemetry Integration](telemetry.md#opentelemetry) -- Production traces, metrics, and structured logs

### Reference
- [API Reference](api-reference.md) -- Module and class reference
- [SOTA Roadmap](SOTA_ROADMAP.md) -- Research-backed improvement roadmap
