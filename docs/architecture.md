# Architecture Overview

## Design Philosophy

Auto-ART follows three core principles:

1. **Defence-in-Depth** -- Multiple independent security layers, each catching what others miss
2. **Automate Everything** -- One YAML file triggers a full evaluation; no manual steps
3. **Framework-Agnostic** -- Works with PyTorch, TensorFlow, Keras, scikit-learn, and any agent with a `process()` method

## System Layers

```
┌─────────────────────────────────────────────────────┐
│                   ORCHESTRATOR                      │
│          (YAML config → phases → gates → report)    │
├─────────────────────────────────────────────────────┤
│                  GUARDRAIL LAYER                    │
│         InputRail │ ExecutionRail │ Sanitizer       │
├─────────────────────────────────────────────────────┤
│                    CORE ENGINE                      │
│ ARTEvaluator │ AttackGenerator │ MetricsCalculator  │
├────────────────┬────────────────┬───────────────────┤
│    ATTACKS     │   DEFENCES     │    TELEMETRY      │
│  Evasion       │  Preprocessor  │  AgentTracer      │
│  Poisoning     │  Postprocessor │  StateTransition  │
│  Extraction    │  Trainer       │  SpanKind         │
│  Inference     │  Detector      │  LoopDetection    │
│  Agentic       │  Agentic       │  MultiAgentRel.   │
│  LLM           │  Certification │                   │
│  Red Team      │                │                   │
├────────────────┴────────────────┴───────────────────┤
│               MODEL IMPLEMENTATIONS                 │
│  PyTorch │ TensorFlow │ Keras │ sklearn │ XGBoost   │
├─────────────────────────────────────────────────────┤
│           ADVERSARIAL ROBUSTNESS TOOLBOX (ART)      │
└─────────────────────────────────────────────────────┘
```

## Component Relationships

### Orchestrator → Core Engine

The `Orchestrator` reads a YAML config and drives the evaluation pipeline. It creates the appropriate attack and defence objects, runs them in phases, evaluates gates, and produces a structured `OrchestratorReport` with JSON and Markdown output.

### Guardrail Layer

The guardrail layer sits between raw inputs and the core engine:

- **InputRail** scans incoming prompts/DOM/environment state for threat patterns
- **ExecutionRail** enforces Principle of Least Privilege on agent tool calls
- **InputSanitizer** chains DOM, visual, and semantic sanitization layers

### Core Engine

- **ARTEvaluator** orchestrates attack execution, metric calculation, and report generation
- **AttackGenerator** creates ART attack instances from configuration (supports 9+ evasion, 4 poisoning, 3 extraction, 3 inference, 1 LLM, 3 black-box attack types)
- **MetricsCalculator** computes robustness metrics (empirical robustness, CLEVER, GREAT Score, Wasserstein distance)

### Telemetry

The `AgentTracer` provides OpenTelemetry-compatible tracing with:
- Span lifecycle (start → events → end)
- Validated state machine transitions
- Infinite loop detection via sliding window pattern matching
- Silent failure analysis

## Data Flow: Full Evaluation

```
YAML Config
  ↓
Orchestrator.from_yaml()
  ↓
Phase 1: Evasion Attacks
  → AttackGenerator creates ART attacks
  → ART attacks generate adversarial examples
  → MetricsCalculator computes success rates
  ↓
Phase 2: Agentic Attacks
  → AdvWebDOMAttack injects into environment
  → AgentPoisonRAGAttack poisons knowledge base
  → Agent responses evaluated for hijacking
  ↓
Phase 3: Red Team
  → RedTeamLLM generates multi-turn attacks
  → Agent responses evaluated for jailbreaks
  → Vulnerabilities logged
  ↓
Gate Evaluation
  → max_attack_success_rate check
  → max_jailbreak_rate check
  ↓
OrchestratorReport
  → .to_json() for CI/CD artifacts
  → .to_markdown() for human review
  → .assert_gates() for pipeline gating
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ART as foundation | 5.9k stars, 39 attacks, 29 defences, active maintenance, Linux Foundation backed |
| Separate evaluation/attacks layers | `core/attacks/` holds ART wrappers; `core/evaluation/attacks/` holds evaluation strategies -- prevents circular dependencies |
| DefenceStrategy ABC with default set_params | Subclasses aren't forced to implement set_params if params are immutable |
| AgenticAttackStrategy bridges to AttackStrategy | `execute()` wraps `execute_agentic()` so agentic attacks can plug into the existing evaluator |
| YAML orchestrator over Python-only config | Enables version-controlled, reproducible evaluations without code changes |
| Circuit breaker with CLOSED/OPEN/HALF_OPEN | Standard circuit breaker pattern from distributed systems, adapted for agent safety |
