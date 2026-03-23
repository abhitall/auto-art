# Architecture Overview

## Design Philosophy

Auto-ART follows six core principles:

1. **Defence-in-Depth** -- Multiple independent security layers, each catching what others miss
2. **Automate Everything** -- One YAML file triggers a full evaluation; no manual steps
3. **Framework-Agnostic** -- Works with PyTorch, TensorFlow, Keras, scikit-learn, XGBoost, LightGBM, CatBoost, GPy, Transformers, and any agent with a `process()` method
4. **Adaptive Intelligence** -- Memory-guided attack selection that learns from past evaluations
5. **Multi-Norm by Default** -- Never evaluate on a single Lp norm; always test across multiple threat models
6. **Compliance-Native** -- Every evaluation maps to NIST AI RMF, OWASP, and EU AI Act requirements

## System Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR v3                                   │
│   YAML config → Execution Engine → Phases → Gates → Compliance          │
│   Adaptive Selection │ Parallel Execution │ Budget Management            │
│   Gradient Masking Detection │ RDI Rapid Screen                          │
├──────────────────────────────────────────────────────────────────────────┤
│                      GUARDRAIL LAYER                                     │
│   InputRail │ ExecutionRail │ Sanitizer │ CircuitBreaker                 │
├──────────────────────────────────────────────────────────────────────────┤
│                       CORE ENGINE                                        │
│ ARTEvaluator │ AttackGenerator │ MetricsCalculator │ ComplianceEngine    │
│ AdaptiveSelector │ ParallelRunner │ AttackMemory │ DriftMonitor          │
├──────────┬──────────┬──────────┬───────────┬──────────┬──────────────────┤
│ ATTACKS  │ DEFENCES │ CERTIF.  │ METRICS   │COMPLIANCE│ MONITORING       │
│ Evasion  │ Pre.     │ RS       │ Empiric.  │NIST RMF  │ DriftDetect     │
│  (37+)   │ Post.    │ DeRS     │ CLEVER    │OWASP LLM │ SupplyChain     │
│ Poison   │ Trainer  │ DeepZ    │ GREAT     │EU AI Act │ Retraining      │
│  (10+)   │ Transf.  │ IBP      │ Wasser.   │          │                 │
│ Extract  │ Detect.  │ Adaptive │ RDI       │          │                 │
│  (3)     │ Agentic  │          │ MultiNorm │          │                 │
│ Infer.   │ Guardr.  │          │ Privacy   │          │                 │
│  (8+)    │ Advanced │          │ GradMask  │          │                 │
│ Audio(2) │ (TRADES  │          │ Security  │          │                 │
│ NLP/Text │  AWP,    │          │ Score     │          │                 │
│ LLM      │  LabelSm │          │           │          │                 │
│ Agentic  │  Thermo) │          │           │          │                 │
│ RedTeam  │          │          │           │          │                 │
├──────────┴──────────┴──────────┴───────────┴──────────┴──────────────────┤
│                  MODEL IMPLEMENTATIONS                                    │
│ PyTorch │ TF v2 │ Keras │ sklearn │ XGBoost │ LightGBM │ CatBoost │ GPy│
├──────────────────────────────────────────────────────────────────────────┤
│                    OUTPUT / REPORTING                                     │
│ JSON │ Markdown │ SARIF 2.1.0 │ HTML Dashboard │ Compliance Reports     │
├──────────────────────────────────────────────────────────────────────────┤
│          ADVERSARIAL ROBUSTNESS TOOLBOX (ART) v1.20+                     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Component Relationships

### Orchestrator → Core Engine

The `Orchestrator` reads a YAML config and drives the evaluation pipeline. It supports three execution modes:
- **Sequential** (default): Attacks run one-by-one
- **Adaptive**: Memory-guided selection escalates from cheap screening to expensive attacks based on model survival
- **Parallel**: Independent attacks execute concurrently via GPU/CPU worker pools

### Guardrail Layer

The guardrail layer sits between raw inputs and the core engine:

- **InputRail** scans incoming prompts/DOM/environment state for threat patterns
- **ExecutionRail** enforces Principle of Least Privilege on agent tool calls
- **InputSanitizer** chains DOM, visual, and semantic sanitization layers

### Core Engine

- **ARTEvaluator** orchestrates attack execution, metric calculation, and report generation
- **AttackGenerator** creates ART attack instances from configuration (37+ evasion, 10+ poisoning, 3 extraction, 8+ inference, 2 audio, 1+ NLP, 1 LLM — 60+ total attack types)
- **MetricsCalculator** computes robustness metrics (empirical robustness, CLEVER, GREAT Score, Wasserstein distance, tree verification, security score)
- **RDICalculator** provides attack-independent robustness evaluation 30x faster than PGD (UAI 2025)
- **MultiNormEvaluator** runs attacks across L1/L2/Linf simultaneously with MultiRobustBench-style metrics
- **GradientMaskingDetector** detects false robustness via FOSC analysis (ICLR 2025)
- **PrivacyMetricsCalculator** computes PDTP and SHAPr membership privacy risk
- **AdaptiveAttackSelector** selects attacks based on model type, budget, and historical performance using tiered escalation
- **ParallelAttackRunner** executes independent attacks concurrently with configurable workers
- **AttackMemory** stores persistent success-rate history per model architecture family

### Compliance Engine

Maps evaluation results to regulatory frameworks:
- **NIST AI RMF** (MEASURE, MAP, MANAGE, GOVERN functions)
- **OWASP LLM Top 10** (2025 risk categories)
- **EU AI Act** (risk classification, Article 15 robustness requirements)

### Production Monitoring

- **RobustnessDriftMonitor** tracks RDI baseline changes over time and triggers alerts
- **ModelSupplyChainScanner** scans model files for pickle RCE, backdoor indicators, and format safety

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
Execution Mode Selection (sequential | adaptive | parallel)
  ↓
Phase 1: Evasion Attacks
  → Model loaded via ModelFactory
  → ART estimator created via ClassifierFactory
  → [Adaptive] Memory-guided attack selection and escalation
  → AttackGenerator creates and executes attacks
  → Adversarial examples generated across multiple norms
  → [Optional] Gradient masking detection
  → MetricsCalculator computes success rates, perturbation sizes, RDI
  ↓
Phase 2: Poisoning Attacks
  → Backdoor, clean label, sleeper agent, hidden trigger, BadDet, DGM
  ↓
Phase 3: Extraction Attacks
  → Copycat CNN, Knockoff Nets, Functionally Equivalent
  ↓
Phase 4: Inference Attacks
  → Membership, attribute, model inversion, label-only, DB reconstruction
  ↓
Phase 5: Defence Evaluation
  → Preprocessors (spatial smoothing, JPEG, label smoothing, thermometer, etc.)
  → Postprocessors (reverse sigmoid, rounding, etc.)
  → Trainers (PGD, TRADES, AWP, OAAT, FBF, Certified AT) applied
  → Robustness improvement measured
  → Security score computed
  ↓
Phase 6: Agentic Attacks (if agent provided)
  → AdvWebDOMAttack injects into environment
  → AgentPoisonRAGAttack poisons knowledge base
  ↓
Phase 7: Red Team (if agent provided)
  → RedTeamLLM generates multi-turn attacks
  → Vulnerabilities logged
  ↓
Gate Evaluation
  → evasion_max_attack_success_rate check
  → agentic_max_attack_success_rate check
  → min_security_score check
  → max_jailbreak_rate check
  → max_membership_leakage check
  → max_poisoning_detection_rate check
  ↓
Compliance Assessment
  → NIST AI RMF mapping
  → OWASP LLM Top 10 coverage
  → EU AI Act conformity
  ↓
OrchestratorReport
  → .to_json() for CI/CD artifacts
  → .to_markdown() for human review
  → .to_sarif() for GitHub/GitLab/Azure integration
  → .assert_gates() for pipeline gating
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ART v1.20+ as foundation | 5.9k stars, 50+ attacks, 30+ defences, GREAT Score, YOLO v8+, active maintenance, Linux Foundation backed |
| Separate evaluation/attacks layers | `core/attacks/` holds ART wrappers; `core/evaluation/attacks/` holds evaluation strategies — prevents circular dependencies |
| DefenceStrategy ABC with default set_params | Subclasses aren't forced to implement set_params if params are immutable |
| YAML orchestrator over Python-only config | Enables version-controlled, reproducible evaluations without code changes |
| Adaptive execution mode | Reduces compute cost by 40-60% while improving attack coverage (AutoRedTeamer-inspired) |
| RDI as rapid screen | 30x faster than PGD for robustness estimation; used by drift monitor too |
| Multi-norm evaluation | SingleLp evaluation is misleading; worst-case across norms is the true robustness |
| Gradient masking detection | ICLR 2025 showed TRADES overestimates robustness; always verify with black-box attacks |
| Compliance-native design | NIST AI RMF and EU AI Act increasingly mandate adversarial testing for high-risk AI |
| Supply chain scanning | Pickle-based RCE in model files is a real threat (2025 incidents) |
| ClassifierFactory extensibility | Factory pattern with framework/model_type tuple keys makes adding new frameworks trivial |
