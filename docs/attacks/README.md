# Attack Overview

Auto-ART provides attacks across 7 categories covering traditional ML, LLMs, and autonomous agents.

## Attack Matrix

| Category | Attack | Type | Requires |
|----------|--------|------|----------|
| **Evasion (White-box)** | FGSM | Gradient | Classifier + gradients |
| | PGD | Gradient | Classifier + gradients |
| | C&W L2 | Optimization | Classifier + gradients |
| | DeepFool | Gradient | Classifier + gradients |
| | AutoAttack | Ensemble | Classifier + gradients |
| **Evasion (Black-box)** | Boundary Attack | Decision-based | Predicted labels only |
| | Square Attack | Score-based | Predicted probabilities |
| | HopSkipJump | Decision-based | Predicted labels only |
| | SimBA | Score-based | Predicted probabilities |
| **Poisoning** | Backdoor | Data manipulation | Training data access |
| | Clean Label | Data manipulation | Training data + model |
| | Feature Collision | Data manipulation | Training data + model |
| | Gradient Matching | Data manipulation | Training data + model |
| **Extraction** | Copycat CNN | Query-based | Prediction API |
| | Knockoff Nets | Query-based | Prediction API |
| | Functionally Equivalent | Query-based | Prediction API |
| **Inference** | Membership Inference | Black-box | Prediction API |
| | Attribute Inference | Black-box | Prediction API |
| | Model Inversion (MIFace) | White-box | Model access |
| **LLM** | HotFlip | Gradient | Token embeddings + gradients |
| **Agentic** | AdvWeb DOM Injection | Environment | Agent + web environment |
| | AgentPoison RAG | Memory | Agent + knowledge base |
| | In-Context Injection | Prompt | Agent process method |
| | Universal Adversarial Patch | Visual | Agent + visual frames |

## How Attacks Are Organized

Auto-ART has two attack layers:

1. **`auto_art/core/attacks/`** -- ART wrapper classes used by `AttackGenerator` for creating attack instances
2. **`auto_art/core/evaluation/attacks/`** -- Evaluation strategy classes used by `ARTEvaluator` for structured evaluation

The `AttackGenerator` handles all the complexity of creating the right ART attack with the right parameters. You typically interact with it through the `ARTEvaluator` or the `Orchestrator`.

## Detailed Guides

- [Evasion Attacks](evasion.md) -- FGSM, PGD, C&W, AutoAttack, Square, HopSkipJump, SimBA
- [Poisoning Attacks](poisoning.md) -- Backdoor, Clean Label, Feature Collision
- [Extraction Attacks](extraction.md) -- Copycat CNN, Knockoff Nets
- [Inference Attacks](inference.md) -- Membership, Attribute, Model Inversion
- [Agentic Attacks](agentic.md) -- DOM injection, RAG poisoning, context injection
- [LLM Attacks](llm.md) -- HotFlip
