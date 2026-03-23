# Attack Overview

Auto-ART ships **43** attack types across evasion, poisoning, extraction, inference, LLMs, and agentic settings. They wrap [IBM Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox) where applicable, plus framework-specific agent and LLM paths.

## Attack Matrix

| Category | Attack | Type | Requires |
|----------|--------|------|----------|
| **Evasion (White-box)** | FGSM | Gradient | Classifier + gradients |
| | PGD | Gradient (iterative) | Classifier + gradients |
| | C&W L2 | Optimization | Classifier + gradients |
| | DeepFool | Gradient | Classifier + gradients |
| | AutoAttack | Ensemble | Classifier + gradients |
| | Auto-PGD | Gradient (adaptive step) | Classifier + gradients |
| | BIM | Gradient (iterative) | Classifier + gradients |
| | Elastic Net (EAD) | Optimization | Classifier + gradients |
| | JSMA | Gradient (saliency) | Classifier + gradients |
| | NewtonFool | Gradient | Classifier + gradients |
| | Universal Perturbation | Gradient | Classifier + gradients |
| | Shadow Attack | Gradient | Classifier + gradients (see ART estimator support) |
| | Wasserstein | Gradient (Sinkhorn) | Classifier + gradients |
| | Feature Adversaries | Gradient (layer features) | Classifier + gradients |
| | Composite | Semantic + L∞ (PyTorch) | PyTorch classifier + gradients |
| | Overload | Gradient (adaptive inference) | PyTorch classifier + gradients (ART 1.18+) |
| **Evasion (Black-box)** | Boundary Attack | Decision-based | Predicted labels |
| | Square Attack | Score-based | Predicted probabilities |
| | HopSkipJump | Decision-based | Predicted labels |
| | SimBA | Score-based | Predicted probabilities |
| | ZOO | Zeroth-order (score) | Predicted probabilities / logits |
| | Brendel-Bethge | Decision / boundary optimisation | Classifier API (ART `BrendelBethgeAttack`) |
| | GeoDA | Decision-based (geometry) | Predicted labels |
| | Pixel Attack | Evolutionary (sparse pixels) | Classifier predictions |
| | Threshold Attack | Evolutionary (thresholded pixels) | Neural-network classifier (ART) |
| | Spatial Transformation | Spatial search | Classifier + differentiable spatial transform (ART) |
| | Adversarial Patch | Optimised patch | Classifier + gradients |
| | Decision Tree Attack | Transfer / tree-specific | Tree-capable setup (ART) |
| **Poisoning** | Backdoor | Data manipulation | Training data access |
| | Clean Label | Data manipulation | Training data + crafting model |
| | Feature Collision | Data manipulation | Training data + crafting model |
| | Gradient Matching | Data manipulation | Training data + crafting model |
| **Extraction** | Copycat CNN | Query-based | Prediction API |
| | Knockoff Nets | Query-based | Prediction API |
| | Functionally Equivalent | Query-based | Prediction API |
| **Inference** | Membership Inference BB | Black-box MI | Prediction API |
| | Attribute Inference BB | Black-box AI | Prediction API |
| | MIFace | Model inversion | Model access (white-box inversion) |
| **LLM** | HotFlip | Gradient | Token embeddings + gradients |
| **Agentic** | AdvWeb DOM Injection | Environment | Agent + web environment |
| | AgentPoison RAG | Memory | Agent + knowledge base |
| | In-Context Injection | Prompt | Agent `process` / prompt path |
| | Universal Adversarial Patch | Visual | Agent + visual frames |

## How Attacks Are Organized

Auto-ART uses two layers:

1. **`auto_art/core/attacks/`** — ART-oriented wrappers used by `AttackGenerator` to build attack objects from `AttackConfig`.
2. **`auto_art/core/evaluation/attacks/`** — Evaluation strategies used by `ARTEvaluator` for structured robustness runs.

`AttackGenerator` maps `attack_type` strings and parameters to the correct ART (or custom) constructor. You usually reach this through `ARTEvaluator` or the `Orchestrator`, not by calling ART directly.

## Detailed Guides

- [Evasion Attacks](evasion.md) — Full white-box, black-box, and ART 1.17+ evasion catalog with `AttackConfig` examples.
- [Poisoning Attacks](poisoning.md) — Backdoor, Clean Label, Feature Collision, Gradient Matching.
- [Extraction Attacks](extraction.md) — Copycat CNN, Knockoff Nets, Functionally Equivalent Extraction.
- [Inference Attacks](inference.md) — Membership Inference BB, Attribute Inference BB, MIFace.
- [Agentic Attacks](agentic.md) — DOM injection, RAG poisoning, in-context injection, universal visual patch.
- [LLM Attacks](llm.md) — HotFlip.
