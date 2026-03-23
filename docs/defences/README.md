# Defence Overview

Auto-ART provides defences across **12 categories**: ART-backed `DefenceStrategy` wrappers (pre/post processing, training, transformers, detectors), agentic hardening, guardrails, and formal certification. The matrix below lists every supported defence type and how it is applied.

## Defence Matrix

| Category | Defence | Applies To | Mechanism |
|----------|---------|------------|-----------|
| **Preprocessor** | Spatial Smoothing | Input images | Median/mean filter removes high-frequency perturbations |
| | Feature Squeezing | Input data | Bit-depth reduction strips subtle noise |
| | JPEG Compression | Input images | Lossy compression removes non-robust features |
| | Gaussian Augmentation | Input data | Additive Gaussian noise smooths adversarial signals |
| | Cutout | Input images | Random square masks improve occlusion / robustness |
| | Mixup | Input + labels | Convex combinations of examples for smoother decision regions |
| | CutMix | Input + labels | Cut-and-paste patches with mixed labels |
| **Postprocessor** | Reverse Sigmoid | Output probabilities | Perturbs confidence to hinder extraction |
| | High Confidence | Output probabilities | Low-confidence predictions collapsed to uniform |
| | Gaussian Noise | Output probabilities | Noise on logits/probs reduces information leakage |
| | Class Labels | Output probabilities | Returns class index only (no probability vector) |
| | Rounding | Output probabilities | Rounds probs to fixed decimals |
| **Trainer** | Adversarial Training PGD (Madry) | Model weights | Trains on PGD adversarial examples |
| | Fast Is Better Than Free (FBF) | Model weights | Single-step FGSM-style AT with random init |
| | Certified Adversarial Training | Model weights | Verified training with configurable bound / loss |
| | Interval Bound Propagation (IBP) | Model weights | Certified training with interval loss |
| | OAAT | Model weights | Once-for-all AT across perturbation regimes |
| **Transformer** | Defensive Distillation | Model + training | Temperature distillation for smoother boundaries |
| | Neural Cleanse | Backdoor analysis | Reverse-engineers triggers; mitigates backdoors |
| | STRIP | Trojan / backdoor | Intentional perturbations; consistency-based detection |
| **Detector — Evasion** | Basic Input Detector | Inference inputs | Binary detector on raw inputs |
| | Activation Detector | Hidden activations | Binary detector on layer activations |
| | Subset Scan | Features / AE stats | Subset scanning over background distribution |
| | BEYOND (ART 1.19+) | Neighborhood density | Bayesian neighborhood outlier detection |
| **Detector — Poisoning** | Activation Defence | Training data | Clusters activations to surface poison |
| | Spectral Signature | Training data | Spectral structure of representations |
| | Data Provenance | Training data + metadata | Trusted vs untrusted source analysis |
| | RONI | Training data | Reject-on-negative-impact filtering |
| **Input Sanitizer** | DOM Sanitizer | HTML / web input | Strips hidden elements and adversarial markup |
| | Visual Denoiser | Images | Smoothing + bit-depth reduction |
| | Semantic Normalizer | Text | Prompt-injection style normalization |
| **RAG Detector** | Cosine Anomaly | Retrieval embeddings | Flags queries far from trusted centroid |
| | Trigger Scanner | Retrieved content | Known backdoor trigger patterns |
| **In-Context** | Defence Exemplars | System / agent prompt | In-prompt examples that reject adversarial inputs |
| **Circuit Breaker** | Error Rate Monitor | Agent runtime | Trips on sustained high error rate |
| | Token Spike Detector | Agent runtime | Trips on abnormal token use |
| | Action Frequency Limiter | Agent runtime | Trips on excessive actions per turn |
| **Guardrails** | InputRail | All inputs | Pattern-based threat detection |
| | ExecutionRail | Agent actions | Policy enforcement (e.g. least privilege) |
| **Certification** | Randomized Smoothing | Classifier | Certified L₂ (smoothed classifier) |
| | GREAT Score | Classifier | Generative robustness metric (ART 1.20+) |

## Detailed Guides

- [ART Defence Wrappers](art-wrappers.md) — Preprocessor, postprocessor, trainer, transformer, evasion and poisoning detectors
- [Agentic Defences](agentic.md) — Input sanitizer, RAG detector, in-context defence, circuit breaker
- [Guardrails](guardrails.md) — InputRail, ExecutionRail, GuardrailPipeline
- [Certification](certification.md) — Randomized Smoothing, GREAT Score, IBP, DeepZ, DeRandomized Smoothing (notes)
