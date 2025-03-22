# Defence Overview

Auto-ART provides defences across 9 categories covering traditional ML hardening, agentic safety, and formal certification.

## Defence Matrix

| Category | Defence | Applies To | Mechanism |
|----------|---------|-----------|-----------|
| **Preprocessor** | Spatial Smoothing | Input images | Median/mean filter removes high-freq perturbations |
| | Feature Squeezing | Input data | Bit-depth reduction strips subtle noise |
| | JPEG Compression | Input images | Lossy compression removes non-robust features |
| | Gaussian Augmentation | Input data | Additive noise smooths adversarial signals |
| **Postprocessor** | Reverse Sigmoid | Output probs | Perturbs confidence to prevent extraction |
| | High Confidence | Output probs | Only returns high-confidence predictions |
| | Gaussian Noise | Output probs | Adds noise to reduce info leakage |
| | Class Labels | Output probs | Returns only class label, no probabilities |
| **Trainer** | Adversarial Training (PGD) | Model weights | Trains on PGD adversarial examples |
| **Detector** | Activation Defence | Training data | Clusters hidden activations to find poison |
| | Spectral Signatures | Training data | Spectral analysis of feature representations |
| **Input Sanitizer** | DOM Sanitizer | HTML/web input | Strips hidden elements, adversarial tags |
| | Visual Denoiser | Image input | Spatial smoothing + bit-depth reduction |
| | Semantic Normalizer | Text input | Detects/removes prompt injections |
| **RAG Detector** | Cosine Anomaly | RAG retrievals | Flags embeddings far from trusted centroid |
| | Trigger Scanner | RAG content | Detects known backdoor trigger patterns |
| **In-Context** | Defence Exemplars | System prompt | Teaches agent to reject adversarial inputs |
| **Circuit Breaker** | Error Rate Monitor | Agent runtime | Trips on high error rate |
| | Token Spike Detector | Agent runtime | Trips on abnormal token consumption |
| | Action Frequency Limiter | Agent runtime | Trips on too-many actions per turn |
| **Guardrails** | InputRail | All inputs | Pattern-based threat detection |
| | ExecutionRail | Agent actions | Policy enforcement (PoLP) |
| **Certification** | Randomized Smoothing | Classifier | Provable L2 robustness with certified radii |
| | GREAT Score | Classifier | Generative AI robustness metric (ART 1.20+) |

## Detailed Guides

- [ART Defence Wrappers](art-wrappers.md) -- Preprocessor, Postprocessor, Trainer, Detector
- [Agentic Defences](agentic.md) -- Input Sanitizer, RAG Detector, In-Context Defence, Circuit Breaker
- [Guardrails](guardrails.md) -- InputRail, ExecutionRail, GuardrailPipeline
- [Certification](certification.md) -- Randomized Smoothing, GREAT Score
