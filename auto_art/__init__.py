"""
Auto-ART: Automated Adversarial Robustness Testing Framework.

Comprehensive adversarial robustness evaluation for ML models and
autonomous AI agents. Built on ART v1.20+.

Features (v0.3.0):
- 50+ evasion attacks (FGSM, PGD, AutoAttack, C&W, DPatch, GRAPHITE, etc.)
- 12+ poisoning attacks (backdoor, clean-label, sleeper agent, etc.)
- 10+ defense strategies (TRADES, AWP, certified AT, diffusion purification, etc.)
- Adaptive attack selection with memory-guided escalation
- Multi-norm evaluation (L1, L2, Linf, semantic, spatial)
- RDI attack-independent robustness metric (30x faster than PGD)
- Gradient masking detection (ICLR 2025)
- NLP semantic perturbation attacks
- Production drift monitoring and supply chain scanning
- NIST AI RMF, OWASP LLM Top 10, EU AI Act compliance mapping
"""

__version__ = "0.3.0"

__all__ = [
    "__version__",
]