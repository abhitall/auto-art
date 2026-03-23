"""
Elastic Net (EAD) evasion attack wrapper.

Wraps ART's ElasticNet into auto-art's wrapper pattern.
EAD combines L1 and L2 regularization to craft adversarial examples
with sparse perturbations.

Reference: Chen et al., "EAD: Elastic-Net Attacks to DNNs via Feature
Selection", AAAI 2018
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import ElasticNet as ARTElasticNet
    from art.estimators.classification import ClassifierMixin
    ART_EAD_AVAILABLE = True
except ImportError:
    ART_EAD_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class ElasticNetWrapper:
    """Wrapper for ART's Elastic Net / EAD (Chen et al., 2018).

    Uses elastic-net regularization (L1 + L2) to produce adversarial
    examples with sparse, imperceptible perturbations.
    """

    def __init__(
        self,
        estimator: Any,
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 0.01,
        max_iter: int = 100,
        binary_search_steps: int = 9,
        initial_const: float = 0.001,
        batch_size: int = 1,
        decision_rule: str = "EN",
        verbose: bool = True,
        beta: float = 0.001,
    ):
        if not ART_EAD_AVAILABLE:
            raise ImportError("ART ElasticNet not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTElasticNet(
            classifier=estimator,
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            max_iter=max_iter,
            binary_search_steps=binary_search_steps,
            initial_const=initial_const,
            batch_size=batch_size,
            decision_rule=decision_rule,
            verbose=verbose,
            beta=beta,
        )
        self.attack_params = {
            "confidence": confidence, "targeted": targeted,
            "learning_rate": learning_rate, "max_iter": max_iter,
            "binary_search_steps": binary_search_steps,
            "initial_const": initial_const, "batch_size": batch_size,
            "decision_rule": decision_rule, "beta": beta,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
