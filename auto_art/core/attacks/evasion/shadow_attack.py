"""
Shadow Attack evasion attack wrapper.

Wraps ART's ShadowAttack into auto-art's wrapper pattern.
Shadow Attack generates adversarial examples that appear natural
by minimising perceptual differences using shadow-based priors.

Reference: Ghiasi et al., "Breaking certified defenses: Semantic
adversarial examples with spoofed robustness certificates", CVPR 2020
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import ShadowAttack as ARTShadowAttack
    from art.estimators.classification import ClassifierMixin
    ART_SHADOW_AVAILABLE = True
except ImportError:
    ART_SHADOW_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class ShadowAttackWrapper:
    """Wrapper for ART's Shadow Attack (Ghiasi et al., 2020).

    Crafts adversarial examples that preserve perceptual similarity
    using total-variation, colour, and smoothness regularization.
    """

    def __init__(
        self,
        estimator: Any,
        sigma: float = 0.5,
        nb_steps: int = 300,
        learning_rate: float = 0.01,
        lambda_tv: float = 0.3,
        lambda_c: float = 1.0,
        lambda_s: float = 0.5,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        if not ART_SHADOW_AVAILABLE:
            raise ImportError("ART ShadowAttack not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTShadowAttack(
            estimator=estimator,
            sigma=sigma,
            nb_steps=nb_steps,
            learning_rate=learning_rate,
            lambda_tv=lambda_tv,
            lambda_c=lambda_c,
            lambda_s=lambda_s,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "sigma": sigma, "nb_steps": nb_steps,
            "learning_rate": learning_rate, "lambda_tv": lambda_tv,
            "lambda_c": lambda_c, "lambda_s": lambda_s,
            "batch_size": batch_size,
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
