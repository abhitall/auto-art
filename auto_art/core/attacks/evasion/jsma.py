"""
Jacobian-based Saliency Map Attack (JSMA) evasion attack wrapper.

Wraps ART's SaliencyMapMethod into auto-art's wrapper pattern.
JSMA uses the Jacobian of the model to identify the most influential
input features to perturb.

Reference: Papernot et al., "The Limitations of Deep Learning in
Adversarial Settings", IEEE Euro S&P 2016
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import SaliencyMapMethod as ARTSaliencyMapMethod
    from art.estimators.classification import ClassifierMixin
    ART_JSMA_AVAILABLE = True
except ImportError:
    ART_JSMA_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class JSMAWrapper:
    """Wrapper for ART's Saliency Map Method / JSMA (Papernot et al., 2016).

    Constructs adversarial examples by iteratively modifying the most
    salient features identified via the Jacobian matrix.
    """

    def __init__(
        self,
        estimator: Any,
        theta: float = 0.1,
        gamma: float = 1.0,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        if not ART_JSMA_AVAILABLE:
            raise ImportError("ART SaliencyMapMethod not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTSaliencyMapMethod(
            classifier=estimator,
            theta=theta,
            gamma=gamma,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "theta": theta, "gamma": gamma, "batch_size": batch_size,
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
