"""
Overload Attack evasion attack wrapper.

Wraps ART's OverloadPyTorch into auto-art's wrapper pattern.
The Overload attack generates adversarial examples designed to
increase computational cost of adaptive inference models.

Reference: Hong et al., "A Framework for Evaluating the Robustness
of Adaptive Inference Models", 2021
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import OverloadPyTorch as ARTOverloadPyTorch
    from art.estimators.classification import ClassifierMixin
    ART_OVERLOAD_AVAILABLE = True
except ImportError:
    ART_OVERLOAD_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class OverloadAttackWrapper:
    """Wrapper for ART's Overload Attack (Hong et al., 2021).

    Crafts adversarial inputs that force adaptive-inference models to
    use their most computationally expensive execution paths.
    """

    def __init__(
        self,
        estimator: Any,
        eps: float = 0.3,
        max_iter: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        if not ART_OVERLOAD_AVAILABLE:
            raise ImportError("ART OverloadPyTorch not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTOverloadPyTorch(
            estimator=estimator,
            eps=eps,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "eps": eps, "max_iter": max_iter, "batch_size": batch_size,
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
