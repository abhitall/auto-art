"""
NewtonFool evasion attack wrapper.

Wraps ART's NewtonFool into auto-art's wrapper pattern.
NewtonFool uses Newton's method to find minimal perturbations
that cross the decision boundary.

Reference: Jang et al., "Objective Metrics and Gradient Descent
Algorithms for Adversarial Examples in Machine Learning", ACSAC 2017
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import NewtonFool as ARTNewtonFool
    from art.estimators.classification import ClassifierMixin
    ART_NEWTONFOOL_AVAILABLE = True
except ImportError:
    ART_NEWTONFOOL_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class NewtonFoolWrapper:
    """Wrapper for ART's NewtonFool (Jang et al., 2017).

    Applies Newton's method to iteratively find the minimal
    perturbation that causes misclassification.
    """

    def __init__(
        self,
        estimator: Any,
        max_iter: int = 100,
        eta: float = 0.01,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        if not ART_NEWTONFOOL_AVAILABLE:
            raise ImportError("ART NewtonFool not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTNewtonFool(
            classifier=estimator,
            max_iter=max_iter,
            eta=eta,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "max_iter": max_iter, "eta": eta, "batch_size": batch_size,
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
