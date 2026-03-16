"""
Wrapper for ART's Fast Gradient Method (FGSM) Attack.
"""
from typing import Any, Optional, Union
import numpy as np

try:
    from art.attacks.evasion import FastGradientMethod as ARTFastGradientMethod
    from art.estimators.classification import ClassifierMixin
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierMixin: pass
    class ARTFastGradientMethod: pass


class FastGradientMethodWrapper:
    """
    A wrapper for the ART Fast Gradient Method (FGSM) attack.
    """
    def __init__(self,
                 estimator: ClassifierMixin,
                 attack_params: dict):
        """
        Initializes the FGSM wrapper.

        Args:
            estimator: An ART-compatible classifier.
            attack_params: Dictionary of attack parameters passed to ART's FastGradientMethod.
                Expected keys: eps, eps_step, batch_size, minimal, targeted, summary_writer, etc.
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. FastGradientMethod cannot be used.")

        if not isinstance(estimator, ClassifierMixin):
            raise TypeError(f"Estimator must be an ART ClassifierMixin, got {type(estimator)}")

        self.attack_params = attack_params

        self.attack = ARTFastGradientMethod(
            estimator=estimator,
            **attack_params
        )

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generates adversarial examples using FGSM.

        Args:
            x: Input samples (numpy array).
            y: Target labels for targeted attacks, or None for untargeted.
            **kwargs: Additional arguments for ART's generate method.

        Returns:
            Adversarial examples (numpy array).
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for FastGradientMethod.generate.")

        return self.attack.generate(x=x, y=y, **kwargs)
