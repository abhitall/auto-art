"""
Feature Adversaries evasion attack wrapper.

Wraps ART's FeatureAdversariesNumpy into auto-art's wrapper pattern.
Feature Adversaries craft inputs whose internal representations
match a target representation at a chosen layer.

Reference: Sabour et al., "Adversarial Manipulation of Deep
Representations", ICLR 2016
"""

from typing import Any, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import FeatureAdversariesNumpy as ARTFeatureAdversariesNumpy
    from art.estimators.classification import ClassifierMixin
    ART_FA_AVAILABLE = True
except ImportError:
    ART_FA_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class FeatureAdversariesWrapper:
    """Wrapper for ART's Feature Adversaries (Sabour et al., 2016).

    Generates adversarial examples by minimising the distance between
    internal feature representations at a specified layer and those of
    a chosen guide image.
    """

    def __init__(
        self,
        estimator: Any,
        delta: float = 0.2,
        layer: Union[int, str] = -1,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        if not ART_FA_AVAILABLE:
            raise ImportError("ART FeatureAdversariesNumpy not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTFeatureAdversariesNumpy(
            estimator=estimator,
            delta=delta,
            layer=layer,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "delta": delta, "layer": layer, "batch_size": batch_size,
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
