"""
Composite Adversarial Attack evasion attack wrapper.

Wraps ART's CompositeAdversarialAttackPyTorch into auto-art's wrapper
pattern. Combines multiple semantic transformations (hue, saturation,
brightness, contrast, rotation) in a single adversarial attack.

Reference: Hsiung et al., "Towards Compositional Adversarial
Robustness: Semantically and Non-Semantically Composed Adversarial
Examples", 2023
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import CompositeAdversarialAttackPyTorch as ARTCompositeAttack
    from art.estimators.classification import ClassifierMixin
    ART_COMPOSITE_AVAILABLE = True
except ImportError:
    ART_COMPOSITE_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class CompositeAdversarialAttackWrapper:
    """Wrapper for ART's Composite Adversarial Attack (Hsiung et al., 2023).

    Combines multiple semantic transformations with Lp perturbations
    in a single optimisation loop for stronger adversarial examples.
    """

    def __init__(
        self,
        estimator: Any,
        enabled_attack: tuple = (0, 1, 2, 3, 4, 5),
        hps_policy: Optional[Any] = None,
        verbose: bool = True,
    ):
        if not ART_COMPOSITE_AVAILABLE:
            raise ImportError(
                "ART CompositeAdversarialAttackPyTorch not available. Install adversarial-robustness-toolbox."
            )
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        kwargs: dict[str, Any] = {
            "classifier": estimator,
            "enabled_attack": enabled_attack,
            "verbose": verbose,
        }
        if hps_policy is not None:
            kwargs["hps_policy"] = hps_policy

        self.art_attack = ARTCompositeAttack(**kwargs)
        self.attack_params = {
            "enabled_attack": enabled_attack,
            "hps_policy": hps_policy,
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
