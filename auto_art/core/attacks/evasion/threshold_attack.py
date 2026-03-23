"""
Threshold Attack evasion attack wrapper.

Wraps ART's ThresholdAttack into auto-art's wrapper pattern.
Threshold Attack uses evolutionary search to perturb pixels whose
values exceed a learned threshold.

Reference: Vargas et al., "Robustness of rotation-equivariant
networks to adversarial perturbations" — adapted threshold-based
approach
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import ThresholdAttack as ARTThresholdAttack
    from art.estimators.classification import ClassifierMixin
    ART_THRESHOLD_AVAILABLE = True
except ImportError:
    ART_THRESHOLD_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class ThresholdAttackWrapper:
    """Wrapper for ART's Threshold Attack.

    An evolutionary-search-based attack that modifies pixel values
    above a computed threshold to cause misclassification.
    """

    def __init__(
        self,
        estimator: Any,
        th: Optional[int] = None,
        es: int = 1,
        targeted: bool = False,
        verbose: bool = True,
    ):
        if not ART_THRESHOLD_AVAILABLE:
            raise ImportError("ART ThresholdAttack not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        kwargs: dict[str, Any] = {
            "classifier": estimator,
            "es": es,
            "targeted": targeted,
            "verbose": verbose,
        }
        if th is not None:
            kwargs["th"] = th

        self.art_attack = ARTThresholdAttack(**kwargs)
        self.attack_params = {
            "th": th, "es": es, "targeted": targeted,
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
