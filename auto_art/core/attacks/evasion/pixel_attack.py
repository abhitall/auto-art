"""
Pixel Attack evasion attack wrapper.

Wraps ART's PixelAttack into auto-art's wrapper pattern.
Pixel Attack uses differential evolution to modify a small number
of pixels to cause misclassification.

Reference: Su et al., "One Pixel Attack for Fooling Deep Neural
Networks", IEEE TEVC 2019
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import PixelAttack as ARTPixelAttack
    from art.estimators.classification import ClassifierMixin
    ART_PIXEL_AVAILABLE = True
except ImportError:
    ART_PIXEL_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class PixelAttackWrapper:
    """Wrapper for ART's Pixel Attack (Su et al., 2019).

    Uses differential evolution to find a minimal set of pixel
    modifications that fool the classifier.
    """

    def __init__(
        self,
        estimator: Any,
        th: Optional[int] = None,
        es: int = 1,
        targeted: bool = False,
        verbose: bool = True,
    ):
        if not ART_PIXEL_AVAILABLE:
            raise ImportError("ART PixelAttack not available. Install adversarial-robustness-toolbox.")
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

        self.art_attack = ARTPixelAttack(**kwargs)
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
