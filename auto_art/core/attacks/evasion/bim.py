"""
Basic Iterative Method (BIM) evasion attack wrapper.

Wraps ART's BasicIterativeMethod into auto-art's wrapper pattern.
BIM extends FGSM by applying it multiple times with a small step size.

Reference: Kurakin et al., "Adversarial examples in the physical world", 2017
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import BasicIterativeMethod as ARTBasicIterativeMethod
    from art.estimators.classification import ClassifierMixin
    ART_BIM_AVAILABLE = True
except ImportError:
    ART_BIM_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class BasicIterativeMethodWrapper:
    """Wrapper for ART's Basic Iterative Method (Kurakin et al., 2017).

    Iteratively applies small FGSM steps, clipping after each iteration
    to stay within an epsilon-ball around the original input.
    """

    def __init__(
        self,
        estimator: Any,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        targeted: bool = False,
        batch_size: int = 32,
        norm: str = "inf",
        verbose: bool = True,
    ):
        if not ART_BIM_AVAILABLE:
            raise ImportError("ART BasicIterativeMethod not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        norm_val: Any = np.inf if norm == "inf" else int(norm)
        self.art_attack = ARTBasicIterativeMethod(
            estimator=estimator,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            targeted=targeted,
            batch_size=batch_size,
            norm=norm_val,
            verbose=verbose,
        )
        self.attack_params = {
            "eps": eps, "eps_step": eps_step, "max_iter": max_iter,
            "targeted": targeted, "batch_size": batch_size, "norm": norm,
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
