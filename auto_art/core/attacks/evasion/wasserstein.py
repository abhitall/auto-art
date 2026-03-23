"""
Wasserstein evasion attack wrapper.

Wraps ART's Wasserstein into auto-art's wrapper pattern.
The Wasserstein attack constrains perturbations using the
Wasserstein distance (earth-mover's distance).

Reference: Wong et al., "Wasserstein Adversarial Examples via Projected
Sinkhorn Iterations", ICML 2019
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import Wasserstein as ARTWasserstein
    from art.estimators.classification import ClassifierMixin
    ART_WASSERSTEIN_AVAILABLE = True
except ImportError:
    ART_WASSERSTEIN_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class WassersteinAttackWrapper:
    """Wrapper for ART's Wasserstein Attack (Wong et al., 2019).

    Generates adversarial examples constrained under the Wasserstein
    distance using projected Sinkhorn iterations.
    """

    def __init__(
        self,
        estimator: Any,
        targeted: bool = False,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 100,
        norm: str = "inf",
        kernel_size: int = 5,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        if not ART_WASSERSTEIN_AVAILABLE:
            raise ImportError("ART Wasserstein not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        norm_val: Any = np.inf if norm == "inf" else int(norm)
        self.art_attack = ARTWasserstein(
            estimator=estimator,
            targeted=targeted,
            eps=eps,
            eps_step=eps_step,
            max_iter=max_iter,
            norm=norm_val,
            kernel_size=kernel_size,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "targeted": targeted, "eps": eps, "eps_step": eps_step,
            "max_iter": max_iter, "norm": norm, "kernel_size": kernel_size,
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
