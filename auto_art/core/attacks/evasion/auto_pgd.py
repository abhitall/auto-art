"""
Auto Projected Gradient Descent (AutoPGD) evasion attack wrapper.

Wraps ART's AutoProjectedGradientDescent into auto-art's wrapper pattern.
AutoPGD uses an adaptive step size schedule and automatic restarts.

Reference: Croce & Hein, "Reliable evaluation of adversarial robustness
with an ensemble of attacks", ICML 2020
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import AutoProjectedGradientDescent as ARTAutoPGD
    from art.estimators.classification import ClassifierMixin
    ART_AUTOPGD_AVAILABLE = True
except ImportError:
    ART_AUTOPGD_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class AutoPGDWrapper:
    """Wrapper for ART's Auto-PGD (Croce & Hein, 2020).

    Automatically tunes the step size during PGD iterations and uses
    multiple random restarts for more reliable adversarial evaluation.
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
        nb_random_start: int = 5,
        loss_type: Optional[str] = None,
        verbose: bool = True,
    ):
        if not ART_AUTOPGD_AVAILABLE:
            raise ImportError("ART AutoProjectedGradientDescent not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        norm_val: Any = np.inf if norm == "inf" else int(norm)
        kwargs: dict[str, Any] = {
            "estimator": estimator,
            "eps": eps,
            "eps_step": eps_step,
            "max_iter": max_iter,
            "targeted": targeted,
            "batch_size": batch_size,
            "norm": norm_val,
            "nb_random_start": nb_random_start,
            "verbose": verbose,
        }
        if loss_type is not None:
            kwargs["loss_type"] = loss_type

        self.art_attack = ARTAutoPGD(**kwargs)
        self.attack_params = {
            "eps": eps, "eps_step": eps_step, "max_iter": max_iter,
            "targeted": targeted, "batch_size": batch_size, "norm": norm,
            "nb_random_start": nb_random_start, "loss_type": loss_type,
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
