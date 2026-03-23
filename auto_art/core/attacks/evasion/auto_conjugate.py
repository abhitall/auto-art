"""
Auto Conjugate Gradient (Auto-CG) attack wrapper.

Reference: ART - AutoConjugateGradient
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import AutoConjugateGradient as ARTAutoCG
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class AutoConjugateGradientWrapper:
    def __init__(self, estimator: Any, eps: float = 0.3, eps_step: float = 0.1,
                 max_iter: int = 100, targeted: bool = False, nb_random_start: int = 5,
                 batch_size: int = 32, loss_type: Optional[str] = None, verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART AutoConjugateGradient not available.")
        kwargs: dict = {"estimator": estimator, "eps": eps, "eps_step": eps_step,
                        "max_iter": max_iter, "targeted": targeted,
                        "nb_random_start": nb_random_start,
                        "batch_size": batch_size, "verbose": verbose}
        if loss_type is not None:
            kwargs["loss_type"] = loss_type
        self.art_attack = ARTAutoCG(**kwargs)
        self.attack_params = {"eps": eps, "eps_step": eps_step, "max_iter": max_iter,
                              "targeted": targeted, "nb_random_start": nb_random_start}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
