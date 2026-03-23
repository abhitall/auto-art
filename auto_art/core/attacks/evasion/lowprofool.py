"""
LowProFool attack wrapper for tabular/low-profile perturbations.

Reference: Ballet et al., "Imperceptible Adversarial Examples on Tabular Data"
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import LowProFool as ARTLowProFool
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class LowProFoolWrapper:
    def __init__(self, estimator: Any, n_steps: int = 100,
                 threshold: Optional[float] = None, lambd: float = 1.5,
                 eta: float = 0.02, eta_decay: float = 0.98,
                 eta_min: float = 1e-7, norm: int = 2,
                 importance: Optional[np.ndarray] = None,
                 batch_size: int = 1, verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART LowProFool not available.")
        kwargs: dict = {"classifier": estimator, "n_steps": n_steps, "lambd": lambd,
                        "eta": eta, "eta_decay": eta_decay, "eta_min": eta_min,
                        "norm": norm, "batch_size": batch_size, "verbose": verbose}
        if threshold is not None:
            kwargs["threshold"] = threshold
        if importance is not None:
            kwargs["importance"] = importance
        self.art_attack = ARTLowProFool(**kwargs)
        self.attack_params = {"n_steps": n_steps, "lambd": lambd, "eta": eta, "norm": norm}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
