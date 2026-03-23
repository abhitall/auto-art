"""
Virtual Adversarial Method wrapper.

Reference: Miyato et al., "Virtual Adversarial Training", ICLR 2016
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import VirtualAdversarialMethod as ARTVirtualAdversarial
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class VirtualAdversarialWrapper:
    def __init__(self, estimator: Any, eps: float = 0.1, finite_diff: float = 1e-6,
                 max_iter: int = 1, batch_size: int = 1, verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART VirtualAdversarialMethod not available.")
        self.art_attack = ARTVirtualAdversarial(
            classifier=estimator, eps=eps, finite_diff=finite_diff,
            max_iter=max_iter, batch_size=batch_size, verbose=verbose,
        )
        self.attack_params = {"eps": eps, "finite_diff": finite_diff, "max_iter": max_iter}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
