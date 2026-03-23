"""
Sign-OPT attack wrapper — query-efficient sign optimization.

Reference: Cheng et al., "Sign-OPT: A Query-Efficient Hard-label Adversarial Attack", 2020
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import SignOPTAttack as ARTSignOPT
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class SignOPTWrapper:
    def __init__(self, estimator: Any, targeted: bool = False, epsilon: float = 0.3,
                 max_iter: int = 1000, num_trial: int = 100, k: int = 200,
                 alpha: float = 0.2, beta: float = 0.001, batch_size: int = 1,
                 verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART SignOPTAttack not available.")
        self.art_attack = ARTSignOPT(
            estimator=estimator, targeted=targeted, epsilon=epsilon,
            max_iter=max_iter, num_trial=num_trial, k=k,
            alpha=alpha, beta=beta, batch_size=batch_size, verbose=verbose,
        )
        self.attack_params = {"targeted": targeted, "epsilon": epsilon, "max_iter": max_iter,
                              "num_trial": num_trial, "k": k}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
