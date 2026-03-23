"""
Bullseye Polytope attack wrapper — targeted data poisoning.

Reference: Aghakhani et al., "Bullseye Polytope", NeurIPS 2021
"""
from typing import Any, Optional, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.poisoning import BullseyePolytopeAttackPyTorch as ARTBullseye
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class BullseyePolytopeWrapper:
    """Wrapper for ART's Bullseye Polytope targeted poisoning attack."""

    def __init__(self, classifier: Any, target: np.ndarray,
                 feature_layer: str = "", opt: str = "adam",
                 max_iter: int = 500, learning_rate: float = 0.04,
                 momentum: float = 0.9, decay_iter: int = 10000,
                 decay_coeff: float = 0.5, epsilon: float = 0.1,
                 dropout: float = 0.3, net_repeat: int = 1,
                 endtoend: bool = True, batch_size: int = 64,
                 verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART BullseyePolytopeAttackPyTorch not available.")
        self.art_attack = ARTBullseye(
            classifier=classifier, target=target, feature_layer=feature_layer,
            opt=opt, max_iter=max_iter, learning_rate=learning_rate,
            momentum=momentum, decay_iter=decay_iter, decay_coeff=decay_coeff,
            epsilon=epsilon, dropout=dropout, net_repeat=net_repeat,
            endtoend=endtoend, batch_size=batch_size, verbose=verbose,
        )
        self.attack_params = {"opt": opt, "max_iter": max_iter, "epsilon": epsilon,
                              "learning_rate": learning_rate}

    def generate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        return self.art_attack.poison(x, y, **kwargs)
