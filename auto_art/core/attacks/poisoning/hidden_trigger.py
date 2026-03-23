"""
Hidden Trigger Backdoor attack wrapper.

Reference: Saha et al., "Hidden Trigger Backdoor Attacks", AAAI 2020
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.poisoning import HiddenTriggerBackdoor as ARTHiddenTrigger
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class HiddenTriggerBackdoorWrapper:
    """Wrapper for ART's Hidden Trigger Backdoor attack."""

    def __init__(self, classifier: Any, target: int = 0, source: int = 1,
                 feature_layer: str = "", eps: float = 0.1,
                 learning_rate: float = 0.01, decay_coeff: float = 0.5,
                 decay_iter: int = 2000, max_iter: int = 5000,
                 batch_size: int = 32, poison_percent: float = 0.1,
                 verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART HiddenTriggerBackdoor not available.")
        self.art_attack = ARTHiddenTrigger(
            classifier=classifier, target=target, source=source,
            feature_layer=feature_layer, eps=eps,
            learning_rate=learning_rate, decay_coeff=decay_coeff,
            decay_iter=decay_iter, max_iter=max_iter,
            batch_size=batch_size, poison_percent=poison_percent,
            verbose=verbose,
        )
        self.attack_params = {"target": target, "source": source, "eps": eps,
                              "max_iter": max_iter, "poison_percent": poison_percent}

    def generate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        return self.art_attack.poison(x, y, **kwargs)
