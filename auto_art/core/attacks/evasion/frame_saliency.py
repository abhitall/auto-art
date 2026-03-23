"""
Frame Saliency attack wrapper for video adversarial examples.

Reference: ART - FrameSaliencyAttack
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import FrameSaliencyAttack as ARTFrameSaliency
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class FrameSaliencyWrapper:
    def __init__(self, estimator: Any, attacker: Any, method: str = "iterative_saliency",
                 frame_index: int = 1, batch_size: int = 1, verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART FrameSaliencyAttack not available.")
        self.art_attack = ARTFrameSaliency(
            classifier=estimator, attacker=attacker, method=method,
            frame_index=frame_index, batch_size=batch_size, verbose=verbose,
        )
        self.attack_params = {"method": method, "frame_index": frame_index}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
