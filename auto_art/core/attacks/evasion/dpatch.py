"""
DPatch and RobustDPatch attack wrappers for object detection.

Reference: Liu et al., "DPatch: An Adversarial Patch Attack on Object Detectors", 2019
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import DPatch as ARTDPatch
    ART_DPATCH_AVAILABLE = True
except ImportError:
    ART_DPATCH_AVAILABLE = False

try:
    from art.attacks.evasion import RobustDPatch as ARTRobustDPatch
    ART_ROBUST_DPATCH_AVAILABLE = True
except ImportError:
    ART_ROBUST_DPATCH_AVAILABLE = False


class DPatchWrapper:
    def __init__(self, estimator: Any, patch_shape: tuple = (40, 40, 3),
                 learning_rate: float = 5.0, max_iter: int = 500,
                 batch_size: int = 16, verbose: bool = True):
        if not ART_DPATCH_AVAILABLE:
            raise ImportError("ART DPatch not available.")
        self.art_attack = ARTDPatch(
            estimator=estimator, patch_shape=patch_shape,
            learning_rate=learning_rate, max_iter=max_iter,
            batch_size=batch_size, verbose=verbose,
        )
        self.attack_params = {"patch_shape": patch_shape, "learning_rate": learning_rate,
                              "max_iter": max_iter}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[Any] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)


class RobustDPatchWrapper:
    def __init__(self, estimator: Any, patch_shape: tuple = (40, 40, 3),
                 patch_location: Optional[tuple] = None, crop_range: tuple = (0, 0),
                 brightness_range: tuple = (1.0, 1.0), rotation_weights: tuple = (1, 0, 0, 0),
                 sample_size: int = 1, learning_rate: float = 5.0, max_iter: int = 500,
                 batch_size: int = 16, verbose: bool = True):
        if not ART_ROBUST_DPATCH_AVAILABLE:
            raise ImportError("ART RobustDPatch not available.")
        self.art_attack = ARTRobustDPatch(
            estimator=estimator, patch_shape=patch_shape,
            patch_location=patch_location, crop_range=crop_range,
            brightness_range=brightness_range, rotation_weights=rotation_weights,
            sample_size=sample_size, learning_rate=learning_rate,
            max_iter=max_iter, batch_size=batch_size, verbose=verbose,
        )
        self.attack_params = {"patch_shape": patch_shape, "learning_rate": learning_rate,
                              "max_iter": max_iter, "sample_size": sample_size}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[Any] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
