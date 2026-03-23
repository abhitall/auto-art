"""
Adversarial Texture attack wrapper for PyTorch.

Generates adversarial textures for 3D objects in physical-world scenarios.
Reference: ART - AdversarialTexturePyTorch
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import AdversarialTexturePyTorch as ARTAdversarialTexture
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class AdversarialTextureWrapper:
    def __init__(self, estimator: Any, patch_height: int = 16, patch_width: int = 16,
                 x_min: float = 0.0, x_max: float = 1.0, step_size: float = 1.0 / 255,
                 max_iter: int = 500, batch_size: int = 16, verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART AdversarialTexturePyTorch not available.")
        self.art_attack = ARTAdversarialTexture(
            estimator=estimator, patch_height=patch_height, patch_width=patch_width,
            x_min=x_min, x_max=x_max, step_size=step_size, max_iter=max_iter,
            batch_size=batch_size, verbose=verbose,
        )
        self.attack_params = {"patch_height": patch_height, "patch_width": patch_width,
                              "max_iter": max_iter, "step_size": step_size}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
