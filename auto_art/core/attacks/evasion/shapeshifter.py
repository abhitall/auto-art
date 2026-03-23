"""
ShapeShifter attack wrapper for object detection misclassification.

Reference: Chen et al., "ShapeShifter: Robust Physical Adversarial Attack on Faster R-CNN"
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import ShapeShifter as ARTShapeShifter
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class ShapeShifterWrapper:
    def __init__(self, estimator: Any, random_transform: Optional[Any] = None,
                 batch_size: int = 1, max_iter: int = 500,
                 texture_as_input: bool = False, verbose: bool = True, **kwargs):
        if not ART_AVAILABLE:
            raise ImportError("ART ShapeShifter not available.")
        init_kwargs: dict = {"estimator": estimator, "batch_size": batch_size,
                             "max_iter": max_iter, "texture_as_input": texture_as_input,
                             "verbose": verbose}
        if random_transform is not None:
            init_kwargs["random_transform"] = random_transform
        init_kwargs.update(kwargs)
        self.art_attack = ARTShapeShifter(**init_kwargs)
        self.attack_params = {"max_iter": max_iter, "texture_as_input": texture_as_input}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(self, x: np.ndarray, y: Optional[Any] = None, **kwargs) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
