"""
Spatial Transformation evasion attack wrapper.

Wraps ART's SpatialTransformation into auto-art's wrapper pattern.
Spatial Transformation attacks fool classifiers via geometric
transformations (translations and rotations) rather than additive
perturbations.

Reference: Engstrom et al., "A Rotation and a Translation Suffice:
Fooling CNNs with Simple Transformations", ICML 2019
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import SpatialTransformation as ARTSpatialTransformation
    from art.estimators.classification import ClassifierMixin
    ART_SPATIAL_AVAILABLE = True
except ImportError:
    ART_SPATIAL_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class SpatialTransformationWrapper:
    """Wrapper for ART's Spatial Transformation (Engstrom et al., 2019).

    Searches over rotation and translation parameters to find
    geometric transformations that cause misclassification.
    """

    def __init__(
        self,
        estimator: Any,
        max_translation: float = 0.3,
        num_translations: int = 5,
        max_rotation: float = 30.0,
        num_rotations: int = 5,
        verbose: bool = True,
    ):
        if not ART_SPATIAL_AVAILABLE:
            raise ImportError("ART SpatialTransformation not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTSpatialTransformation(
            classifier=estimator,
            max_translation=max_translation,
            num_translations=num_translations,
            max_rotation=max_rotation,
            num_rotations=num_rotations,
            verbose=verbose,
        )
        self.attack_params = {
            "max_translation": max_translation,
            "num_translations": num_translations,
            "max_rotation": max_rotation,
            "num_rotations": num_rotations,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
