"""
GeoDA (Geometric Decision-based Attack) evasion attack wrapper.

Wraps ART's GeoDA into auto-art's wrapper pattern.
GeoDA is a decision-based attack that estimates the normal vector
to the decision boundary using geometric properties.

Reference: Rahmati et al., "GeoDA: A Geometric Framework for
Black-box Adversarial Attacks", CVPR 2020
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import GeoDA as ARTGeoDA
    from art.estimators.classification import ClassifierMixin
    ART_GEODA_AVAILABLE = True
except ImportError:
    ART_GEODA_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class GeoDAWrapper:
    """Wrapper for ART's GeoDA (Rahmati et al., 2020).

    A decision-based black-box attack that estimates the decision
    boundary's geometry to craft minimal adversarial perturbations.
    """

    def __init__(
        self,
        estimator: Any,
        norm: str = "2",
        sub_dim: int = 75,
        max_iter: int = 4000,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        if not ART_GEODA_AVAILABLE:
            raise ImportError("ART GeoDA not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        norm_val: Any = np.inf if norm == "inf" else int(norm)
        self.art_attack = ARTGeoDA(
            estimator=estimator,
            norm=norm_val,
            sub_dim=sub_dim,
            max_iter=max_iter,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "norm": norm, "sub_dim": sub_dim,
            "max_iter": max_iter, "batch_size": batch_size,
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
