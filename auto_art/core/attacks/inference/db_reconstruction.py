"""
Database reconstruction attack wrapper.

Wraps ART's DatabaseReconstruction attack that attempts to reconstruct
training data from a target model's learned parameters.

Reference: ART documentation on reconstruction-based inference attacks.
"""

from typing import Any, Optional
import numpy as np

try:
    from art.attacks.inference.reconstruction import (
        DatabaseReconstruction as ARTDatabaseReconstruction,
    )
    DB_RECONSTRUCTION_AVAILABLE = True
except ImportError:
    DB_RECONSTRUCTION_AVAILABLE = False


class DatabaseReconstructionWrapper:
    """Wrapper for ART's DatabaseReconstruction attack.

    Reconstructs training data samples from a target model by exploiting
    information leaked through the model's parameters and predictions.
    """

    def __init__(
        self,
        estimator: Any,
        **kwargs,
    ):
        if not DB_RECONSTRUCTION_AVAILABLE:
            raise ImportError(
                "ART DatabaseReconstruction not available. "
                "Ensure adversarial-robustness-toolbox is installed."
            )
        self.estimator = estimator
        self.art_attack = ARTDatabaseReconstruction(
            estimator=estimator,
            **kwargs,
        )
        self.attack_params: dict = {}

    @property
    def attack(self) -> Any:
        return self.art_attack

    def reconstruct(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.reconstruct(x=x, y=y, **kwargs)
