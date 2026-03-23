"""
ART BEYOND Detector defence wrapper.

Wraps ART's BeyondDetectorPyTorch into the auto-art DefenceStrategy
interface for detecting adversarial examples using BEYOND's neighborhood
analysis approach.

Reference: Yin et al., 2023 - "BEYOND: Detecting Adversarial Examples
with Bayesian Estimation of Yielded Outliers from Neighborhood Density"
"""

from typing import Any, Dict, Optional
import logging
import numpy as np

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.detector.evasion import (
        BeyondDetectorPyTorch as ARTBeyondDetectorPyTorch,
    )
    ART_BEYOND_DETECTOR_AVAILABLE = True
except ImportError:
    ART_BEYOND_DETECTOR_AVAILABLE = False


class BEYONDDetectorWrapper(DefenceStrategy):
    """BEYOND Detector for adversarial example detection (Yin et al., 2023).

    Detects adversarial examples using Bayesian estimation of yielded
    outliers from neighborhood density. Analyzes the local neighborhood
    of each input in the feature space to identify outliers that are
    likely adversarial.
    """

    def __init__(
        self,
        classifier: Optional[Any] = None,
        nb_classes: int = 10,
        nb_neighbors: int = 50,
        batch_size: int = 128,
    ):
        super().__init__(defence_name="BEYONDDetector")
        self.classifier = classifier
        self.nb_classes = nb_classes
        self.nb_neighbors = nb_neighbors
        self.batch_size = batch_size
        self._detector: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply BEYOND detection to the estimator.

        Uses the provided art_estimator (or self.classifier) as the base
        classifier for neighborhood density analysis.
        """
        if not ART_BEYOND_DETECTOR_AVAILABLE:
            raise ImportError(
                "ART BeyondDetectorPyTorch not available. "
                "Install adversarial-robustness-toolbox with PyTorch support."
            )

        target_classifier = self.classifier or art_estimator

        self._detector = ARTBeyondDetectorPyTorch(
            classifier=target_classifier,
            nb_classes=self.nb_classes,
            nb_neighbors=self.nb_neighbors,
            batch_size=self.batch_size,
        )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is not None and y_train is not None:
            logger.info(
                f"Fitting BEYOND detector on {len(x_train)} samples with "
                f"{self.nb_neighbors} neighbors."
            )
            self._detector.fit(x_train, y_train)

        return art_estimator

    def detect(self, x: np.ndarray) -> np.ndarray:
        """Detect adversarial examples using BEYOND neighborhood analysis.

        Returns:
            Binary array where 1 indicates adversarial, 0 indicates clean.
        """
        if self._detector is None:
            raise RuntimeError(
                "Detector not initialized. Call apply() first."
            )
        _, is_adversarial = self._detector.detect(x)
        return is_adversarial

    def get_params(self) -> Dict[str, Any]:
        return {
            "nb_classes": self.nb_classes,
            "nb_neighbors": self.nb_neighbors,
            "batch_size": self.batch_size,
        }
