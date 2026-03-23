"""
ART Poison Provenance Detection defence wrappers.

Wraps ART's provenance-based poisoning detection defences (ProvenanceDefense,
RONIDefense) into the auto-art DefenceStrategy interface.

These detectors use data provenance information or training data influence
analysis to identify and filter poisoned training samples.

Reference:
  - ProvenanceDefense: Baracaldo et al., 2017 - "Mitigating Poisoning Attacks
    on Machine Learning Models: A Data Provenance Based Approach"
  - RONI: Nelson et al., 2008 - "Exploiting Machine Learning to Subvert Your
    Spam Filter" (Reject on Negative Impact)
"""

from typing import Any, Dict, Optional
import logging
import numpy as np

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.detector.poison import (
        ProvenanceDefense as ARTProvenanceDefense,
    )
    ART_PROVENANCE_AVAILABLE = True
except ImportError:
    ART_PROVENANCE_AVAILABLE = False

try:
    from art.defences.detector.poison import (
        RONIDefense as ARTRONIDefense,
    )
    ART_RONI_AVAILABLE = True
except ImportError:
    ART_RONI_AVAILABLE = False


class DataProvenanceDefenceWrapper(DefenceStrategy):
    """Provenance-based poison detection defence (Baracaldo et al., 2017).

    Uses data provenance information (trusted source labels for each
    training sample) to identify potentially poisoned samples from
    untrusted sources. Combines provenance metadata with statistical
    analysis to flag suspicious data points.
    """

    def __init__(
        self,
        classifier: Optional[Any] = None,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        p_train: Optional[np.ndarray] = None,
    ):
        super().__init__(defence_name="DataProvenanceDefence")
        self.classifier = classifier
        self.x_train = x_train
        self.y_train = y_train
        self.p_train = p_train
        self._detector: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply provenance-based poison detection.

        Requires x_train, y_train, and p_train (provenance labels).
        These can be set at init or passed via kwargs.
        """
        if not ART_PROVENANCE_AVAILABLE:
            raise ImportError(
                "ART ProvenanceDefense not available. "
                "Install adversarial-robustness-toolbox."
            )

        target_classifier = self.classifier or art_estimator
        x_train = kwargs.get("x_train", self.x_train)
        y_train = kwargs.get("y_train", self.y_train)
        p_train = kwargs.get("p_train", self.p_train)

        if x_train is None or y_train is None or p_train is None:
            logger.warning(
                "DataProvenanceDefence requires x_train, y_train, and p_train "
                "(provenance labels). Returning estimator without detection."
            )
            return art_estimator

        self._detector = ARTProvenanceDefense(
            classifier=target_classifier,
            x_train=x_train,
            y_train=y_train,
            p_train=p_train,
        )

        logger.info(
            f"Provenance defence initialized with {len(x_train)} samples."
        )

        return art_estimator

    def detect_poison(self) -> Dict[str, Any]:
        """Run provenance-based poison detection.

        Returns:
            Dictionary with detection results including is_clean mask
            and detection report.
        """
        if self._detector is None:
            raise RuntimeError(
                "Detector not initialized. Call apply() first."
            )
        report = self._detector.detect_poison()
        is_clean = self._detector.is_clean
        return {
            "report": report,
            "is_clean": np.array(is_clean) if is_clean is not None else None,
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "x_train_shape": self.x_train.shape if self.x_train is not None else None,
            "y_train_shape": self.y_train.shape if self.y_train is not None else None,
            "p_train_shape": self.p_train.shape if self.p_train is not None else None,
        }


class RONIDefenceWrapper(DefenceStrategy):
    """RONI (Reject on Negative Impact) poison detection (Nelson et al., 2008).

    Identifies poisoned training samples by measuring the impact of each
    sample on model performance. Samples that negatively impact
    performance when included in training are flagged as potentially
    poisoned and rejected.
    """

    def __init__(
        self,
        classifier: Optional[Any] = None,
        x_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ):
        super().__init__(defence_name="RONIDefence")
        self.classifier = classifier
        self.x_train = x_train
        self.y_train = y_train
        self._detector: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply RONI-based poison detection.

        Requires x_train and y_train. These can be set at init or passed
        via kwargs.
        """
        if not ART_RONI_AVAILABLE:
            raise ImportError(
                "ART RONIDefense not available. "
                "Install adversarial-robustness-toolbox."
            )

        target_classifier = self.classifier or art_estimator
        x_train = kwargs.get("x_train", self.x_train)
        y_train = kwargs.get("y_train", self.y_train)

        if x_train is None or y_train is None:
            logger.warning(
                "RONIDefence requires x_train and y_train. "
                "Returning estimator without detection."
            )
            return art_estimator

        self._detector = ARTRONIDefense(
            classifier=target_classifier,
            x_train=x_train,
            y_train=y_train,
        )

        logger.info(
            f"RONI defence initialized with {len(x_train)} samples."
        )

        return art_estimator

    def detect_poison(self) -> Dict[str, Any]:
        """Run RONI-based poison detection.

        Returns:
            Dictionary with detection results including is_clean mask
            and detection report.
        """
        if self._detector is None:
            raise RuntimeError(
                "Detector not initialized. Call apply() first."
            )
        report = self._detector.detect_poison()
        is_clean = self._detector.is_clean
        return {
            "report": report,
            "is_clean": np.array(is_clean) if is_clean is not None else None,
        }

    def get_params(self) -> Dict[str, Any]:
        return {
            "x_train_shape": self.x_train.shape if self.x_train is not None else None,
            "y_train_shape": self.y_train.shape if self.y_train is not None else None,
        }
