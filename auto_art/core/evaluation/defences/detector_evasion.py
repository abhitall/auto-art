"""
ART Evasion Detection defence wrappers.

Wraps ART's evasion detection defences (basic input detection, binary
activation detection, subset scanning detection) into the auto-art
DefenceStrategy interface.

These detectors identify adversarial examples at inference time.

Reference:
  - Binary Activation Detector: Sperl et al., 2020 - "DLA: Dense-Layer-Analysis
    for Adversarial Example Detection"
  - Subset Scanning: Cintas et al., 2020 - "Detecting Adversarial Attacks via
    Subset Scanning of Autoencoder Activations and Reconstruction Error"
"""

from typing import Any, Dict, Optional
import logging
import numpy as np

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.detector.evasion import (
        BinaryInputDetector as ARTBinaryInputDetector,
    )
    ART_BASIC_DETECTOR_AVAILABLE = True
except ImportError:
    ART_BASIC_DETECTOR_AVAILABLE = False

try:
    from art.defences.detector.evasion import (
        BinaryActivationDetector as ARTBinaryActivationDetector,
    )
    ART_ACTIVATION_DETECTOR_AVAILABLE = True
except ImportError:
    ART_ACTIVATION_DETECTOR_AVAILABLE = False

try:
    from art.defences.detector.evasion import (
        SubsetScanningDetector as ARTSubsetScanningDetector,
    )
    ART_SUBSET_SCAN_DETECTOR_AVAILABLE = True
except ImportError:
    ART_SUBSET_SCAN_DETECTOR_AVAILABLE = False


class BasicInputDetectorWrapper(DefenceStrategy):
    """Basic binary input detector for evasion detection.

    Uses a binary classifier trained to distinguish between clean and
    adversarial inputs. The detector classifier is trained separately
    and wrapped around the target model.
    """

    def __init__(self):
        super().__init__(defence_name="BasicInputDetector")
        self._detector: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply basic input detection to the estimator.

        Requires a `detector_classifier` in kwargs — a trained binary
        classifier that distinguishes clean from adversarial inputs.
        """
        if not ART_BASIC_DETECTOR_AVAILABLE:
            raise ImportError(
                "ART BinaryInputDetector not available. "
                "Install adversarial-robustness-toolbox."
            )

        detector_classifier = kwargs.get("detector_classifier")
        if detector_classifier is None:
            logger.warning(
                "BasicInputDetector requires a detector_classifier in kwargs. "
                "Returning estimator without detection."
            )
            return art_estimator

        self._detector = ARTBinaryInputDetector(
            detector=detector_classifier,
        )

        return art_estimator

    def detect(self, x: np.ndarray) -> np.ndarray:
        """Detect adversarial examples in input batch.

        Returns:
            Binary array where 1 indicates adversarial, 0 indicates clean.
        """
        if self._detector is None:
            raise RuntimeError(
                "Detector not initialized. Call apply() first."
            )
        return self._detector.detect(x)

    def get_params(self) -> Dict[str, Any]:
        return {}


class ActivationDetectorWrapper(DefenceStrategy):
    """Binary Activation Detector for evasion detection (Sperl et al., 2020).

    Detects adversarial examples by analyzing the activation patterns of
    hidden layers. A binary classifier is trained on the activations
    produced by clean vs adversarial inputs.
    """

    def __init__(self, hidden_layer_index: int = -1):
        super().__init__(defence_name="ActivationDetector")
        self.hidden_layer_index = hidden_layer_index
        self._detector: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply activation-based evasion detection.

        Requires `detector_classifier` (trained binary activation detector)
        and optionally `x_train`/`y_train` for calibration.
        """
        if not ART_ACTIVATION_DETECTOR_AVAILABLE:
            raise ImportError(
                "ART BinaryActivationDetector not available. "
                "Install adversarial-robustness-toolbox."
            )

        detector_classifier = kwargs.get("detector_classifier")
        if detector_classifier is None:
            logger.warning(
                "ActivationDetector requires a detector_classifier in kwargs. "
                "Returning estimator without detection."
            )
            return art_estimator

        self._detector = ARTBinaryActivationDetector(
            classifier=art_estimator,
            detector=detector_classifier,
        )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")
        if x_train is not None and y_train is not None:
            logger.info(
                f"Fitting activation detector on {len(x_train)} samples."
            )
            self._detector.fit(x_train, y_train, batch_size=128, nb_epochs=10)

        return art_estimator

    def detect(self, x: np.ndarray) -> np.ndarray:
        """Detect adversarial examples via activation analysis."""
        if self._detector is None:
            raise RuntimeError(
                "Detector not initialized. Call apply() first."
            )
        return self._detector.detect(x)

    def get_params(self) -> Dict[str, Any]:
        return {
            "hidden_layer_index": self.hidden_layer_index,
        }


class SubsetScanDetectorWrapper(DefenceStrategy):
    """Subset Scanning Detector for evasion detection (Cintas et al., 2020).

    Detects adversarial examples using subset scanning over autoencoder
    activations and reconstruction errors. Identifies anomalous subsets
    of features that are indicative of adversarial manipulation.
    """

    def __init__(self, bgd_data: Optional[np.ndarray] = None):
        super().__init__(defence_name="SubsetScanDetector")
        self.bgd_data = bgd_data
        self._detector: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply subset scanning evasion detection.

        Requires `bgd_data` (background/clean data) either at init or in kwargs.
        """
        if not ART_SUBSET_SCAN_DETECTOR_AVAILABLE:
            raise ImportError(
                "ART SubsetScanningDetector not available. "
                "Install adversarial-robustness-toolbox."
            )

        bgd_data = kwargs.get("bgd_data", self.bgd_data)
        if bgd_data is None:
            logger.warning(
                "SubsetScanDetector requires bgd_data (background data). "
                "Returning estimator without detection."
            )
            return art_estimator

        self._detector = ARTSubsetScanningDetector(
            classifier=art_estimator,
            bgd_data=bgd_data,
        )

        return art_estimator

    def detect(self, x: np.ndarray) -> np.ndarray:
        """Detect adversarial examples via subset scanning."""
        if self._detector is None:
            raise RuntimeError(
                "Detector not initialized. Call apply() first."
            )
        _, is_adversarial = self._detector.detect(x)
        return is_adversarial

    def get_params(self) -> Dict[str, Any]:
        return {
            "bgd_data_shape": self.bgd_data.shape if self.bgd_data is not None else None,
        }
