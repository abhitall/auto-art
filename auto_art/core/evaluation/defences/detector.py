"""
ART Poisoning Detection defence wrappers.

Wraps ART's poisoning detection defences (Activation Defence,
Spectral Signatures, Provenance-based) into the auto-art DefenceStrategy
interface.

These detectors identify and filter poisoned samples from training data
to prevent backdoor attacks.

Reference: ART Defences Wiki - Detector section
"""

from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging
import numpy as np

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.detector.poison import (
        ActivationDefence as ARTActivationDefence,
        SpectralSignatureDefense as ARTSpectralSignatureDefense,
    )
    ART_DETECTORS_AVAILABLE = True
except ImportError:
    ART_DETECTORS_AVAILABLE = False


@dataclass
class PoisonDetectionReport:
    """Report from a poisoning detection analysis."""
    total_samples: int
    detected_poison: int
    clean_samples: int
    detection_rate: float
    confidence_scores: Optional[np.ndarray] = None
    is_clean_mask: Optional[np.ndarray] = None
    cluster_info: Dict[str, Any] = field(default_factory=dict)


class ActivationDefenceWrapper(DefenceStrategy):
    """Activation Defence for poisoning detection (Chen et al., 2018).

    Detects poisoned training samples by clustering the activations
    of a neural network's hidden layers. Poisoned samples tend to form
    distinct clusters in the activation space.
    """

    def __init__(
        self,
        nb_clusters: int = 2,
        clustering_method: str = "KMeans",
        nb_dims: int = 10,
        reduce: str = "PCA",
        cluster_analysis: str = "smaller",
    ):
        super().__init__(defence_name="ActivationDefence")
        self.nb_clusters = nb_clusters
        self.clustering_method = clustering_method
        self.nb_dims = nb_dims
        self.reduce = reduce
        self.cluster_analysis = cluster_analysis
        self._detector: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply activation-based poison detection.

        Requires x_train and y_train in kwargs for the detection analysis.
        Returns the original estimator (detection is analytical, not a
        model transformation).
        """
        if not ART_DETECTORS_AVAILABLE:
            raise ImportError("ART detector defences not available.")

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is None or y_train is None:
            logger.warning(
                "ActivationDefence requires x_train and y_train. "
                "Returning estimator without detection."
            )
            return art_estimator

        self._detector = ARTActivationDefence(
            classifier=art_estimator,
            x_train=x_train,
            y_train=y_train,
            nb_clusters=self.nb_clusters,
            clustering_method=self.clustering_method,
            nb_dims=self.nb_dims,
            reduce=self.reduce,
            cluster_analysis=self.cluster_analysis,
        )

        return art_estimator

    def detect_poison(
        self,
        art_estimator: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> PoisonDetectionReport:
        """Run the full poison detection pipeline.

        Returns:
            PoisonDetectionReport with detection results.
        """
        if not ART_DETECTORS_AVAILABLE:
            raise ImportError("ART detector defences not available.")

        self._detector = ARTActivationDefence(
            classifier=art_estimator,
            x_train=x_train,
            y_train=y_train,
            nb_clusters=self.nb_clusters,
            clustering_method=self.clustering_method,
            nb_dims=self.nb_dims,
            reduce=self.reduce,
            cluster_analysis=self.cluster_analysis,
        )

        report = self._detector.detect_poison(
            nb_clusters=self.nb_clusters,
            nb_dims=self.nb_dims,
            reduce=self.reduce,
        )

        is_clean = self._detector.is_clean
        is_clean_arr = np.array(is_clean) if is_clean is not None else None
        clean_count = int(np.sum(is_clean_arr)) if is_clean_arr is not None else 0
        poison_count = len(x_train) - clean_count

        return PoisonDetectionReport(
            total_samples=len(x_train),
            detected_poison=poison_count,
            clean_samples=clean_count,
            detection_rate=poison_count / len(x_train) if len(x_train) > 0 else 0.0,
            is_clean_mask=is_clean_arr,
            cluster_info=report if isinstance(report, dict) else {},
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "nb_clusters": self.nb_clusters,
            "clustering_method": self.clustering_method,
            "nb_dims": self.nb_dims,
            "reduce": self.reduce,
            "cluster_analysis": self.cluster_analysis,
        }


class SpectralSignatureDefenceWrapper(DefenceStrategy):
    """Spectral Signature Defence for poisoning detection (Tran et al., 2018).

    Detects poisoned training samples using spectral signatures of
    the learned representations. Backdoor attacks leave detectable
    spectral traces in the covariance spectrum of feature representations.
    """

    def __init__(
        self,
        expected_pp_poison: float = 0.1,
        batch_size: int = 128,
    ):
        super().__init__(defence_name="SpectralSignatureDefence")
        self.expected_pp_poison = expected_pp_poison
        self.batch_size = batch_size
        self._detector: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_DETECTORS_AVAILABLE:
            raise ImportError("ART detector defences not available.")

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is None or y_train is None:
            logger.warning(
                "SpectralSignatureDefence requires x_train and y_train."
            )
            return art_estimator

        self._detector = ARTSpectralSignatureDefense(
            classifier=art_estimator,
            x_train=x_train,
            y_train=y_train,
            expected_pp_poison=self.expected_pp_poison,
            batch_size=self.batch_size,
        )

        return art_estimator

    def detect_poison(
        self,
        art_estimator: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> PoisonDetectionReport:
        """Run spectral signature poison detection."""
        if not ART_DETECTORS_AVAILABLE:
            raise ImportError("ART detector defences not available.")

        self._detector = ARTSpectralSignatureDefense(
            classifier=art_estimator,
            x_train=x_train,
            y_train=y_train,
            expected_pp_poison=self.expected_pp_poison,
            batch_size=self.batch_size,
        )

        report = self._detector.detect_poison()

        is_clean = self._detector.is_clean
        is_clean_arr = np.array(is_clean) if is_clean is not None else None
        clean_count = int(np.sum(is_clean_arr)) if is_clean_arr is not None else 0
        poison_count = len(x_train) - clean_count

        return PoisonDetectionReport(
            total_samples=len(x_train),
            detected_poison=poison_count,
            clean_samples=clean_count,
            detection_rate=poison_count / len(x_train) if len(x_train) > 0 else 0.0,
            is_clean_mask=is_clean_arr,
            cluster_info=report if isinstance(report, dict) else {},
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "expected_pp_poison": self.expected_pp_poison,
            "batch_size": self.batch_size,
        }
