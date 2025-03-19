"""
Robustness certification metrics.

Implements certification-based robustness evaluation using Randomized
Smoothing and the GREAT Score (ART 1.20) for generative AI robustness.

References:
- Randomized Smoothing: Cohen et al., 2019
- GREAT Score: ART 1.20.0 release
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.metrics import great_score as art_great_score
    GREAT_SCORE_AVAILABLE = True
except ImportError:
    GREAT_SCORE_AVAILABLE = False

try:
    from art.estimators.certification import RandomizedSmoothingMixin  # noqa: F401
    CERT_AVAILABLE = True
except ImportError:
    CERT_AVAILABLE = False


def compute_great_score(
    classifier: Any,
    x: np.ndarray,
    y: np.ndarray,
    nb_samples: int = 100,
    nb_classes: Optional[int] = None,
    **kwargs,
) -> Optional[float]:
    """Compute the GREAT Score for robustness evaluation with Generative AI.

    The GREAT Score (Global Robustness Evaluation of Adversarial Perturbation
    using Generative Models) provides a comprehensive robustness metric that
    leverages generative models to assess model vulnerability.

    Added in ART 1.20.0.

    Args:
        classifier: ART classifier estimator.
        x: Input data samples.
        y: True labels.
        nb_samples: Number of samples for evaluation.
        nb_classes: Number of classes. Auto-detected if None.

    Returns:
        GREAT score as float, or None if not available.
    """
    if not GREAT_SCORE_AVAILABLE:
        logger.warning(
            "GREAT Score not available. Requires ART >= 1.20.0. "
            "Install with: pip install adversarial-robustness-toolbox>=1.20.0"
        )
        return None

    try:
        score = art_great_score(
            classifier=classifier,
            x=x,
            y=y,
            nb_samples=nb_samples,
            **kwargs,
        )
        return float(score)
    except Exception as e:
        logger.error(f"GREAT Score computation failed: {e}")
        return None


class RandomizedSmoothingCertifier:
    """Randomized Smoothing certification for provable robustness.

    Provides certified robustness guarantees by constructing a smoothed
    classifier that is provably robust to L2 perturbations within a
    certified radius.

    Reference: Cohen et al., 2019 - "Certified Adversarial Robustness
    via Randomized Smoothing"
    """

    def __init__(
        self,
        sigma: float = 0.25,
        nb_samples: int = 100,
        alpha: float = 0.001,
        batch_size: int = 64,
    ):
        self.sigma = sigma
        self.nb_samples = nb_samples
        self.alpha = alpha
        self.batch_size = batch_size
        self.logger = logging.getLogger("auto_art.certification")

    def certify(
        self,
        smoothed_classifier: Any,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute certified radii for input samples.

        Args:
            smoothed_classifier: An ART smoothed classifier (must be a
                                 RandomizedSmoothingMixin instance).
            x: Input samples to certify.
            y: True labels (optional, for accuracy calculation).

        Returns:
            Dict with certification results including certified radii,
            certified accuracy at various epsilon thresholds, and summary.
        """
        if not CERT_AVAILABLE:
            self.logger.warning(
                "ART certification not available. "
                "Returning empty certification results."
            )
            return {"error": "ART certification not available"}

        predictions = smoothed_classifier.predict(x, batch_size=self.batch_size)
        predicted_classes = np.argmax(predictions, axis=1)

        certified_radii: List[float] = []

        for i in range(len(x)):
            try:
                pred, radius = smoothed_classifier.certify(
                    x[i:i + 1],
                    n=self.nb_samples,
                    batch_size=self.batch_size,
                )
                certified_radii.append(float(radius))
            except Exception as e:
                self.logger.debug(f"Certification failed for sample {i}: {e}")
                certified_radii.append(0.0)

        radii_array = np.array(certified_radii)

        epsilons = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        certified_accuracy: Dict[str, float] = {}
        for eps in epsilons:
            if y is not None:
                true_classes = np.argmax(y, axis=1) if y.ndim > 1 else y
                correct = predicted_classes == true_classes
                certified = radii_array >= eps
                cert_acc = float(np.mean(correct & certified))
            else:
                cert_acc = float(np.mean(radii_array >= eps))
            certified_accuracy[f"eps_{eps}"] = cert_acc

        return {
            "certified_radii": radii_array,
            "mean_certified_radius": float(np.mean(radii_array)),
            "median_certified_radius": float(np.median(radii_array)),
            "max_certified_radius": float(np.max(radii_array)),
            "min_certified_radius": float(np.min(radii_array)),
            "certified_accuracy": certified_accuracy,
            "sigma": self.sigma,
            "nb_samples": self.nb_samples,
            "total_samples": len(x),
        }

    def create_smoothed_classifier_pytorch(
        self,
        model: Any,
        nb_classes: int,
        input_shape: Tuple[int, ...],
        loss: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        clip_values: Tuple[float, float] = (0.0, 1.0),
    ) -> Any:
        """Create a PyTorch Randomized Smoothing classifier.

        Args:
            model: PyTorch nn.Module.
            nb_classes: Number of output classes.
            input_shape: Shape of a single input (without batch dim).
            loss: Loss function (torch criterion).
            optimizer: Optimizer instance.
            clip_values: Min/max values for input clipping.

        Returns:
            ART PyTorchRandomizedSmoothing estimator.
        """
        try:
            from art.estimators.certification.randomized_smoothing import (
                PyTorchRandomizedSmoothing,
            )
        except ImportError:
            raise ImportError(
                "PyTorchRandomizedSmoothing not available. "
                "Ensure ART is installed with PyTorch support."
            )

        import torch
        if loss is None:
            loss = torch.nn.CrossEntropyLoss()

        return PyTorchRandomizedSmoothing(
            model=model,
            loss=loss,
            optimizer=optimizer,
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=clip_values,
            sample_size=self.nb_samples,
            scale=self.sigma,
            alpha=self.alpha,
        )
