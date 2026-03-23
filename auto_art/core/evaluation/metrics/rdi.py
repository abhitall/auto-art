"""
Robustness Difference Index (RDI) — attack-independent robustness metric.

Based on Song et al., UAI 2025: "RDI: An adversarial robustness evaluation
metric for deep neural networks based on model statistical features."

RDI measures robustness by analyzing the ratio of inter-class to intra-class
feature distances at the model's decision boundary. Unlike attack-based
metrics, RDI requires no adversarial example generation, achieving ~30x
speedup over PGD-based evaluation with strong ASR correlation.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RDIReport:
    """RDI computation results."""
    rdi_score: float = 0.0
    intra_class_distance: float = 0.0
    inter_class_distance: float = 0.0
    per_class_rdi: dict[int, float] = field(default_factory=dict)
    num_samples_used: int = 0
    computation_time: float = 0.0
    interpretation: str = ""


class RDICalculator:
    """Computes the Robustness Difference Index for a classifier.
    
    RDI = (inter_class_distance - intra_class_distance) / (inter_class_distance + epsilon)
    
    Higher RDI indicates wider decision boundary margins and thus
    greater adversarial robustness. Values range from ~0 (fragile) to ~1 (robust).
    """

    def __init__(
        self,
        num_samples: int = 500,
        feature_layer: Optional[str] = None,
        epsilon: float = 1e-8,
    ):
        self.num_samples = num_samples
        self.feature_layer = feature_layer
        self.epsilon = epsilon

    def compute(
        self,
        classifier: Any,
        x: np.ndarray,
        y: np.ndarray,
    ) -> RDIReport:
        """Compute RDI score for the given classifier and data.
        
        Args:
            classifier: ART classifier instance with predict method.
            x: Input data samples.
            y: Labels (one-hot encoded).
        
        Returns:
            RDIReport with computed RDI score and breakdown.
        """
        import time
        start = time.time()

        n = min(len(x), self.num_samples)
        indices = np.random.choice(len(x), n, replace=False) if len(x) > n else np.arange(len(x))
        x_sub = x[indices]
        y_sub = y[indices]

        features = self._extract_features(classifier, x_sub)
        labels = np.argmax(y_sub, axis=1) if y_sub.ndim > 1 else y_sub

        unique_classes = np.unique(labels)
        class_features: dict[int, np.ndarray] = {}
        for c in unique_classes:
            mask = labels == c
            class_features[int(c)] = features[mask]

        intra_distances = []
        per_class_rdi = {}
        for _c, feats in class_features.items():
            if len(feats) < 2:
                continue
            centroid = np.mean(feats, axis=0)
            dists = np.sqrt(np.sum((feats - centroid) ** 2, axis=1))
            intra_distances.append(float(np.mean(dists)))

        inter_distances = []
        class_list = list(class_features.keys())
        for i in range(len(class_list)):
            for j in range(i + 1, len(class_list)):
                ci, cj = class_list[i], class_list[j]
                centroid_i = np.mean(class_features[ci], axis=0)
                centroid_j = np.mean(class_features[cj], axis=0)
                dist = float(np.sqrt(np.sum((centroid_i - centroid_j) ** 2)))
                inter_distances.append(dist)

        avg_intra = float(np.mean(intra_distances)) if intra_distances else 0.0
        avg_inter = float(np.mean(inter_distances)) if inter_distances else 0.0

        rdi = (avg_inter - avg_intra) / (avg_inter + self.epsilon)
        rdi = max(0.0, min(1.0, rdi))

        for c, feats in class_features.items():
            if len(feats) < 2:
                per_class_rdi[c] = 0.0
                continue
            centroid_c = np.mean(feats, axis=0)
            intra_c = float(np.mean(np.sqrt(np.sum((feats - centroid_c) ** 2, axis=1))))
            other_centroids = [
                np.mean(class_features[oc], axis=0) for oc in class_features if oc != c and len(class_features[oc]) > 0
            ]
            if other_centroids:
                inter_c = float(np.mean([
                    np.sqrt(np.sum((centroid_c - oc) ** 2)) for oc in other_centroids
                ]))
                per_class_rdi[c] = max(0.0, min(1.0, (inter_c - intra_c) / (inter_c + self.epsilon)))
            else:
                per_class_rdi[c] = 0.0

        elapsed = time.time() - start

        if rdi > 0.7:
            interp = "High robustness — wide decision boundary margins."
        elif rdi > 0.4:
            interp = "Moderate robustness — some vulnerability to adversarial perturbation."
        elif rdi > 0.2:
            interp = "Low robustness — narrow decision boundary margins, likely vulnerable."
        else:
            interp = "Very low robustness — decision boundaries are extremely tight."

        return RDIReport(
            rdi_score=rdi,
            intra_class_distance=avg_intra,
            inter_class_distance=avg_inter,
            per_class_rdi=per_class_rdi,
            num_samples_used=n,
            computation_time=elapsed,
            interpretation=interp,
        )

    def _extract_features(self, classifier: Any, x: np.ndarray) -> np.ndarray:
        """Extract feature representations from the classifier.
        
        Uses the last hidden layer or prediction logits as features.
        """
        if self.feature_layer and hasattr(classifier, 'get_activations'):
            try:
                return classifier.get_activations(x, layer=self.feature_layer)
            except Exception:
                pass

        predictions = classifier.predict(x)
        return predictions.astype(np.float64)
