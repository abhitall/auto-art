"""
Metrics calculator for model evaluation.
"""

from typing import Any, Dict, List, Optional
from functools import lru_cache
import numpy as np
from art.metrics import (
    empirical_robustness,
    loss_sensitivity,
    clever_u,
    RobustnessVerificationTreeModelsCliqueMethod,
)

# Constants
CACHE_SIZE = 128

class MetricsCalculator:
    """Calculator for various robustness metrics."""
    
    def __init__(self, cache_size: int = CACHE_SIZE):
        self.cache_size = cache_size
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize cached methods."""
        self.calculate_empirical_robustness = lru_cache(maxsize=self.cache_size)(
            self._calculate_empirical_robustness
        )
        self.calculate_clever_score = lru_cache(maxsize=self.cache_size)(
            self._calculate_clever_score
        )

    def calculate_basic_metrics(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate basic model metrics."""
        predictions = classifier.predict(data)
        
        # Calculate accuracy
        accuracy = float(np.mean(
            np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)
        ))
        
        # Calculate confidence
        confidence = float(np.mean(np.max(predictions, axis=1)))
        
        return {
            'accuracy': accuracy,
            'average_confidence': confidence
        }

    def calculate_robustness_metrics(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray,
        num_samples: int = 5
    ) -> Dict[str, float]:
        """Calculate advanced robustness metrics."""
        metrics = {}
        
        # Empirical robustness
        metrics['empirical_robustness'] = self.calculate_empirical_robustness(
            classifier,
            data
        )
        
        # Loss sensitivity
        metrics['loss_sensitivity'] = self._calculate_loss_sensitivity(
            classifier,
            data,
            labels
        )
        
        # CLEVER score for subset of samples
        clever_scores = []
        for i in range(min(num_samples, len(data))):
            try:
                score = self.calculate_clever_score(
                    classifier,
                    data[i]
                )
                clever_scores.append(score)
            except Exception:
                continue
        
        if clever_scores:
            metrics['average_clever_score'] = float(np.mean(clever_scores))
        
        # Tree model verification if applicable
        if hasattr(classifier, 'model') and hasattr(classifier.model, 'tree_'):
            metrics['tree_verification'] = self._calculate_tree_verification(
                classifier,
                data,
                labels
            )
        
        return metrics

    def _calculate_empirical_robustness(
        self,
        classifier: Any,
        data: np.ndarray,
        eps: float = 0.3
    ) -> float:
        """Calculate empirical robustness."""
        return float(empirical_robustness(
            classifier=classifier,
            x=data,
            attack_name="FastGradientMethod",
            attack_params={"eps": eps}
        ))

    def _calculate_loss_sensitivity(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate loss sensitivity."""
        return float(loss_sensitivity(classifier, data, labels))

    def _calculate_clever_score(
        self,
        classifier: Any,
        sample: np.ndarray,
        nb_batches: int = 10,
        batch_size: int = 100,
        radius: float = 0.3,
        norm: int = 2
    ) -> float:
        """Calculate CLEVER score for a sample."""
        return float(clever_u(
            classifier=classifier,
            x=sample,
            nb_batches=nb_batches,
            batch_size=batch_size,
            radius=radius,
            norm=norm
        ))

    def _calculate_tree_verification(
        self,
        classifier: Any,
        data: np.ndarray,
        labels: np.ndarray
    ) -> Optional[float]:
        """Calculate verification score for tree models."""
        try:
            verification = RobustnessVerificationTreeModelsCliqueMethod(classifier)
            return float(verification(data, labels))
        except Exception:
            return None 