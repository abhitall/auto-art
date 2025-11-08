"""
Metrics calculator for model evaluation.
"""

from typing import Any, Dict, List, Optional
from functools import lru_cache
import numpy as np
import logging
from art.metrics import (
    empirical_robustness,
    loss_sensitivity,
    clever_u,
    RobustnessVerificationTreeModelsCliqueMethod,
    wasserstein_distance # Added import
)
import sys # For printing warnings if needed, though commented out in prompt

# Configure logger for this module
logger = logging.getLogger(__name__)

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

        accuracy = float(np.mean(
            np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)
        ))

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

        metrics['empirical_robustness'] = self.calculate_empirical_robustness(
            classifier,
            data
        )

        metrics['loss_sensitivity'] = self._calculate_loss_sensitivity(
            classifier,
            data,
            labels
        )

        clever_scores = []
        for i in range(min(num_samples, len(data))):
            try:
                score = self.calculate_clever_score(
                    classifier,
                    data[i]
                )
                clever_scores.append(score)
            except (ValueError, RuntimeError, AttributeError) as e:
                # Log specific errors but continue with remaining samples
                logger.warning(f"Failed to calculate CLEVER score for sample {i}: {type(e).__name__}: {e}")
                continue
            except Exception as e:
                # Log unexpected errors but continue processing
                logger.error(f"Unexpected error calculating CLEVER score for sample {i}: {type(e).__name__}: {e}")
                continue

        if clever_scores:
            metrics['average_clever_score'] = float(np.mean(clever_scores))
        else:
            logger.warning("No CLEVER scores could be calculated for any samples")

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
        """Calculate tree verification metric for tree-based models."""
        try:
            verification = RobustnessVerificationTreeModelsCliqueMethod(classifier)
            return float(verification(data, labels))
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(f"Tree verification not applicable or failed: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in tree verification: {type(e).__name__}: {e}")
            return None

    def calculate_security_score(self,
                                 base_accuracy: float,
                                 attack_results: Dict[str, Dict[str, Any]],
                                 robustness_metrics: Dict[str, float]
                                ) -> float:
        score = 0.0
        weights = {'accuracy': 0.4, 'attack_defense': 0.4, 'certified_robustness': 0.2}
        score += weights['accuracy'] * base_accuracy * 100
        defense_scores = []
        attacks_evaluated_count = 0
        if attack_results:
            for _attack_name, res in attack_results.items():
                if res and 'success_rate' in res and isinstance(res['success_rate'], (float, int)):
                    attacks_evaluated_count += 1
                    defense_scores.append(1.0 - float(res['success_rate']))

        if attacks_evaluated_count > 0:
            avg_defense_score = np.mean(defense_scores) if defense_scores else 0.0
            score += weights['attack_defense'] * avg_defense_score * 100
        elif not attack_results:
             score += weights['attack_defense'] * 100

        num_robust_metrics = 0
        robust_score_sum = 0.0
        if 'empirical_robustness' in robustness_metrics and robustness_metrics['empirical_robustness'] is not None:
            robust_score_sum += robustness_metrics['empirical_robustness']
            num_robust_metrics += 1

        if num_robust_metrics > 0:
            avg_robust_contrib = robust_score_sum / num_robust_metrics
            score += weights['certified_robustness'] * avg_robust_contrib * 100
        elif not robustness_metrics :
             score += weights['certified_robustness'] * 50

        return max(0.0, min(100.0, score))

    def calculate_wasserstein_distance(self,
                                     data_batch_1: np.ndarray,
                                     data_batch_2: np.ndarray) -> Optional[float]:
        """
        Calculates the Wasserstein distance between two batches of data.
        Useful for comparing distributions, e.g., original vs. adversarial.
        Requires SciPy to be installed.

        Args:
            data_batch_1: First batch of data (e.g., original samples).
            data_batch_2: Second batch of data (e.g., adversarial samples).

        Returns:
            The Wasserstein distance as a float, or None if calculation fails (e.g., SciPy not found).
        """
        if not isinstance(data_batch_1, np.ndarray) or not isinstance(data_batch_2, np.ndarray):
            # print("Warning: Wasserstein distance requires numpy array inputs.", file=sys.stderr)
            return None

        if data_batch_1.shape[0] == 0 or data_batch_2.shape[0] == 0:
            # print("Warning: Wasserstein distance cannot be computed on empty batches.", file=sys.stderr)
            return None

        shape1 = data_batch_1.shape
        if len(shape1) > 2:
            data_batch_1_flat = data_batch_1.reshape(shape1[0], -1)
        elif len(shape1) == 1:
            data_batch_1_flat = data_batch_1.reshape(1, -1)
        else:
            data_batch_1_flat = data_batch_1

        shape2 = data_batch_2.shape
        if len(shape2) > 2:
            data_batch_2_flat = data_batch_2.reshape(shape2[0], -1)
        elif len(shape2) == 1:
            data_batch_2_flat = data_batch_2.reshape(1, -1)
        else:
            data_batch_2_flat = data_batch_2

        if data_batch_1_flat.shape[1] != data_batch_2_flat.shape[1] and data_batch_1_flat.size > 0 and data_batch_2_flat.size > 0 :
            # print(f"Warning: Feature dimensions for Wasserstein distance differ: {data_batch_1_flat.shape[1]} vs {data_batch_2_flat.shape[1]}", file=sys.stderr)
            pass # Let ART handle or raise error.

        try:
            distance = wasserstein_distance(data_batch_1_flat, data_batch_2_flat)
            return float(distance)
        except ImportError as e:
            # SciPy not installed or import error
            logger.warning(f"Cannot calculate Wasserstein distance: SciPy not available - {e}")
            return None
        except (ValueError, TypeError) as e:
            # Invalid data shapes or types
            logger.warning(f"Invalid data for Wasserstein distance calculation: {type(e).__name__}: {e}")
            return None
        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error calculating Wasserstein distance: {type(e).__name__}: {e}")
            return None
