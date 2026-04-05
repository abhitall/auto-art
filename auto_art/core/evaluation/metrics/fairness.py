"""
Fairness evaluation under adversarial perturbation.

Measures the intersection of:
1. Adversarial robustness (input perturbations)
2. Privacy (membership inference risk)
3. Fairness (group fairness metrics under attack)

Key research findings:
- FD-MIA: Fairness interventions can paradoxically increase privacy risk
- DP-SGD: 2-5% accuracy drop for strong membership privacy
- Adversarial perturbations disproportionately affect minority groups

Metrics:
- Group accuracy disparity (clean vs adversarial)
- Equalized robustness (robustness gap between groups)
- Fairness-robustness Pareto frontier
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FairnessResult:
    """Result of fairness evaluation under adversarial perturbation."""
    clean_accuracy_by_group: Dict[str, float] = field(default_factory=dict)
    adversarial_accuracy_by_group: Dict[str, float] = field(default_factory=dict)
    robustness_by_group: Dict[str, float] = field(default_factory=dict)
    accuracy_disparity_clean: float = 0.0
    accuracy_disparity_adversarial: float = 0.0
    robustness_gap: float = 0.0
    equalized_robustness: bool = False
    demographic_parity_clean: float = 0.0
    demographic_parity_adversarial: float = 0.0
    fairness_robustness_tradeoff: Dict[str, Any] = field(default_factory=dict)


class FairnessEvaluator:
    """Evaluates fairness under adversarial perturbation.

    Measures whether adversarial attacks disproportionately affect
    certain demographic groups, and whether defenses maintain fairness.
    """

    def __init__(self, robustness_gap_threshold: float = 0.1):
        """
        Args:
            robustness_gap_threshold: Max allowed robustness gap between groups
                for equalized robustness to be considered satisfied.
        """
        self.robustness_gap_threshold = robustness_gap_threshold

    def evaluate(
        self,
        clean_predictions: np.ndarray,
        adversarial_predictions: np.ndarray,
        true_labels: np.ndarray,
        group_labels: np.ndarray,
    ) -> FairnessResult:
        """Evaluate fairness metrics on clean and adversarial predictions.

        Args:
            clean_predictions: Model predictions on clean data (N,)
            adversarial_predictions: Model predictions on adversarial data (N,)
            true_labels: Ground truth labels (N,)
            group_labels: Demographic group labels (N,) — integer or string

        Returns:
            FairnessResult with per-group and aggregate metrics
        """
        groups = np.unique(group_labels)
        result = FairnessResult()

        for group in groups:
            mask = group_labels == group
            group_name = str(group)

            # Per-group accuracy
            clean_acc = np.mean(clean_predictions[mask] == true_labels[mask])
            adv_acc = np.mean(adversarial_predictions[mask] == true_labels[mask])

            result.clean_accuracy_by_group[group_name] = float(clean_acc)
            result.adversarial_accuracy_by_group[group_name] = float(adv_acc)
            result.robustness_by_group[group_name] = float(clean_acc - adv_acc)

        # Accuracy disparity (max gap between groups)
        clean_accs = list(result.clean_accuracy_by_group.values())
        adv_accs = list(result.adversarial_accuracy_by_group.values())
        robustness_vals = list(result.robustness_by_group.values())

        if clean_accs:
            result.accuracy_disparity_clean = float(max(clean_accs) - min(clean_accs))
            result.accuracy_disparity_adversarial = float(max(adv_accs) - min(adv_accs))
            result.robustness_gap = float(max(robustness_vals) - min(robustness_vals))
            result.equalized_robustness = result.robustness_gap <= self.robustness_gap_threshold

            # Demographic parity (prediction rate disparity)
            result.demographic_parity_clean = self._demographic_parity(
                clean_predictions, group_labels, groups
            )
            result.demographic_parity_adversarial = self._demographic_parity(
                adversarial_predictions, group_labels, groups
            )

        # Tradeoff summary
        result.fairness_robustness_tradeoff = {
            "fairness_degrades_under_attack": (
                result.accuracy_disparity_adversarial > result.accuracy_disparity_clean * 1.1
            ),
            "worst_group_clean": min(result.clean_accuracy_by_group.items(), key=lambda x: x[1])
                if result.clean_accuracy_by_group else ("N/A", 0),
            "worst_group_adversarial": min(result.adversarial_accuracy_by_group.items(), key=lambda x: x[1])
                if result.adversarial_accuracy_by_group else ("N/A", 0),
            "recommendation": self._recommend(result),
        }

        return result

    @staticmethod
    def _demographic_parity(
        predictions: np.ndarray,
        group_labels: np.ndarray,
        groups: np.ndarray,
    ) -> float:
        """Compute demographic parity difference (max positive prediction rate gap)."""
        rates = []
        for group in groups:
            mask = group_labels == group
            if np.sum(mask) > 0:
                rate = np.mean(predictions[mask] == 1)  # positive prediction rate
                rates.append(float(rate))
        return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0

    @staticmethod
    def _recommend(result: FairnessResult) -> str:
        """Generate recommendation based on fairness-robustness analysis."""
        if result.robustness_gap > 0.2:
            return (
                "CRITICAL: Large robustness gap between groups. "
                "Adversarial attacks disproportionately affect certain demographics. "
                "Consider group-aware adversarial training or fairness-constrained defenses."
            )
        if result.accuracy_disparity_adversarial > result.accuracy_disparity_clean * 1.5:
            return (
                "WARNING: Fairness degrades significantly under attack. "
                "The model's fairness properties are not robust. "
                "Consider robustness-aware fairness constraints."
            )
        if result.equalized_robustness:
            return "GOOD: Robustness is approximately equalized across groups."
        return "MODERATE: Some robustness disparity detected. Monitor during deployment."
