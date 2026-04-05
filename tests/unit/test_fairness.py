"""Tests for fairness evaluation under adversarial perturbation."""

import numpy as np
import pytest
from auto_art.core.evaluation.metrics.fairness import FairnessEvaluator, FairnessResult


class TestFairnessEvaluator:
    def setup_method(self):
        self.evaluator = FairnessEvaluator(robustness_gap_threshold=0.1)
        np.random.seed(42)

    def test_equal_robustness(self):
        """Both groups equally affected by attack — equalized robustness."""
        n = 200
        true_labels = np.random.randint(0, 2, n)
        # Both groups have similar accuracy
        clean_preds = true_labels.copy()
        clean_preds[np.random.choice(n, 20, replace=False)] = 1 - clean_preds[np.random.choice(n, 20, replace=False)]

        adv_preds = true_labels.copy()
        adv_preds[np.random.choice(n, 50, replace=False)] = 1 - adv_preds[np.random.choice(n, 50, replace=False)]

        groups = np.array([0] * 100 + [1] * 100)

        result = self.evaluator.evaluate(clean_preds, adv_preds, true_labels, groups)

        assert isinstance(result, FairnessResult)
        assert len(result.clean_accuracy_by_group) == 2
        assert len(result.adversarial_accuracy_by_group) == 2
        assert isinstance(result.robustness_gap, float)

    def test_unequal_robustness(self):
        """One group much more affected by attacks."""
        n = 200
        true_labels = np.ones(n, dtype=int)
        groups = np.array([0] * 100 + [1] * 100)

        # Group 0: high clean accuracy, moderate adversarial accuracy
        clean_preds = np.ones(n, dtype=int)
        adv_preds = np.ones(n, dtype=int)

        # Group 1 takes much more damage from attacks
        adv_preds[100:] = 0  # All adversarial examples for group 1 are wrong

        result = self.evaluator.evaluate(clean_preds, adv_preds, true_labels, groups)

        assert result.robustness_gap > 0.5  # Large gap
        assert result.equalized_robustness is False
        assert "CRITICAL" in result.fairness_robustness_tradeoff["recommendation"]

    def test_result_fields(self):
        n = 100
        true_labels = np.random.randint(0, 2, n)
        clean_preds = true_labels.copy()
        adv_preds = true_labels.copy()
        groups = np.array([0] * 50 + [1] * 50)

        result = self.evaluator.evaluate(clean_preds, adv_preds, true_labels, groups)

        assert "0" in result.clean_accuracy_by_group
        assert "1" in result.clean_accuracy_by_group
        assert result.accuracy_disparity_clean >= 0
        assert result.demographic_parity_clean >= 0
