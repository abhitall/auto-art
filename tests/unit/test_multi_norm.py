"""Tests for multi-norm evaluation engine."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from auto_art.core.evaluation.metrics.multi_norm import (
    MultiNormEvaluator, MultiNormReport, NormEvalResult,
)


class TestNormEvalResult:
    def test_defaults(self):
        r = NormEvalResult(norm="linf", epsilon=0.3, attack_name="pgd")
        assert r.success_rate == 0.0
        assert r.error is None


class TestMultiNormEvaluator:
    def setup_method(self):
        self.evaluator = MultiNormEvaluator(norms=["linf", "l2"])

    def test_init_defaults(self):
        e = MultiNormEvaluator()
        assert "linf" in e.norms
        assert "l2" in e.norms

    def test_init_custom_norms(self):
        e = MultiNormEvaluator(norms=["l1"])
        assert e.norms == ["l1"]

    def test_evaluate_with_fallback(self):
        with patch("art.attacks.evasion.ProjectedGradientDescent") as mock_pgd_class:
            classifier = MagicMock()
            preds_clean = np.eye(3)[[0, 1, 2, 0, 1]]
            preds_adv = np.eye(3)[[1, 2, 0, 1, 2]]
            classifier.predict.side_effect = [preds_clean, preds_adv] * 20

            mock_pgd = MagicMock()
            mock_pgd.generate.return_value = np.random.rand(5, 3, 32, 32).astype(np.float32)
            mock_pgd_class.return_value = mock_pgd

            x = np.random.rand(5, 3, 32, 32).astype(np.float32)
            y = np.eye(3)[[0, 1, 2, 0, 1]]

            evaluator = MultiNormEvaluator(norms=["linf"], epsilons={"linf": [0.03]})
            report = evaluator.evaluate(classifier, x, y)

            assert isinstance(report, MultiNormReport)
            assert report.num_configurations > 0
            assert report.total_duration > 0

    def test_report_tracks_errors_gracefully(self):
        """When all attacks fail due to mock classifier, results contain errors."""
        evaluator = MultiNormEvaluator(norms=["linf"], epsilons={"linf": [0.1]})
        classifier = MagicMock()
        x = np.random.rand(5, 10).astype(np.float32)
        y = np.eye(2)[[0, 1, 0, 1, 0]]
        report = evaluator.evaluate(classifier, x, y)
        assert report.num_configurations > 0
        assert all(r.error is not None for r in report.results)
