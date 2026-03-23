"""Tests for privacy metrics calculator."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from auto_art.core.evaluation.metrics.privacy import (
    PrivacyMetricsCalculator, PrivacyReport,
)


class TestPrivacyReport:
    def test_defaults(self):
        r = PrivacyReport()
        assert r.pdtp_mean == 0.0
        assert r.shapr_mean == 0.0
        assert r.high_risk_fraction == 0.0


class TestPrivacyMetricsCalculator:
    def setup_method(self):
        self.calc = PrivacyMetricsCalculator(risk_threshold=0.7)

    def test_compute_pdtp_success(self):
        import art.metrics as art_metrics
        original = getattr(art_metrics, 'pdtp', None)
        try:
            art_metrics.pdtp = MagicMock(return_value=np.array([0.1, 0.5, 0.8, 0.3]))
            classifier = MagicMock()
            x = np.random.rand(4, 10)
            y = np.eye(2)[[0, 1, 0, 1]]
            result = self.calc.compute_pdtp(classifier, x, y)
            assert "pdtp_mean" in result
            assert result["high_risk_count"] == 1
        finally:
            if original is not None:
                art_metrics.pdtp = original
            elif hasattr(art_metrics, 'pdtp'):
                delattr(art_metrics, 'pdtp')

    def test_compute_pdtp_handles_exception(self):
        classifier = MagicMock()
        result = self.calc.compute_pdtp(classifier, np.zeros((2, 3)), np.zeros((2, 2)))
        assert isinstance(result, dict)

    def test_compute_shapr_success(self):
        with patch("art.metrics.SHAPr", return_value=np.array([0.2, 0.8, 0.5, 0.9])):
            classifier = MagicMock()
            x_train = np.random.rand(4, 10)
            y_train = np.eye(2)[[0, 1, 0, 1]]
            x_test = np.random.rand(4, 10)
            y_test = np.eye(2)[[0, 1, 0, 1]]
            result = self.calc.compute_shapr(classifier, x_train, y_train, x_test, y_test)
            assert "shapr_mean" in result
            assert result["high_risk_fraction"] == 0.5

    def test_compute_all(self):
        classifier = MagicMock()
        x = np.random.rand(4, 10)
        y = np.eye(2)[[0, 1, 0, 1]]
        report = self.calc.compute_all(classifier, x, y, x, y)
        assert isinstance(report, PrivacyReport)
