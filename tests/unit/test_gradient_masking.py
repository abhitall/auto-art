"""Tests for GradientMaskingDetector."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from auto_art.core.gradient_masking import GradientMaskingDetector, MaskingReport


class TestMaskingReport:
    def test_defaults(self):
        report = MaskingReport()
        assert report.is_masked is False
        assert report.confidence == 0.0
        assert report.fosc_score == 0.0
        assert report.recommendation == ""

    def test_custom_values(self):
        report = MaskingReport(is_masked=True, confidence=0.8, fosc_score=0.5)
        assert report.is_masked is True
        assert report.confidence == 0.8


class TestGradientMaskingDetector:
    def setup_method(self):
        self.detector = GradientMaskingDetector(
            fosc_threshold=0.1, discrepancy_threshold=0.15,
        )

    def test_init_defaults(self):
        d = GradientMaskingDetector()
        assert d.fosc_threshold == 0.1
        assert d.discrepancy_threshold == 0.15
        assert d.noise_sigma == 0.01

    def test_detect_no_masking(self):
        classifier = MagicMock()
        classifier.loss_gradient.return_value = np.random.randn(10, 3, 32, 32).astype(np.float32)
        x = np.random.rand(10, 3, 32, 32).astype(np.float32)
        y = np.eye(10).astype(np.float32)

        wb_results = {"success_rate": 0.4}
        bb_results = {"success_rate": 0.35}

        report = self.detector.detect(classifier, x, y, wb_results, bb_results)
        assert isinstance(report, MaskingReport)
        assert report.discrepancy < 0.0

    def test_detect_with_masking(self):
        classifier = MagicMock()
        classifier.loss_gradient.return_value = np.zeros((10, 3, 32, 32), dtype=np.float32)

        x = np.random.rand(10, 3, 32, 32).astype(np.float32)
        y = np.eye(10).astype(np.float32)

        wb_results = {"success_rate": 0.05}
        bb_results = {"success_rate": 0.6}

        report = self.detector.detect(classifier, x, y, wb_results, bb_results)
        assert report.discrepancy > 0.15
        assert report.wb_success_rate == 0.05
        assert report.bb_success_rate == 0.6

    def test_detect_from_attack_results_no_masking(self):
        attack_results = [
            {"attack": "pgd", "success_rate": 0.5, "status": "completed"},
            {"attack": "fgsm", "success_rate": 0.4, "status": "completed"},
            {"attack": "square_attack", "success_rate": 0.3, "status": "completed"},
        ]
        report = self.detector.detect_from_attack_results(attack_results)
        assert report.is_masked is False

    def test_detect_from_attack_results_with_masking(self):
        attack_results = [
            {"attack": "pgd", "success_rate": 0.02, "status": "completed"},
            {"attack": "fgsm", "success_rate": 0.01, "status": "completed"},
            {"attack": "square_attack", "success_rate": 0.6, "status": "completed"},
            {"attack": "hopskipjump", "success_rate": 0.5, "status": "completed"},
        ]
        report = self.detector.detect_from_attack_results(attack_results)
        assert report.is_masked is True
        assert "black-box" in report.recommendation.lower() or "gradient" in report.recommendation.lower()

    def test_detect_from_empty_results(self):
        report = self.detector.detect_from_attack_results([])
        assert report.is_masked is False

    def test_detect_skips_errored_attacks(self):
        attack_results = [
            {"attack": "pgd", "success_rate": 0.5, "status": "error"},
            {"attack": "square_attack", "success_rate": 0.8, "status": "completed"},
        ]
        report = self.detector.detect_from_attack_results(attack_results)
        assert report.bb_success_rate == 0.8
        assert report.wb_success_rate == 0.0
