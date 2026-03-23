"""Tests for the RDI (Robustness Difference Index) calculator."""
import pytest
import numpy as np
from unittest.mock import MagicMock

from auto_art.core.evaluation.metrics.rdi import RDICalculator, RDIReport


class TestRDIReport:
    def test_defaults(self):
        r = RDIReport()
        assert r.rdi_score == 0.0
        assert r.per_class_rdi == {}
        assert r.interpretation == ""


class TestRDICalculator:
    def setup_method(self):
        self.calc = RDICalculator(num_samples=50)

    def test_init(self):
        c = RDICalculator(num_samples=100, feature_layer="fc1")
        assert c.num_samples == 100
        assert c.feature_layer == "fc1"

    def test_compute_well_separated(self):
        classifier = MagicMock()
        preds = np.zeros((20, 2))
        preds[:10, 0] = 0.9
        preds[:10, 1] = 0.1
        preds[10:, 0] = 0.1
        preds[10:, 1] = 0.9
        classifier.predict.return_value = preds

        x = np.random.rand(20, 10).astype(np.float32)
        y = np.eye(2)[np.array([0]*10 + [1]*10)]

        report = self.calc.compute(classifier, x, y)
        assert isinstance(report, RDIReport)
        assert report.rdi_score > 0.0
        assert report.computation_time > 0.0
        assert report.num_samples_used == 20

    def test_compute_overlapping(self):
        classifier = MagicMock()
        preds = np.random.rand(20, 2).astype(np.float32)
        classifier.predict.return_value = preds

        x = np.random.rand(20, 10).astype(np.float32)
        y = np.eye(2)[np.random.randint(0, 2, 20)]

        report = self.calc.compute(classifier, x, y)
        assert isinstance(report, RDIReport)
        assert 0.0 <= report.rdi_score <= 1.0

    def test_compute_single_class(self):
        classifier = MagicMock()
        preds = np.ones((10, 2)) * 0.5
        classifier.predict.return_value = preds

        x = np.random.rand(10, 5).astype(np.float32)
        y = np.eye(2)[np.zeros(10, dtype=int)]

        report = self.calc.compute(classifier, x, y)
        assert isinstance(report, RDIReport)

    def test_interpretation_levels(self):
        classifier = MagicMock()

        preds = np.zeros((20, 2))
        preds[:10, 0] = 10.0
        preds[:10, 1] = 0.0
        preds[10:, 0] = 0.0
        preds[10:, 1] = 10.0
        classifier.predict.return_value = preds

        x = np.random.rand(20, 10).astype(np.float32)
        y = np.eye(2)[np.array([0]*10 + [1]*10)]

        report = self.calc.compute(classifier, x, y)
        assert len(report.interpretation) > 0
