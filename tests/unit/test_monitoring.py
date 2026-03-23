"""Tests for production monitoring module."""
import pytest
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch

from auto_art.core.monitoring import (
    RobustnessDriftMonitor, DriftReport,
    ModelSupplyChainScanner, SupplyChainReport,
)


class TestDriftReport:
    def test_defaults(self):
        r = DriftReport()
        assert r.drift_detected is False
        assert r.rdi_current == 0.0


class TestRobustnessDriftMonitor:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.baseline_path = os.path.join(self.tmpdir, "baseline.json")
        self.monitor = RobustnessDriftMonitor(
            baseline_path=self.baseline_path,
            drift_threshold=0.1,
        )

    def test_set_baseline(self):
        self.monitor.set_baseline(0.75, 0.95)
        assert os.path.exists(self.baseline_path)

    @patch("auto_art.core.evaluation.metrics.rdi.RDICalculator.compute")
    def test_check_drift_no_baseline(self, mock_compute):
        mock_compute.return_value = MagicMock(rdi_score=0.6)

        classifier = MagicMock()
        classifier.predict.return_value = np.eye(2)[[0, 1]]

        x = np.random.rand(2, 10).astype(np.float32)
        y = np.eye(2)

        report = self.monitor.check_drift(classifier, x, y)
        assert isinstance(report, DriftReport)
        assert "baseline" in report.recommendation.lower()

    @patch("auto_art.core.evaluation.metrics.rdi.RDICalculator.compute")
    def test_check_drift_with_baseline(self, mock_compute):
        self.monitor.set_baseline(0.8, 0.95)
        mock_compute.return_value = MagicMock(rdi_score=0.5)

        classifier = MagicMock()
        classifier.predict.return_value = np.eye(2)[[0, 1]]

        x = np.random.rand(2, 10).astype(np.float32)
        y = np.eye(2)

        report = self.monitor.check_drift(classifier, x, y)
        assert report.drift_detected is True
        assert report.rdi_delta > 0


class TestSupplyChainReport:
    def test_defaults(self):
        r = SupplyChainReport()
        assert r.is_safe is True


class TestModelSupplyChainScanner:
    def setup_method(self):
        self.scanner = ModelSupplyChainScanner()

    def test_scan_nonexistent(self):
        report = self.scanner.scan("/nonexistent/model.pt")
        assert report.is_safe is False

    def test_scan_pickle_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"fake pickle data")
            path = f.name
        try:
            report = self.scanner.scan(path)
            assert report.risk_level == "high"
            assert report.is_safe is False
            assert len(report.backdoor_indicators) > 0
        finally:
            os.unlink(path)

    def test_scan_safetensors_file(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            f.write(b"fake safetensor data")
            path = f.name
        try:
            report = self.scanner.scan(path)
            assert report.risk_level == "low"
            assert report.is_safe is True
            assert report.provenance_verified is True
        finally:
            os.unlink(path)

    def test_scan_pytorch_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake model")
            path = f.name
        try:
            report = self.scanner.scan(path)
            assert report.risk_level == "medium"
        finally:
            os.unlink(path)
