"""
Tests for poisoning detection defence wrappers.
"""

import numpy as np
from unittest.mock import MagicMock, patch

from auto_art.core.evaluation.defences.detector import (
    ActivationDefenceWrapper,
    SpectralSignatureDefenceWrapper,
    PoisonDetectionReport,
)


class TestActivationDefenceWrapper:

    def test_initialization(self):
        defence = ActivationDefenceWrapper(nb_clusters=3)
        assert defence.nb_clusters == 3
        assert defence.name == "ActivationDefence"

    def test_get_params(self):
        defence = ActivationDefenceWrapper(
            nb_clusters=4, clustering_method="KMeans",
            nb_dims=20, reduce="PCA",
        )
        params = defence.get_params()
        assert params["nb_clusters"] == 4
        assert params["clustering_method"] == "KMeans"
        assert params["nb_dims"] == 20

    def test_apply_without_data_returns_estimator(self):
        defence = ActivationDefenceWrapper()
        estimator = MagicMock()
        result = defence.apply(estimator)
        assert result is estimator

    @patch(
        'auto_art.core.evaluation.defences.detector.ART_DETECTORS_AVAILABLE',
        False,
    )
    def test_detect_poison_raises_without_art(self):
        defence = ActivationDefenceWrapper()
        try:
            defence.detect_poison(
                MagicMock(),
                np.zeros((10, 5)),
                np.zeros(10),
            )
            assert False, "Should have raised ImportError"
        except ImportError:
            pass


class TestSpectralSignatureDefenceWrapper:

    def test_initialization(self):
        defence = SpectralSignatureDefenceWrapper(expected_pp_poison=0.2)
        assert defence.expected_pp_poison == 0.2
        assert defence.name == "SpectralSignatureDefence"

    def test_get_params(self):
        defence = SpectralSignatureDefenceWrapper(
            expected_pp_poison=0.15, batch_size=64,
        )
        params = defence.get_params()
        assert params["expected_pp_poison"] == 0.15
        assert params["batch_size"] == 64

    def test_apply_without_data_returns_estimator(self):
        defence = SpectralSignatureDefenceWrapper()
        estimator = MagicMock()
        result = defence.apply(estimator)
        assert result is estimator


class TestPoisonDetectionReport:

    def test_report_creation(self):
        report = PoisonDetectionReport(
            total_samples=100,
            detected_poison=5,
            clean_samples=95,
            detection_rate=0.05,
        )
        assert report.total_samples == 100
        assert report.detected_poison == 5
        assert report.detection_rate == 0.05

    def test_report_with_mask(self):
        mask = np.array([True, True, False, True, False])
        report = PoisonDetectionReport(
            total_samples=5,
            detected_poison=2,
            clean_samples=3,
            detection_rate=0.4,
            is_clean_mask=mask,
        )
        assert np.sum(report.is_clean_mask) == 3
