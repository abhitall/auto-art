"""
Tests for certification metrics and Randomized Smoothing.
"""

import numpy as np
from unittest.mock import MagicMock, patch

from auto_art.core.evaluation.metrics.certification import (
    compute_great_score,
    RandomizedSmoothingCertifier,
)


class TestGreatScore:

    @patch(
        'auto_art.core.evaluation.metrics.certification.GREAT_SCORE_AVAILABLE',
        False,
    )
    def test_returns_none_when_unavailable(self):
        classifier = MagicMock()
        x = np.random.rand(10, 3, 32, 32).astype(np.float32)
        y = np.eye(10)
        result = compute_great_score(classifier, x, y)
        assert result is None

    def test_calls_art_great_score(self):
        import auto_art.core.evaluation.metrics.certification as cert_mod
        mock_fn = MagicMock(return_value=0.85)
        original_available = cert_mod.GREAT_SCORE_AVAILABLE
        original_fn = getattr(cert_mod, 'art_great_score', None)
        try:
            cert_mod.GREAT_SCORE_AVAILABLE = True
            cert_mod.art_great_score = mock_fn
            classifier = MagicMock()
            x = np.random.rand(10, 3, 32, 32).astype(np.float32)
            y = np.eye(10)
            result = compute_great_score(classifier, x, y, nb_samples=50)
            assert result == 0.85
            mock_fn.assert_called_once()
        finally:
            cert_mod.GREAT_SCORE_AVAILABLE = original_available
            if original_fn is not None:
                cert_mod.art_great_score = original_fn
            elif hasattr(cert_mod, 'art_great_score'):
                delattr(cert_mod, 'art_great_score')

    def test_handles_exception(self):
        import auto_art.core.evaluation.metrics.certification as cert_mod
        mock_fn = MagicMock(side_effect=RuntimeError("compute failed"))
        original_available = cert_mod.GREAT_SCORE_AVAILABLE
        original_fn = getattr(cert_mod, 'art_great_score', None)
        try:
            cert_mod.GREAT_SCORE_AVAILABLE = True
            cert_mod.art_great_score = mock_fn
            result = compute_great_score(
                MagicMock(), np.zeros((5, 10)), np.zeros(5),
            )
            assert result is None
        finally:
            cert_mod.GREAT_SCORE_AVAILABLE = original_available
            if original_fn is not None:
                cert_mod.art_great_score = original_fn
            elif hasattr(cert_mod, 'art_great_score'):
                delattr(cert_mod, 'art_great_score')


class TestRandomizedSmoothingCertifier:

    def test_initialization(self):
        certifier = RandomizedSmoothingCertifier(sigma=0.5, nb_samples=200)
        assert certifier.sigma == 0.5
        assert certifier.nb_samples == 200

    @patch(
        'auto_art.core.evaluation.metrics.certification.CERT_AVAILABLE',
        False,
    )
    def test_certify_returns_error_when_unavailable(self):
        certifier = RandomizedSmoothingCertifier()
        smoothed = MagicMock()
        x = np.random.rand(5, 10).astype(np.float32)
        result = certifier.certify(smoothed, x)
        assert "error" in result

    @patch(
        'auto_art.core.evaluation.metrics.certification.CERT_AVAILABLE',
        True,
    )
    def test_certify_with_mock_classifier(self):
        certifier = RandomizedSmoothingCertifier(sigma=0.25, nb_samples=10)
        smoothed = MagicMock()
        smoothed.predict.return_value = np.eye(5)
        smoothed.certify.return_value = (0, 0.5)

        x = np.random.rand(5, 10).astype(np.float32)
        y = np.eye(5)
        result = certifier.certify(smoothed, x, y)

        assert "certified_radii" in result
        assert result["total_samples"] == 5
        assert "certified_accuracy" in result
        assert result["mean_certified_radius"] >= 0.0

    @patch(
        'auto_art.core.evaluation.metrics.certification.CERT_AVAILABLE',
        True,
    )
    def test_certify_handles_failed_samples(self):
        certifier = RandomizedSmoothingCertifier()
        smoothed = MagicMock()
        smoothed.predict.return_value = np.eye(3)
        smoothed.certify.side_effect = RuntimeError("cert failed")

        x = np.random.rand(3, 10).astype(np.float32)
        result = certifier.certify(smoothed, x)

        assert result["total_samples"] == 3
        assert all(r == 0.0 for r in result["certified_radii"])
