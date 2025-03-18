"""
Tests for ART preprocessor defence wrappers.
"""

import numpy as np
from unittest.mock import MagicMock, patch

from auto_art.core.evaluation.defences.preprocessor import (
    SpatialSmoothingDefence,
    FeatureSqueezingDefence,
    JpegCompressionDefence,
    GaussianAugmentationDefence,
)


class TestSpatialSmoothingDefence:

    def test_initialization(self):
        defence = SpatialSmoothingDefence(window_size=5)
        assert defence.window_size == 5
        assert defence.name == "SpatialSmoothing"

    def test_get_params(self):
        defence = SpatialSmoothingDefence(window_size=3, channels_first=True)
        params = defence.get_params()
        assert params["window_size"] == 3
        assert params["channels_first"] is True

    @patch(
        'auto_art.core.evaluation.defences.preprocessor.ART_PREPROCESSORS_AVAILABLE',
        False,
    )
    def test_apply_raises_without_art(self):
        defence = SpatialSmoothingDefence()
        try:
            defence.apply(MagicMock())
            assert False, "Should have raised ImportError"
        except ImportError:
            pass

    @patch(
        'auto_art.core.evaluation.defences.preprocessor.ART_PREPROCESSORS_AVAILABLE',
        False,
    )
    def test_transform_raises_without_art(self):
        defence = SpatialSmoothingDefence()
        try:
            defence.transform(np.zeros((10, 3)))
            assert False, "Should have raised ImportError"
        except ImportError:
            pass


class TestFeatureSqueezingDefence:

    def test_initialization(self):
        defence = FeatureSqueezingDefence(bit_depth=8)
        assert defence.bit_depth == 8
        assert defence.name == "FeatureSqueezing"

    def test_get_params(self):
        defence = FeatureSqueezingDefence(bit_depth=4)
        params = defence.get_params()
        assert params["bit_depth"] == 4
        assert params["clip_values"] == (0.0, 1.0)


class TestJpegCompressionDefence:

    def test_initialization(self):
        defence = JpegCompressionDefence(quality=75)
        assert defence.quality == 75
        assert defence.name == "JpegCompression"

    def test_get_params(self):
        defence = JpegCompressionDefence(quality=50, channels_first=True)
        params = defence.get_params()
        assert params["quality"] == 50
        assert params["channels_first"] is True


class TestGaussianAugmentationDefence:

    def test_initialization(self):
        defence = GaussianAugmentationDefence(sigma=0.2)
        assert defence.sigma == 0.2
        assert defence.name == "GaussianAugmentation"

    def test_get_params(self):
        defence = GaussianAugmentationDefence(sigma=0.5, ratio=0.8)
        params = defence.get_params()
        assert params["sigma"] == 0.5
        assert params["ratio"] == 0.8
        assert params["augmentation"] is True
