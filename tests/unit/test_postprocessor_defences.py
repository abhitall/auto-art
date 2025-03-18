"""
Tests for ART postprocessor defence wrappers.
"""

from unittest.mock import MagicMock, patch

from auto_art.core.evaluation.defences.postprocessor import (
    ReverseSigmoidDefence,
    HighConfidenceDefence,
    GaussianNoiseDefence,
    ClassLabelsDefence,
)


class TestReverseSigmoidDefence:

    def test_initialization(self):
        defence = ReverseSigmoidDefence(beta=2.0, gamma=0.2)
        assert defence.beta == 2.0
        assert defence.gamma == 0.2
        assert defence.name == "ReverseSigmoid"

    def test_get_params(self):
        defence = ReverseSigmoidDefence(beta=1.5)
        params = defence.get_params()
        assert params["beta"] == 1.5
        assert params["gamma"] == 0.1

    @patch(
        'auto_art.core.evaluation.defences.postprocessor.ART_POSTPROCESSORS_AVAILABLE',
        False,
    )
    def test_apply_raises_without_art(self):
        defence = ReverseSigmoidDefence()
        try:
            defence.apply(MagicMock())
            assert False, "Should have raised ImportError"
        except ImportError:
            pass


class TestHighConfidenceDefence:

    def test_initialization(self):
        defence = HighConfidenceDefence(cutoff=0.5)
        assert defence.cutoff == 0.5
        assert defence.name == "HighConfidence"

    def test_get_params(self):
        defence = HighConfidenceDefence(cutoff=0.3)
        params = defence.get_params()
        assert params["cutoff"] == 0.3


class TestGaussianNoiseDefence:

    def test_initialization(self):
        defence = GaussianNoiseDefence(scale=0.2)
        assert defence.scale == 0.2
        assert defence.name == "GaussianNoise"

    def test_get_params(self):
        defence = GaussianNoiseDefence(scale=0.5)
        params = defence.get_params()
        assert params["scale"] == 0.5


class TestClassLabelsDefence:

    def test_initialization(self):
        defence = ClassLabelsDefence()
        assert defence.name == "ClassLabels"

    def test_get_params_empty(self):
        defence = ClassLabelsDefence()
        assert defence.get_params() == {}
