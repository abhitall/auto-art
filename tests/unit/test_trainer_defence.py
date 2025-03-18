"""
Tests for adversarial training defence wrapper.
"""

from unittest.mock import MagicMock, patch

from auto_art.core.evaluation.defences.trainer import (
    AdversarialTrainingPGDDefence,
)


class TestAdversarialTrainingPGDDefence:

    def test_initialization(self):
        defence = AdversarialTrainingPGDDefence(
            nb_epochs=10, eps=0.1, eps_step=0.01,
        )
        assert defence.nb_epochs == 10
        assert defence.eps == 0.1
        assert defence.name == "AdversarialTrainingPGD"

    def test_get_params(self):
        defence = AdversarialTrainingPGDDefence(
            nb_epochs=30, eps=0.3, max_iter=10, batch_size=64,
        )
        params = defence.get_params()
        assert params["nb_epochs"] == 30
        assert params["eps"] == 0.3
        assert params["max_iter"] == 10
        assert params["batch_size"] == 64
        assert params["ratio"] == 1.0

    @patch(
        'auto_art.core.evaluation.defences.trainer.ART_TRAINERS_AVAILABLE',
        False,
    )
    def test_apply_raises_without_art(self):
        defence = AdversarialTrainingPGDDefence()
        try:
            defence.apply(MagicMock())
            assert False, "Should have raised ImportError"
        except ImportError:
            pass

    def test_apply_without_data_returns_estimator(self):
        defence = AdversarialTrainingPGDDefence()
        estimator = MagicMock()
        with patch(
            'auto_art.core.evaluation.defences.trainer.ART_TRAINERS_AVAILABLE',
            True,
        ), patch(
            'auto_art.core.evaluation.defences.trainer.'
            'ARTAdversarialTrainerMadryPGD',
        ), patch(
            'art.attacks.evasion.ProjectedGradientDescent',
        ):
            result = defence.apply(estimator)
            assert result is estimator
