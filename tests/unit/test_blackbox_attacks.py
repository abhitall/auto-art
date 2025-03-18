"""
Tests for black-box evasion attack wrappers.
"""

import numpy as np
from unittest.mock import MagicMock, patch

from auto_art.core.attacks.evasion.blackbox import (
    SquareAttackWrapper,
    HopSkipJumpWrapper,
    SimBAWrapper,
)


class TestSquareAttackWrapper:

    @patch(
        'auto_art.core.attacks.evasion.blackbox.ART_BLACKBOX_AVAILABLE',
        False,
    )
    def test_raises_without_art(self):
        try:
            SquareAttackWrapper(estimator=MagicMock())
            assert False, "Should have raised ImportError"
        except ImportError:
            pass

    def test_raises_with_non_classifier(self):
        try:
            SquareAttackWrapper(estimator="not_a_classifier")
            assert False, "Should have raised TypeError"
        except (TypeError, ImportError):
            pass

    def test_params_stored(self):
        wrapper = MagicMock(spec=SquareAttackWrapper)
        wrapper.attack_params = {
            "norm": "inf", "eps": 0.3,
            "max_iter": 100, "nb_restarts": 1, "batch_size": 1,
        }
        assert wrapper.attack_params["eps"] == 0.3


class TestHopSkipJumpWrapper:

    @patch(
        'auto_art.core.attacks.evasion.blackbox.ART_BLACKBOX_AVAILABLE',
        False,
    )
    def test_raises_without_art(self):
        try:
            HopSkipJumpWrapper(estimator=MagicMock())
            assert False, "Should have raised ImportError"
        except ImportError:
            pass

    def test_params_stored(self):
        wrapper = MagicMock(spec=HopSkipJumpWrapper)
        wrapper.attack_params = {
            "targeted": False, "norm": "2",
            "max_iter": 50, "max_eval": 10000,
        }
        assert wrapper.attack_params["norm"] == "2"


class TestSimBAWrapper:

    @patch(
        'auto_art.core.attacks.evasion.blackbox.ART_BLACKBOX_AVAILABLE',
        False,
    )
    def test_raises_without_art(self):
        try:
            SimBAWrapper(estimator=MagicMock())
            assert False, "Should have raised ImportError"
        except ImportError:
            pass

    def test_params_stored(self):
        wrapper = MagicMock(spec=SimBAWrapper)
        wrapper.attack_params = {
            "attack": "dct", "max_iter": 3000, "epsilon": 0.1,
        }
        assert wrapper.attack_params["attack"] == "dct"
