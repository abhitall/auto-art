"""
ART Fast is Better than Free (FBF) Adversarial Training defence wrapper.

Wraps ART's AdversarialTrainerFBFPyTorch into the auto-art DefenceStrategy
interface for single-step FGSM-based adversarial training with random
initialization.

Reference: Wong et al., 2020 - "Fast is better than free: Revisiting
Adversarial Training"
"""

from typing import Any, Dict, Optional
import logging

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.trainer import (
        AdversarialTrainerFBFPyTorch as ARTAdversarialTrainerFBFPyTorch,
    )
    ART_TRAINER_FBF_AVAILABLE = True
except ImportError:
    ART_TRAINER_FBF_AVAILABLE = False


class FastIsBetterThanFreeDefence(DefenceStrategy):
    """Fast is Better than Free (FBF) adversarial training (Wong et al., 2020).

    Uses single-step FGSM with random initialization for adversarial
    training, achieving comparable robustness to multi-step PGD training
    at a fraction of the computational cost.
    """

    def __init__(
        self,
        nb_epochs: int = 20,
        eps: float = 0.3,
        batch_size: int = 128,
    ):
        super().__init__(defence_name="FastIsBetterThanFree")
        self.nb_epochs = nb_epochs
        self.eps = eps
        self.batch_size = batch_size
        self._trainer: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply FBF adversarial training to the estimator.

        Requires x_train and y_train in kwargs.
        """
        if not ART_TRAINER_FBF_AVAILABLE:
            raise ImportError(
                "ART AdversarialTrainerFBFPyTorch not available. "
                "Install adversarial-robustness-toolbox with PyTorch support."
            )

        self._trainer = ARTAdversarialTrainerFBFPyTorch(
            classifier=art_estimator,
            eps=self.eps,
        )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is not None and y_train is not None:
            logger.info(
                f"Running FBF adversarial training for {self.nb_epochs} epochs "
                f"on {len(x_train)} samples."
            )
            self._trainer.fit(
                x_train, y_train,
                nb_epochs=self.nb_epochs,
                batch_size=self.batch_size,
            )
            return self._trainer.get_classifier()
        else:
            logger.warning(
                "No training data provided (x_train, y_train). "
                "Returning trainer without training."
            )
            return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {
            "nb_epochs": self.nb_epochs,
            "eps": self.eps,
            "batch_size": self.batch_size,
        }
