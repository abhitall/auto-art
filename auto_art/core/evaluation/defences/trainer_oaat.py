"""
ART OAAT (Once-for-all Adversarial Training) defence wrapper.

Wraps ART's AdversarialTrainerOAATPyTorch into the auto-art DefenceStrategy
interface for universal adversarial training that achieves robustness across
multiple perturbation types simultaneously.

Reference: Addepalli et al., 2022 - "Scaling Adversarial Training to Large
Perturbation Bounds"
"""

from typing import Any, Dict, Optional
import logging

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.trainer import (
        AdversarialTrainerOAATPyTorch as ARTAdversarialTrainerOAATPyTorch,
    )
    ART_TRAINER_OAAT_AVAILABLE = True
except ImportError:
    ART_TRAINER_OAAT_AVAILABLE = False


class OAATDefence(DefenceStrategy):
    """Once-for-all Adversarial Training (OAAT) defence (Addepalli et al., 2022).

    Performs adversarial training that scales to large perturbation bounds
    by using a single training procedure that defends against multiple
    attack types simultaneously with progressive scaling of perturbation
    budgets.
    """

    def __init__(
        self,
        nb_epochs: int = 20,
        eps: float = 0.3,
        eps_step: float = 0.1,
        max_iter: int = 10,
        batch_size: int = 128,
    ):
        super().__init__(defence_name="OAAT")
        self.nb_epochs = nb_epochs
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.batch_size = batch_size
        self._trainer: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply OAAT adversarial training to the estimator.

        Requires x_train and y_train in kwargs.
        """
        if not ART_TRAINER_OAAT_AVAILABLE:
            raise ImportError(
                "ART AdversarialTrainerOAATPyTorch not available. "
                "Install adversarial-robustness-toolbox with PyTorch support."
            )

        from art.attacks.evasion import ProjectedGradientDescent

        pgd = ProjectedGradientDescent(
            estimator=art_estimator,
            eps=self.eps,
            eps_step=self.eps_step,
            max_iter=self.max_iter,
            batch_size=self.batch_size,
        )

        self._trainer = ARTAdversarialTrainerOAATPyTorch(
            classifier=art_estimator,
            attacks=pgd,
        )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is not None and y_train is not None:
            logger.info(
                f"Running OAAT adversarial training for {self.nb_epochs} "
                f"epochs on {len(x_train)} samples."
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
            "eps_step": self.eps_step,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
        }
