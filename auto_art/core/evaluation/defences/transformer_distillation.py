"""
ART Defensive Distillation transformer defence wrapper.

Wraps ART's DefensiveDistillation into the auto-art DefenceStrategy
interface for mitigating evasion attacks through knowledge distillation
with temperature scaling.

Reference: Papernot et al., 2016 - "Distillation as a Defense to Adversarial
Perturbations Against Deep Neural Networks"
"""

from typing import Any, Dict, Optional
import logging

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.transformer.evasion import (
        DefensiveDistillation as ARTDefensiveDistillation,
    )
    ART_DEFENSIVE_DISTILLATION_AVAILABLE = True
except ImportError:
    ART_DEFENSIVE_DISTILLATION_AVAILABLE = False


class DefensiveDistillationDefence(DefenceStrategy):
    """Defensive Distillation defence (Papernot et al., 2016).

    Trains a distilled model at high temperature to smooth the decision
    boundaries, making the model more robust to small adversarial
    perturbations. The teacher model's soft probabilities guide the
    student model to learn smoother representations.
    """

    def __init__(
        self,
        batch_size: int = 128,
        nb_epochs: int = 10,
        temperature: float = 10.0,
    ):
        super().__init__(defence_name="DefensiveDistillation")
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.temperature = temperature
        self._transformer: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply defensive distillation to the estimator.

        Requires x_train and y_train in kwargs to train the distilled model.
        """
        if not ART_DEFENSIVE_DISTILLATION_AVAILABLE:
            raise ImportError(
                "ART DefensiveDistillation transformer defence not available. "
                "Install adversarial-robustness-toolbox."
            )

        self._transformer = ARTDefensiveDistillation(
            classifier=art_estimator,
            batch_size=self.batch_size,
            nb_epochs=self.nb_epochs,
        )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is not None and y_train is not None:
            logger.info(
                f"Running defensive distillation for {self.nb_epochs} epochs "
                f"with temperature={self.temperature} on {len(x_train)} samples."
            )
            distilled = self._transformer(x_train, y_train)
            return distilled
        else:
            logger.warning(
                "No training data provided (x_train, y_train). "
                "Returning estimator without distillation."
            )
            return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "nb_epochs": self.nb_epochs,
            "temperature": self.temperature,
        }
