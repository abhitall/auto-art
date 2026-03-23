"""
TRADES adversarial training defence wrapper.

Reference: Zhang et al., "Theoretically Principled Trade-off between
Robustness and Accuracy", ICML 2019
"""
import logging
from typing import Any

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.trainer import AdversarialTrainerTRADESPyTorch as ARTTrainerTRADES
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class TRADESDefence(DefenceStrategy):
    """TRADES adversarial training — theoretically principled trade-off
    between robustness and accuracy.

    Optimizes: natural loss + beta * KL-divergence(clean, adversarial).
    """

    def __init__(self, nb_epochs: int = 20, eps: float = 0.3,
                 eps_step: float = 0.1, max_iter: int = 7,
                 beta: float = 6.0, batch_size: int = 128):
        super().__init__(defence_name="TRADES")
        self.nb_epochs = nb_epochs
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.beta = beta
        self.batch_size = batch_size

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_AVAILABLE:
            raise ImportError("ART TRADES trainer not available. Requires PyTorch.")
        self._trainer = ARTTrainerTRADES(
            classifier=art_estimator, eps=self.eps,
            eps_step=self.eps_step, max_iter=self.max_iter,
            beta=self.beta, batch_size=self.batch_size,
        )
        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")
        if x_train is not None and y_train is not None:
            logger.info(f"TRADES training for {self.nb_epochs} epochs on {len(x_train)} samples.")
            self._trainer.fit(x_train, y_train, nb_epochs=self.nb_epochs, batch_size=self.batch_size)
            return self._trainer.get_classifier()
        logger.warning("No training data provided. Returning estimator unchanged.")
        return art_estimator

    def get_params(self) -> dict[str, Any]:
        return {"nb_epochs": self.nb_epochs, "eps": self.eps, "eps_step": self.eps_step,
                "max_iter": self.max_iter, "beta": self.beta, "batch_size": self.batch_size}
