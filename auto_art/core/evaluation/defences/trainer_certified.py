"""
ART Certified Adversarial Training defence wrappers.

Wraps ART's AdversarialTrainerCertifiedPytorch into the auto-art
DefenceStrategy interface for provably robust adversarial training
via interval bound propagation (IBP) and related methods.

Reference:
  - Gowal et al., 2019 - "Scalable Verified Training for Provably Robust Image Classification"
  - Mirman et al., 2018 - "Differentiable Abstract Interpretation for Provably Robust Neural Networks"
"""

from typing import Any, Dict, Optional
import logging

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.trainer import (
        AdversarialTrainerCertifiedPytorch as ARTAdversarialTrainerCertifiedPytorch,
    )
    ART_TRAINER_CERTIFIED_AVAILABLE = True
except ImportError:
    ART_TRAINER_CERTIFIED_AVAILABLE = False


class CertifiedAdversarialTrainingDefence(DefenceStrategy):
    """Certified adversarial training defence (Gowal et al., 2019).

    Trains models with provable robustness guarantees using certified
    bounds on the worst-case loss within an epsilon-ball around each
    input. Supports multiple bound propagation methods and loss types.
    """

    def __init__(
        self,
        nb_epochs: int = 20,
        batch_size: int = 128,
        bound: float = 0.1,
        loss_type: str = "interval",
    ):
        super().__init__(defence_name="CertifiedAdversarialTraining")
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.bound = bound
        self.loss_type = loss_type
        self._trainer: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply certified adversarial training to the estimator.

        Requires x_train and y_train in kwargs.
        """
        if not ART_TRAINER_CERTIFIED_AVAILABLE:
            raise ImportError(
                "ART AdversarialTrainerCertifiedPytorch not available. "
                "Install adversarial-robustness-toolbox with PyTorch support."
            )

        self._trainer = ARTAdversarialTrainerCertifiedPytorch(
            classifier=art_estimator,
            bound=self.bound,
            loss_type=self.loss_type,
        )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is not None and y_train is not None:
            logger.info(
                f"Running certified adversarial training for {self.nb_epochs} "
                f"epochs on {len(x_train)} samples (bound={self.bound}, "
                f"loss_type={self.loss_type})."
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
            "batch_size": self.batch_size,
            "bound": self.bound,
            "loss_type": self.loss_type,
        }


class IntervalBoundPropagationDefence(DefenceStrategy):
    """Interval Bound Propagation (IBP) adversarial training (Mirman et al., 2018).

    A specialization of certified adversarial training that uses interval
    bound propagation for computing certified bounds. IBP propagates
    interval constraints through each layer to bound the network output.
    """

    def __init__(
        self,
        nb_epochs: int = 20,
        batch_size: int = 128,
    ):
        super().__init__(defence_name="IntervalBoundPropagation")
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self._trainer: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply IBP adversarial training to the estimator.

        Requires x_train and y_train in kwargs.
        """
        if not ART_TRAINER_CERTIFIED_AVAILABLE:
            raise ImportError(
                "ART AdversarialTrainerCertifiedPytorch not available. "
                "Install adversarial-robustness-toolbox with PyTorch support."
            )

        self._trainer = ARTAdversarialTrainerCertifiedPytorch(
            classifier=art_estimator,
            bound=0.1,
            loss_type="interval",
        )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is not None and y_train is not None:
            logger.info(
                f"Running IBP adversarial training for {self.nb_epochs} "
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
            "batch_size": self.batch_size,
        }
