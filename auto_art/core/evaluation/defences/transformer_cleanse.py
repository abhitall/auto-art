"""
ART Poisoning Transformer defence wrappers (Neural Cleanse, STRIP).

Wraps ART's poisoning transformer defences into the auto-art DefenceStrategy
interface for detecting and mitigating backdoor attacks.

Reference:
  - Neural Cleanse: Wang et al., 2019 - "Neural Cleanse: Identifying and
    Mitigating Backdoor Attacks in Neural Networks"
  - STRIP: Gao et al., 2019 - "STRIP: A Defence Against Trojan Attacks on
    Deep Neural Networks"
"""

from typing import Any, Dict, Optional
import logging

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.transformer.poisoning import (
        NeuralCleanse as ARTNeuralCleanse,
    )
    ART_NEURAL_CLEANSE_AVAILABLE = True
except ImportError:
    ART_NEURAL_CLEANSE_AVAILABLE = False

try:
    from art.defences.transformer.poisoning import (
        STRIP as ARTSTRIP,
    )
    ART_STRIP_AVAILABLE = True
except ImportError:
    ART_STRIP_AVAILABLE = False


class NeuralCleanseDefence(DefenceStrategy):
    """Neural Cleanse defence (Wang et al., 2019).

    Reverse-engineers potential triggers for each output class by finding
    the minimal perturbation that causes misclassification to that class.
    Classes with anomalously small triggers are flagged as potentially
    backdoored.
    """

    def __init__(
        self,
        steps: int = 1000,
        init_cost: float = 1e-3,
        norm: int = 2,
        learning_rate: float = 0.1,
        attack_size: float = 0.01,
        early_stop: bool = True,
    ):
        super().__init__(defence_name="NeuralCleanse")
        self.steps = steps
        self.init_cost = init_cost
        self.norm = norm
        self.learning_rate = learning_rate
        self.attack_size = attack_size
        self.early_stop = early_stop
        self._transformer: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply Neural Cleanse to the estimator.

        Analyzes the model for potential backdoor triggers and returns
        the mitigated estimator.
        """
        if not ART_NEURAL_CLEANSE_AVAILABLE:
            raise ImportError(
                "ART NeuralCleanse transformer defence not available. "
                "Install adversarial-robustness-toolbox."
            )

        self._transformer = ARTNeuralCleanse(
            classifier=art_estimator,
            steps=self.steps,
            init_cost=self.init_cost,
            norm=self.norm,
            learning_rate=self.learning_rate,
            attack_size=self.attack_size,
            early_stop=self.early_stop,
        )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        if x_train is not None and y_train is not None:
            logger.info(
                f"Running Neural Cleanse analysis on {len(x_train)} samples "
                f"for {self.steps} steps."
            )
            mitigated = self._transformer.mitigate(x_train, y_train)
            return mitigated
        else:
            logger.warning(
                "No training data provided (x_train, y_train). "
                "Returning estimator without mitigation."
            )
            return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "init_cost": self.init_cost,
            "norm": self.norm,
            "learning_rate": self.learning_rate,
            "attack_size": self.attack_size,
            "early_stop": self.early_stop,
        }


class STRIPDefence(DefenceStrategy):
    """STRIP defence (Gao et al., 2019).

    STRong Intentional Perturbation detects Trojan attacks at inference
    time by perturbing inputs and observing prediction consistency.
    Clean inputs show variable predictions under perturbation, while
    trojaned inputs maintain consistent predictions.
    """

    def __init__(self, num_samples: int = 20):
        super().__init__(defence_name="STRIP")
        self.num_samples = num_samples
        self._transformer: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        """Apply STRIP defence to the estimator."""
        if not ART_STRIP_AVAILABLE:
            raise ImportError(
                "ART STRIP transformer defence not available. "
                "Install adversarial-robustness-toolbox."
            )

        x_train = kwargs.get("x_train")
        y_train = kwargs.get("y_train")

        self._transformer = ARTSTRIP(
            classifier=art_estimator,
            num_samples=self.num_samples,
        )

        if x_train is not None and y_train is not None:
            logger.info(
                f"Running STRIP defence with {self.num_samples} perturbation "
                f"samples on {len(x_train)} training examples."
            )
            mitigated = self._transformer.mitigate(x_train, y_train)
            return mitigated
        else:
            logger.warning(
                "No training data provided (x_train, y_train). "
                "Returning estimator without STRIP mitigation."
            )
            return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {"num_samples": self.num_samples}
