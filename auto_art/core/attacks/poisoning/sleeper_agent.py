"""
Sleeper Agent attack wrapper — dormant backdoor that activates on trigger.

Reference: Souri et al., "Sleeper Agent: Scalable Hidden Trigger Backdoors"
"""
from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.poisoning import SleeperAgentAttack as ARTSleeperAgent
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False


class SleeperAgentWrapper:
    """Wrapper for ART's Sleeper Agent dormant backdoor attack."""

    def __init__(self, classifier: Any, percent_poison: float = 0.1,
                 epsilon: float = 0.1, max_trials: int = 8,
                 max_epochs: int = 250, learning_rate_schedule: Optional[list] = None,
                 batch_size: int = 128, verbose: bool = True):
        if not ART_AVAILABLE:
            raise ImportError("ART SleeperAgentAttack not available.")
        kwargs: dict = {"classifier": classifier, "percent_poison": percent_poison,
                        "epsilon": epsilon, "max_trials": max_trials,
                        "max_epochs": max_epochs, "batch_size": batch_size,
                        "verbose": verbose}
        if learning_rate_schedule is not None:
            kwargs["learning_rate_schedule"] = learning_rate_schedule
        self.art_attack = ARTSleeperAgent(**kwargs)
        self.attack_params = {"percent_poison": percent_poison, "epsilon": epsilon,
                              "max_trials": max_trials, "max_epochs": max_epochs}

    def generate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> tuple:
        return self.art_attack.poison(x, y, **kwargs)
