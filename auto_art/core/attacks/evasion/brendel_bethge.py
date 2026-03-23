"""
Brendel & Bethge Attack evasion attack wrapper.

Wraps ART's BrendelBethgeAttack into auto-art's wrapper pattern.
A powerful minimum-norm adversarial attack that operates along the
decision boundary using gradient-based optimisation.

Reference: Brendel et al., "Accurate, Large Minibatch SGD:
Training ImageNet in 1 Hour" — attack from "Adversarial Vision
Challenge" / Brendel & Bethge, 2019
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import BrendelBethgeAttack as ARTBrendelBethgeAttack
    from art.estimators.classification import ClassifierMixin
    ART_BB_AVAILABLE = True
except ImportError:
    ART_BB_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class BrendelBethgeWrapper:
    """Wrapper for ART's Brendel & Bethge Attack (Brendel et al., 2019).

    A boundary attack that uses gradient-based optimisation to walk
    along the decision boundary towards the minimum-norm adversarial
    perturbation.
    """

    def __init__(
        self,
        estimator: Any,
        norm: str = "2",
        targeted: bool = False,
        overshoot: float = 1.1,
        steps: int = 1000,
        lr: float = 0.001,
        lr_decay: float = 0.5,
        lr_num_decay: int = 20,
        momentum: float = 0.8,
        binary_search_steps: int = 10,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        if not ART_BB_AVAILABLE:
            raise ImportError("ART BrendelBethgeAttack not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        norm_val: Any = np.inf if norm == "inf" else int(norm)
        self.art_attack = ARTBrendelBethgeAttack(
            estimator=estimator,
            norm=norm_val,
            targeted=targeted,
            overshoot=overshoot,
            steps=steps,
            lr=lr,
            lr_decay=lr_decay,
            lr_num_decay=lr_num_decay,
            momentum=momentum,
            binary_search_steps=binary_search_steps,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "norm": norm, "targeted": targeted, "overshoot": overshoot,
            "steps": steps, "lr": lr, "lr_decay": lr_decay,
            "lr_num_decay": lr_num_decay, "momentum": momentum,
            "binary_search_steps": binary_search_steps,
            "batch_size": batch_size,
        }

    @property
    def attack(self) -> Any:
        return self.art_attack

    def generate(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.art_attack.generate(x=x, y=y, **kwargs)
