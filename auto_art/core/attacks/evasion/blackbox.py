"""
Black-box evasion attack wrappers.

Wraps ART's black-box evasion attacks (Square Attack, HopSkipJump, SimBA)
into auto-art's wrapper pattern. These attacks do not require gradient
access and work against deployed models via query access only.

Reference: ART Attacks Wiki - Black-box section
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import (
        SquareAttack as ARTSquareAttack,
        HopSkipJump as ARTHopSkipJump,
        SimBA as ARTSimBA,
    )
    from art.estimators.classification import ClassifierMixin
    ART_BLACKBOX_AVAILABLE = True
except ImportError:
    ART_BLACKBOX_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class SquareAttackWrapper:
    """Wrapper for ART's Square Attack (Andriushchenko et al., 2020).

    A score-based black-box attack that uses random search in a reduced
    space of perturbations. Achieves state-of-the-art query efficiency
    among black-box attacks.
    """

    def __init__(
        self,
        estimator: Any,
        norm: str = "inf",
        eps: float = 0.3,
        max_iter: int = 100,
        nb_restarts: int = 1,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        if not ART_BLACKBOX_AVAILABLE:
            raise ImportError("ART black-box attacks not available.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        norm_val: Any = np.inf if norm == "inf" else int(norm)
        self.art_attack = ARTSquareAttack(
            estimator=estimator,
            norm=norm_val,
            eps=eps,
            max_iter=max_iter,
            nb_restarts=nb_restarts,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "norm": norm, "eps": eps, "max_iter": max_iter,
            "nb_restarts": nb_restarts, "batch_size": batch_size,
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


class HopSkipJumpWrapper:
    """Wrapper for ART's HopSkipJump Attack (Chen et al., 2019).

    A decision-based black-box attack that only requires the final
    predicted label (no confidence scores). Uses a binary search
    along the decision boundary.
    """

    def __init__(
        self,
        estimator: Any,
        targeted: bool = False,
        norm: str = "2",
        max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        batch_size: int = 64,
        verbose: bool = True,
    ):
        if not ART_BLACKBOX_AVAILABLE:
            raise ImportError("ART black-box attacks not available.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        norm_val: Any = np.inf if norm == "inf" else int(norm)
        self.art_attack = ARTHopSkipJump(
            classifier=estimator,
            targeted=targeted,
            norm=norm_val,
            max_iter=max_iter,
            max_eval=max_eval,
            init_eval=init_eval,
            init_size=init_size,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "targeted": targeted, "norm": norm,
            "max_iter": max_iter, "max_eval": max_eval,
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


class SimBAWrapper:
    """Wrapper for ART's SimBA Attack (Guo et al., 2019).

    Simple Black-box Adversarial attack that uses a simple iterative
    approach, perturbing one dimension at a time. Very query-efficient
    for untargeted attacks.
    """

    def __init__(
        self,
        estimator: Any,
        attack: str = "dct",
        max_iter: int = 3000,
        epsilon: float = 0.1,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        if not ART_BLACKBOX_AVAILABLE:
            raise ImportError("ART black-box attacks not available.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTSimBA(
            classifier=estimator,
            attack=attack,
            max_iter=max_iter,
            epsilon=epsilon,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "attack": attack, "max_iter": max_iter,
            "epsilon": epsilon,
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
