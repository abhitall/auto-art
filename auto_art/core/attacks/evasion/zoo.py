"""
Zeroth Order Optimization (ZOO) evasion attack wrapper.

Wraps ART's ZooAttack into auto-art's wrapper pattern.
ZOO is a black-box attack that uses zeroth-order optimization
to estimate gradients via finite differences.

Reference: Chen et al., "ZOO: Zeroth Order Optimization Based
Black-box Attacks to Deep Neural Networks without Training
Substitute Models", ACM CCS Workshop AISec 2017
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import ZooAttack as ARTZooAttack
    from art.estimators.classification import ClassifierMixin
    ART_ZOO_AVAILABLE = True
except ImportError:
    ART_ZOO_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class ZOOWrapper:
    """Wrapper for ART's ZOO Attack (Chen et al., 2017).

    Performs a black-box C&W-style attack using zeroth-order
    gradient estimation via coordinate-wise finite differences.
    """

    def __init__(
        self,
        estimator: Any,
        confidence: float = 0.0,
        targeted: bool = False,
        learning_rate: float = 0.01,
        max_iter: int = 10,
        binary_search_steps: int = 1,
        initial_const: float = 0.001,
        abort_early: bool = True,
        use_resize: bool = True,
        use_importance: bool = True,
        nb_parallel: int = 128,
        batch_size: int = 1,
        variable_h: float = 0.0001,
        verbose: bool = True,
    ):
        if not ART_ZOO_AVAILABLE:
            raise ImportError("ART ZooAttack not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        self.art_attack = ARTZooAttack(
            classifier=estimator,
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            max_iter=max_iter,
            binary_search_steps=binary_search_steps,
            initial_const=initial_const,
            abort_early=abort_early,
            use_resize=use_resize,
            use_importance=use_importance,
            nb_parallel=nb_parallel,
            batch_size=batch_size,
            variable_h=variable_h,
            verbose=verbose,
        )
        self.attack_params = {
            "confidence": confidence, "targeted": targeted,
            "learning_rate": learning_rate, "max_iter": max_iter,
            "binary_search_steps": binary_search_steps,
            "initial_const": initial_const, "abort_early": abort_early,
            "use_resize": use_resize, "use_importance": use_importance,
            "nb_parallel": nb_parallel, "batch_size": batch_size,
            "variable_h": variable_h,
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
