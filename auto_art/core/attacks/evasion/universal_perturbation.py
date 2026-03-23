"""
Universal Adversarial Perturbation evasion attack wrapper.

Wraps ART's UniversalPerturbation into auto-art's wrapper pattern.
Computes a single image-agnostic perturbation that fools the model
on most inputs.

Reference: Moosavi-Dezfooli et al., "Universal Adversarial
Perturbations", CVPR 2017
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import UniversalPerturbation as ARTUniversalPerturbation
    from art.estimators.classification import ClassifierMixin
    ART_UAP_AVAILABLE = True
except ImportError:
    ART_UAP_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class UniversalPerturbationWrapper:
    """Wrapper for ART's Universal Perturbation (Moosavi-Dezfooli et al., 2017).

    Computes a single universal perturbation vector that, when added
    to most inputs, causes the classifier to misclassify them.
    """

    def __init__(
        self,
        estimator: Any,
        attacker: str = "deepfool",
        attacker_params: Optional[dict] = None,
        delta: float = 0.2,
        max_iter: int = 20,
        eps: float = 10.0,
        norm: str = "inf",
        batch_size: int = 32,
        verbose: bool = True,
    ):
        if not ART_UAP_AVAILABLE:
            raise ImportError("ART UniversalPerturbation not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        norm_val: Any = np.inf if norm == "inf" else int(norm)
        self.art_attack = ARTUniversalPerturbation(
            classifier=estimator,
            attacker=attacker,
            attacker_params=attacker_params,
            delta=delta,
            max_iter=max_iter,
            eps=eps,
            norm=norm_val,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.attack_params = {
            "attacker": attacker, "attacker_params": attacker_params,
            "delta": delta, "max_iter": max_iter, "eps": eps,
            "norm": norm, "batch_size": batch_size,
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
