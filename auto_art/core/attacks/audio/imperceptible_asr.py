"""
Imperceptible ASR attack wrapper.

Wraps ART's implementation of the imperceptible adversarial attack against
automatic speech recognition systems. Uses psychoacoustic masking to produce
adversarial audio that is indistinguishable from benign audio to human ears.

Reference: Qin et al., "Imperceptible, Robust, and Targeted Adversarial
Examples for Automatic Speech Recognition", ICML 2019.
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import ImperceptibleASR as ARTImperceptibleASR
    IMPERCEPTIBLE_ASR_AVAILABLE = True
except ImportError:
    try:
        from art.attacks.evasion.imperceptible_asr import (
            ImperceptibleASR as ARTImperceptibleASR,
        )
        IMPERCEPTIBLE_ASR_AVAILABLE = True
    except ImportError:
        IMPERCEPTIBLE_ASR_AVAILABLE = False


class ImperceptibleASRWrapper:
    """Wrapper for ART's ImperceptibleASR attack (Qin et al., 2019).

    Generates adversarial audio using psychoacoustic hiding to ensure
    perturbations fall below human hearing thresholds while still
    fooling ASR systems into producing attacker-chosen transcriptions.
    """

    def __init__(
        self,
        estimator: Any,
        eps: float = 0.05,
        max_iter: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 1,
        verbose: bool = True,
        **kwargs,
    ):
        if not IMPERCEPTIBLE_ASR_AVAILABLE:
            raise ImportError(
                "ART ImperceptibleASR not available. Ensure "
                "adversarial-robustness-toolbox is installed with audio support."
            )
        self.art_attack = ARTImperceptibleASR(
            estimator=estimator,
            eps=eps,
            max_iter=max_iter,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=verbose,
            **kwargs,
        )
        self.attack_params = {
            "eps": eps,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
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
