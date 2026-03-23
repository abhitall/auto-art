"""
Carlini & Wagner audio adversarial attack wrapper.

Wraps ART's implementation of the Carlini & Wagner attack adapted for
automatic speech recognition (ASR) systems. Generates adversarial audio
that is misrecognized by the target ASR model.

Reference: Carlini and Wagner, "Audio Adversarial Examples: Targeted Attacks
on Speech-to-Text", IEEE SPW 2018.
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import CarliniWagnerASR as ARTCarliniWagnerASR
    CW_AUDIO_AVAILABLE = True
except ImportError:
    try:
        from art.attacks.evasion.carlini_wagner_audio import (
            CarliniWagnerASR as ARTCarliniWagnerASR,
        )
        CW_AUDIO_AVAILABLE = True
    except ImportError:
        CW_AUDIO_AVAILABLE = False


class CarliniWagnerAudioWrapper:
    """Wrapper for ART's CarliniWagnerASR attack (Carlini and Wagner, 2018).

    Generates adversarial audio examples that cause ASR systems to
    transcribe attacker-chosen text while remaining perceptually
    similar to the original audio.
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
        if not CW_AUDIO_AVAILABLE:
            raise ImportError(
                "ART CarliniWagnerASR not available. Ensure "
                "adversarial-robustness-toolbox is installed with audio support."
            )
        self.art_attack = ARTCarliniWagnerASR(
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
