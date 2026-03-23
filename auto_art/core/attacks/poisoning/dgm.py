"""
DGM poisoning attack wrappers for generative models.

Wraps ART's DGM (Deep Generative Model) poisoning attacks:
- DGMReD: Poisoning via red-team data injection into generative model training
- DGMTrail: Trail-based poisoning targeting the latent space of generative models

Reference: ART documentation on DGM poisoning attacks for generative models.
"""

from typing import Any, Optional, Tuple
import numpy as np

try:
    from art.attacks.poisoning import DGMReD as ARTDGMReD
    DGM_RED_AVAILABLE = True
except ImportError:
    DGM_RED_AVAILABLE = False

try:
    from art.attacks.poisoning import DGMTrail as ARTDGMTrail
    DGM_TRAIL_AVAILABLE = True
except ImportError:
    DGM_TRAIL_AVAILABLE = False


class DGMReDWrapper:
    """Wrapper for ART's DGMReD poisoning attack.

    Targets generative models by injecting adversarial data into the
    training pipeline, causing the generator to produce attacker-chosen
    outputs when given a specific trigger in the latent space.
    """

    def __init__(
        self,
        generator: Any,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        **kwargs,
    ):
        if not DGM_RED_AVAILABLE:
            raise ImportError(
                "ART DGMReD not available. Ensure adversarial-robustness-toolbox "
                "is installed with generative model support."
            )
        self.generator = generator
        self.z_trigger = z_trigger
        self.x_target = x_target
        self.art_attack = ARTDGMReD(
            generator=generator,
            z_trigger=z_trigger,
            x_target=x_target,
            **kwargs,
        )
        self.attack_params = {
            "z_trigger_shape": z_trigger.shape,
            "x_target_shape": x_target.shape,
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
        return self.art_attack.poison(x=x, y=y, **kwargs)


class DGMTrailWrapper:
    """Wrapper for ART's DGMTrail poisoning attack.

    Uses trail-based optimization to poison the latent space of generative
    models, causing targeted outputs when specific latent triggers are used.
    """

    def __init__(
        self,
        generator: Any,
        z_trigger: np.ndarray,
        x_target: np.ndarray,
        **kwargs,
    ):
        if not DGM_TRAIL_AVAILABLE:
            raise ImportError(
                "ART DGMTrail not available. Ensure adversarial-robustness-toolbox "
                "is installed with generative model support."
            )
        self.generator = generator
        self.z_trigger = z_trigger
        self.x_target = x_target
        self.art_attack = ARTDGMTrail(
            generator=generator,
            z_trigger=z_trigger,
            x_target=x_target,
            **kwargs,
        )
        self.attack_params = {
            "z_trigger_shape": z_trigger.shape,
            "x_target_shape": x_target.shape,
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
        return self.art_attack.poison(x=x, y=y, **kwargs)
