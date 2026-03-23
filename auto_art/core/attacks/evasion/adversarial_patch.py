"""
Adversarial Patch evasion attack wrapper.

Wraps ART's AdversarialPatch (numpy) or AdversarialPatchPyTorch (pytorch)
into auto-art's wrapper pattern. Generates a universal adversarial patch
that can be applied to any input.

Reference: Brown et al., "Adversarial Patch", NeurIPS 2017
"""

from typing import Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from art.attacks.evasion import AdversarialPatch as ARTAdversarialPatch
    from art.estimators.classification import ClassifierMixin
    ART_PATCH_AVAILABLE = True
    try:
        from art.attacks.evasion import AdversarialPatchPyTorch as ARTAdversarialPatchPyTorch
        ART_PATCH_PYTORCH_AVAILABLE = True
    except ImportError:
        ART_PATCH_PYTORCH_AVAILABLE = False
except ImportError:
    ART_PATCH_AVAILABLE = False
    ART_PATCH_PYTORCH_AVAILABLE = False

    class ClassifierMixin:  # type: ignore
        pass


class AdversarialPatchWrapper:
    """Wrapper for ART's Adversarial Patch (Brown et al., 2017).

    Generates a universal adversarial patch optimised to cause
    misclassification when physically applied to inputs. Uses the
    PyTorch backend when available, falling back to the numpy version.
    """

    def __init__(
        self,
        estimator: Any,
        rotation_max: float = 22.5,
        scale_min: float = 0.3,
        scale_max: float = 1.0,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        targeted: bool = True,
        verbose: bool = True,
    ):
        if not ART_PATCH_AVAILABLE:
            raise ImportError("ART AdversarialPatch not available. Install adversarial-robustness-toolbox.")
        if not isinstance(estimator, ClassifierMixin):
            raise TypeError("estimator must be an ART ClassifierMixin.")

        patch_kwargs: dict[str, Any] = {
            "estimator": estimator,
            "rotation_max": rotation_max,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "batch_size": batch_size,
            "targeted": targeted,
            "verbose": verbose,
        }

        if ART_PATCH_PYTORCH_AVAILABLE:
            try:
                self.art_attack = ARTAdversarialPatchPyTorch(**patch_kwargs)
            except Exception:
                self.art_attack = ARTAdversarialPatch(**patch_kwargs)
        else:
            self.art_attack = ARTAdversarialPatch(**patch_kwargs)

        self.attack_params = {
            "rotation_max": rotation_max, "scale_min": scale_min,
            "scale_max": scale_max, "learning_rate": learning_rate,
            "max_iter": max_iter, "batch_size": batch_size,
            "targeted": targeted,
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
