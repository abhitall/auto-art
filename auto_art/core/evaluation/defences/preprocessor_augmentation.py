"""
ART Augmentation Preprocessor defence wrappers.

Wraps ART's augmentation-based preprocessor defences (Cutout, Mixup,
CutMix) into the auto-art DefenceStrategy interface so they can be
used with ARTEvaluator.

Reference:
  - Cutout: DeVries & Taylor, 2017 - "Improved Regularization of CNNs with Cutout"
  - Mixup: Zhang et al., 2018 - "mixup: Beyond Empirical Risk Minimization"
  - CutMix: Yun et al., 2019 - "CutMix: Regularization Strategy to Train Strong Classifiers"
"""

from typing import Any, Dict, Optional
import logging
import numpy as np

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.preprocessor import (
        CutoutPyTorch as ARTCutoutPyTorch,
        MixupPyTorch as ARTMixupPyTorch,
        CutMixPyTorch as ARTCutMixPyTorch,
    )
    ART_AUGMENTATION_PREPROCESSORS_AVAILABLE = True
except ImportError:
    ART_AUGMENTATION_PREPROCESSORS_AVAILABLE = False


class CutoutDefence(DefenceStrategy):
    """Cutout preprocessor defence (DeVries & Taylor, 2017).

    Randomly masks out square regions of input images during training,
    acting as a regularizer that improves robustness to occlusion and
    adversarial perturbations.
    """

    def __init__(self, length: int = 16, channels_first: bool = False):
        super().__init__(defence_name="Cutout")
        self.length = length
        self.channels_first = channels_first
        self._preprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_AUGMENTATION_PREPROCESSORS_AVAILABLE:
            raise ImportError(
                "ART augmentation preprocessor defences not available. "
                "Install adversarial-robustness-toolbox with PyTorch support."
            )
        self._preprocessor = ARTCutoutPyTorch(
            length=self.length,
            channels_first=self.channels_first,
        )
        if hasattr(art_estimator, 'set_params'):
            current = getattr(art_estimator, 'preprocessing_defences', None)
            defences = list(current) if current else []
            defences.append(self._preprocessor)
            art_estimator.set_params(preprocessing_defences=defences)
        return art_estimator

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply cutout directly to data."""
        if self._preprocessor is None:
            if not ART_AUGMENTATION_PREPROCESSORS_AVAILABLE:
                raise ImportError(
                    "ART augmentation preprocessor defences not available."
                )
            self._preprocessor = ARTCutoutPyTorch(
                length=self.length,
                channels_first=self.channels_first,
            )
        result, _ = self._preprocessor(x)
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "channels_first": self.channels_first,
        }


class MixupDefence(DefenceStrategy):
    """Mixup preprocessor defence (Zhang et al., 2018).

    Creates virtual training examples by taking convex combinations
    of pairs of training samples and their labels, encouraging the
    model to learn linear behavior between training examples.
    """

    def __init__(
        self,
        num_classes: int = 10,
        alpha: float = 1.0,
        channels_first: bool = False,
    ):
        super().__init__(defence_name="Mixup")
        self.num_classes = num_classes
        self.alpha = alpha
        self.channels_first = channels_first
        self._preprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_AUGMENTATION_PREPROCESSORS_AVAILABLE:
            raise ImportError(
                "ART augmentation preprocessor defences not available. "
                "Install adversarial-robustness-toolbox with PyTorch support."
            )
        self._preprocessor = ARTMixupPyTorch(
            num_classes=self.num_classes,
            alpha=self.alpha,
            channels_first=self.channels_first,
        )
        if hasattr(art_estimator, 'set_params'):
            current = getattr(art_estimator, 'preprocessing_defences', None)
            defences = list(current) if current else []
            defences.append(self._preprocessor)
            art_estimator.set_params(preprocessing_defences=defences)
        return art_estimator

    def transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply mixup directly to data."""
        if self._preprocessor is None:
            if not ART_AUGMENTATION_PREPROCESSORS_AVAILABLE:
                raise ImportError(
                    "ART augmentation preprocessor defences not available."
                )
            self._preprocessor = ARTMixupPyTorch(
                num_classes=self.num_classes,
                alpha=self.alpha,
                channels_first=self.channels_first,
            )
        result, _ = self._preprocessor(x, y)
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "channels_first": self.channels_first,
        }


class CutMixDefence(DefenceStrategy):
    """CutMix preprocessor defence (Yun et al., 2019).

    Combines Cutout and Mixup by cutting and pasting patches between
    training images, with ground truth labels mixed proportionally to
    the area of the patches.
    """

    def __init__(
        self,
        num_classes: int = 10,
        alpha: float = 1.0,
        channels_first: bool = False,
    ):
        super().__init__(defence_name="CutMix")
        self.num_classes = num_classes
        self.alpha = alpha
        self.channels_first = channels_first
        self._preprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_AUGMENTATION_PREPROCESSORS_AVAILABLE:
            raise ImportError(
                "ART augmentation preprocessor defences not available. "
                "Install adversarial-robustness-toolbox with PyTorch support."
            )
        self._preprocessor = ARTCutMixPyTorch(
            num_classes=self.num_classes,
            alpha=self.alpha,
            channels_first=self.channels_first,
        )
        if hasattr(art_estimator, 'set_params'):
            current = getattr(art_estimator, 'preprocessing_defences', None)
            defences = list(current) if current else []
            defences.append(self._preprocessor)
            art_estimator.set_params(preprocessing_defences=defences)
        return art_estimator

    def transform(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply CutMix directly to data."""
        if self._preprocessor is None:
            if not ART_AUGMENTATION_PREPROCESSORS_AVAILABLE:
                raise ImportError(
                    "ART augmentation preprocessor defences not available."
                )
            self._preprocessor = ARTCutMixPyTorch(
                num_classes=self.num_classes,
                alpha=self.alpha,
                channels_first=self.channels_first,
            )
        result, _ = self._preprocessor(x, y)
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "num_classes": self.num_classes,
            "alpha": self.alpha,
            "channels_first": self.channels_first,
        }
