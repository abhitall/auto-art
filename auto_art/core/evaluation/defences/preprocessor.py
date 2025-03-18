"""
ART Preprocessor defence wrappers.

Wraps ART's preprocessor defences (Spatial Smoothing, Feature Squeezing,
JPEG Compression, Gaussian Augmentation) into the auto-art DefenceStrategy
interface so they can be used with ARTEvaluator.

Reference: ART Defences Wiki - https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Defences
"""

from typing import Any, Dict, Optional
import logging
import numpy as np

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.preprocessor import (
        SpatialSmoothing as ARTSpatialSmoothing,
        FeatureSqueezing as ARTFeatureSqueezing,
        JpegCompression as ARTJpegCompression,
        GaussianAugmentation as ARTGaussianAugmentation,
    )
    ART_PREPROCESSORS_AVAILABLE = True
except ImportError:
    ART_PREPROCESSORS_AVAILABLE = False


class SpatialSmoothingDefence(DefenceStrategy):
    """Spatial smoothing preprocessor defence (Xu et al., 2017).

    Applies a spatial smoothing filter to input data to remove
    high-frequency adversarial perturbations.
    """

    def __init__(self, window_size: int = 3, channels_first: bool = False):
        super().__init__(defence_name="SpatialSmoothing")
        self.window_size = window_size
        self.channels_first = channels_first
        self._preprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_PREPROCESSORS_AVAILABLE:
            raise ImportError("ART preprocessor defences not available.")
        self._preprocessor = ARTSpatialSmoothing(
            window_size=self.window_size,
            channels_first=self.channels_first,
        )
        if hasattr(art_estimator, 'set_params'):
            current = getattr(art_estimator, 'preprocessing_defences', None)
            defences = list(current) if current else []
            defences.append(self._preprocessor)
            art_estimator.set_params(preprocessing_defences=defences)
        return art_estimator

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply spatial smoothing directly to data."""
        if self._preprocessor is None:
            if not ART_PREPROCESSORS_AVAILABLE:
                raise ImportError("ART preprocessor defences not available.")
            self._preprocessor = ARTSpatialSmoothing(
                window_size=self.window_size,
                channels_first=self.channels_first,
            )
        result, _ = self._preprocessor(x)
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "channels_first": self.channels_first,
        }


class FeatureSqueezingDefence(DefenceStrategy):
    """Feature squeezing preprocessor defence.

    Reduces the color bit depth of inputs to remove adversarial
    perturbations that rely on subtle pixel value changes.
    """

    def __init__(self, bit_depth: int = 4, clip_values: tuple = (0.0, 1.0)):
        super().__init__(defence_name="FeatureSqueezing")
        self.bit_depth = bit_depth
        self.clip_values = clip_values
        self._preprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_PREPROCESSORS_AVAILABLE:
            raise ImportError("ART preprocessor defences not available.")
        self._preprocessor = ARTFeatureSqueezing(
            bit_depth=self.bit_depth,
            clip_values=self.clip_values,
        )
        if hasattr(art_estimator, 'set_params'):
            current = getattr(art_estimator, 'preprocessing_defences', None)
            defences = list(current) if current else []
            defences.append(self._preprocessor)
            art_estimator.set_params(preprocessing_defences=defences)
        return art_estimator

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._preprocessor is None:
            if not ART_PREPROCESSORS_AVAILABLE:
                raise ImportError("ART preprocessor defences not available.")
            self._preprocessor = ARTFeatureSqueezing(
                bit_depth=self.bit_depth,
                clip_values=self.clip_values,
            )
        result, _ = self._preprocessor(x)
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "bit_depth": self.bit_depth,
            "clip_values": self.clip_values,
        }


class JpegCompressionDefence(DefenceStrategy):
    """JPEG compression preprocessor defence.

    Applies JPEG compression/decompression to remove adversarial
    perturbations that do not survive lossy compression.
    """

    def __init__(
        self,
        quality: int = 50,
        channels_first: bool = False,
        clip_values: tuple = (0.0, 1.0),
    ):
        super().__init__(defence_name="JpegCompression")
        self.quality = quality
        self.channels_first = channels_first
        self.clip_values = clip_values
        self._preprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_PREPROCESSORS_AVAILABLE:
            raise ImportError("ART preprocessor defences not available.")
        self._preprocessor = ARTJpegCompression(
            quality=self.quality,
            channels_first=self.channels_first,
            clip_values=self.clip_values,
        )
        if hasattr(art_estimator, 'set_params'):
            current = getattr(art_estimator, 'preprocessing_defences', None)
            defences = list(current) if current else []
            defences.append(self._preprocessor)
            art_estimator.set_params(preprocessing_defences=defences)
        return art_estimator

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._preprocessor is None:
            if not ART_PREPROCESSORS_AVAILABLE:
                raise ImportError("ART preprocessor defences not available.")
            self._preprocessor = ARTJpegCompression(
                quality=self.quality,
                channels_first=self.channels_first,
                clip_values=self.clip_values,
            )
        result, _ = self._preprocessor(x)
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "quality": self.quality,
            "channels_first": self.channels_first,
            "clip_values": self.clip_values,
        }


class GaussianAugmentationDefence(DefenceStrategy):
    """Gaussian augmentation preprocessor defence.

    Adds Gaussian noise to inputs at test time to smooth adversarial
    perturbations. Can also be used for training augmentation.
    """

    def __init__(
        self,
        sigma: float = 0.1,
        augmentation: bool = True,
        ratio: float = 1.0,
        clip_values: tuple = (0.0, 1.0),
    ):
        super().__init__(defence_name="GaussianAugmentation")
        self.sigma = sigma
        self.augmentation = augmentation
        self.ratio = ratio
        self.clip_values = clip_values
        self._preprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_PREPROCESSORS_AVAILABLE:
            raise ImportError("ART preprocessor defences not available.")
        self._preprocessor = ARTGaussianAugmentation(
            sigma=self.sigma,
            augmentation=self.augmentation,
            ratio=self.ratio,
            clip_values=self.clip_values,
        )
        if hasattr(art_estimator, 'set_params'):
            current = getattr(art_estimator, 'preprocessing_defences', None)
            defences = list(current) if current else []
            defences.append(self._preprocessor)
            art_estimator.set_params(preprocessing_defences=defences)
        return art_estimator

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._preprocessor is None:
            if not ART_PREPROCESSORS_AVAILABLE:
                raise ImportError("ART preprocessor defences not available.")
            self._preprocessor = ARTGaussianAugmentation(
                sigma=self.sigma,
                augmentation=self.augmentation,
                ratio=self.ratio,
                clip_values=self.clip_values,
            )
        result, _ = self._preprocessor(x)
        return result

    def get_params(self) -> Dict[str, Any]:
        return {
            "sigma": self.sigma,
            "augmentation": self.augmentation,
            "ratio": self.ratio,
            "clip_values": self.clip_values,
        }
