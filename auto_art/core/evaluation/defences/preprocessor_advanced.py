"""
Advanced ART preprocessor defence wrappers.

Covers: LabelSmoothing, ThermometerEncoding, TotalVarianceMinimization,
VideoCompression, Mp3Compression, InverseGAN/DefenseGAN.
"""
import logging
from typing import Any

from .base import DefenceStrategy

logger = logging.getLogger(__name__)


class LabelSmoothingDefence(DefenceStrategy):
    """Label Smoothing — softens hard labels to reduce overconfidence."""

    def __init__(self, max_value: float = 0.9):
        super().__init__(defence_name="LabelSmoothing")
        self.max_value = max_value

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        from art.defences.preprocessor import LabelSmoothing
        preprocessor = LabelSmoothing(max_value=self.max_value)
        return self._apply_preprocessor(art_estimator, preprocessor)

    def get_params(self) -> dict[str, Any]:
        return {"max_value": self.max_value}

    @staticmethod
    def _apply_preprocessor(estimator: Any, preprocessor: Any) -> Any:
        existing = list(estimator.preprocessing_defences or [])
        existing.append(preprocessor)
        estimator.preprocessing_defences = existing
        return estimator


class ThermometerEncodingDefence(DefenceStrategy):
    """Thermometer Encoding — encodes inputs as thermometer codes."""

    def __init__(self, num_space: int = 10, clip_values: tuple = (0.0, 1.0)):
        super().__init__(defence_name="ThermometerEncoding")
        self.num_space = num_space
        self.clip_values = clip_values

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        from art.defences.preprocessor import ThermometerEncoding
        preprocessor = ThermometerEncoding(
            num_space=self.num_space, clip_values=self.clip_values,
        )
        existing = list(art_estimator.preprocessing_defences or [])
        existing.append(preprocessor)
        art_estimator.preprocessing_defences = existing
        return art_estimator

    def get_params(self) -> dict[str, Any]:
        return {"num_space": self.num_space, "clip_values": self.clip_values}


class TotalVarianceMinimizationDefence(DefenceStrategy):
    """Total Variance Minimization — TV denoising for adversarial purification."""

    def __init__(self, prob: float = 0.3, norm: int = 2, lam: float = 0.5,
                 solver: str = "L-BFGS-B", max_iter: int = 10,
                 clip_values: tuple = (0.0, 1.0)):
        super().__init__(defence_name="TotalVarianceMinimization")
        self.prob = prob
        self.norm = norm
        self.lam = lam
        self.solver = solver
        self.max_iter = max_iter
        self.clip_values = clip_values

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        from art.defences.preprocessor import TotalVarMin
        preprocessor = TotalVarMin(
            prob=self.prob, norm=self.norm, lam=self.lam,
            solver=self.solver, max_iter=self.max_iter,
            clip_values=self.clip_values,
        )
        existing = list(art_estimator.preprocessing_defences or [])
        existing.append(preprocessor)
        art_estimator.preprocessing_defences = existing
        return art_estimator

    def get_params(self) -> dict[str, Any]:
        return {"prob": self.prob, "norm": self.norm, "lam": self.lam,
                "solver": self.solver, "max_iter": self.max_iter}


class VideoCompressionDefence(DefenceStrategy):
    """Video Compression — uses video codec to remove adversarial perturbations."""

    def __init__(self, video_format: str = "avi",
                 constant_rate_factor: int = 28,
                 clip_values: tuple = (0.0, 1.0)):
        super().__init__(defence_name="VideoCompression")
        self.video_format = video_format
        self.constant_rate_factor = constant_rate_factor
        self.clip_values = clip_values

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        from art.defences.preprocessor import VideoCompression
        preprocessor = VideoCompression(
            video_format=self.video_format,
            constant_rate_factor=self.constant_rate_factor,
            clip_values=self.clip_values,
        )
        existing = list(art_estimator.preprocessing_defences or [])
        existing.append(preprocessor)
        art_estimator.preprocessing_defences = existing
        return art_estimator

    def get_params(self) -> dict[str, Any]:
        return {"video_format": self.video_format,
                "constant_rate_factor": self.constant_rate_factor}


class Mp3CompressionDefence(DefenceStrategy):
    """MP3 Compression — uses lossy audio compression to remove perturbations."""

    def __init__(self, sample_rate: int = 16000,
                 clip_values: tuple = (0.0, 1.0)):
        super().__init__(defence_name="Mp3Compression")
        self.sample_rate = sample_rate
        self.clip_values = clip_values

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        from art.defences.preprocessor import Mp3Compression
        preprocessor = Mp3Compression(
            sample_rate=self.sample_rate, clip_values=self.clip_values,
        )
        existing = list(art_estimator.preprocessing_defences or [])
        existing.append(preprocessor)
        art_estimator.preprocessing_defences = existing
        return art_estimator

    def get_params(self) -> dict[str, Any]:
        return {"sample_rate": self.sample_rate}
