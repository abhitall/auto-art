"""
ART Postprocessor defence wrappers.

Wraps ART's postprocessor defences (Reverse Sigmoid, High Confidence,
Gaussian Noise, Class Labels) into the auto-art DefenceStrategy interface.

These defences modify the model's output predictions to reduce information
leakage that enables extraction and inference attacks.

Reference: ART Defences Wiki - https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/ART-Defences
"""

from typing import Any, Dict, Optional
import logging

from .base import DefenceStrategy

logger = logging.getLogger(__name__)

try:
    from art.defences.postprocessor import (
        ReverseSigmoid as ARTReverseSigmoid,
        HighConfidence as ARTHighConfidence,
        GaussianNoise as ARTGaussianNoise,
        ClassLabels as ARTClassLabels,
    )
    ART_POSTPROCESSORS_AVAILABLE = True
except ImportError:
    ART_POSTPROCESSORS_AVAILABLE = False


class ReverseSigmoidDefence(DefenceStrategy):
    """Reverse Sigmoid postprocessor defence (Lee et al., 2018).

    Applies a reverse sigmoid perturbation to model output probabilities,
    reducing the confidence information available to adversaries attempting
    model extraction.
    """

    def __init__(
        self,
        beta: float = 1.0,
        gamma: float = 0.1,
        clip_values: tuple = (0.0, 1.0),
    ):
        super().__init__(defence_name="ReverseSigmoid")
        self.beta = beta
        self.gamma = gamma
        self.clip_values = clip_values
        self._postprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_POSTPROCESSORS_AVAILABLE:
            raise ImportError("ART postprocessor defences not available.")
        self._postprocessor = ARTReverseSigmoid(
            beta=self.beta,
            gamma=self.gamma,
        )
        if hasattr(art_estimator, 'set_params'):
            current = getattr(
                art_estimator, 'postprocessing_defences', None
            )
            defences = list(current) if current else []
            defences.append(self._postprocessor)
            art_estimator.set_params(postprocessing_defences=defences)
        return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "gamma": self.gamma,
            "clip_values": self.clip_values,
        }


class HighConfidenceDefence(DefenceStrategy):
    """High Confidence postprocessor defence (Tramer et al., 2016).

    Only returns predictions with high confidence, replacing low-confidence
    predictions with uniform distributions to prevent model extraction.
    """

    def __init__(self, cutoff: float = 0.25):
        super().__init__(defence_name="HighConfidence")
        self.cutoff = cutoff
        self._postprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_POSTPROCESSORS_AVAILABLE:
            raise ImportError("ART postprocessor defences not available.")
        self._postprocessor = ARTHighConfidence(cutoff=self.cutoff)
        if hasattr(art_estimator, 'set_params'):
            current = getattr(
                art_estimator, 'postprocessing_defences', None
            )
            defences = list(current) if current else []
            defences.append(self._postprocessor)
            art_estimator.set_params(postprocessing_defences=defences)
        return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {"cutoff": self.cutoff}


class GaussianNoiseDefence(DefenceStrategy):
    """Gaussian Noise postprocessor defence (Chandrasekaran et al., 2018).

    Adds Gaussian noise to model output probabilities to reduce
    information leakage while preserving classification accuracy.
    """

    def __init__(self, scale: float = 0.1):
        super().__init__(defence_name="GaussianNoise")
        self.scale = scale
        self._postprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_POSTPROCESSORS_AVAILABLE:
            raise ImportError("ART postprocessor defences not available.")
        self._postprocessor = ARTGaussianNoise(scale=self.scale)
        if hasattr(art_estimator, 'set_params'):
            current = getattr(
                art_estimator, 'postprocessing_defences', None
            )
            defences = list(current) if current else []
            defences.append(self._postprocessor)
            art_estimator.set_params(postprocessing_defences=defences)
        return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {"scale": self.scale}


class ClassLabelsDefence(DefenceStrategy):
    """Class Labels postprocessor defence (Tramer et al., 2016).

    Returns only the predicted class label instead of full probability
    vectors, eliminating confidence information available to adversaries.
    """

    def __init__(self):
        super().__init__(defence_name="ClassLabels")
        self._postprocessor: Optional[Any] = None

    def apply(self, art_estimator: Any, **kwargs) -> Any:
        if not ART_POSTPROCESSORS_AVAILABLE:
            raise ImportError("ART postprocessor defences not available.")
        self._postprocessor = ARTClassLabels()
        if hasattr(art_estimator, 'set_params'):
            current = getattr(
                art_estimator, 'postprocessing_defences', None
            )
            defences = list(current) if current else []
            defences.append(self._postprocessor)
            art_estimator.set_params(postprocessing_defences=defences)
        return art_estimator

    def get_params(self) -> Dict[str, Any]:
        return {}
