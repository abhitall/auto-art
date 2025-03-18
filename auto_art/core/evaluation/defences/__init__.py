# This file makes the 'defences' directory a Python package.

from .base import DefenceStrategy
from .input_sanitizer import (
    InputSanitizer, DOMSanitizer, VisualDenoiser,
    SemanticNormalizer, SanitizationResult,
)
from .rag_poisoning_detector import RAGPoisoningDetector, RAGDetectionReport
from .in_context_defence import (
    InContextDefence, InContextDefenceLibrary, DefenceExemplar,
)
from .circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState, AgentStateSnapshot,
)
from .preprocessor import (
    SpatialSmoothingDefence, FeatureSqueezingDefence,
    JpegCompressionDefence, GaussianAugmentationDefence,
)
from .postprocessor import (
    ReverseSigmoidDefence, HighConfidenceDefence,
    GaussianNoiseDefence, ClassLabelsDefence,
)
from .trainer import AdversarialTrainingPGDDefence
from .detector import (
    ActivationDefenceWrapper, SpectralSignatureDefenceWrapper,
    PoisonDetectionReport,
)

__all__ = [
    "DefenceStrategy",
    "InputSanitizer",
    "DOMSanitizer",
    "VisualDenoiser",
    "SemanticNormalizer",
    "SanitizationResult",
    "RAGPoisoningDetector",
    "RAGDetectionReport",
    "InContextDefence",
    "InContextDefenceLibrary",
    "DefenceExemplar",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "AgentStateSnapshot",
    "SpatialSmoothingDefence",
    "FeatureSqueezingDefence",
    "JpegCompressionDefence",
    "GaussianAugmentationDefence",
    "ReverseSigmoidDefence",
    "HighConfidenceDefence",
    "GaussianNoiseDefence",
    "ClassLabelsDefence",
    "AdversarialTrainingPGDDefence",
    "ActivationDefenceWrapper",
    "SpectralSignatureDefenceWrapper",
    "PoisonDetectionReport",
]
