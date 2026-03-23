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
from .preprocessor_augmentation import (
    CutoutDefence, MixupDefence, CutMixDefence,
)
from .postprocessor import (
    ReverseSigmoidDefence, HighConfidenceDefence,
    GaussianNoiseDefence, ClassLabelsDefence,
)
from .postprocessor_rounding import RoundingDefence
from .trainer import AdversarialTrainingPGDDefence
from .trainer_fbf import FastIsBetterThanFreeDefence
from .trainer_certified import (
    CertifiedAdversarialTrainingDefence, IntervalBoundPropagationDefence,
)
from .trainer_oaat import OAATDefence
from .detector import (
    ActivationDefenceWrapper, SpectralSignatureDefenceWrapper,
    PoisonDetectionReport,
)
from .detector_evasion import (
    BasicInputDetectorWrapper, ActivationDetectorWrapper,
    SubsetScanDetectorWrapper,
)
from .detector_beyond import BEYONDDetectorWrapper
from .detector_provenance import DataProvenanceDefenceWrapper, RONIDefenceWrapper
from .transformer_cleanse import NeuralCleanseDefence, STRIPDefence
from .transformer_distillation import DefensiveDistillationDefence
from .trainer_trades import TRADESDefence
from .trainer_awp import AWPDefence
from .preprocessor_advanced import (
    LabelSmoothingDefence, ThermometerEncodingDefence,
    TotalVarianceMinimizationDefence, VideoCompressionDefence,
    Mp3CompressionDefence,
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
    "CutoutDefence",
    "MixupDefence",
    "CutMixDefence",
    "ReverseSigmoidDefence",
    "HighConfidenceDefence",
    "GaussianNoiseDefence",
    "ClassLabelsDefence",
    "RoundingDefence",
    "AdversarialTrainingPGDDefence",
    "FastIsBetterThanFreeDefence",
    "CertifiedAdversarialTrainingDefence",
    "IntervalBoundPropagationDefence",
    "OAATDefence",
    "ActivationDefenceWrapper",
    "SpectralSignatureDefenceWrapper",
    "PoisonDetectionReport",
    "BasicInputDetectorWrapper",
    "ActivationDetectorWrapper",
    "SubsetScanDetectorWrapper",
    "BEYONDDetectorWrapper",
    "DataProvenanceDefenceWrapper",
    "RONIDefenceWrapper",
    "NeuralCleanseDefence",
    "STRIPDefence",
    "DefensiveDistillationDefence",
    "TRADESDefence",
    "AWPDefence",
    "LabelSmoothingDefence",
    "ThermometerEncodingDefence",
    "TotalVarianceMinimizationDefence",
    "VideoCompressionDefence",
    "Mp3CompressionDefence",
]
