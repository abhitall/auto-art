# This file makes the 'config' directory a Python package.
# It can also be used to expose certain classes or functions at the package level.

from .evaluation_config import EvaluationConfig, EvaluationResult, ModelType, Framework, EvaluationBuilder

__all__ = [
    "EvaluationConfig",
    "EvaluationResult",
    "ModelType",
    "Framework",
    "EvaluationBuilder",
]
