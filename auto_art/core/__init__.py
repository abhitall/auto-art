"""
Core functionality for the Auto-ART framework.
"""
# Import core classes and functions only when needed to prevent circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis.model_analyzer import ModelMetadata
    from .testing.data_generator import DataGenerator, TestData

from .base import BaseModel, BaseAttack, BaseEvaluator, BaseTestGenerator
from .interfaces import ModelInterface, AttackInterface, EvaluatorInterface

__all__ = [
    'BaseModel',
    'BaseAttack',
    'BaseEvaluator',
    'BaseTestGenerator',
    'ModelInterface',
    'AttackInterface',
    'EvaluatorInterface'
]