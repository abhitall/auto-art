"""
Factory for creating model implementations.
"""

from typing import Dict, Type, Optional
from ...core.base import BaseModel
from ...core.interfaces import ModelInterface
from .pytorch import PyTorchModel
from .tensorflow import TensorFlowModel
from .transformers import TransformersModel
from .keras import KerasModel
from .sklearn import SklearnModel
from .mxnet import MXNetModel # Added
from .xgboost import XGBoostModel # Added
from .lightgbm import LightGBMModel # Added
from .catboost import CatBoostModel # Added
from .gpy import GPyModel # Added

class ModelFactory:
    """Factory class for creating model implementations."""

    _implementations: Dict[str, Type[BaseModel]] = {
        'pytorch': PyTorchModel,
        'tensorflow': TensorFlowModel,
        'transformers': TransformersModel,
        'keras': KerasModel,
        'sklearn': SklearnModel,
        'mxnet': MXNetModel, # Added
        'xgboost': XGBoostModel, # Added
        'lightgbm': LightGBMModel, # Added
        'catboost': CatBoostModel, # Added
        'gpy': GPyModel # Added
    }

    @classmethod
    def create_model(cls, framework: str) -> Optional[BaseModel]:
        """Create a model implementation for the specified framework."""
        framework_key = framework.lower()

        if framework_key not in cls._implementations:
            if framework_key == 'tf.keras':
                if 'keras' in cls._implementations:
                    framework_key = 'keras'
                elif 'tensorflow' in cls._implementations:
                    framework_key = 'tensorflow'

        if framework_key not in cls._implementations:
            raise ValueError(f"Unsupported framework: {framework}. Supported: {list(cls._implementations.keys())}")

        return cls._implementations[framework_key]()

    @classmethod
    def register_implementation(cls, framework: str, implementation: Type[BaseModel]) -> None:
        """Register a new model implementation. Framework name is stored in lowercase."""
        if not issubclass(implementation, BaseModel):
            raise TypeError("Implementation must inherit from BaseModel")
        cls._implementations[framework.lower()] = implementation

    @classmethod
    def get_supported_frameworks(cls) -> list:
        """Get list of supported frameworks."""
        return list(cls._implementations.keys())

    @classmethod
    def get_implementation(cls, framework: str) -> Optional[Type[BaseModel]]:
        """Get the implementation class for a framework (case-insensitive)."""
        return cls._implementations.get(framework.lower())
