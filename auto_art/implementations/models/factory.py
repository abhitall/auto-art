"""
Factory for creating model implementations.
"""

from typing import Dict, Type, Optional
from ...core.base import BaseModel
from ...core.interfaces import ModelInterface
from .pytorch import PyTorchModel
from .tensorflow import TensorFlowModel
from .transformers import TransformersModel

class ModelFactory:
    """Factory class for creating model implementations."""
    
    _implementations: Dict[str, Type[BaseModel]] = {
        'pytorch': PyTorchModel,
        'tensorflow': TensorFlowModel,
        'transformers': TransformersModel
    }
    
    @classmethod
    def create_model(cls, framework: str) -> Optional[BaseModel]:
        """Create a model implementation for the specified framework."""
        if framework not in cls._implementations:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return cls._implementations[framework]()
    
    @classmethod
    def register_implementation(cls, framework: str, implementation: Type[BaseModel]) -> None:
        """Register a new model implementation."""
        if not issubclass(implementation, (BaseModel, ModelInterface)):
            raise TypeError("Implementation must inherit from BaseModel and ModelInterface")
        cls._implementations[framework] = implementation
    
    @classmethod
    def get_supported_frameworks(cls) -> list:
        """Get list of supported frameworks."""
        return list(cls._implementations.keys())
    
    @classmethod
    def get_implementation(cls, framework: str) -> Optional[Type[BaseModel]]:
        """Get the implementation class for a framework."""
        return cls._implementations.get(framework) 