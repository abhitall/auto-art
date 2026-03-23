"""
Model implementations for different frameworks.

All imports are lazy to avoid hard dependencies on optional frameworks
like PyTorch, TensorFlow, etc.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pytorch import PyTorchModel
    from .tensorflow import TensorFlowModel
    from .transformers import TransformersModel

__all__ = [
    "PyTorchModel",
    "TensorFlowModel",
    "TransformersModel",
]


def __getattr__(name: str):
    if name == "PyTorchModel":
        from .pytorch import PyTorchModel
        return PyTorchModel
    if name == "TensorFlowModel":
        from .tensorflow import TensorFlowModel
        return TensorFlowModel
    if name == "TransformersModel":
        from .transformers import TransformersModel
        return TransformersModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
