"""
Model analyzer module for extracting metadata and properties from ML models.
"""
import sys
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass

print("Starting model_analyzer.py initialization", file=sys.stderr)

# Framework imports will be loaded on demand
_FRAMEWORKS = {}

def _load_framework(name: str) -> Any:
    """Load a machine learning framework on demand."""
    if name in _FRAMEWORKS:
        return _FRAMEWORKS[name]
    
    try:
        if name == 'pytorch':
            import torch
            _FRAMEWORKS['pytorch'] = torch
        elif name == 'tensorflow':
            import tensorflow
            _FRAMEWORKS['tensorflow'] = tensorflow
        elif name == 'transformers':
            from transformers import AutoModel, AutoTokenizer
            _FRAMEWORKS['transformers'] = (AutoModel, AutoTokenizer)
        return _FRAMEWORKS[name]
    except ImportError:
        print(f"Failed to import {name}", file=sys.stderr)
        return None

@dataclass
class ModelMetadata:
    """Container for model metadata."""
    input_type: str  # 'image', 'text', 'multimodal'
    input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
    framework: str   # 'pytorch', 'tensorflow', 'transformers'
    model_type: str = 'classifier'  # 'classification', 'regression', 'generator', 'llm'
    output_shape: Optional[Tuple[int, ...]] = None
    output_type: Optional[str] = None  # 'image', 'text', 'multimodal', 'logits', 'embeddings'
    layer_info: Optional[List[Dict[str, Any]]] = None

print("ModelMetadata defined", file=sys.stderr)

def analyze_model(model: Any, framework: str) -> ModelMetadata:
    """Analyzes a model to determine its metadata."""
    # Only load the framework when needed for analysis
    framework_module = _load_framework(framework)
    if not framework_module:
        return ModelMetadata(
            input_type='unknown',
            input_shape=(1,),
            framework=framework
        )
    
    # Basic analysis without loading full framework features
    return ModelMetadata(
        input_type='unknown',
        input_shape=(1,),
        framework=framework
    )