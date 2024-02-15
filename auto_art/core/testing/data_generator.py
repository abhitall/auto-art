"""
Data generator module for creating test inputs based on model type.
"""
import sys
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass

print("Starting data_generator.py initialization", file=sys.stderr)

# Only import frameworks when needed, not at module level
_FRAMEWORKS = {
    'torch': None,
    'tensorflow': None,
    'transformers': None
}

def _import_framework(name: str):
    """Import a framework only when needed."""
    if name not in _FRAMEWORKS or _FRAMEWORKS[name] is not None:
        return _FRAMEWORKS.get(name)
    
    try:
        if name == 'torch':
            import torch
            _FRAMEWORKS['torch'] = torch
        elif name == 'tensorflow':
            import tensorflow
            _FRAMEWORKS['tensorflow'] = tensorflow
        elif name == 'transformers':
            from transformers import AutoModel, AutoTokenizer
            _FRAMEWORKS['transformers'] = (AutoModel, AutoTokenizer)
    except ImportError:
        print(f"Failed to import {name}", file=sys.stderr)
    
    return _FRAMEWORKS.get(name)

@dataclass
class TestData:
    """Container for generated test data."""
    inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    expected_outputs: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

print("TestData class defined", file=sys.stderr)

class DataGenerator:
    """Generates appropriate test data based on model type and input requirements."""
    
    def __init__(self):
        print("Initializing DataGenerator", file=sys.stderr)
        self.supported_input_types = ['image', 'text', 'multimodal']

    def generate_data(self, metadata: Any, num_samples: int = 100) -> TestData:
        """Generates test data appropriate for the model type and input format."""
        print(f"Generating {metadata.input_type} data", file=sys.stderr)
        
        # Validate input parameters
        if num_samples <= 0:
            raise ValueError(f"Number of samples must be positive, got {num_samples}")
        
        if not metadata.input_shape or (isinstance(metadata.input_shape, tuple) and not metadata.input_shape):
            raise ValueError("Input shape cannot be empty")
        
        if metadata.input_type == 'image':
            return self._generate_image_data(metadata, num_samples)
        elif metadata.input_type == 'text':
            return self._generate_text_data(metadata, num_samples)
        elif metadata.input_type == 'multimodal':
            return self._generate_multimodal_data(metadata, num_samples)
        else:
            raise ValueError(f"Unsupported input type: {metadata.input_type}")
    
    def _generate_image_data(self, metadata: Any, num_samples: int) -> TestData:
        """Generates synthetic image data matching the model's input requirements."""
        input_shape = (num_samples,) + metadata.input_shape
        images = np.random.uniform(0, 1, input_shape)
        images = (images - images.min()) / (images.max() - images.min())
        
        return TestData(
            inputs=images,
            metadata={
                'input_type': 'image',
                'normalization': 'minmax',
                'shape': input_shape
            }
        )
    
    def _generate_text_data(self, metadata: Any, num_samples: int) -> TestData:
        """Generates synthetic text data matching the model's input requirements."""
        if metadata.framework == 'transformers':
            vocab_size = metadata.input_shape[-1]
            sequence_length = metadata.input_shape[0]
            tokens = np.random.randint(0, vocab_size, (num_samples, sequence_length))
            return TestData(
                inputs=tokens,
                metadata={
                    'input_type': 'text',
                    'sequence_length': sequence_length,
                    'vocab_size': vocab_size
                }
            )
        else:
            embedding_dim = metadata.input_shape[-1]
            sequence_length = metadata.input_shape[0]
            embeddings = np.random.randn(num_samples, sequence_length, embedding_dim)
            return TestData(
                inputs=embeddings,
                metadata={
                    'input_type': 'text',
                    'sequence_length': sequence_length,
                    'embedding_dim': embedding_dim
                }
            )
    
    def _generate_multimodal_data(self, metadata: Any, num_samples: int) -> TestData:
        """Generates synthetic multimodal data (e.g., image + text pairs)."""
        from ..analysis.model_analyzer import ModelMetadata
        
        image_metadata = ModelMetadata(
            input_type='image',
            input_shape=metadata.input_shape[0],
            framework=metadata.framework
        )
        text_metadata = ModelMetadata(
            input_type='text',
            input_shape=metadata.input_shape[1],
            framework=metadata.framework
        )
        
        image_data = self._generate_image_data(image_metadata, num_samples)
        text_data = self._generate_text_data(text_metadata, num_samples)
        
        # Get text shape from the appropriate metadata field
        text_shape = text_data.inputs.shape
        
        return TestData(
            inputs={
                'image': image_data.inputs,
                'text': text_data.inputs
            },
            metadata={
                'input_type': 'multimodal',
                'components': ['image', 'text'],
                'image_shape': image_data.metadata['shape'],
                'text_shape': text_shape
            }
        )