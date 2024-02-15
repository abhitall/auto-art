"""
Test data generator module for creating appropriate test inputs based on model type.
"""

import numpy as np
import torch
try:
    import tensorflow as tf
except ImportError:
    tf = None
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from ..analysis.model_analyzer import ModelMetadata

@dataclass
class TestData:
    """Container for generated test data."""
    inputs: np.ndarray
    expected_outputs: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

class TestDataGenerator:
    """Generates appropriate test data based on model type and input requirements."""
    
    def __init__(self):
        self.supported_input_types = ['image', 'text', 'multimodal']
    
    def generate_test_data(self, model_metadata: ModelMetadata, num_samples: int = 100) -> TestData:
        """Generates test data appropriate for the model type and input format."""
        if model_metadata.input_type == 'image':
            return self._generate_image_data(model_metadata, num_samples)
        elif model_metadata.input_type == 'text':
            return self._generate_text_data(model_metadata, num_samples)
        elif model_metadata.input_type == 'multimodal':
            return self._generate_multimodal_data(model_metadata, num_samples)
        else:
            raise ValueError(f"Unsupported input type: {model_metadata.input_type}")
    
    def _generate_image_data(self, metadata: ModelMetadata, num_samples: int) -> TestData:
        """Generates synthetic image data matching the model's input requirements."""
        # Create random images with appropriate shape and normalization
        input_shape = (num_samples,) + metadata.input_shape
        if len(metadata.input_shape) == 3:  # RGB image
            images = np.random.uniform(0, 1, input_shape)
            # Normalize to [0, 1] range
            images = (images - images.min()) / (images.max() - images.min())
        else:
            images = np.random.randn(*input_shape)
        
        return TestData(
            inputs=images,
            metadata={
                'input_type': 'image',
                'normalization': 'minmax',
                'shape': input_shape
            }
        )
    
    def _generate_text_data(self, metadata: ModelMetadata, num_samples: int) -> TestData:
        """Generates synthetic text data matching the model's input requirements."""
        # For text models, we'll generate random token sequences
        if metadata.framework == 'transformers':
            # Use a simple vocabulary for demonstration
            vocab_size = metadata.input_shape[-1]
            sequence_length = metadata.input_shape[0]
            
            # Generate random token sequences
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
            # For other frameworks, generate random embeddings
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
    
    def _generate_multimodal_data(self, metadata: ModelMetadata, num_samples: int) -> TestData:
        """Generates synthetic multimodal data (e.g., image + text pairs)."""
        # This is a simplified version - in practice, you'd need to handle
        # specific multimodal architectures and their requirements
        
        # Generate both image and text components
        image_data = self._generate_image_data(metadata, num_samples)
        text_data = self._generate_text_data(metadata, num_samples)
        
        # Combine the data
        combined_inputs = {
            'image': image_data.inputs,
            'text': text_data.inputs
        }
        
        return TestData(
            inputs=combined_inputs,
            metadata={
                'input_type': 'multimodal',
                'components': ['image', 'text'],
                'image_shape': image_data.metadata['shape'],
                'text_shape': text_data.metadata['shape']
            }
        )
    
    def generate_expected_outputs(self, model: Any, test_data: TestData) -> np.ndarray:
        """Generates expected outputs for the test data using the model."""
        if isinstance(model, tuple) and len(model) == 2:  # Transformers model with tokenizer
            model, tokenizer = model
        else:
            tokenizer = None
        
        with torch.no_grad():
            if isinstance(test_data.inputs, dict):  # Multimodal input
                outputs = model(**test_data.inputs)
            else:
                if tokenizer is not None:
                    # Handle text inputs with tokenizer
                    outputs = model(**tokenizer(test_data.inputs, return_tensors="pt"))
                else:
                    # Handle regular inputs
                    outputs = model(torch.from_numpy(test_data.inputs))
            
            # Convert outputs to numpy array
            if isinstance(outputs, torch.Tensor):
                return outputs.numpy()
            elif isinstance(outputs, tf.Tensor):
                return outputs.numpy()
            else:
                # Handle other output types (e.g., transformers outputs)
                return outputs.logits.numpy()