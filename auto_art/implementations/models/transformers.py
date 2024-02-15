"""
Transformers model implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, TypeVar
from ...core.base import BaseModel, ModelMetadata
from ...core.interfaces import ModelInterface

T = TypeVar('T')  # Generic type for model inputs/outputs

class TransformersModel(BaseModel[T], ModelInterface[T]):
    """Transformers model implementation with Template Method pattern."""
    
    def __init__(self):
        self.supported_extensions = {'.bin', '.pt', '.pth'}
    
    def load_model(self, model_path: str) -> Tuple[Any, str]:
        """Load Transformers model from path."""
        if not any(model_path.endswith(ext) for ext in self.supported_extensions):
            raise ValueError(f"Unsupported model file extension. Supported: {self.supported_extensions}")
        
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_path)
            return model, 'transformers'
        except Exception as e:
            raise RuntimeError(f"Failed to load Transformers model: {str(e)}")
    
    def analyze_architecture(self, model: Any, framework: str) -> ModelMetadata:
        """Analyze Transformers model architecture."""
        if not hasattr(model, 'config'):
            raise ValueError("Model must be a Transformers model with config")
        
        layer_info = []
        for name, layer in model.named_modules():
            if hasattr(layer, 'in_features') and hasattr(layer, 'out_features'):
                layer_info.append({
                    'name': name,
                    'type': layer.__class__.__name__,
                    'in_features': getattr(layer, 'in_features', None),
                    'out_features': getattr(layer, 'out_features', None)
                })
        
        return ModelMetadata(
            model_type=self._determine_model_type(model),
            framework='transformers',
            input_shape=model.config.hidden_size,
            output_shape=model.config.hidden_size,
            input_type='text',
            output_type='text',
            layer_info=layer_info,
            additional_info=self._get_additional_info(model)
        )
    
    def preprocess_input(self, input_data: T) -> T:
        """Preprocess input data for Transformers model."""
        return input_data
    
    def postprocess_output(self, output_data: T) -> T:
        """Postprocess output data from Transformers model."""
        return output_data
    
    def validate_model(self, model: Any) -> bool:
        """Validate Transformers model structure."""
        required_attrs = {'config', 'forward', 'named_modules'}
        return all(hasattr(model, attr) for attr in required_attrs)
    
    def _determine_model_type(self, model: Any) -> str:
        """Determine the type of Transformers model."""
        if hasattr(model.config, 'is_encoder_decoder') and model.config.is_encoder_decoder:
            return 'sequence-to-sequence'
        return 'transformer'
    
    def _get_additional_info(self, model: Any) -> Dict[str, Any]:
        """Get additional model information."""
        return {
            'model_type': model.config.model_type,
            'vocab_size': model.config.vocab_size,
            'hidden_size': model.config.hidden_size,
            'num_attention_heads': model.config.num_attention_heads,
            'num_hidden_layers': model.config.num_hidden_layers
        } 