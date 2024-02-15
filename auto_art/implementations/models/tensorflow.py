"""
TensorFlow model implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, TypeVar
from ...core.base import BaseModel, ModelMetadata
from ...core.interfaces import ModelInterface

T = TypeVar('T')  # Generic type for model inputs/outputs

class TensorFlowModel(BaseModel[T], ModelInterface[T]):
    """TensorFlow model implementation with Template Method pattern."""
    
    def __init__(self):
        self.supported_extensions = {'.h5', '.pb', '.savedmodel'}
    
    def load_model(self, model_path: str) -> Tuple[Any, str]:
        """Load TensorFlow model from path."""
        if not any(model_path.endswith(ext) for ext in self.supported_extensions):
            raise ValueError(f"Unsupported model file extension. Supported: {self.supported_extensions}")
        
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            return model, 'tensorflow'
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow model: {str(e)}")
    
    def analyze_architecture(self, model: Any, framework: str) -> ModelMetadata:
        """Analyze TensorFlow model architecture."""
        if not hasattr(model, 'layers'):
            raise ValueError("Model must be a TensorFlow model with layers")
        
        layer_info = []
        for layer in model.layers:
            layer_info.append({
                'name': layer.name,
                'type': layer.__class__.__name__,
                'input_shape': layer.input_shape,
                'output_shape': layer.output_shape
            })
        
        return ModelMetadata(
            model_type=self._determine_model_type(model),
            framework='tensorflow',
            input_shape=model.input_shape[1:],
            output_shape=model.output_shape[1:],
            input_type=self._determine_input_type(model),
            output_type=self._determine_output_type(model),
            layer_info=layer_info,
            additional_info=self._get_additional_info(model)
        )
    
    def preprocess_input(self, input_data: T) -> T:
        """Preprocess input data for TensorFlow model."""
        import tensorflow as tf
        if isinstance(input_data, tf.Tensor):
            return input_data
        return tf.convert_to_tensor(input_data)
    
    def postprocess_output(self, output_data: T) -> T:
        """Postprocess output data from TensorFlow model."""
        import tensorflow as tf
        if isinstance(output_data, tf.Tensor):
            return output_data.numpy()
        return output_data
    
    def validate_model(self, model: Any) -> bool:
        """Validate TensorFlow model structure."""
        required_attrs = {'layers', 'input_shape', 'output_shape', 'predict'}
        return all(hasattr(model, attr) for attr in required_attrs)
    
    def _determine_model_type(self, model: Any) -> str:
        """Determine the type of TensorFlow model."""
        if hasattr(model, 'loss') and 'categorical_crossentropy' in str(model.loss):
            return 'classification'
        elif hasattr(model, 'loss') and 'mse' in str(model.loss):
            return 'regression'
        return 'unknown'
    
    def _determine_input_type(self, model: Any) -> str:
        """Determine the input type expected by the model."""
        if len(model.input_shape) == 4:  # (batch, height, width, channels)
            return 'image'
        elif len(model.input_shape) == 2:  # (batch, features)
            return 'tabular'
        return 'unknown'
    
    def _determine_output_type(self, model: Any) -> str:
        """Determine the output type produced by the model."""
        if len(model.output_shape) == 4:  # (batch, height, width, channels)
            return 'image'
        elif len(model.output_shape) == 2:  # (batch, features)
            return 'tabular'
        return 'unknown'
    
    def _get_additional_info(self, model: Any) -> Dict[str, Any]:
        """Get additional model information."""
        return {
            'num_layers': len(model.layers),
            'total_params': model.count_params(),
            'optimizer': model.optimizer.__class__.__name__ if hasattr(model, 'optimizer') else None,
            'loss': str(model.loss) if hasattr(model, 'loss') else None
        } 