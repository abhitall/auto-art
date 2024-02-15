"""
PyTorch model implementation.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, TypeVar
from ...core.base import BaseModel, ModelMetadata
from ...core.interfaces import ModelInterface

T = TypeVar('T')  # Generic type for model inputs/outputs

class PyTorchModel(BaseModel[T], ModelInterface[T]):
    """PyTorch model implementation with Template Method pattern."""
    
    def __init__(self):
        self.supported_extensions = {'.pt', '.pth', '.ckpt'}
    
    def load_model(self, model_path: str) -> Tuple[Any, str]:
        """Load PyTorch model from path."""
        if not any(model_path.endswith(ext) for ext in self.supported_extensions):
            raise ValueError(f"Unsupported model file extension. Supported: {self.supported_extensions}")
        
        try:
            model = torch.load(model_path)
            if not isinstance(model, nn.Module):
                raise ValueError("Loaded file is not a PyTorch model")
            return model, 'pytorch'
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {str(e)}")
    
    def analyze_architecture(self, model: Any, framework: str) -> ModelMetadata:
        """Analyze PyTorch model architecture."""
        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a PyTorch module")
        
        layer_info = []
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer_info.append({
                    'name': name,
                    'type': layer.__class__.__name__,
                    'in_features': getattr(layer, 'in_features', None),
                    'out_features': getattr(layer, 'out_features', None)
                })
        
        # Determine input/output shapes from first/last layers
        try:
            first_param = next(model.parameters())
            last_param = list(model.parameters())[-1]
            input_shape = first_param.shape
            output_shape = last_param.shape
        except StopIteration:
            # Model has no parameters, use default shapes
            input_shape = (1,)  # Default input shape
            output_shape = (1,)  # Default output shape
        
        return ModelMetadata(
            model_type=self._determine_model_type(model),
            framework='pytorch',
            input_shape=input_shape,
            output_shape=output_shape,
            input_type=self._determine_input_type(model),
            output_type=self._determine_output_type(model),
            layer_info=layer_info,
            additional_info=self._get_additional_info(model)
        )
    
    def preprocess_input(self, input_data: T) -> T:
        """Preprocess input data for PyTorch model."""
        if isinstance(input_data, torch.Tensor):
            return input_data
        elif isinstance(input_data, (list, tuple)):
            return torch.stack([torch.tensor(x) for x in input_data])
        else:
            return torch.tensor(input_data)
    
    def postprocess_output(self, output_data: T) -> T:
        """Postprocess output data from PyTorch model."""
        if isinstance(output_data, torch.Tensor):
            return output_data.detach().cpu().numpy()
        return output_data
    
    def validate_model(self, model: Any) -> bool:
        """Validate PyTorch model structure."""
        if not isinstance(model, nn.Module):
            return False
        
        # Check if model has required methods
        required_methods = {'forward', 'parameters', 'named_modules'}
        return all(hasattr(model, method) for method in required_methods)
    
    def _determine_model_type(self, model: nn.Module) -> str:
        """Determine the type of PyTorch model."""
        # Check for classification models
        if any(isinstance(layer, nn.Sigmoid) for layer in model.modules()):
            return 'classification'
        
        # Check for regression models
        if any(isinstance(layer, nn.Linear) for layer in model.modules()):
            return 'regression'
        
        # Check for generative models
        if 'generator' in model.__class__.__name__.lower():
            return 'generator'
        
        return 'unknown'
    
    def _determine_input_type(self, model: nn.Module) -> str:
        """Determine the input type expected by the model."""
        try:
            # Check first layer type
            first_param = next(model.parameters())
            if len(first_param.shape) == 3:  # Assuming (batch, height, channels)
                return 'image'
            elif len(first_param.shape) == 2:  # Assuming (batch, sequence)
                return 'text'
        except StopIteration:
            pass
        return 'unknown'
    
    def _determine_output_type(self, model: nn.Module) -> str:
        """Determine the output type produced by the model."""
        try:
            # Check last layer type
            last_param = list(model.parameters())[-1]
            if len(last_param.shape) == 3:  # Assuming (batch, height, channels)
                return 'image'
            elif len(last_param.shape) == 2:  # Assuming (batch, sequence)
                return 'text'
        except (StopIteration, IndexError):
            pass
        return 'unknown'
    
    def _get_additional_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get additional model information."""
        try:
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            device = next(model.parameters()).device.type
        except StopIteration:
            num_params = 0
            num_trainable = 0
            device = 'cpu'
        
        return {
            'num_parameters': num_params,
            'num_trainable_parameters': num_trainable,
            'device': device,
            'model_class': model.__class__.__name__
        } 