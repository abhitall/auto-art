"""
Keras model implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from ...core.base import BaseModel, ModelMetadata
import numpy as np # Added for preprocess_input if standalone Keras

T = TypeVar('T')

class KerasModel(BaseModel[T]):
    """Keras model implementation."""

    def __init__(self):
        self.supported_extensions = {'.h5', '.keras'}
        self._tf_module = None # Store the imported module (tf or keras)
        self._model_instance: Optional[Any] = None


    def _ensure_keras_module(self):
        """Ensures TensorFlow or standalone Keras is imported and returns the Keras API entry point."""
        if self._tf_module is None:
            try:
                import tensorflow
                self._tf_module = tensorflow
                if not hasattr(self._tf_module, 'keras'):
                     raise ImportError("TensorFlow installed but tf.keras not found.")
                return self._tf_module.keras # Return tf.keras
            except ImportError:
                try:
                    import keras # Standalone Keras
                    self._tf_module = keras
                    return self._tf_module # Return keras
                except ImportError:
                    raise RuntimeError("TensorFlow or Keras not found. Please install TensorFlow or Keras.")

        # Module already imported, return the correct Keras API entry
        if hasattr(self._tf_module, 'keras'): # It's TensorFlow
            return self._tf_module.keras
        else: # It's standalone Keras
            return self._tf_module


    def load_model(self, model_path: str) -> Tuple[Any, str]:
        """Load Keras model from path."""
        if not any(model_path.endswith(ext) for ext in self.supported_extensions):
            raise ValueError(f"Unsupported Keras model file extension. Supported: {self.supported_extensions}")

        keras_api = self._ensure_keras_module()
        try:
            self._model_instance = keras_api.models.load_model(model_path)
            return self._model_instance, 'keras'
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {str(e)}")

    def analyze_architecture(self, model: Any, framework: str) -> ModelMetadata:
        """Analyze Keras model architecture."""
        current_model = model if model is not None else self._model_instance
        if current_model is None: raise ValueError("Model not loaded or provided.")

        keras_api = self._ensure_keras_module()

        if not isinstance(current_model, keras_api.Model):
            raise ValueError(f"Model must be a Keras Model instance. Got {type(current_model)}")

        layer_info_list = []
        try:
            for layer in current_model.layers:
                config = {}
                try: config = layer.get_config()
                except: pass # Some custom layers might not serialize config well

                layer_info_list.append({
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'input_shape': getattr(layer, 'input_shape', None),
                    'output_shape': getattr(layer, 'output_shape', None),
                    'config': config,
                    'trainable': layer.trainable, # Added
                    'num_params': layer.count_params() if hasattr(layer, 'count_params') else 0 # Added
                })
        except AttributeError:
            pass

        raw_input_shape = current_model.input_shape
        raw_output_shape = current_model.output_shape

        def normalize_shape(shape_val):
            if isinstance(shape_val, list): # Multi-input/output
                 return tuple(s[1:] if isinstance(s, (list,tuple)) and len(s) > 1 and s[0] is None else s for s in shape_val)
            elif isinstance(shape_val, tuple) and len(shape_val) > 1 and shape_val[0] is None: # Single input/output with batch
                 return shape_val[1:]
            return shape_val if isinstance(shape_val, tuple) else (shape_val,) # Ensure tuple, keep as is if no batch or already processed

        input_shape_tup = normalize_shape(raw_input_shape)
        output_shape_tup = normalize_shape(raw_output_shape)

        return ModelMetadata(
            model_type=self._determine_model_type(current_model),
            framework='keras',
            input_shape=input_shape_tup if input_shape_tup else (0,),
            output_shape=output_shape_tup if output_shape_tup else (0,),
            input_type=self._determine_input_output_type(input_shape_tup if input_shape_tup else (0,)),
            output_type=self._determine_input_output_type(output_shape_tup if output_shape_tup else (0,)),
            layer_info=layer_info_list,
            additional_info=self._get_additional_info(current_model)
        )

    def preprocess_input(self, input_data: T) -> T:
        """Preprocess input data for Keras model."""
        keras_api = self._ensure_keras_module()
        # Check if input_data is already a tensor type for the backend (e.g. tf.Tensor)
        if hasattr(self._tf_module, 'is_tensor') and self._tf_module.is_tensor(input_data):
            return input_data
        if hasattr(self._tf_module, 'keras') and isinstance(input_data, self._tf_module.Tensor): # tf.Tensor
            return input_data

        # Standard preprocessing: convert to numpy array, ensure float32 for ART
        if isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32) if input_data.dtype != np.float32 else input_data
        if isinstance(input_data, list):
             # if list of numbers, convert to numpy array
            if all(isinstance(x, (int, float)) for x in input_data):
                return np.array(input_data, dtype=np.float32)
            # if list of numpy arrays (e.g. for multi-input models), ensure they are float32
            if all(isinstance(x, np.ndarray) for x in input_data):
                return [x.astype(np.float32) if x.dtype != np.float32 else x for x in input_data]

        try: # General fallback
            return np.array(input_data, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Could not preprocess input for Keras model. Input type: {type(input_data)}. Error: {e}")


    def postprocess_output(self, output_data: T) -> T:
        """Postprocess output data from Keras model (typically numpy arrays)."""
        keras_api = self._ensure_keras_module()
        # If using tf.keras, output might be tf.Tensor
        if hasattr(self._tf_module, 'keras') and isinstance(output_data, self._tf_module.Tensor):
            return output_data.numpy()
        if isinstance(output_data, list) and all(isinstance(t, getattr(self._tf_module, 'Tensor', type(None))) for t in output_data):
            return [t.numpy() for t in output_data]
        return output_data # Assuming it's already numpy or compatible

    def validate_model(self, model: Any) -> bool:
        """Validate Keras model structure. BaseModel already checks for predict/forward."""
        keras_api = self._ensure_keras_module()
        return isinstance(model, keras_api.Model) and super().validate_model(model)

    def get_model_predictions(self, model: Any, data: T) -> T:
        """Get predictions from the Keras model."""
        current_model = model if model is not None else self._model_instance
        if current_model is None: raise ValueError("Model not loaded or provided for prediction.")

        processed_data = self.preprocess_input(data)
        return self.postprocess_output(current_model.predict(processed_data))

    def _determine_model_type(self, model: Any) -> str:
        """Determine the type of Keras model."""
        if hasattr(model, 'loss') and model.loss:
            loss_str = str(model.loss).lower()
            if any(kw in loss_str for kw in ['categorical_crossentropy', 'binary_crossentropy', 'sparse_categorical_crossentropy']): return 'classification'
            if any(kw in loss_str for kw in ['mse', 'mean_squared_error', 'mae', 'mean_absolute_error']): return 'regression'

        # Check output layer activation if model has layers
        if hasattr(model, 'layers') and model.layers:
            try:
                last_layer = model.layers[-1]
                if hasattr(last_layer, 'activation'):
                    activation_name = getattr(last_layer.activation, '__name__', '').lower()
                    if 'softmax' in activation_name or 'sigmoid' in activation_name: return 'classification'
            except IndexError: pass
        return 'unknown'

    def _determine_input_output_type(self, shape_info: Union[Tuple, List[Tuple]]) -> str:
        """Determine input/output type based on shape (after batch dim removal)."""
        if not shape_info: return 'unknown'

        current_shape = shape_info[0] if isinstance(shape_info, list) and shape_info and isinstance(shape_info[0], tuple) else shape_info

        if not isinstance(current_shape, tuple) or not current_shape : return 'unknown_non_tuple_shape'

        # Shape here is assumed to be without batch dim
        if len(current_shape) == 3: # (H, W, C)
            if current_shape[-1] in [1,3,4]: return 'image_2d_rgb_or_gray_alpha'
            return 'volumetric_or_3d_feature_map'
        if len(current_shape) == 2: # (SeqLen, Features) or (H,W)
            return 'sequence_or_grayscale_image'
        if len(current_shape) == 1 and current_shape[0] is not None and current_shape[0] > 0 : # (Features,)
            return 'tabular_or_flattened'
        return 'unknown_shape_details'


    def _get_additional_info(self, model: Any) -> Dict[str, Any]:
        """Get additional Keras model information."""
        info = {'model_class': model.__class__.__name__}
        try:
            info['num_layers'] = len(model.layers)
            info['total_params'] = model.count_params()
            info['trainable_params'] = sum(np.prod(w.shape.as_list()) for w in model.trainable_weights)
            info['non_trainable_params'] = sum(np.prod(w.shape.as_list()) for w in model.non_trainable_weights)

        except AttributeError: pass # model might not be fully Keras-like

        if hasattr(model, 'optimizer') and model.optimizer is not None:
            info['optimizer_name'] = model.optimizer.__class__.__name__
            try: info['optimizer_config'] = model.optimizer.get_config()
            except: pass # Some optimizers might not have simple config
        else: info['optimizer'] = None

        info['loss_config'] = str(model.loss) if hasattr(model, 'loss') else None
        # Keras model.metrics_names is only populated after compile + training/eval step
        # info['metrics_names'] = model.metrics_names if hasattr(model, 'metrics_names') and model.metrics_names else None
        return info
