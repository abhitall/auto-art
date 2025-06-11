"""
TensorFlow model implementation.
"""

from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from ...core.base import BaseModel, ModelMetadata
import numpy as np

_tf_module = None
try:
    import tensorflow as tf
    _tf_module = tf
except ImportError:
    pass # Allow import to fail if TF not installed, __init__ will raise error.

T = TypeVar('T')

class TensorFlowModel(BaseModel[T]):
    """TensorFlow model implementation with Template Method pattern."""

    def __init__(self):
        self.supported_extensions = {'.h5', '.keras', '.pb', '.savedmodel'}
        self._model_instance: Optional[Any] = None
        if _tf_module is None:
            raise ImportError("TensorFlow is not installed. Please install TensorFlow to use TensorFlowModel.")

    def load_model(self, model_path: str) -> Tuple[Any, str]:
        """Load TensorFlow model from path."""
        if not any(model_path.endswith(ext) for ext in self.supported_extensions) and not tf.io.gfile.isdir(model_path):
            raise ValueError(f"Unsupported model file extension or path type. Supported extensions: {self.supported_extensions} or a directory for SavedModel.")

        try:
            if tf.io.gfile.isdir(model_path) or model_path.lower().endswith('.pb'): # .pb is tricky, can be frozen or part of SavedModel
                try:
                    loaded_model = tf.saved_model.load(model_path)
                    # Check if this is a Keras model instance within the SavedModel structure
                    if isinstance(loaded_model, tf.keras.Model):
                        self._model_instance = loaded_model
                    # Check for standard serving signature
                    elif hasattr(loaded_model, 'signatures') and 'serving_default' in loaded_model.signatures:
                        self._model_instance = loaded_model.signatures['serving_default']
                    # Check if the loaded object itself is callable (e.g. a restored Keras model not under signatures)
                    elif callable(loaded_model):
                         self._model_instance = loaded_model
                    else:
                        # If it's a trackable object with sub-modules, we might need to find the Keras model
                        # This can get complex, e.g. loaded_model.keras_model if saved that way
                        # For now, if not directly a Keras model or serving_default, we might have an issue using it generically
                        raise ValueError("Loaded SavedModel is not a Keras model and does not have a 'serving_default' signature. Manual inspection needed.")

                except Exception as e_sm:
                    try: # Fallback for .pb that might be a Keras H5 or older format, or if SavedModel loading failed
                        self._model_instance = tf.keras.models.load_model(model_path)
                    except Exception as e_h5:
                        raise RuntimeError(f"Failed to load TensorFlow model. SavedModel/PB error: {e_sm}. Keras H5 error: {e_h5}")
            else:
                self._model_instance = tf.keras.models.load_model(model_path)

            if self._model_instance is None:
                 raise RuntimeError("Model could not be loaded or is not of a recognized TensorFlow type.")

            return self._model_instance, 'tensorflow'
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorFlow model from {model_path}: {str(e)}")

    def _get_layer_config_tf(self, layer: Any) -> Dict[str, Any]:
        config = {}
        try: config = layer.get_config()
        except AttributeError: pass

        if 'name' not in config and hasattr(layer, 'name'): config['name'] = layer.name
        if hasattr(layer, 'trainable'): config['trainable'] = layer.trainable
        if hasattr(layer, 'dtype'): config['dtype'] = layer.dtype

        if not config and not isinstance(layer, tf.keras.layers.Layer):
            if hasattr(layer, 'variables'): config['num_variables'] = len(layer.variables)
            if hasattr(layer, 'trainable_variables'): config['num_trainable_variables'] = len(layer.trainable_variables)
        return config

    def analyze_architecture(self, model: Any, framework: str) -> ModelMetadata:
        current_model = model if model is not None else self._model_instance
        if current_model is None: raise ValueError("Model not loaded or provided.")

        layer_info_list: List[Dict[str, Any]] = []
        input_s: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]] = (None,) # Default with batch
        output_s: Union[Tuple[Optional[int], ...], List[Tuple[Optional[int], ...]]] = (None,) # Default with batch

        if isinstance(current_model, tf.keras.Model):
            for layer in current_model.layers:
                layer_entry = {
                    'name': layer.name, 'type': layer.__class__.__name__,
                    'input_shape': layer.input_shape, 'output_shape': layer.output_shape,
                    'config': self._get_layer_config_tf(layer),
                    'num_params': layer.count_params() if hasattr(layer, 'count_params') else sum(tf.keras.backend.count_params(w) for w in layer.weights),
                    'trainable': layer.trainable,
                }
                layer_info_list.append(layer_entry)

            raw_input_shape = current_model.input_shape
            raw_output_shape = current_model.output_shape

            if isinstance(raw_input_shape, list):
                input_s = tuple(s if isinstance(s, tuple) else (None,) for s in raw_input_shape) # Keep batch dim as None
            elif isinstance(raw_input_shape, tuple):
                input_s = raw_input_shape
            else: input_s = (None,)

            if isinstance(raw_output_shape, list):
                output_s = tuple(s if isinstance(s, tuple) else (None,) for s in raw_output_shape)
            elif isinstance(raw_output_shape, tuple):
                output_s = raw_output_shape
            else: output_s = (None,)

        elif callable(current_model) and hasattr(current_model, 'inputs') and hasattr(current_model, 'outputs') and isinstance(current_model.inputs, list) and isinstance(current_model.outputs, list):
            # TF function from SavedModel (concrete function)
            try:
                # Get shapes including batch dimension (often None)
                inp_shapes_with_batch = [tuple(inp.shape.as_list()) for inp in current_model.inputs]
                out_shapes_with_batch = [tuple(out.shape.as_list()) for out in current_model.outputs]

                input_s = inp_shapes_with_batch[0] if len(inp_shapes_with_batch) == 1 else tuple(inp_shapes_with_batch)
                output_s = out_shapes_with_batch[0] if len(out_shapes_with_batch) == 1 else tuple(out_shapes_with_batch)

                layer_info_list.append({
                    'name': getattr(current_model, 'name', current_model.__class__.__name__),
                    'type': 'TensorFlowConcreteFunction',
                    'inputs_spec': str(current_model.inputs), 'outputs_spec': str(current_model.outputs),
                    'num_params': 'N/A', 'trainable': 'N/A'
                })
            except Exception: input_s, output_s = (None,), (None,)
        else: input_s, output_s = (None,), (None,)


        # Remove batch dimension for metadata if it's the first one and typically None or -1
        def _strip_batch(s):
            if isinstance(s, list) or isinstance(s, tuple) and len(s) > 0 and (isinstance(s[0],list) or isinstance(s[0],tuple)): #  multi-input/output
                return tuple(_strip_batch(sub_s) for sub_s in s)
            if isinstance(s, tuple) and len(s) > 1 and (s[0] is None or s[0] == -1):
                return s[1:]
            return s if isinstance(s, tuple) else (s,)


        final_input_shape = _strip_batch(input_s)
        final_output_shape = _strip_batch(output_s)


        return ModelMetadata(
            model_type=self._determine_model_type(current_model),
            framework='tensorflow',
            input_shape=final_input_shape if final_input_shape else (0,),
            output_shape=final_output_shape if final_output_shape else (0,),
            input_type=self._determine_input_output_type_tf(final_input_shape if final_input_shape else (0,)),
            output_type=self._determine_input_output_type_tf(final_output_shape if final_output_shape else (0,)),
            layer_info=layer_info_list,
            additional_info=self._get_additional_info(current_model)
        )

    def preprocess_input(self, input_data: T) -> T:
        if isinstance(input_data, tf.Tensor): return input_data
        try:
            if isinstance(input_data, np.ndarray):
                return tf.convert_to_tensor(input_data, dtype=tf.float32)
            # Handle lists of numpy arrays for multi-input models
            if isinstance(input_data, list) and all(isinstance(x, np.ndarray) for x in input_data):
                return [tf.convert_to_tensor(x, dtype=tf.float32) for x in input_data]
            return tf.convert_to_tensor(input_data)
        except Exception as e:
            raise ValueError(f"Could not convert input to TensorFlow tensor: {e}")

    def postprocess_output(self, output_data: T) -> T:
        if isinstance(output_data, tf.Tensor): return output_data.numpy()
        if isinstance(output_data, list) and all(isinstance(t, tf.Tensor) for t in output_data):
            return [t.numpy() for t in output_data]
        if isinstance(output_data, dict) and all(isinstance(t, tf.Tensor) for t in output_data.values()):
            return {k: v.numpy() for k, v in output_data.items()}
        # If output from TF function is a flat list of tensors (e.g. from model.outputs)
        if isinstance(output_data, tuple) and all(isinstance(t, tf.Tensor) for t in output_data):
             return tuple(t.numpy() for t in output_data)
        return output_data

    def get_model_predictions(self, model: Any, data: T) -> T:
        current_model = model if model is not None else self._model_instance
        if current_model is None: raise ValueError("Model not loaded or provided for prediction.")

        processed_data = self.preprocess_input(data)

        if callable(current_model):
            # If it's a Keras model or a TF function that expects a dict for multiple inputs
            if isinstance(processed_data, list) and hasattr(current_model, 'input_names') and len(current_model.input_names) == len(processed_data):
                 input_dict = {name: tensor for name, tensor in zip(current_model.input_names, processed_data)}
                 predictions = current_model(**input_dict)
            elif isinstance(processed_data, dict) and hasattr(current_model, 'input_names'):
                 predictions = current_model(**processed_data)
            else: # Standard call for single tensor input or when model handles list/tuple directly
                 predictions = current_model(processed_data)
        else:
            raise TypeError(f"Model of type {type(current_model)} is not callable for predictions.")
        return self.postprocess_output(predictions)

    def _determine_model_type(self, model: Any) -> str:
        if isinstance(model, tf.keras.Model):
            if hasattr(model, 'loss') and model.loss:
                loss_str = str(model.loss).lower()
                if any(kw in loss_str for kw in ['categorical_crossentropy', 'binary_crossentropy', 'sparse_categorical_crossentropy']): return 'classification'
                if any(kw in loss_str for kw in ['mse', 'mean_squared_error', 'mae', 'mean_absolute_error']): return 'regression'
            if model.layers:
                try:
                    last_layer = model.layers[-1]
                    if hasattr(last_layer, 'activation'):
                        activation_name = getattr(last_layer.activation, '__name__', '').lower()
                        if 'softmax' in activation_name or 'sigmoid' in activation_name: return 'classification'
                except IndexError: pass
        elif callable(model) and hasattr(model, 'outputs'): return 'generic_tf_function'
        return 'unknown'

    def _determine_input_output_type_tf(self, shape_info: Union[Tuple, List[Tuple]]) -> str:
        if not shape_info: return 'unknown'

        # Handle single shape tuple or list of shape tuples (for multi-input/output)
        # We analyze the first shape in case of multiple inputs/outputs for simplicity
        current_shape = shape_info[0] if isinstance(shape_info, list) and shape_info else shape_info

        if not isinstance(current_shape, tuple) or not current_shape: return 'unknown' # Ensure it's a non-empty tuple

        # Shape usually excludes batch size here
        # Example: (H, W, C) for image, (SeqLen, Features) for text, (Features,) for tabular
        if len(current_shape) == 3: # (H, W, C) or (Seq, H, W) etc.
            if current_shape[-1] in [1, 3, 4]: return 'image_2d_rgb_or_gray_alpha'
            return 'volumetric_or_sequence_with_channels'
        if len(current_shape) == 2: # (SeqLen, Features) or (H,W)
            return 'sequence_or_grayscale_image'
        if len(current_shape) == 1 and current_shape[0] is not None and current_shape[0] > 0 : # (Features,)
            return 'tabular_or_flattened'
        if all(isinstance(dim, int) and dim > 0 for dim in current_shape): # If all dims are concrete and positive
             if len(current_shape) >= 2: return 'tensor_generic_rank{}'.format(len(current_shape))

        return 'unknown_shape_details'


    def _get_additional_info(self, model: Any) -> Dict[str, Any]:
        info = {'model_class': model.__class__.__name__ if hasattr(model, '__class__') else str(type(model))}
        if isinstance(model, tf.keras.Model):
            try:
                info['num_layers'] = len(model.layers)
                info['total_params'] = model.count_params()
                trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
                non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
                info['trainable_params'] = trainable_params
                info['non_trainable_params'] = non_trainable_params
            except Exception: pass
            if hasattr(model, 'optimizer') and model.optimizer:
                info['optimizer_name'] = model.optimizer.__class__.__name__
                try: info['optimizer_config'] = model.optimizer.get_config()
                except: pass
            info['loss_config'] = str(model.loss) if hasattr(model, 'loss') else None
        elif callable(model) and hasattr(model, 'graph') and hasattr(model, 'inputs') and hasattr(model, 'outputs'):
            try:
                info['function_inputs'] = str(model.inputs)
                info['function_outputs'] = str(model.outputs)
                info['total_params'] = "N/A (TF Function)"
            except Exception: pass
        info['device_info'] = "TF manages device placement."
        return info
