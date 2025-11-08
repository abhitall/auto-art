"""
Test data generator module for creating appropriate test inputs based on model type,
and loading data from various sources.
"""

import numpy as np
import torch # For generate_expected_outputs, consider making this optional if model is not always PyTorch
try:
    import tensorflow as tf
except ImportError:
    tf = None # Allow TensorFlow to be optional

from typing import Dict, Any, Tuple, List, Optional, Union, Callable # Added Union, Callable, List
from dataclasses import dataclass, field # Added field
from pathlib import Path
import sys # For printing warnings if needed

from ...core.base import BaseTestGenerator, ModelMetadata # Assuming this is the correct relative path

# Attempt to import pandas for CSV loading
_PANDAS_AVAILABLE = False
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    # print("Warning: pandas library not found. CSV loading will not be available.", file=sys.stderr)
    pass


class SecurityError(Exception):
    """Exception raised for security-related issues in file loading."""
    pass


@dataclass
class TestData:
    """Container for generated test data."""
    inputs: Union[np.ndarray, Dict[str, np.ndarray]]
    expected_outputs: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict) # Initialize with empty dict


class TestDataGenerator(BaseTestGenerator): # Inherit from BaseTestGenerator
    """Generates test data synthetically or loads from sources."""

    def __init__(self):
        self.supported_input_types_synthetic = ['image', 'text', 'multimodal', 'tabular_or_flattened_features',
                                                'sequence_features', 'image_2d', 'time_series_or_audio_or_text_embeddings',
                                                'text_token_ids', 'text_embeddings', 'tabular'] # Added more known types
        # self.model_loader = None # If direct model loading/analysis is needed here

    # --- Implementation of BaseTestGenerator abstract methods (existing ones) ---
    def generate_test_data(self, model_metadata: ModelMetadata, num_samples: int = 100) -> TestData:
        """Generates test data synthetically appropriate for the model type and input format."""
        input_type_lower = model_metadata.input_type.lower() if model_metadata.input_type else 'unknown'

        if input_type_lower == 'image' or input_type_lower == 'image_2d': # Consolidate
            return self._generate_image_data(model_metadata, num_samples)
        elif input_type_lower in ['text', 'text_token_ids', 'text_embeddings']: # Consolidate
            return self._generate_text_data(model_metadata, num_samples)
        elif input_type_lower == 'multimodal':
            return self._generate_multimodal_data(model_metadata, num_samples)
        elif input_type_lower in self.supported_input_types_synthetic:
            # Generic fallback for other known structured types (tabular, sequence, etc.)
            # print(f"Synthetic generation for input_type '{model_metadata.input_type}' using random numpy data based on shape.", file=sys.stderr)
            if model_metadata.input_shape and model_metadata.input_shape != (0,) and model_metadata.input_shape != (None,):
                # Ensure input_shape is usable (without None as first element if it's batch)
                actual_input_shape = model_metadata.input_shape
                if actual_input_shape and actual_input_shape[0] is None and len(actual_input_shape) > 1:
                    actual_input_shape = actual_input_shape[1:]

                if not all(isinstance(dim, int) and dim > 0 for dim in actual_input_shape): # type: ignore
                     raise ValueError(f"Cannot generate synthetic data for input_type '{model_metadata.input_type}' due to non-concrete shape: {actual_input_shape}")

                dummy_data = np.random.rand(num_samples, *actual_input_shape).astype(np.float32) # type: ignore
                return TestData(inputs=dummy_data, metadata={'generation_method': 'random_fallback_structured', 'shape': dummy_data.shape, 'original_input_type': model_metadata.input_type})
            raise ValueError(f"Cannot generate synthetic data for input_type '{model_metadata.input_type}' due to insufficient shape info: {model_metadata.input_shape}")
        else:
            raise ValueError(f"Unsupported input_type '{model_metadata.input_type}' for synthetic data generation.")


    def generate_expected_outputs(self, model: Any, test_data: TestData) -> Optional[np.ndarray]:
        """
        Generates expected outputs for the test data.
        If ground truth labels are already in test_data.expected_outputs, returns them.
        Otherwise, uses the provided model to make predictions.
        """
        if test_data.expected_outputs is not None:
            # print("Using pre-loaded expected_outputs (ground truth labels).", file=sys.stderr)
            return test_data.expected_outputs

        # print("Generating expected_outputs using model predictions (not ground truth).", file=sys.stderr)

        model_obj = model # Assuming model is the raw, callable model

        try:
            if hasattr(torch, 'Tensor') and isinstance(model_obj, torch.nn.Module):
                with torch.no_grad():
                    current_inputs = test_data.inputs
                    # Move to model's device
                    model_device = next(model_obj.parameters()).device
                    if isinstance(current_inputs, dict):
                        torch_inputs = {k: torch.from_numpy(v.astype(np.float32)).to(model_device) for k,v in current_inputs.items()}
                        outputs = model_obj(**torch_inputs)
                    else:
                        torch_inputs = torch.from_numpy(current_inputs.astype(np.float32)).to(model_device) # type: ignore
                        outputs = model_obj(torch_inputs)

                    if isinstance(outputs, torch.Tensor): return outputs.detach().cpu().numpy()
                    if hasattr(outputs, 'logits') and isinstance(outputs.logits, torch.Tensor):
                        return outputs.logits.detach().cpu().numpy()
                    return np.array(outputs)

            elif tf is not None and (isinstance(model_obj, tf.Module) or hasattr(model_obj, "predict")): # Check for Keras model too
                current_inputs = test_data.inputs
                if isinstance(current_inputs, dict):
                    tf_inputs = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k,v in current_inputs.items()}
                else:
                    tf_inputs = tf.convert_to_tensor(current_inputs, dtype=tf.float32) # type: ignore

                outputs = model_obj(tf_inputs) # Or model_obj.predict(tf_inputs) for some Keras models

                if isinstance(outputs, tf.Tensor): return outputs.numpy()
                if isinstance(outputs, dict) and 'logits' in outputs and isinstance(outputs['logits'], tf.Tensor):
                    return outputs['logits'].numpy()
                if isinstance(outputs, list) and all(isinstance(o, tf.Tensor) for o in outputs):
                    return [o.numpy() for o in outputs] # type: ignore
                return np.array(outputs)
            else: # Fallback for sklearn or other model types that have a .predict method
                if hasattr(model_obj, 'predict'):
                    # print(f"Using generic model.predict() for {type(model_obj)}", file=sys.stderr)
                    return model_obj.predict(test_data.inputs) # type: ignore
                # print(f"Warning: Model type {type(model_obj)} not directly supported for auto-prediction. Returning None.", file=sys.stderr)
                return None
        except Exception as e:
            # print(f"Error during model prediction in generate_expected_outputs: {e}", file=sys.stderr)
            return None


    def load_data_from_source(self,
                              source: Union[str, Path, Tuple[np.ndarray, np.ndarray], Any],
                              data_type: str = 'test',
                              num_samples: Optional[int] = None,
                              feature_columns: Optional[List[Union[int, str]]] = None,
                              label_columns: Optional[Union[int, str, List[Union[int, str]]]] = None,
                              preprocessing_fn: Optional[Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]]] = None
                             ) -> TestData:
        loaded_inputs: Any = None # Can be np.ndarray or Dict[str, np.ndarray]
        loaded_outputs: Optional[np.ndarray] = None
        source_metadata: Dict[str, Any] = {'source_type': str(type(source)), 'data_type_requested': data_type}

        if isinstance(source, (str, Path)):
            source_path = Path(source)
            source_metadata['source_path'] = str(source_path)

            # Security: Validate and sanitize file path
            try:
                # Resolve to absolute path to prevent path traversal
                source_path = source_path.resolve(strict=True)
            except (OSError, RuntimeError) as e:
                raise ValueError(f"Invalid or inaccessible file path: {source_path}") from e

            # Security: Check file size to prevent resource exhaustion
            MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB limit
            file_size = source_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                raise ValueError(f"File size ({file_size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)")

            if not source_path.exists():
                raise FileNotFoundError(f"Data source file not found: {source_path}")

            ext = source_path.suffix.lower()
            if ext == '.npy':
                # Security: Disable pickle for safety. Use NPY files with structured data types instead.
                # If pickle is absolutely required, implement additional validation.
                try:
                    loaded_inputs = np.load(source_path, allow_pickle=False)
                except ValueError as e:
                    # If file requires pickle, raise security error
                    raise SecurityError(
                        f"File {source_path} requires pickle deserialization, which is a security risk. "
                        f"Please use NPY files with standard numpy dtypes or implement additional security validation."
                    ) from e
                source_metadata['format'] = 'npy'
            elif ext == '.npz':
                source_metadata['format'] = 'npz'
                try:
                    data_archive = np.load(source_path, allow_pickle=False)
                except ValueError as e:
                    # If file requires pickle, raise security error
                    raise SecurityError(
                        f"File {source_path} requires pickle deserialization, which is a security risk. "
                        f"Please use NPZ files with standard numpy dtypes or implement additional security validation."
                    ) from e
                # Heuristic key finding, can be made more robust or configurable
                x_keys = ['x', 'features', 'inputs', 'x_train', 'x_test', 'arr_0'] # arr_0 for unnamed single array
                y_keys = ['y', 'labels', 'outputs', 'y_train', 'y_test', 'arr_1'] # arr_1 for unnamed second array

                # Try to find inputs
                for key in x_keys:
                    if key in data_archive:
                        loaded_inputs = data_archive[key]
                        source_metadata['x_key_used'] = key
                        break
                # Try to find outputs
                for key in y_keys:
                    if key in data_archive:
                        loaded_outputs = data_archive[key]
                        source_metadata['y_key_used'] = key
                        break

                if loaded_inputs is None and len(data_archive.files) == 1: # Single array NPZ
                    loaded_inputs = data_archive[data_archive.files[0]]
                    source_metadata['x_key_used'] = data_archive.files[0]
                elif loaded_inputs is None and len(data_archive.files) > 1 and loaded_outputs is not None:
                    # If outputs were found, try to infer inputs from remaining keys
                    possible_x_keys = [k for k in data_archive.files if k not in (source_metadata.get('y_key_used'),)]
                    if len(possible_x_keys) == 1:
                        loaded_inputs = data_archive[possible_x_keys[0]]
                        source_metadata['x_key_used'] = possible_x_keys[0]

                if loaded_inputs is None: # Still not found
                    raise ValueError(f"Could not find suitable data array in NPZ file: {source_path}. Found keys: {data_archive.files}")

            elif ext == '.csv':
                source_metadata['format'] = 'csv'
                if not _PANDAS_AVAILABLE:
                    raise ImportError("Pandas is required to load CSV files. Please install pandas.")
                df = pd.read_csv(source_path)

                if label_columns is not None:
                    if isinstance(label_columns, list):
                        loaded_outputs = df[label_columns].values
                    else:
                        loaded_outputs = df[label_columns].values
                        if len(loaded_outputs.shape) == 1: loaded_outputs = loaded_outputs.reshape(-1, 1)

                    if feature_columns is None: # Assume all other columns are features
                        feature_columns = [col for col in df.columns if col not in (label_columns if isinstance(label_columns, list) else [label_columns])]

                if feature_columns is not None:
                    loaded_inputs = df[feature_columns].values
                elif label_columns is None: # Auto-detect: assume last column is label, rest are features
                    loaded_outputs = df.iloc[:, -1].values
                    if len(loaded_outputs.shape) == 1: loaded_outputs = loaded_outputs.reshape(-1,1)
                    loaded_inputs = df.iloc[:, :-1].values
                elif loaded_inputs is None: # label_columns was specified, but feature_columns was not (so inputs still None)
                     raise ValueError("If label_columns are specified for CSV, feature_columns must also be specified or be derivable.")

                source_metadata['feature_columns_used'] = feature_columns
                source_metadata['label_columns_used'] = label_columns
            else:
                raise ValueError(f"Unsupported file extension: {ext}. Supported: .npy, .npz, .csv")

        elif isinstance(source, tuple) and len(source) == 2 and isinstance(source[0], np.ndarray):
            loaded_inputs, loaded_outputs = source[0], source[1] # Ensure outputs is also ndarray or None
            if loaded_outputs is not None and not isinstance(loaded_outputs, np.ndarray):
                 raise TypeError("If source is a tuple, the second element (labels) must be np.ndarray or None.")
            source_metadata['format'] = 'numpy_tuple'
        elif isinstance(source, np.ndarray):
            loaded_inputs = source
            source_metadata['format'] = 'numpy_array_inputs_only'
        else:
            raise TypeError(f"Unsupported source type: {type(source)}. Provide filepath, (features, labels) tuple, or features array.")

        if loaded_inputs is None:
             raise ValueError("Failed to load inputs from the source.")

        if num_samples is not None and num_samples > 0 and num_samples < loaded_inputs.shape[0]:
            indices = np.random.choice(loaded_inputs.shape[0], num_samples, replace=False)
            loaded_inputs = loaded_inputs[indices]
            if loaded_outputs is not None:
                loaded_outputs = loaded_outputs[indices]
            source_metadata['num_samples_loaded'] = num_samples
        else:
            source_metadata['num_samples_loaded'] = loaded_inputs.shape[0]

        if preprocessing_fn:
            try:
                processed_inputs, processed_outputs = preprocessing_fn(loaded_inputs, loaded_outputs)
                loaded_inputs, loaded_outputs = processed_inputs, processed_outputs
                source_metadata['preprocessing_applied'] = getattr(preprocessing_fn, '__name__', 'custom_function')
            except Exception as e:
                raise RuntimeError(f"Error applying preprocessing_fn: {e}")
        else:
            source_metadata['preprocessing_applied'] = None

        if isinstance(loaded_inputs, np.ndarray) and loaded_inputs.dtype != np.float32 and np.issubdtype(loaded_inputs.dtype, np.number):
            loaded_inputs = loaded_inputs.astype(np.float32)
        # For dict inputs, apply to each array
        elif isinstance(loaded_inputs, dict):
            for k, v_arr in loaded_inputs.items():
                if isinstance(v_arr, np.ndarray) and v_arr.dtype != np.float32 and np.issubdtype(v_arr.dtype, np.number):
                    loaded_inputs[k] = v_arr.astype(np.float32)

        if loaded_outputs is not None:
            if np.issubdtype(loaded_outputs.dtype, np.floating):
                 loaded_outputs = loaded_outputs.astype(np.float32)

        return TestData(inputs=loaded_inputs, expected_outputs=loaded_outputs, metadata=source_metadata)

    def _generate_image_data(self, metadata: ModelMetadata, num_samples: int) -> TestData:
        actual_input_shape = metadata.input_shape
        if actual_input_shape and actual_input_shape[0] is None and len(actual_input_shape) > 1:
            actual_input_shape = actual_input_shape[1:]
        elif not actual_input_shape or not all(isinstance(d, int) and d > 0 for d in actual_input_shape): # type: ignore
             raise ValueError("Cannot generate image data without valid concrete input_shape in ModelMetadata.")

        final_shape = (num_samples,) + actual_input_shape

        is_likely_pixel_image = (len(actual_input_shape) == 3 and (actual_input_shape[0] in [1,3,4] or actual_input_shape[-1] in [1,3,4])) or \
                                (len(actual_input_shape) == 2 and min(actual_input_shape) > 16) # type: ignore

        if is_likely_pixel_image:
            images = np.random.uniform(0, 1, final_shape).astype(np.float32)
        else:
            images = np.random.randn(*final_shape).astype(np.float32)

        return TestData(inputs=images, metadata={'input_type': 'image', 'generation_method': 'synthetic_random', 'shape': final_shape})

    def _generate_text_data(self, metadata: ModelMetadata, num_samples: int) -> TestData:
        actual_input_shape = metadata.input_shape
        if actual_input_shape and actual_input_shape[0] is None and len(actual_input_shape) > 1:
            actual_input_shape = actual_input_shape[1:]
        elif not actual_input_shape or not all(isinstance(d, int) and d > 0 for d in actual_input_shape if d is not None): # Allow None for seq len if embed_dim is present
             # Special case: (None,) could be valid for ragged token sequences, but we need a length to generate.
             # For (seq_len, embed_dim), seq_len might be None.
             if not (len(actual_input_shape) == 2 and actual_input_shape[1] is not None): # type: ignore
                raise ValueError(f"Cannot generate text data without valid input_shape (e.g., (seq_len,) or (seq_len, embed_dim)). Got: {actual_input_shape}")


        if len(actual_input_shape) == 1:
            sequence_length = actual_input_shape[0] if actual_input_shape[0] is not None else 50 # Default seq_len
            vocab_size = metadata.additional_info.get('vocab_size', 2000)
            tokens = np.random.randint(0, vocab_size, (num_samples, sequence_length)).astype(np.int32)
            return TestData(inputs=tokens, metadata={'input_type': 'text_token_ids', 'generation_method': 'synthetic_random_tokens',
                                                    'sequence_length': sequence_length, 'vocab_size_assumed': vocab_size})
        elif len(actual_input_shape) == 2:
            sequence_length = actual_input_shape[0] if actual_input_shape[0] is not None else 50
            embedding_dim = actual_input_shape[1]
            if embedding_dim is None: raise ValueError("Embedding dimension cannot be None for pre-embedded text data.")
            embeddings = np.random.randn(num_samples, sequence_length, embedding_dim).astype(np.float32) # type: ignore
            return TestData(inputs=embeddings, metadata={'input_type': 'text_embeddings', 'generation_method': 'synthetic_random_embeddings',
                                                        'sequence_length': sequence_length, 'embedding_dim': embedding_dim})
        else:
            raise ValueError(f"Unsupported input_shape for synthetic text data: {metadata.input_shape}")

    def _generate_multimodal_data(self, metadata: ModelMetadata, num_samples: int) -> TestData:
        # This requires metadata.input_shape to be a Dict[str, Tuple] or similar structured info.
        # The current ModelMetadata.input_shape is Tuple.
        # This method needs significant rework if ModelMetadata isn't changed.
        # For now, raising NotImplementedError as the previous fallback was not truly multimodal.
        # print("Warning: True synthetic multimodal data generation requires structured input_shape in ModelMetadata (e.g., a Dict).", file=sys.stderr)
        # A placeholder that returns a dictionary of inputs as required by TestData type hint
        # This is still very basic and assumes two modalities for demonstration.

        # Try to parse additional_info for multimodal structure if available
        mm_config = metadata.additional_info.get('multimodal_config')
        if isinstance(mm_config, dict):
            inputs_dict: Dict[str, np.ndarray] = {}
            for key, mod_conf_any in mm_config.items():
                if not isinstance(mod_conf_any, dict):
                    # print(f"Skipping multimodal part '{key}': config not a dict.", file=sys.stderr)
                    continue

                mod_conf: Dict[str, Any] = mod_conf_any
                mod_input_type = mod_conf.get('input_type', 'unknown')
                mod_input_shape = mod_conf.get('input_shape') # Should be without batch
                mod_additional_info = mod_conf.get('additional_info', {})

                # Create temporary ModelMetadata for this modality
                temp_meta = ModelMetadata(
                    model_type=metadata.model_type, framework=metadata.framework, # Inherit parent's
                    input_shape=tuple(mod_input_shape) if mod_input_shape else (0,), # Ensure tuple
                    output_shape=(0,), # Not relevant for input generation
                    input_type=mod_input_type, output_type='unknown',
                    layer_info=[], additional_info=mod_additional_info
                )
                try:
                    test_data_part = self.generate_test_data(temp_meta, num_samples)
                    inputs_dict[key] = test_data_part.inputs # type: ignore # inputs here will be np.ndarray
                except ValueError as e:
                    # print(f"Could not generate synthetic data for multimodal part '{key}': {e}", file=sys.stderr)
                    # Fallback to random data for this part if shape is known
                    if mod_input_shape and all(isinstance(d,int) and d > 0 for d in mod_input_shape):
                        inputs_dict[key] = np.random.rand(num_samples, *mod_input_shape).astype(np.float32)
                    else:
                        inputs_dict[key] = np.array([]) # Empty if shape unknown

            if not inputs_dict:
                 raise ValueError("Multimodal generation failed: no valid modality configurations found in additional_info.multimodal_config.")
            return TestData(inputs=inputs_dict, metadata={'input_type': 'multimodal', 'generation_method': 'synthetic_structured_multimodal'})

        else:
            # print("Warning: No 'multimodal_config' in additional_info. Attempting basic fallback if input_type is 'multimodal'.", file=sys.stderr)
            # This part remains a very rough fallback as in the prompt, not truly multimodal.
            if metadata.input_shape and len(metadata.input_shape) > 1 and metadata.input_shape[0] is not None : # e.g. (total_feature_len,)
                try:
                    img_features = metadata.input_shape[0] // 2 # type: ignore
                    text_features = metadata.input_shape[0] - img_features # type: ignore

                    inputs_dict = {
                        "modality1_image_approx": np.random.rand(num_samples, img_features).astype(np.float32),
                        "modality2_text_approx": np.random.rand(num_samples, text_features).astype(np.float32)
                    }
                    return TestData(inputs=inputs_dict, metadata={'input_type': 'multimodal_fallback_dict',
                                                                        'generation_method': 'synthetic_multimodal_simple_split'})
                except (ValueError, TypeError, AttributeError) as e:
                     raise ValueError(f"Fallback multimodal generation failed for input_shape {metadata.input_shape}. Error: {e}") from e


            raise NotImplementedError("Synthetic multimodal data generation requires 'multimodal_config' in ModelMetadata.additional_info, or a simple tuple input_shape for basic fallback.")
