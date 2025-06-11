"""
Scikit-learn model implementation.
"""

import joblib
from typing import Any, Dict, List, Optional, Tuple, TypeVar
from ...core.base import BaseModel, ModelMetadata # Corrected import path
import numpy as np

T = TypeVar('T')

class SklearnModel(BaseModel[T]):
    """Scikit-learn model implementation."""

    def __init__(self):
        self.supported_extensions = {'.joblib', '.pkl', '.sav'}
        self._model_instance = None

    def load_model(self, model_path: str) -> Tuple[Any, str]:
        """Load Scikit-learn model from path using joblib."""
        if not any(model_path.endswith(ext) for ext in self.supported_extensions):
            raise ValueError(f"Unsupported Scikit-learn model file extension. Supported: {self.supported_extensions}")

        try:
            self._model_instance = joblib.load(model_path)
            if not (hasattr(self._model_instance, 'predict') or hasattr(self._model_instance, 'transform')):
                 raise ValueError("Loaded object does not appear to be a Scikit-learn estimator (missing predict/transform).")
            return self._model_instance, 'sklearn'
        except Exception as e:
            raise RuntimeError(f"Failed to load Scikit-learn model from {model_path}: {str(e)}")

    def analyze_architecture(self, model: Any, framework: str) -> ModelMetadata:
        """Analyze Scikit-learn model."""
        if model is None:
            model = self._model_instance
        if model is None:
            raise ValueError("Model instance not loaded or provided.")

        model_type_str = 'unknown'
        if hasattr(model, '_estimator_type'):
            model_type_str = model._estimator_type
        elif hasattr(model, 'predict_proba') or (hasattr(model, 'classes_') and hasattr(model.classes_, '__len__') and len(model.classes_) > 1) :
            model_type_str = 'classifier'
        elif hasattr(model, 'predict') and not (hasattr(model, 'classes_') and hasattr(model.classes_, '__len__')):
            model_type_str = 'regressor'
        elif hasattr(model, 'transform'):
            model_type_str = 'transformer'

        n_features_in = getattr(model, 'n_features_in_', getattr(model, 'n_features_', None) ) # Handles older sklearn versions too
        input_shape_tup = (None, n_features_in) if n_features_in is not None else (None, None)

        output_shape_val = (None, None)
        if model_type_str == 'classifier' and hasattr(model, 'classes_') and hasattr(model.classes_, '__len__'):
            num_classes = len(model.classes_)
            if num_classes > 2: # Multi-class
                # Check if predict_proba exists and its output shape
                if hasattr(model, 'predict_proba'):
                    try:
                        if input_shape_tup[1] is not None and input_shape_tup[1] > 0:
                            dummy_input = np.zeros((1, input_shape_tup[1]))
                            proba_output_shape = model.predict_proba(dummy_input).shape
                            output_shape_val = (None, proba_output_shape[1]) if len(proba_output_shape) > 1 else (None,1)
                        else: # Cannot determine n_features_in_
                            output_shape_val = (None, num_classes) # Best guess
                    except Exception:
                         output_shape_val = (None, num_classes) # Fallback
                else: # No predict_proba, likely outputs class labels directly
                    output_shape_val = (None, 1)
            else: # Binary classifier
                 output_shape_val = (None, 1)
        elif model_type_str == 'regressor':
            try:
                if input_shape_tup[1] is not None and input_shape_tup[1] > 0:
                    dummy_input = np.zeros((1, input_shape_tup[1]))
                    pred = model.predict(dummy_input)
                    output_shape_val = (None,) + pred.shape[1:] if len(pred.shape) > 1 else (None,1)
                else:
                    output_shape_val = (None, None)
            except Exception:
                output_shape_val = (None, None)
        elif model_type_str == 'transformer' and hasattr(model, 'transform'):
             try:
                if input_shape_tup[1] is not None and input_shape_tup[1] > 0:
                    dummy_input = np.zeros((1, input_shape_tup[1]))
                    transformed = model.transform(dummy_input)
                    output_shape_val = (None,) + transformed.shape[1:] if len(transformed.shape) > 1 else (None,1)
                else:
                    output_shape_val = (None, None)
             except Exception:
                output_shape_val = (None, None)

        layer_info = []
        if hasattr(model, 'steps') and isinstance(model.steps, list):
            for step_name, step_estimator in model.steps:
                layer_info.append({
                    'name': step_name,
                    'type': step_estimator.__class__.__name__,
                    'params': step_estimator.get_params(deep=False)
                })
        else:
            layer_info.append({
                'name': model.__class__.__name__,
                'type': model.__class__.__name__,
                'params': model.get_params(deep=False) if hasattr(model, 'get_params') else {}
            })

        return ModelMetadata(
            model_type=model_type_str,
            framework='sklearn',
            input_shape=input_shape_tup,
            output_shape=output_shape_val,
            input_type='tabular',
            output_type='tabular',
            layer_info=layer_info,
            additional_info={'model_class': model.__class__.__name__, 'params': model.get_params() if hasattr(model, 'get_params') else {}}
        )

    def preprocess_input(self, input_data: T) -> T:
        """Preprocess input data for Scikit-learn model (typically expects NumPy array)."""
        if isinstance(input_data, np.ndarray):
            return input_data
        try:
            # Handle pandas DataFrames or Series
            if hasattr(input_data, 'values') and isinstance(getattr(input_data, 'values', None), np.ndarray):
                return np.asarray(input_data.values)
            return np.asarray(input_data)
        except Exception as e:
            raise ValueError(f"Input data could not be converted to NumPy array: {e}")

    def postprocess_output(self, output_data: T) -> T:
        """Postprocess output data from Scikit-learn model (typically NumPy array)."""
        return output_data

    def validate_model(self, model: Any) -> bool:
        """Validate Scikit-learn model structure. BaseModel checks for predict/forward."""
        has_predict = hasattr(model, 'predict')
        has_transform = hasattr(model, 'transform')
        # Ensure it's not a class type itself, but an instance
        is_instance = not isinstance(model, type)
        return is_instance and (has_predict or has_transform) and super().validate_model(model)

    def get_model_predictions(self, model: Any, data: T) -> T:
        """Get predictions from the Scikit-learn model."""
        current_model = model if model is not None else self._model_instance
        if current_model is None:
            raise ValueError("Model instance not loaded or provided.")

        if hasattr(current_model, 'predict'):
            return current_model.predict(data)
        elif hasattr(current_model, 'transform'):
            return current_model.transform(data)
        else:
            raise AttributeError(f"{current_model.__class__.__name__} has neither 'predict' nor 'transform' method.")
