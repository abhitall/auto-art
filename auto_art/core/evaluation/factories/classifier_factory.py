"""
Factory for creating ART classifiers.
"""

from typing import Any, Dict, Optional, Type, Callable
import torch
import torch.nn as nn

# ART Imports
from art.estimators.classification import (
    PyTorchClassifier,
    TensorFlowV2Classifier,
    KerasClassifier,
)
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.object_detection import PyTorchObjectDetector

# Local Imports
from ...config.manager import ConfigManager
from ..config.evaluation_config import ModelType, Framework
import sys # For printing warnings if needed (commented out for now)

# Try to import TensorFlow to check for its availability for device checking
_tf_available_for_gpu_check = False
try:
    import tensorflow as tf
    _tf_available_for_gpu_check = True
except ImportError:
    pass


class ClassifierFactory:
    """Factory for creating ART classifiers, with device placement awareness."""

    @staticmethod
    def _determine_actual_device(requested_device: Optional[str] = None) -> str:
        """Determines the actual device ('cpu' or 'gpu') based on request and availability."""
        cfg = ConfigManager().config
        # Fallback to config default_device if requested_device is None or empty
        effective_request = requested_device if requested_device else cfg.default_device.lower()
        if not effective_request: # If still None or empty (e.g. default_device was also empty)
            effective_request = "auto" # Final fallback to auto

        final_device = 'cpu' # Default to CPU

        if effective_request == "gpu":
            if cfg.use_gpu:
                if torch.cuda.is_available():
                    final_device = "gpu"
                elif _tf_available_for_gpu_check:
                    try:
                        if tf.config.list_physical_devices('GPU'):
                            final_device = "gpu"
                        else:
                            # print("Warning: GPU requested but TensorFlow found no GPU. Falling back to CPU.", file=sys.stderr)
                            pass # Stays cpu
                    except Exception:
                        # print("Warning: Error checking TensorFlow GPU. Falling back to CPU.", file=sys.stderr)
                        pass
                else:
                    # print("Warning: GPU requested but no compatible GPU backend (PyTorch/TensorFlow) found or usable. Falling back to CPU.", file=sys.stderr)
                    pass
            else:
                # print("Info: GPU usage is disabled by global 'use_gpu' config. Using CPU.", file=sys.stderr)
                pass
        elif effective_request == "cpu":
            final_device = "cpu"
        elif effective_request == "auto":
            if cfg.use_gpu:
                if torch.cuda.is_available():
                    final_device = "gpu"
                elif _tf_available_for_gpu_check:
                    try:
                        if tf.config.list_physical_devices('GPU'):
                            final_device = "gpu"
                    except Exception:
                        pass
            # If no GPU found or use_gpu is false, it remains 'cpu'
            # print(f"Info: Device 'auto' resolved to '{final_device}'. (use_gpu: {cfg.use_gpu})", file=sys.stderr)
        else:
            # print(f"Warning: Invalid requested_device '{effective_request}'. Defaulting to CPU.", file=sys.stderr)
            final_device = "cpu" # Fallback for invalid string

        return final_device

    @staticmethod
    def create_classifier(
        model: Any,
        model_type: ModelType,
        framework: Framework,
        device_type: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Create appropriate ART classifier based on framework, model type, and device preference.
        """
        resolved_device = "unknown" # Placeholder for error message if it fails early
        try:
            resolved_device = ClassifierFactory._determine_actual_device(device_type)
            kwargs_with_device = kwargs.copy()
            kwargs_with_device['device_type'] = resolved_device

            creator: Callable[..., Any] = ClassifierFactory._get_creator(framework, model_type)
            return creator(model, **kwargs_with_device)
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            raise ValueError(f"Failed to create classifier for {framework.value}/{model_type.value} (requested device: '{device_type}', resolved: '{resolved_device}'): {str(e)}")


    @staticmethod
    def _get_creator(
        framework: Framework,
        model_type: ModelType
    ) -> Callable[..., Any]:
        """Get the appropriate classifier creator function."""
        creators: Dict[Tuple[Framework, ModelType], Callable[..., Any]] = {
            (Framework.PYTORCH, ModelType.CLASSIFICATION): ClassifierFactory._create_pytorch_classifier,
            (Framework.PYTORCH, ModelType.OBJECT_DETECTION): ClassifierFactory._create_pytorch_object_detector,
            (Framework.TENSORFLOW, ModelType.CLASSIFICATION): ClassifierFactory._create_tensorflow_classifier,
            (Framework.KERAS, ModelType.CLASSIFICATION): ClassifierFactory._create_keras_classifier,
            (Framework.SKLEARN, ModelType.CLASSIFICATION): ClassifierFactory._create_sklearn_classifier,
        }

        key = (framework, model_type)
        if key not in creators:
            if framework == Framework.KERAS and (Framework.TENSORFLOW, model_type) in creators:
                # print(f"Info: No specific Keras creator for {model_type.value}, using TensorFlow creator as fallback.", file=sys.stderr)
                return creators[(Framework.TENSORFLOW, model_type)]
            raise ValueError(f"Unsupported framework/model type combination for classifier factory: {framework.value}/{model_type.value}")

        return creators[key]

    @staticmethod
    def _create_pytorch_classifier(
        model: nn.Module,
        **kwargs: Any
    ) -> PyTorchClassifier:
        loss_function = kwargs.pop('loss_fn', nn.CrossEntropyLoss()) # Pop loss_fn, provide default
        device_to_use = kwargs.pop('device_type', 'cpu') # Pop device_type

        # Ensure essential kwargs for PyTorchClassifier are present or raise error
        if 'input_shape' not in kwargs or 'nb_classes' not in kwargs:
            raise ValueError("PyTorchClassifier requires 'input_shape' and 'nb_classes' in kwargs.")

        return PyTorchClassifier(
            model=model,
            loss=loss_function,
            input_shape=kwargs.get('input_shape'),
            nb_classes=kwargs.get('nb_classes'),
            optimizer=kwargs.get('optimizer'),
            clip_values=kwargs.get('clip_values', (0.0, 1.0)),
            device_type=device_to_use,
            **kwargs # Pass remaining kwargs like channels_first, preprocessing_defences etc.
        )

    @staticmethod
    def _create_pytorch_object_detector(
        model: nn.Module,
        **kwargs: Any
    ) -> PyTorchObjectDetector:
        device_to_use = kwargs.pop('device_type', 'cpu')
        # PyTorchObjectDetector in ART 1.x does not take input_shape, nb_classes directly.
        # It infers them or they are implicit.
        return PyTorchObjectDetector(
            model=model,
            clip_values=kwargs.get('clip_values', (0.0, 1.0)),
            channels_first=kwargs.get('channels_first', True),
            attack_losses=kwargs.get('attack_losses', ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg')),
            device_type=device_to_use,
            **kwargs # Pass remaining kwargs
        )

    @staticmethod
    def _create_tensorflow_classifier(
        model: Any,
        **kwargs: Any
    ) -> TensorFlowV2Classifier:
        resolved_device_type = kwargs.pop('device_type', 'cpu')
        # if resolved_device_type == 'gpu':
            # print("Note: TensorFlow device placement via tf.device. 'gpu' preference is informational.", file=sys.stderr)

        if 'input_shape' not in kwargs or 'nb_classes' not in kwargs:
            raise ValueError("TensorFlowV2Classifier requires 'input_shape' and 'nb_classes' in kwargs.")

        return TensorFlowV2Classifier(
            model=model,
            nb_classes=kwargs.get('nb_classes'),
            input_shape=kwargs.get('input_shape'),
            loss_object=kwargs.get('loss_object'),
            optimizer=kwargs.get('optimizer'),
            clip_values=kwargs.get('clip_values', (0.0, 1.0)),
            # device_type is not a direct arg for TensorFlowV2Classifier constructor
            **kwargs # Pass remaining kwargs
        )

    @staticmethod
    def _create_keras_classifier(
        model: Any,
        **kwargs: Any
    ) -> KerasClassifier:
        kwargs.pop('device_type', None) # KerasClassifier doesn't use it
        # if resolved_device_type == 'gpu':
            # print("Note: Keras device placement follows TensorFlow backend.", file=sys.stderr)
        return KerasClassifier(
            model=model,
            clip_values=kwargs.get('clip_values', (0.0, 1.0)),
            use_logits=kwargs.get('use_logits', False),
            channels_first=kwargs.get('channels_first'),
            **kwargs # Pass remaining kwargs
        )

    @staticmethod
    def _create_sklearn_classifier(
        model: Any,
        **kwargs: Any
    ) -> ScikitlearnClassifier:
        resolved_device_type = kwargs.pop('device_type', 'cpu')
        if resolved_device_type == 'gpu':
            # print("Warning: ScikitlearnClassifier runs on CPU. 'gpu' device request ignored.", file=sys.stderr)
            pass
        return ScikitlearnClassifier(
            model=model,
            clip_values=kwargs.get('clip_values', (0.0, 1.0)),
            **kwargs # Pass remaining kwargs like preprocessing, postprocessing
        )
```
The `_determine_actual_device` method was slightly refined to handle `requested_device` being `None` or empty string by defaulting to `cfg.default_device` or "auto". The creator methods now explicitly pop `device_type` and `loss_fn` (for PyTorch) to avoid passing them twice if they are also in `**kwargs`, and raise `ValueError` if essential parameters like `input_shape` or `nb_classes` are missing.The temporary file `auto_art/core/evaluation/factories/classifier_factory_py_device.py` has been created with the new content for `ClassifierFactory`.

Key changes included:
- Addition of `ConfigManager` import.
- Conditional import of `tensorflow` for GPU availability checks.
- New `_determine_actual_device` static method to resolve 'cpu' or 'gpu' based on user request, global config, and hardware availability.
- `create_classifier` method updated to take an optional `device_type` argument, call `_determine_actual_device`, and pass the resolved device to specific creator methods via `kwargs`.
- Specific creator methods (`_create_pytorch_classifier`, `_create_tensorflow_classifier`, etc.) updated to:
    - Accept `device_type` from `kwargs`.
    - Pass `device_type` to `PyTorchClassifier` and `PyTorchObjectDetector` constructors.
    - Note that TensorFlow/Keras ART classifiers manage devices internally, so `device_type` is not passed to them.
    - Scikit-learn runs on CPU, so `device_type='gpu'` is informational but ignored.
    - Essential kwargs like `input_shape` and `nb_classes` are now explicitly checked in PyTorch and TensorFlow/Keras creators.

The next step is to move this temporary file to the correct location, overwriting the original `auto_art/core/evaluation/factories/classifier_factory.py`.
