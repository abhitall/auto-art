"""
Factory for creating ART classifiers.
"""

from typing import Any, Dict, Optional, Tuple, Type, Callable
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

try:
    from art.estimators.classification.xgboost import XGBoostClassifier as ARTXGBoostClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from art.estimators.classification.lightgbm import LightGBMClassifier as ARTLightGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from art.estimators.classification.catboost import CatBoostARTClassifier as ARTCatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from art.estimators.classification.GPy import GPyGaussianProcessClassifier as ARTGPyClassifier
    GPY_AVAILABLE = True
except ImportError:
    GPY_AVAILABLE = False

try:
    from art.estimators.classification.blackbox import BlackBoxClassifier as ARTBlackBoxClassifier
    BLACKBOX_AVAILABLE = True
except ImportError:
    BLACKBOX_AVAILABLE = False

try:
    from art.estimators.classification.hugging_face import HuggingFaceClassifierPyTorch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Local Imports
from ....config.manager import ConfigManager
from ..config.evaluation_config import ModelType, Framework
import sys # For printing warnings if needed (commented out for now)

# Try to import TensorFlow to check for its availability for device checking
_tf_available_for_gpu_check = False
try:
    import tensorflow as tf
    _tf_available_for_gpu_check = True
except ImportError:
    pass # Handled by _tf_available_for_gpu_check flag

import logging # Import logging
logger = logging.getLogger(__name__) # Get a logger instance


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
                            logger.info("GPU requested but TensorFlow found no GPU. Falling back to CPU.")
                            # pass # Stays cpu
                    except Exception as e_tf_gpu:
                        logger.warning(f"Error checking TensorFlow GPU: {e_tf_gpu}. Falling back to CPU.")
                        # pass
                else:
                    logger.info("GPU requested but no compatible GPU backend (PyTorch/TensorFlow) found or usable. Falling back to CPU.")
                    # pass
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
                    except Exception as e_tf_gpu_auto:
                        logger.warning(f"Error checking TensorFlow GPU during 'auto' detection: {e_tf_gpu_auto}. Assuming CPU.")
                        # pass
            # If no GPU found or use_gpu is false, it remains 'cpu'
            logger.info(f"Device 'auto' resolved to '{final_device}'. (use_gpu: {cfg.use_gpu})")
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
            (Framework.XGBOOST, ModelType.CLASSIFICATION): ClassifierFactory._create_xgboost_classifier,
            (Framework.LIGHTGBM, ModelType.CLASSIFICATION): ClassifierFactory._create_lightgbm_classifier,
            (Framework.CATBOOST, ModelType.CLASSIFICATION): ClassifierFactory._create_catboost_classifier,
            (Framework.GPY, ModelType.CLASSIFICATION): ClassifierFactory._create_gpy_classifier,
            (Framework.TRANSFORMERS, ModelType.CLASSIFICATION): ClassifierFactory._create_transformers_classifier,
        }

        key = (framework, model_type)
        if key not in creators:
            if framework == Framework.KERAS and (Framework.TENSORFLOW, model_type) in creators:
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

        input_shape = kwargs.pop('input_shape')
        nb_classes = kwargs.pop('nb_classes')
        optimizer = kwargs.pop('optimizer', None)
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))

        return PyTorchClassifier(
            model=model,
            loss=loss_function,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            clip_values=clip_values,
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
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))
        channels_first = kwargs.pop('channels_first', True)
        attack_losses = kwargs.pop('attack_losses', ('loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg'))

        return PyTorchObjectDetector(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            attack_losses=attack_losses,
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

        nb_classes = kwargs.pop('nb_classes')
        input_shape = kwargs.pop('input_shape')
        loss_object = kwargs.pop('loss_object', None)
        optimizer = kwargs.pop('optimizer', None)
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))

        return TensorFlowV2Classifier(
            model=model,
            nb_classes=nb_classes,
            input_shape=input_shape,
            loss_object=loss_object,
            optimizer=optimizer,
            clip_values=clip_values,
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
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))
        use_logits = kwargs.pop('use_logits', False)
        channels_first = kwargs.pop('channels_first', None)

        return KerasClassifier(
            model=model,
            clip_values=clip_values,
            use_logits=use_logits,
            channels_first=channels_first,
            **kwargs # Pass remaining kwargs
        )

    @staticmethod
    def _create_sklearn_classifier(
        model: Any,
        **kwargs: Any
    ) -> ScikitlearnClassifier:
        resolved_device_type = kwargs.pop('device_type', 'cpu')
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))

        return ScikitlearnClassifier(
            model=model,
            clip_values=clip_values,
            **kwargs
        )

    @staticmethod
    def _create_xgboost_classifier(model: Any, **kwargs: Any) -> Any:
        kwargs.pop('device_type', None)
        if not XGBOOST_AVAILABLE:
            raise ImportError("ART XGBoostClassifier not available. Install xgboost.")
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))
        nb_classes = kwargs.pop('nb_classes', 2)
        nb_features = kwargs.pop('nb_features', None)
        return ARTXGBoostClassifier(
            model=model, clip_values=clip_values,
            nb_classes=nb_classes, nb_features=nb_features,
            **kwargs
        )

    @staticmethod
    def _create_lightgbm_classifier(model: Any, **kwargs: Any) -> Any:
        kwargs.pop('device_type', None)
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("ART LightGBMClassifier not available. Install lightgbm.")
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))
        nb_classes = kwargs.pop('nb_classes', 2)
        return ARTLightGBMClassifier(
            model=model, clip_values=clip_values,
            nb_classes=nb_classes,
            **kwargs
        )

    @staticmethod
    def _create_catboost_classifier(model: Any, **kwargs: Any) -> Any:
        kwargs.pop('device_type', None)
        if not CATBOOST_AVAILABLE:
            raise ImportError("ART CatBoostClassifier not available. Install catboost.")
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))
        nb_classes = kwargs.pop('nb_classes', 2)
        return ARTCatBoostClassifier(
            model=model, clip_values=clip_values,
            nb_classes=nb_classes,
            **kwargs
        )

    @staticmethod
    def _create_gpy_classifier(model: Any, **kwargs: Any) -> Any:
        kwargs.pop('device_type', None)
        if not GPY_AVAILABLE:
            raise ImportError("ART GPyGaussianProcessClassifier not available. Install GPy.")
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))
        return ARTGPyClassifier(
            model=model, clip_values=clip_values,
            **kwargs
        )

    @staticmethod
    def _create_transformers_classifier(model: Any, **kwargs: Any) -> Any:
        """Create an ART classifier wrapping a HuggingFace Transformers model.

        Requires ``art>=1.16.0`` with the ``hugging_face`` estimator and
        ``transformers`` + ``torch`` installed.
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "ART HuggingFaceClassifierPyTorch not available. "
                "Install art>=1.16.0 and transformers."
            )

        device_to_use = kwargs.pop('device_type', 'cpu')

        if 'input_shape' not in kwargs or 'nb_classes' not in kwargs:
            raise ValueError(
                "HuggingFaceClassifierPyTorch requires 'input_shape' and "
                "'nb_classes' in kwargs."
            )

        input_shape = kwargs.pop('input_shape')
        nb_classes = kwargs.pop('nb_classes')
        loss_function = kwargs.pop('loss_fn', nn.CrossEntropyLoss())
        optimizer = kwargs.pop('optimizer', None)
        clip_values = kwargs.pop('clip_values', (0.0, 1.0))

        return HuggingFaceClassifierPyTorch(
            model=model,
            loss=loss_function,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            clip_values=clip_values,
            device_type=device_to_use,
            **kwargs
        )
