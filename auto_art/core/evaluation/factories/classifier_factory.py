"""
Factory for creating ART classifiers.
"""

from typing import Any, Dict, Optional, Type
import torch.nn as nn
from art.estimators.classification import (
    PyTorchClassifier,
    TensorFlowV2Classifier,
    KerasClassifier,
)
from art.estimators.classification.scikitlearn import ScikitlearnClassifier
from art.estimators.object_detection import PyTorchObjectDetector

from ..config.evaluation_config import ModelType, Framework

class ClassifierFactory:
    """Factory for creating ART classifiers."""
    
    @staticmethod
    def create_classifier(
        model: Any,
        model_type: ModelType,
        framework: Framework,
        **kwargs: Any
    ) -> Any:
        """Create appropriate ART classifier based on framework and model type."""
        try:
            creator = ClassifierFactory._get_creator(framework, model_type)
            return creator(model, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create classifier: {str(e)}")

    @staticmethod
    def _get_creator(
        framework: Framework,
        model_type: ModelType
    ) -> Type[Any]:
        """Get the appropriate classifier creator function."""
        creators = {
            (Framework.PYTORCH, ModelType.CLASSIFICATION): ClassifierFactory._create_pytorch_classifier,
            (Framework.PYTORCH, ModelType.OBJECT_DETECTION): ClassifierFactory._create_pytorch_object_detector,
            (Framework.TENSORFLOW, ModelType.CLASSIFICATION): ClassifierFactory._create_tensorflow_classifier,
            (Framework.KERAS, ModelType.CLASSIFICATION): ClassifierFactory._create_keras_classifier,
            (Framework.SKLEARN, ModelType.CLASSIFICATION): ClassifierFactory._create_sklearn_classifier,
        }
        
        key = (framework, model_type)
        if key not in creators:
            raise ValueError(f"Unsupported framework/model type combination: {framework}/{model_type}")
        
        return creators[key]

    @staticmethod
    def _create_pytorch_classifier(
        model: nn.Module,
        **kwargs: Any
    ) -> PyTorchClassifier:
        """Create PyTorch classifier."""
        return PyTorchClassifier(
            model=model,
            loss=kwargs.get('loss_fn', nn.CrossEntropyLoss()),
            input_shape=kwargs.get('input_shape'),
            nb_classes=kwargs.get('nb_classes'),
            optimizer=kwargs.get('optimizer'),
            clip_values=kwargs.get('clip_values', (0, 1))
        )

    @staticmethod
    def _create_pytorch_object_detector(
        model: nn.Module,
        **kwargs: Any
    ) -> PyTorchObjectDetector:
        """Create PyTorch object detector."""
        return PyTorchObjectDetector(
            model=model,
            clip_values=kwargs.get('clip_values', (0, 1)),
            channels_first=kwargs.get('channels_first', True),
            **kwargs
        )

    @staticmethod
    def _create_tensorflow_classifier(
        model: Any,
        **kwargs: Any
    ) -> TensorFlowV2Classifier:
        """Create TensorFlow classifier."""
        return TensorFlowV2Classifier(
            model=model,
            nb_classes=kwargs.get('nb_classes'),
            input_shape=kwargs.get('input_shape'),
            loss_object=kwargs.get('loss_object'),
            optimizer=kwargs.get('optimizer'),
            clip_values=kwargs.get('clip_values', (0, 1))
        )

    @staticmethod
    def _create_keras_classifier(
        model: Any,
        **kwargs: Any
    ) -> KerasClassifier:
        """Create Keras classifier."""
        return KerasClassifier(
            model=model,
            clip_values=kwargs.get('clip_values', (0, 1)),
            use_logits=kwargs.get('use_logits', False),
            **kwargs
        )

    @staticmethod
    def _create_sklearn_classifier(
        model: Any,
        **kwargs: Any
    ) -> ScikitlearnClassifier:
        """Create scikit-learn classifier."""
        return ScikitlearnClassifier(
            model=model,
            clip_values=kwargs.get('clip_values', (0, 1)),
            preprocessing=kwargs.get('preprocessing'),
            postprocessing=kwargs.get('postprocessing')
        ) 