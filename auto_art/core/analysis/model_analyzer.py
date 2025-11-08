"""
Model analyzer module for extracting metadata and properties from ML models.
Orchestrates analysis by calling framework-specific handlers.
"""
from typing import Any, Dict, Optional, Union, List # Added List
from ...core.base import ModelMetadata
from ...implementations.models.factory import ModelFactory
import numpy as np # For data conversion

# Delayed imports for metric functions to avoid heavy load at module import
# These will be imported inside the functions that need them.

import sys
# print("Initializing model_analyzer.py (refactored v2)", file=sys.stderr)


class ModelAnalyzer:
    """
    Analyzer class for extracting metadata and analyzing ML model properties.
    Provides a high-level interface for model analysis operations.
    """

    def analyze(self, model_obj: Any, handler: Any) -> ModelMetadata:
        """
        Analyzes a model using the provided handler.

        Args:
            model_obj: The model instance to analyze.
            handler: Model handler instance with analysis methods.

        Returns:
            ModelMetadata object containing analysis results.
        """
        try:
            # Extract framework from handler
            framework_name = handler.get_framework() if hasattr(handler, 'get_framework') else 'unknown'

            # Use the handler to extract metadata
            metadata_dict = {
                'model_type': handler.get_model_type() if hasattr(handler, 'get_model_type') else 'unknown',
                'framework': framework_name,
                'input_shape': handler.get_input_shape() if hasattr(handler, 'get_input_shape') else None,
                'output_shape': handler.get_output_shape() if hasattr(handler, 'get_output_shape') else None,
                'layer_info': handler.get_layer_info() if hasattr(handler, 'get_layer_info') else [],
                'input_type': 'tensor',  # Default
                'output_type': 'tensor',  # Default
                'additional_info': {}
            }

            return ModelMetadata(**metadata_dict)
        except Exception as e:
            # Return fallback metadata on error
            return ModelMetadata(
                model_type='unknown',
                framework='unknown',
                input_shape=(0,),
                output_shape=(0,),
                input_type='unknown',
                output_type='unknown',
                layer_info=[],
                additional_info={'error': f"Analysis failed: {str(e)}"}
            )

    def analyze_architecture(self, model_obj: Any, framework_name: str) -> ModelMetadata:
        """
        Analyzes model architecture directly using framework name.

        Args:
            model_obj: The model instance to analyze.
            framework_name: Name of the framework (e.g., 'pytorch', 'tensorflow').

        Returns:
            ModelMetadata object containing architecture details.
        """
        return analyze_model_architecture(model_obj, framework_name)


def analyze_model_architecture(model_obj: Any, framework_name: str) -> ModelMetadata:
    """
    Analyzes a loaded model object to determine its architecture and metadata.
    Args:
        model_obj: The loaded model instance.
        framework_name: The name of the framework (e.g., 'pytorch', 'tensorflow', 'keras').
    Returns:
        A ModelMetadata object containing details about the model.
    """
    try:
        model_handler_class = ModelFactory.get_implementation(framework_name)
        if not model_handler_class:
            if framework_name.lower() == 'keras' and ModelFactory.get_implementation('tensorflow'):
                model_handler_class = ModelFactory.get_implementation('tensorflow')
            elif framework_name.lower() == 'tf.keras':
                keras_handler = ModelFactory.get_implementation('keras')
                tf_handler = ModelFactory.get_implementation('tensorflow')
                model_handler_class = keras_handler if keras_handler else tf_handler

        if not model_handler_class:
             raise ValueError(f"No model handler implementation found for framework: {framework_name}")

        model_handler_instance = model_handler_class()
        metadata = model_handler_instance.analyze_architecture(model=model_obj, framework=framework_name)
        return metadata
    except Exception as e:
        # print(f"Error during model architecture analysis for {framework_name}: {e}", file=sys.stderr)
        return ModelMetadata(
            model_type='unknown', framework=framework_name,
            input_shape=(0,), output_shape=(0,), input_type='unknown', output_type='unknown',
            layer_info=[], additional_info={'error': f"Analysis failed: {str(e)}", 'notes': 'Fallback metadata due to error.'}
        )

def _get_default_loss_for_model(model_metadata: ModelMetadata, model_obj: Any):
    # This is a simplified placeholder. Robust loss determination is complex.
    from ...core.evaluation.config.evaluation_config import Framework as FrameworkEnum
    from ...core.evaluation.config.evaluation_config import ModelType as ModelTypeEnum
    import torch # For PyTorch default

    if model_metadata.framework == FrameworkEnum.PYTORCH.value:
        if model_metadata.model_type == ModelTypeEnum.CLASSIFICATION.value:
            return torch.nn.CrossEntropyLoss() # Common default for PyTorch classifiers
        return torch.nn.MSELoss() # Generic fallback for PyTorch
    # For TF/Keras, loss is often part of the model if compiled. ART wrappers handle this.
    # If not compiled, ART might require it or fail.
    return None # Let ART try to infer or use its defaults for TF/Keras.

def estimate_clever_score(model_obj: Any, model_metadata: ModelMetadata, sample_input: Any,
                          nb_batches: int = 10, batch_size: int = 100, radius: float = 0.3, norm: Union[int, float, str] = 2) -> Optional[float]:
    """
    Estimates the CLEVER score for a model and a sample input.
    """
    from ...core.evaluation.config.evaluation_config import Framework as FrameworkEnum
    from ...core.evaluation.config.evaluation_config import ModelType as ModelTypeEnum
    from ...core.evaluation.factories.classifier_factory import ClassifierFactory
    from art.metrics import clever_u

    if model_metadata.model_type != ModelTypeEnum.CLASSIFICATION.value:
        # print(f"Warning: CLEVER score is designed for classification models. Model type is {model_metadata.model_type}.", file=sys.stderr)
        # Allow to proceed, ART will error out if estimator type is incompatible.
        pass

    try:
        fw_enum = FrameworkEnum(model_metadata.framework)
        nb_classes = 1
        if model_metadata.output_shape and isinstance(model_metadata.output_shape[-1], int) and model_metadata.output_shape[-1] > 0:
            nb_classes = model_metadata.output_shape[-1]
        elif model_metadata.model_type == ModelTypeEnum.CLASSIFICATION.value:
             nb_classes = 2 # Fallback for classifiers if output shape unclear, very heuristic.

        loss_fn = _get_default_loss_for_model(model_metadata, model_obj)

        # Ensure input_shape in metadata doesn't have batch dimension for ClassifierFactory
        meta_input_shape = model_metadata.input_shape
        if isinstance(meta_input_shape, tuple) and len(meta_input_shape)>0 and meta_input_shape[0] is None :
            meta_input_shape = meta_input_shape[1:]


        art_classifier = ClassifierFactory.create_classifier(
            model=model_obj, model_type=ModelTypeEnum(model_metadata.model_type), framework=fw_enum,
            input_shape=meta_input_shape, nb_classes=nb_classes, loss_fn=loss_fn
        )

        current_sample_input = sample_input
        if not isinstance(current_sample_input, np.ndarray):
            if hasattr(current_sample_input, "numpy"): current_sample_input = current_sample_input.numpy()
            else: current_sample_input = np.array(current_sample_input)

        # ART's clever_u expects a single sample (no batch dim)
        if len(current_sample_input.shape) == len(art_classifier.input_shape) + 1: # Has batch dim
            current_sample_input = current_sample_input[0]

        if current_sample_input.shape != art_classifier.input_shape:
             raise ValueError(f"Sample input shape {current_sample_input.shape} incompatible with ART classifier input shape {art_classifier.input_shape}")

        score = clever_u(art_classifier, current_sample_input, nb_batches, batch_size, radius, norm)
        return float(score)
    except Exception as e:
        # print(f"Error estimating CLEVER score for {model_metadata.framework} model: {e}", file=sys.stderr)
        return None

def evaluate_empirical_robustness(model_obj: Any, model_metadata: ModelMetadata,
                                  test_data: Any, test_labels: Any, attack_name: str = "fgsm") -> Dict[str, Any]:
    """
    Evaluates empirical robustness of the model against a specified attack.
    """
    from ...core.evaluation.config.evaluation_config import Framework as FrameworkEnum
    from ...core.evaluation.config.evaluation_config import ModelType as ModelTypeEnum
    from ...core.evaluation.factories.classifier_factory import ClassifierFactory
    from art.metrics import empirical_robustness

    if model_metadata.model_type != ModelTypeEnum.CLASSIFICATION.value:
        # print(f"Warning: Empirical robustness is typically for classification. Model type: {model_metadata.model_type}", file=sys.stderr)
        pass # Allow attempt

    try:
        fw_enum = FrameworkEnum(model_metadata.framework)
        nb_classes = 1
        if model_metadata.output_shape and isinstance(model_metadata.output_shape[-1], int) and model_metadata.output_shape[-1] > 0:
            nb_classes = model_metadata.output_shape[-1]
        elif model_metadata.model_type == ModelTypeEnum.CLASSIFICATION.value:
            nb_classes = 2

        loss_fn = _get_default_loss_for_model(model_metadata, model_obj)
        meta_input_shape = model_metadata.input_shape
        if isinstance(meta_input_shape, tuple) and len(meta_input_shape)>0 and meta_input_shape[0] is None :
            meta_input_shape = meta_input_shape[1:]

        art_classifier = ClassifierFactory.create_classifier(
            model=model_obj, model_type=ModelTypeEnum(model_metadata.model_type), framework=fw_enum,
            input_shape=meta_input_shape, nb_classes=nb_classes, loss_fn=loss_fn
        )

        current_test_data, current_test_labels = test_data, test_labels
        if not isinstance(current_test_data, np.ndarray): current_test_data = np.array(current_test_data)
        if not isinstance(current_test_labels, np.ndarray): current_test_labels = np.array(current_test_labels)

        # empirical_robustness expects x (test_data) to have batch dim.
        # It expects labels to match.
        # Ensure test_data has batch dim if classifier expects it (usually does)
        if len(current_test_data.shape) == len(art_classifier.input_shape) : # Missing batch
             current_test_data = np.expand_dims(current_test_data, axis=0)
             # If labels also need batching for a single sample, this might be needed, but usually not for labels.

        robustness_score = empirical_robustness(art_classifier, current_test_data, attack_name=attack_name.lower())
        return {"notes": "Empirical robustness calculated.", "robustness_score": float(robustness_score)}
    except Exception as e:
        # print(f"Error evaluating empirical robustness for {model_metadata.framework} model: {e}", file=sys.stderr)
        return {"notes": f"Empirical robustness evaluation failed: {str(e)}", "robustness_score": None}

# Parameter sensitivity and vulnerability assessment placeholders remain as previously defined.
def calculate_parameter_sensitivity(model_metadata: ModelMetadata, model_obj: Any, data: Any, grad_source: str = 'loss') -> Dict[str, Any]:
    return {"notes": "Parameter sensitivity analysis not yet fully implemented.", "sensitivity_scores": {}}

def assess_vulnerability(model_metadata: ModelMetadata, model_obj: Any, typical_input: Optional[Any]=None) -> Dict[str, Any]:
    vulnerabilities = {}
    if model_metadata and model_metadata.layer_info:
        for layer in model_metadata.layer_info:
            layer_type_str = layer.get('type', '').lower()
            # Example check: ReLU without BatchNorm (very basic)
            if 'relu' in layer_type_str and not any('batchnorm' in l.get('type','').lower() for l in model_metadata.layer_info if l.get('name', '').startswith(layer.get('name', '___xxx___').split('.')[0])): # Check if BN is in same "block"
                vulnerabilities.setdefault('warnings', []).append(f"Layer {layer.get('name')} is ReLU without obvious preceding BatchNorm, potential for dying ReLU.")
            # Example check: Untrained Embedding layer (if identifiable)
            if 'embedding' in layer_type_str and layer.get('num_params', 0) > 0:
                # This would require checking weights, not just config. For future.
                pass
    return {"notes": "Vulnerability assessment needs more depth.", "identified_issues": vulnerabilities}
