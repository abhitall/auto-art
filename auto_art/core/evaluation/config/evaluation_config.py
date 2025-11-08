"""
Configuration classes for evaluation module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any # Added Dict, Optional, Tuple, Any

class ModelType(Enum):
    """Enum for supported model types."""
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    GENERATION = "generation"
    REGRESSION = "regression"
    LLM = "llm"

class Framework(Enum):
    """Enum for supported frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    SKLEARN = "sklearn"
    MXNET = "mxnet"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    GPY = "gpy"

@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable configuration for evaluation."""
    model_type: ModelType
    framework: Framework
    batch_size: int = 32
    num_workers: int = 4 # Harmonized default with builder
    use_cache: bool = True
    timeout: float = 300.0
    metrics_to_calculate: List[str] = field(default_factory=lambda: ["empirical_robustness"]) # Specific robust. metrics
    device_preference: Optional[str] = None

    input_shape: Optional[Tuple[int, ...]] = None
    nb_classes: Optional[int] = None
    loss_function: Optional[Any] = None
    num_samples_for_adv_metrics: int = 5

    # General 'metrics' list from original config, might be for overall selection of what to report/calculate broadly
    # Kept for backward compatibility or if it serves a different purpose than metrics_to_calculate
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "robustness"])


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    success: bool
    metrics_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0

class EvaluationBuilder:
    """Builder for configuring evaluations."""

    def __init__(self):
        self._config: Dict[str, Any] = {
            'model_type': None,
            'framework': None,
            'batch_size': 32,
            'num_workers': 4,
            'use_cache': True,
            'timeout': 300.0,
            'metrics_to_calculate': ["empirical_robustness"],
            'device_preference': None,
            'input_shape': None,
            'nb_classes': None,
            'loss_function': None,
            'num_samples_for_adv_metrics': 5,
            'metrics': ["accuracy", "robustness"] # Original general metrics list
        }

    def with_model_type(self, model_type: ModelType) -> 'EvaluationBuilder':
        self._config['model_type'] = model_type
        return self

    def with_framework(self, framework: Framework) -> 'EvaluationBuilder':
        self._config['framework'] = framework
        return self

    def with_batch_size(self, batch_size: int) -> 'EvaluationBuilder':
        self._config['batch_size'] = batch_size
        return self

    def with_num_workers(self, num_workers: int) -> 'EvaluationBuilder':
        self._config['num_workers'] = num_workers
        return self

    def with_cache(self, use_cache: bool) -> 'EvaluationBuilder':
        self._config['use_cache'] = use_cache
        return self

    def with_timeout(self, timeout: float) -> 'EvaluationBuilder':
        self._config['timeout'] = timeout
        return self

    def with_metrics_to_calculate(self, metrics_list: List[str]) -> 'EvaluationBuilder':
        self._config['metrics_to_calculate'] = metrics_list
        return self

    def with_device_preference(self, device: Optional[str]) -> 'EvaluationBuilder':
        self._config['device_preference'] = device
        return self

    def with_input_shape(self, shape: Tuple[int, ...]) -> 'EvaluationBuilder':
        self._config['input_shape'] = shape
        return self

    def with_nb_classes(self, num_classes: int) -> 'EvaluationBuilder':
        self._config['nb_classes'] = num_classes
        return self

    def with_loss_function(self, loss_fn: Any) -> 'EvaluationBuilder':
        self._config['loss_function'] = loss_fn
        return self

    def with_num_samples_for_adv_metrics(self, num_samples: int) -> 'EvaluationBuilder':
        self._config['num_samples_for_adv_metrics'] = num_samples
        return self

    def with_metrics_overall_list(self, metrics_list: List[str]) -> 'EvaluationBuilder': # For the original 'metrics' field
        self._config['metrics'] = metrics_list
        return self

    def build(self) -> EvaluationConfig:
        if not all([self._config['model_type'], self._config['framework']]):
            raise ValueError("Model type and framework must be specified for EvaluationConfig.")

        # Ensure all keys defined in EvaluationConfig dataclass are present in self._config
        # This is crucial because EvaluationConfig is frozen=True
        config_fields = EvaluationConfig.__dataclass_fields__.keys()
        for fld in config_fields:
            if fld not in self._config:
                # This case should ideally not happen if __init__ initializes all keys
                # Or if EvaluationConfig fields have defaults for all non-mandatory ones.
                # For frozen dataclasses, all fields must be passed to __init__.
                # If a field in EvaluationConfig has no default and is not in self._config, this will error.
                # The current EvaluationConfig has defaults for most, model_type/framework are checked above.
                pass

        return EvaluationConfig(**self._config)
