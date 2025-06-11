"""
Configuration classes for evaluation module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional # Ensure Optional is imported

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

@dataclass(frozen=True) # Keep frozen=True if these are meant to be immutable after creation via builder
class EvaluationConfig:
    """Immutable configuration for evaluation."""
    model_type: ModelType
    framework: Framework
    batch_size: int = 32
    num_workers: int = 1 # For tasks like parallel attack evaluation
    use_cache: bool = True # General caching hint
    timeout: float = 300.0 # Timeout for long operations
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "robustness"])
    device_preference: Optional[str] = None # Added: "auto", "cpu", "gpu"

    # Added fields that ARTEvaluator's art_estimator property expects, ensure they are here or handled
    input_shape: Optional[Tuple[int, ...]] = None # e.g. (3, 32, 32) - must not include batch size for ART
    nb_classes: Optional[int] = None
    loss_function: Optional[Any] = None # type: ignore # Can be framework specific, e.g. torch.nn.CrossEntropyLoss instance
    num_samples_for_adv_metrics: int = 5 # For metrics like CLEVER that run on samples
    metrics_to_calculate: List[str] = field(default_factory=lambda: ["empirical_robustness"]) # Specific metrics to run

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    success: bool
    metrics_data: Dict[str, Any] = field(default_factory=dict) # Changed from 'metrics: dict' to match ARTEvaluator
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0

class EvaluationBuilder:
    """Builder for configuring evaluations."""

    def __init__(self):
        # Initialize with all fields from EvaluationConfig to allow building a complete object
        self._config: Dict[str, Any] = {
            'model_type': None,
            'framework': None,
            'batch_size': 32,
            'num_workers': 4, # Default from ARTEvaluator
            'use_cache': True,
            'timeout': 300.0, # Default from EvaluationConfig, was 3600 in ARTEvaluator, harmonizing
            'metrics': ["accuracy", "robustness"], # Default from EvaluationConfig
            'device_preference': None, # Added
            'input_shape': None, # Added for completeness
            'nb_classes': None,  # Added for completeness
            'loss_function': None, # Added for completeness
            'num_samples_for_adv_metrics': 5, # Added
            'metrics_to_calculate': ["empirical_robustness"] # Added
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

    def with_metrics(self, metrics: List[str]) -> 'EvaluationBuilder': # This is for the general metrics list
        self._config['metrics'] = metrics
        return self

    def with_metrics_to_calculate(self, metrics_list: List[str]) -> 'EvaluationBuilder': # Specific list
        self._config['metrics_to_calculate'] = metrics_list
        return self

    def with_device_preference(self, device: Optional[str]) -> 'EvaluationBuilder': # Added
        self._config['device_preference'] = device
        return self

    def with_input_shape(self, shape: Tuple[int, ...]) -> 'EvaluationBuilder': # Added
        self._config['input_shape'] = shape
        return self

    def with_nb_classes(self, num_classes: int) -> 'EvaluationBuilder': # Added
        self._config['nb_classes'] = num_classes
        return self

    def with_loss_function(self, loss_fn: Any) -> 'EvaluationBuilder': # Added
        self._config['loss_function'] = loss_fn
        return self

    def with_num_samples_for_adv_metrics(self, num_samples: int) -> 'EvaluationBuilder': # Added
        self._config['num_samples_for_adv_metrics'] = num_samples
        return self

    def build(self) -> EvaluationConfig:
        if not all([self._config['model_type'], self._config['framework']]):
            raise ValueError("Model type and framework must be specified for EvaluationConfig.")
        # Ensure all keys expected by EvaluationConfig are present in self._config
        # This is important if EvaluationConfig is frozen=True.
        # For non-frozen, missing keys would just use dataclass defaults if any.
        # Since it's frozen, all fields must be provided to constructor.
        return EvaluationConfig(**self._config)

```
I've also taken the liberty to add other fields to `EvaluationConfig` and `EvaluationBuilder` that `ARTEvaluator` seemed to expect from its `self.config` (like `input_shape`, `nb_classes`, `loss_function`, `num_samples_for_adv_metrics`, `metrics_to_calculate`). This makes the `EvaluationConfig` more complete for `ARTEvaluator`'s needs. I also updated `EvaluationResult` to use `metrics_data` to match what `ARTEvaluator` was changed to previously. `Any` from `typing` was also added for `loss_function`.
