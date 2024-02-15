"""
Configuration classes for evaluation module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List

class ModelType(Enum):
    """Enum for supported model types."""
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    GENERATION = "generation"

class Framework(Enum):
    """Enum for supported frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    KERAS = "keras"
    SKLEARN = "sklearn"

@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable configuration for evaluation."""
    model_type: ModelType
    framework: Framework
    batch_size: int = 32
    num_workers: int = 1
    use_cache: bool = True
    timeout: float = 300.0
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "robustness"])

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    success: bool
    metrics: dict
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0

class EvaluationBuilder:
    """Builder for configuring evaluations."""
    
    def __init__(self):
        self._config = {
            'model_type': None,
            'framework': None,
            'batch_size': 32,
            'num_workers': 1,
            'use_cache': True,
            'timeout': 300.0,
            'metrics': ["accuracy", "robustness"]
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

    def with_metrics(self, metrics: List[str]) -> 'EvaluationBuilder':
        self._config['metrics'] = metrics
        return self

    def build(self) -> EvaluationConfig:
        if not all([self._config['model_type'], self._config['framework']]):
            raise ValueError("Model type and framework must be specified")
        return EvaluationConfig(**self._config) 