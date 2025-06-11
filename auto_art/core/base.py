"""
Base classes implementing the Template Method pattern for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic, Union, Callable # Added Union, Callable
from pathlib import Path # Added Path
from dataclasses import dataclass
import numpy as np

T = TypeVar('T')  # Generic type for model inputs/outputs

@dataclass
class ModelMetadata:
    """Metadata about the analyzed model."""
    model_type: str
    framework: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    input_type: str
    output_type: str
    layer_info: List[Dict[str, Any]]
    additional_info: Optional[Dict[str, Any]] = None

class BaseModel(ABC, Generic[T]):
    """Base class for model handling with Template Method pattern."""

    @abstractmethod
    def load_model(self, model_path: str) -> Tuple[Any, str]:
        """Load model from path and return model object and framework name."""
        pass

    @abstractmethod
    def analyze_architecture(self, model: Any, framework: str) -> ModelMetadata:
        """Analyze model architecture and return metadata."""
        pass

    def preprocess_input(self, input_data: T) -> T:
        """Preprocess input data according to model requirements."""
        return input_data

    def postprocess_output(self, output_data: T) -> T:
        """Postprocess output data according to model requirements."""
        return output_data

    def validate_model(self, model: Any) -> bool:
        """Validate model structure and requirements."""
        if hasattr(model, 'predict') or hasattr(model, 'forward'):
            return True
        return False

    @abstractmethod
    def get_model_predictions(self, model: Any, data: T) -> T:
        """Get predictions from the model for the given data."""
        pass

class BaseAttack(ABC, Generic[T]):
    """Base class for adversarial attacks with Template Method pattern."""

    @abstractmethod
    def create_attack(self, model: Any, metadata: ModelMetadata, config: Dict[str, Any]) -> Any:
        """Create attack object with given configuration."""
        pass

    @abstractmethod
    def apply_attack(self, attack: Any, test_data: T) -> Tuple[T, T]: # test_data is T, which could be TestData obj or np.ndarray
        """Apply attack to test data and return adversarial and clean examples."""
        pass

    def validate_attack(self, attack: Any, test_data: T) -> bool:
        """Validate attack configuration and test data."""
        return True

    def get_attack_metrics(self, clean_data: T, adversarial_data: T) -> Dict[str, float]:
        """Calculate attack-specific metrics."""
        return {
            "perturbation_norm_mean": np.nan,
            "attack_success_rate": np.nan
        }

class BaseTestGenerator(ABC, Generic[T]):
    """Base class for test data generation with Template Method pattern."""

    @abstractmethod
    def generate_test_data(self, metadata: ModelMetadata, num_samples: int) -> T: # T is likely TestData obj
        """Generate test data based on model metadata."""
        pass

    @abstractmethod
    def generate_expected_outputs(self, model: Any, test_data: T) -> T: # test_data is T (e.g. TestData), returns T (e.g. np.ndarray for labels)
        """Generate expected outputs for test data."""
        pass

    def validate_test_data(self, test_data: T, metadata: ModelMetadata) -> bool: # test_data is T (e.g. TestData)
        """Validate generated test data."""
        return True

    def augment_test_data(self, test_data: T) -> T: # test_data is T (e.g. TestData)
        """Augment test data with additional variations."""
        return test_data

    @abstractmethod
    def load_data_from_source(self,
                              source: Union[str, Path, Tuple[np.ndarray, np.ndarray], Any],
                              data_type: str = 'test',
                              num_samples: Optional[int] = None,
                              feature_columns: Optional[List[Union[int, str]]] = None,
                              label_columns: Optional[Union[int, str, List[Union[int, str]]]] = None,
                              preprocessing_fn: Optional[Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]]] = None
                             ) -> T:
        """
        Loads data from a specified source.
        The returned type T should encapsulate inputs and optionally labels.
        Implementations should handle different source types (filepaths, numpy arrays).
        """
        pass

class BaseEvaluator(ABC, Generic[T]):
    """Base class for model evaluation with Template Method pattern."""

    def __init__(self):
        self.model_handler: Optional[BaseModel[T]] = None
        self.attack_handler: Optional[BaseAttack[T]] = None
        self.test_generator: Optional[BaseTestGenerator[T]] = None # This would be an instance of a concrete TestDataGenerator

    @abstractmethod
    def evaluate_model(self, model_path: str, num_samples: int) -> Dict[str, Any]: # num_samples here might relate to synthetic or loaded data count
        """Evaluate model robustness against attacks."""
        pass

    @abstractmethod
    def calculate_metrics(self, clean_data: T, adversarial_data: T,
                         expected_outputs: T) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        pass

    @abstractmethod
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]: # Should be -> str based on ARTEvaluator
        """Generate comprehensive evaluation report."""
        pass

    def validate_evaluation(self, results: Dict[str, Any]) -> bool:
        """Validate evaluation results."""
        return True
