"""
Base classes implementing the Template Method pattern for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic
from dataclasses import dataclass

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
    
    @abstractmethod
    def preprocess_input(self, input_data: T) -> T:
        """Preprocess input data according to model requirements."""
        pass
    
    @abstractmethod
    def postprocess_output(self, output_data: T) -> T:
        """Postprocess output data according to model requirements."""
        pass
    
    def validate_model(self, model: Any) -> bool:
        """Validate model structure and requirements."""
        return True  # Default implementation

class BaseAttack(ABC, Generic[T]):
    """Base class for adversarial attacks with Template Method pattern."""
    
    @abstractmethod
    def create_attack(self, model: Any, metadata: ModelMetadata, config: Dict[str, Any]) -> Any:
        """Create attack object with given configuration."""
        pass
    
    @abstractmethod
    def apply_attack(self, attack: Any, test_data: T) -> Tuple[T, T]:
        """Apply attack to test data and return adversarial and clean examples."""
        pass
    
    @abstractmethod
    def validate_attack(self, attack: Any, test_data: T) -> bool:
        """Validate attack configuration and test data."""
        pass
    
    def get_attack_metrics(self, clean_data: T, adversarial_data: T) -> Dict[str, float]:
        """Calculate attack-specific metrics."""
        return {}  # Default implementation

class BaseTestGenerator(ABC, Generic[T]):
    """Base class for test data generation with Template Method pattern."""
    
    @abstractmethod
    def generate_test_data(self, metadata: ModelMetadata, num_samples: int) -> T:
        """Generate test data based on model metadata."""
        pass
    
    @abstractmethod
    def generate_expected_outputs(self, model: Any, test_data: T) -> T:
        """Generate expected outputs for test data."""
        pass
    
    @abstractmethod
    def validate_test_data(self, test_data: T, metadata: ModelMetadata) -> bool:
        """Validate generated test data."""
        pass
    
    def augment_test_data(self, test_data: T) -> T:
        """Augment test data with additional variations."""
        return test_data  # Default implementation

class BaseEvaluator(ABC, Generic[T]):
    """Base class for model evaluation with Template Method pattern."""
    
    def __init__(self):
        self.model_handler: Optional[BaseModel[T]] = None
        self.attack_handler: Optional[BaseAttack[T]] = None
        self.test_generator: Optional[BaseTestGenerator[T]] = None
    
    @abstractmethod
    def evaluate_model(self, model_path: str, num_samples: int) -> Dict[str, Any]:
        """Evaluate model robustness against attacks."""
        pass
    
    @abstractmethod
    def calculate_metrics(self, clean_data: T, adversarial_data: T, 
                         expected_outputs: T) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        pass
    
    @abstractmethod
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        pass
    
    def validate_evaluation(self, results: Dict[str, Any]) -> bool:
        """Validate evaluation results."""
        return True  # Default implementation 