"""
Interfaces defining the contract for different components using Protocol classes.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, TypeVar, runtime_checkable
from dataclasses import dataclass

T = TypeVar('T')  # Generic type for model inputs/outputs

@runtime_checkable
class ModelInterface(Protocol[T]):
    """Interface for model handling components."""
    
    def load_model(self, model_path: str) -> Tuple[Any, str]:
        """Load model from path and return model object and framework name."""
        ...
    
    def analyze_architecture(self, model: Any, framework: str) -> Dict[str, Any]:
        """Analyze model architecture and return metadata."""
        ...
    
    def preprocess_input(self, input_data: T) -> T:
        """Preprocess input data according to model requirements."""
        ...
    
    def postprocess_output(self, output_data: T) -> T:
        """Postprocess output data according to model requirements."""
        ...

@runtime_checkable
class AttackInterface(Protocol[T]):
    """Interface for adversarial attack components."""
    
    def create_attack(self, model: Any, metadata: Dict[str, Any], 
                     config: Dict[str, Any]) -> Any:
        """Create attack object with given configuration."""
        ...
    
    def apply_attack(self, attack: Any, test_data: T) -> Tuple[T, T]:
        """Apply attack to test data and return adversarial and clean examples."""
        ...
    
    def validate_attack(self, attack: Any, test_data: T) -> bool:
        """Validate attack configuration and test data."""
        ...
    
    def get_attack_metrics(self, clean_data: T, adversarial_data: T) -> Dict[str, float]:
        """Calculate attack-specific metrics."""
        ...

@runtime_checkable
class EvaluatorInterface(Protocol[T]):
    """Interface for model evaluation components."""
    
    def evaluate_model(self, model_path: str, num_samples: int) -> Dict[str, Any]:
        """Evaluate model robustness against attacks."""
        ...
    
    def calculate_metrics(self, clean_data: T, adversarial_data: T, 
                         expected_outputs: T) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        ...
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        ...
    
    def validate_evaluation(self, results: Dict[str, Any]) -> bool:
        """Validate evaluation results."""
        ...

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    num_samples: int = 100
    attack_configs: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[List[str]] = None
    report_format: str = 'json'
    save_results: bool = True
    output_dir: Optional[str] = None

@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    attack_type: str
    epsilon: float = 0.3
    eps_step: float = 0.01
    max_iter: int = 100
    targeted: bool = False
    num_random_init: int = 0
    batch_size: int = 32
    additional_params: Optional[Dict[str, Any]] = None 