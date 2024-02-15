"""
Base class for attack strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from art.utils import compute_success

class AttackStrategy(ABC):
    """Base class for all attack strategies."""
    
    def __init__(self):
        """Initialize attack strategy."""
        self._attack = None
    
    @abstractmethod
    def execute(
        self,
        classifier: Any,
        x: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Execute the attack strategy.
        
        Args:
            classifier: The target classifier
            x: Input samples
            y: True labels (optional)
            
        Returns:
            Tuple of (adversarial examples, success rate)
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get attack parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        pass
    
    @abstractmethod
    def set_params(self, **params) -> None:
        """
        Set attack parameters.
        
        Args:
            **params: Parameter names and values
        """
        pass

class BaseAttackStrategy(AttackStrategy):
    """Base implementation for attack strategies."""
    
    def __init__(self, attack_class: Any, params: Dict[str, Any]):
        super().__init__()
        self.attack_class = attack_class
        self.params = params.copy()
        self._attack_instance = None

    def get_params(self) -> Dict[str, Any]:
        return self.params.copy()

    def set_params(self, **params: Any) -> None:
        self.params.update(params)
        self._attack_instance = None  # Reset instance when params change

    @property
    def attack_instance(self) -> Any:
        """Lazy initialization of attack instance."""
        if self._attack_instance is None:
            self._attack_instance = self.attack_class(**self.params)
        return self._attack_instance

    def execute(
        self,
        classifier: Any,
        x: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Execute attack and return adversarial examples and success rate."""
        self._attack_instance = self.attack_class(classifier, **self.params)
        adversarial_examples = self._attack_instance.generate(x=data)
        success_rate = compute_success(
            classifier,
            data,
            labels,
            adversarial_examples
        )
        return adversarial_examples, float(success_rate) 