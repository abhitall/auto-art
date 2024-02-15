"""
Poisoning attack strategies implementation.
"""

from abc import ABC
from typing import Tuple, Any, Dict, Optional
import numpy as np
from art.attacks.poisoning import PoisoningAttackBackdoor, PoisoningAttackCleanLabelBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd
from .base import AttackStrategy

class PoisoningAttackStrategy(AttackStrategy, ABC):
    """Base class for poisoning attack strategies."""
    def __init__(self):
        super().__init__()
        self._attack = None

class BackdoorAttack(PoisoningAttackStrategy):
    """Backdoor poisoning attack implementation."""
    def __init__(self, 
                 trigger_pattern: np.ndarray,
                 percent_poison: float = 0.1,
                 channels_first: bool = False):
        """
        Initialize backdoor attack.
        
        Args:
            trigger_pattern: Pattern to be used as the trigger
            percent_poison: Percentage of poisoned samples
            channels_first: Whether channels are first in the image format
        """
        super().__init__()
        self.trigger_pattern = trigger_pattern
        self.percent_poison = percent_poison
        self.channels_first = channels_first

    def execute(self, classifier: Any, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Execute the backdoor attack.
        
        Args:
            classifier: Target classifier
            x: Input samples
            y: True labels
            
        Returns:
            Tuple of (poisoned samples, success rate)
        """
        self._attack = PoisoningAttackBackdoor(
            classifier,
            perturbation=add_pattern_bd,
            percent_poison=self.percent_poison,
            channels_first=self.channels_first
        )
        
        poisoned_data = self._attack.poison(x, y, trigger_pattern=self.trigger_pattern)
        success_rate = self._evaluate_attack_success(classifier, poisoned_data, y)
        
        return poisoned_data, success_rate

class CleanLabelBackdoorAttack(PoisoningAttackStrategy):
    """Clean label backdoor attack implementation."""
    def __init__(self,
                 trigger_pattern: np.ndarray,
                 target: int,
                 percent_poison: float = 0.1,
                 channels_first: bool = False):
        """
        Initialize clean label backdoor attack.
        
        Args:
            trigger_pattern: Pattern to be used as the trigger
            target: Target label
            percent_poison: Percentage of poisoned samples
            channels_first: Whether channels are first in the image format
        """
        super().__init__()
        self.trigger_pattern = trigger_pattern
        self.target = target
        self.percent_poison = percent_poison
        self.channels_first = channels_first

    def execute(self, classifier: Any, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Execute the clean label backdoor attack.
        
        Args:
            classifier: Target classifier
            x: Input samples
            y: True labels
            
        Returns:
            Tuple of (poisoned samples, success rate)
        """
        self._attack = PoisoningAttackCleanLabelBackdoor(
            classifier,
            target=self.target,
            percent_poison=self.percent_poison,
            channels_first=self.channels_first
        )
        
        poisoned_data = self._attack.poison(x, y, trigger_pattern=self.trigger_pattern)
        success_rate = self._evaluate_attack_success(classifier, poisoned_data, y)
        
        return poisoned_data, success_rate

    def get_params(self) -> Dict[str, Any]:
        """Get attack parameters."""
        return {
            'target': self.target,
            'percent_poison': self.percent_poison,
            'channels_first': self.channels_first
        } 