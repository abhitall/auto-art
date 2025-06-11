"""
Wrapper for ART's Carlini & Wagner L2 (C&W L2) Attack.
"""
from typing import Any, Dict, Optional, Union
import numpy as np

try:
    from art.attacks.evasion import CarliniL2Method as ARTCarliniL2
    from art.estimators.classification import ClassifierMixin
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    # Define dummy types for type hinting if ART is not available
    class ClassifierMixin: pass
    class ARTCarliniL2: pass

class CarliniWagnerL2Wrapper:
    """
    A wrapper for the ART Carlini & Wagner L2 attack.
    """
    def __init__(self,
                 estimator: ClassifierMixin, # ART C&W uses 'classifier' in its constructor
                 confidence: float = 0.0,
                 targeted: bool = False,
                 learning_rate: float = 0.01, # ART Default: 0.01
                 binary_search_steps: int = 9, # ART Default: 9
                 max_iter: int = 100, # ART Default: 1000. Adjusted for potentially faster runs.
                 initial_const: float = 0.01, # ART Default: 0.001. Adjusted.
                 max_halvings: int = 5, # ART Default: 5
                 max_doublings: int = 5, # ART Default: 5
                 batch_size: int = 32, # ART Default: 1. Adjusted for consistency.
                 verbose: bool = True):
        """
        Initializes the C&W L2 wrapper.
        Args:
            estimator: An ART-compatible classifier.
            confidence: Confidence of the adversarial examples.
            targeted: Should the attack be targeted.
            learning_rate: The learning rate for the attack algorithm.
            binary_search_steps: Number of binary search steps to find the optimal const.
            max_iter: Maximum number of iterations for the gradient descent.
            initial_const: The initial trade-off constant c.
            max_halvings: Maximum number of halvings of the constant c.
            max_doublings: Maximum number of doublings of the constant c.
            batch_size: Size of the batch on which adversarial samples are generated.
            verbose: Show progress bars.
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. CarliniWagnerL2Method cannot be used.")

        if not isinstance(estimator, ClassifierMixin):
            raise TypeError(f"Estimator must be an ART ClassifierMixin, got {type(estimator)}")

        self.attack = ARTCarliniL2(
            classifier=estimator, # Pass as 'classifier'
            confidence=confidence,
            targeted=targeted,
            learning_rate=learning_rate,
            binary_search_steps=binary_search_steps,
            max_iter=max_iter,
            initial_const=initial_const,
            max_halvings=max_halvings,
            max_doublings=max_doublings,
            batch_size=batch_size,
            verbose=verbose
        )
        self._targeted_at_init = targeted # Store initial targeted flag

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generates adversarial examples.

        Args:
            x: Input samples (numpy array).
            y: Target labels for targeted attacks. If self._targeted_at_init is True, y must be provided.
               If self._targeted_at_init is False, y is ignored by C&W L2.
            **kwargs: Additional arguments for the ART attack's generate method.
                      (C&W L2 generate method in ART 1.x does not take many extra **kwargs like 'mask').
        Returns:
            Adversarial examples.
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for CarliniWagnerL2.generate.")

        if self._targeted_at_init and y is None:
            raise ValueError("Target labels 'y' must be provided for a targeted C&W L2 attack initialized with targeted=True.")

        # If not targeted at init, ART's C&W L2 generate method ignores 'y'.
        # If targeted at init, 'y' is used.
        return self.attack.generate(x=x, y=y, **kwargs)
