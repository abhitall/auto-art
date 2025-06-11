"""
Wrapper for ART's Boundary Attack.
"""
from typing import Any, Optional, Dict
import numpy as np

try:
    from art.attacks.evasion import BoundaryAttack as ARTBoundaryAttack
    from art.estimators.classification import ClassifierMixin
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False
    class ClassifierMixin: pass
    class ARTBoundaryAttack: pass

class BoundaryAttackWrapper:
    """
    A wrapper for the ART Boundary Attack.
    This attack is query-based and typically requires many queries.
    """
    def __init__(self,
                 estimator: ClassifierMixin,
                 targeted: bool = False,
                 delta: float = 0.01,
                 epsilon: float = 0.01,
                 step_adapt: float = 0.667,
                 max_iter: int = 100, # ART Default: 5000. Significantly reduced.
                 num_trial: int = 25, # ART Default: 25
                 sample_size: int = 20, # ART Default: 20
                 init_size: int = 100, # ART Default: 100
                 # batch_size for generate method, not __init__ for ART's BoundaryAttack
                 verbose: bool = True):
        """
        Initializes the Boundary Attack wrapper.
        Args:
            estimator: An ART-compatible classifier.
            targeted: Should the attack be targeted.
            delta: Initial step size for the orthogonal step.
            epsilon: Initial step size for the step towards the target.
            step_adapt: Factor by which the step sizes are multiplied or divided.
            max_iter: Maximum number of iterations.
            num_trial: Maximum number of trials to find a starting point (if x_adv_init is None).
            sample_size: Number of samples to draw for a new starting point (if x_adv_init is None).
            init_size: Number of initial points to draw (if x_adv_init is None).
            verbose: Show progress bars.
        """
        if not ART_AVAILABLE:
            raise ImportError("Adversarial Robustness Toolbox (ART) is not installed. BoundaryAttack cannot be used.")

        if not isinstance(estimator, ClassifierMixin):
            raise TypeError(f"Estimator must be an ART ClassifierMixin, got {type(estimator)}")

        self.attack = ARTBoundaryAttack(
            estimator=estimator,
            targeted=targeted,
            delta=delta,
            epsilon=epsilon,
            step_adapt=step_adapt,
            max_iter=max_iter,
            num_trial=num_trial,
            sample_size=sample_size,
            init_size=init_size,
            verbose=verbose
        )
        self._targeted_at_init = targeted # Store for generate method logic

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None,
                 x_adv_init: Optional[np.ndarray] = None,
                 batch_size: int = 1, # Default batch_size for generate, can be overridden
                 **kwargs) -> np.ndarray:
        """
        Generates adversarial examples.

        Args:
            x: Input samples (numpy array).
            y: Target labels for targeted attacks or original labels for untargeted.
               Boundary attack needs `y`.
            x_adv_init: Initial adversarial examples (optional). If None, starts from random noise.
            batch_size: Batch size for generating adversarial examples.
            **kwargs: Placeholder for any other future arguments to ART's generate.
        Returns:
            Adversarial examples.
        """
        if not ART_AVAILABLE:
            raise ImportError("ART not available for BoundaryAttack.generate.")

        if y is None: # Boundary attack requires y for both targeted and untargeted modes in ART
            raise ValueError("Labels 'y' must be provided for Boundary Attack.")

        # ART's BoundaryAttack.generate takes x, y, x_adv_init, and batch_size
        return self.attack.generate(x=x, y=y, x_adv_init=x_adv_init, batch_size=batch_size)
